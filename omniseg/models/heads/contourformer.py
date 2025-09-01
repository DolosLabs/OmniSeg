"""ContourFormer segmentation head implementation."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional
from collections import OrderedDict

from scipy.optimize import linear_sum_assignment
from scipy.interpolate import interp1d
from skimage.measure import find_contours
from PIL import Image, ImageDraw
from torchvision.ops import FeaturePyramidNetwork

from ..base import BaseHead
from ..backbones import get_backbone


class GenericBackboneWithFPN(nn.Module):
    """Generic wrapper to add FPN to any backbone."""
    
    def __init__(self, backbone_type: str = 'dino', fpn_out_channels: int = 256, dummy_input_size: int = 224):
        super().__init__()
        self.backbone = get_backbone(backbone_type)
        
        # Discover output channels by running a dummy forward pass
        dummy = torch.randn(1, 3, dummy_input_size, dummy_input_size)
        with torch.no_grad():
            feat_dict = self.backbone(dummy)
        
        feat_keys = sorted(list(feat_dict.keys()))
        in_channels_list = [feat_dict[k].shape[1] for k in feat_keys]
        self.out_channels = fpn_out_channels
        self.fpn = FeaturePyramidNetwork(in_channels_list=in_channels_list, out_channels=self.out_channels)
        self._orig_keys = feat_keys
        self._num_levels = len(feat_keys)

    def forward(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        feats = self.backbone(x)
        ordered_input = OrderedDict()
        
        for k in self._orig_keys:
            if k not in feats:
                raise KeyError(f"Expected feature '{k}' from backbone but got keys: {list(feats.keys())}")
            ordered_input[k] = feats[k]
        
        fpn_out = self.fpn(ordered_input)
        # Remap keys to strings '0', '1', '2', ... for DETR compatibility
        remapped = OrderedDict([(str(i), val) for i, (key, val) in enumerate(fpn_out.items())])
        return remapped


def bilinear_sample(feat, coords):
    """Simple bilinear sampler helper (for grid_sample)."""
    # feat: [N, C, H, W], coords: [N, L, 2] in [-1,1]
    N, C, H, W = feat.shape
    L = coords.shape[1]
    grid = coords.view(N, 1, L, 2)
    sampled = F.grid_sample(feat, grid, align_corners=True, mode='bilinear')
    sampled = sampled.view(N, C, L).permute(0, 2, 1)
    return sampled


class DeformableCrossAttention(nn.Module):
    """Deformable-style cross-attention (single-level)."""
    
    def __init__(self, d_model, n_heads, n_points):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_points = n_points
        self.head_dim = d_model // n_heads
        self.to_q = nn.Linear(d_model, d_model)
        self.to_kv = nn.Linear(d_model, 2 * d_model)
        self.offset_mlp = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim),
            nn.ReLU(),
            nn.Linear(self.head_dim, n_points * 2)
        )
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, query, memory, reference_points, spatial_shape):
        N, Len_q, _ = query.shape
        q = self.to_q(query)
        kv = self.to_kv(memory)
        k, v = kv.chunk(2, dim=-1)
        q = q.view(N, Len_q, self.n_heads, self.head_dim)
        k = k.view(N, -1, self.n_heads, self.head_dim)
        v = v.view(N, -1, self.n_heads, self.head_dim)
        offsets = self.offset_mlp(q)
        offsets = offsets.view(N, Len_q, self.n_heads, self.n_points, 2)
        if reference_points.dim() == 3:
            reference_points = reference_points.unsqueeze(2).unsqueeze(3)
        elif reference_points.dim() == 4:
            reference_points = reference_points.unsqueeze(3)
        reference_points = reference_points.expand(-1, -1, self.n_heads, self.n_points, -1)
        sampling_locations = reference_points + offsets.tanh()
        q = q.unsqueeze(3).expand(-1, -1, -1, self.n_points, -1)
        q = q.contiguous().view(N, Len_q * self.n_heads * self.n_points, self.head_dim)
        v = v.contiguous().view(N, -1, self.head_dim)
        attn_output = q
        attn_output = attn_output.view(N, Len_q, self.n_heads, self.n_points, self.head_dim)
        attn_output = attn_output.mean(dim=3)
        attn_output = attn_output.contiguous().view(N, Len_q, self.d_model)
        attn_output = self.output_proj(attn_output)
        return attn_output


class DeformableDecoderLayer(nn.Module):
    """Deformable decoder layer."""
    
    def __init__(self, d_model=256, n_heads=8, n_points=4, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = DeformableCrossAttention(d_model, n_heads, n_points)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory_feats, reference_points, spatial_shape):
        q = k = tgt
        tgt2, _ = self.self_attn(q, k, value=tgt)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.cross_attn(tgt, memory_feats, reference_points, spatial_shape)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class SimpleDeformableTransformer(nn.Module):
    """Simple deformable transformer decoder stack."""
    
    def __init__(self, d_model=256, nheads=8, num_decoder_layers=6, n_points=4, dim_feedforward=2048):
        super().__init__()
        self.decoder_layers = nn.ModuleList(
            [DeformableDecoderLayer(d_model, nheads, n_points, dim_feedforward) for _ in range(num_decoder_layers)]
        )

    def forward(self, query_embed, memory_feats, reference_points, spatial_shape):
        hs = []
        tgt = query_embed
        for layer in self.decoder_layers:
            tgt = layer(tgt, memory_feats, reference_points, spatial_shape)
            hs.append(tgt)
        return torch.stack(hs)


class HungarianMatcher(nn.Module):
    """Hungarian matcher with annealing / point-only warmup."""
    
    def __init__(self, cost_class=1, cost_point=1, cost_giou=1, cost_point_init=10, cost_decay=0.95, warmup_epochs=10):
        super().__init__()
        self.base_cost_class = cost_class
        self.base_cost_point = cost_point
        self.cost_giou = cost_giou
        self.cost_point_init = cost_point_init
        self.cost_decay = cost_decay
        self.epoch = 0
        self.warmup_epochs = warmup_epochs

    def step_epoch(self):
        self.epoch += 1

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        out_prob = (outputs["pred_logits"].flatten(0, 1) / 0.7).softmax(-1)
        out_points = outputs["pred_coords"].flatten(0, 1)
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_points = torch.cat([v["coords"] for v in targets])
        cost_class = -out_prob[:, tgt_ids]
        cost_point = torch.cdist(out_points.flatten(1), tgt_points.flatten(1), p=1)
        if self.epoch < self.warmup_epochs:
            C = cost_point
        else:
            cost_point_w = max(self.base_cost_point, self.cost_point_init * (self.cost_decay ** self.epoch))
            C = self.base_cost_class * cost_class + cost_point_w * cost_point
        C = C.view(bs, num_queries, -1).cpu()
        sizes = [len(v["coords"]) for v in targets]
        indices = []
        for i, c in enumerate(C.split(sizes, -1)):
            row_ind, col_ind = linear_sum_assignment(c[i].numpy())
            indices.append((torch.as_tensor(row_ind, dtype=torch.int64), torch.as_tensor(col_ind, dtype=torch.int64)))
        return indices


class SetCriterionContourFormer(nn.Module):
    """Criterion with normalized point loss and deep supervision support."""
    
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, num_points):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.num_points = num_points
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_instances):
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        if len(idx[0]) == 0:
            target_classes_o = torch.empty(0, dtype=torch.int64, device=src_logits.device)
        else:
            target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, weight=self.empty_weight.to(src_logits.device))
        return {'loss_ce': loss_ce}

    def loss_points(self, outputs, targets, indices, num_instances):
        idx = self._get_src_permutation_idx(indices)
        if len(idx[0]) == 0:
            return {'loss_point': torch.tensor(0., device=next(iter(outputs.values())).device)}
        src_points = outputs['pred_coords'][idx]
        target_points = torch.cat([t['coords'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_point = F.l1_loss(src_points, target_points, reduction='sum') / (num_instances * self.num_points)
        return {'loss_point': loss_point}

    def _get_src_permutation_idx(self, indices):
        if len(indices) == 0:
            return (torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64))
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def forward(self, outputs, targets, indices=None):
        if indices is None:
            indices = self.matcher(outputs, targets)
        num_instances = sum(len(t["labels"]) for t in targets)
        num_instances = torch.as_tensor([num_instances], dtype=torch.float, device=next(iter(outputs.values())).device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(num_instances)
            world_size = torch.distributed.get_world_size()
        else:
            world_size = 1
        num_instances = torch.clamp(num_instances / world_size, min=1).item()
        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices, num_instances))
        losses.update(self.loss_points(outputs, targets, indices, num_instances))
        return {k: v * self.weight_dict.get(k, 1.0) for k, v in losses.items()}


def masks_to_contours(masks, num_points=50):
    """Convert binary masks to fixed-size contour point sets."""
    contours_list, valid_indices = [], []
    for i, mask in enumerate(masks):
        if mask.sum() == 0: continue
        mask_np = mask.cpu().numpy().astype(np.uint8)
        contours = find_contours(mask_np, 0.5)
        if not contours: 
            # Fallback: create a simple bounding box contour if no contours found
            ys, xs = torch.where(mask > 0)
            if len(ys) > 0:
                y_min, y_max = ys.min().item(), ys.max().item()
                x_min, x_max = xs.min().item(), xs.max().item()
                # Create a rectangular contour
                contour = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], dtype=np.float64)
            else:
                continue
        else:
            sorted_contours = sorted(contours, key=lambda x: len(x), reverse=True)
            contour = sorted_contours[0]
            if len(contour) < 3:  # Reduced from 4 to 3
                # Fallback: create a simple bounding box contour
                ys, xs = torch.where(mask > 0)
                if len(ys) > 0:
                    y_min, y_max = ys.min().item(), ys.max().item()
                    x_min, x_max = xs.min().item(), xs.max().item()
                    contour = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], dtype=np.float64)
                else:
                    continue
            else:
                contour = np.flip(contour, axis=1).astype(np.float64)
        
        # Ensure contour is closed
        if not np.allclose(contour[0], contour[-1], atol=1e-6):
            contour = np.vstack([contour, contour[0:1]])
        
        contour = torch.tensor(contour, dtype=torch.float32)
        contour[:, 0] = torch.clamp(contour[:, 0] / max(mask.shape[1], 1), 0.0, 1.0)
        contour[:, 1] = torch.clamp(contour[:, 1] / max(mask.shape[0], 1), 0.0, 1.0)
        path = contour.t()
        diffs = path[:, 1:] - path[:, :-1]
        distances = torch.sqrt(torch.sum(diffs**2, dim=0))
        distance = torch.cumsum(distances, dim=0)
        distance = torch.cat([torch.tensor([0.0]), distance])
        
        # More lenient distance check
        if distance[-1] < 1e-8: 
            # If distance is too small, create uniform sampling
            sampled_points = contour[:min(len(contour), num_points)]
            if len(sampled_points) < num_points:
                # Repeat last point to reach num_points
                padding = sampled_points[-1:].repeat(num_points - len(sampled_points), 1)
                sampled_points = torch.cat([sampled_points, padding])
        else:
            try:
                f = interp1d(distance.numpy(), path.numpy(), kind='linear', axis=1, fill_value="extrapolate")
                new_distances = np.linspace(0, distance[-1].item(), num_points)
                sampled_points = torch.from_numpy(f(new_distances).T).to(torch.float32)
            except:
                indices = torch.linspace(0, len(contour)-1, num_points).long()
                indices = torch.clamp(indices, 0, len(contour)-1)
                sampled_points = contour[indices]
        
        sampled_points = torch.clamp(sampled_points, 0.0, 1.0)
        # More robust NaN/Inf check with fallback
        if torch.isnan(sampled_points).any() or torch.isinf(sampled_points).any(): 
            # Final fallback: create a simple centered square
            center_x, center_y = 0.5, 0.5
            size = 0.1
            square_points = torch.tensor([
                [center_x - size, center_y - size],
                [center_x + size, center_y - size], 
                [center_x + size, center_y + size],
                [center_x - size, center_y + size]
            ], dtype=torch.float32)
            # Repeat to reach num_points
            repeats = num_points // 4 + 1
            sampled_points = square_points.repeat(repeats, 1)[:num_points]
            
        contours_list.append(sampled_points)
        valid_indices.append(i)
    
    # Always return something, even if empty
    if not contours_list: 
        # Return empty tensors with correct shape instead of None
        return torch.empty((0, num_points, 2), dtype=torch.float32), []
    return torch.stack(contours_list), valid_indices


def contours_to_masks(contours, img_shape):
    """Convert sets of contour points to binary masks."""
    h, w = img_shape
    masks = []
    for contour_set in contours:
        if contour_set.numel() == 0:
            masks.append(torch.zeros(h, w, dtype=torch.bool)); continue
        points = contour_set.clone()
        points[:, 0] = torch.clamp(points[:, 0] * w, 0, w - 0.01)
        points[:, 1] = torch.clamp(points[:, 1] * h, 0, h - 0.01)
        points_np = points.cpu().numpy()
        if len(points_np) <= 10 or np.ptp(points_np[:, 0]) < 2 or np.ptp(points_np[:, 1]) < 2:
            points_rounded = points_np
            if np.ptp(points_np[:, 0]) < 1:
                center_x = np.mean(points_np[:, 0])
                points_rounded[:, 0] = np.linspace(center_x - 1, center_x + 1, len(points_np))
            if np.ptp(points_np[:, 1]) < 1:
                center_y = np.mean(points_np[:, 1])
                points_rounded[:, 1] = np.linspace(center_y - 1, center_y + 1, len(points_np))
        else:
            points_rounded = np.round(points_np).astype(np.float64)
        unique_points = [points_rounded[0]]
        for point in points_rounded[1:]:
            if not np.allclose(point, unique_points[-1], atol=0.5): unique_points.append(point)
        if len(unique_points) >= 3 and not np.allclose(unique_points[0], unique_points[-1], atol=0.5):
            unique_points.append(unique_points[0])
        if len(unique_points) < 3:
            center_x, center_y = np.mean(points_np[:, 0]), np.mean(points_np[:, 1])
            unique_points = [[center_x - 1, center_y - 1],[center_x + 1, center_y - 1],[center_x + 1, center_y + 1],[center_x - 1, center_y + 1],[center_x - 1, center_y - 1]]
        points_flat = np.array(unique_points).flatten().tolist()
        try:
            img = Image.new('L', (w, h), 0)
            ImageDraw.Draw(img).polygon(points_flat, outline=1, fill=1)
            masks.append(torch.from_numpy(np.array(img)).bool())
        except Exception as e:
            print(f"Warning: Error drawing polygon: {e}")
            center_x, center_y = int(np.mean(points_np[:, 0])), int(np.mean(points_np[:, 1]))
            mask = torch.zeros(h, w, dtype=torch.bool)
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    x, y = center_x + dx, center_y + dy
                    if 0 <= x < w and 0 <= y < h: mask[y, x] = True
            masks.append(mask)
    return torch.stack(masks) if masks else torch.empty(0, h, w, dtype=torch.bool)


class ContourFormerHead(BaseHead, nn.Module):
    """ContourFormer segmentation head."""
    
    def __init__(self, num_classes: int, backbone_type: str = 'dino', image_size: int = 224,
                 num_queries: int = 100, num_points: int = 50, hidden_dim: int = 256, 
                 nheads: int = 8, num_decoder_layers: int = 6, n_points: int = 4):
        BaseHead.__init__(self, num_classes, backbone_type, image_size)
        nn.Module.__init__(self)
        
        self.num_queries = num_queries
        self.num_points = num_points
        self.hidden_dim = hidden_dim
        
        # --- FIX #3: ADAPTER LOGIC FOR SIMPLE BACKBONES ---
        # This logic checks if a complex FPN is needed or if a simple adapter will suffice.
        self.use_fpn = backbone_type not in ['simple'] # Add other simple backbones here if needed

        if self.use_fpn:
            self.backbone_net = GenericBackboneWithFPN(backbone_type, fpn_out_channels=hidden_dim, dummy_input_size=image_size)
            self.adapter = None # FPN handles adaptation
        else:
            # For simple backbones, we don't use the FPN wrapper.
            self.backbone_net = get_backbone(backbone_type)
            
            # Discover the output channels of the simple backbone
            dummy = torch.randn(1, 3, image_size, image_size)
            with torch.no_grad():
                backbone_out = self.backbone_net(dummy)
                # Handle both tensor and dictionary output from simple backbones
                if isinstance(backbone_out, dict):
                    backbone_out_channels = list(backbone_out.values())[0].shape[1]
                else:
                    backbone_out_channels = backbone_out.shape[1]

            # Create a 1x1 Conv adapter to project features to the required hidden_dim
            self.adapter = nn.Conv2d(backbone_out_channels, hidden_dim, kernel_size=1)
        # --- END FIX ---

        self.input_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.ref_point_embed = nn.Embedding(num_queries, 2)
        
        self.transformer = SimpleDeformableTransformer(
            d_model=hidden_dim, 
            nheads=nheads,
            num_decoder_layers=num_decoder_layers, 
            n_points=n_points
        )
        
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.coord_embed = nn.Linear(hidden_dim, num_points * 2)
        
        # Loss computation setup
        matcher = HungarianMatcher(
            cost_class=1, 
            cost_point=5, 
            cost_giou=2, 
            cost_point_init=10, 
            cost_decay=0.95, 
            warmup_epochs=10
        )
        weight_dict = {'loss_ce': 1.0, 'loss_point': 1.0}
        self.criterion = SetCriterionContourFormer(
            num_classes, 
            matcher=matcher, 
            weight_dict=weight_dict, 
            eos_coef=0.1, 
            num_points=num_points
        )

    def build_position_encoding(self, feature: torch.Tensor) -> torch.Tensor:
        """Build 2D positional encoding for the feature map."""
        B, C, H, W = feature.shape
        mask = torch.ones(B, H, W, device=feature.device)
        y_embed = mask.cumsum(1, dtype=torch.float32)
        x_embed = mask.cumsum(2, dtype=torch.float32)
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * 2 * math.pi
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * 2 * math.pi
        
        dim_t = torch.arange(C // 2, device=feature.device)
        dim_t = 10000 ** (2 * (dim_t // 2) / (C // 2))
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

    def forward(self, pixel_values: torch.Tensor, targets: Optional[List[Dict]] = None) -> Dict[str, torch.Tensor]:
        # --- FIX: Use the correct path based on whether we have an FPN or an adapter ---
        features = self.backbone_net(pixel_values)

        if self.use_fpn:
            # For complex backbones, take the first feature map from the FPN
            src = features['0']
        else:
            # For simple backbones, adapt the single output feature map
            if isinstance(features, dict):
                features = list(features.values())[0]
            src = self.adapter(features)
        # --- END FIX ---

        src = self.input_proj(src)
        B, C, H, W = src.shape
        
        # Add positional encoding
        pos_embed = self.build_position_encoding(src)
        
        # --- FIX: Flatten spatial dimensions for the transformer ---
        # The transformer expects a sequence of shape [Batch, Height*Width, Channels]
        src_flat = src.flatten(2).permute(0, 2, 1)
        pos_embed_flat = pos_embed.flatten(2).permute(0, 2, 1)
        # --- END FIX ---

        # Prepare queries and reference points
        query_input = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)
        ref_points = self.ref_point_embed.weight.unsqueeze(0).repeat(B, 1, 1).sigmoid()
        
        # Apply transformer using the flattened features
        hs = self.transformer(query_input, src_flat + pos_embed_flat, ref_points, (H, W))
        
        # Generate predictions
        outputs_classes = [self.class_embed(h) for h in hs]
        outputs_coords = [self.coord_embed(h).sigmoid().view(B, self.num_queries, self.num_points, 2) for h in hs]
        
        out = {
            'pred_logits': outputs_classes[-1], 
            'pred_coords': outputs_coords[-1]
        }
        
        if self.training and targets is not None:
            indices = self.criterion.matcher(out, targets)
            losses = self.criterion(out, targets, indices=indices)
            
            # Add auxiliary losses
            for i, (c, p) in enumerate(zip(outputs_classes[:-1], outputs_coords[:-1])):
                aux_out = {'pred_logits': c, 'pred_coords': p}
                aux_losses = self.criterion(aux_out, targets, indices=indices)
                for k, v in aux_losses.items():
                    losses[f"{k}_aux{i}"] = v
            return out, losses
        return out
