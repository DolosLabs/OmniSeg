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


class DeformableCrossAttention(nn.Module):
    """
    FIXED: Functional Deformable Cross-Attention.
    This module now correctly samples features from memory using query-predicted
    offsets and attention weights, which is critical for convergence.
    """
    def __init__(self, d_model, n_heads, n_points):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model must be divisible by n_heads, but got {d_model} and {n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_points = n_points
        self.head_dim = d_model // n_heads

        self.value_proj = nn.Linear(d_model, d_model)
        self.sampling_offset_proj = nn.Linear(d_model, n_heads * n_points * 2)
        self.attention_weight_proj = nn.Linear(d_model, n_heads * n_points)
        self.output_proj = nn.Linear(d_model, d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        # Initialize sampling offsets to zero to start with local attention
        nn.init.zeros_(self.sampling_offset_proj.weight)
        nn.init.zeros_(self.sampling_offset_proj.bias)
        # Initialize attention weights to be uniform
        nn.init.zeros_(self.attention_weight_proj.weight)
        nn.init.constant_(self.attention_weight_proj.bias, 1.0 / self.n_points)

    def forward(self, query, memory, reference_points, spatial_shape):
        N, Len_q, C = query.shape
        H, W = spatial_shape

        value = self.value_proj(memory)
        value = value.view(N, H, W, self.n_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        value = value.reshape(N * self.n_heads, H, W, self.head_dim).permute(0, 3, 1, 2)

        sampling_offsets = self.sampling_offset_proj(query).view(N, Len_q, self.n_heads, self.n_points, 2)
        attention_weights = self.attention_weight_proj(query).view(N, Len_q, self.n_heads, self.n_points).softmax(-1)

        ref_points_expanded = reference_points.unsqueeze(2).unsqueeze(3).expand(-1, -1, self.n_heads, self.n_points, -1)
        
        # Scale offsets to be relative to the feature map size
        spatial_scale = torch.tensor([W, H], device=query.device).view(1, 1, 1, 1, 2)
        sampling_locations = ref_points_expanded + (sampling_offsets / spatial_scale)
        
        grid = 2 * sampling_locations - 1
        grid = grid.permute(0, 2, 1, 3, 4).flatten(0, 1)

        sampled_features = F.grid_sample(
            value, grid, mode='bilinear', padding_mode='zeros', align_corners=False
        )
        sampled_features = sampled_features.view(N, self.n_heads, self.head_dim, Len_q, self.n_points)
        sampled_features = sampled_features.permute(0, 3, 1, 4, 2).contiguous()

        attention_weights = attention_weights.unsqueeze(-1)
        output = (sampled_features * attention_weights).sum(dim=3)
        output = output.permute(0, 2, 1, 3).contiguous().view(N, Len_q, C)
        
        return self.output_proj(output)


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
        
        if not targets or all(len(t.get("labels", [])) == 0 for t in targets):
            return [(torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)) for _ in range(bs)]
        
        # OPTIMIZATION: Removed temperature scaling for simplicity and stability.
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        out_points = outputs["pred_coords"].flatten(0, 1)
        
        tgt_ids = torch.cat([v["labels"] for v in targets if "labels" in v and len(v["labels"]) > 0])
        tgt_points = torch.cat([v["coords"] for v in targets if "coords" in v and len(v["coords"]) > 0])
        
        if len(tgt_ids) == 0:
            return [(torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)) for _ in range(bs)]
        
        cost_class = -out_prob[:, tgt_ids]
        cost_point = torch.cdist(out_points.flatten(1), tgt_points.flatten(1), p=1)
        
        if self.epoch < self.warmup_epochs:
            C = self.cost_point_init * cost_point
        else:
            cost_point_w = self.base_cost_point
            C = self.base_cost_class * cost_class + cost_point_w * cost_point
        
        C = C.view(bs, num_queries, -1).cpu()
        sizes = [len(v["coords"]) for v in targets]
        indices = []
        
        for i, c in enumerate(C.split(sizes, -1)):
            if sizes[i] == 0:
                indices.append((torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)))
            else:
                row_ind, col_ind = linear_sum_assignment(c[i])
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
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, weight=self.empty_weight)
        return {'loss_ce': loss_ce}

    def loss_points(self, outputs, targets, indices, num_instances):
        idx = self._get_src_permutation_idx(indices)
        if len(idx[0]) == 0:
            return {'loss_point': torch.tensor(0., device=outputs['pred_coords'].device)}
        
        src_points = outputs['pred_coords'][idx]
        target_points = torch.cat([t['coords'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        # Normalize by number of instances and points per instance for consistent loss scale
        loss_point = F.l1_loss(src_points, target_points, reduction='sum') / (num_instances * self.num_points)
        return {'loss_point': loss_point}

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_num_instances(self, targets, device):
        num_instances = sum(len(t["labels"]) for t in targets)
        num_instances = torch.as_tensor([num_instances], dtype=torch.float, device=device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(num_instances)
        return torch.clamp(num_instances, min=1).item()

    def forward(self, outputs, targets):
        indices = self.matcher(outputs, targets)
        num_instances = self.get_num_instances(targets, outputs['pred_logits'].device)
        
        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices, num_instances))
        losses.update(self.loss_points(outputs, targets, indices, num_instances))
        
        total_loss = sum(losses[k] * self.weight_dict.get(k, 1.0) for k in losses.keys())
        losses['total_loss'] = total_loss
        return losses, indices


def masks_to_contours(masks, num_points=50):
    """(Robust) Convert binary masks to fixed-size contour point sets."""
    contours_list, valid_indices = [], []
    for i, mask in enumerate(masks):
        if mask.sum() == 0: continue
        
        mask_np = mask.cpu().numpy().astype(np.uint8)
        contours = find_contours(mask_np, 0.5)
        
        if not contours: continue
        
        contour = sorted(contours, key=len, reverse=True)[0]
        if len(contour) < 4: continue
        
        contour = np.flip(contour, axis=1).astype(np.float64) # (y,x) -> (x,y)
        if not np.allclose(contour[0], contour[-1]):
            contour = np.vstack([contour, contour[0:1]])
        
        contour_tensor = torch.from_numpy(contour).float()
        contour_tensor[:, 0].clamp_(0, mask.shape[1] - 1).div_(mask.shape[1])
        contour_tensor[:, 1].clamp_(0, mask.shape[0] - 1).div_(mask.shape[0])

        path = contour_tensor.t()
        distances = (path[:, 1:] - path[:, :-1]).pow(2).sum(dim=0).sqrt()
        cum_distances = torch.cat([torch.zeros(1), distances.cumsum(0)])

        if cum_distances[-1] < 1e-4: continue
        
        try:
            f = interp1d(cum_distances.numpy(), path.numpy(), kind='linear', axis=1, fill_value="extrapolate")
            new_distances = np.linspace(0, cum_distances[-1].item(), num_points)
            sampled_points = torch.from_numpy(f(new_distances).T).float()
        except Exception:
            # Fallback to linear sampling of indices
            indices = torch.linspace(0, len(contour)-1, num_points).long()
            sampled_points = contour_tensor[indices]

        if not torch.isfinite(sampled_points).all(): continue

        contours_list.append(sampled_points.clamp_(0.0, 1.0))
        valid_indices.append(i)

    if not contours_list:
        return torch.empty((0, num_points, 2), dtype=torch.float32), []
    return torch.stack(contours_list), valid_indices


def contours_to_masks(contours, img_shape):
    """(Robust) Convert sets of contour points to binary masks."""
    h, w = img_shape
    masks = []
    
    if len(contours) == 0:
        return torch.empty(0, h, w, dtype=torch.bool)
        
    for contour_set in contours:
        points = contour_set.clone()
        points[:, 0] = torch.clamp(points[:, 0] * w, 0, w - 1)
        points[:, 1] = torch.clamp(points[:, 1] * h, 0, h - 1)
        
        points_list = points.cpu().numpy().tolist()
        
        if len(points_list) < 3:
            # Not enough points to form a polygon, return empty mask
            masks.append(torch.zeros(h, w, dtype=torch.bool))
            continue
            
        try:
            img = Image.new('L', (w, h), 0)
            ImageDraw.Draw(img).polygon([tuple(p) for p in points_list], outline=1, fill=1)
            mask = torch.from_numpy(np.array(img)).bool()
            masks.append(mask)
        except Exception:
            masks.append(torch.zeros(h, w, dtype=torch.bool))
            
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
        
        self.use_fpn = backbone_type not in ['simple']
        if self.use_fpn:
            self.backbone_net = GenericBackboneWithFPN(backbone_type, fpn_out_channels=hidden_dim, dummy_input_size=image_size)
        else:
            self.backbone_net = get_backbone(backbone_type)
            dummy = torch.randn(1, 3, image_size, image_size)
            with torch.no_grad():
                backbone_out = self.backbone_net(dummy)
                backbone_out_channels = list(backbone_out.values())[0].shape[1] if isinstance(backbone_out, dict) else backbone_out.shape[1]
            self.adapter = nn.Conv2d(backbone_out_channels, hidden_dim, kernel_size=1)

        self.input_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.ref_point_embed = nn.Embedding(num_queries, 2)
        
        self.transformer = SimpleDeformableTransformer(
            d_model=hidden_dim, 
            nheads=nheads,
            num_decoder_layers=num_decoder_layers, 
            n_points=n_points,
            dim_feedforward=hidden_dim * 4 # Standard FFN dimension
        )
        
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.coord_embed = nn.Linear(hidden_dim, num_points * 2) # Predicts offsets
        
        matcher = HungarianMatcher(cost_class=2.0, cost_point=5.0) # Adjusted weights
        weight_dict = {'loss_ce': 2.0, 'loss_point': 5.0}
        self.criterion = SetCriterionContourFormer(
            num_classes, 
            matcher=matcher, 
            weight_dict=weight_dict, 
            eos_coef=0.1, 
            num_points=num_points
        )

    def build_position_encoding(self, feature: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feature.shape
        mask = torch.ones(B, H, W, device=feature.device, dtype=torch.bool)
        y_embed = mask.cumsum(1, dtype=torch.float32)
        x_embed = mask.cumsum(2, dtype=torch.float32)
        eps = 1e-6
        y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps)
        x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps)

        dim_t = torch.arange(C // 4, dtype=torch.float32, device=feature.device)
        dim_t = 10000 ** (2 * (dim_t // 2) / (C // 4))
        
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        return torch.cat((pos_y, posx), dim=3).permute(0, 3, 1, 2)

    def forward(self, pixel_values: torch.Tensor, targets: Optional[List[Dict]] = None) -> Dict[str, torch.Tensor]:
        features = self.backbone_net(pixel_values)

        if self.use_fpn:
            src = features['0']
        else:
            features = list(features.values())[0] if isinstance(features, dict) else features
            src = self.adapter(features)

        src = self.input_proj(src)
        B, C, H, W = src.shape
        
        pos_embed = self.build_position_encoding(src)
        src_flat = src.flatten(2).permute(0, 2, 1)
        pos_embed_flat = pos_embed.flatten(2).permute(0, 2, 1)
        
        query_input = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)
        ref_points = self.ref_point_embed.weight.unsqueeze(0).repeat(B, 1, 1).sigmoid()
        
        hs = self.transformer(query_input, src_flat + pos_embed_flat, ref_points, (H, W))
        
        outputs_classes = []
        outputs_coords = []
        expanded_ref_points = ref_points.unsqueeze(2).expand(-1, -1, self.num_points, -1)
        
        for h in hs:
            outputs_classes.append(self.class_embed(h))
            # OPTIMIZATION: Predict offsets from the reference point for stability
            offsets = self.coord_embed(h).tanh().view(B, self.num_queries, self.num_points, 2)
            pred_coords = (expanded_ref_points + offsets).sigmoid() # Sigmoid to keep in [0,1]
            outputs_coords.append(pred_coords)
            
        out = {
            'pred_logits': outputs_classes[-1], 
            'pred_coords': outputs_coords[-1]
        }
        
        if self.training and targets is not None:
            losses, indices = self.criterion(out, targets)
            
            # Add auxiliary losses for deep supervision
            for i, (c, p) in enumerate(zip(outputs_classes[:-1], outputs_coords[:-1])):
                aux_out = {'pred_logits': c, 'pred_coords': p}
                aux_losses, _ = self.criterion(aux_out, targets)
                for k, v in aux_losses.items():
                    if k != 'total_loss':
                        losses[f"{k}_aux{i}"] = v
            
            # Recalculate total loss with auxiliary losses
            total_loss = sum(losses[k] * self.criterion.weight_dict.get(k.split('_aux')[0], 1.0) for k in losses)
            losses['total_loss'] = total_loss
            return out, losses
            
        return out
