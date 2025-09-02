"""ContourFormer segmentation head implementation."""
# --- Fully corrected and refactored ---

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

# Assuming these are in your project structure
# from ..base import BaseHead
# from ..backbones import get_backbone

# Dummy base classes for standalone execution
class BaseHead:
    def __init__(self, num_classes, backbone_type, image_size):
        pass
def get_backbone(backbone_type):
    # This is a placeholder for your actual backbone loader
    import torchvision.models as models
    from torch.nn.modules.container import Sequential
    from torchvision.models._utils import IntermediateLayerGetter
    
    # Using ResNet50 as an example backbone
    backbone = models.resnet50(weights=None, replace_stride_with_dilation=[False, False, True])
    # The keys '0', '1', '2', '3' are just examples; they should match what your FPN expects.
    # Typically, these are layer names like 'layer1', 'layer2', etc.
    return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
    return IntermediateLayerGetter(backbone, return_layers=return_layers)


# ==============================================================================
# UNCHANGED HELPER MODULES
# ==============================================================================

def masks_to_contours(masks, num_points=64):
    """Convert binary masks to fixed-size contour point sets."""
    contours_list, valid_indices = [], []
    for i, mask in enumerate(masks):
        if mask.sum() == 0: continue
        mask_np = mask.cpu().numpy().astype(np.uint8)
        contours = find_contours(mask_np, 0.5)
        if not contours:
            ys, xs = torch.where(mask > 0)
            if len(ys) > 0:
                y_min, y_max = ys.min().item(), ys.max().item()
                x_min, x_max = xs.min().item(), xs.max().item()
                contour = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], dtype=np.float64)
            else:
                continue
        else:
            contour = sorted(contours, key=len, reverse=True)[0]
            if len(contour) < 3:
                ys, xs = torch.where(mask > 0)
                if len(ys) > 0:
                    y_min, y_max = ys.min().item(), ys.max().item()
                    x_min, x_max = xs.min().item(), xs.max().item()
                    contour = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], dtype=np.float64)
                else:
                    continue
            else:
                contour = np.flip(contour, axis=1).astype(np.float64)
        
        if not np.allclose(contour[0], contour[-1], atol=1e-6):
            contour = np.vstack([contour, contour[0:1]])
        
        contour = torch.tensor(contour, dtype=torch.float32)
        contour[:, 0] = torch.clamp(contour[:, 0] / max(mask.shape[1], 1), 0.0, 1.0)
        contour[:, 1] = torch.clamp(contour[:, 1] / max(mask.shape[0], 1), 0.0, 1.0)
        
        path = contour.t()
        distances = torch.sqrt(torch.sum((path[:, 1:] - path[:, :-1])**2, dim=0))
        distance = torch.cat([torch.tensor([0.0]), torch.cumsum(distances, dim=0)])
        
        if distance[-1] < 1e-8:
            sampled_points = contour[torch.linspace(0, len(contour)-1, num_points).long()]
        else:
            try:
                f = interp1d(distance.numpy(), path.numpy(), kind='linear', axis=1, fill_value="extrapolate")
                new_distances = np.linspace(0, distance[-1].item(), num_points)
                sampled_points = torch.from_numpy(f(new_distances).T).to(torch.float32)
            except:
                indices = torch.linspace(0, len(contour)-1, num_points).long()
                sampled_points = contour[torch.clamp(indices, 0, len(contour)-1)]

        if torch.isnan(sampled_points).any() or torch.isinf(sampled_points).any():
            continue # Skip corrupted data

        contours_list.append(torch.clamp(sampled_points, 0.0, 1.0))
        valid_indices.append(i)
    
    if not contours_list:
        return torch.empty((0, num_points, 2), dtype=torch.float32), []
    return torch.stack(contours_list), valid_indices


def contours_to_masks(contours, img_shape):
    """Convert sets of contour points to binary masks."""
    h, w = img_shape
    masks = []
    
    if len(contours) == 0 or (hasattr(contours, 'numel') and contours.numel() == 0):
        return torch.empty(0, h, w, dtype=torch.bool)
    
    for contour_set in contours:
        if contour_set.numel() == 0:
            masks.append(torch.zeros(h, w, dtype=torch.bool))
            continue
            
        points = contour_set.clone()
        points[:, 0] = torch.clamp(points[:, 0] * w, 0, w - 1)
        points[:, 1] = torch.clamp(points[:, 1] * h, 0, h - 1)
        points_np = points.cpu().numpy()
        
        unique_points = [points_np[0]]
        for point in points_np[1:]:
            if not np.allclose(point, unique_points[-1], atol=0.5):
                unique_points.append(point)
        
        if len(unique_points) < 3:
            masks.append(torch.zeros(h, w, dtype=torch.bool))
            continue
            
        points_flat = np.array(unique_points).flatten().tolist()
        
        try:
            img = Image.new('L', (w, h), 0)
            ImageDraw.Draw(img).polygon(points_flat, outline=1, fill=1)
            mask = torch.from_numpy(np.array(img)).bool()
            masks.append(mask)
        except Exception:
            masks.append(torch.zeros(h, w, dtype=torch.bool))
            
    return torch.stack(masks) if masks else torch.empty(0, h, w, dtype=torch.bool)
    
class GenericBackboneWithFPN(nn.Module):
    """Generic wrapper to add FPN to any backbone."""
    
    def __init__(self, backbone_type: str = 'resnet50', fpn_out_channels: int = 256, dummy_input_size: int = 224):
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


class HungarianMatcher(nn.Module):
    """Hungarian matcher (unchanged from original)."""
    
    def __init__(self, cost_class=1, cost_point=1, cost_giou=1, cost_point_init=10, cost_decay=0.95, warmup_epochs=10):
        super().__init__()
        self.base_cost_class = cost_class
        self.base_cost_point = cost_point
        self.cost_giou = cost_giou # Note: This is unused in the contour-only version
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
        
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        out_points = outputs["pred_coords"].flatten(0, 1) # [B*Nq, Nv, 2]
        
        tgt_ids = torch.cat([v["labels"] for v in targets if len(v["labels"]) > 0])
        tgt_points = torch.cat([v["coords"] for v in targets if len(v["coords"]) > 0]) # [total_gt, Nv, 2]

        if len(tgt_ids) == 0:
            return [(torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)) for _ in range(bs)]

        cost_class = -out_prob[:, tgt_ids]
        cost_point = torch.cdist(out_points.flatten(1), tgt_points.flatten(1), p=1)
        
        if self.epoch < self.warmup_epochs:
            C = cost_point
        else:
            cost_point_w = max(self.base_cost_point, self.cost_point_init * (self.cost_decay ** self.epoch))
            C = self.base_cost_class * cost_class + cost_point_w * cost_point
        
        C = C.view(bs, num_queries, -1).cpu()
        sizes = [len(v["coords"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


# ==============================================================================
# CORRECTED AND REFACTORED MODULES
# ==============================================================================

class DecoupledSelfAttention(nn.Module):
    """
    Performs decoupled self-attention as described in the ContourFormer paper.
    1. Self-attention among instance queries.
    2. Self-attention among sub-contour queries.
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.inst_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.sub_contour_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

    def forward(self, inst_query, sub_contour_query):
        # inst_query: [B, Nq, C]
        # sub_contour_query: [B, Nq, Nc, C]
        B, Nq, Nc, C = sub_contour_query.shape
        
        # 1. Self-attention among instance queries
        inst_q_out, _ = self.inst_attn(inst_query, inst_query, value=inst_query)
        
        # 2. Self-attention among sub-contour queries (shared across all instances)
        # Reshape to treat instances as part of the batch for parallel processing
        sc_q = sub_contour_query.permute(0, 2, 1, 3).reshape(B * Nc, Nq, C)
        sc_q_out, _ = self.sub_contour_attn(sc_q, sc_q, value=sc_q)
        sc_q_out = sc_q_out.view(B, Nc, Nq, C).permute(0, 2, 1, 3)

        return sc_q_out + inst_q_out.unsqueeze(2)


class DeformableCrossAttention(nn.Module):
    """
    **CORRECTED** DeformableCrossAttention.
    - Uses bounding boxes as reference areas.
    - Fixed to perform correct feature sampling using grid_sample.
    """
    def __init__(self, d_model, n_heads, n_points):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_points = n_points
        self.head_dim = d_model // n_heads

        self.sampling_offset_mlp = nn.Linear(d_model, n_heads * n_points * 2)
        self.attention_weight_mlp = nn.Linear(d_model, n_heads * n_points)
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, query, memory, reference_boxes, spatial_shape):
        """
        query: [B, Lq, C] where Lq = Nq*Nc
        memory: [B, H*W, C]
        reference_boxes: [B, Lq, 4] with (cx, cy, w, h) in [0,1] format
        """
        B, Lq, C = query.shape
        H, W = spatial_shape
        
        memory = memory.view(B, H, W, C).permute(0, 3, 1, 2) # [B, C, H, W]

        offsets = self.sampling_offset_mlp(query).view(B, Lq, self.n_heads, self.n_points, 2)
        attn_weights = self.attention_weight_mlp(query).view(B, Lq, self.n_heads, self.n_points).softmax(-1)

        ref_boxes = reference_boxes.unsqueeze(2).unsqueeze(3) # [B, Lq, 1, 1, 4]
        ref_xy, ref_wh = ref_boxes[..., :2], ref_boxes[..., 2:]
        
        sampling_locations = ref_xy + offsets * ref_wh * 0.5 # [B, Lq, n_heads, n_points, 2]
        
        # F.grid_sample expects coords in [-1, 1]
        sampling_grid = sampling_locations * 2 - 1
        
        # --- CORRECTED SAMPLING LOGIC ---
        # Reshape grid for sampling: [B * n_heads, Lq, n_points, 2]
        grid = sampling_grid.permute(0, 2, 1, 3, 4).reshape(B * self.n_heads, Lq, self.n_points, 2)
        
        # Reshape memory to match the new batch dimension for multi-head attention
        memory_reshaped = memory.view(B, self.n_heads, self.head_dim, H, W)
        memory_reshaped = memory_reshaped.reshape(B * self.n_heads, self.head_dim, H, W)

        sampled_features = F.grid_sample(
            memory_reshaped, grid, mode='bilinear', padding_mode='zeros', align_corners=False
        ) # Shape: [B * n_heads, C/n_heads, Lq, n_points]
        
        sampled_features = sampled_features.reshape(B, self.n_heads, self.head_dim, Lq, self.n_points)
        sampled_features = sampled_features.permute(0, 3, 1, 4, 2) # -> [B, Lq, n_heads, n_points, C/n_heads]
        
        weighted_features = (sampled_features * attn_weights.unsqueeze(-1)).sum(dim=3) # [B, Lq, n_heads, C/n_heads]
        
        # Concatenate heads and project
        output = weighted_features.reshape(B, Lq, C)
        return self.output_proj(output)


class ContourFormerDecoderLayer(nn.Module):
    """A single decoder layer for ContourFormer using decoupled attention."""
    def __init__(self, d_model=256, n_heads=8, n_points=4, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = DecoupledSelfAttention(d_model, n_heads, dropout)
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

    def forward(self, inst_query, sub_contour_query, memory_feats, reference_boxes, spatial_shape):
        B, Nq, Nc, C = sub_contour_query.shape
        
        # 1. Decoupled Self-Attention (Post-Norm Style)
        q_sa = self.self_attn(inst_query, sub_contour_query)
        combined_query = sub_contour_query + q_sa # The paper implies this combination
        tgt = combined_query.reshape(B, Nq * Nc, C) # Flatten for next layers
        tgt = self.norm1(tgt + self.dropout1(q_sa.reshape(B, Nq * Nc, C)))
        
        # 2. Deformable Cross-Attention
        tgt2 = self.cross_attn(tgt, memory_feats, reference_boxes, spatial_shape)
        tgt = self.norm2(tgt + self.dropout2(tgt2))
        
        # 3. FFN
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout3(tgt2))
        
        return inst_query, tgt.view(B, Nq, Nc, C)


class CFDRHead(nn.Module):
    """Predicts residual logits for the contour point distribution."""
    def __init__(self, d_model, num_points_per_subcontour, num_bins=10):
        super().__init__()
        self.num_bins = num_bins
        self.logit_predictor = nn.Linear(d_model, num_points_per_subcontour * 2 * num_bins)

    def forward(self, query_features):
        return self.logit_predictor(query_features)


def calculate_coords_from_logits(logits, prev_layer_coords, sub_contour_boxes):
    """
    Calculates final coordinates from logits using the CFDR formula.
    **CORRECTED** to use clamp instead of sigmoid.
    """
    B, Nq, Nc, _ = logits.shape
    Nv = prev_layer_coords.shape[2]
    Ns = Nv // Nc
    num_bins = logits.shape[-1] // (Ns * 2)

    dist = logits.reshape(B, Nq, Nc, Ns, 2, num_bins).softmax(dim=-1)
    bin_weights = torch.linspace(-0.5, 0.5, num_bins, device=logits.device)
    
    offsets_dist = (dist * bin_weights).sum(dim=-1)
    
    wh = sub_contour_boxes[..., 2:].unsqueeze(3) # [B, Nq, Nc, 1, 2]
    scaled_offsets = offsets_dist * wh
    
    total_offsets = scaled_offsets.reshape(B, Nq, Nv, 2)
    
    # This is now an iterative refinement: v^l = v^{l-1} + Δv^l
    final_coords = prev_layer_coords + total_offsets
    
    # Use clamp for better boundary behavior
    return torch.clamp(final_coords, 0.0, 1.0)


class SetCriterionContourFormer(nn.Module):
    """MODIFIED Criterion with added Shape Loss."""
    
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

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def loss_labels(self, outputs, targets, indices, num_instances):
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        return {'loss_ce': loss_ce}

    def loss_points(self, outputs, targets, indices, num_instances):
        idx = self._get_src_permutation_idx(indices)
        src_points = outputs['pred_coords'][idx]
        target_points = torch.cat([t['coords'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_point = F.l1_loss(src_points, target_points, reduction='none')
        return {'loss_point': loss_point.sum() / (num_instances * self.num_points * 2)}

    def loss_shape(self, outputs, targets, indices, num_instances):
        """Computes shape loss based on cosine similarity of adjacent point offsets."""
        idx = self._get_src_permutation_idx(indices)
        src_points = outputs['pred_coords'][idx]
        target_points = torch.cat([t['coords'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        src_offsets = F.normalize(src_points - torch.roll(src_points, shifts=1, dims=1), p=2, dim=2)
        target_offsets = F.normalize(target_points - torch.roll(target_points, shifts=1, dims=1), p=2, dim=2)
        
        loss = 1 - (src_offsets * target_offsets).sum(dim=2)
        return {'loss_shape': loss.sum() / (num_instances * self.num_points)}

    def forward(self, outputs, targets):
        indices = self.matcher(outputs, targets)
        num_instances = sum(len(t["labels"]) for t in targets)
        num_instances = torch.as_tensor([num_instances], dtype=torch.float, device=next(iter(outputs.values())).device)
        
        # Handle distributed training
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(num_instances)
            world_size = torch.distributed.get_world_size()
        else:
            world_size = 1
            
        num_instances = torch.clamp(num_instances / world_size, min=1).item()

        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices, num_instances))
        losses.update(self.loss_points(outputs, targets, indices, num_instances))
        losses.update(self.loss_shape(outputs, targets, indices, num_instances))
        
        return {k: v * self.weight_dict[k] for k, v in losses.items() if k in self.weight_dict}

# ==============================================================================
# FINAL INTEGRATED AND CORRECTED CONTOURFORMER HEAD
# ==============================================================================

class ContourFormerHead(BaseHead, nn.Module):
    """
    ContourFormer segmentation head.
    **CORRECTED** with iterative refinement logic.
    """
    
    def __init__(self, num_classes: int, backbone_type: str = 'resnet50', image_size: int = 224,
                 num_queries: int = 300, num_points: int = 64, hidden_dim: int = 256,
                 nheads: int = 8, num_decoder_layers: int = 6, n_points_attn: int = 4,
                 num_sub_contours: int = 8):
        
        BaseHead.__init__(self, num_classes, backbone_type, image_size)
        nn.Module.__init__(self)
        
        self.num_queries = num_queries
        self.num_points = num_points
        self.num_sub_contours = num_sub_contours
        self.num_points_per_sc = num_points // num_sub_contours
        assert num_points % num_sub_contours == 0, "num_points must be divisible by num_sub_contours"

        self.backbone_net = GenericBackboneWithFPN(backbone_type, fpn_out_channels=hidden_dim, dummy_input_size=image_size)
        self.input_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        
        self.instance_query_embed = nn.Embedding(num_queries, hidden_dim)
        self.subcontour_query_embed = nn.Embedding(num_sub_contours, hidden_dim)

        self.decoder_layers = nn.ModuleList([
            ContourFormerDecoderLayer(hidden_dim, nheads, n_points_attn) for _ in range(num_decoder_layers)
        ])
        
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = nn.Linear(hidden_dim, 4)
        self.cfdr_head = CFDRHead(hidden_dim, self.num_points_per_sc)
        
        matcher = HungarianMatcher(cost_class=1, cost_point=5)
        weight_dict = {'loss_ce': 1.0, 'loss_point': 5.0, 'loss_shape': 0.25}
        self.criterion = SetCriterionContourFormer(
            num_classes, matcher, weight_dict, eos_coef=0.1, num_points=num_points
        )

    def build_position_encoding(self, feature: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feature.shape
        mask = torch.ones(B, H, W, device=feature.device)
        y_embed = mask.cumsum(1, dtype=torch.float32)
        x_embed = mask.cumsum(2, dtype=torch.float32)
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * 2 * math.pi
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * 2 * math.pi
        dim_t = torch.arange(C // 2, dtype=torch.float32, device=feature.device)
        dim_t = 10000 ** (2 * (dim_t // 2) / (C // 2))
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

    def forward(self, pixel_values: torch.Tensor, targets: Optional[List[Dict]] = None):
        B = pixel_values.shape[0]
        features = self.backbone_net(pixel_values)
        src = self.input_proj(features['0'])
        H, W = src.shape[-2:]
        
        pos_embed = self.build_position_encoding(src)
        memory = src.flatten(2).permute(0, 2, 1) + pos_embed.flatten(2).permute(0, 2, 1)

        inst_query = self.instance_query_embed.weight.unsqueeze(0).repeat(B, 1, 1)
        sc_query_base = self.subcontour_query_embed.weight
        sub_contour_query = inst_query.unsqueeze(2) + sc_query_base

        # Initial base contour (small circle in the center)
        t = torch.linspace(0, 2 * torch.pi, self.num_points, device=pixel_values.device)
        base_coords = torch.stack([0.5 + 0.2*t.cos(), 0.5 + 0.2*t.sin()], dim=1)
        pred_coords = base_coords.unsqueeze(0).unsqueeze(0).repeat(B, self.num_queries, 1, 1)

        # Initial reference is the whole image for the first layer
        reference_boxes = torch.tensor([0.5, 0.5, 1.0, 1.0], device=pixel_values.device)
        reference_boxes = reference_boxes.view(1, 1, 4).repeat(B, self.num_queries * self.num_sub_contours, 1)
        
        intermediate_outputs = []

        for layer in self.decoder_layers:
            # Flatten sub-contour query for the attention layers
            flat_sc_query = sub_contour_query.reshape(B, self.num_queries * self.num_sub_contours, -1)
            
            inst_query, sub_contour_query = layer(inst_query, sub_contour_query, memory, reference_boxes, (H, W))
            
            # Predict bounding boxes from the updated instance-level query
            avg_inst_query = sub_contour_query.mean(dim=2) # [B, Nq, C]
            pred_boxes = self.bbox_embed(avg_inst_query).sigmoid() # [B, Nq, 4]

            # Predict residual logits for this layer's refinement
            residual_logits = self.cfdr_head(sub_contour_query)
            
            # Sub-contour boxes are just the instance-level boxes repeated
            sc_boxes = pred_boxes.unsqueeze(2).repeat(1, 1, self.num_sub_contours, 1)
            
            # --- ITERATIVE REFINEMENT ---
            # Update the previous layer's coordinates with the new offsets
            pred_coords = calculate_coords_from_logits(residual_logits, pred_coords, sc_boxes)

            # Predict classes from the averaged instance query
            pred_logits = self.class_embed(avg_inst_query)
            
            intermediate_outputs.append({'pred_logits': pred_logits, 'pred_coords': pred_coords})

            # Update reference boxes for the next layer's cross-attention
            reference_boxes = sc_boxes.reshape(B, -1, 4)
            
        out = intermediate_outputs[-1]

        if self.training:
            losses = self.criterion(out, targets)
            
            # Add auxiliary losses from intermediate layers
            for i, aux_out in enumerate(intermediate_outputs[:-1]):
                aux_losses = self.criterion(aux_out, targets)
                for k, v in aux_losses.items():
                    losses[f"{k}_aux{i}"] = v
            return out, losses

        return out
