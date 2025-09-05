"""
Lightweight DETR-style detection head based on LW-DETR.

This module implements a DETR-style detection head inspired by the paper:
"LW-DETR: A Transformer Replacement to YOLO for Real-Time Detection"
(arXiv:2406.03459v1) .

Features:
- Flexible backbone wrapper: Supports ViT (including DINO) and CNN backbones.
- Deformable-attention decoder, as used in LW-DETR.
- Hungarian matcher and a loss set including the IoU-aware BCE loss (IA-BCE)
  from the LW-DETR paper.
- Minimal external dependencies: timm, scipy.
- Preserves DINO-style single-stage query and box prediction logic.

Install requirements:
    pip install timm scipy
"""

from copy import deepcopy
from collections import OrderedDict
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision.ops import FeaturePyramidNetwork, box_convert, generalized_box_iou

# optional: timm-based/backbone helper. Replace with your project's get_backbone.
import timm  # noqa: F401
from ..backbones import get_backbone  # project-specific

# --- BaseHead (dummy for self-contained example) ---


class BaseHead:
    def __init__(self, num_classes: int, backbone_type: str, image_size: int, *args, **kwargs):
        self.num_classes = num_classes
        self.backbone_type = backbone_type
        self.image_size = image_size
        print(f"Initialized BaseHead with {num_classes} classes and backbone '{backbone_type}'.")


# --- Deformable attention (attempt import, fallback if needed) ---


try:
    from torchvision.ops.deformable_attention import DeformableAttention
    print("INFO: Successfully imported torchvision DeformableAttention.")
except ImportError as e:
    print("=" * 60)
    print("CRITICAL WARNING: Failed to import the official DeformableAttention.")
    print(f"THE IMPORT ERROR WAS: {e}")
    print("Falling back to the pure-PyTorch implementation, which is causing errors.")
    print("Please check your torch/torchvision/CUDA installation.")
    print("=" * 60)
    class PurePyTorchDeformableAttention(nn.Module):
        def __init__(self, d_model, n_levels, n_heads, n_points):
            super().__init__()
            self.n_heads = n_heads
            self.n_levels = n_levels
            self.n_points = n_points
            self.d_model = d_model
            self.head_dim = d_model // n_heads
        
            self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
            self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
            self.value_proj = nn.Linear(d_model, d_model)
            self.output_proj = nn.Linear(d_model, d_model)
            self._reset_parameters()
        
        def _reset_parameters(self):
            nn.init.constant_(self.sampling_offsets.weight, 0.0)
            nn.init.constant_(self.sampling_offsets.bias, 0.0)
            nn.init.xavier_uniform_(self.attention_weights.weight)
            nn.init.constant_(self.attention_weights.bias, 0.0)
            nn.init.xavier_uniform_(self.value_proj.weight)
            nn.init.constant_(self.value_proj.bias, 0.0)
            nn.init.xavier_uniform_(self.output_proj.weight)
            nn.init.constant_(self.output_proj.bias, 0.0)
        
        def forward(
            self,
            query,
            reference_points,
            input_flatten,
            input_spatial_shapes,
            input_level_start_index,
            input_padding_mask=None,
        ):
            N, Lq, C = query.shape
            N2, L, C2 = input_flatten.shape
            assert N == N2 and C == C2, "Shape mismatch between query and input_flatten"
            value = self.value_proj(input_flatten).view(N, L, self.n_heads, self.head_dim)
            offsets = self.sampling_offsets(query).view(N, Lq, self.n_heads, self.n_levels, self.n_points, 2)
            attention_weights = self.attention_weights(query).view(N, Lq, self.n_heads, self.n_levels, self.n_points)
            attention_weights = F.softmax(attention_weights, -1)
            
            if reference_points.shape[-1] == 2:
                ref = reference_points.unsqueeze(2).unsqueeze(4)
                offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
                offset_normalizer = offset_normalizer[None, None, None, :, None, :]
                sampling_locations = ref + offsets / offset_normalizer
            else:
                ref_xy = reference_points[..., :2].unsqueeze(2).unsqueeze(3).unsqueeze(4)
                ref_wh = reference_points[..., 2:].unsqueeze(2).unsqueeze(3).unsqueeze(4)
                sampling_locations = ref_xy + offsets / self.n_points * ref_wh * 0.5
            
            splits = [H * W for H, W in input_spatial_shapes]
            value_list = value.split(splits, dim=1)
            sampling_grids = 2 * sampling_locations - 1
            sampled_value_list = []
            
            for lvl, (H, W) in enumerate(input_spatial_shapes):
                v_l = value_list[lvl].transpose(1, 2).reshape(N * self.n_heads, self.head_dim, H, W)
                grid_l = sampling_grids[:, :, :, lvl].permute(0, 2, 1, 3, 4).reshape(N * self.n_heads, Lq * self.n_points, 2)
                grid_l = grid_l.unsqueeze(1)
                sampled = F.grid_sample(v_l, grid_l, mode="bilinear", padding_mode="zeros", align_corners=False)
                sampled = sampled.squeeze(2).reshape(N, self.n_heads, self.head_dim, Lq, self.n_points)
                sampled_value_list.append(sampled)
            
            sampled_values = torch.stack(sampled_value_list, dim=4)  # (N, n_heads, head_dim, Lq, n_levels, n_points)
            attn = attention_weights.permute(0, 2, 1, 3, 4)  # (N, n_heads, Lq, n_levels, n_points)
            attn = attn.unsqueeze(2)  # (N, n_heads, 1, Lq, n_levels, n_points)
            # Broadcast multiplication and sum over n_points and n_levels
            output = (sampled_values * attn).sum(-1).sum(-1)  # Sum over n_points then n_levels
            output = output.permute(0, 3, 1, 2).reshape(N, Lq, C)
            return self.output_proj(output)

    DeformableAttention = PurePyTorchDeformableAttention  # type: ignore


# --- Helpers ---


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(min=0.0, max=1.0)
    x1 = x.clamp(min=eps)
    x2 = (1.0 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def _get_clones(module: nn.Module, N: int) -> nn.ModuleList:
    return nn.ModuleList([deepcopy(module) for _ in range(N)])


# --- Backbone wrapper ---

# 🚀 NEW: Path Aggregation Network (PANet) Module
class PANetFPN(nn.Module):
    """
    A Path Aggregation Network (PANet) that enhances a standard FPN.
    This module takes multi-scale features, processes them through a top-down
    FPN pathway, and then adds a bottom-up pathway to improve localization.
    """
    def __init__(self, in_channels_list: list, out_channels: int):
        super().__init__()
        self.out_channels = out_channels

        # 1. Standard FPN for the top-down pathway
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
        )

        # 2. Layers for the new bottom-up pathway
        num_panet_blocks = len(in_channels_list) - 1
        self.panet_convs = nn.ModuleList()
        for _ in range(num_panet_blocks):
            self.panet_convs.append(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
            )
        
        # 3. Output convolution for each final level
        self.output_convs = nn.ModuleList()
        for _ in range(len(in_channels_list)):
            self.output_convs.append(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )

    def forward(self, features_dict: OrderedDict) -> OrderedDict:
        # Run top-down pathway (FPN)
        fpn_outputs = self.fpn(features_dict)
        fpn_output_list = list(fpn_outputs.values())

        # Run bottom-up pathway (PANet)
        panet_outputs = [fpn_output_list[0]] # The first level is unchanged
        for i in range(len(fpn_output_list) - 1):
            downsampled_prev = self.panet_convs[i](panet_outputs[-1])
            fused_feature = fpn_output_list[i + 1] + downsampled_prev
            panet_outputs.append(fused_feature)

        # Apply final convolutions and format output
        final_outputs = OrderedDict()
        for i, (feat, conv) in enumerate(zip(panet_outputs, self.output_convs)):
             final_outputs[str(i)] = conv(feat)

        return final_outputs

# UPDATED: This class now uses the PANetFPN
class GenericBackboneWithFPN(nn.Module):
    """
    A generic wrapper to add a Feature Pyramid Network to a backbone.
    This version has been updated to use the more powerful PANetFPN.
    """
    def __init__(
        self, backbone_type: str = 'dino', fpn_out_channels: int = 256, dummy_input_size: int = 224
    ):
        super().__init__()
        self.backbone = get_backbone(backbone_type)
        self.out_channels = fpn_out_channels

        # Discover output channels by running a dummy forward pass
        dummy_input = torch.randn(1, 3, dummy_input_size, dummy_input_size)
        with torch.no_grad():
            feat_dict = self.backbone(dummy_input)

        self._feature_keys = sorted(feat_dict.keys())
        in_channels_list = [feat_dict[k].shape[1] for k in self._feature_keys]

        # Use the new PANetFPN module
        self.fpn = PANetFPN(
            in_channels_list=in_channels_list, out_channels=self.out_channels
        )

    @property
    def num_feature_levels(self) -> int:
        return len(self._feature_keys)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extracts features and processes them through the PANetFPN."""
        features = self.backbone(x)
        ordered_features = OrderedDict((k, features[k]) for k in self._feature_keys)
        # The call is the same, but now it uses the more powerful PANet
        fpn_output = self.fpn(ordered_features)
        return fpn_output


# --- Decoder layer ---


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int = 256, n_heads: int = 8, d_ffn: int = 1024, dropout: float = 0.1, n_levels: int = 4, n_points: int = 4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.cross_attn = DeformableAttention(d_model, n_levels, n_heads, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, tgt, memory, memory_padding_mask=None, ref_points_c=None, memory_spatial_shapes=None, level_start_index=None):
        q = k = v = tgt
        tgt2 = self.self_attn(q, k, v)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.cross_attn(
            query=tgt,
            reference_points=ref_points_c,
            input_flatten=memory,
            input_spatial_shapes=memory_spatial_shapes,
            input_level_start_index=level_start_index,
            input_padding_mask=memory_padding_mask,
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt


# --- Matcher & losses ---


class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 2.0, cost_bbox: float = 5.0, cost_giou: float = 2.0, cost_mask: float = 2.0, cost_dice: float = 2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
        out_bbox = outputs["pred_boxes"].flatten(0, 1)

        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        cost_class = -out_prob[:, tgt_ids]
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_giou = -generalized_box_iou(box_convert(out_bbox, "cxcywh", "xyxy"), box_convert(tgt_bbox, "cxcywh", "xyxy"))

        C = self.cost_bbox * cost_bbox + self.cost_giou * cost_giou + self.cost_class * cost_class

        if "pred_masks" in outputs and "masks" in targets[0]:
            out_masks = outputs["pred_masks"].flatten(0, 1)
            tgt_masks = torch.cat([v["masks"] for v in targets], dim=0)
            tgt_masks = F.interpolate(tgt_masks.unsqueeze(1), size=out_masks.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
            out_flat, tgt_flat = out_masks.sigmoid().flatten(1), tgt_masks.float().flatten(1)

            numerator = 2 * (out_flat @ tgt_flat.T)
            denominator = out_flat.sum(-1)[:, None] + tgt_flat.sum(-1)[None, :]
            cost_dice = -(numerator + 1) / (denominator.clamp(min=1e-6) + 1)

            n_pred, n_tgt = out_flat.shape[0], tgt_flat.shape[0]
            out_expanded = out_flat.unsqueeze(1).expand(n_pred, n_tgt, -1)
            tgt_expanded = tgt_flat.unsqueeze(0).expand(n_pred, n_tgt, -1)
            cost_mask = F.binary_cross_entropy(out_expanded, tgt_expanded, reduction="none").mean(-1)

            C = C + (self.cost_mask * cost_mask + self.cost_dice * cost_dice)

        C = C.view(bs, num_queries, -1).cpu()
        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def ia_bce_loss(pred_scores: torch.Tensor, target_classes: torch.Tensor, target_ious: torch.Tensor, alpha: float = 0.25) -> torch.Tensor:
    """Implements the IoU-aware BCE loss from the LW-DETR paper."""
    pred_prob = pred_scores.sigmoid()
    target_t = torch.zeros_like(pred_prob)

    pos_mask = target_classes >= 0
    if pos_mask.any():
        b_idx, q_idx = pos_mask.nonzero(as_tuple=True)
        cls_idx = target_classes[b_idx, q_idx]
        u = target_ious[b_idx, q_idx].clamp(0.0, 1.0)
        t_vals = (pred_prob[b_idx, q_idx, cls_idx].clamp(1e-6) ** alpha) * (u ** (1 - alpha))
        target_t[b_idx, q_idx, cls_idx] = t_vals

    bce = F.binary_cross_entropy(pred_prob, target_t, reduction="none")
    neg_mask = target_classes < 0
    neg_weights = torch.where(neg_mask.unsqueeze(-1), pred_prob ** 2, torch.ones_like(pred_prob))
    return (bce * neg_weights).sum() / max(1, pos_mask.numel())


def dice_loss(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute Dice loss for masks (improved version matching deformable_detr)."""
    inputs = inputs.sigmoid().flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    # Add small epsilon for numerical stability and handle empty case
    loss = 1 - (numerator + 1e-5) / (denominator + 1e-5)
    return loss.mean()


class SetCriterion(nn.Module):
    def __init__(self, num_classes: int, matcher: nn.Module, weight_dict: Dict[str, float], eos_coef: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        
        # Weight for the "no object" class in classification loss (like deformable_detr)
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    @staticmethod
    def _get_permutation_idx(indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def loss_labels(self, outputs, targets, indices, num_objects):
        """Classification loss using standard cross-entropy (like deformable_detr) instead of IoU-aware BCE."""
        src_logits = outputs['pred_logits']
        idx = self._get_permutation_idx(indices)
        
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        return {"loss_cls": loss_ce}

    def loss_boxes(self, outputs, targets, indices, num_objects):
        """Bounding box losses (same as deformable_detr)."""
        idx = self._get_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        
        src_boxes_xyxy = box_convert(src_boxes, in_fmt="cxcywh", out_fmt="xyxy")
        target_boxes_xyxy = box_convert(target_boxes, in_fmt="cxcywh", out_fmt="xyxy")
        loss_giou = 1 - torch.diag(generalized_box_iou(src_boxes_xyxy, target_boxes_xyxy))

        return {
            "loss_bbox": loss_bbox.sum() / num_objects,
            "loss_giou": loss_giou.sum() / num_objects
        }

    def loss_masks(self, outputs, targets, indices, num_objects):
        """Mask losses (same as deformable_detr)."""
        if "pred_masks" not in outputs or "masks" not in targets[0]:
            return {"loss_mask": torch.tensor(0.0, device=next(iter(outputs.values())).device),
                    "loss_dice": torch.tensor(0.0, device=next(iter(outputs.values())).device)}
            
        src_idx = self._get_permutation_idx(indices)
        src_masks = outputs["pred_masks"][src_idx]
        
        target_masks = torch.cat([t['masks'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        # Resize target masks to match predicted mask size
        target_masks_resized = F.interpolate(
            target_masks.unsqueeze(1), 
            size=src_masks.shape[-2:], 
            mode="bilinear", 
            align_corners=False
        ).squeeze(1)
        
        # Use float() for BCE and Dice compatibility
        target_masks_resized = target_masks_resized.to(src_masks.dtype)
        
        # BCE loss
        loss_mask = F.binary_cross_entropy_with_logits(src_masks, target_masks_resized, reduction='none')
        loss_mask = loss_mask.mean(dim=(1, 2))  # Average over pixels

        # Dice loss
        loss_dice = dice_loss(src_masks, target_masks_resized)
        
        return {
            "loss_mask": loss_mask.sum() / num_objects,
            "loss_dice": loss_dice
        }

    def forward(self, outputs, targets):
        """Compute losses (following deformable_detr pattern)."""
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        indices = self.matcher(outputs_without_aux, targets)

        # Sum objects across all GPUs for consistent normalization
        num_objects = sum(len(t["labels"]) for t in targets)
        num_objects_tensor = torch.as_tensor(
            [num_objects], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        num_objects = torch.clamp(num_objects_tensor, min=1).item()

        losses = {}
        for loss_fn_name in ['labels', 'boxes', 'masks']:
            loss_map = getattr(self, f"loss_{loss_fn_name}")(outputs, targets, indices, num_objects)
            losses.update(loss_map)
                    
        # Apply final weights
        weighted_losses = {k: v * self.weight_dict[k] for k, v in losses.items() if k in self.weight_dict}
        return weighted_losses


# --- Main DETR head ---
class LWDETRHead(BaseHead, nn.Module):
    """
    Main LW-DETR detection head with optional Group-DETR sequential decoding.
    """
    def __init__(
        self,
        num_classes: int,
        backbone_type: str = "timm-vit_tiny_patch16_224",
        image_size: int = 640,
        # --- Architecture Hyperparameters (Defaults for LW-DETR-tiny) ---
        d_model: int = 192,
        num_queries: int = 100,
        num_decoder_layers: int = 3,
        n_heads: int = 8,
        d_ffn: int = 1024,
        mask_dim: int = 16,
        # --- Group-DETR Specific Parameter ---
        num_groups: int = 1,
        **kwargs,
    ):
        BaseHead.__init__(self, num_classes, backbone_type, image_size, **kwargs)
        nn.Module.__init__(self)

        self.d_model = d_model
        self.num_queries = num_queries
        self.num_groups = num_groups
        
        if self.num_groups > 1:
            assert self.num_queries % self.num_groups == 0, \
                f"num_queries ({self.num_queries}) must be divisible by num_groups ({self.num_groups})"

        # This line automatically uses the new PANet-based backbone wrapper
        self.backbone = GenericBackboneWithFPN(
            backbone_type=backbone_type, fpn_out_channels=d_model, dummy_input_size=image_size
        )
        self.num_feature_levels = self.backbone.num_feature_levels

        self.query_embed = nn.Embedding(num_queries, d_model)

        decoder_layer = DeformableTransformerDecoderLayer(d_model=d_model, n_heads=n_heads, d_ffn=d_ffn, n_levels=self.num_feature_levels)
        self.transformer_decoder = nn.ModuleList(_get_clones(decoder_layer, num_decoder_layers))

        # Prediction heads
        self.class_embed = nn.Linear(d_model, num_classes + 1)  # +1 for background class (like deformable_detr)
        self.bbox_embed = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 4))
        self.iou_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, 1))
        self.query_scale = nn.Linear(d_model, 4)
        self.mask_embed = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, mask_dim))
        self.pixel_proj = nn.Conv2d(d_model, mask_dim, kernel_size=1)

        self._init_criterion()

    def _init_criterion(self):
        # Align with deformable_detr for better convergence
        matcher = HungarianMatcher(cost_class=1.0, cost_bbox=5.0, cost_giou=2.0)  # Reduced cost_class from 2.0 to 1.0
        
        # Align loss weights with deformable_detr for better convergence
        weight_dict = {"loss_cls": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0, "loss_mask": 1.0, "loss_dice": 1.0}
        
        self.criterion = SetCriterion(self.num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=0.1)

    def forward(self, pixel_values: torch.Tensor, targets: Optional[List[Dict]] = None):
        device = pixel_values.device
        feature_maps = self.backbone(pixel_values)
        srcs = list(feature_maps.values())

        memory_spatial_shapes_list = []
        memory_list = []

        for src in srcs:
            bs, c, h, w = src.shape
            memory_list.append(src.flatten(2).transpose(1, 2))
            memory_spatial_shapes_list.append([h, w])

        memory = torch.cat(memory_list, dim=1)
        memory_spatial_shapes = torch.as_tensor(memory_spatial_shapes_list, dtype=torch.long, device=device)
        sizes = [h * w for h, w in memory_spatial_shapes_list]
        level_start_index = torch.as_tensor([0] + sizes[:-1], dtype=torch.long, device=device).cumsum(0)

        # Decoder
        query_embeds = self.query_embed.weight.unsqueeze(0).expand(pixel_values.shape[0], -1, -1)
        reference_points = self.query_scale(query_embeds).sigmoid()
        decoder_output = query_embeds

        if self.num_groups == 1:
            for layer in self.transformer_decoder:
                decoder_output = layer(
                    tgt=decoder_output,
                    memory=memory,
                    ref_points_c=reference_points,
                    memory_spatial_shapes=memory_spatial_shapes,
                    level_start_index=level_start_index,
                )
        else: # Group-DETR behavior
            group_size = self.num_queries // self.num_groups
            all_group_outputs = []
            
            for i in range(self.num_groups):
                start, end = i * group_size, (i + 1) * group_size
                group_queries = decoder_output[:, start:end, :]
                group_ref_points = reference_points[:, start:end, :]

                current_tgt = group_queries
                for layer in self.transformer_decoder:
                    current_tgt = layer(
                        tgt=current_tgt,
                        memory=memory,
                        ref_points_c=group_ref_points,
                        memory_spatial_shapes=memory_spatial_shapes,
                        level_start_index=level_start_index,
                    )
                
                all_group_outputs.append(current_tgt)

                if i < self.num_groups - 1:
                    # Fix: Remove .detach() to allow gradient flow between groups
                    # This enables proper inter-group learning in Group-DETR
                    decoder_output = torch.cat([
                        decoder_output[:, :start, :], 
                        current_tgt, 
                        decoder_output[:, end:, :]
                    ], dim=1)
                    
                    # Optional: Apply gradient scaling to prevent instability
                    # This scales gradients by 0.5 to maintain stability while preserving flow
                    decoder_output = decoder_output * 0.5 + decoder_output.detach() * 0.5

            decoder_output = torch.cat(all_group_outputs, dim=1)

        logits = self.class_embed(decoder_output)
        pred_boxes_delta = self.bbox_embed(decoder_output)

        prop_center = reference_points[..., :2]
        prop_size = reference_points[..., 2:]
        delta_center = pred_boxes_delta[..., :2]
        delta_size = pred_boxes_delta[..., 2:]
        pred_center = delta_center * prop_size + prop_center
        pred_size = delta_size.exp() * prop_size
        pred_boxes = torch.cat([pred_center, pred_size], dim=-1).sigmoid()

        pred_iou = self.iou_head(decoder_output)
        mask_embeds = self.mask_embed(decoder_output)
        pixel_feats = self.pixel_proj(srcs[0])
        pred_masks = torch.einsum("bqd,bdhw->bqhw", mask_embeds, pixel_feats)

        outputs = {"pred_logits": logits, "pred_boxes": pred_boxes, "pred_iou": pred_iou, "pred_masks": pred_masks}

        if self.training and targets is not None:
            losses = self.criterion(outputs, targets)
            return outputs, losses
        return outputs
