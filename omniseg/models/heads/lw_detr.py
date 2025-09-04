"""
Lightweight DETR-style detection head based on LW-DETR.

This module implements a DETR-style detection head inspired by the paper:
"LW-DETR: A Transformer Replacement to YOLO for Real-Time Detection"
(arXiv:2406.03459v1) [cite_start][cite: 2].

Features:
- Flexible backbone wrapper: Supports ViT (including DINO) and CNN backbones.
- [cite_start]Deformable-attention decoder, as used in LW-DETR[cite: 101].
- Hungarian matcher and a loss set including the IoU-aware BCE loss (IA-BCE)
  [cite_start]from the LW-DETR paper[cite: 121].
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
            """
            query: (N, Lq, C)
            reference_points: (N, Lq, n_levels, 2/4)
            input_flatten: (N, L, C)
            input_spatial_shapes: (n_levels, 2)  # (H, W)
            input_level_start_index: (n_levels,)
            """
            N, Lq, C = query.shape
            N2, L, C2 = input_flatten.shape
            assert N == N2 and C == C2, "Shape mismatch between query and input_flatten"
    
            # (N, L, n_heads, head_dim)
            value = self.value_proj(input_flatten).view(N, L, self.n_heads, self.head_dim)
    
            # (N, Lq, n_heads, n_levels, n_points, 2)
            offsets = self.sampling_offsets(query).view(N, Lq, self.n_heads, self.n_levels, self.n_points, 2)
    
            # (N, Lq, n_heads, n_levels, n_points)
            attention_weights = self.attention_weights(query).view(N, Lq, self.n_heads, self.n_levels, self.n_points)
            attention_weights = F.softmax(attention_weights, -1)
    
            # compute sampling locations
            if reference_points.shape[-1] == 2:
                ref = reference_points.unsqueeze(2).unsqueeze(4)  # (N, Lq, 1, n_levels, 1, 2)
                offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
                offset_normalizer = offset_normalizer[None, None, None, :, None, :]
                sampling_locations = ref + offsets / offset_normalizer
            else:
                ref_xy = reference_points[..., :2].unsqueeze(2).unsqueeze(3).unsqueeze(4)
                ref_wh = reference_points[..., 2:].unsqueeze(2).unsqueeze(3).unsqueeze(4)
                sampling_locations = ref_xy + offsets / self.n_points * ref_wh * 0.5
    
            # split per level
            splits = [H * W for H, W in input_spatial_shapes]
            value_list = value.split(splits, dim=1)
    
            sampling_grids = 2 * sampling_locations - 1
            sampled_value_list = []
    
            for lvl, (H, W) in enumerate(input_spatial_shapes):
                v_l = value_list[lvl].transpose(1, 2).reshape(N * self.n_heads, self.head_dim, H, W)
                grid_l = sampling_grids[:, :, :, lvl]  # (N, Lq, n_heads, n_points, 2)
                grid_l = grid_l.permute(0, 2, 1, 3, 4).reshape(N * self.n_heads, Lq * self.n_points, 2)
                grid_l = grid_l.unsqueeze(1)  # (N*heads, 1, Lq*n_points, 2)
    
                sampled = F.grid_sample(
                    v_l, grid_l, mode="bilinear", padding_mode="zeros", align_corners=False
                )  # (N*heads, head_dim, 1, Lq*n_points)
    
                sampled = sampled.squeeze(2).reshape(N, self.n_heads, self.head_dim, Lq, self.n_points)
                sampled_value_list.append(sampled)
    
            # (N, n_heads, head_dim, Lq, n_levels, n_points)
            sampled_values = torch.stack(sampled_value_list, dim=4)
    
            # (N, Lq, n_heads, n_levels, n_points) -> align with sampled_values
            attn = attention_weights.permute(0, 2, 3, 1, 4).unsqueeze(2)
            # (N, n_heads, head_dim, Lq, n_levels, n_points) * (N, n_heads, 1, Lq, n_levels, n_points)
            output = (sampled_values * attn).sum(-1).sum(4)  # sum over points and levels
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


class BackboneWithOptionalFPN(nn.Module):
    """
    Flexible backbone wrapper.
    - ViT-like (e.g., DINO, SAM): extracts patch tokens and uses a single projection layer.
      NOTE: This is a simplification of the multi-level feature aggregation and C2f
      [cite_start]projector described in the LW-DETR paper[cite: 98, 119].
    - CNN backbone: processes multi-scale features with a Feature Pyramid Network (FPN).
    """

    def __init__(self, backbone_type: str, out_channels: int, image_size: int, fpn_feature_indices=(1, 2, 3)):
        super().__init__()
        self.backbone_type = backbone_type.lower()
        self.is_vit = "sam" in self.backbone_type or "dino" in self.backbone_type or "vit" in self.backbone_type
        self.out_channels = out_channels
        self.image_size = image_size

        self.backbone = get_backbone(backbone_type)

        if self.is_vit:
            # Determine embedding dim
            if hasattr(self.backbone, "embed_dim"):
                self.embed_dim = self.backbone.embed_dim
            elif hasattr(self.backbone, "dino"): # DINO-specific handling
                self.embed_dim = self.backbone.dino.embeddings.patch_embeddings.out_channels
            else:
                raise AttributeError(f"Cannot determine embed_dim for {type(self.backbone).__name__}")

            # Determine patch size
            if hasattr(self.backbone, "patch_embed"):
                self.patch_size = self.backbone.patch_embed.patch_size[0]
            elif hasattr(self.backbone, "conv_proj"):
                # some timm models use conv_proj
                self.patch_size = self.backbone.conv_proj.kernel_size[0]
            elif hasattr(self.backbone, "dino"): # DINO-specific handling
                self.patch_size = self.backbone.dino.embeddings.patch_embeddings.kernel_size[0]
            else:
                raise AttributeError(f"Cannot determine patch size for {type(self.backbone).__name__}")

            self.grid_size = self.image_size // self.patch_size
            self.fpn = None
            self.proj = nn.Conv2d(self.embed_dim, out_channels, kernel_size=1)
        else:
            # CNN + FPN path
            feature_info = self.backbone.feature_info.channels()
            self.proj = None
            self.fpn = FeaturePyramidNetwork(in_channels_list=feature_info, out_channels=out_channels)

    @property
    def num_feature_levels(self) -> int:
        return 1 if self.is_vit else len(self.backbone.feature_info)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.is_vit:
            # ViT forward (support models with forward_features or plain forward)
            model_output = self.backbone.forward_features(x) if hasattr(self.backbone, "forward_features") else self.backbone(x)

            # Extract a tensor robustly from various possible return types
            features_tensor = None
            if torch.is_tensor(model_output):
                features_tensor = model_output
            elif isinstance(model_output, dict):
                for key in ("last_hidden_state", "hidden_states", "features", "res5"):
                    if key in model_output:
                        features_tensor = model_output[key] if key != "hidden_states" else model_output["hidden_states"][-1]
                        break
                if features_tensor is None:
                    raise KeyError(f"Backbone output dict missing known keys. Available: {list(model_output.keys())}")
            elif isinstance(model_output, (list, tuple)):
                features_tensor = model_output[0]
            else:
                raise TypeError(f"Unsupported backbone output type: {type(model_output)}")

            if features_tensor.ndim == 3:
                # patch tokens: [B, N, C]
                B, N, C = features_tensor.shape
                num_prefix = getattr(self.backbone, "num_prefix_tokens", 1)
                patch_tokens = features_tensor[:, num_prefix:, :]
                h = w = self.grid_size
                if patch_tokens.shape[1] != h * w:
                    raise ValueError(f"Patch token count ({patch_tokens.shape[1]}) != grid ({h*w})")
                feature_map = patch_tokens.transpose(1, 2).reshape(B, C, h, w)
            elif features_tensor.ndim == 4:
                # feature map: [B, C, H, W]
                B, C, H, W = features_tensor.shape
                if (H, W) == (1, 1):
                    h = w = self.grid_size
                    feature_map = F.interpolate(features_tensor, size=(h, w), mode="bilinear", align_corners=False)
                else:
                    feature_map = features_tensor
            else:
                raise TypeError(f"Unsupported feature tensor shape: {features_tensor.shape}")

            projected = self.proj(feature_map)
            return OrderedDict([("0", projected)])
        else:
            # CNN + FPN
            features = self.backbone(x)
            ordered = OrderedDict((str(i), feat) for i, feat in enumerate(features))
            fpn_out = self.fpn(ordered)
            return fpn_out


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
    """Implements the IoU-aware BCE loss from the LW-DETR paper[cite: 121, 122, 125]."""
    pred_prob = pred_scores.sigmoid()
    target_t = torch.zeros_like(pred_prob)

    pos_mask = target_classes >= 0
    if pos_mask.any():
        b_idx, q_idx = pos_mask.nonzero(as_tuple=True)
        cls_idx = target_classes[b_idx, q_idx]
        # 'u' is the IoU score with the ground truth
        u = target_ious[b_idx, q_idx].clamp(0.0, 1.0)
        # t = s^alpha * u^(1-alpha)
        t_vals = (pred_prob[b_idx, q_idx, cls_idx].clamp(1e-6) ** alpha) * (u ** (1 - alpha))
        target_t[b_idx, q_idx, cls_idx] = t_vals

    bce = F.binary_cross_entropy(pred_prob, target_t, reduction="none")
    # Weight negative samples by s^2
    neg_mask = target_classes < 0
    neg_weights = torch.where(neg_mask.unsqueeze(-1), pred_prob ** 2, torch.ones_like(pred_prob))
    return (bce * neg_weights).sum() / max(1, pos_mask.numel())


def dice_loss(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    inputs = inputs.sigmoid().flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    return (1 - (numerator + 1) / (denominator + 1)).mean()


class SetCriterion(nn.Module):
    def __init__(self, num_classes: int, matcher: nn.Module, weight_dict: Dict[str, float]):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict

    @staticmethod
    def _get_permutation_idx(indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def forward(self, outputs, targets):
        indices = self.matcher(outputs, targets)
        num_objects = sum(len(t["labels"]) for t in targets)
        num_objects = torch.as_tensor([num_objects], dtype=torch.float, device=next(iter(outputs.values())).device).clamp(min=1).item()

        B, Q, _ = outputs["pred_logits"].shape
        device = outputs["pred_logits"].device

        target_classes = torch.full((B, Q), -1, dtype=torch.long, device=device)
        target_ious = torch.zeros((B, Q), device=device)

        for i, (src_idx, tgt_idx) in enumerate(indices):
            target_classes[i, src_idx] = targets[i]["labels"][tgt_idx]
            ious = generalized_box_iou(
                box_convert(outputs["pred_boxes"][i, src_idx], "cxcywh", "xyxy"),
                box_convert(targets[i]["boxes"][tgt_idx], "cxcywh", "xyxy"),
            ).diag()
            target_ious[i, src_idx] = ious

        losses = {"loss_cls": ia_bce_loss(outputs["pred_logits"], target_classes, target_ious)}

        idx = self._get_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        losses["loss_bbox"] = F.l1_loss(src_boxes, target_boxes, reduction="none").sum() / num_objects
        giou_vals = generalized_box_iou(box_convert(src_boxes, "cxcywh", "xyxy"), box_convert(target_boxes, "cxcywh", "xyxy"))
        losses["loss_giou"] = (1 - torch.diag(giou_vals)).sum() / num_objects

        if "pred_masks" in outputs and "masks" in targets[0] and idx[0].numel() > 0:
            src_masks = outputs["pred_masks"][idx]
            target_masks = torch.cat([t["masks"][i] for t, (_, i) in zip(targets, indices)], dim=0)
            target_masks_resized = F.interpolate(target_masks.unsqueeze(1), size=src_masks.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
            losses["loss_mask"] = F.binary_cross_entropy_with_logits(src_masks, target_masks_resized.float(), reduction="mean")
            losses["loss_dice"] = dice_loss(src_masks, target_masks_resized.float())

        return {k: v * self.weight_dict[k] for k, v in losses.items() if k in self.weight_dict}


# --- Main DETR head ---
class LWDETRHead(BaseHead, nn.Module):
    """
    Main LW-DETR detection head with optional Group-DETR sequential decoding.

    This class implements the main components of the LW-DETR architecture, including
    the backbone wrapper, deformable decoder, and prediction heads. The default parameters
    are set to match the 'LW-DETR-tiny' configuration from the paper.

    LW-DETR Configurations (from Table 1):
    - tiny:   d_model=192, num_queries=100, num_decoder_layers=3
    - small:  d_model=192, num_queries=300, num_decoder_layers=3
    - medium: d_model=384, num_queries=300, num_decoder_layers=3
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
        num_groups: int = 1, # Add this parameter
        **kwargs,
    ):
        # Pass kwargs to BaseHead to be robust
        BaseHead.__init__(self, num_classes, backbone_type, image_size, **kwargs)
        nn.Module.__init__(self)

        self.d_model = d_model
        self.num_queries = num_queries
        self.num_groups = num_groups # Store the number of groups
        
        # Add an assertion to ensure queries can be split evenly
        if self.num_groups > 1:
            assert self.num_queries % self.num_groups == 0, \
                f"num_queries ({self.num_queries}) must be divisible by num_groups ({self.num_groups})"

        self.backbone = BackboneWithOptionalFPN(backbone_type, out_channels=d_model, image_size=image_size)
        self.num_feature_levels = self.backbone.num_feature_levels

        self.query_embed = nn.Embedding(num_queries, d_model)

        decoder_layer = DeformableTransformerDecoderLayer(d_model=d_model, n_heads=n_heads, d_ffn=d_ffn, n_levels=self.num_feature_levels)
        self.transformer_decoder = nn.ModuleList(_get_clones(decoder_layer, num_decoder_layers))

        # Prediction heads
        self.class_embed = nn.Linear(d_model, num_classes)
        self.bbox_embed = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 4))
        self.iou_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, 1))
        self.query_scale = nn.Linear(d_model, 4)
        self.mask_embed = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, mask_dim))
        self.pixel_proj = nn.Conv2d(d_model, mask_dim, kernel_size=1)

        self._init_criterion()

    def _init_criterion(self):
        matcher = HungarianMatcher()
        weight_dict = {"loss_cls": 2.0, "loss_bbox": 6.0, "loss_giou": 3.0, "loss_mask": 2.0, "loss_dice": 2.0}
        self.criterion = SetCriterion(self.num_classes, matcher=matcher, weight_dict=weight_dict)

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

        # Compute level start indices
        sizes = [h * w for h, w in memory_spatial_shapes_list]
        level_start_index = torch.as_tensor([0] + sizes[:-1], dtype=torch.long, device=device).cumsum(0)

        # Decoder
        query_embeds = self.query_embed.weight.unsqueeze(0).expand(pixel_values.shape[0], -1, -1)
        reference_points = self.query_scale(query_embeds).sigmoid()
        decoder_output = query_embeds

        if self.num_groups == 1:
            # Original behavior: pass all queries through all layers
            for layer in self.transformer_decoder:
                decoder_output = layer(
                    tgt=decoder_output,
                    memory=memory,
                    ref_points_c=reference_points,
                    memory_spatial_shapes=memory_spatial_shapes,
                    level_start_index=level_start_index,
                )
        else:
            # Group-DETR behavior: sequential group decoding
            group_size = self.num_queries // self.num_groups
            all_group_outputs = []
            
            # Iterate through each group
            for i in range(self.num_groups):
                start, end = i * group_size, (i + 1) * group_size
                
                # Select the current group of queries and their reference points
                group_queries = decoder_output[:, start:end, :]
                group_ref_points = reference_points[:, start:end, :]

                # Pass this single group through the entire stack of decoder layers
                current_tgt = group_queries
                for layer in self.transformer_decoder:
                    current_tgt = layer(
                        tgt=current_tgt,
                        memory=memory,
                        ref_points_c=group_ref_points,
                        memory_spatial_shapes=memory_spatial_shapes,
                        level_start_index=level_start_index,
                    )
                
                # Store the final refined output for this group
                all_group_outputs.append(current_tgt)

                # CRITICAL: Update the main decoder_output tensor in place.
                # This allows the self-attention in the next group's processing
                # to see the refined results of the current group.
                if i < self.num_groups - 1:
                     decoder_output = torch.cat([
                        decoder_output[:, :start, :], 
                        current_tgt, 
                        decoder_output[:, end:, :]
                    ], dim=1).detach() # Detach to prevent re-computing gradients

            # Combine the final outputs from all groups
            decoder_output = torch.cat(all_group_outputs, dim=1)
        
        # --- END OF REPLACED LOGIC ---

        logits = self.class_embed(decoder_output)
        pred_boxes_delta = self.bbox_embed(decoder_output)

        # Implements the box regression reparameterization from the LW-DETR paper
        # reference_points are the proposal boxes 'p' in cxcywh format
        # pred_boxes_delta are the predicted deltas [dx, dy, dw, dh]
        prop_center = reference_points[..., :2]
        prop_size = reference_points[..., 2:]
        delta_center = pred_boxes_delta[..., :2]
        delta_size = pred_boxes_delta[..., 2:]

        # b_cx = delta_x * p_w + p_cx
        pred_center = delta_center * prop_size + prop_center

        # b_w = exp(delta_w) * p_w
        pred_size = delta_size.exp() * prop_size

        # Combine and apply sigmoid to ensure the output is a valid normalized box
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
