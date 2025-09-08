"""
Optimized implementation of a DETR-style segmentation head.

NOTE: The original 'DeformableDETRHead' was a misnomer. This implementation uses a
standard Transformer decoder, not the multi-scale deformable attention from the
Deformable DETR paper. The class has been renamed to `DETRSegmentationHead` for clarity.
"""
# PATCHED: This file has been updated to use a proper Deformable Transformer Decoder.

from collections import OrderedDict
from typing import Dict, List, Optional
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision.ops import FeaturePyramidNetwork, box_convert, generalized_box_iou

# Assuming these are defined in your project structure
from ..base import BaseHead
from ..backbones import get_backbone


# --- Deformable attention (from lw_detr.py) ---
try:
    from torchvision.ops.deformable_attention import DeformableAttention
    print("INFO: Successfully imported torchvision DeformableAttention.")
except ImportError as e:
    print("=" * 60)
    print("CRITICAL WARNING: Failed to import the official DeformableAttention.")
    print(f"THE IMPORT ERROR WAS: {e}")
    print("Falling back to a pure-PyTorch implementation, which may be slow or buggy.")
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
            assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
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
            reference_points: (N, Lq, 2) or (N, Lq, 4) if normalized boxes (cx,cy,w,h)
            input_flatten: (N, L, C)
            input_spatial_shapes: Tensor[(num_levels, 2)] with (H, W) per level
            input_level_start_index: Tensor with start index for each level (not used heavily here)
            """
            N, Lq, C = query.shape
            N2, L, C2 = input_flatten.shape
            assert N == N2 and C == C2, "Shape mismatch between query and input_flatten"

            # project values and reshape to (N, L, n_heads, head_dim)
            value = self.value_proj(input_flatten).view(N, L, self.n_heads, self.head_dim)

            # sampling offsets and attention weights from queries
            offsets = self.sampling_offsets(query).view(N, Lq, self.n_heads, self.n_levels, self.n_points, 2)
            attention_weights = self.attention_weights(query).view(N, Lq, self.n_heads, self.n_levels, self.n_points)
            attention_weights = F.softmax(attention_weights, -1)  # normalize over n_levels * n_points

            # compute sampling locations normalized to [0, 1] coordinates per level
            if reference_points.shape[-1] == 2:
                # reference_points are (x, y) normalized
                ref = reference_points.unsqueeze(2).unsqueeze(4)  # (N, Lq, 1, 1, 1, 2)
                # offset normalizer is [W, H] per level to convert offsets to normalized offsets
                offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)  # (num_levels, 2) as (W, H)
                offset_normalizer = offset_normalizer[None, None, None, :, None, :].to(offsets.device)  # (1,1,1,n_levels,1,2)
                sampling_locations = ref + offsets / offset_normalizer  # normalized sampling locations
            else:
                # reference_points contain (cx, cy, w, h)
                ref_xy = reference_points[..., :2].unsqueeze(2).unsqueeze(3).unsqueeze(4)  # (N,Lq,1,1,1,2)
                ref_wh = reference_points[..., 2:].unsqueeze(2).unsqueeze(3).unsqueeze(4)   # (N,Lq,1,1,1,2)
                # offsets normalized relative to box size
                sampling_locations = ref_xy + offsets / self.n_points * ref_wh * 0.5

            # prepare splitting of flattened value into per-level tensors
            splits = [int(H * W) for (H, W) in input_spatial_shapes.tolist()]
            value_list = value.split(splits, dim=1)  # list of (N, L_lvl, n_heads, head_dim)

            # sampling: for each level, grid-sample from the level's feature map
            sampling_grids = 2 * sampling_locations - 1  # grid_sample expects coords in [-1,1]
            sampled_value_list = []

            for lvl, (H, W) in enumerate(input_spatial_shapes.tolist()):
                # prepare the value for this level: (N, L_lvl, n_heads, head_dim) -> (N*n_heads, head_dim, H, W)
                v_l = value_list[lvl].permute(0, 2, 3, 1).contiguous()  # (N, n_heads, head_dim, L_lvl)
                # reshape into spatial H, W
                v_l = v_l.view(N * self.n_heads, self.head_dim, H, W)

                # sampling grid for this level:
                # sampling_grids[:, :, :, lvl] -> (N, Lq, n_heads, n_points, 2)
                grid_l = sampling_grids[:, :, :, lvl]  # (N, Lq, n_heads, n_points, 2)
                # permute to (N, n_heads, Lq, n_points, 2) then collapse batch*head
                grid_l = grid_l.permute(0, 2, 1, 3, 4).contiguous().view(N * self.n_heads, Lq, self.n_points, 2)

                # grid_sample expects grid shape (N, H_out, W_out, 2); here H_out = Lq, W_out = n_points
                # sample -> (N*n_heads, head_dim, Lq, n_points)
                sampled = F.grid_sample(v_l, grid_l, mode="bilinear", padding_mode="zeros", align_corners=False)

                # reshape back to (N, n_heads, head_dim, Lq, n_points)
                sampled = sampled.view(N, self.n_heads, self.head_dim, Lq, self.n_points)
                sampled_value_list.append(sampled)

            # stack over levels -> (N, n_heads, head_dim, Lq, n_points, n_levels)
            sampled_values = torch.stack(sampled_value_list, dim=-1)

            # attention weights: (N, Lq, n_heads, n_levels, n_points) -> (N, n_heads, 1, Lq, n_levels, n_points)
            attn = attention_weights.permute(0, 2, 3, 1, 4).unsqueeze(2)

            # weighted sum over points and levels -> result shape (N, n_heads, head_dim, Lq)
            weighted = (sampled_values * attn).sum(-1).sum(-1)

            # permute to (N, Lq, n_heads, head_dim) then reshape to (N, Lq, C)
            output = weighted.permute(0, 3, 1, 2).contiguous().view(N, Lq, C)
            return self.output_proj(output)

    DeformableAttention = PurePyTorchDeformableAttention


# --- Deformable Decoder Layer (from lw_detr.py) ---
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


class GenericBackboneWithFPN(nn.Module):
    """
    A generic wrapper to add a Feature Pyramid Network (FPN) to a backbone.

    This module automatically discovers the output feature channels of the provided
    backbone and connects them to an FPN to generate a multi-scale feature pyramid.
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

        # Ensure features are sorted to maintain consistent order for the FPN
        self._feature_keys = sorted(feat_dict.keys())
        self.num_feature_levels = len(self._feature_keys)
        in_channels_list = [feat_dict[k].shape[1] for k in self._feature_keys]

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list, out_channels=self.out_channels
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extracts features and processes them through the FPN."""
        features = self.backbone(x)
        
        # Create an ordered dictionary of features for the FPN
        ordered_features = OrderedDict((k, features[k]) for k in self._feature_keys)
        
        fpn_output = self.fpn(ordered_features)
        
        # Remap keys to strings '0', '1', '2', etc., for DETR compatibility
        return OrderedDict(zip(map(str, range(len(fpn_output))), fpn_output.values()))


# --- Matcher & Losses (from lw_detr.py) ---
class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 1.0, cost_bbox: float = 5.0, cost_giou: float = 2.0, cost_mask: float = 5.0, cost_dice: float = 2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
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
            cost_dice = -(numerator + 1e-5) / (denominator.clamp(min=1e-6) + 1e-5)

            n_pred, n_tgt = out_flat.shape[0], tgt_flat.shape[0]
            out_expanded = out_flat.unsqueeze(1).expand(n_pred, n_tgt, -1)
            tgt_expanded = tgt_flat.unsqueeze(0).expand(n_pred, n_tgt, -1)
            cost_mask = F.binary_cross_entropy(out_expanded, tgt_expanded, reduction="none").mean(-1)

            C = C + (self.cost_mask * cost_mask + self.cost_dice * cost_dice)

        C = C.view(bs, num_queries, -1).cpu()
        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def dice_loss(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    inputs = inputs.sigmoid().flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1e-5) / (denominator + 1e-5)
    return loss.mean()


class SetCriterion(nn.Module):
    def __init__(self, num_classes: int, matcher: nn.Module, weight_dict: Dict[str, float], eos_coef: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    @staticmethod
    def _get_permutation_idx(indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def loss_labels(self, outputs, targets, indices, num_objects):
        src_logits = outputs['pred_logits']
        idx = self._get_permutation_idx(indices)
        
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        return {"loss_cls": loss_ce}

    def loss_boxes(self, outputs, targets, indices, num_objects):
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
        if "pred_masks" not in outputs or "masks" not in targets[0]:
            return {"loss_mask": torch.tensor(0.0), "loss_dice": torch.tensor(0.0)}
            
        src_idx = self._get_permutation_idx(indices)
        src_masks = outputs["pred_masks"][src_idx]
        
        target_masks = torch.cat([t['masks'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        target_masks_resized = F.interpolate(
            target_masks.unsqueeze(1), size=src_masks.shape[-2:], mode="bilinear", align_corners=False
        ).squeeze(1)
        
        target_masks_resized = target_masks_resized.to(src_masks.dtype)
        
        loss_mask = F.binary_cross_entropy_with_logits(src_masks, target_masks_resized, reduction='none').mean(dim=(1, 2))
        loss_dice = dice_loss(src_masks, target_masks_resized)
        
        return {
            "loss_mask": loss_mask.sum() / num_objects,
            "loss_dice": loss_dice
        }

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        indices = self.matcher(outputs_without_aux, targets)

        num_objects = sum(len(t["labels"]) for t in targets)
        num_objects_tensor = torch.as_tensor(
            [num_objects], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(num_objects_tensor)
        
        world_size = torch.distributed.get_world_size() if torch.distributed.is_available() and torch.distributed.is_initialized() else 1
        num_objects = torch.clamp(num_objects_tensor / world_size, min=1).item()

        losses = {}
        for loss_fn_name in ['labels', 'boxes', 'masks']:
            loss_map = getattr(self, f"loss_{loss_fn_name}")(outputs, targets, indices, num_objects)
            losses.update(loss_map)
                    
        weighted_losses = {k: v * self.weight_dict[k] for k, v in losses.items() if k in self.weight_dict}
        return weighted_losses


def _get_clones(module: nn.Module, N: int) -> nn.ModuleList:
    """Helper function to clone a module N times."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class DETRSegmentationHead(BaseHead, nn.Module):
    """
    A DETR-style segmentation head with a Deformable Transformer decoder and
    optional Group-DETR sequential decoding.

    This module combines a backbone with FPN, a Deformable Transformer decoder,
    and prediction heads for object detection and instance segmentation. It
    attends to a small set of key sampling points from a multi-scale feature
    pyramid, making it much more efficient than a standard Transformer.
    """
    def __init__(
        self,
        num_classes: int,
        backbone_type: str = 'dino',
        image_size: int = 224,
        d_model: int = 256,
        num_queries: int = 100,
        num_decoder_layers: int = 6,
        n_heads: int = 8,
        d_ffn: int = 1024,
        mask_dim: int = 256,
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

        self.backbone = GenericBackboneWithFPN(
            backbone_type, fpn_out_channels=d_model, dummy_input_size=image_size
        )
        self.num_feature_levels = self.backbone.num_feature_levels

        self.query_embed = nn.Embedding(num_queries, d_model)
        self.query_scale = nn.Linear(d_model, 4)

        decoder_layer = DeformableTransformerDecoderLayer(
            d_model=d_model, n_heads=n_heads, d_ffn=d_ffn, n_levels=self.num_feature_levels
        )
        self.transformer_decoder = _get_clones(decoder_layer, num_decoder_layers)

        # Prediction heads
        self.class_embed = nn.Linear(d_model, num_classes + 1)
        self.bbox_embed = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 4))
        self.mask_embed = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, mask_dim))
        self.pixel_proj = nn.Conv2d(d_model, mask_dim, kernel_size=1)

        self._init_criterion()

    def _init_criterion(self) -> None:
        """Initializes the loss criterion and matcher."""
        matcher = HungarianMatcher(cost_class=1.0, cost_bbox=5.0, cost_giou=2.0, cost_mask=5.0, cost_dice=2.0)

        weight_dict = {
            "loss_cls": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0, "loss_mask": 5.0, "loss_dice": 2.0
        }
        
        self.criterion = SetCriterion(
            self.num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=0.1,
        )

    def forward(
        self, pixel_values: torch.Tensor, targets: Optional[List[Dict]] = None
    ):
        device = pixel_values.device
        
        # 1. Get multi-scale features from backbone + FPN
        feature_maps = self.backbone(pixel_values)
        srcs = list(feature_maps.values())

        # 2. Prepare multi-scale memory for the decoder
        memory_list, memory_spatial_shapes_list = [], []
        for src in srcs:
            bs, c, h, w = src.shape
            memory_list.append(src.flatten(2).transpose(1, 2))
            memory_spatial_shapes_list.append([h, w])

        memory = torch.cat(memory_list, dim=1)
        memory_spatial_shapes = torch.as_tensor(memory_spatial_shapes_list, dtype=torch.long, device=device)
        level_start_index = torch.cat((
            memory_spatial_shapes.new_zeros((1,)),
            memory_spatial_shapes.prod(1).cumsum(0)[:-1]
        ))

        # 3. Prepare query embeddings and reference points
        query_embeds = self.query_embed.weight.unsqueeze(0).expand(pixel_values.shape[0], -1, -1)
        reference_points = self.query_scale(query_embeds).sigmoid()
        
        # 4. Pass through the Deformable Transformer decoder
        decoder_output = query_embeds
        if self.num_groups == 1:
            for layer in self.transformer_decoder:
                decoder_output = layer(
                    tgt=decoder_output, memory=memory, ref_points_c=reference_points,
                    memory_spatial_shapes=memory_spatial_shapes, level_start_index=level_start_index
                )
        else: # Group-DETR behavior
            group_size = self.num_queries // self.num_groups
            all_group_outputs = []
            
            for i in range(self.num_groups):
                start, end = i * group_size, (i + 1) * group_size
                group_ref_points = reference_points[:, start:end, :]

                current_tgt = decoder_output[:, start:end, :]
                for layer in self.transformer_decoder:
                    current_tgt = layer(
                        tgt=current_tgt, memory=memory, ref_points_c=group_ref_points,
                        memory_spatial_shapes=memory_spatial_shapes, level_start_index=level_start_index
                    )
                
                all_group_outputs.append(current_tgt)

                if i < self.num_groups - 1:
                    decoder_output = torch.cat([
                        decoder_output[:, :start, :], current_tgt, decoder_output[:, end:, :]
                    ], dim=1)
                    decoder_output = decoder_output * 0.5 + decoder_output.detach() * 0.5

            decoder_output = torch.cat(all_group_outputs, dim=1)

        # 5. Generate predictions from the decoder output
        logits = self.class_embed(decoder_output)
        
        # Box prediction (delta-based)
        pred_boxes_delta = self.bbox_embed(decoder_output)
        pred_center = (pred_boxes_delta[..., :2] * reference_points[..., 2:]) + reference_points[..., :2]
        pred_size = (pred_boxes_delta[..., 2:].exp() * reference_points[..., 2:])
        pred_boxes = torch.cat([pred_center, pred_size], dim=-1).sigmoid()

        # Mask prediction
        mask_embeds = self.mask_embed(decoder_output)
        pixel_feats = self.pixel_proj(srcs[0])
        pred_masks_lowres = torch.einsum("bqd,bdhw->bqhw", mask_embeds, pixel_feats)
        
        # 6. Upsample masks to the original image size for segmentation
        pred_masks = F.interpolate(
            pred_masks_lowres, size=pixel_values.shape[-2:], mode='bilinear', align_corners=False
        )
        
        outputs = {"pred_logits": logits, "pred_boxes": pred_boxes, "pred_masks": pred_masks}
        
        losses = {}
        if targets is not None:
            losses = self.criterion(outputs, targets)
        
        return outputs, losses
