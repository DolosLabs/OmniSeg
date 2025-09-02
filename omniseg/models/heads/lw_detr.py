"""
Optimized implementation of an LW-DETR-style detection head with optional instance segmentation.
This is the full, high-performance version, re-integrating the advanced loss and
box prediction mechanisms into the stable architecture.
"""
from collections import OrderedDict
from typing import Dict, List, Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision.ops import FeaturePyramidNetwork, box_convert, generalized_box_iou

from ..base import BaseHead
from ..backbones import get_backbone

# --- FALLBACK IMPLEMENTATION ---
try:
    from torchvision.ops.deformable_attention import DeformableAttention
    print("INFO: Successfully imported official torchvision DeformableAttention.")
except ImportError:
    print("WARNING: Could not import official DeformableAttention. Using pure-PyTorch fallback.")
    class PurePyTorchDeformableAttention(nn.Module):
        def __init__(self, d_model, n_levels, n_heads, n_points):
            super().__init__()
            self.n_heads, self.n_levels, self.n_points, self.d_model = n_heads, n_levels, n_points, d_model
            self.head_dim = d_model // n_heads
            self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
            self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
            self.value_proj = nn.Linear(d_model, d_model)
            self.output_proj = nn.Linear(d_model, d_model)
            self._reset_parameters()
        def _reset_parameters(self):
            nn.init.constant_(self.sampling_offsets.weight.data, 0.)
            nn.init.constant_(self.sampling_offsets.bias.data, 0.)
            nn.init.xavier_uniform_(self.attention_weights.weight.data)
            nn.init.constant_(self.attention_weights.bias.data, 0.)
            nn.init.xavier_uniform_(self.value_proj.weight.data)
            nn.init.constant_(self.value_proj.bias.data, 0.)
            nn.init.xavier_uniform_(self.output_proj.weight.data)
            nn.init.constant_(self.output_proj.bias.data, 0.)
        def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):
            N, Lq, C = query.shape
            N, L, C = input_flatten.shape
            value = self.value_proj(input_flatten).view(N, L, self.n_heads, self.head_dim)
            offsets = self.sampling_offsets(query).view(N, Lq, self.n_heads, self.n_levels, self.n_points, 2)
            attention_weights = self.attention_weights(query).view(N, Lq, self.n_heads, self.n_levels * self.n_points)
            attention_weights = F.softmax(attention_weights, -1).view(N, Lq, self.n_heads, self.n_levels, self.n_points)
            if reference_points.shape[-1] == 2:
                if len(reference_points.shape) == 3: reference_points = reference_points.unsqueeze(2)
                reference_points_expanded = reference_points.unsqueeze(2).unsqueeze(4)
                offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
                offset_normalizer = offset_normalizer[None, None, None, :, None, :]
                sampling_locations = reference_points_expanded + offsets / offset_normalizer
            else:
                 sampling_locations = reference_points[:, :, None, :, None, :2] + offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
            value_list = value.split([H * W for H, W in input_spatial_shapes], dim=1)
            sampling_grids = 2 * sampling_locations - 1
            sampled_value_list = []
            for level, (H, W) in enumerate(input_spatial_shapes):
                value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(N * self.n_heads, self.head_dim, H, W)
                sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
                sampled_value = F.grid_sample(value_l_, sampling_grid_l_, mode='bilinear', padding_mode='zeros', align_corners=False)
                sampled_value_list.append(sampled_value)
            attention_weights = attention_weights.unsqueeze(-1)
            sampled_values = torch.stack(sampled_value_list, dim=-1).view(N, self.n_heads, self.head_dim, Lq, self.n_levels, self.n_points)
            output = (sampled_values.permute(0, 3, 1, 4, 5, 2) * attention_weights).sum(-2).view(N, Lq, C)
            return self.output_proj(output)
    DeformableAttention = PurePyTorchDeformableAttention


# --- HELPER COMPONENTS ---

def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)

def _get_clones(module, N):
    return nn.ModuleList([module for i in range(N)])

class GenericBackboneWithFPN(nn.Module):
    def __init__(self, backbone_type: str = 'dino', fpn_out_channels: int = 256, dummy_input_size: int = 224):
        super().__init__()
        self.backbone = get_backbone(backbone_type)
        self.out_channels = fpn_out_channels
        dummy_input = torch.randn(1, 3, dummy_input_size, dummy_input_size)
        with torch.no_grad():
            feat_dict = self.backbone(dummy_input)
        self._feature_keys = sorted(feat_dict.keys())
        in_channels_list = [feat_dict[k].shape[1] for k in self._feature_keys]
        self.fpn = FeaturePyramidNetwork(in_channels_list=in_channels_list, out_channels=self.out_channels)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(x)
        ordered_features = OrderedDict((k, features[k]) for k in self._feature_keys)
        fpn_output = self.fpn(ordered_features)
        return OrderedDict(zip(map(str, range(len(fpn_output))), fpn_output.values()))


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, n_heads=8, d_ffn=1024, dropout=0.1, n_levels=1, n_points=4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout1, self.norm1 = nn.Dropout(dropout), nn.LayerNorm(d_model)
        self.cross_attn = DeformableAttention(d_model, n_levels, n_heads, n_points)
        self.dropout2, self.norm2 = nn.Dropout(dropout), nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4, self.norm3 = nn.Dropout(dropout), nn.LayerNorm(d_model)

    def forward(self, tgt, memory, memory_padding_mask=None,
                ref_points_c=None, memory_spatial_shapes=None, level_start_index=None):
        q = k = v = tgt
        tgt2 = self.self_attn(q, k, v)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.cross_attn(query=tgt, reference_points=ref_points_c,
                               input_flatten=memory, input_spatial_shapes=memory_spatial_shapes,
                               input_level_start_index=level_start_index, input_padding_mask=memory_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

# ### --- ADVANCED COMPONENT: IoU-Aware Matcher and Loss --- ###

class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 2.0, cost_bbox: float = 5.0, cost_giou: float = 2.0, cost_mask: float = 2.0, cost_dice: float = 2.0):
        super().__init__()
        self.cost_class, self.cost_bbox, self.cost_giou, self.cost_mask, self.cost_dice = cost_class, cost_bbox, cost_giou, cost_mask, cost_dice

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
        C = (self.cost_bbox * cost_bbox + self.cost_giou * cost_giou + self.cost_class * cost_class)
        if "pred_masks" in outputs and "masks" in targets[0]:
            out_masks = outputs["pred_masks"].flatten(0, 1)
            tgt_masks = torch.cat([v["masks"] for v in targets], dim=0)
            tgt_masks = F.interpolate(tgt_masks.unsqueeze(1), size=out_masks.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
            out_flat, tgt_flat = out_masks.sigmoid().flatten(1), tgt_masks.float().flatten(1)
            numerator = 2 * (out_flat @ tgt_flat.T)
            denominator = out_flat.sum(-1)[:, None] + tgt_flat.sum(-1)[None, :]
            cost_dice = -(numerator + 1) / (denominator.clamp(min=1e-6) + 1)
            n_pred, n_tgt = out_flat.shape[0], tgt_flat.shape[0]
            out_expanded, tgt_expanded = out_flat.unsqueeze(1).expand(n_pred, n_tgt, -1), tgt_flat.unsqueeze(0).expand(n_pred, n_tgt, -1)
            cost_mask = F.binary_cross_entropy(out_expanded, tgt_expanded, reduction="none").mean(-1)
            C += (self.cost_mask * cost_mask + self.cost_dice * cost_dice)
        C = C.view(bs, num_queries, -1).cpu()
        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

def ia_bce_loss(pred_scores, pred_iou, target_classes, target_ious, alpha: float = 0.25):
    pred_prob, pred_iou = pred_scores.sigmoid(), pred_iou.squeeze(-1)
    target_t = torch.zeros_like(pred_prob)
    pos_mask = target_classes >= 0
    if pos_mask.any():
        b_idx, q_idx = pos_mask.nonzero(as_tuple=True)
        cls_idx = target_classes[b_idx, q_idx]
        u = target_ious[b_idx, q_idx].clamp(0, 1)
        t_vals = (pred_prob[b_idx, q_idx, cls_idx].clamp(1e-6) ** alpha) * (u ** (1 - alpha))
        target_t[b_idx, q_idx, cls_idx] = t_vals
    bce = F.binary_cross_entropy(pred_prob, target_t, reduction="none")
    neg_mask = target_classes < 0
    neg_weights = torch.where(neg_mask.unsqueeze(-1), pred_prob ** 2, torch.ones_like(pred_prob))
    return (bce * neg_weights).sum() / pos_mask.numel()

def dice_loss(inputs, targets):
    inputs, targets = inputs.sigmoid().flatten(1), targets.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    return (1 - (numerator + 1) / (denominator + 1)).mean()

class SetCriterion(nn.Module):
    def __init__(self, num_classes: int, matcher: nn.Module, weight_dict: Dict[str, float]):
        super().__init__()
        self.num_classes, self.matcher, self.weight_dict = num_classes, matcher, weight_dict

    def _get_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def forward(self, outputs, targets):
        indices = self.matcher(outputs, targets)
        num_objects = sum(len(t["labels"]) for t in targets)
        num_objects = torch.as_tensor([num_objects], dtype=torch.float, device=next(iter(outputs.values())).device).clamp(min=1).item()
        
        B, Q, _ = outputs["pred_logits"].shape
        target_classes = torch.full((B, Q), -1, dtype=torch.long, device=outputs["pred_logits"].device)
        target_ious = torch.zeros((B, Q), device=outputs["pred_logits"].device)
        for i, (src_idx, tgt_idx) in enumerate(indices):
            target_classes[i, src_idx] = targets[i]["labels"][tgt_idx]
            ious = generalized_box_iou(
                box_convert(outputs["pred_boxes"][i, src_idx], "cxcywh", "xyxy"),
                box_convert(targets[i]["boxes"][tgt_idx], "cxcywh", "xyxy"),
            ).diag()
            target_ious[i, src_idx] = ious

        losses = {'loss_cls': ia_bce_loss(outputs["pred_logits"], outputs["pred_iou"], target_classes, target_ious)}
        
        idx = self._get_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        losses['loss_bbox'] = F.l1_loss(src_boxes, target_boxes, reduction='none').sum() / num_objects
        losses['loss_giou'] = (1 - torch.diag(generalized_box_iou(
            box_convert(src_boxes, "cxcywh", "xyxy"),
            box_convert(target_boxes, "cxcywh", "xyxy")
        ))).sum() / num_objects
        
        if "pred_masks" in outputs and "masks" in targets[0] and idx[0].numel() > 0:
            src_masks = outputs["pred_masks"][idx]
            target_masks = torch.cat([t["masks"][i] for t, (_, i) in zip(targets, indices)], dim=0)
            target_masks_resized = F.interpolate(target_masks.unsqueeze(1), size=src_masks.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
            losses['loss_mask'] = F.binary_cross_entropy_with_logits(src_masks, target_masks_resized.float(), reduction='mean')
            losses['loss_dice'] = dice_loss(src_masks, target_masks_resized.float())
        
        return {k: v * self.weight_dict[k] for k, v in losses.items() if k in self.weight_dict}


# ----------------------------
# UPGRADED LW-DETR Head
# ----------------------------
class LWDETRHead(BaseHead, nn.Module):
    def __init__(self, num_classes: int, backbone_type: str = "dino", image_size: int = 224,
                 d_model: int = 256, num_queries: int = 100, num_decoder_layers: int = 3, mask_dim: int = 16):
        BaseHead.__init__(self, num_classes, backbone_type, image_size)
        nn.Module.__init__(self)

        self.d_model, self.num_queries = d_model, num_queries
        self.backbone = GenericBackboneWithFPN(backbone_type, fpn_out_channels=d_model, dummy_input_size=image_size)
        self.query_embed = nn.Embedding(num_queries, d_model)
        decoder_layer = DeformableTransformerDecoderLayer(d_model=d_model, n_heads=8, d_ffn=2048, n_levels=1)
        self.transformer_decoder = nn.ModuleList([_get_clones(decoder_layer, num_decoder_layers)])

        # ### --- ADVANCED COMPONENT: Prediction Heads --- ###
        self.class_embed = nn.Linear(d_model, num_classes) # No +1 for "no object"
        self.bbox_embed = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 4))
        self.iou_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, 1))
        self.query_scale = nn.Linear(d_model, 2) # To learn reference points
        self.mask_embed = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, mask_dim))
        self.pixel_proj = nn.Conv2d(d_model, mask_dim, kernel_size=1)
        self._init_criterion()

    def _init_criterion(self):
        # ### --- ADVANCED COMPONENT: Matcher and Criterion --- ###
        matcher = HungarianMatcher(cost_class=2.0, cost_bbox=5.0, cost_giou=2.0, cost_mask=2.0, cost_dice=2.0)
        weight_dict = {"loss_cls": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0, "loss_mask": 2.0, "loss_dice": 2.0}
        self.criterion = SetCriterion(self.num_classes, matcher=matcher, weight_dict=weight_dict)

    def forward(self, pixel_values: torch.Tensor, targets: Optional[List[Dict]] = None):
        feature_maps = self.backbone(pixel_values)
        src = feature_maps["0"]
        B, C, H, W = src.shape
        memory = src.flatten(2).transpose(1, 2)
        device = pixel_values.device

        # ### --- ADVANCED COMPONENT: Delta-Based Box Prediction --- ###
        query_embeds = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        reference_points = self.query_scale(query_embeds).sigmoid()

        decoder_output = query_embeds
        for layer in self.transformer_decoder[0]:
            decoder_output = layer(
                tgt=decoder_output, memory=memory, ref_points_c=reference_points,
                memory_spatial_shapes=torch.as_tensor([[H, W]], device=device),
                level_start_index=torch.as_tensor([0], device=device)
            )
        
        logits = self.class_embed(decoder_output)
        pred_boxes_delta = self.bbox_embed(decoder_output)
        
        # Use the stable, numerically sound formula
        reference_points_logit = inverse_sigmoid(reference_points)
        delta_xy, delta_wh = pred_boxes_delta[..., :2], pred_boxes_delta[..., 2:]
        new_center_logit = reference_points_logit + delta_xy
        pred_boxes_logit = torch.cat([new_center_logit, delta_wh], dim=-1)
        pred_boxes = pred_boxes_logit.sigmoid()
        
        pred_iou = self.iou_head(decoder_output) # Note: loss expects logits
        
        mask_embeds = self.mask_embed(decoder_output)
        pixel_feats = self.pixel_proj(src)
        pred_masks = torch.einsum("bqd,bdhw->bqhw", mask_embeds, pixel_feats)
        
        outputs = {"pred_logits": logits, "pred_boxes": pred_boxes, "pred_iou": pred_iou, "pred_masks": pred_masks}

        if self.training and targets is not None:
            losses = self.criterion(outputs, targets)
            return outputs, losses
        return outputs
