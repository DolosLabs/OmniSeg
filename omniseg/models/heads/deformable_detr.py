"""Deformable DETR segmentation head implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
from collections import OrderedDict
from types import SimpleNamespace

from scipy.optimize import linear_sum_assignment
from torchvision.ops import FeaturePyramidNetwork, box_convert, generalized_box_iou

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


class HungarianMatcherDETR(nn.Module):
    """Hungarian matcher for DETR-style models."""
    
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        out_bbox = outputs["pred_boxes"].flatten(0, 1)
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        cost_class = -out_prob[:, tgt_ids]
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_giou = -generalized_box_iou(box_convert(out_bbox, in_fmt="cxcywh", out_fmt="xyxy"), box_convert(tgt_bbox, in_fmt="cxcywh", out_fmt="xyxy"))
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()
        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def dice_loss(inputs, targets, num_boxes):
    """Compute dice loss."""
    inputs = inputs.sigmoid().flatten(1)
    # Flatten the target masks to match the input masks
    targets = targets.flatten(1)
    
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


class SetCriterionDETR(nn.Module):
    """Loss criterion for DETR-style models."""
    
    def __init__(self, num_classes, matcher, weight_dict, eos_coef):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def loss_labels(self, outputs, targets, indices, num_boxes):
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        return {"loss_ce": loss_ce}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_giou = 1 - torch.diag(generalized_box_iou(box_convert(src_boxes, in_fmt="cxcywh", out_fmt="xyxy"), box_convert(target_boxes, in_fmt="cxcywh", out_fmt="xyxy")))
        return {"loss_bbox": loss_bbox.sum() / num_boxes, "loss_giou": loss_giou.sum() / num_boxes}

    def loss_masks(self, outputs, targets, indices, num_boxes):
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"][src_idx]
        target_masks = torch.cat([t['masks'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_mask = F.binary_cross_entropy_with_logits(src_masks, target_masks.to(src_masks.dtype), reduction='none').mean(dim=[1,2])
        loss_dice = dice_loss(src_masks, target_masks.to(src_masks.dtype), num_boxes)
        return {"loss_mask": loss_mask.sum() / num_boxes, "loss_dice": loss_dice}

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        indices = self.matcher(outputs_without_aux, targets)
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / (torch.distributed.get_world_size() if torch.distributed.is_available() and torch.distributed.is_initialized() else 1), min=1).item()
        losses = {}
        for loss in ['labels', 'boxes', 'masks']:
            losses.update(getattr(self, f"loss_{loss}")(outputs, targets, indices, num_boxes))
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in ['labels', 'boxes', 'masks']:
                    l_dict = getattr(self, f"loss_{loss}")(aux_outputs, targets, indices, num_boxes)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return {k: v * self.weight_dict[k] for k, v in losses.items() if k in self.weight_dict}


class DeformableDETRHead(BaseHead, nn.Module):
    """Deformable DETR segmentation head."""
    
    def __init__(self, num_classes: int, backbone_type: str = 'dino', image_size: int = 224,
                 d_model: int = 256, num_queries: int = 100, num_decoder_layers: int = 6):
        BaseHead.__init__(self, num_classes, backbone_type, image_size)
        nn.Module.__init__(self)
        
        self.d_model = d_model
        self.num_queries = num_queries
        self.backbone = GenericBackboneWithFPN(backbone_type, fpn_out_channels=d_model, dummy_input_size=image_size)
        
        # Query embeddings 
        self.query_embed = nn.Embedding(num_queries, d_model)
        
        # Simple transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # Prediction heads
        self.class_embed = nn.Linear(d_model, num_classes + 1)
        self.bbox_embed = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 4)
        )
        self.mask_head = nn.Sequential(
            nn.Linear(d_model, d_model), 
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # Loss computation setup
        config = SimpleNamespace(
            class_cost=1.0,
            bbox_cost=5.0,
            giou_cost=2.0,
            bbox_loss_coefficient=5.0,
            giou_loss_coefficient=2.0,
            eos_coefficient=0.1
        )
        
        matcher = HungarianMatcherDETR(
            cost_class=config.class_cost, 
            cost_bbox=config.bbox_cost, 
            cost_giou=config.giou_cost
        )
        weight_dict = {
            "loss_ce": 1.0, 
            "loss_bbox": config.bbox_loss_coefficient, 
            "loss_giou": config.giou_loss_coefficient, 
            "loss_mask": 1.0, 
            "loss_dice": 1.0
        }
        self.criterion = SetCriterionDETR(
            num_classes, 
            matcher=matcher, 
            weight_dict=weight_dict, 
            eos_coef=config.eos_coefficient
        )

    def forward(self, pixel_values: torch.Tensor, targets: Optional[List[Dict]] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for Deformable DETR head.
        
        Args:
            pixel_values: Input images, shape [B, 3, H, W]
            targets: Optional list of dicts with 'boxes', 'labels', 'masks'
    
        Returns:
            outputs: Dict containing 'pred_logits', 'pred_boxes', 'pred_masks', and optionally 'aux_outputs'
            losses: Dict of losses if targets are provided
        """
        # 1. Extract backbone features
        feature_maps = self.backbone(pixel_values)
        
        # 2. Use the finest feature map as memory for the transformer
        finest_features = feature_maps['0']  # Shape: [B, d_model, H_f, W_f]
        B, C, H_f, W_f = finest_features.shape
        
        # Flatten spatial dimensions for transformer
        memory = finest_features.flatten(2).transpose(1, 2)  # [B, H_f*W_f, d_model]
        
        # 3. Get query embeddings
        query_embeds = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)  # [B, num_queries, d_model]
        
        # 4. Apply transformer decoder
        memory_key_padding_mask = torch.zeros(B, H_f*W_f, dtype=torch.bool, device=pixel_values.device)
        
        decoder_output = self.transformer_decoder(
            tgt=query_embeds,
            memory=memory,
            memory_key_padding_mask=memory_key_padding_mask
        )  # [B, num_queries, d_model]
        
        # 5. Apply prediction heads
        logits = self.class_embed(decoder_output)
        pred_boxes = self.bbox_embed(decoder_output).sigmoid()
        mask_embeds = self.mask_head(decoder_output)  # [B, num_queries, d_model]
        
        # 6. Compute mask predictions using finest feature map
        pred_masks_lowres = (mask_embeds @ finest_features.view(B, C, H_f * W_f)).view(B, -1, H_f, W_f)
        
        # 7. Upsample masks to match input image size
        pred_masks = F.interpolate(
            pred_masks_lowres,
            size=(pixel_values.shape[2], pixel_values.shape[3]),
            mode='bilinear',
            align_corners=False
        )
        
        outputs = {
            "pred_logits": logits,
            "pred_boxes": pred_boxes,
            "pred_masks": pred_masks,
            "aux_outputs": [],
        }
        
        if targets is not None:
            losses = self.criterion(outputs, targets)
            return outputs, losses
        
        return outputs
