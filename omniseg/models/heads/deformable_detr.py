"""
Optimized implementation of a DETR-style segmentation head.

NOTE: The original 'DeformableDETRHead' was a misnomer. This implementation uses a
standard Transformer decoder, not the multi-scale deformable attention from the
Deformable DETR paper. The class has been renamed to `DETRSegmentationHead` for clarity.
"""

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


class HungarianMatcher(nn.Module):
    """
    Computes an optimal bipartite matching between predictions and targets.

    This matcher is used in DETR to assign ground truth boxes to the best-matching
    model predictions.
    """

    def __init__(self, cost_class: float = 1.0, cost_bbox: float = 5.0, cost_giou: float = 2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "All costs cannot be zero."

    @torch.no_grad()
    def forward(
        self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]
    ) -> List[tuple[torch.Tensor, torch.Tensor]]:
        """
        Performs the matching.

        Args:
            outputs: A dict with 'pred_logits' (B, N, C) and 'pred_boxes' (B, N, 4).
            targets: A list of dicts, each with 'labels' and 'boxes'.

        Returns:
            A list of (source_idx, target_idx) tuples for each batch element.
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # Flatten predictions and concatenate targets for batch processing
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        out_bbox = outputs["pred_boxes"].flatten(0, 1)

        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute classification cost (negative log-probability)
        cost_class = -out_prob[:, tgt_ids]

        # Compute L1 cost for bounding boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute GIoU cost for bounding boxes
        cost_giou = -generalized_box_iou(
            box_convert(out_bbox, in_fmt="cxcywh", out_fmt="xyxy"),
            box_convert(tgt_bbox, in_fmt="cxcywh", out_fmt="xyxy"),
        )

        # Final cost matrix
        C = (
            self.cost_bbox * cost_bbox
            + self.cost_class * cost_class
            + self.cost_giou * cost_giou
        )
        C = C.view(bs, num_queries, -1).cpu()

        # Perform matching for each batch element
        sizes = [len(v["boxes"]) for v in targets]
        indices = [
            linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))
        ]
        
        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

def dice_loss(inputs: torch.Tensor, targets: torch.Tensor, num_objects: float) -> torch.Tensor:
    """
    Computes the Dice loss for semantic segmentation.

    Args:
        inputs: Predicted masks, shape (N, H, W). Logits are expected.
        targets: Ground truth masks, shape (N, H, W).
        num_objects: The number of objects to normalize the loss by.

    Returns:
        The computed dice loss.
    """
    inputs = inputs.sigmoid().flatten(1)
    targets = targets.flatten(1)
    
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    
    # Add a small epsilon for numerical stability
    loss = 1 - (numerator + 1e-5) / (denominator + 1e-5)
    return loss.sum() / num_objects


class SetCriterion(nn.Module):
    """
    The loss computation criterion for DETR-style models.

    This module computes the final loss as a weighted sum of classification,
    bounding box, and mask losses.
    """
    def __init__(self, num_classes: int, matcher: nn.Module, weight_dict: Dict[str, float], eos_coef: float):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        
        # Weight for the "no object" class in classification loss
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def _get_permutation_idx(self, indices: List[tuple]) -> tuple[torch.Tensor, torch.Tensor]:
        """Helper to convert matcher output to flat indices."""
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
        return {"loss_ce": loss_ce}

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
        src_idx = self._get_permutation_idx(indices)
        src_masks = outputs["pred_masks"][src_idx]
        
        target_masks = torch.cat([t['masks'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        # Use float() for BCE and Dice compatibility
        target_masks = target_masks.to(src_masks.dtype)
        
        # BCE loss
        loss_mask = F.binary_cross_entropy_with_logits(src_masks, target_masks, reduction='none')
        loss_mask = loss_mask.mean(dim=(1, 2)) # Average over pixels

        # Dice loss
        loss_dice = dice_loss(src_masks, target_masks, num_objects)
        
        return {
            "loss_mask": loss_mask.sum() / num_objects,
            "loss_dice": loss_dice
        }

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        indices = self.matcher(outputs_without_aux, targets)

        # Sum objects across all GPUs for consistent normalization
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

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss_fn_name in ['labels', 'boxes', 'masks']:
                    l_dict = getattr(self, f"loss_{loss_fn_name}")(aux_outputs, targets, indices, num_objects)
                    l_dict = {f"{k}_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)
                    
        # Apply final weights
        weighted_losses = {k: v * self.weight_dict[k] for k, v in losses.items() if k in self.weight_dict}
        return weighted_losses


def _get_clones(module: nn.Module, N: int) -> nn.ModuleList:
    """Helper function to clone a module N times."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class DETRSegmentationHead(BaseHead, nn.Module):
    """
    A DETR-style segmentation head with a standard Transformer decoder and optional
    Group-DETR sequential decoding.

    This module combines a backbone with FPN, a Transformer decoder, and prediction
    heads for object detection and instance segmentation.

    Note: This is NOT a Deformable DETR. It uses a standard, computationally-intensive
    Transformer decoder that attends to all pixels in the highest-resolution feature map.
    """

    def __init__(
        self,
        num_classes: int,
        backbone_type: str = 'dino',
        image_size: int = 224,
        d_model: int = 256,
        num_queries: int = 100,
        num_decoder_layers: int = 6,
        num_groups: int = 1, # Add this parameter
        **kwargs,
    ):
        # Call initializers for both parent classes
        BaseHead.__init__(self, num_classes, backbone_type, image_size, **kwargs)
        nn.Module.__init__(self)

        self.d_model = d_model
        self.num_queries = num_queries
        self.num_groups = num_groups # Store the number of groups

        # Add an assertion to ensure queries can be split evenly
        if self.num_groups > 1:
            assert self.num_queries % self.num_groups == 0, \
                f"num_queries ({self.num_queries}) must be divisible by num_groups ({self.num_groups})"

        self.backbone = GenericBackboneWithFPN(
            backbone_type, fpn_out_channels=d_model, dummy_input_size=image_size
        )

        self.query_embed = nn.Embedding(num_queries, d_model)

        # --- REFACTORED DECODER ---
        # Refactored from a single nn.TransformerDecoder to a ModuleList of layers
        # to enable the group-wise decoding logic.
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer_decoder = nn.ModuleList(
            _get_clones(decoder_layer, num_decoder_layers)
        )

        # Prediction heads
        self.class_embed = nn.Linear(d_model, num_classes + 1)
        self.bbox_embed = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 4)
        )
        self.mask_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model)
        )

        self._init_criterion()

    def _init_criterion(self) -> None:
        """Initializes the loss criterion and matcher."""
        matcher = HungarianMatcher(cost_class=1.0, cost_bbox=5.0, cost_giou=2.0)

        weight_dict = {
            "loss_ce": 1.0,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
            "loss_mask": 1.0,
            "loss_dice": 1.0,
        }
        
        # Add weights for auxiliary losses if they are ever used
        for i in range(len(self.transformer_decoder) - 1):
            weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})


        self.criterion = SetCriterion(
            self.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=0.1,
        )

    def forward(
        self, pixel_values: torch.Tensor, targets: Optional[List[Dict]] = None
    ):
        """
        Forward pass for the DETR segmentation head.

        Args:
            pixel_values: Input images, shape (B, 3, H, W).
            targets: Optional list of ground truth dicts for training.

        Returns:
            If targets are None, a dict of predictions.
            If targets are provided, a tuple of (predictions, losses).
        """
        # 1. Get multi-scale features from backbone + FPN
        feature_maps = self.backbone(pixel_values)

        # 2. Use the highest-resolution feature map as memory for the decoder
        finest_features = feature_maps['0']
        B, C, H_f, W_f = finest_features.shape
        memory = finest_features.flatten(2).transpose(1, 2)  # (B, H_f*W_f, C)

        # 3. Prepare query embeddings
        query_embeds = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)

        # 4. Pass through the Transformer decoder with optional group logic
        decoder_output = query_embeds

        # --- REPLACED DECODER LOGIC ---
        if self.num_groups == 1:
            # Original behavior: pass all queries through all layers sequentially
            for layer in self.transformer_decoder:
                decoder_output = layer(tgt=decoder_output, memory=memory)
        else:
            # Group-DETR behavior: sequential group decoding
            group_size = self.num_queries // self.num_groups
            all_group_outputs = []

            # Iterate through each group
            for i in range(self.num_groups):
                start, end = i * group_size, (i + 1) * group_size
                
                # The queries for this group are a slice of the running `decoder_output` tensor
                group_queries = decoder_output[:, start:end, :]

                # Pass this single group through the entire stack of decoder layers
                current_tgt = group_queries
                for layer in self.transformer_decoder:
                    current_tgt = layer(tgt=current_tgt, memory=memory)
                
                # Store the final refined output for this group
                all_group_outputs.append(current_tgt)

                # CRITICAL: Update the main decoder_output tensor in place.
                # This allows self-attention in the next group's processing
                # to see the refined results of the current group.
                if i < self.num_groups - 1:
                    decoder_output = torch.cat([
                        decoder_output[:, :start, :], 
                        current_tgt, 
                        decoder_output[:, end:, :]
                    ], dim=1).detach()

            # Combine the final outputs from all groups
            decoder_output = torch.cat(all_group_outputs, dim=1)

        # 5. Generate predictions from the decoder output
        logits = self.class_embed(decoder_output)
        pred_boxes = self.bbox_embed(decoder_output).sigmoid()
        mask_embeds = self.mask_head(decoder_output)

        # 6. Compute masks via dot product with finest features
        pred_masks_lowres = (mask_embeds @ finest_features.view(B, C, -1)).view(B, -1, H_f, W_f)

        # 7. Upsample masks to the original image size
        pred_masks = F.interpolate(
            pred_masks_lowres,
            size=pixel_values.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )

        outputs = {
            "pred_logits": logits,
            "pred_boxes": pred_boxes,
            "pred_masks": pred_masks,
        }

        if targets is not None:
            losses = self.criterion(outputs, targets)
            return outputs, losses

        return outputs
