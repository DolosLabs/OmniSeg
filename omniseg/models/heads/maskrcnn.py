"""Mask R-CNN head implementation."""

from typing import Dict, List, Optional, Tuple, Union
from collections import OrderedDict
import warnings
import torch
import torch.nn as nn

# Import the original MaskRCNN model to subclass it
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import FeaturePyramidNetwork

from ..base import BaseHead
from ..backbones import get_backbone


class ClippingMaskRCNN(MaskRCNN):
    """
    A subclass of MaskRCNN that includes two critical fixes:
    1. Clips proposal boxes to image boundaries to prevent CUDA assert errors.
    2. Handles the case where the RPN produces no proposals to prevent crashes.
    """
    def forward(self, images, targets=None):
        if self.training:
            if targets is None:
                raise ValueError("In training mode, targets should be passed")
            for target in targets:
                boxes = target["boxes"]
                if not isinstance(boxes, torch.Tensor):
                    raise TypeError(f"Expected target boxes to be of type Tensor, got {type(boxes)}.")
                if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                    raise ValueError(f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.")

        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            torch._assert(len(val) == 2, "expecting the last two dimensions of the Tensor to be H and W")
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        
        proposals, proposal_losses = self.rpn(images, features, targets)

        # Safety check: if RPN produces no proposals, return early
        if not proposals:
            losses = {}
            losses.update(proposal_losses)
            # In this case, detections are empty
            return losses, []

        clipped_proposals = []
        for i, proposals_per_image in enumerate(proposals):
            image_h, image_w = images.image_sizes[i]
            proposals_per_image[:, [0, 2]] = proposals_per_image[:, [0, 2]].clamp(min=0, max=image_w)
            proposals_per_image[:, [1, 3]] = proposals_per_image[:, [1, 3]].clamp(min=0, max=image_h)
            clipped_proposals.append(proposals_per_image)
        
        detections, detector_losses = self.roi_heads(features, clipped_proposals, images.image_sizes, targets)
        
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        # ============================ START: FINAL FIX ============================
        # The original `eager_outputs` returns only one item.
        # We must return a tuple of (losses, detections) to match what the
        # calling code `losses, _ = self.model(...)` expects.
        return losses, detections
        # ============================= END: FINAL FIX =============================


# --- The rest of your file remains the same ---

class GenericBackboneWithFPN(nn.Module):
    """Generic wrapper to add FPN to any backbone."""
    def __init__(self, backbone_type: str = 'dino', fpn_out_channels: int = 256, dummy_input_size: int = 224):
        super().__init__()
        self.backbone = get_backbone(backbone_type)
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
        remapped = OrderedDict([(str(i), val) for i, (key, val) in enumerate(fpn_out.items())])
        return remapped


class MaskRCNNHead(BaseHead, nn.Module):
    """Mask R-CNN segmentation head."""
    def __init__(self, num_classes: int, backbone_type: str = 'dino', image_size: int = 224, anchor_base_size: int = 32, **kwargs):
        BaseHead.__init__(self, num_classes, backbone_type, image_size)
        nn.Module.__init__(self)
        backbone_with_fpn = GenericBackboneWithFPN(backbone_type, dummy_input_size=image_size)
        dummy = torch.randn(1, 3, image_size, image_size)
        with torch.no_grad():
            feats = backbone_with_fpn(dummy)
        num_maps = len(feats)
        if num_maps == 0:
            raise RuntimeError("Backbone+FPN returned no feature maps; expected >=1")
        sizes = tuple([(anchor_base_size * (2 ** i),) for i in range(num_maps)])
        aspect_ratios = ((0.5, 1.0, 2.0),) * num_maps
        anchor_generator = AnchorGenerator(sizes=sizes, aspect_ratios=aspect_ratios)
        self.model = ClippingMaskRCNN(
            backbone=backbone_with_fpn, 
            num_classes=num_classes + 1, 
            rpn_anchor_generator=anchor_generator,
            **kwargs
        )

    def forward(self, 
                pixel_values: torch.Tensor, 
                targets: Optional[List[Dict]] = None
    ) -> Tuple[Union[List[Dict], Dict], Dict]:
        if self.training and targets is not None:
            losses, _ = self.model(pixel_values, targets)
            return {}, losses
        else:
            _, predictions = self.model(pixel_values, targets)
            return predictions, {}
