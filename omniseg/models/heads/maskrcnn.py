"""Mask R-CNN head implementation."""

from typing import Dict, List, Optional
from collections import OrderedDict
import torch
import torch.nn as nn
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
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


class MaskRCNNHead(BaseHead, nn.Module):
    """Mask R-CNN segmentation head."""
    
    def __init__(self, num_classes: int, backbone_type: str = 'dino', image_size: int = 224, anchor_base_size: int = 32):
        BaseHead.__init__(self, num_classes, backbone_type, image_size)
        nn.Module.__init__(self)
        
        backbone_with_fpn = GenericBackboneWithFPN(backbone_type, dummy_input_size=image_size)
        backbone_with_fpn.out_channels = backbone_with_fpn.out_channels
        
        # Discover number of feature maps
        dummy = torch.randn(1, 3, image_size, image_size)
        with torch.no_grad():
            feats = backbone_with_fpn(dummy)
        
        num_maps = len(feats)
        if num_maps == 0:
            raise RuntimeError("Backbone+FPN returned no feature maps; expected >=1")
        
        sizes = tuple([(anchor_base_size * (2 ** i),) for i in range(num_maps)])
        aspect_ratios = ((0.5, 1.0, 2.0),) * num_maps
        anchor_generator = AnchorGenerator(sizes=sizes, aspect_ratios=aspect_ratios)
        
        self.model = MaskRCNN(
            backbone=backbone_with_fpn, 
            num_classes=num_classes + 1, 
            rpn_anchor_generator=anchor_generator
        )

    def forward(self, pixel_values: torch.Tensor, targets: Optional[List[Dict]] = None) -> Dict[str, torch.Tensor]:
        return self.model(pixel_values, targets)