"""Backbone factory and imports."""

from typing import List
from ..base import BaseBackbone
from ...config import get_available_backbones

# Import backbone implementations
from .dino import DinoVisionTransformerBackbone
from .sam import SamVisionTransformerBackbone
from .swin import SwinTransformerBackbone
from .convnext import ConvNeXtBackbone
from .repvgg import RepVGGBackbone  
from .resnet import ResNetBackbone


def get_backbone(backbone_type: str, **kwargs) -> BaseBackbone:
    """Factory function to create a backbone instance."""
    backbone_classes = {
        'dino': DinoVisionTransformerBackbone,
        'sam': SamVisionTransformerBackbone,
        'swin': SwinTransformerBackbone,
        'convnext': ConvNeXtBackbone,
        'repvgg': RepVGGBackbone,
        'resnet': ResNetBackbone
    }
    
    if backbone_type not in backbone_classes:
        raise ValueError(f"Unsupported backbone_type: {backbone_type}. "
                        f"Available options: {list(backbone_classes.keys())}")
    
    return backbone_classes[backbone_type](**kwargs)


__all__ = [
    'get_backbone',
    'DinoVisionTransformerBackbone',
    'SamVisionTransformerBackbone', 
    'SwinTransformerBackbone',
    'ConvNeXtBackbone',
    'RepVGGBackbone',
    'ResNetBackbone'
]