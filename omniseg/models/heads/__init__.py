"""Segmentation head models and factory functions."""

from typing import Dict, Any
from ..base import BaseHead
from .maskrcnn import MaskRCNNHead
from .deformable_detr import DETRSegmentationHead
from .contourformer import ContourFormerHead
from .lw_detr import LWDETRHead


def get_head(head_type: str, num_classes: int, **kwargs) -> BaseHead:
    """Factory function to create a segmentation head instance."""
    head_classes = {
        'maskrcnn': MaskRCNNHead,
        'deformable_detr': DETRSegmentationHead,
        'contourformer': ContourFormerHead,
        'lw_detr': LWDETRHead
    }
    
    if head_type not in head_classes:
        raise ValueError(f"Unsupported head_type: {head_type}. "
                        f"Available options: {list(head_classes.keys())}")
    
    return head_classes[head_type](num_classes, **kwargs)


__all__ = [
    'BaseHead',
    'MaskRCNNHead',
    'DETRSegmentationHead', 
    'ContourFormerHead',
    'LWDETRSegmentationHead',
    'get_head'
]
