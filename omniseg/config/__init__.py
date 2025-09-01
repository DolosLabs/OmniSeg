"""Configuration settings for OmniSeg."""

from typing import Dict, Any, List

# --- Main Configuration ---
PROJECT_DIR = './SSL_Instance_Segmentation'
EPOCHS = 300
NUM_CLASSES = 10

# Backbone and Head Configuration
BACKBONE_CONFIGS = {
    'dino': {'default_size': 224, 'model_name': 'facebook/dinov3-vitl16-pretrain-lvd1689m'},
    'sam': {'default_size': 1024, 'model_name': 'facebook/sam-vit-huge'},
    'swin': {'default_size': 224, 'model_name': 'microsoft/swin-base-patch4-window7-224-in22k'},
    'convnext': {'default_size': 224, 'model_name': 'convnext_base.fb_in22k'},
    'repvgg': {'default_size': 224, 'model_name': 'repvgg_b0.rvgg_in1k'},
    'resnet': {'default_size': 224, 'model_name': 'resnet50.a1_in1k'}
}

HEAD_CONFIGS = {
    'maskrcnn': {'precision': '16-mixed', 'batch_size': 8},
    'contourformer': {'precision': '32', 'batch_size': 4},
    'deformable_detr': {'precision': '32', 'batch_size': 4}
}


def get_default_config(backbone_type: str, head_type: str) -> Dict[str, Any]:
    """Get default configuration for a backbone-head combination."""
    if backbone_type not in BACKBONE_CONFIGS:
        raise ValueError(f"Unknown backbone: {backbone_type}")
    if head_type not in HEAD_CONFIGS:
        raise ValueError(f"Unknown head: {head_type}")
    
    return {
        'backbone': BACKBONE_CONFIGS[backbone_type],
        'head': HEAD_CONFIGS[head_type],
        'image_size': BACKBONE_CONFIGS[backbone_type]['default_size'],
        'batch_size': HEAD_CONFIGS[head_type]['batch_size'],
        'precision': HEAD_CONFIGS[head_type]['precision']
    }


def get_available_backbones() -> List[str]:
    """Return list of available backbone types."""
    return list(BACKBONE_CONFIGS.keys())


def get_available_heads() -> List[str]:
    """Return list of available head types."""
    return list(HEAD_CONFIGS.keys())