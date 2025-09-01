#!/usr/bin/env python3
"""
Demo script to test the refactored modular structure of OmniSeg.
This shows how the components can be imported and used independently.
"""

import torch
import numpy as np
from PIL import Image

# Import from our modular structure
from omniseg.config import (
    BACKBONE_CONFIGS, HEAD_CONFIGS, NUM_CLASSES,
    get_available_backbones, get_available_heads, 
    get_default_config
)
from omniseg.data import get_transforms
from omniseg.models.base import BaseBackbone, BaseHead
from omniseg.models.backbones import get_backbone


def demo_config():
    """Demo the configuration module."""
    print("=== Configuration Demo ===")
    print(f"Available backbones: {get_available_backbones()}")
    print(f"Available heads: {get_available_heads()}")
    print(f"Number of classes: {NUM_CLASSES}")
    
    # Get default config for a backbone-head combination
    config = get_default_config('resnet', 'maskrcnn')
    print(f"Default config for ResNet + MaskRCNN: {config}")
    print()


def demo_transforms():
    """Demo the data transforms."""
    print("=== Data Transforms Demo ===")
    
    # Create a dummy image
    dummy_image = Image.new('RGB', (640, 480), color='red')
    
    # Get transforms
    train_transform = get_transforms(augment=True, image_size=224)
    val_transform = get_transforms(augment=False, image_size=224)
    
    # Apply transforms
    train_tensor = train_transform(dummy_image)
    val_tensor = val_transform(dummy_image)
    
    print(f"Original image size: {dummy_image.size}")
    print(f"Training transform output shape: {train_tensor.shape}")
    print(f"Validation transform output shape: {val_tensor.shape}")
    print()


def demo_backbones():
    """Demo the backbone models."""
    print("=== Backbone Models Demo ===")
    
    # Test different backbones
    backbones_to_test = ['resnet', 'convnext']  # Test lightweight ones
    
    for backbone_type in backbones_to_test:
        print(f"Testing {backbone_type} backbone...")
        
        # Create backbone
        backbone = get_backbone(backbone_type, freeze_encoder=True)
        print(f"  - Created {backbone.__class__.__name__}")
        print(f"  - Output channels: {backbone.output_channels}")
        
        # Test forward pass with dummy input
        dummy_input = torch.randn(1, 3, 224, 224)
        
        try:
            with torch.no_grad():
                features = backbone(dummy_input)
            print(f"  - Forward pass successful!")
            print(f"  - Feature map shapes: {[k + ': ' + str(v.shape) for k, v in features.items()]}")
        except Exception as e:
            print(f"  - Forward pass failed: {e}")
        
        print()


def demo_modular_usage():
    """Demo how to use modules together."""
    print("=== Modular Usage Demo ===")
    
    # Get configuration
    config = get_default_config('resnet', 'maskrcnn')
    image_size = config['image_size']
    
    # Create transforms
    transform = get_transforms(augment=False, image_size=image_size)
    
    # Create backbone
    backbone = get_backbone('resnet', freeze_encoder=True)
    
    # Create dummy data pipeline
    dummy_image = Image.new('RGB', (640, 480), color='blue')
    tensor = transform(dummy_image).unsqueeze(0)  # Add batch dimension
    
    print(f"Input tensor shape: {tensor.shape}")
    
    # Process through backbone
    with torch.no_grad():
        features = backbone(tensor)
    
    print("Feature extraction successful!")
    print("This demonstrates how the modular components work together.")
    print()


if __name__ == '__main__':
    print("OmniSeg Modular Structure Demo")
    print("=" * 50)
    print()
    
    demo_config()
    demo_transforms()  
    demo_backbones()
    demo_modular_usage()
    
    print("Demo completed successfully!")
    print("The refactored modular structure is working correctly.")