#!/usr/bin/env python3
# train_refactored.py
# Semi-Supervised Instance Segmentation with Multiple Backbones and Heads (PyTorch Lightning Version)
#
# This is the refactored version that demonstrates modular imports.
# The complex head implementations remain in the original file for brevity.

import os
import sys
import copy
import math
import json
import argparse
import random
from collections import defaultdict, OrderedDict
from types import SimpleNamespace
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.v2 as T

import pytorch_lightning as pl
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import AutoModel, AutoConfig
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import FeaturePyramidNetwork, box_convert, generalized_box_iou
import timm
from scipy.optimize import linear_sum_assignment
from skimage.measure import find_contours
from scipy.interpolate import interp1d

# Import from our modular structure
from omniseg.config import (
    PROJECT_DIR, EPOCHS, NUM_CLASSES, BACKBONE_CONFIGS, HEAD_CONFIGS,
    get_default_config, get_available_backbones, get_available_heads
)
from omniseg.data import COCODataModule, download_coco2017, get_transforms
from omniseg.models.base import BaseBackbone, BaseHead
from omniseg.models.backbones import get_backbone


# For now, keep the complex head implementations here
# In a full refactoring, these would be moved to separate files as well

# --- Head Architectures ---
class GenericBackboneWithFPN(nn.Module):
    """Generic backbone with Feature Pyramid Network."""
    
    def __init__(self, backbone_type: str, image_size: int = 224):
        super().__init__()
        self.backbone = get_backbone(backbone_type, image_size=image_size)
        
        backbone_channels = self.backbone.output_channels
        fpn_input_channels = list(backbone_channels.values())[-4:]  # Use last 4 levels
        
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=fpn_input_channels,
            out_channels=256,
            extra_blocks=None
        )
    
    def forward(self, x):
        features = self.backbone(x)
        # Get the last 4 feature levels for FPN
        fpn_inputs = OrderedDict()
        feature_keys = sorted(features.keys())[-4:]
        for i, key in enumerate(feature_keys):
            fpn_inputs[str(i)] = features[key]
        
        return self.fpn(fpn_inputs)


class MaskRCNNHead(BaseHead, nn.Module):
    """Mask R-CNN head implementation."""
    
    def __init__(self, num_classes: int, backbone_type: str = 'dino', image_size: int = 224):
        BaseHead.__init__(self, num_classes, backbone_type, image_size)
        nn.Module.__init__(self)
        
        self.backbone_fpn = GenericBackboneWithFPN(backbone_type, image_size)
        
        anchor_generator = AnchorGenerator(
            sizes=((32,), (64,), (128,), (256,), (512,)),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5
        )
        
        self.maskrcnn = MaskRCNN(
            backbone=self.backbone_fpn,
            num_classes=num_classes + 1,  # +1 for background
            rpn_anchor_generator=anchor_generator,
            box_detections_per_img=100,
            rpn_pre_nms_top_n_train=2000,
            rpn_pre_nms_top_n_test=1000,
            rpn_post_nms_top_n_train=2000,
            rpn_post_nms_top_n_test=1000
        )
    
    def forward(self, pixel_values: torch.Tensor, targets: Optional[List[Dict]] = None) -> Dict[str, torch.Tensor]:
        if targets is not None:
            # Training mode
            return self.maskrcnn(pixel_values, targets)
        else:
            # Inference mode
            return self.maskrcnn(pixel_values)


# For the complex DETR and ContourFormer heads, they would need to be extracted
# to separate files. For now, we'll include a simplified factory

def get_head(head_type: str, num_classes: int, **kwargs) -> BaseHead:
    """Factory function to create a head instance."""
    if head_type == 'maskrcnn':
        return MaskRCNNHead(num_classes, **kwargs)
    else:
        # Import the full original heads for complex ones
        from train import DeformableDETRHead, ContourFormerHead
        if head_type == 'deformable_detr':
            return DeformableDETRHead(num_classes, **kwargs)
        elif head_type == 'contourformer':
            return ContourFormerHead(num_classes, **kwargs)
        else:
            raise ValueError(f"Unsupported head_type: {head_type}")


# Import the full training module for now
from train import SSLSegmentationLightning


# --- Main Training Script ---
def main():
    parser = argparse.ArgumentParser(description="Train a Semi-Supervised Instance Segmentation Model.")
    parser.add_argument('--backbone', type=str, choices=get_available_backbones(), default='resnet',
                       help="Choose the backbone model.")
    parser.add_argument('--head', type=str, choices=get_available_heads(), default='maskrcnn',
                       help="Choose the segmentation head.")
    parser.add_argument('--learning_rate', type=float, default=5e-5, help="Learning rate for the optimizer.")
    parser.add_argument('--image_size', type=int, default=None, 
                       help="Custom image size for resizing. Overrides backbone-specific defaults.")
    parser.add_argument('--batch_size', type=int, default=None, help="Override the default batch size.")
    parser.add_argument('--num_workers', type=int, default=2, help="Number of workers for data loading.")
    parser.add_argument('--fast_dev_run', action='store_true', 
                       help="Run a single batch for training and validation to check for errors.")
    parser.add_argument('--max_steps', type=int, default=-1, 
                       help="Total number of training steps to perform. Overrides max_epochs.")
    parser.add_argument('--num_labeled_images', type=int, default=-1, 
                       help="Number of labeled images to use for training. -1 for all.")
    parser.add_argument('--num_unlabeled_images', type=int, default=-1, 
                       help="Number of unlabeled images to use for training. -1 for all.")
    parser.add_argument('--accumulate_grad_batches', type=int, default=1, 
                       help="Accumulate gradients over N batches.")
    parser.add_argument('--find_unused_parameters', action='store_true', 
                       help="Enable 'find_unused_parameters' for DDP. May slightly slow down training.")
    parser.add_argument('--warmup_steps', type=int, default=500, 
                       help="Number of initial steps to train on labeled data only.")
    parser.add_argument('--unsup_rampup_steps', type=int, default=1000, 
                       help="Number of steps to ramp up unsupervised loss weight after warmup.")
    parser.add_argument('--val_every_n_epoch', type=int, default=1, help="Run validation every N epochs.")
    args = parser.parse_args()
    
    # Get default configuration for the backbone-head combination
    default_config = get_default_config(args.backbone, args.head)
    
    # Apply user overrides or use defaults
    image_size = args.image_size if args.image_size is not None else default_config['image_size']
    batch_size = args.batch_size if args.batch_size is not None else default_config['batch_size']
    precision_setting = default_config['precision']

    if args.backbone == 'swin':
        swin_model_name = 'microsoft/swin-base-patch4-window7-224-in22k'
        try:
            config = AutoConfig.from_pretrained(swin_model_name)
            patch_size, window_size = config.patch_size, config.window_size
            divisor = patch_size * window_size
            if image_size % divisor != 0:
                adjusted_size = ((image_size // divisor) + 1) * divisor
                print(f"Adjusting image size from {image_size} to {adjusted_size} for Swin compatibility.")
                image_size = adjusted_size
        except Exception as e:
            print(f"Could not adjust image size for Swin: {e}")

    print(f"\n--- Training Configuration ---")
    print(f"Backbone: {args.backbone}")
    print(f"Head: {args.head}")
    print(f"Image size: {image_size}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Precision: {precision_setting}")

    # Create run directory
    run_name = f"{args.backbone}_{args.head}_{image_size}"
    run_dir = os.path.join(PROJECT_DIR, 'runs', run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Initialize data module using our modular imports
    coco_datamodule = COCODataModule(
        project_dir=PROJECT_DIR, 
        batch_size=batch_size, 
        image_size=image_size, 
        num_workers=args.num_workers,
        num_labeled_images=args.num_labeled_images, 
        num_unlabeled_images=args.num_unlabeled_images
    )
    
    # Initialize model
    model = SSLSegmentationLightning(
        num_classes=NUM_CLASSES, 
        lr=args.learning_rate, 
        backbone_type=args.backbone,
        head_type=args.head, 
        image_size=image_size, 
        warmup_steps=args.warmup_steps,
        unsup_rampup_steps=args.unsup_rampup_steps
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=run_dir, 
        filename='best-model-{epoch:02d}-{val_mAP:.4f}',
        save_top_k=1, 
        verbose=True, 
        monitor='val_mAP', 
        mode='max', 
        save_last=True
    )
    
    strategy = "ddp_find_unused_parameters_true" if args.find_unused_parameters else "ddp"

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        max_steps=args.max_steps,
        accelerator="auto",
        devices="auto",
        strategy=strategy,
        accumulate_grad_batches=args.accumulate_grad_batches,
        check_val_every_n_epoch=args.val_every_n_epoch,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10,
        precision=precision_setting,
        default_root_dir=run_dir,
        fast_dev_run=args.fast_dev_run,
        logger=True
    )

    last_ckpt_path = os.path.join(run_dir, "last.ckpt")
    resume_path = last_ckpt_path if os.path.exists(last_ckpt_path) else None
    trainer.fit(model, datamodule=coco_datamodule, ckpt_path=resume_path)

    if not args.fast_dev_run:
        print("\n--- Running final validation on the best checkpoint ---")
        trainer.validate(model, datamodule=coco_datamodule, ckpt_path='best')


if __name__ == '__main__':
    main()