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
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping # MODIFIED: Import EarlyStopping
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
from omniseg.models.heads import get_head
from omniseg.training import SSLSegmentationLightning

# The complex head implementations have been extracted to separate files
# in omniseg/models/heads/


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
    parser.add_argument('--use_tiny_data', action='store_true', 
                        help="Use the tiny synthetic dataset for quick training/testing (run generate_tiny_data.py first).")
    
    # NEW: Add arguments for early stopping
    parser.add_argument('--early_stopping_patience', type=int, default=25,
                        help="Patience for early stopping. Training stops after this many epochs of no improvement.")
    parser.add_argument('--early_stopping_monitor', type=str, default='val_mAP',
                        help="Metric to monitor for early stopping.")
    parser.add_argument('--early_stopping_mode', type=str, choices=['min', 'max'], default='max',
                        help="Mode for early stopping ('min' for loss, 'max' for accuracy/mAP).")
    
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
    print(f"Use tiny data: {args.use_tiny_data}")
    print(f"Early Stopping Patience: {args.early_stopping_patience} epochs monitoring '{args.early_stopping_monitor}'")


    # Set number of classes based on dataset
    num_classes = 3 if args.use_tiny_data else NUM_CLASSES
    print(f"Number of classes: {num_classes}")

    # Create run directory
    dataset_suffix = "_tiny" if args.use_tiny_data else ""
    run_name = f"{args.backbone}_{args.head}_{image_size}{dataset_suffix}"
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
        num_classes=num_classes, 
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
    
    # NEW: Instantiate the EarlyStopping callback
    early_stopping_callback = EarlyStopping(
        monitor=args.early_stopping_monitor,
        patience=args.early_stopping_patience,
        mode=args.early_stopping_mode,
        verbose=True
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
        # MODIFIED: Add the early stopping callback to the list
        callbacks=[checkpoint_callback, early_stopping_callback],
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
