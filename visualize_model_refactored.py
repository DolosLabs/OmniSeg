# visualize_model_refactored.py
import os
import sys
import argparse
import random
import copy
import math
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from types import SimpleNamespace
import torchvision.transforms.v2 as T

# Import from our modular structure
from omniseg.config import PROJECT_DIR
from omniseg.data import SemiCOCODataset, COCODataModule, get_transforms
from omniseg.models.base import BaseBackbone, BaseHead
from omniseg.models.backbones import get_backbone

# Import complex classes from original train.py for now
# Import from the modular structure
from omniseg.training import SSLSegmentationLightning, masks_to_boxes  
from omniseg.models.heads.contourformer import masks_to_contours, contours_to_masks


def visualize_model(checkpoint_path, project_dir, num_images_to_viz=5):
    """
    Loads a trained model checkpoint and visualizes predictions on a few
    random validation images using our refactored modular structure.
    """
    # Load the model from checkpoint
    model = SSLSegmentationLightning.load_from_checkpoint(checkpoint_path)
    model.eval()
    
    # Setup data module using our refactored data classes
    coco_datamodule = COCODataModule(project_dir=project_dir, batch_size=1, num_workers=0)
    coco_datamodule.prepare_data()
    coco_datamodule.setup()
    
    val_dataset = coco_datamodule.val_ds
    val_transform = get_transforms(augment=False, image_size=model.hparams.image_size)
    
    # Randomly sample images
    random.seed(42)
    sample_indices = random.sample(range(len(val_dataset)), min(num_images_to_viz, len(val_dataset)))
    
    fig, axes = plt.subplots(num_images_to_viz, 3, figsize=(15, 5 * num_images_to_viz))
    if num_images_to_viz == 1:
        axes = axes.reshape(1, -1)
    
    for i, idx in enumerate(sample_indices):
        # Get original image and target
        image_pil, target = val_dataset[idx]
        original_size = image_pil.size[::-1]  # (height, width)
        
        # Transform for model input
        image_tensor = val_transform(image_pil).unsqueeze(0)
        
        # Get predictions
        with torch.no_grad():
            outputs = model.student(image_tensor)

            # Handle different head types for predictions
            if model.hparams.head_type == 'maskrcnn':
                predictions = outputs[0]
                pred_masks = (predictions["masks"] > 0.5).squeeze(1).cpu().numpy()
                pred_labels = predictions["labels"].cpu().numpy()
                pred_scores = predictions["scores"].cpu().numpy()
                
                # Filter by confidence
                keep_mask = pred_scores > 0.5
                pred_masks = pred_masks[keep_mask]
                pred_labels = pred_labels[keep_mask]
                pred_scores = pred_scores[keep_mask]
                
            elif model.hparams.head_type == 'contourformer':
                scores = outputs['pred_logits'].softmax(-1)[0, :, :-1]
                max_scores, labels = scores.max(-1)
                
                keep_mask = max_scores > 0.5
                pred_contours = outputs['pred_contours'][0][keep_mask].cpu().numpy()
                pred_labels = labels[keep_mask].cpu().numpy()
                pred_scores = max_scores[keep_mask].cpu().numpy()
                
                # Convert contours to masks
                pred_masks = contours_to_masks(pred_contours, original_size)
                
            else:  # deformable_detr or other DETR variants
                # Simplified handling for demo
                pred_masks = np.zeros((1, original_size[0], original_size[1]))
                pred_labels = np.array([0])
                pred_scores = np.array([0.0])
        
        # Visualize
        # Original image
        axes[i, 0].imshow(image_pil)
        axes[i, 0].set_title(f"Original Image {idx}")
        axes[i, 0].axis('off')
        
        # Ground truth masks
        if 'masks' in target and len(target['masks']) > 0:
            gt_masks = target['masks'].numpy()
            combined_gt = np.zeros(original_size + (3,))
            colors = plt.cm.Set3(np.linspace(0, 1, len(gt_masks)))
            
            for mask, color in zip(gt_masks, colors):
                mask_resized = np.array(Image.fromarray(mask.astype(np.uint8) * 255).resize(image_pil.size[::-1]))
                for c in range(3):
                    combined_gt[:, :, c] += (mask_resized > 0) * color[c]
            
            combined_gt = np.clip(combined_gt, 0, 1)
            overlay_gt = 0.6 * np.array(image_pil.resize(original_size[::-1])) / 255.0 + 0.4 * combined_gt
        else:
            overlay_gt = np.array(image_pil)
            
        axes[i, 1].imshow(overlay_gt)
        axes[i, 1].set_title(f"Ground Truth ({len(target.get('masks', []))} masks)")
        axes[i, 1].axis('off')
        
        # Predicted masks
        if len(pred_masks) > 0:
            combined_pred = np.zeros(original_size + (3,))
            colors = plt.cm.Set1(np.linspace(0, 1, len(pred_masks)))
            
            for mask, color, score in zip(pred_masks, colors, pred_scores):
                mask_resized = np.array(Image.fromarray((mask > 0.5).astype(np.uint8) * 255).resize((original_size[1], original_size[0])))
                for c in range(3):
                    combined_pred[:, :, c] += (mask_resized > 0) * color[c]
            
            combined_pred = np.clip(combined_pred, 0, 1)
            overlay_pred = 0.6 * np.array(image_pil.resize(original_size[::-1])) / 255.0 + 0.4 * combined_pred
        else:
            overlay_pred = np.array(image_pil)
            
        axes[i, 2].imshow(overlay_pred)
        axes[i, 2].set_title(f"Predictions ({len(pred_masks)} masks)")
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    # Save the visualization
    output_dir = os.path.join(project_dir, 'visualizations')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'model_predictions_refactored.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Visualization saved to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize predictions of a trained model using refactored modular structure.")
    parser.add_argument('checkpoint', type=str, help="Path to the model checkpoint (.ckpt) file.")
    parser.add_argument('--project_dir', type=str, default=PROJECT_DIR, help="Project directory where data is stored.")
    args = parser.parse_args()
    
    visualize_model(args.checkpoint, args.project_dir)