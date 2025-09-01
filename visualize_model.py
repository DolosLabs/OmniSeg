# visualize_model.py
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

# Adjust the Python path to import from your main script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from the modular structure
from omniseg.config import get_available_backbones, get_default_config
from omniseg.data import SemiCOCODataset, COCODataModule, get_transforms
from omniseg.models.backbones import get_backbone
from omniseg.models.heads import get_head
from omniseg.models.heads.contourformer import masks_to_contours, contours_to_masks  
from omniseg.training import SSLSegmentationLightning, masks_to_boxes


def visualize_model(checkpoint_path, project_dir, num_images_to_viz=5):
    """
    Loads a trained model checkpoint and visualizes predictions on a few
    random validation images.
    """
    print(f"Loading model from checkpoint: {checkpoint_path}")
    
    # 1. Load the model from the checkpoint. This automatically sets up the
    #    architecture and loads the weights.
    model = SSLSegmentationLightning.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.freeze()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 2. Instantiate the DataModule to get the validation dataset
    image_size = model.hparams.image_size
    coco_datamodule = COCODataModule(project_dir=project_dir, image_size=image_size)
    coco_datamodule.prepare_data()
    coco_datamodule.setup()
    val_dataset = coco_datamodule.val_ds
    
    print(f"Using validation dataset with {len(val_dataset)} images.")

    # 3. Randomly select a few images from the validation dataset
    random_indices = random.sample(range(len(val_dataset)), min(num_images_to_viz, len(val_dataset)))

    # 4. Create a figure to plot the visualizations
    fig, axes = plt.subplots(num_images_to_viz, 3, figsize=(15, 5 * num_images_to_viz))
    if num_images_to_viz == 1:
        axes = axes.reshape(1, 3)

    # 5. Iterate through the selected images, predict, and plot
    for i, idx in enumerate(random_indices):
        image_pil, target = val_dataset[idx]

        # Get original image size for post-processing
        img_info = coco_datamodule.coco_gt_val.loadImgs(target['image_id'])[0]
        original_size = (img_info['height'], img_info['width'])
        
        # --- PREDICT ---
        with torch.no_grad():
            pixel_values = model.val_aug(image_pil, None)[0].unsqueeze(0).to(device)
            outputs = model.student(pixel_values)

            # --- Handle different head types for predictions ---
            if model.hparams.head_type == 'mask2former':
                outputs_ns = SimpleNamespace(**outputs)
                results = model.image_processor_m2f.post_process_instance_segmentation(outputs_ns, target_sizes=[original_size])[0]
                pred_masks = results['segmentation'].cpu().numpy() if 'segmentation' in results else np.zeros((1, original_size[0], original_size[1]))
            elif model.hparams.head_type == 'maskrcnn':
                # The output for maskrcnn is a list of dictionaries. We take the first one.
                predictions = outputs[0]
                pred_masks = (predictions["masks"] > 0.5).squeeze(1).cpu().numpy()
            elif model.hparams.head_type == 'contourformer':
                scores = outputs['pred_logits'].softmax(-1)[0, :, :-1]
                max_scores, labels = scores.max(-1)
                keep = max_scores > 0.05
                pred_coords = outputs['pred_coords'][0][keep]
                if pred_coords.shape[0] > 0:
                    pred_masks = contours_to_masks(pred_coords, (image_size, image_size))
                    pred_masks = F.interpolate(pred_masks.unsqueeze(1).float(), size=original_size, mode='bilinear', align_corners=False)[:, 0].bool().cpu().numpy()
                else:
                    pred_masks = np.empty((0, original_size[0], original_size[1]), dtype=bool)

        # --- PLOT ---
        # 1. Original Image
        axes[i, 0].imshow(image_pil)
        axes[i, 0].set_title(f'Original Image {i + 1}', fontsize=10)
        axes[i, 0].axis('off')

        # 2. Ground Truth Masks
        gt_masks = target['masks'].cpu().numpy().astype(bool)
        ax2 = axes[i, 1]
        ax2.imshow(image_pil)
        ax2.set_title(f'Actual Masks {i + 1}', fontsize=10)
        ax2.axis('off')
        colors = plt.cm.get_cmap('Paired', len(gt_masks))
        for k, mask in enumerate(gt_masks):
            colored_mask = np.zeros((*original_size, 4))
            color = colors(k)
            colored_mask[mask] = color
            ax2.imshow(colored_mask, alpha=0.5)

        # 3. Predicted Masks
        ax3 = axes[i, 2]
        ax3.imshow(image_pil)
        ax3.set_title(f'Predicted Masks {i + 1}', fontsize=10)
        ax3.axis('off')
        if pred_masks.ndim == 3 and pred_masks.shape[0] > 0:
            colors_pred = plt.cm.get_cmap('hsv', pred_masks.shape[0])
            for k, mask in enumerate(pred_masks):
                if mask.sum() > 0:
                    colored_mask = np.zeros((*original_size, 4))
                    color = colors_pred(k)
                    colored_mask[mask] = color
                    ax3.imshow(colored_mask, alpha=0.5)

    plt.tight_layout()
    output_dir = os.path.dirname(checkpoint_path)
    viz_path = os.path.join(output_dir, f"model_predictions_viz.png")
    print(f"Saving visualization to {viz_path}")
    plt.savefig(viz_path)
    plt.show()
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize predictions of a trained model.")
    parser.add_argument('checkpoint', type=str, help="Path to the model checkpoint (.ckpt) file.")
    parser.add_argument('--project_dir', type=str, default='./SSL_Instance_Segmentation', help="Project directory where data is stored.")
    args = parser.parse_args()
    
    visualize_model(args.checkpoint, args.project_dir)
