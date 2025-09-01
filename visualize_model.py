# visualize_model.py

import argparse
import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import Dict, Any, Tuple

# Import from our modular structure
from omniseg.config import PROJECT_DIR
from omniseg.data import COCODataModule, get_transforms
from omniseg.training import SSLSegmentationLightning
from omniseg.models.heads.contourformer import contours_to_masks

# --- Constants ---
CONFIDENCE_THRESHOLD = 0.5
RANDOM_SEED = 42

# --- Helper Functions ---

def get_device() -> torch.device:
    """Determines the available compute device (GPU or CPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

def process_predictions(model_hparams: Dict[str, Any], raw_outputs: Any, original_size: Tuple[int, int]) -> Dict[str, np.ndarray]:
    """
    Processes raw model outputs into a standardized dictionary of numpy arrays.

    Args:
        model_hparams: Hyperparameters from the loaded model.
        raw_outputs: The raw output from the model's forward pass.
        original_size: The (height, width) of the original image.

    Returns:
        A dictionary containing processed 'masks', 'labels', and 'scores'.
    """
    head_type = model_hparams.get('head_type', 'unknown')

    if head_type == 'maskrcnn':
        predictions = raw_outputs[0]
        scores = predictions["scores"].detach()
        keep_mask = scores > CONFIDENCE_THRESHOLD
        
        return {
            "masks": (predictions["masks"][keep_mask] > 0.5).squeeze(1).cpu().numpy(),
            "labels": predictions["labels"][keep_mask].detach().cpu().numpy(),
            "scores": scores[keep_mask].cpu().numpy()
        }
        
    elif head_type == 'contourformer':
        # Handle tuple vs dict outputs
        if isinstance(raw_outputs, (tuple, list)) and len(raw_outputs) == 2:
            logits, contours = raw_outputs
        elif isinstance(raw_outputs, dict):
            logits = raw_outputs.get("pred_logits")
            # accept both naming variants
            contours = raw_outputs.get("pred_contours") or raw_outputs.get("pred_coords")
        else:
            logits, contours = None, None

        if logits is None or contours is None:
            print("⚠️ Unexpected ContourFormer raw_outputs format")
            if isinstance(raw_outputs, dict):
                print("Dict keys:", list(raw_outputs.keys()))
            else:
                print("Type:", type(raw_outputs))
            raise ValueError("ContourFormer outputs missing 'logits' or 'contours/coords'")

        scores_softmax = logits.softmax(-1)[0, :, :-1]
        scores, labels = scores_softmax.max(-1)
        keep_mask = scores > CONFIDENCE_THRESHOLD

        contours_np = contours[0][keep_mask].detach().cpu()

        return {
            "masks": contours_to_masks(contours_np, original_size),
            "labels": labels[keep_mask].detach().cpu().numpy(),
            "scores": scores[keep_mask].detach().cpu().numpy()
        }

    else:  # Fallback for other heads
        print(f"Warning: Unsupported head type '{head_type}'. Returning empty predictions.")
        return {
            "masks": np.array([]),
            "labels": np.array([]),
            "scores": np.array([])
        }

def create_mask_overlay(image: Image.Image, masks: np.ndarray, colors: np.ndarray) -> np.ndarray:
    """Creates a colored overlay from masks on top of an image."""
    if masks.ndim == 2:  # Handle single mask case
        masks = masks[np.newaxis, ...]
        
    combined_mask_color = np.zeros((*image.size[::-1], 3), dtype=np.float32)
    
    for i, mask in enumerate(masks):
        color = colors[i % len(colors), :3]  # Cycle through colors
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()

        mask_pil = Image.fromarray(((mask > 0.5).astype(np.uint8)) * 255)
        resized_mask = np.array(mask_pil.resize(image.size)) > 0
        
        for c in range(3):
            combined_mask_color[resized_mask, c] = color[c]
            
    image_arr = np.array(image) / 255.0
    overlay = 0.6 * image_arr + 0.4 * combined_mask_color
    return np.clip(overlay, 0, 1)

# --- Main Visualization Logic ---

def visualize_model(checkpoint_path: str, project_dir: str, num_images: int):
    """
    Loads a model and visualizes its predictions on validation images.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at: {checkpoint_path}")

    device = get_device()
    
    model = SSLSegmentationLightning.load_from_checkpoint(checkpoint_path, map_location=device)
    model.eval()

    datamodule = COCODataModule(project_dir=project_dir, batch_size=1, num_workers=0)
    datamodule.setup()
    val_dataset = datamodule.val_ds
    val_transform = get_transforms(augment=False, image_size=model.hparams.image_size)

    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    indices = random.sample(range(len(val_dataset)), min(num_images, len(val_dataset)))
    fig, axes = plt.subplots(len(indices), 3, figsize=(15, 5 * len(indices)), squeeze=False)
    
    cmap = plt.colormaps.get('gist_rainbow')
    colors_list = cmap(np.linspace(0, 1, 50))  # more colors available

    for i, idx in enumerate(indices):
        image_pil, target = val_dataset[idx]
        original_size = image_pil.size[::-1]
        
        image_tensor = val_transform(image_pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            raw_outputs = model.student(image_tensor)
            predictions = process_predictions(model.hparams, raw_outputs, original_size)

        axes[i, 0].imshow(image_pil)
        axes[i, 0].set_title(f"Original Image (Index: {idx})")

        gt_masks = target.get('masks')
        if gt_masks is not None and len(gt_masks) > 0:
            gt_overlay = create_mask_overlay(image_pil, gt_masks.numpy(), colors_list)
            axes[i, 1].imshow(gt_overlay)
        else:
            axes[i, 1].imshow(image_pil)
        axes[i, 1].set_title(f"Ground Truth ({0 if gt_masks is None else len(gt_masks)} masks)")

        pred_masks = predictions['masks']
        if pred_masks.shape[0] > 0:
            pred_overlay = create_mask_overlay(image_pil, pred_masks, colors_list)
            axes[i, 2].imshow(pred_overlay)
        else:
            axes[i, 2].imshow(image_pil)
        axes[i, 2].set_title(f"Predictions ({len(pred_masks)} masks)")
        
        for ax in axes[i]:
            ax.axis('off')

    plt.tight_layout()
    output_dir = os.path.join(project_dir, 'visualizations')
    os.makedirs(output_dir, exist_ok=True)
    
    backbone = model.hparams.get('backbone_type', 'unknown_backbone')
    head = model.hparams.get('head_type', 'unknown_head')
    image_prefix = f"{backbone}_{head}"
    
    output_path = os.path.join(output_dir, f'{image_prefix}_predictions_{os.path.basename(checkpoint_path)}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Visualization saved successfully to: {output_path}")

# --- Script Entrypoint ---

def main():
    """Parses command-line arguments and runs the visualization."""
    parser = argparse.ArgumentParser(description="Visualize predictions of a trained OmniSeg model.")
    parser.add_argument("checkpoint", type=str, help="Path to the model checkpoint (.ckpt) file.")
    parser.add_argument("--project_dir", type=str, default=PROJECT_DIR, help="Project directory where data is stored.")
    parser.add_argument("--num_images", type=int, default=5, help="Number of random images to visualize.")
    args = parser.parse_args()
    
    visualize_model(args.checkpoint, args.project_dir, args.num_images)

if __name__ == '__main__':
    main()
