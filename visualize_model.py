import argparse
import os
import random
import time
from typing import Dict, Any, Tuple

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Import from the omniseg project structure
from omniseg.config import PROJECT_DIR
from omniseg.data import COCODataModule
from omniseg.training import SSLSegmentationLightning
from omniseg.models.heads.contourformer import contours_to_masks

# Import all possible head modules for the factory function
from omniseg.models.heads.deformable_detr import DETRSegmentationHead
from omniseg.models.heads.contourformer import ContourFormerHead
from omniseg.models.heads.maskrcnn import MaskRCNNHead
from omniseg.models.heads.lw_detr import LWDETRHead
from omniseg.models.heads.sparrow_seg import SparrowSegHead

# --- Constants ---
CONFIDENCE_THRESHOLD = 0.5
RANDOM_SEED = 42

# --- Model & Prediction Helpers (Largely unchanged) ---

def build_head_from_hparams(hparams: Dict[str, Any]) -> torch.nn.Module:
    """Instantiates the correct model head based on hyperparameters."""
    head_type = hparams.get("head_type")
    if not head_type:
        raise ValueError("Could not find 'head_type' in model hyperparameters.")

    num_classes = hparams.get("num_classes", 80)
    print(f"Building model head of type: '{head_type}'")

    if head_type == "deformable_detr":
        return DETRSegmentationHead(num_classes=num_classes, backbone_type=hparams.get("backbone_type", "dino"), image_size=hparams.get("image_size", 128), hidden_dim=hparams.get("d_model", 256), num_queries=hparams.get("num_queries", 52), dec_layers=hparams.get("num_decoder_layers", 4), num_groups=hparams.get("num_groups", 13), nheads=hparams.get("n_heads", 8), dim_feedforward=hparams.get("d_ffn", 2048), enc_layers=6, pre_norm=False, enforce_input_project=False, mask_classification=True)
    elif head_type == "lw_detr":
        return LWDETRHead(num_classes=num_classes, backbone_type=hparams.get("backbone_type", "dino"), image_size=hparams.get("image_size", 128), d_model=hparams.get("d_model", 256), num_queries=hparams.get("num_queries", 52), num_decoder_layers=hparams.get("num_decoder_layers", 4), n_heads=hparams.get("n_heads", 8), d_ffn=hparams.get("d_ffn", 1024), mask_dim=hparams.get("mask_dim", 16), num_groups=hparams.get("num_groups", 13))
    elif head_type == "contourformer":
        return ContourFormerHead(num_classes=num_classes, hidden_dim=384, num_queries=100, nheads=8, dim_feedforward=2048, enc_layers=4, dec_layers=4, pre_norm=False, control_points=16)
    elif head_type == "sparrow_seg":
        return SparrowSegHead(num_classes=num_classes, backbone_type=hparams.get("backbone_type", "dino"), image_size=hparams.get("image_size", 512), d_model=hparams.get("d_model", 256), num_queries=hparams.get("num_queries", 100), num_decoder_layers=hparams.get("num_decoder_layers", 2), n_heads=hparams.get("n_heads", 8), d_ffn=hparams.get("d_ffn", 1024), mask_dim=hparams.get("mask_dim", 256))
    elif head_type == "maskrcnn":
        return MaskRCNNHead(backbone_type=hparams.get("backbone_type", "dino"), num_classes=num_classes)
    else:
        raise NotImplementedError(f"Head type '{head_type}' is not supported by this script.")

def get_device() -> torch.device:
    """Determines the available compute device (GPU or CPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

def masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
    """Computes bounding boxes from masks."""
    if masks.numel() == 0:
        return torch.zeros((0, 4), dtype=torch.float32, device=masks.device)
    n = masks.shape[0]
    boxes = torch.zeros(n, 4, dtype=torch.float32, device=masks.device)
    for i in range(n):
        y, x = torch.where(masks[i])
        if x.numel() == 0:
            continue
        boxes[i] = torch.tensor([x.min(), y.min(), x.max(), y.max()], dtype=torch.float32)
    return boxes

def process_predictions(model_hparams: Dict[str, Any], raw_outputs: Any, original_size: Tuple[int, int]) -> Dict[str, np.ndarray]:
    """Processes raw model outputs into a standardized dictionary of numpy arrays."""
    head_type = model_hparams.get("head_type", "unknown")
    empty_preds = {"masks": np.array([]), "labels": np.array([]), "scores": np.array([]), "boxes": np.array([])}

    if head_type == "maskrcnn":
        predictions = raw_outputs[0]
        scores = predictions["scores"].detach()
        keep = scores > CONFIDENCE_THRESHOLD
        
        # --- FIX APPLIED ---
        return {
            "masks": (predictions["masks"][keep] > 0.5).squeeze(1).cpu().numpy(),
            "labels": (predictions["labels"][keep] + 1).cpu().numpy(),
            "scores": scores[keep].cpu().numpy(),
            "boxes": predictions["boxes"][keep].cpu().numpy()
        }

    elif head_type == "contourformer":
        logits = raw_outputs.get("pred_logits")
        contours = raw_outputs.get("pred_contours") or raw_outputs.get("pred_coords")
        if logits is None or contours is None: return empty_preds
        
        scores_softmax = logits.softmax(-1)[0, :, :-1]
        scores, labels = scores_softmax.max(-1)
        keep = scores > CONFIDENCE_THRESHOLD
        if not torch.any(keep): return empty_preds
        
        contours_tensor = contours[0, keep].detach().cpu()
        masks_tensor = contours_to_masks(contours_tensor, original_size)
        
        # --- FIX APPLIED ---
        return {
            "masks": masks_tensor.numpy(),
            "labels": (labels[keep] + 1).cpu().numpy(),
            "scores": scores[keep].cpu().numpy(),
            "boxes": masks_to_boxes(masks_tensor).cpu().numpy()
        }

    elif head_type in ["deformable_detr", "lw_detr", "sparrow_seg"]:
        logits, masks = raw_outputs.get("pred_logits"), raw_outputs.get("pred_masks")
        if logits is None or masks is None: return empty_preds
        
        scores_softmax = logits[0].softmax(-1)[:, :-1]
        scores, labels = scores_softmax.max(-1)
        keep = scores > CONFIDENCE_THRESHOLD
        if not torch.any(keep): return empty_preds
        
        selected_masks = F.interpolate(masks[0, keep].unsqueeze(1), size=original_size, mode="bilinear", align_corners=False).squeeze(1)
        binary_masks = (selected_masks > 0.5)
        
        # --- FIX APPLIED ---
        return {
            "masks": binary_masks.cpu().numpy(),
            "labels": (labels[keep] + 1).cpu().numpy(),
            "scores": scores[keep].cpu().numpy(),
            "boxes": masks_to_boxes(binary_masks).cpu().numpy()
        }
    
    return empty_preds
# --- Plotting Functions ---

def apply_masks_to_image(image_arr: np.ndarray, masks: np.ndarray, colors: np.ndarray) -> np.ndarray:
    """Creates a colored overlay from masks on top of an image array."""
    overlay = image_arr.copy()
    for i, mask in enumerate(masks):
        color = colors[i % len(colors)]
        overlay[mask] = (1 - 0.5) * overlay[mask] + 0.5 * np.array(color[:3])
    return np.clip(overlay, 0, 1)

def plot_coco_image(ax: plt.Axes, image: np.ndarray, title: str):
    """Plots the original COCO image."""
    ax.imshow(image)
    ax.set_title(title)
    ax.axis("off")

def plot_ground_truth_masks(ax: plt.Axes, image: np.ndarray, target: Dict, colors: np.ndarray):
    """Plots the ground truth masks on the image."""
    gt_masks = target.get("masks")
    num_masks = len(gt_masks) if gt_masks is not None else 0
    
    if num_masks > 0:
        overlay = apply_masks_to_image(image, gt_masks.numpy().astype(bool), colors)
        ax.imshow(overlay)
    else:
        ax.imshow(image)
        
    ax.set_title(f"Ground Truth ({num_masks} masks)")
    ax.axis("off")

def plot_predicted_masks(ax: plt.Axes, image: np.ndarray, predictions: Dict, colors: np.ndarray, category_map: Dict):
    """Plots the predicted masks, bounding boxes, and labels on the image."""
    pred_masks = predictions["masks"]
    num_masks = len(pred_masks)
    
    if num_masks > 0:
        overlay = apply_masks_to_image(image, pred_masks, colors)
        ax.imshow(overlay)
        
        for i in range(num_masks):
            box, label_id, score = predictions["boxes"][i], predictions["labels"][i], predictions["scores"][i]
            color = colors[i % len(colors)]
            class_name = category_map.get(label_id, f"ID: {label_id}")
            x_min, y_min, x_max, y_max = box
            
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor=color, facecolor="none")
            ax.add_patch(rect)
            ax.text(x_min, y_min, f"{class_name}: {score:.2f}", bbox=dict(facecolor=color, alpha=0.7), fontsize=7, color="white", va="bottom")
    else:
        ax.imshow(image)
        
    ax.set_title(f"Predictions ({num_masks} masks)")
    ax.axis("off")

# --- Main Visualization Logic ---

def visualize_model(checkpoint_path: str, project_dir: str, num_images: int, backbone: str):
    """Loads a model and visualizes its predictions on COCO validation images."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # --- Setup ---
    device = get_device()
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # --- Load Model ---
    print("Loading model from checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    hparams = checkpoint["hyper_parameters"]
    hparams["backbone_type"] = backbone
    head = build_head_from_hparams(hparams)
    model = SSLSegmentationLightning.load_from_checkpoint(checkpoint_path, map_location=device, head=head, strict=False)
    model.eval().to(device)

    # --- Load Data ---
    print("Setting up COCO data module...")
    datamodule = COCODataModule(project_dir=project_dir, batch_size=1, num_workers=0)
    datamodule.setup()
    val_dataset = datamodule.val_ds
    category_map = {cat["id"]: cat["name"] for cat in datamodule.val_ds.coco.cats.values()}
    
    # --- Plotting ---
    indices = random.sample(range(len(val_dataset)), min(num_images, len(val_dataset)))
    fig, axes = plt.subplots(len(indices), 3, figsize=(18, 6 * len(indices)), squeeze=False)
    cmap = plt.get_cmap("gist_rainbow")
    colors = cmap(np.linspace(0, 1, 80))

    for i, idx in enumerate(indices):
        image_tensor, target = val_dataset[idx]
        
        # --- FIXED LINE ---
        # Handle the case where 'size' may or may not be in the target dict
        size_from_target = target.get("size")
        if size_from_target is not None:
            original_size = tuple(size_from_target.tolist())
        else:
            original_size = tuple(image_tensor.shape[1:])
        # --- END FIX ---

        display_image = np.clip(image_tensor.permute(1, 2, 0).cpu().numpy(), 0, 1)

        # --- Get Predictions ---
        with torch.no_grad():
            raw_outputs = model.student(image_tensor.unsqueeze(0).to(device))
            predictions = process_predictions(model.hparams, raw_outputs, original_size)

        # --- Create Plots for one image ---
        plot_coco_image(axes[i, 0], display_image, f"Original Image (Index: {idx})")
        plot_ground_truth_masks(axes[i, 1], display_image, target, colors)
        plot_predicted_masks(axes[i, 2], display_image, predictions, colors, category_map)

    # --- Save and Show Figure ---
    plt.tight_layout()
    output_dir = os.path.join(project_dir, "visualizations")
    os.makedirs(output_dir, exist_ok=True)
    
    head_name = model.hparams.get("head_type", "unknown_head")
    ckpt_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_path = os.path.join(output_dir, f"{backbone}_{head_name}_{ckpt_name}_{timestamp}.png")
    
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"✅ Visualization saved successfully to: {output_path}")

# --- Script Entrypoint ---
def main():
    parser = argparse.ArgumentParser(description="Visualize predictions of a trained OmniSeg model.")
    parser.add_argument("checkpoint", type=str, help="Path to the model checkpoint (.ckpt) file.")
    parser.add_argument("--project_dir", type=str, default=PROJECT_DIR, help="Project directory where data is stored.")
    parser.add_argument("--num_images", type=int, default=5, help="Number of random images to visualize.")
    parser.add_argument("--backbone", type=str, default="dino", help="Backbone type (e.g., 'dino').")
    args = parser.parse_args()

    visualize_model(args.checkpoint, args.project_dir, args.num_images, args.backbone)

if __name__ == "__main__":
    main()
