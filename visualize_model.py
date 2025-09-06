import argparse
import time
import os
import random
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from typing import Dict, Any, Tuple, List

# Import from our modular structure
from omniseg.config import PROJECT_DIR
from omniseg.data import COCODataModule
from omniseg.training import SSLSegmentationLightning
from omniseg.models.heads.contourformer import contours_to_masks

# --- Import all possible head modules ---
from omniseg.models.heads.deformable_detr import DETRSegmentationHead
from omniseg.models.heads.contourformer import ContourFormerHead
from omniseg.models.heads.maskrcnn import MaskRCNNHead
from omniseg.models.heads.lw_detr import LWDETRHead
from omniseg.models.heads.sparrow_seg import SparrowSegHead

# --- Constants ---
RANDOM_SEED = 42

# --- Helper Functions ---

def build_head_from_hparams(hparams: Dict[str, Any]) -> torch.nn.Module:
    """Instantiates the correct model head based on hyperparameters."""
    head_type = hparams.get('head_type')
    if not head_type:
        raise ValueError("Could not find 'head_type' in model hyperparameters.")
    num_classes = hparams.get('num_classes', 80)
    print(f"Building model head of type: '{head_type}'")
    if head_type == 'deformable_detr':
        return DETRSegmentationHead(
            num_classes=num_classes,
            backbone_type=hparams.get('backbone_type', 'dino'),
            image_size=hparams.get('image_size', 128),
            d_model=hparams.get('d_model', 256),
            num_queries=hparams.get('num_queries', 52),
            num_decoder_layers=hparams.get('num_decoder_layers', 4),
            num_groups=hparams.get('num_groups', 13),
            nhead=hparams.get('n_heads', 8),
            dim_feedforward=hparams.get('d_ffn', 2048)
        )
    elif head_type == 'lw_detr':
        return LWDETRHead(
            num_classes=num_classes,
            backbone_type=hparams.get('backbone_type', 'dino'),
            image_size=hparams.get('image_size', 128),
            d_model=hparams.get('d_model', 256),
            num_queries=hparams.get('num_queries', 52),
            num_decoder_layers=hparams.get('num_decoder_layers', 4),
            n_heads=hparams.get('n_heads', 8),
            d_ffn=hparams.get('d_ffn', 1024),
            mask_dim=hparams.get('mask_dim', 16),
            num_groups=hparams.get('num_groups', 13)
        )
    elif head_type == 'contourformer':
        return ContourFormerHead(
            num_classes=num_classes,
            hidden_dim=384,
            num_queries=100,
            nheads=8,
            dim_feedforward=2048,
            enc_layers=4,
            dec_layers=4,
            pre_norm=False,
            control_points=16
        )
    elif head_type == 'sparrow_seg':
        return SparrowSegHead(
            num_classes=num_classes,
            backbone_type=hparams.get('backbone_type', 'dino'),
            image_size=hparams.get('image_size', 512),
            d_model=hparams.get('d_model', 256),
            num_queries=hparams.get('num_queries', 100),
            num_decoder_layers=hparams.get('num_decoder_layers', 2),
            n_heads=hparams.get('n_heads', 8),
            d_ffn=hparams.get('d_ffn', 1024),
            mask_dim=hparams.get('mask_dim', 256)
        )
    elif head_type == 'maskrcnn':
        return MaskRCNNHead(
            backbone_type=hparams.get('backbone_type', 'dino'),
            num_classes=num_classes
        )
    else:
        raise NotImplementedError(f"Head type '{head_type}' is not supported by this script.")

def get_device() -> torch.device:
    """Determines the available compute device (GPU or CPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

def masks_to_boxes(masks: np.ndarray) -> np.ndarray:
    """Computes bounding boxes from masks in (x_min, y_min, x_max, y_max) format."""
    if masks.size == 0:
        return np.zeros((0, 4), dtype=np.float32)

    boxes = np.zeros((masks.shape[0], 4), dtype=np.float32)
    for i, mask in enumerate(masks):
        y, x = np.where(mask)
        if len(x) == 0:
            boxes[i] = np.array([0, 0, 1, 1], dtype=np.float32)
        else:
            boxes[i] = np.array([x.min(), y.min(), x.max(), y.max()], dtype=np.float32)
    return boxes

def process_predictions(model_hparams: Dict[str, Any], raw_outputs: Any, original_size: Tuple[int, int], threshold: float) -> Dict[str, np.ndarray]:
    """Processes raw model outputs into a standardized dictionary of numpy arrays."""
    head_type = model_hparams.get('head_type', 'unknown')
    processed_preds = {"masks": np.array([]), "labels": np.array([]), "scores": np.array([]), "boxes": np.array([])}

    try:
        if head_type == 'maskrcnn':
            predictions = raw_outputs[0]
            scores = predictions["scores"].detach()
            keep = scores > threshold
            masks = (predictions["masks"][keep] > 0.5).squeeze(1).cpu().numpy()

            processed_preds = {
                "scores": scores[keep].cpu().numpy(),
                "labels": predictions["labels"][keep].cpu().numpy(),
                "masks": masks,
                "boxes": masks_to_boxes(masks)
            }
        elif head_type in ['deformable_detr', 'lw_detr', 'sparrow_seg', 'contourformer']:
            if head_type == 'contourformer':
                logits, coords = raw_outputs["pred_logits"][0], raw_outputs["pred_coords"][0]
                masks_torch = contours_to_masks(coords, original_size)
            else: # DETR-style
                logits, masks = raw_outputs["pred_logits"][0], raw_outputs["pred_masks"][0]
                masks_torch = F.interpolate(masks.unsqueeze(0), size=original_size, mode='bilinear', align_corners=False).squeeze(0)
                masks_torch = masks_torch.sigmoid()

            scores_softmax = logits.softmax(-1)[:, :-1]
            scores, labels = scores_softmax.max(-1)
            keep = scores > threshold

            processed_preds = {
                "scores": scores[keep].cpu().numpy(),
                "labels": labels[keep].cpu().numpy(),
                "masks": (masks_torch[keep] > 0.5).cpu().numpy(),
            }
            processed_preds["boxes"] = masks_to_boxes(processed_preds["masks"])
    except Exception as e:
        print(f"⚠️ Error processing predictions for head '{head_type}': {e}")

    return processed_preds

def draw_predictions(ax: plt.Axes, image: np.ndarray, preds: Dict[str, np.ndarray], colors: List, label_map: Dict[int, str]):
    """Draws masks, boxes, and labels on a matplotlib axis."""
    ax.imshow(image)
    ax.axis('off')

    masks = preds.get("masks", np.array([]))
    boxes = preds.get("boxes", np.array([]))
    labels = preds.get("labels", np.array([]))
    scores = preds.get("scores", np.array([]))

    # Draw masks first
    if len(masks) > 0:
        overlay = image.copy()
        alpha = 0.4
        for i, mask in enumerate(masks):
            # Use the label if available, otherwise cycle through colors
            color_idx = labels[i] if i < len(labels) else i
            color = colors[color_idx % len(colors)]
            bool_mask = mask > 0.5
            overlay[bool_mask] = overlay[bool_mask] * (1 - alpha) + np.array(color[:3]) * alpha
        ax.imshow(overlay)

    # Draw boxes and labels on top (only if labels are available)
    if len(labels) > 0:
        for i in range(len(boxes)):
            box = boxes[i]
            label_idx = labels[i]
            score = scores[i]
            color = colors[label_idx % len(colors)]

            x_min, y_min, x_max, y_max = box
            rect = patches.Rectangle(
                (x_min, y_min), x_max - x_min, y_max - y_min,
                linewidth=1.5, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)

            label_name = label_map.get(label_idx, f"Label {label_idx}")
            caption = f"{label_name}: {score:.2f}"
            ax.text(
                x_min, y_min - 5, caption,
                color='white', fontsize=8,
                bbox=dict(facecolor=color, alpha=0.6, pad=1, edgecolor='none')
            )

# --- Main Visualization Logic ---

def visualize_model(checkpoint_path: str, project_dir: str, num_images: int, backbone: str, threshold: float):
    """Loads a model and visualizes its predictions on validation images."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at: {checkpoint_path}")
    device = get_device()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    hparams = checkpoint['hyper_parameters']
    hparams['backbone_type'] = backbone
    head = build_head_from_hparams(hparams)
    model = SSLSegmentationLightning.load_from_checkpoint(
        checkpoint_path, map_location=device, head=head, strict=False
    )
    model.eval()

    datamodule = COCODataModule(project_dir=project_dir, batch_size=1, num_workers=0, image_size=hparams.get('image_size', 224))
    datamodule.setup()
    val_dataset = datamodule.val_ds

    label_map = {k: datamodule.coco_gt_val.loadCats(v)[0]['name'] for k, v in datamodule.label2cat.items()}

    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    indices = random.sample(range(len(val_dataset)), min(num_images, len(val_dataset)))
    fig, axes = plt.subplots(len(indices), 3, figsize=(18, 6 * len(indices)), squeeze=False)

    cmap = plt.colormaps.get('gist_rainbow')
    colors_list = cmap(np.linspace(0, 1, 80))
    random.shuffle(colors_list)

    inference_times = []

    for i, idx in enumerate(indices):
        image_as_tensor, target = val_dataset[idx]
        original_size = image_as_tensor.shape[1:]
        display_image = np.clip(image_as_tensor.permute(1, 2, 0).cpu().numpy(), 0, 1)
        image_tensor = image_as_tensor.unsqueeze(0).to(device)

        # --- Measure inference time ---
        torch.cuda.synchronize(device) if device.type == "cuda" else None
        start = time.perf_counter()
        with torch.no_grad():
            raw_outputs = model.student(image_tensor)
        torch.cuda.synchronize(device) if device.type == "cuda" else None
        elapsed = (time.perf_counter() - start) * 1000  # ms
        inference_times.append(elapsed)

        predictions = process_predictions(model.hparams, raw_outputs, original_size, threshold)

        # Plot Original Image
        axes[i, 0].imshow(display_image)
        axes[i, 0].set_title(f"Original Image (Index: {idx})")
        axes[i, 0].axis('off')

        # Plot Ground Truth
        gt_preds = {
            "masks": target.get('masks', torch.Tensor()).numpy(),
            "labels": target.get('labels', torch.Tensor()).numpy()
        }
        axes[i, 1].set_title(f"Ground Truth ({len(gt_preds['masks'])} masks)")
        draw_predictions(axes[i, 1], display_image, gt_preds, colors_list, {})

        # Plot Predictions
        axes[i, 2].set_title(f"Predictions ({len(predictions['boxes'])} objs @ {threshold} thr)\n{elapsed:.1f} ms")
        draw_predictions(axes[i, 2], display_image, predictions, colors_list, label_map)

    plt.tight_layout(pad=1.5)
    output_dir = os.path.join(project_dir, 'visualizations')
    os.makedirs(output_dir, exist_ok=True)
    head_name = model.hparams.get('head_type', 'unknown_head')
    ckpt_name = os.path.splitext(os.path.basename(checkpoint_path))[0]

    # --- Add avg inference time to filename ---
    avg_time = np.mean(inference_times) if inference_times else 0.0
    output_path = os.path.join(
        output_dir,
        f"{backbone}_{head_name}_predictions_{ckpt_name}_{avg_time:.1f}ms.png"
    )

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"✅ Visualization saved successfully to: {output_path}")
    print(f"📊 Inference times per image (ms): {['%.1f' % t for t in inference_times]}")
    print(f"📊 Average inference time: {avg_time:.1f} ms")
# --- Script Entrypoint ---

def main():
    """Parses command-line arguments and runs the visualization."""
    parser = argparse.ArgumentParser(description="Visualize predictions of a trained OmniSeg model.")
    parser.add_argument("checkpoint", type=str, help="Path to the model checkpoint (.ckpt) file.")
    parser.add_argument("--project_dir", type=str, default=PROJECT_DIR, help="Project directory where data is stored.")
    parser.add_argument("--num_images", type=int, default=3, help="Number of random images to visualize.")
    parser.add_argument("--backbone", type=str, default="dino", help="Backbone type to use for the model.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold for filtering predictions.")
    args = parser.parse_args()

    visualize_model(args.checkpoint, args.project_dir, args.num_images, args.backbone, args.threshold)

if __name__ == '__main__':
    main()
