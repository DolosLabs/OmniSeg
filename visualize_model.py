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

# --- New Imports for Validation ---
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import masks_to_boxes as torch_masks_to_boxes
from tqdm import tqdm

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

def process_predictions(model_hparams: Dict[str, Any], raw_outputs: Any, original_size: Tuple[int, int], threshold: float, device: torch.device, return_format: str = 'numpy') -> Any:
    """
    Processes raw model outputs into a standardized format.
    """
    head_type = model_hparams.get('head_type', 'unknown')
    
    try:
        outputs = raw_outputs[0] if isinstance(raw_outputs, (tuple, list)) else raw_outputs

        if head_type == 'maskrcnn':
            predictions = outputs[0]
            scores = predictions["scores"].detach()
            keep = scores > threshold
            final_scores = scores[keep]
            final_labels = predictions["labels"][keep]
            final_masks = (predictions["masks"][keep] > 0.5).squeeze(1)

        elif head_type in ['deformable_detr', 'lw_detr', 'sparrow_seg', 'contourformer']:
            if head_type == 'contourformer':
                logits, coords = outputs["pred_logits"][0], outputs["pred_coords"][0]
                masks_torch = contours_to_masks(coords, original_size)
            else:
                logits, masks = outputs["pred_logits"][0], outputs["pred_masks"][0]
                # --- KEY FIX 1: Prevent error on empty model output for certain images ---
                if logits.numel() == 0:
                    empty_result = {"scores": torch.tensor([], device=device), "labels": torch.tensor([], device=device, dtype=torch.long), "masks": torch.tensor([], device=device), "boxes": torch.tensor([], device=device)}
                    return [empty_result] if return_format == 'torch' else {k: v.cpu().numpy() for k, v in empty_result.items()}
                
                masks_torch = F.interpolate(masks.unsqueeze(0), size=original_size, mode='bilinear', align_corners=False).squeeze(0)
                masks_torch = masks_torch.sigmoid()
            
            scores_softmax = logits.softmax(-1)[:, :-1]
            scores, labels = scores_softmax.max(-1)
            keep = scores > threshold
            final_scores = scores[keep]
            final_labels = labels[keep]
            final_masks = masks_torch[keep] > 0.5
        
        else:
             raise NotImplementedError(f"Prediction processing for '{head_type}' is not implemented.")

        if final_masks.numel() == 0:
            final_boxes = torch.empty((0, 4), device=final_masks.device)
        else:
            final_boxes = torch_masks_to_boxes(final_masks)
        
        if return_format == 'torch':
            return [{"scores": final_scores, "labels": final_labels, "masks": final_masks.bool(), "boxes": final_boxes}]
        else:
            return {"scores": final_scores.cpu().numpy(), "labels": final_labels.cpu().numpy(), "masks": final_masks.cpu().numpy(), "boxes": final_boxes.cpu().numpy()}

    except Exception as e:
        print(f"⚠️ Error processing predictions for head '{head_type}': {e}")
        # --- KEY FIX 2: Ensure empty tensors in the error case are on the correct device ---
        empty_result = {"scores": torch.tensor([], device=device), "labels": torch.tensor([], device=device, dtype=torch.long), "masks": torch.tensor([], device=device), "boxes": torch.tensor([], device=device)}
        if return_format == 'torch':
            return [empty_result]
        else:
            return {k: v.cpu().numpy() for k, v in empty_result.items()}


def draw_predictions(ax: plt.Axes, image: np.ndarray, preds: Dict[str, np.ndarray], colors: List, label_map: Dict[int, str]):
    """Draws masks, boxes, and labels on a matplotlib axis."""
    ax.imshow(image)
    ax.axis('off')

    masks = preds.get("masks", np.array([]))
    boxes = preds.get("boxes", np.array([]))
    labels = preds.get("labels", np.array([]))
    scores = preds.get("scores", np.array([]))

    if len(masks) > 0:
        overlay = image.copy()
        alpha = 0.4
        for i, mask in enumerate(masks):
            color_idx = labels[i] if i < len(labels) else i
            color = colors[color_idx % len(colors)]
            overlay[mask > 0.5] = overlay[mask > 0.5] * (1 - alpha) + np.array(color[:3]) * alpha
        ax.imshow(overlay)

    if len(labels) > 0 and len(boxes) > 0:
        for i in range(len(boxes)):
            box = boxes[i]
            label_idx = labels[i]
            color = colors[label_idx % len(colors)]
            x_min, y_min, x_max, y_max = box
            rect = patches.Rectangle(
                (x_min, y_min), x_max - x_min, y_max - y_min,
                linewidth=1.5, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            
            label_name = label_map.get(label_idx, f"Label {label_idx}")
            if i < len(scores):
                score = scores[i]
                caption = f"{label_name}: {score:.2f}"
            else:
                caption = label_name

            ax.text(
                x_min, y_min - 5, caption,
                color='white', fontsize=8,
                bbox=dict(facecolor=color, alpha=0.6, pad=1, edgecolor='none')
            )

def format_targets_for_metric(targets: List[Dict[str, torch.Tensor]], device: torch.device) -> List[Dict[str, torch.Tensor]]:
    """Formats ground truth targets from the dataloader for torchmetrics."""
    formatted_targets = []
    for t in targets:
        masks = t["masks"]
        if "boxes" not in t:
            boxes = torch_masks_to_boxes(masks.bool())
        else:
            boxes = t["boxes"]
        
        formatted_targets.append({
            "boxes": boxes.to(device),
            "labels": t["labels"].to(device),
            "masks": masks.to(device).bool()
        })
    return formatted_targets

def run_validation(model, dataloader, device, threshold) -> Dict[str, Any]:
    """Runs validation across the entire dataset and computes mAP."""
    print("\n🚀 Starting validation run...")
    metric = MeanAveragePrecision(box_format='xyxy', iou_type='segm').to(device)
    
    for images, targets in tqdm(dataloader, desc="Validating"):
        images = images.to(device)
        original_size = images.shape[2:]

        with torch.no_grad():
            raw_outputs = model.student(images)
        
        preds = process_predictions(model.hparams, raw_outputs, original_size, threshold, device, return_format='torch')
        targets_formatted = format_targets_for_metric(targets, device)
        
        metric.update(preds, targets_formatted)
        
    results = metric.compute()
    return results

def visualize_model(checkpoint_path: str, project_dir: str, num_images: int, backbone: str, threshold: float, run_val: bool):
    """Loads a model, optionally runs validation, and visualizes predictions."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at: {checkpoint_path}")
    
    device = get_device()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    hparams = checkpoint['hyper_parameters']
    hparams['backbone_type'] = backbone
    head = build_head_from_hparams(hparams)
    
    model = SSLSegmentationLightning.load_from_checkpoint(
        checkpoint_path, map_location=device, head=head, strict=False
    ).to(device)
    model.eval()

    datamodule = COCODataModule(project_dir=project_dir, batch_size=1, num_workers=2, image_size=hparams.get('image_size', 224))
    datamodule.setup()
    val_dataset = datamodule.val_ds
    label_map = {k: datamodule.coco_gt_val.loadCats(v)[0]['name'] for k, v in datamodule.label2cat.items()}

    val_results = None
    if run_val:
        val_loader = datamodule.val_dataloader()
        val_results = run_validation(model, val_loader, device, threshold)
        
        print("\n--- ✅ Validation Results (Scalar Metrics) ---")
        for k, v in val_results.items():
            if v.numel() == 1:
                print(f"{k}: {v.item():.4f}")
        print("--------------------------------------------\n")

    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    indices = random.sample(range(len(val_dataset)), min(num_images, len(val_dataset)))
    
    fig, axes = plt.subplots(len(indices), 3, figsize=(18, 6 * len(indices)), squeeze=False)
    
    head_name = model.hparams.get('head_type', 'unknown_head')
    if val_results:
        map = val_results.get('map', torch.tensor(-1.0)).item()
        fig.suptitle(f"Model: {head_name} | Backbone: {backbone}\nValidation mAP@50 (segm): {map:.3f}", fontsize=16)

    cmap = plt.colormaps.get('gist_rainbow')
    colors_list = cmap(np.linspace(0, 1, 80))
    random.shuffle(colors_list)
    inference_times = []

    for i, idx in enumerate(tqdm(indices, desc="Generating visualizations")):
        image_as_tensor, target = val_dataset[idx] 

        original_size = image_as_tensor.shape[1:]
        display_image = np.clip(image_as_tensor.permute(1, 2, 0).cpu().numpy(), 0, 1)
        image_tensor = image_as_tensor.unsqueeze(0).to(device)

        torch.cuda.synchronize(device) if device.type == "cuda" else None
        start = time.perf_counter()
        with torch.no_grad():
            raw_outputs = model.student(image_tensor)
        torch.cuda.synchronize(device) if device.type == "cuda" else None
        elapsed = (time.perf_counter() - start) * 1000
        inference_times.append(elapsed)

        predictions = process_predictions(model.hparams, raw_outputs, original_size, threshold, device, return_format='numpy')

        axes[i, 0].imshow(display_image)
        axes[i, 0].set_title(f"Original Image (Index: {idx})")
        axes[i, 0].axis('off')

        gt_preds = {"masks": target.get('masks', torch.Tensor()).numpy(), "labels": target.get('labels', torch.Tensor()).numpy()}
        if gt_preds["masks"].shape[0] > 0:
            gt_preds["boxes"] = torch_masks_to_boxes(torch.from_numpy(gt_preds["masks"])).numpy()
        else:
            gt_preds["boxes"] = np.array([])
        axes[i, 1].set_title(f"Ground Truth ({len(gt_preds['masks'])} masks)")
        draw_predictions(axes[i, 1], display_image, gt_preds, colors_list, {})

        axes[i, 2].set_title(f"Predictions ({len(predictions.get('boxes', []))} objs @ {threshold} thr)\n{elapsed:.1f} ms")
        draw_predictions(axes[i, 2], display_image, predictions, colors_list, label_map)

    plt.tight_layout(rect=[0, 0, 1, 0.96] if val_results else None)
    output_dir = os.path.join(project_dir, 'visualizations')
    os.makedirs(output_dir, exist_ok=True)
    ckpt_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    avg_time = np.mean(inference_times) if inference_times else 0.0
    
    score_str = f"_map_{map:.3f}" if (val_results and 'map' in val_results) else ""
    output_path = os.path.join(output_dir, f"{backbone}_{head_name}_{ckpt_name}_{avg_time:.1f}ms{score_str}.png")

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"✅ Visualization saved successfully to: {output_path}")
    print(f"📊 Average inference time: {avg_time:.1f} ms")

def main():
    parser = argparse.ArgumentParser(description="Visualize predictions of a trained OmniSeg model.")
    parser.add_argument("checkpoint", type=str, help="Path to the model checkpoint (.ckpt) file.")
    parser.add_argument("--project_dir", type=str, default=PROJECT_DIR, help="Project directory where data is stored.")
    parser.add_argument("--num_images", type=int, default=3, help="Number of random images to visualize.")
    parser.add_argument("--backbone", type=str, default="dino", help="Backbone type to use for the model.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold for filtering predictions.")
    parser.add_argument("--validate", action="store_true", help="Run full validation and add mAP score to the plot.")
    args = parser.parse_args()

    visualize_model(args.checkpoint, args.project_dir, args.num_images, args.backbone, args.threshold, args.validate)

if __name__ == '__main__':
    main()
