# visualize_model.py

import argparse
import os
import random
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import Dict, Any, Tuple

# Import from our modular structure
from omniseg.config import PROJECT_DIR
from omniseg.data import COCODataModule, get_transforms
from omniseg.training import SSLSegmentationLightning
from omniseg.models.heads.contourformer import contours_to_masks

# --- Import all possible head modules ---
from omniseg.models.heads.deformable_detr import DETRSegmentationHead
from omniseg.models.heads.contourformer import ContourFormerHead
from omniseg.models.heads.maskrcnn import MaskRCNNHead
from omniseg.models.heads.lw_detr import LWDETRHead
from omniseg.models.heads.sparrow_seg import SparrowSegHead # MODIFIED: Import SparrowSegHead

# --- Constants ---
CONFIDENCE_THRESHOLD = 0.5
RANDOM_SEED = 42

# --- HELPER FUNCTION ---
def build_head_from_hparams(hparams: Dict[str, Any]) -> torch.nn.Module:
    """
    Instantiates the correct model head based on hyperparameters.
    """
    head_type = hparams.get('head_type')
    if not head_type:
        raise ValueError("Could not find 'head_type' in model hyperparameters.")

    num_classes = hparams.get('num_classes', 80)
    
    print(f"Building model head of type: '{head_type}'")

    if head_type == 'deformable_detr':
        # NOTE: Ensure these parameters match those used during training!
        return DETRSegmentationHead(
            num_classes=num_classes,
            backbone_type=hparams.get('backbone_type', 'dino'),
            image_size=hparams.get('image_size', 128),
            hidden_dim=hparams.get('d_model', 256),
            num_queries=hparams.get('num_queries', 52),
            dec_layers=hparams.get('num_decoder_layers', 4),
            num_groups=hparams.get('num_groups', 13),
            nheads=hparams.get('n_heads', 8),
            dim_feedforward=hparams.get('d_ffn', 2048),
            enc_layers=6, # Default/legacy value
            pre_norm=False, # Default/legacy value
            enforce_input_project=False, # Default/legacy value
            mask_classification=True # Default/legacy value
        )
    elif head_type == 'lw_detr':
        # NOTE: Ensure these parameters match those used during training!
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
        # NOTE: Adjust these parameters as needed
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
    # MODIFIED: Correctly instantiate SparrowSegHead
    elif head_type == 'sparrow_seg':
        # NOTE: Ensure these parameters match those used during training!
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
        # NOTE: Adjust these parameters as needed
        return MaskRCNNHead(
            backbone_type=hparams.get('backbone_type', 'dino'),
            num_classes=num_classes
        )
    else:
        raise NotImplementedError(f"Head type '{head_type}' is not supported by this script.")


# --- Helper Functions ---

def get_device() -> torch.device:
    """Determines the available compute device (GPU or CPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

def process_predictions(model_hparams: Dict[str, Any], raw_outputs: Any, original_size: Tuple[int, int]) -> Dict[str, np.ndarray]:
    """
    Processes raw model outputs into a standardized dictionary of numpy arrays.
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
        if isinstance(raw_outputs, (tuple, list)) and len(raw_outputs) == 2:
            logits, contours = raw_outputs
        elif isinstance(raw_outputs, dict):
            logits = raw_outputs.get("pred_logits")
            contours = raw_outputs.get("pred_contours") or raw_outputs.get("pred_coords")
        else:
            logits, contours = None, None

        if logits is None or contours is None:
            print("⚠️ Unexpected ContourFormer raw_outputs format")
            return {"masks": np.array([]), "labels": np.array([]), "scores": np.array([])}

        try:
            if logits.dim() == 3:
                scores_softmax = logits.softmax(-1)[0, :, :-1]
            else:
                scores_softmax = logits.softmax(-1)[:, :-1]
                
            scores, labels = scores_softmax.max(-1)
            keep_mask = scores > CONFIDENCE_THRESHOLD

            if contours.dim() == 4:
                contours_np = contours[0][keep_mask].detach().cpu()
            else:
                contours_np = contours[keep_mask].detach().cpu()

            masks = contours_to_masks(contours_np, original_size)
            
            return {
                "masks": masks.numpy() if isinstance(masks, torch.Tensor) else masks,
                "labels": labels[keep_mask].detach().cpu().numpy(),
                "scores": scores[keep_mask].detach().cpu().numpy()
            }
        except Exception as e:
            print(f"Error processing ContourFormer outputs: {e}")
            return {"masks": np.array([]), "labels": np.array([]), "scores": np.array([])}

    # MODIFIED: Add 'sparrow_seg' to the list of DETR-like models
    elif head_type in ['deformable_detr', 'lw_detr', 'sparrow_seg']:
        try:
            if isinstance(raw_outputs, dict):
                logits = raw_outputs.get("pred_logits")
                masks = raw_outputs.get("pred_masks")
                
                if logits is not None and masks is not None:
                    if logits.dim() == 3: logits = logits[0]
                    if masks.dim() == 4: masks = masks[0]
                    
                    # For DETR-like models, last class is "no object"
                    scores_softmax = logits.softmax(-1)[:, :-1]
                    scores, labels = scores_softmax.max(-1)
                    keep_mask = scores > CONFIDENCE_THRESHOLD
                    
                    selected_masks = masks[keep_mask]
                    if len(selected_masks) > 0:
                        resized_masks = F.interpolate(
                            selected_masks.unsqueeze(0), 
                            size=original_size, 
                            mode='bilinear', 
                            align_corners=False
                        ).squeeze(0)
                        binary_masks = (resized_masks > 0.5).cpu().numpy()
                    else:
                        binary_masks = np.array([]).reshape(0, *original_size)
                    
                    return {
                        "masks": binary_masks,
                        "labels": labels[keep_mask].detach().cpu().numpy(),
                        "scores": scores[keep_mask].detach().cpu().numpy()
                    }
            print(f"⚠️ Unexpected {head_type} raw_outputs format")
            return {"masks": np.array([]), "labels": np.array([]), "scores": np.array([])}
        except Exception as e:
            print(f"Error processing {head_type} outputs: {e}")
            return {"masks": np.array([]), "labels": np.array([]), "scores": np.array([])}

    else:
        print(f"Warning: Unsupported head type '{head_type}'. Returning empty predictions.")
        return {"masks": np.array([]), "labels": np.array([]), "scores": np.array([])}

def create_mask_overlay(image: Image.Image, masks: np.ndarray, colors: np.ndarray) -> np.ndarray:
    """Creates a colored overlay from masks on top of an image."""
    if masks.ndim == 2:
        masks = masks[np.newaxis, ...]
        
    combined_mask_color = np.zeros((*image.size[::-1], 3), dtype=np.float32)
    
    # MODIFIED: Use a single, distinct color (white) and adjust transparency
    # for better visibility. Original colors_list can still be passed but
    # we'll override for the overlay effect.
    overlay_color = np.array([1.0, 1.0, 1.0]) # White color
    alpha = 0.6 # Increased transparency for better visibility

    for i, mask in enumerate(masks):
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()

        resized_mask = (mask > 0.5)
        
        for c in range(3):
            combined_mask_color[resized_mask, c] = overlay_color[c]
            
    image_arr = np.array(image) / 255.0
    overlay = image_arr.copy()
    
    # Apply the white overlay only where masks exist
    overlay_area = np.any(combined_mask_color > 0, axis=-1)
    overlay[overlay_area] = (1 - alpha) * image_arr[overlay_area] + alpha * combined_mask_color[overlay_area]
    
    return np.clip(overlay, 0, 1)

# --- Main Visualization Logic ---

def visualize_model(checkpoint_path: str, project_dir: str, num_images: int, backbone: str):
    """
    Loads a model and visualizes its predictions on validation images.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at: {checkpoint_path}")

    device = get_device()
    
    # --- MODIFIED: New model loading logic ---
    print("Loading checkpoint to extract hyperparameters...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    hparams = checkpoint['hyper_parameters']
    hparams['backbone_type'] = backbone
    
    # 1. Build the model head from hyperparameters
    head = build_head_from_hparams(hparams)
    
    # 2. Load the LightningModule, injecting the head
    print("Loading Lightning module with injected head...")
    model = SSLSegmentationLightning.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
        head=head,
        strict=False # Use strict=False if checkpoint has extra keys not in model
    )
    
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
    colors_list = cmap(np.linspace(0, 1, 50))

    for i, idx in enumerate(indices):
        image_pil, target = val_dataset[idx]
        original_size = image_pil.size[::-1] # (height, width)
        
        image_tensor = val_transform(image_pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            model.to(device)
            # Forward pass through the student model (backbone + head)
            raw_outputs = model.student(image_tensor)
            
            try:
                predictions = process_predictions(model.hparams, raw_outputs, original_size)
            except Exception as e:
                print(f"Warning: Error processing predictions for image {idx}: {e}")
                predictions = {"masks": np.array([]), "labels": np.array([]), "scores": np.array([])}

        axes[i, 0].imshow(image_pil)
        axes[i, 0].set_title(f"Original Image (Index: {idx})")

        gt_masks = target.get('masks')
        if gt_masks is not None and len(gt_masks) > 0:
            gt_overlay = create_mask_overlay(image_pil, gt_masks.numpy(), colors_list) # Keep original colors for GT
            axes[i, 1].imshow(gt_overlay)
        else:
            axes[i, 1].imshow(image_pil)
        axes[i, 1].set_title(f"Ground Truth ({0 if gt_masks is None else len(gt_masks)} masks)")

        pred_masks = predictions['masks']
        if pred_masks.shape[0] > 0:
            pred_overlay = create_mask_overlay(image_pil, pred_masks, colors_list) # Now this will be white
            axes[i, 2].imshow(pred_overlay)
        else:
            axes[i, 2].imshow(image_pil)
        axes[i, 2].set_title(f"Predictions ({len(pred_masks)} masks)")
        
        for ax in axes[i]:
            ax.axis('off')

    plt.tight_layout()
    output_dir = os.path.join(project_dir, 'visualizations')
    os.makedirs(output_dir, exist_ok=True)
    
    head_name = model.hparams.get('head_type', 'unknown_head')
    ckpt_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    image_prefix = f"{backbone}_{head_name}"
    
    output_path = os.path.join(output_dir, f'{image_prefix}_predictions_{ckpt_name}.png')
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
    parser.add_argument("--backbone", type=str, default="dino", help="Number of random images to visualize.")
    args = parser.parse_args()
    
    visualize_model(args.checkpoint, args.project_dir, args.num_images, args.backbone)

if __name__ == '__main__':
    main()
