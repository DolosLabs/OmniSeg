"""Training utilities and PyTorch Lightning training module."""

import copy
import math
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union

import pytorch_lightning as pl
from torchvision.ops import box_convert
from torchmetrics.detection import MeanAveragePrecision  # <-- ADDED: Import for mAP calculation

from ..data import get_transforms
from ..models.heads import get_head
from ..models.heads.contourformer import masks_to_contours, contours_to_masks


def masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
    """Convert masks to bounding boxes."""
    if masks.numel() == 0:
        return torch.zeros((0, 4), dtype=torch.float32)
    n = masks.shape[0]
    boxes = torch.zeros(n, 4, dtype=masks.dtype, device=masks.device)
    for i, mask in enumerate(masks):
        ys, xs = torch.where(mask)
        if len(xs) == 0:
            boxes[i] = torch.tensor([0, 0, 1, 1], dtype=masks.dtype, device=masks.device)
        else:
            boxes[i] = torch.tensor([xs.min(), ys.min(), xs.max(), ys.max()], dtype=masks.dtype, device=masks.device)
    return boxes


class SSLSegmentationLightning(pl.LightningModule):
    """Semi-supervised segmentation training with PyTorch Lightning."""

    def __init__(self, num_classes: int = 80, lr: float = 1e-4, ema_decay: float = 0.999,
                 backbone_type: str = 'dino', head_type: str = 'maskrcnn', image_size: int = 224,
                 warmup_steps: int = 500, unsup_rampup_steps: int = 5000):
        super().__init__()
        self.save_hyperparameters()

        # Create student model using the factory pattern
        head_kwargs = {
            'backbone_type': backbone_type,
            'image_size': image_size
        }

        # Add head-specific parameters
        if head_type == 'contourformer':
            head_kwargs['hidden_dim'] = 32  # Special case for ContourFormer

        self.student = get_head(head_type, num_classes, **head_kwargs)

        # Create teacher model as a copy of student
        self.teacher = copy.deepcopy(self.student)
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

        self.train_aug = get_transforms(augment=True, image_size=self.hparams.image_size)
        self.val_aug = get_transforms(augment=False, image_size=self.hparams.image_size)
        
        # --- MODIFIED: Initialize torchmetrics for mAP calculation ---
        # Using iou_type='segm' is crucial for calculating mask mAP.
        self.val_map = MeanAveragePrecision(box_format='xyxy', iou_type='segm')
        # --- REMOVED old validation lists ---

    def configure_optimizers(self):
        return torch.optim.AdamW([p for p in self.student.parameters() if p.requires_grad], lr=self.hparams.lr)

    @torch.no_grad()
    def _update_teacher(self):
        for t_param, s_param in zip(self.teacher.parameters(), self.student.parameters()):
            t_param.data.mul_(self.hparams.ema_decay).add_(s_param.data, alpha=1 - self.hparams.ema_decay)

    def on_before_optimizer_step(self, optimizer):
        self._update_teacher()

    def training_step(self, batch, batch_idx):
        if self.hparams.head_type == 'maskrcnn':
            return self._training_step_mrcnn(batch, batch_idx)
        elif self.hparams.head_type == 'contourformer':
            return self._training_step_cf(batch, batch_idx)
        elif self.hparams.head_type == 'deformable_detr':
            return self._training_step_detr(batch, batch_idx)

    # --- ADDED/REFACTORED: Unified validation_step to calculate mAP ---
    def validation_step(self, batch: Tuple[torch.Tensor, List[Dict]], batch_idx: int):
        """
        Perform a validation step, calculate predictions, and update the mAP metric.
        """
        images, targets = batch
        pixel_values = torch.stack([self.val_aug(img, None)[0] for img in images]).to(device=self.device, dtype=self.dtype)
        
        # Run inference
        self.student.eval()
        with torch.no_grad():
            outputs = self.student(pixel_values)
        
        # Format ground truth targets into the format required by torchmetrics
        targets_formatted = []
        for target in targets:
            masks = target['masks'].to(self.device)
            labels = target['labels'].to(self.device)
            boxes = masks_to_boxes(masks)
            # --- FIXED: Convert ground truth masks to uint8 ---
            targets_formatted.append({'boxes': boxes, 'labels': labels, 'masks': masks.to(torch.uint8)})
        
        # Get original image sizes for post-processing
        original_sizes = torch.tensor([img.size[::-1] for img in images], device=self.device)

        # Format predictions based on the specific model head
        preds = []
        
        if self.hparams.head_type == 'maskrcnn':
            # --- FIXED: Post-process Mask R-CNN output ---
            # Threshold the float masks and convert to uint8
            for pred in outputs:
                pred['masks'] = (pred['masks'] > 0.5).squeeze(1).to(torch.uint8)
                preds.append(pred)
        
        elif self.hparams.head_type == 'deformable_detr':
            # Assumes the student model has a post-processing method for segmentation
            processed_preds = self.student.post_process_segmentation(outputs, original_sizes)
            # --- FIXED: Ensure DETR masks are also uint8 ---
            for pred in processed_preds:
                # DETR mask outputs can also be floats that need thresholding
                pred['masks'] = (pred['masks'] > 0.5).to(torch.uint8)
                preds.append(pred)

        elif self.hparams.head_type == 'contourformer':
            # Post-process ContourFormer output from contours to masks and boxes
            for i, (h, w) in enumerate(original_sizes):
                pred_scores = outputs[i]['scores']
                pred_labels = outputs[i]['labels']
                pred_coords = outputs[i]['coords'] # Assumes model output includes predicted coordinates
                pred_masks = contours_to_masks(pred_coords, (h.item(), w.item()))
                pred_boxes = masks_to_boxes(pred_masks)
                # --- FIXED: Convert predicted masks to uint8 ---
                preds.append({
                    'scores': pred_scores, 
                    'labels': pred_labels, 
                    'masks': pred_masks.to(torch.uint8), 
                    'boxes': pred_boxes
                })
        
        # Update the metric with predictions and targets for this batch
        self.val_map.update(preds, targets_formatted)

    # --- REMOVED old, empty _validation_step_* helper methods ---

    # --- MODIFIED: on_validation_epoch_end to compute and log the final mAP ---
    def on_validation_epoch_end(self):
            """
            Called at the end of the validation epoch.
            Computes the final mAP score, logs it, and resets the metric.
            """
            # Compute the final mAP score from all validation steps
            map_dict = self.val_map.compute()
            
            # --- FIXED: Move computed metrics to the correct device before logging ---
            # This is crucial for multi-GPU training (sync_dist=True)
            val_mAP = map_dict['map'].to(self.device)
            val_mAP_50 = map_dict['map_50'].to(self.device)
            val_mAP_75 = map_dict['map_75'].to(self.device)
            val_mAP_large = map_dict['map_large'].to(self.device)

            # Log the metrics. 'val_mAP' is the key that ModelCheckpoint is monitoring.
            self.log_dict({
                'val_mAP': val_mAP,
                'val_mAP_50': val_mAP_50,
                'val_mAP_75': val_mAP_75,
                'val_mAP_large': val_mAP_large,
            }, on_epoch=True, prog_bar=True, sync_dist=True)
            
            # Reset the metric for the next validation epoch
            self.val_map.reset()

    def _training_step_detr(self, batch, batch_idx):
        """Training step for Deformable DETR."""
        # 1. Labeled data (supervised)
        images_l, targets_l = batch["labeled"]
        pixel_values_l = torch.stack([self.train_aug(img, None)[0] for img in images_l]).to(device=self.device, dtype=self.dtype)
    
        targets_l_detr = []
        h, w = self.hparams.image_size, self.hparams.image_size
        for target in targets_l:
            boxes_xyxy = masks_to_boxes(target["masks"])
            boxes_cxcywh = box_convert(boxes_xyxy, in_fmt="xyxy", out_fmt="cxcywh")
            boxes_cxcywh[:, 0::2] /= w
            boxes_cxcywh[:, 1::2] /= h
            mask_target_size = (h // 2, w // 2)
            resized_masks = F.interpolate(target["masks"].unsqueeze(1), size=mask_target_size, mode="bilinear", align_corners=False).squeeze(1)
            targets_l_detr.append({
                "labels": target["labels"].to(self.device),
                "boxes": boxes_cxcywh.to(self.device),
                "masks": resized_masks.to(self.device)
            })
    
        _, losses_sup = self.student(pixel_values=pixel_values_l, targets=targets_l_detr)
        
        # For simplicity, we focus on supervised loss for DETR
        supervised_loss = sum(losses_sup.values())
        total_loss = supervised_loss
        
        self.log_dict({
            "train_loss": total_loss,
            "sup_loss": supervised_loss,
            **{f"train_{k}": v for k, v in losses_sup.items()}
        }, on_step=True, on_epoch=True, prog_bar=True)
        
        return total_loss

    def _training_step_cf(self, batch, batch_idx):
        """Training step for ContourFormer."""
        # 1. Labeled data (supervised)
        images_l, targets_l = batch["labeled"]
        pixel_values_l = torch.stack([self.train_aug(img, None)[0] for img in images_l]).to(device=self.device, dtype=self.dtype)
    
        targets_l_cf = []
        for target in targets_l:
            try:
                contours, valid_indices = masks_to_contours(target["masks"])
                if len(valid_indices) > 0:
                    valid_labels = target["labels"][valid_indices]
                    targets_l_cf.append({
                        "labels": valid_labels.to(self.device),
                        "coords": contours.to(self.device)
                    })
                else:
                    # Create empty target
                    targets_l_cf.append({
                        "labels": torch.empty(0, dtype=torch.long, device=self.device),
                        "coords": torch.empty(0, 50, 2, device=self.device)
                    })
            except Exception as e:
                print(f"Warning: Error processing ContourFormer target: {e}")
                targets_l_cf.append({
                    "labels": torch.empty(0, dtype=torch.long, device=self.device),
                    "coords": torch.empty(0, 50, 2, device=self.device)
                })
    
        _, losses_sup = self.student(pixel_values=pixel_values_l, targets=targets_l_cf)
        
        supervised_loss = sum(losses_sup.values())
        total_loss = supervised_loss
        
        self.log_dict({
            "train_loss": total_loss,
            "sup_loss": supervised_loss,
            **{f"train_{k}": v for k, v in losses_sup.items()}
        }, on_step=True, on_epoch=True, prog_bar=True)
        
        return total_loss

    def _training_step_mrcnn(self, batch, batch_idx):
        """Training step for Mask R-CNN."""
        # 1. Labeled data (supervised)
        images_l, targets_l = batch["labeled"]
        pixel_values_l = torch.stack([self.train_aug(img, None)[0] for img in images_l]).to(device=self.device, dtype=self.dtype)
    
        targets_l_mrcnn = []
        for target in targets_l:
            boxes_xyxy = masks_to_boxes(target["masks"])
            targets_l_mrcnn.append({
                "boxes": boxes_xyxy.to(self.device),
                "labels": target["labels"].to(self.device),
                "masks": target["masks"].to(self.device)
            })
    
        losses_sup = self.student(pixel_values=pixel_values_l, targets=targets_l_mrcnn)
        
        supervised_loss = sum(losses_sup.values())
        total_loss = supervised_loss
        
        self.log_dict({
            "train_loss": total_loss,
            "sup_loss": supervised_loss,
            **{f"train_{k}": v for k, v in losses_sup.items()}
        }, on_step=True, on_epoch=True, prog_bar=True)
        
        return total_loss


__all__ = ['SSLSegmentationLightning', 'masks_to_boxes']
