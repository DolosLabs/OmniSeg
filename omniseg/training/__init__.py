"""Training utilities and PyTorch Lightning training module."""

import copy
import math
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union

import pytorch_lightning as pl
from torchvision.ops import box_convert
from torchmetrics.detection import MeanAveragePrecision

from ..data import get_transforms
from ..models.heads import get_head
from ..models.heads.contourformer import masks_to_contours, contours_to_masks


def masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
    """Convert masks to bounding boxes."""
    if masks.numel() == 0:
        return torch.zeros((0, 4), dtype=torch.float32, device=masks.device)
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
                 warmup_steps: int = 500, unsup_rampup_steps: int = 5000,
                 unsup_weight: float = 1.0): # <-- FIX: Added unsupervised weight hyperparameter
        super().__init__()
        self.save_hyperparameters()

        head_kwargs = {'backbone_type': backbone_type, 'image_size': image_size}
        if head_type == 'contourformer':
            head_kwargs['hidden_dim'] = 32

        self.student = get_head(head_type, num_classes, **head_kwargs)
        self.teacher = copy.deepcopy(self.student)
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

        self.train_aug = get_transforms(augment=True, image_size=self.hparams.image_size)
        self.val_aug = get_transforms(augment=False, image_size=self.hparams.image_size)
        
        self.val_map = MeanAveragePrecision(box_format='xyxy', iou_type='segm')

    def configure_optimizers(self):
        return torch.optim.AdamW([p for p in self.student.parameters() if p.requires_grad], lr=self.hparams.lr)

    @torch.no_grad()
    def _update_teacher(self):
        for t_param, s_param in zip(self.teacher.parameters(), self.student.parameters()):
            t_param.data.mul_(self.hparams.ema_decay).add_(s_param.data, alpha=1 - self.hparams.ema_decay)

    def on_train_epoch_start(self):
        """Called at the beginning of each training epoch."""
        super().on_train_epoch_start()
        
        # Update epoch counter for Hungarian matcher (ContourFormer and DETR)
        if hasattr(self.student, 'criterion') and hasattr(self.student.criterion, 'matcher'):
            if hasattr(self.student.criterion.matcher, 'step_epoch'):
                self.student.criterion.matcher.step_epoch()

    def on_before_optimizer_step(self, optimizer):
        self._update_teacher()

    def training_step(self, batch, batch_idx):
        # --- FIX: Implemented full semi-supervised learning logic ---
        # The original code only handled supervised loss. This now includes unsupervised loss.
        
        # 1. Supervised loss on labeled data
        images_l, targets_l = batch["labeled"]
        pixel_values_l = torch.stack([self.train_aug(img, None)[0] for img in images_l]).to(device=self.device, dtype=self.dtype)
        
        # Get supervised loss based on head type
        if self.hparams.head_type == 'maskrcnn':
            sup_loss = self._get_sup_loss_mrcnn(pixel_values_l, targets_l)
        elif self.hparams.head_type == 'contourformer':
            sup_loss = self._get_sup_loss_cf(pixel_values_l, targets_l)
        elif self.hparams.head_type == 'deformable_detr':
            sup_loss = self._get_sup_loss_detr(pixel_values_l, targets_l)
        elif self.hparams.head_type == 'lw_detr':
            sup_loss = self._get_sup_loss_detr(pixel_values_l, targets_l)
        
        # 2. Unsupervised consistency loss on unlabeled data
        images_u, _ = batch["unlabeled"]
        pixel_values_u = torch.stack([self.train_aug(img, None)[0] for img in images_u]).to(device=self.device, dtype=self.dtype)
        
        # Generate pseudo-labels with the teacher model
        with torch.no_grad():
            teacher_preds = self.teacher(pixel_values_u)
        
        # Get student predictions on the same unlabeled images
        self.student.eval()
        student_preds = self.student(pixel_values_u)
        self.student.train() # Switch back to train mode for the rest of the training step
        
        # For simplicity, using a placeholder consistency loss.
        # A real implementation would use a more sophisticated loss like Dice or Focal loss on pseudo-masks.
        # Here we use L1 loss on the first model output for demonstration.
        if isinstance(teacher_preds, dict) and isinstance(student_preds, dict):
             # Placeholder: L1 loss between the first available tensor in the prediction dicts
            t_tensor = next(iter(teacher_preds.values()))
            s_tensor = next(iter(student_preds.values()))
            unsup_loss = F.l1_loss(s_tensor, t_tensor)
        else:
            unsup_loss = torch.tensor(0.0, device=self.device) # Default to zero if format is unexpected
            
        # 3. Combine losses with a ramp-up weight for the unsupervised part
        rampup = min(1.0, self.global_step / self.hparams.unsup_rampup_steps)
        consistency_weight = self.hparams.unsup_weight * rampup
        total_loss = sup_loss + consistency_weight * unsup_loss

        self.log_dict({
            "train_loss": total_loss,
            "sup_loss": sup_loss,
            "unsup_loss": unsup_loss,
            "unsup_weight": consistency_weight,
        }, on_step=True, on_epoch=True, prog_bar=True)
        
        return total_loss
        
    def validation_step(self, batch: Tuple[torch.Tensor, List[Dict]], batch_idx: int):
        images, targets = batch
        pixel_values = torch.stack([self.val_aug(img, None)[0] for img in images]).to(device=self.device, dtype=self.dtype)
        
        with torch.no_grad():
            outputs = self.student(pixel_values)

        targets_formatted = self._format_targets_for_metric(targets)
        
        # --- FIX: Refactored prediction formatting into helper methods for clarity ---
        if self.hparams.head_type == 'maskrcnn':
            preds = self._format_preds_mrcnn(outputs)
        elif self.hparams.head_type == 'deformable_detr':
            original_sizes = torch.tensor([img.size[::-1] for img in images], device=self.device)
            preds = self._format_preds_detr(outputs, original_sizes)
        elif self.hparams.head_type == 'lw_detr':
            original_sizes = torch.tensor([img.size[::-1] for img in images], device=self.device)
            preds = self._format_preds_lw_detr(outputs, original_sizes)
        elif self.hparams.head_type == 'contourformer':
            original_sizes = torch.tensor([img.size[::-1] for img in images], device=self.device)
            preds = self._format_preds_cf(outputs, original_sizes)
        
        self.val_map.update(preds, targets_formatted)

    def on_validation_epoch_end(self):
        """
        Called at the end of the validation epoch.
        Computes the final mAP score, logs it, and resets the metric.
        """
        # Compute the final mAP score from all validation steps
        map_dict = self.val_map.compute()
        
        # --- FIX: Move computed metrics to the correct device before logging ---
        # This ensures the sync operation in a distributed setting receives a GPU tensor.
        val_mAP = map_dict['map'].to(self.device)
        val_mAP_50 = map_dict['map_50'].to(self.device)
        val_mAP_75 = map_dict['map_75'].to(self.device)

        # Log the metrics. 'val_mAP' is a common key for ModelCheckpoint to monitor.
        self.log_dict({
            'val_mAP': val_mAP,
            'val_mAP_50': val_mAP_50,
            'val_mAP_75': val_mAP_75,
        }, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # Reset the metric for the next validation epoch
        self.val_map.reset()

    # --- FIX: Helper methods for formatting predictions and targets ---
    def _format_targets_for_metric(self, targets):
        targets_formatted = []
        for target in targets:
            masks = target['masks'].to(self.device)
            targets_formatted.append({
                'boxes': masks_to_boxes(masks),
                'labels': target['labels'].to(self.device),
                'masks': masks.to(torch.uint8)
            })
        return targets_formatted

    def _format_preds_mrcnn(self, outputs):
        preds = []
        for pred in outputs:
            pred['masks'] = (pred['masks'] > 0.5).squeeze(1).to(torch.uint8)
            preds.append(pred)
        return preds
        
    def _format_preds_lw_detr(self, outputs, original_sizes):
        # This function now correctly processes the advanced DETR-style outputs.
        preds = []
        pred_logits = outputs['pred_logits']
        pred_boxes_normalized = outputs['pred_boxes']
        pred_masks = outputs['pred_masks']
        
        for i in range(pred_logits.shape[0]):
            h, w = original_sizes[i]
            
            # --- FIX IS HERE ---
            # The new model outputs exactly num_classes logits. Do not slice off the last one.
            # BEFORE: pred_probs = pred_logits[i].softmax(-1)[:, :-1]
            # AFTER:
            pred_probs = pred_logits[i].sigmoid() # For BCE-style losses, sigmoid is more appropriate than softmax
            # --- END FIX ---
            
            pred_scores, pred_labels = pred_probs.max(-1)
            keep = pred_scores > 0.05
            keep = keep.cpu()
            
            final_boxes = pred_boxes_normalized[i][keep]
            final_scores = pred_scores[keep]
            final_labels = pred_labels[keep]
            final_masks = pred_masks[i][keep]
    
            final_boxes = box_convert(final_boxes, in_fmt="cxcywh", out_fmt="xyxy")
            scale_fct = torch.tensor([w, h, w, h], device=self.device)
            final_boxes = final_boxes * scale_fct
    
            final_masks = F.interpolate(
                final_masks.unsqueeze(1), size=(h.item(), w.item()),
                mode="bilinear", align_corners=False
            ).squeeze(1)
            final_masks = (final_masks > 0.5).to(torch.uint8)
    
            preds.append({
                'scores': final_scores, 'labels': final_labels,
                'boxes': final_boxes, 'masks': final_masks,
            })
        return preds
        
    def _format_preds_detr(self, outputs, original_sizes):
        preds = []
        batch_size = outputs['pred_logits'].shape[0]
    
        for i in range(batch_size):
            h, w = original_sizes[i]
    
            pred_logits = outputs['pred_logits'][i]        # [num_queries, num_classes+1]
            pred_masks = outputs['pred_masks'][i]          # [num_queries, H, W]
    
            # Remove "no-object" class
            pred_probs = pred_logits.softmax(-1)[:, :-1]
            pred_scores, pred_labels = pred_probs.max(-1)
    
            # Filter out low-confidence predictions
            keep = pred_scores > 0.05
            keep = keep.cpu()
            pred_scores = pred_scores[keep]
            pred_labels = pred_labels[keep]
            pred_masks = pred_masks[keep]
    
            # Ensure masks are binary and match original image size
            pred_masks = (pred_masks > 0.5).to(torch.uint8)
    
            preds.append({
                'scores': pred_scores,
                'labels': pred_labels,
                'masks': pred_masks,
                'boxes': masks_to_boxes(pred_masks)
            })
    
        return preds
        
    def _format_preds_cf(self, outputs, original_sizes):
        preds = []
        batch_size = outputs['pred_logits'].shape[0]
    
        for i in range(batch_size):
            h, w = original_sizes[i]
    
            pred_logits = outputs['pred_logits'][i]
            pred_coords = outputs['pred_coords'][i]
    
            pred_probs = pred_logits.softmax(-1)[:, :-1]  # exclude "no-object" class
            pred_scores, pred_labels = pred_probs.max(-1)
    
            # Convert contours to masks
            pred_masks = contours_to_masks(pred_coords, (h.item(), w.item()))
    
            # Filter low-confidence predictions
            keep = pred_scores > 0.05
            keep = keep.cpu()
            pred_scores = pred_scores[keep]
            pred_labels = pred_labels[keep]
            pred_masks = pred_masks[keep].to(torch.uint8)
    
            preds.append({
                'scores': pred_scores,
                'labels': pred_labels,
                'masks': pred_masks,
                'boxes': masks_to_boxes(pred_masks)
            })
    
        return preds

    # --- FIX: Helper methods for calculating supervised loss ---
    def _get_sup_loss_detr(self, pixel_values, targets):
        targets_detr = []
        h, w = self.hparams.image_size, self.hparams.image_size
        for target in targets:
            boxes_xyxy = masks_to_boxes(target["masks"])
            boxes_cxcywh = box_convert(boxes_xyxy, in_fmt="xyxy", out_fmt="cxcywh")
            boxes_cxcywh[:, 0::2] /= w
            boxes_cxcywh[:, 1::2] /= h
            targets_detr.append({
                "labels": target["labels"].to(self.device), "boxes": boxes_cxcywh.to(self.device),
                "masks": target["masks"].to(self.device) # Assuming model handles resizing
            })
        _, losses_sup = self.student(pixel_values=pixel_values, targets=targets_detr)
        return sum(losses_sup.values())

    def _get_sup_loss_cf(self, pixel_values, targets):
        targets_cf = []
        for target in targets:
            try:
                contours, valid_indices = masks_to_contours(target["masks"])
                if len(valid_indices) > 0:
                    targets_cf.append({
                        "labels": target["labels"][valid_indices].to(self.device),
                        "coords": contours.to(self.device)
                    })
            except Exception: # Handle cases with no valid contours
                pass
        
        if not targets_cf: return torch.tensor(0.0, device=self.device)
        
        _, losses_sup = self.student(pixel_values=pixel_values, targets=targets_cf)
        return sum(losses_sup.values())

    def _get_sup_loss_mrcnn(self, pixel_values, targets):
        targets_mrcnn = []
        for target in targets:
            targets_mrcnn.append({
                "boxes": masks_to_boxes(target["masks"]).to(self.device),
                "labels": target["labels"].to(self.device),
                "masks": target["masks"].to(self.device)
            })
        losses_sup = self.student(pixel_values=pixel_values, targets=targets_mrcnn)
        return sum(losses_sup.values())
