import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import transforms
from torchvision.ops import box_convert, masks_to_boxes
from torchmetrics.detection import MeanAveragePrecision
from typing import Dict, List, Tuple, Any, Union

class SSLSegmentationLightning(pl.LightningModule):
    def __init__(self,
                 head: nn.Module,
                 head_type: str,
                 image_size: int,
                 num_classes: int = 80,
                 lr: float = 1e-4,
                 ema_decay: float = 0.999,
                 warmup_steps: int = 500,
                 unsup_rampup_steps: int = 5000,
                 unsup_weight: float = 1.0,
                 pseudo_label_thresh: float = 0.7):
        super().__init__()
        self.save_hyperparameters(ignore=['head'])

        self.student = head
        self.teacher = copy.deepcopy(self.student)
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

        self.val_map = MeanAveragePrecision(box_format='xyxy', iou_type='segm')

        # ============================ NEW: On-the-fly Strong Augmentation ============================
        # This pipeline runs on the GPU to create the "strong" view for the student model.
        self.strong_augment = nn.Sequential(
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0),
        )
        # =========================================================================================

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [p for p in self.student.parameters() if p.requires_grad],
            lr=self.hparams.lr,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "train_loss_epoch"}
        }

    @torch.no_grad()
    def _update_teacher(self):
        for t_param, s_param in zip(self.teacher.parameters(), self.student.parameters()):
            t_param.data.mul_(self.hparams.ema_decay).add_(s_param.data, alpha=1 - self.hparams.ema_decay)

    def on_before_optimizer_step(self, optimizer):
        self._update_teacher()

    def training_step(self, batch, batch_idx):
        # --- Unpack Data ---
        pixel_values_l, targets_l = batch["labeled"]
        # Correctly unpacks the unlabeled data as (images, dummy_targets)
        pixel_values_u, _ = batch["unlabeled"]
        
        batch_size = pixel_values_l.size(0) if isinstance(pixel_values_l, torch.Tensor) else len(pixel_values_l)

        # Ensure all inputs are batched tensors for DETR models
        if self.hparams.head_type != 'maskrcnn':
            if not isinstance(pixel_values_l, torch.Tensor):
                pixel_values_l = torch.stack(pixel_values_l, dim=0)
            if not isinstance(pixel_values_u, torch.Tensor):
                pixel_values_u = torch.stack(pixel_values_u, dim=0)
        
        # --- Supervised Loss ---
        if self.hparams.head_type == 'maskrcnn':
            sup_loss = self._get_sup_loss_mrcnn(pixel_values_l, targets_l)
        else:
            sup_loss = self._get_sup_loss_detr(pixel_values_l, targets_l)

        # --- Unsupervised Loss (New Strategy) ---
        # 1. Generate pseudo-labels from the teacher on the original ("weak") unlabeled images
        with torch.no_grad():
            self.teacher.eval()
            # Handle different output formats for teacher predictions
            teacher_output = self.teacher(pixel_values=pixel_values_u)
            teacher_preds = teacher_output[0] if self.hparams.head_type == 'maskrcnn' else teacher_output

        pseudo_targets = self._create_pseudo_targets(teacher_preds, self.hparams.pseudo_label_thresh)
        num_pseudo_boxes = sum(len(t.get('labels', [])) for t in pseudo_targets)

        if num_pseudo_boxes > 0:
            # 2. Create a strongly augmented view of the images on the fly
            pixel_values_u_strong = self.strong_augment(pixel_values_u)

            # 3. Calculate student's loss on the strongly augmented view with the pseudo-labels
            unsup_loss = self._get_unsup_loss_detr(pixel_values_u_strong, pseudo_targets)
        else:
            # If no high-confidence pseudo-labels, unsupervised loss is zero for this batch
            unsup_loss = torch.tensor(0.0, device=self.device)

        # --- Combine and Log Losses ---
        rampup = min(1.0, self.global_step / self.hparams.unsup_rampup_steps)
        consistency_weight = self.hparams.unsup_weight * rampup
        total_loss = sup_loss + consistency_weight * unsup_loss

        self.log_dict({
            "train_loss": total_loss,
            "sup_loss": sup_loss.detach(),
            "unsup_loss": unsup_loss.detach(),
            "unsup_weight": consistency_weight,
            "lr": self.trainer.optimizers[0].param_groups[0]['lr']
        }, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)

        return total_loss

    def validation_step(self, batch: Tuple[torch.Tensor, List[Dict]], batch_idx: int):
        pixel_values, targets = batch
    
        if self.hparams.head_type != 'maskrcnn' and not isinstance(pixel_values, torch.Tensor):
            pixel_values = torch.stack(pixel_values, dim=0)
    
        # --- Calculate validation loss ---
        self.student.train() # Set to train mode for loss calculation
        with torch.no_grad():
            if self.hparams.head_type == 'maskrcnn':
                # Filter images and targets that become empty after preparation
                prepared_targets = self._prepare_mrcnn_targets(targets)
                valid_pixel_values = [img for img, t in zip(pixel_values, prepared_targets) if t['labels'].numel() > 0]
                valid_targets = [t for t in prepared_targets if t['labels'].numel() > 0]
                if valid_targets:
                    _, val_loss_dict = self.student(valid_pixel_values, valid_targets)
                    val_loss = sum(val_loss_dict.values())
                else:
                    val_loss = torch.tensor(0.0, device=self.device)
            else:
                targets_detr = self._prepare_detr_targets(targets)
                if targets_detr:
                    _, val_losses_dict = self.student(pixel_values=pixel_values, targets=targets_detr)
                    val_loss = sum(val_losses_dict.values())
                else:
                    val_loss = torch.tensor(0.0, device=self.device)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
    
        # --- Calculate mAP ---
        self.student.eval() # Set to eval mode for predictions
        with torch.no_grad():
            # The model output format can differ; assuming it returns a tuple
            # where the first element contains the predictions.
            # Adjust if your model's forward pass in eval mode is different.
            outputs = self.student(pixel_values)
            if isinstance(outputs, tuple):
                 outputs = outputs[0]
    
    
        targets_formatted = self._format_targets_for_metric(targets)
        # Ensure original_sizes are computed correctly even for empty targets
        original_sizes = []
        for t in targets:
            if 'masks' in t and t['masks'].numel() > 0:
                original_sizes.append(t.get('original_size', t['masks'].shape[-2:]))
            elif 'original_size' in t:
                 original_sizes.append(t['original_size'])
            else:
                # Fallback for empty targets: use the model's input image size
                original_sizes.append((self.hparams.image_size, self.hparams.image_size))
        original_sizes = torch.tensor(original_sizes, device=self.device)
    
    
        if self.hparams.head_type == 'maskrcnn':
            preds = self._format_preds_mrcnn(outputs)
        else:
            preds = self._format_preds_detr_style(outputs, original_sizes)
        
        # --- THIS IS THE CORRECTED PART ---
        # Update the metric unconditionally.
        # torchmetrics is designed to handle empty predictions/targets correctly.
        self.val_map.update(preds, targets_formatted)
        
    def on_validation_epoch_end(self):
        map_dict = self.val_map.compute()
        self.log_dict({
            'val_mAP': map_dict['map'].to(self.device),
            'val_mAP_50': map_dict['map_50'].to(self.device),
            'val_mAP_75': map_dict['map_75'].to(self.device),
        }, on_epoch=True, prog_bar=True, sync_dist=True)
        self.val_map.reset()

    def _get_sup_loss_detr(self, pixel_values: torch.Tensor, targets: List[Dict]) -> torch.Tensor:
        targets_detr = self._prepare_detr_targets(targets)
        if not targets_detr:
             return torch.tensor(0.0, device=self.device)
        
        _, sup_losses_dict = self.student(pixel_values=pixel_values, targets=targets_detr)
        return sum(sup_losses_dict.values())

    def _get_unsup_loss_detr(self, strong_images: torch.Tensor, pseudo_targets: List[Dict]) -> torch.Tensor:
        self.student.train()
        # pseudo_targets from _create_pseudo_targets are already in cxcywh format
        # and filtered. We just need to prepare them for the model.
        targets_detr = self._prepare_detr_targets(pseudo_targets)
        if not targets_detr:
            return torch.tensor(0.0, device=strong_images.device)

        _, unsup_losses_dict = self.student(pixel_values=strong_images, targets=targets_detr)
        return sum(unsup_losses_dict.values())

    def _get_sup_loss_mrcnn(self, pixel_values: torch.Tensor, targets: List[Dict]) -> torch.Tensor:
        targets_mrcnn = self._prepare_mrcnn_targets(targets)
        
        # Filter out images that have no valid targets after preparation
        valid_pixel_values = [img for img, t in zip(pixel_values, targets_mrcnn) if t['labels'].numel() > 0]
        valid_targets = [t for t in targets_mrcnn if t['labels'].numel() > 0]

        if not valid_targets:
            return torch.tensor(0.0, device=self.device)

        _, losses_sup_dict = self.student(valid_pixel_values, valid_targets)
        return sum(losses_sup_dict.values())
    
    def _create_pseudo_targets(self, preds: Union[Dict, List[Dict]], conf_thresh: float) -> List[Dict]:
        pseudo_targets = []
        H, W = self.hparams.image_size, self.hparams.image_size
        device = self.device
    
        # CASE 1: DETR-style (batched dict)
        if isinstance(preds, dict):
            logits = preds['pred_logits']
            boxes_cxcywh = preds['pred_boxes']
            masks_logits = preds.get('pred_masks', None)
    
            for i in range(logits.shape[0]):
                probs = logits[i].softmax(-1)[:, :-1]
                scores, labels = probs.max(-1)
                keep = scores > conf_thresh
    
                if not keep.any():
                    pseudo_targets.append({}) # Append empty dict for placeholder
                    continue
    
                masks_bin = (masks_logits[i][keep].sigmoid() > 0.5).float().detach() if masks_logits is not None else torch.empty((keep.sum(), H, W), device=device)
                pseudo_targets.append({
                    "labels": labels[keep].detach(),
                    "boxes": boxes_cxcywh[i][keep].detach(),
                    "masks": masks_bin
                })
    
        # CASE 2: Mask R-CNN outputs (list of dicts)
        elif isinstance(preds, list):
            for pred_dict in preds:
                keep = pred_dict['scores'] > conf_thresh
                if not keep.any():
                    pseudo_targets.append({})
                    continue
    
                boxes_xyxy = pred_dict['boxes'][keep]
                # Convert boxes to normalized cxcywh for consistency with DETR pipeline
                boxes_cxcywh = box_convert(boxes_xyxy, in_fmt="xyxy", out_fmt="cxcywh")
                boxes_cxcywh[:, 0::2] /= W
                boxes_cxcywh[:, 1::2] /= H
    
                masks_bin = (pred_dict['masks'][keep] > 0.5).squeeze(1).float()
                pseudo_targets.append({
                    "labels": pred_dict['labels'][keep].detach(),
                    "boxes": boxes_cxcywh.detach(), # Already in cxcywh
                    "masks": masks_bin.detach()
                })
        return pseudo_targets

    def _prepare_detr_targets(self, targets: List[Dict]) -> List[Dict]:
        targets_detr = []
        h, w = self.hparams.image_size, self.hparams.image_size
        device = self.device
    
        for target in targets:
            # Check for essential keys; if missing, it's an empty pseudo-target
            if "masks" not in target or "labels" not in target:
                continue
            
            masks = target["masks"]
            labels = target["labels"]
            
            # Skip if there are no annotations for this image
            if masks.numel() == 0 or labels.numel() == 0:
                continue

            # Filter out masks that are completely empty
            keep = masks.any(dim=(-1, -2))
            if not keep.any():
                continue
            
            valid_masks = masks[keep]
            valid_labels = labels[keep]

            # Safely generate boxes from valid masks
            boxes_xyxy = masks_to_boxes(valid_masks)
            
            # Filter out boxes with zero area (width or height is 0)
            valid_box_indices = (boxes_xyxy[:, 2] > boxes_xyxy[:, 0]) & (boxes_xyxy[:, 3] > boxes_xyxy[:, 1])
            if not valid_box_indices.any():
                continue

            # Final filtering
            final_masks = valid_masks[valid_box_indices]
            final_labels = valid_labels[valid_box_indices]
            final_boxes_xyxy = boxes_xyxy[valid_box_indices]

            # Convert to normalized cxcywh format
            boxes_cxcywh = box_convert(final_boxes_xyxy, in_fmt="xyxy", out_fmt="cxcywh")
            boxes_cxcywh[:, 0::2] /= w
            boxes_cxcywh[:, 1::2] /= h

            targets_detr.append({
                "labels": final_labels.to(device),
                "boxes": boxes_cxcywh.to(device),
                "masks": final_masks.to(device).float()
            })
        return targets_detr
        
    def _prepare_mrcnn_targets(self, targets: List[Dict]) -> List[Dict]:
        prepared_targets = []
        for t in targets:
            masks = t.get("masks")
            labels = t.get("labels")

            # Skip if essential data is missing or empty
            if masks is None or labels is None or masks.numel() == 0 or labels.numel() == 0:
                prepared_targets.append({
                    "boxes": torch.zeros((0, 4), dtype=torch.float32, device=self.device),
                    "labels": torch.zeros((0,), dtype=torch.long, device=self.device),
                    "masks": torch.zeros((0, self.hparams.image_size, self.hparams.image_size), dtype=torch.uint8, device=self.device)
                })
                continue
            
            # Filter out completely empty masks
            keep = masks.any(dim=(-1, -2))
            if not keep.any():
                prepared_targets.append({
                    "boxes": torch.zeros((0, 4), dtype=torch.float32, device=self.device),
                    "labels": torch.zeros((0,), dtype=torch.long, device=self.device),
                    "masks": torch.zeros((0, self.hparams.image_size, self.hparams.image_size), dtype=torch.uint8, device=self.device)
                })
                continue

            valid_masks = masks[keep].to(self.device)
            valid_labels = labels[keep].to(self.device)

            # Safely generate boxes and filter for valid ones
            boxes = masks_to_boxes(valid_masks)
            valid_box_indices = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            
            prepared_targets.append({
                "boxes": boxes[valid_box_indices],
                "labels": valid_labels[valid_box_indices],
                "masks": valid_masks[valid_box_indices].to(torch.uint8)
            })
        return prepared_targets

    def _format_targets_for_metric(self, targets: List[Dict]) -> List[Dict]:
        # Reuse the robust mrcnn target prep for consistent formatting
        return self._prepare_mrcnn_targets(targets)

    def _format_preds_mrcnn(self, outputs: List[Dict]) -> List[Dict]:
        formatted = []
        for pred in outputs:
            # Ensure all keys exist, providing empty tensors as defaults
            scores = pred.get('scores', torch.tensor([], device=self.device))
            labels = pred.get('labels', torch.tensor([], dtype=torch.long, device=self.device))
            boxes = pred.get('boxes', torch.zeros((0, 4), device=self.device))
            masks = pred.get('masks', torch.zeros((0, 1, self.hparams.image_size, self.hparams.image_size), device=self.device))

            formatted.append({
                'scores': scores,
                'labels': labels,
                'boxes': boxes,
                'masks': (masks > 0.5).squeeze(1).to(torch.uint8) if masks.numel() > 0 else torch.empty((0, self.hparams.image_size, self.hparams.image_size), dtype=torch.uint8, device=self.device)
            })
        return formatted

    def _format_preds_detr_style(self, outputs: Dict, original_sizes: torch.Tensor) -> List[Dict]:
        preds = []
        pred_logits, pred_boxes, pred_masks = outputs['pred_logits'], outputs['pred_boxes'], outputs['pred_masks']

        for i, (h, w) in enumerate(original_sizes):
            logits, boxes_norm, masks_logits = pred_logits[i], pred_boxes[i], pred_masks[i]
            probs = logits.softmax(-1)[..., :-1]
            scores, labels = probs.max(-1)
            
            boxes = box_convert(boxes_norm, in_fmt="cxcywh", out_fmt="xyxy")
            boxes *= torch.tensor([w.item(), h.item(), w.item(), h.item()], device=boxes.device)
            
            masks = F.interpolate(
                masks_logits.unsqueeze(1),
                size=(h.item(), w.item()),
                mode="bilinear",
                align_corners=False
            ).squeeze(1)
            masks = (masks.sigmoid() > 0.5).to(torch.uint8)
            
            preds.append({"scores": scores, "labels": labels, "boxes": boxes, "masks": masks})
        return preds