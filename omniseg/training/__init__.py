import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import transforms
from torchvision.ops import box_convert, masks_to_boxes
from torchmetrics.detection import MeanAveragePrecision
from typing import Dict, List, Tuple, Union

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
            teacher_preds, _ = self.teacher(pixel_values=pixel_values_u)
        
        pseudo_targets = self._create_pseudo_targets(teacher_preds, self.hparams.pseudo_label_thresh)
        num_pseudo_boxes = sum(len(t['labels']) for t in pseudo_targets)

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
            "sup_loss": sup_loss,
            "unsup_loss": unsup_loss,
            "unsup_weight": consistency_weight,
            "lr": self.trainer.optimizers[0].param_groups[0]['lr']
        }, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)

        return total_loss

    def validation_step(self, batch: Tuple[torch.Tensor, List[Dict]], batch_idx: int):
        pixel_values, targets = batch

        if self.hparams.head_type != 'maskrcnn' and not isinstance(pixel_values, torch.Tensor):
            pixel_values = torch.stack(pixel_values, dim=0)

        self.student.eval()

        with torch.no_grad():
            if self.hparams.head_type == 'maskrcnn':
                self.student.train()
                # The model now returns a tuple: (predictions, losses).
                # In training mode, our workaround returns (loss_dict, loss_dict).
                # We only need the second element here for the loss values.
                _, val_loss_dict = self.student(pixel_values, self._prepare_mrcnn_targets(targets))
                val_loss = sum(val_loss_dict.values())
                
                self.student.eval()
                # In eval mode, the model returns (predictions, {}).
                # We only need the first element here for the output predictions.
                outputs, _ = self.student(pixel_values)
            else:
                targets_detr = self._prepare_detr_targets(targets)
                outputs, val_losses_dict = self.student(pixel_values=pixel_values, targets=targets_detr)
                val_loss = sum(val_losses_dict.values()) if val_losses_dict else torch.tensor(0.0, device=self.device)

        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        targets_formatted = self._format_targets_for_metric(targets)
        original_sizes = torch.tensor([t['masks'].shape[-2:] for t in targets], device=self.device)

        if self.hparams.head_type == 'maskrcnn':
            preds = self._format_preds_mrcnn(outputs)
        else:
            preds = self._format_preds_detr_style(outputs, original_sizes)

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
        """Compute supervised loss for DETR-style models.
        
        Args:
            pixel_values: Input images tensor
            targets: Raw target annotations that need preprocessing
            
        Returns:
            Total supervised loss
        """
        targets_detr = self._prepare_detr_targets(targets)
        _, sup_losses_dict = self.student(pixel_values=pixel_values, targets=targets_detr)
        return sum(sup_losses_dict.values())

    def _get_unsup_loss_detr(self, strong_images: torch.Tensor, pseudo_targets: List[Dict]) -> torch.Tensor:
        """Compute unsupervised loss for DETR-style models.
        
        Args:
            strong_images: Strongly augmented images tensor
            pseudo_targets: Pseudo-labels from teacher model
            
        Returns:
            Total unsupervised loss
        """
        self.student.train()
        
        # Filter pseudo-targets to remove invalid boxes
        final_pseudo_targets = []
        device = strong_images.device
    
        for target in pseudo_targets:
            boxes = target.get("boxes")
            
            if boxes is not None and boxes.shape[0] > 0:
                # Keep only valid boxes (x2 > x1 and y2 > y1)
                valid_indices = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
                
                if valid_indices.any():
                    # Filter target to keep only valid entries
                    new_target = {key: value[valid_indices] for key, value in target.items()}
                    final_pseudo_targets.append(new_target)
                    continue
            
            # Create empty target for images with no valid boxes
            final_pseudo_targets.append({
                "boxes": torch.empty((0, 4), dtype=torch.float32, device=device),
                "labels": torch.empty(0, dtype=torch.long, device=device),
                "masks": torch.empty(0, self.hparams.image_size, self.hparams.image_size, device=device)
            })
            
        # Handle edge case of completely empty batch
        if not final_pseudo_targets:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        _, unsup_losses_dict = self.student(pixel_values=strong_images, targets=final_pseudo_targets)
        return sum(unsup_losses_dict.values())

    def _get_sup_loss_mrcnn(self, pixel_values: torch.Tensor, targets: List[Dict]) -> torch.Tensor:
        """Compute supervised loss for Mask R-CNN model.
        
        Args:
            pixel_values: Input images tensor
            targets: Target annotations for Mask R-CNN
            
        Returns:
            Total supervised loss
        """
        targets_mrcnn = self._prepare_mrcnn_targets(targets)
        _, losses_sup_dict = self.student(pixel_values, targets_mrcnn)
        return sum(losses_sup_dict.values())
    
    def _create_pseudo_targets(self, preds: Union[Dict, List[Dict]], conf_thresh: float) -> List[Dict]:
        pseudo_targets = []
    
        # CASE 1: Handle DETR-style output (a single dictionary of batched tensors)
        if isinstance(preds, dict):
            logits, boxes_cxcywh, masks_logits = preds['pred_logits'], preds['pred_boxes'], preds['pred_masks']
            for i in range(logits.shape[0]): # Iterate over batch
                probs = logits[i].softmax(-1)[:, :-1]
                scores, labels = probs.max(-1)
                keep = scores > conf_thresh
    
                if not keep.any():
                    pseudo_targets.append({
                        "labels": torch.tensor([], dtype=torch.long, device=self.device),
                        "boxes": torch.tensor([], device=self.device),
                        "masks": torch.tensor([], device=self.device)
                    })
                    continue
    
                pseudo_targets.append({
                    "labels": labels[keep].detach(),
                    "boxes": boxes_cxcywh[i][keep].detach(),
                    "masks": (masks_logits[i][keep].sigmoid() > 0.5).float().detach()
                })
    
        # CASE 2: Handle Mask R-CNN-style output (a list of dictionaries)
        elif isinstance(preds, list):
            h, w = self.hparams.image_size, self.hparams.image_size
            for pred_dict in preds: # Iterate over predictions for each image
                keep = pred_dict['scores'] > conf_thresh
    
                if not keep.any():
                    pseudo_targets.append({
                        "labels": torch.tensor([], dtype=torch.long, device=self.device),
                        "boxes": torch.tensor([], device=self.device),
                        "masks": torch.tensor([], device=self.device)
                    })
                    continue
                
                # The unsupervised loss expects DETR format, so we must convert the boxes
                boxes_xyxy = pred_dict['boxes'][keep]
                boxes_cxcywh = box_convert(boxes_xyxy, in_fmt="xyxy", out_fmt="cxcywh")
                boxes_cxcywh[:, 0::2] /= w
                boxes_cxcywh[:, 1::2] /= h
                
                pseudo_targets.append({
                    "labels": pred_dict['labels'][keep].detach(),
                    "boxes": boxes_cxcywh.detach(),
                    "masks": (pred_dict['masks'][keep] > 0.5).squeeze(1).float().detach()
                })
                
        return pseudo_targets

    def _prepare_detr_targets(self, targets: List[Dict]) -> List[Dict]:
        targets_detr = []
        h, w = self.hparams.image_size, self.hparams.image_size
        device = self.device
        for target in targets:
            boxes_xyxy = masks_to_boxes(target["masks"])
            boxes_cxcywh = box_convert(boxes_xyxy, in_fmt="xyxy", out_fmt="cxcywh")
            boxes_cxcywh[:, 0::2] /= w
            boxes_cxcywh[:, 1::2] /= h
            targets_detr.append({
                "labels": target["labels"].to(device),
                "boxes": boxes_cxcywh.to(device),
                "masks": target["masks"].to(device).float()
            })
        return targets_detr

    def _prepare_mrcnn_targets(self, targets: List[Dict]) -> List[Dict]:
        return [{
            "boxes": masks_to_boxes(t["masks"].to(self.device)),
            "labels": t["labels"].to(self.device),
            "masks": t["masks"].to(self.device, torch.uint8)
        } for t in targets]

    def _format_targets_for_metric(self, targets: List[Dict]) -> List[Dict]:
        return [{
            'boxes': masks_to_boxes(t['masks']),
            'labels': t['labels'],
            'masks': t['masks'].to(torch.uint8)
        } for t in targets]

    def _format_preds_mrcnn(self, outputs: List[Dict]) -> List[Dict]:
        return [{
            'scores': pred['scores'],
            'labels': pred['labels'],
            'boxes': pred['boxes'],
            'masks': (pred['masks'] > 0.5).squeeze(1).to(torch.uint8)
        } for pred in outputs]

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
