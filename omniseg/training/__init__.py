import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union

import pytorch_lightning as pl
from torchvision.ops import box_convert, masks_to_boxes
from torchmetrics.detection import MeanAveragePrecision

from ..models.heads.contourformer import masks_to_contours, contours_to_masks


class SSLSegmentationLightning(pl.LightningModule):
    """
    Semi-supervised segmentation training with PyTorch Lightning.
    This version is compatible with a dataloader that handles all augmentations
    and is robust to different target batching formats (list of dicts or dict of tensors).
    """

    def __init__(self,
                 head: nn.Module,
                 head_type: str,
                 image_size: int,
                 num_classes: int = 80,
                 lr: float = 1e-4,
                 ema_decay: float = 0.999,
                 warmup_steps: int = 500,
                 unsup_rampup_steps: int = 5000,
                 unsup_weight: float = 1.0
                ):
        super().__init__()
        self.save_hyperparameters('head_type', 'image_size', 'num_classes', 'lr', 'ema_decay', 'warmup_steps', 'unsup_rampup_steps', 'unsup_weight')

        self.student = head
        
        self.teacher = copy.deepcopy(self.student)
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

        self.val_map = MeanAveragePrecision(box_format='xyxy', iou_type='segm')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [p for p in self.student.parameters() if p.requires_grad], 
            lr=self.hparams.lr
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=3,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss_epoch",
            },
        }

    @torch.no_grad()
    def _update_teacher(self):
        for t_param, s_param in zip(self.teacher.parameters(), self.student.parameters()):
            t_param.data.mul_(self.hparams.ema_decay).add_(s_param.data, alpha=1 - self.hparams.ema_decay)

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        if hasattr(self.student, 'criterion') and hasattr(self.student.criterion, 'matcher'):
            if hasattr(self.student.criterion.matcher, 'step_epoch'):
                self.student.criterion.matcher.step_epoch()

    def on_before_optimizer_step(self, optimizer):
        self._update_teacher()

    def training_step(self, batch, batch_idx):
        pixel_values_l, targets_l = batch["labeled"]
        pixel_values_u, _ = batch["unlabeled"]
        
        batch_size = pixel_values_l.size(0)
        sup_loss = 0.0
        if self.hparams.head_type == 'maskrcnn':
            sup_loss = self._get_sup_loss_mrcnn(pixel_values_l, targets_l)
        elif self.hparams.head_type in ['contourformer', 'deformable_detr', 'lw_detr', 'sparrow_seg']:
            sup_loss = self._get_sup_loss_detr_style(pixel_values_l, targets_l)
        
        with torch.no_grad():
            self.teacher.eval()
            teacher_preds = self.teacher(pixel_values_u)
        
        self.student.eval()
        student_preds = self.student(pixel_values_u)
        self.student.train()
        
        unsup_loss = torch.tensor(0.0, device=self.device)
        if isinstance(teacher_preds, dict) and isinstance(student_preds, dict):
            t_tensor = next(iter(teacher_preds.values()))
            s_tensor = next(iter(student_preds.values()))
            unsup_loss = F.l1_loss(s_tensor, t_tensor)
            
        rampup = min(1.0, self.global_step / self.hparams.unsup_rampup_steps)
        consistency_weight = self.hparams.unsup_weight * rampup
        total_loss = sup_loss + consistency_weight * unsup_loss

        self.log_dict({
            "train_loss": total_loss, "sup_loss": sup_loss, "unsup_loss": unsup_loss,
            "unsup_weight": consistency_weight,
        }, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        
        return total_loss
        
    def validation_step(self, batch: Tuple[torch.Tensor, Union[List[Dict], Dict]], batch_idx: int):
        pixel_values, targets = batch

        if isinstance(targets, dict):
            targets = self._unbatch_targets(targets)
        
        self.student.train()
        val_loss = 0.0
        if self.hparams.head_type == 'maskrcnn':
            val_loss = self._get_sup_loss_mrcnn(pixel_values, targets)
        elif self.hparams.head_type in ['contourformer', 'deformable_detr', 'lw_detr', 'sparrow_seg']:
            val_loss = self._get_sup_loss_detr_style(pixel_values, targets)
        self.student.eval()
        
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        outputs = self.student(pixel_values)
        
        targets_formatted = self._format_targets_for_metric(targets)
        
        coco_api = self.trainer.datamodule.coco_gt_val
        
        img_ids = [t['image_id'].item() for t in targets] 
        img_infos = coco_api.loadImgs(img_ids)
        original_sizes = torch.tensor([[info['height'], info['width']] for info in img_infos], device=self.device)

        preds = []
        if self.hparams.head_type == 'maskrcnn':
            preds = self._format_preds_mrcnn(outputs)
        elif self.hparams.head_type == 'deformable_detr':
            preds = self._format_preds_detr(outputs, original_sizes)
        elif self.hparams.head_type == 'lw_detr':
            preds = self._format_preds_lw_detr(outputs, original_sizes)
        elif self.hparams.head_type == 'contourformer':
            preds = self._format_preds_cf(outputs, original_sizes)
        elif self.hparams.head_type == 'sparrow_seg':
            preds = self._format_preds_sparrow_seg(outputs, original_sizes)
        
        self.val_map.update(preds, targets_formatted)

    def on_validation_epoch_end(self):
        map_dict = self.val_map.compute()
        val_mAP = map_dict['map'].to(self.device)
        val_mAP_50 = map_dict['map_50'].to(self.device)
        val_mAP_75 = map_dict['map_75'].to(self.device)

        self.log_dict({
            'val_mAP': val_mAP, 'val_mAP_50': val_mAP_50, 'val_mAP_75': val_mAP_75,
        }, on_epoch=True, prog_bar=True, sync_dist=True)
        
        self.val_map.reset()

    def _unbatch_targets(self, targets_dict: Dict[str, torch.Tensor]) -> List[Dict[str, Any]]:
        unbatched_targets = []
        if 'image_id' in targets_dict:
            batch_size = targets_dict['image_id'].shape[0]
        elif 'labels' in targets_dict:
            batch_size = targets_dict['labels'].shape[0]
        else:
            return [targets_dict]
            
        keys = targets_dict.keys()
        for i in range(batch_size):
            target = {key: targets_dict[key][i] for key in keys}
            unbatched_targets.append(target)
            
        return unbatched_targets

    def _format_targets_for_metric(self, targets):
        targets_formatted = []
        for target in targets:
            masks = target['masks'].to(self.device)
            targets_formatted.append({
                'boxes': masks_to_boxes(masks), 'labels': target['labels'].to(self.device),
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
        logits, boxes_norm, masks_logits = outputs["pred_logits"], outputs["pred_boxes"], outputs["pred_masks"]
        results = []
        for i, (h, w) in enumerate(original_sizes):
            logits_classes = logits[i][..., :-1]
            probs = logits_classes.sigmoid()
            
            scores, labels = probs.max(-1)
            keep = scores > 0.05

            scores = scores[keep]
            labels = labels[keep]
            boxes = box_convert(boxes_norm[i][keep], in_fmt="cxcywh", out_fmt="xyxy")
            boxes *= torch.tensor([w, h, w, h], device=boxes.device)
            masks = masks_logits[i][keep]

            if masks.numel() > 0:
                masks = F.interpolate(masks.unsqueeze(1), size=(h.item(), w.item()), mode="bilinear", align_corners=False).squeeze(1)
                masks = (masks.sigmoid() > 0.1).to(torch.uint8)
            else:
                masks = torch.empty((0, h, w), device=masks.device, dtype=torch.uint8)

            results.append({"scores": scores, "labels": labels, "boxes": boxes, "masks": masks})
        return results
        
    def _format_preds_detr(self, outputs, original_sizes):
        preds = []
        for i in range(outputs['pred_logits'].shape[0]):
            pred_logits, pred_masks_raw = outputs['pred_logits'][i], outputs['pred_masks'][i]
            
            pred_probs = pred_logits.softmax(-1)[:, :-1]
            pred_scores, pred_labels = pred_probs.max(-1)
            
            keep = pred_scores > 0.05
            
            scores = pred_scores[keep]
            labels = pred_labels[keep]
            masks_unfiltered = (pred_masks_raw[keep] > 0.5)

            non_empty_indices = [j for j, m in enumerate(masks_unfiltered) if m.any()]
            if not non_empty_indices:
                preds.append({'scores': torch.tensor([]), 'labels': torch.tensor([]), 'masks': torch.tensor([]), 'boxes': torch.tensor([])})
                continue

            final_scores = scores[non_empty_indices]
            final_labels = labels[non_empty_indices]
            final_masks = masks_unfiltered[non_empty_indices]
            
            preds.append({
                'scores': final_scores,
                'labels': final_labels,
                'masks': final_masks.to(torch.uint8),
                'boxes': masks_to_boxes(final_masks)
            })
        return preds
        
    def _format_preds_cf(self, outputs, original_sizes):
        preds = []
        for i in range(outputs['pred_logits'].shape[0]):
            h, w = original_sizes[i]
            pred_logits, pred_coords = outputs['pred_logits'][i], outputs['pred_coords'][i]
            
            pred_probs = pred_logits.softmax(-1)[:, :-1]
            pred_scores, pred_labels = pred_probs.max(-1)
            
            keep = (pred_scores > 0.05)

            scores = pred_scores[keep]
            labels = pred_labels[keep]
            masks_unfiltered = contours_to_masks(pred_coords[keep], (h.item(), w.item()))

            non_empty_indices = [j for j, m in enumerate(masks_unfiltered) if m.any()]
            if not non_empty_indices:
                preds.append({'scores': torch.tensor([]), 'labels': torch.tensor([]), 'masks': torch.tensor([]), 'boxes': torch.tensor([])})
                continue
                
            final_scores = scores[non_empty_indices]
            final_labels = labels[non_empty_indices]
            final_masks = masks_unfiltered[non_empty_indices]

            preds.append({
                'scores': final_scores,
                'labels': final_labels,
                'masks': final_masks.to(torch.uint8),
                'boxes': masks_to_boxes(final_masks)
            })
        return preds
        
    def _format_preds_sparrow_seg(self, outputs, original_sizes):
        pred_logits, pred_boxes_norm, pred_masks_logits = outputs["pred_logits"], outputs["pred_boxes"], outputs["pred_masks"]
        results = []
        for i, (h, w) in enumerate(original_sizes):
            probs = pred_logits[i].softmax(-1)[..., :-1]
            scores, labels = probs.max(-1)
            
            keep = scores > 0.1
            scores, labels = scores[keep], labels[keep]
            boxes_norm, masks_logits = pred_boxes_norm[i][keep], pred_masks_logits[i][keep]

            if scores.numel() == 0:
                results.append({"scores": torch.tensor([]), "labels": torch.tensor([]), "boxes": torch.tensor([]), "masks": torch.tensor([])})
                continue
            
            boxes = box_convert(boxes_norm, in_fmt="cxcywh", out_fmt="xyxy")
            boxes *= torch.tensor([w, h, w, h], device=boxes.device)
            masks = F.interpolate(masks_logits.unsqueeze(1), size=(h.item(), w.item()), mode="bilinear", align_corners=False).squeeze(1)
            masks = (masks.sigmoid() > 0.5).to(torch.uint8)
            results.append({"scores": scores, "labels": labels, "boxes": boxes, "masks": masks})
        return results
        
    def _get_sup_loss_detr_style(self, pixel_values, targets):
        targets_detr = []
        h, w = self.hparams.image_size, self.hparams.image_size
        device = pixel_values.device
        for target in targets:
            boxes_xyxy = masks_to_boxes(target["masks"].to(device))
            boxes_cxcywh = box_convert(boxes_xyxy, in_fmt="xyxy", out_fmt="cxcywh")
            boxes_cxcywh[:, 0::2] /= w
            boxes_cxcywh[:, 1::2] /= h
            targets_detr.append({
                "labels": target["labels"].to(device), 
                "boxes": boxes_cxcywh.to(device),
                "masks": target["masks"].to(device).float() 
            })
        _, losses_sup = self.student(pixel_values=pixel_values, targets=targets_detr)
        return sum(losses_sup.values())

    def _get_sup_loss_mrcnn(self, pixel_values, targets):
        targets_mrcnn = []
        device = pixel_values.device
        for target in targets:
            targets_mrcnn.append({
                "boxes": masks_to_boxes(target["masks"].to(device)).to(device),
                "labels": target["labels"].to(device),
                "masks": target["masks"].to(device).float()
            })
        losses_sup = self.student(pixel_values=pixel_values, targets=targets_mrcnn)
        return sum(losses_sup.values())
