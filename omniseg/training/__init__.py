"""Training utilities and PyTorch Lightning training module."""

import copy
import math
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union

import pytorch_lightning as pl
from pycocotools.cocoeval import COCOeval
from torchvision.ops import box_convert

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
        self.validation_step_outputs = []
        self.validation_step_losses = []

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
    
    def validation_step(self, batch, batch_idx):
        if self.hparams.head_type == 'maskrcnn': 
            self._validation_step_mrcnn(batch, batch_idx)
        elif self.hparams.head_type == 'contourformer': 
            self._validation_step_cf(batch, batch_idx)
        elif self.hparams.head_type == 'deformable_detr': 
            return self._validation_step_detr(batch, batch_idx)

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

    def _validation_step_detr(self, batch, batch_idx):
        """Validation step for Deformable DETR."""
        return {"val_loss": torch.tensor(0.0, device=self.device)}

    def _validation_step_cf(self, batch, batch_idx):
        """Validation step for ContourFormer."""
        pass

    def _validation_step_mrcnn(self, batch, batch_idx):
        """Validation step for Mask R-CNN."""
        pass

    def on_validation_epoch_end(self):
        """Called at the end of the validation epoch."""
        if self.validation_step_losses:
            avg_val_loss = torch.stack(self.validation_step_losses).mean()
            self.log("val_loss", avg_val_loss, on_epoch=True, prog_bar=True)
        self.validation_step_outputs.clear()
        self.validation_step_losses.clear()


__all__ = ['SSLSegmentationLightning', 'masks_to_boxes']
