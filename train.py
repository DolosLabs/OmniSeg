# train.py
# Semi-Supervised Instance Segmentation with Multiple Backbones and Heads (PyTorch Lightning Version)
#
# To run this script, first install the required dependencies:
# pip install transformers torch torchvision tqdm pycocotools pytorch-lightning timm scikit-image scipy
#
# Then, execute the script from your terminal using command-line arguments.
# Examples:
#    python train.py --backbone resnet --head maskrcnn
#    python train.py --backbone dino --head contourformer
#    python train.py --backbone swin --head deformable_detr --image_size 448

import os
import sys
import copy
import math
import json
import zipfile
import urllib.request
import argparse
import random
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.v2 as T
import tqdm

import pytorch_lightning as pl
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import (AutoModel, AutoConfig, DeformableDetrConfig, DeformableDetrModel, DeformableDetrImageProcessor)
from transformers.models.deformable_detr.modeling_deformable_detr import DeformableDetrSinePositionEmbedding
# --- FIX: Add specific import for BaseModelOutput ---
from transformers.modeling_outputs import BaseModelOutput


from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import FeaturePyramidNetwork, box_convert, generalized_box_iou
import timm
from collections import OrderedDict
from scipy.optimize import linear_sum_assignment
from skimage.measure import find_contours
from scipy.interpolate import interp1d

# --- Configuration ---
PROJECT_DIR = './SSL_Instance_Segmentation'
EPOCHS = 300
NUM_CLASSES = 10


# --- Step 1: COCO Dataset Utility ---
def download_coco2017(root_dir=".", splits=['train', 'val', 'test']):
    base_dir = os.path.join(root_dir, 'coco2017')
    annotations_dir = os.path.join(base_dir, 'annotations')
    class TqdmUpTo(tqdm.tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None: self.total = tsize
            self.update(b * bsize - self.n)
    def download_url(url, output_path, desc):
        with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=desc) as t:
            urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
    if not os.path.exists(annotations_dir):
        print("Downloading COCO 2017 annotations..."); os.makedirs(base_dir, exist_ok=True)
        url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        zip_path = os.path.join(base_dir, 'annotations.zip'); download_url(url, zip_path, "annotations_trainval2017.zip")
        print("Extracting annotations...");
        with zipfile.ZipFile(zip_path, 'r') as z: z.extractall(base_dir)
        os.remove(zip_path)
    else: print("COCO 2017 annotations already exist.")
    for split in splits:
        images_dir = os.path.join(base_dir, f'{split}2017')
        if not os.path.exists(images_dir):
            print(f"Downloading COCO {split}2017 images...")
            url = f"http://images.cocodataset.org/zips/{split}2017.zip"; zip_path = os.path.join(base_dir, f'{split}2017.zip')
            download_url(url, zip_path, f"{split}2017.zip"); print(f"Extracting {split}2017 images...")
            with zipfile.ZipFile(zip_path, 'r') as z: z.extractall(base_dir)
            os.remove(zip_path)
        else: print(f"COCO {split}2017 images already exist.")
    return base_dir


# --- Step 2: Model Architecture Definition ---
class DinoVisionTransformerBackbone(nn.Module):
    def __init__(self, model_name='facebook/dinov3-vitl16-pretrain-lvd1689m', feature_layer=8):
        super().__init__()
        print(f"Loading backbone: {model_name}")
        self.dino = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
        for p in self.dino.parameters(): p.requires_grad = False
        self.dino.eval(); self.feature_layer_index = feature_layer; self.num_register_tokens = self.dino.config.num_register_tokens
        embed_dim = self.dino.config.hidden_size; decoder_channels = [128, 256, 512, 1024]
        self.pyramid_layers = nn.ModuleList(); self.pyramid_layers.append(nn.Conv2d(embed_dim, decoder_channels[0], kernel_size=1))
        for i in range(1, len(decoder_channels)):
            self.pyramid_layers.append(nn.Sequential(nn.Conv2d(decoder_channels[i-1], decoder_channels[i], kernel_size=3, stride=2, padding=1), nn.ReLU()))
    def forward(self, x):
        outputs = self.dino(pixel_values=x, output_hidden_states=True); hidden_state = outputs.hidden_states[self.feature_layer_index]
        B, N, D = hidden_state.shape; tokens_to_skip = 1 + self.num_register_tokens; patch_tokens = hidden_state[:, tokens_to_skip:, :]
        seq_len_patches = N - tokens_to_skip; side_len = int(math.sqrt(seq_len_patches))
        if side_len * side_len != seq_len_patches: raise ValueError("Patch tokens do not form a perfect square.")
        fmap = patch_tokens.permute(0, 2, 1).reshape(B, D, side_len, side_len); features = {}; current_fmap = fmap
        for i, layer in enumerate(self.pyramid_layers):
            current_fmap = layer(current_fmap); features[f"res{i+2}"] = current_fmap
        return features

class SamVisionTransformerBackbone(nn.Module):
    def __init__(self, model_name='facebook/sam-vit-huge', freeze_encoder=True):
        super().__init__()
        print(f"Loading backbone: {model_name}")
        self.sam = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
        if freeze_encoder:
            for p in self.sam.parameters(): p.requires_grad = False
            self.sam.eval()
        embed_dim = self.sam.config.vision_config.hidden_size
        decoder_channels = [128, 256, 512, 1024]
        self.proj = nn.Conv2d(embed_dim, decoder_channels[0], kernel_size=1); self.pyramid_layers = nn.ModuleList()
        for i in range(1, len(decoder_channels)):
            self.pyramid_layers.append(nn.Sequential(nn.Conv2d(decoder_channels[i-1], decoder_channels[i], kernel_size=3, stride=2, padding=1), nn.ReLU()))
    def forward(self, x):
        outputs = self.sam.vision_encoder(pixel_values=x); last_hidden_state = outputs.last_hidden_state
        B, N, D = last_hidden_state.shape; side_len = int(math.sqrt(N))
        if side_len * side_len != N: raise ValueError("Patch tokens do not form a perfect square.")
        fmap = last_hidden_state.permute(0, 2, 1).reshape(B, D, side_len, side_len)
        features = {}; current_fmap = self.proj(fmap); features["res2"] = current_fmap
        for i, layer in enumerate(self.pyramid_layers):
            current_fmap = layer(current_fmap); features[f"res{i+3}"] = current_fmap
        return features

class SwinTransformerBackbone(nn.Module):
    def __init__(self, model_name='microsoft/swin-base-patch4-window7-224-in22k', freeze_encoder=True):
        super().__init__()
        print(f"Loading backbone: {model_name}")
        self.swin = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32, out_indices=(0, 1, 2, 3))
        if freeze_encoder:
            for p in self.swin.parameters(): p.requires_grad = False
            self.swin.eval()
    def forward(self, x):
        outputs = self.swin(pixel_values=x)
        return {f"res{i+2}": hs for i, hs in enumerate(outputs.hidden_states)}

class ConvNeXtBackbone(nn.Module):
    def __init__(self, model_name='convnext_base.fb_in22k', freeze_encoder=True):
        super().__init__()
        print(f"Loading backbone: {model_name}")
        self.convnext = timm.create_model(model_name, pretrained=True, features_only=True, out_indices=(0, 1, 2, 3))
        if freeze_encoder:
            for p in self.convnext.parameters(): p.requires_grad = False
            self.convnext.eval()
    def forward(self, x):
        outputs = self.convnext(x)
        return {f"res{i+2}": out for i, out in enumerate(outputs)}

class RepVGGBackbone(nn.Module):
    def __init__(self, model_name='repvgg_b0.rvgg_in1k', freeze_encoder=True):
        super().__init__()
        print(f"Loading backbone: {model_name}")
        self.repvgg = timm.create_model(model_name, pretrained=True, features_only=True, out_indices=(0, 1, 2, 3))
        if freeze_encoder:
            for p in self.repvgg.parameters(): p.requires_grad = False
            self.repvgg.eval()
    def forward(self, x):
        outputs = self.repvgg(x)
        return {f"res{i+2}": out for i, out in enumerate(outputs)}

class ResNetBackbone(nn.Module):
    def __init__(self, model_name='resnet50.a1_in1k', freeze_encoder=True):
        super().__init__()
        print(f"Loading backbone: {model_name}")
        self.resnet = timm.create_model(model_name, pretrained=True, features_only=True, out_indices=(0, 1, 2, 3))
        if freeze_encoder:
            for p in self.resnet.parameters(): p.requires_grad = False
            self.resnet.eval()
    def forward(self, x):
        outputs = self.resnet(x)
        return {f"res{i+2}": out for i, out in enumerate(outputs)}

# --- Head Architectures ---
def get_backbone(backbone_type):
    """Factory function to create a backbone instance."""
    if backbone_type == 'dino':
        return DinoVisionTransformerBackbone()
    elif backbone_type == 'sam':
        return SamVisionTransformerBackbone()
    elif backbone_type == 'swin':
        return SwinTransformerBackbone()
    elif backbone_type == 'convnext':
        return ConvNeXtBackbone()
    elif backbone_type == 'repvgg':
        return RepVGGBackbone()
    elif backbone_type == 'resnet':
        return ResNetBackbone()
    else:
        raise ValueError(f"Unsupported backbone_type: {backbone_type}")

class GenericBackboneWithFPN(nn.Module):
    def __init__(self, backbone_type='dino', fpn_out_channels=256, dummy_input_size=224):
        super().__init__()
        self.backbone = get_backbone(backbone_type)
        dummy = torch.randn(1, 3, dummy_input_size, dummy_input_size)
        with torch.no_grad():
            feat_dict = self.backbone(dummy)
        feat_keys = sorted(list(feat_dict.keys()))
        in_channels_list = [feat_dict[k].shape[1] for k in feat_keys]
        self.out_channels = fpn_out_channels
        self.fpn = FeaturePyramidNetwork(in_channels_list=in_channels_list, out_channels=self.out_channels)
        self._orig_keys = feat_keys
        self._num_levels = len(feat_keys)
    def forward(self, x):
        feats = self.backbone(x)
        ordered_input = OrderedDict()
        for k in self._orig_keys:
            if k not in feats:
                raise KeyError(f"Expected feature '{k}' from backbone but got keys: {list(feats.keys())}")
            ordered_input[k] = feats[k]
        fpn_out = self.fpn(ordered_input)
        # Remap keys to strings '0', '1', '2', ... for DETR compatibility
        remapped = OrderedDict([(str(i), val) for i, (key, val) in enumerate(fpn_out.items())])
        return remapped

class GenericMaskRCNN(nn.Module):
    def __init__(self, num_classes, backbone_type='dino', image_size=224, anchor_base_size=32):
        super().__init__()
        backbone_with_fpn = GenericBackboneWithFPN(backbone_type, dummy_input_size=image_size)
        backbone_with_fpn.out_channels = backbone_with_fpn.out_channels
        dummy = torch.randn(1, 3, image_size, image_size)
        with torch.no_grad():
            feats = backbone_with_fpn(dummy)
        num_maps = len(feats)
        if num_maps == 0:
            raise RuntimeError("Backbone+FPN returned no feature maps; expected >=1")
        sizes = tuple([(anchor_base_size * (2 ** i),) for i in range(num_maps)])
        aspect_ratios = ((0.5, 1.0, 2.0),) * num_maps
        anchor_generator = AnchorGenerator(sizes=sizes, aspect_ratios=aspect_ratios)
        self.model = MaskRCNN(backbone=backbone_with_fpn, num_classes=num_classes + 1, rpn_anchor_generator=anchor_generator)
    def forward(self, images, targets=None):
        return self.model(images, targets)


# --- START: Deformable DETR Head ---
class HungarianMatcherDETR(nn.Module):
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        out_bbox = outputs["pred_boxes"].flatten(0, 1)
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        cost_class = -out_prob[:, tgt_ids]
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_giou = -generalized_box_iou(box_convert(out_bbox, in_fmt="cxcywh", out_fmt="xyxy"), box_convert(tgt_bbox, in_fmt="cxcywh", out_fmt="xyxy"))
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()
        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

def dice_loss(inputs, targets, num_boxes):
    inputs = inputs.sigmoid().flatten(1)
    # --- FIX: Flatten the target masks to match the input masks ---
    targets = targets.flatten(1)
    
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes

class SetCriterionDETR(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, eos_coef):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)
    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx
    def loss_labels(self, outputs, targets, indices, num_boxes):
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        return {"loss_ce": loss_ce}
    def loss_boxes(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_giou = 1 - torch.diag(generalized_box_iou(box_convert(src_boxes, in_fmt="cxcywh", out_fmt="xyxy"), box_convert(target_boxes, in_fmt="cxcywh", out_fmt="xyxy")))
        return {"loss_bbox": loss_bbox.sum() / num_boxes, "loss_giou": loss_giou.sum() / num_boxes}
    def loss_masks(self, outputs, targets, indices, num_boxes):
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"][src_idx]
        target_masks = torch.cat([t['masks'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_mask = F.binary_cross_entropy_with_logits(src_masks, target_masks.to(src_masks.dtype), reduction='none').mean(dim=[1,2])
        loss_dice = dice_loss(src_masks, target_masks.to(src_masks.dtype), num_boxes)
        return {"loss_mask": loss_mask.sum() / num_boxes, "loss_dice": loss_dice}
    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        indices = self.matcher(outputs_without_aux, targets)
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / (torch.distributed.get_world_size() if torch.distributed.is_available() and torch.distributed.is_initialized() else 1), min=1).item()
        losses = {}
        for loss in ['labels', 'boxes', 'masks']:
            losses.update(getattr(self, f"loss_{loss}")(outputs, targets, indices, num_boxes))
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in ['labels', 'boxes', 'masks']:
                    l_dict = getattr(self, f"loss_{loss}")(aux_outputs, targets, indices, num_boxes)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return {k: v * self.weight_dict[k.split('_')[0]] for k, v in losses.items() if k.split('_')[0] in self.weight_dict}

class SSLDeformableDETR(nn.Module):
    def __init__(self, num_classes, backbone_type='dino', image_size=224):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = GenericBackboneWithFPN(backbone_type, fpn_out_channels=256, dummy_input_size=image_size)
        
        config = DeformableDetrConfig(num_labels=num_classes)
        self.detr = DeformableDetrModel(config)
        
        self.position_embedding = DeformableDetrSinePositionEmbedding(config.d_model // 2, normalize=True)
        
        self.class_embed = nn.Linear(config.d_model, num_classes + 1)
        self.bbox_embed = nn.Sequential(nn.Linear(config.d_model, config.d_model), nn.ReLU(), nn.Linear(config.d_model, 4))
        self.mask_head = nn.Sequential(nn.Linear(config.d_model, config.d_model), nn.ReLU(), nn.Linear(config.d_model, 256))
        self.mask_out_stride = 8
        self.mask_pred_stride = 4 # From the F.interpolate in the forward pass
        
        matcher = HungarianMatcherDETR(cost_class=config.class_cost, cost_bbox=config.bbox_cost, cost_giou=config.giou_cost)
        weight_dict = {"loss_ce": 1.0, "loss_bbox": config.bbox_loss_coefficient, "loss_giou": config.giou_loss_coefficient, "loss_mask": 1.0, "loss_dice": 1.0}
        self.criterion = SetCriterionDETR(num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=config.eos_coefficient)

    def forward(self, pixel_values, pixel_mask=None, targets=None):
        # 1. Run the backbone and FPN ONCE to get all feature maps.
        feature_maps = self.backbone(pixel_values)
        
        # The DETR model from Hugging Face expects a list of tensors, not a dictionary.
        features_list = list(feature_maps.values())

        # --- FIX START ---
        # 2. Create a default pixel_mask if one isn't provided.
        # The model requires this mask when multi-scale features are passed as a list.
        if pixel_mask is None:
            batch_size, _, height, width = pixel_values.shape
            pixel_mask = torch.ones((batch_size, height, width), dtype=torch.long, device=pixel_values.device)
        # --- FIX END ---

        # 3. Pass the feature maps AND the mask to the DETR transformer.
        outputs = self.detr(pixel_values=pixel_values, pixel_mask=pixel_mask)
    
        decoder_output = outputs.last_hidden_state
        
        # Prediction heads
        logits = self.class_embed(decoder_output)
        pred_boxes = self.bbox_embed(decoder_output).sigmoid()
        mask_embeds = self.mask_head(decoder_output)
    
        # Reuse the finest feature map for mask prediction
        fpn_finest = feature_maps['0'] 
        B, C, H, W = fpn_finest.shape
        pred_masks = (mask_embeds @ fpn_finest.view(B, C, H * W)).view(B, -1, H, W)
    
        outputs = {
            "pred_logits": logits,
            "pred_boxes": pred_boxes,
            "pred_masks": pred_masks,
            "aux_outputs": [],
        }
        
        if targets is not None:
            losses = self.criterion(outputs, targets)
            return outputs, losses
        return outputs

# --- END: Deformable DETR Head ---


# --- START: ContourFormer Head ---
# ... (rest of ContourFormer code remains unchanged)
# ------------------------
# Simple bilinear sampler helper (for grid_sample)
# ------------------------
def bilinear_sample(feat, coords):
    # feat: [N, C, H, W], coords: [N, L, 2] in [-1,1]
    N, C, H, W = feat.shape
    L = coords.shape[1]
    grid = coords.view(N, 1, L, 2)
    sampled = F.grid_sample(feat, grid, align_corners=True, mode='bilinear')
    sampled = sampled.view(N, C, L).permute(0, 2, 1)
    return sampled


# ------------------------
# Deformable-style cross-attention (single-level)
# ------------------------
class DeformableCrossAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_points):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_points = n_points
        self.head_dim = d_model // n_heads
        self.to_q = nn.Linear(d_model, d_model)
        self.to_kv = nn.Linear(d_model, 2 * d_model)
        self.offset_mlp = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim),
            nn.ReLU(),
            nn.Linear(self.head_dim, n_points * 2)
        )
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, query, memory, reference_points, spatial_shape):
        N, Len_q, _ = query.shape
        q = self.to_q(query)
        kv = self.to_kv(memory)
        k, v = kv.chunk(2, dim=-1)
        q = q.view(N, Len_q, self.n_heads, self.head_dim)
        k = k.view(N, -1, self.n_heads, self.head_dim)
        v = v.view(N, -1, self.n_heads, self.head_dim)
        offsets = self.offset_mlp(q)
        offsets = offsets.view(N, Len_q, self.n_heads, self.n_points, 2)
        if reference_points.dim() == 3:
            reference_points = reference_points.unsqueeze(2).unsqueeze(3)
        elif reference_points.dim() == 4:
            reference_points = reference_points.unsqueeze(3)
        reference_points = reference_points.expand(-1, -1, self.n_heads, self.n_points, -1)
        sampling_locations = reference_points + offsets.tanh()
        q = q.unsqueeze(3).expand(-1, -1, -1, self.n_points, -1)
        q = q.contiguous().view(N, Len_q * self.n_heads * self.n_points, self.head_dim)
        v = v.contiguous().view(N, -1, self.head_dim)
        attn_output = q
        attn_output = attn_output.view(N, Len_q, self.n_heads, self.n_points, self.head_dim)
        attn_output = attn_output.mean(dim=3)
        attn_output = attn_output.contiguous().view(N, Len_q, self.d_model)
        attn_output = self.output_proj(attn_output)
        return attn_output


# ------------------------
# Deformable decoder layer + stack
# ------------------------
class DeformableDecoderLayer(nn.Module):
    def __init__(self, d_model=256, n_heads=8, n_points=4, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = DeformableCrossAttention(d_model, n_heads, n_points)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory_feats, reference_points, spatial_shape):
        q = k = tgt
        tgt2, _ = self.self_attn(q, k, value=tgt)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.cross_attn(tgt, memory_feats, reference_points, spatial_shape)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class SimpleDeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nheads=8, num_decoder_layers=6, n_points=4, dim_feedforward=2048):
        super().__init__()
        self.decoder_layers = nn.ModuleList(
            [DeformableDecoderLayer(d_model, nheads, n_points, dim_feedforward) for _ in range(num_decoder_layers)]
        )

    def forward(self, query_embed, memory_feats, reference_points, spatial_shape):
        hs = []
        tgt = query_embed
        for layer in self.decoder_layers:
            tgt = layer(tgt, memory_feats, reference_points, spatial_shape)
            hs.append(tgt)
        return torch.stack(hs)


# ------------------------
# Hungarian matcher (with annealing / point-only warmup)
# ------------------------
class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=1, cost_point=1, cost_giou=1, cost_point_init=10, cost_decay=0.95, warmup_epochs=10):
        super().__init__()
        self.base_cost_class = cost_class
        self.base_cost_point = cost_point
        self.cost_giou = cost_giou
        self.cost_point_init = cost_point_init
        self.cost_decay = cost_decay
        self.epoch = 0
        self.warmup_epochs = warmup_epochs

    def step_epoch(self):
        self.epoch += 1

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        out_prob = (outputs["pred_logits"].flatten(0, 1) / 0.7).softmax(-1)
        out_points = outputs["pred_coords"].flatten(0, 1)
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_points = torch.cat([v["coords"] for v in targets])
        cost_class = -out_prob[:, tgt_ids]
        cost_point = torch.cdist(out_points.flatten(1), tgt_points.flatten(1), p=1)
        if self.epoch < self.warmup_epochs:
            C = cost_point
        else:
            cost_point_w = max(self.base_cost_point, self.cost_point_init * (self.cost_decay ** self.epoch))
            C = self.base_cost_class * cost_class + cost_point_w * cost_point
        C = C.view(bs, num_queries, -1).cpu()
        sizes = [len(v["coords"]) for v in targets]
        indices = []
        for i, c in enumerate(C.split(sizes, -1)):
            row_ind, col_ind = linear_sum_assignment(c[i].numpy())
            indices.append((torch.as_tensor(row_ind, dtype=torch.int64), torch.as_tensor(col_ind, dtype=torch.int64)))
        return indices


# ------------------------
# Criterion with normalized point loss and deep supervision support
# ------------------------
class SetCriterionContourFormer(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, num_points):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.num_points = num_points
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_instances):
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        if len(idx[0]) == 0:
            target_classes_o = torch.empty(0, dtype=torch.int64, device=src_logits.device)
        else:
            target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, weight=self.empty_weight.to(src_logits.device))
        return {'loss_ce': loss_ce}

    def loss_points(self, outputs, targets, indices, num_instances):
        idx = self._get_src_permutation_idx(indices)
        if len(idx[0]) == 0:
            return {'loss_point': torch.tensor(0., device=next(iter(outputs.values())).device)}
        src_points = outputs['pred_coords'][idx]
        target_points = torch.cat([t['coords'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_point = F.l1_loss(src_points, target_points, reduction='sum') / (num_instances * self.num_points)
        return {'loss_point': loss_point}

    def _get_src_permutation_idx(self, indices):
        if len(indices) == 0:
            return (torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64))
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def forward(self, outputs, targets, indices=None):
        if indices is None:
            indices = self.matcher(outputs, targets)
        num_instances = sum(len(t["labels"]) for t in targets)
        num_instances = torch.as_tensor([num_instances], dtype=torch.float, device=next(iter(outputs.values())).device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(num_instances)
            world_size = torch.distributed.get_world_size()
        else:
            world_size = 1
        num_instances = torch.clamp(num_instances / world_size, min=1).item()
        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices, num_instances))
        losses.update(self.loss_points(outputs, targets, indices, num_instances))
        return {k: v * self.weight_dict.get(k, 1.0) for k, v in losses.items()}


# ------------------------
# SSLContourFormer using the SimpleDeformableTransformer
# ------------------------
class SSLContourFormer(nn.Module):
    def __init__(self, num_classes, backbone_type='dino', image_size=224,
                 num_queries=100, num_points=50, hidden_dim=256, nheads=8,
                 num_decoder_layers=6, n_points=4):
        super().__init__()
        self.num_queries = num_queries
        self.num_points = num_points
        self.hidden_dim = hidden_dim
        self.backbone = GenericBackboneWithFPN(backbone_type, fpn_out_channels=hidden_dim, dummy_input_size=image_size)
        self.input_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.ref_point_embed = nn.Embedding(num_queries, 2)
        self.transformer = SimpleDeformableTransformer(d_model=hidden_dim, nheads=nheads,
                                                       num_decoder_layers=num_decoder_layers, n_points=n_points)
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.coord_embed = nn.Linear(hidden_dim, num_points * 2)
        matcher = HungarianMatcher(cost_class=1, cost_point=5, cost_giou=2, cost_point_init=10, cost_decay=0.95, warmup_epochs=10)
        weight_dict = {'loss_ce': 1.0, 'loss_point': 1.0}
        self.criterion = SetCriterionContourFormer(num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=0.1, num_points=num_points)

    def build_position_encoding(self, feature):
        B, C, H, W = feature.shape
        mask = torch.ones(B, H, W, device=feature.device)
        y_embed = mask.cumsum(1, dtype=torch.float32)
        x_embed = mask.cumsum(2, dtype=torch.float32)
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * 2 * math.pi
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * 2 * math.pi
        dim_t = torch.arange(C // 2, device=feature.device)
        dim_t = 10000 ** (2 * (dim_t // 2) / (C // 2))
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

    def forward(self, pixel_values, targets=None):
        features = self.backbone(pixel_values)
        src = self.input_proj(features['0'])
        B, C, H, W = src.shape
        pos_embed = self.build_position_encoding(src)
        src_with_pos = src + pos_embed
        query_input = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)
        ref_points = self.ref_point_embed.weight.unsqueeze(0).repeat(B, 1, 1).sigmoid()
        hs = self.transformer(query_input, src, ref_points, (H, W))
        outputs_classes = [self.class_embed(h) for h in hs]
        outputs_coords = [self.coord_embed(h).sigmoid().view(B, self.num_queries, self.num_points, 2) for h in hs]
        out = {'pred_logits': outputs_classes[-1], 'pred_coords': outputs_coords[-1]}
        if self.training and targets is not None:
            indices = self.criterion.matcher(out, targets)
            losses = self.criterion(out, targets, indices=indices)
            for i, (c, p) in enumerate(zip(outputs_classes[:-1], outputs_coords[:-1])):
                aux_out = {'pred_logits': c, 'pred_coords': p}
                aux_losses = self.criterion(aux_out, targets, indices=indices)
                for k, v in aux_losses.items():
                    losses[f"{k}_aux{i}"] = v
            return out, losses
        return out
# --- END: ContourFormer Head ---


# --- Step 3: Data Handling ---
# --- START: ContourFormer Utilities ---
def masks_to_contours(masks, num_points=50):
    """Convert binary masks to fixed-size contour point sets."""
    contours_list, valid_indices = [], []
    for i, mask in enumerate(masks):
        if mask.sum() == 0: continue
        mask_np = mask.cpu().numpy().astype(np.uint8)
        contours = find_contours(mask_np, 0.5)
        if not contours: continue
        sorted_contours = sorted(contours, key=lambda x: len(x), reverse=True)
        contour = sorted_contours[0]
        if len(contour) < 4: continue
        contour = np.flip(contour, axis=1).astype(np.float64)
        if not np.allclose(contour[0], contour[-1], atol=1e-6):
            contour = np.vstack([contour, contour[0:1]])
        contour = torch.tensor(contour, dtype=torch.float32)
        contour[:, 0] = torch.clamp(contour[:, 0] / max(mask.shape[1], 1), 0.0, 1.0)
        contour[:, 1] = torch.clamp(contour[:, 1] / max(mask.shape[0], 1), 0.0, 1.0)
        path = contour.t()
        diffs = path[:, 1:] - path[:, :-1]
        distances = torch.sqrt(torch.sum(diffs**2, dim=0))
        distance = torch.cumsum(distances, dim=0)
        distance = torch.cat([torch.tensor([0.0]), distance])
        if distance[-1] < 1e-6: continue
        try:
            f = interp1d(distance.numpy(), path.numpy(), kind='linear', axis=1, fill_value="extrapolate")
            new_distances = np.linspace(0, distance[-1].item(), num_points)
            sampled_points = torch.from_numpy(f(new_distances).T).to(torch.float32)
        except:
            indices = torch.linspace(0, len(contour)-1, num_points).long()
            indices = torch.clamp(indices, 0, len(contour)-1)
            sampled_points = contour[indices]
        sampled_points = torch.clamp(sampled_points, 0.0, 1.0)
        if torch.isnan(sampled_points).any() or torch.isinf(sampled_points).any(): continue
        contours_list.append(sampled_points)
        valid_indices.append(i)
    if not contours_list: return None, None
    return torch.stack(contours_list), valid_indices

def contours_to_masks(contours, img_shape):
    """Convert sets of contour points to binary masks."""
    h, w = img_shape
    masks = []
    for contour_set in contours:
        if contour_set.numel() == 0:
            masks.append(torch.zeros(h, w, dtype=torch.bool)); continue
        points = contour_set.clone()
        points[:, 0] = torch.clamp(points[:, 0] * w, 0, w - 0.01)
        points[:, 1] = torch.clamp(points[:, 1] * h, 0, h - 0.01)
        points_np = points.cpu().numpy()
        if len(points_np) <= 10 or np.ptp(points_np[:, 0]) < 2 or np.ptp(points_np[:, 1]) < 2:
            points_rounded = points_np
            if np.ptp(points_np[:, 0]) < 1:
                center_x = np.mean(points_np[:, 0])
                points_rounded[:, 0] = np.linspace(center_x - 1, center_x + 1, len(points_np))
            if np.ptp(points_np[:, 1]) < 1:
                center_y = np.mean(points_np[:, 1])
                points_rounded[:, 1] = np.linspace(center_y - 1, center_y + 1, len(points_np))
        else:
            points_rounded = np.round(points_np).astype(np.float64)
        unique_points = [points_rounded[0]]
        for point in points_rounded[1:]:
            if not np.allclose(point, unique_points[-1], atol=0.5): unique_points.append(point)
        if len(unique_points) >= 3 and not np.allclose(unique_points[0], unique_points[-1], atol=0.5):
            unique_points.append(unique_points[0])
        if len(unique_points) < 3:
            center_x, center_y = np.mean(points_np[:, 0]), np.mean(points_np[:, 1])
            unique_points = [[center_x - 1, center_y - 1],[center_x + 1, center_y - 1],[center_x + 1, center_y + 1],[center_x - 1, center_y + 1],[center_x - 1, center_y - 1]]
        points_flat = np.array(unique_points).flatten().tolist()
        try:
            img = Image.new('L', (w, h), 0)
            ImageDraw.Draw(img).polygon(points_flat, outline=1, fill=1)
            masks.append(torch.from_numpy(np.array(img)).bool())
        except Exception as e:
            print(f"Warning: Error drawing polygon: {e}")
            center_x, center_y = int(np.mean(points_np[:, 0])), int(np.mean(points_np[:, 1]))
            mask = torch.zeros(h, w, dtype=torch.bool)
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    x, y = center_x + dx, center_y + dy
                    if 0 <= x < w and 0 <= y < h: mask[y, x] = True
            masks.append(mask)
    return torch.stack(masks) if masks else torch.empty(0, h, w, dtype=torch.bool)
# --- END: ContourFormer Utilities ---

def masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
    if masks.numel() == 0: return torch.zeros((0, 4), dtype=torch.float32)
    n = masks.shape[0]
    boxes = torch.zeros((n, 4), dtype=torch.float32, device=masks.device)
    for i in range(n):
        y, x = torch.where(masks[i])
        if y.numel() > 0 and x.numel() > 0:
            x1, y1 = torch.min(x), torch.min(y)
            x2, y2 = torch.max(x), torch.max(y)
            if x1 == x2: x2 = x1 + 1
            if y1 == y2: y2 = y1 + 1
            boxes[i, 0], boxes[i, 1] = x1, y1
            boxes[i, 2], boxes[i, 3] = x2, y2
    return boxes

def get_transforms(augment=False, image_size=224):
    transforms = []
    if augment:
        transforms.append(T.RandomHorizontalFlip(p=0.5))
        transforms.append(T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1))
    transforms.append(T.Resize((image_size, image_size), antialias=True))
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return T.Compose(transforms)

class SemiCOCODataset(Dataset):
    def __init__(self, images_dir, ann_file=None, is_unlabeled=False, num_images=-1):
        self.images_dir, self.is_unlabeled = images_dir, is_unlabeled
        if self.is_unlabeled:
            self.img_files = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith('.jpg')]
            if num_images > 0 and len(self.img_files) > num_images:
                random.seed(42); random.shuffle(self.img_files)
                self.img_files = self.img_files[:num_images]
                print(f"--- Using a random subset of {len(self.img_files)} unlabeled images for training. ---")
        else:
            self.coco = COCO(ann_file); all_img_ids = sorted(self.coco.getImgIds())
            cat_ids = self.coco.getCatIds()
            self.cat2label = {cat_id: i for i, cat_id in enumerate(cat_ids)}
            self.label2cat = {v: k for k, v in self.cat2label.items()}
            self.img_ids = [img_id for img_id in all_img_ids if len(self.coco.getAnnIds(imgIds=img_id, iscrowd=False)) > 0]
            if num_images > 0 and len(self.img_ids) > num_images:
                random.seed(42); random.shuffle(self.img_ids)
                self.img_ids = self.img_ids[:num_images]
                print(f"--- Using a random subset of {len(self.img_ids)} labeled images for training. ---")
    def __len__(self):
        return len(self.img_files) if self.is_unlabeled else len(self.img_ids)
    def __getitem__(self, idx):
        if self.is_unlabeled:
            return Image.open(self.img_files[idx]).convert("RGB"), {}
        img_id = self.img_ids[idx]; img_info = self.coco.loadImgs(img_id)[0]
        image = Image.open(os.path.join(self.images_dir, img_info['file_name'])).convert("RGB")
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False); anns = self.coco.loadAnns(ann_ids)
        masks, labels = [], []
        for ann in anns:
            mask = self.coco.annToMask(ann)
            if mask.sum() > 0:
                masks.append(mask); labels.append(self.cat2label[ann['category_id']])
        if not masks: return self.__getitem__((idx + 1) % len(self))
        target = {"image_id": img_id, "labels": torch.tensor(labels, dtype=torch.int64), "masks": torch.tensor(np.array(masks), dtype=torch.float32)}
        return image, target

class COCODataModule(pl.LightningDataModule):
    def __init__(self, project_dir, batch_size=32, num_workers=2, image_size=224, num_labeled_images=-1, num_unlabeled_images=-1):
        super().__init__()
        self.project_dir, self.batch_size, self.num_workers, self.image_size = project_dir, batch_size, num_workers, image_size
        self.num_labeled_images, self.num_unlabeled_images = num_labeled_images, num_unlabeled_images
        self.train_aug = get_transforms(augment=True, image_size=self.image_size)
        self.val_aug = get_transforms(augment=False, image_size=self.image_size)
    def prepare_data(self): download_coco2017(root_dir=self.project_dir, splits=['train', 'val', 'test'])
    def setup(self, stage=None):
        base_dir = os.path.join(self.project_dir, 'coco2017'); train_images_dir = os.path.join(base_dir, 'train2017'); val_images_dir = os.path.join(base_dir, 'val2017'); test_images_dir = os.path.join(base_dir, 'test2017')
        train_ann_file = os.path.join(base_dir, 'annotations', 'instances_train2017.json'); val_ann_file = os.path.join(base_dir, 'annotations', 'instances_val2017.json')
        self.labeled_ds = SemiCOCODataset(train_images_dir, train_ann_file, num_images=self.num_labeled_images)
        self.unlabeled_ds = SemiCOCODataset(test_images_dir, is_unlabeled=True, num_images=self.num_unlabeled_images)
        self.val_ds = SemiCOCODataset(val_images_dir, val_ann_file)
        self.cat2label, self.label2cat, self.coco_gt_val = self.labeled_ds.cat2label, self.labeled_ds.label2cat, self.val_ds.coco
    def train_dataloader(self):
        labeled_loader = DataLoader(self.labeled_ds, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn, num_workers=self.num_workers, pin_memory=True, persistent_workers=False)
        unlabeled_loader = DataLoader(self.unlabeled_ds, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn, num_workers=self.num_workers, pin_memory=True, persistent_workers=False)
        return CombinedLoader({"labeled": labeled_loader, "unlabeled": unlabeled_loader}, mode="max_size_cycle")
    def val_dataloader(self): return DataLoader(self.val_ds, batch_size=1, shuffle=False, collate_fn=self.collate_fn, num_workers=self.num_workers, persistent_workers=False)
    def collate_fn(self, batch): return tuple(zip(*batch))


# --- Step 4: PyTorch Lightning Module ---
class SSLSegmentationLightning(pl.LightningModule):
    def __init__(self, num_classes=80, lr=1e-4, ema_decay=0.999, backbone_type='dino', head_type='maskrcnn', image_size=224, warmup_steps=500, unsup_rampup_steps=5000):
        super().__init__()
        self.save_hyperparameters()
        if self.hparams.head_type == 'maskrcnn':
            self.student = GenericMaskRCNN(num_classes, backbone_type=self.hparams.backbone_type, image_size=self.hparams.image_size)
        elif self.hparams.head_type == 'contourformer':
             self.student = SSLContourFormer(num_classes, backbone_type=self.hparams.backbone_type, image_size=self.hparams.image_size,hidden_dim=32)
        elif self.hparams.head_type == 'deformable_detr':
             self.student = SSLDeformableDETR(num_classes, backbone_type=self.hparams.backbone_type, image_size=self.hparams.image_size)
        else: raise ValueError(f"Unsupported head_type: {self.hparams.head_type}")
        self.teacher = copy.deepcopy(self.student)
        for p in self.teacher.parameters(): p.requires_grad = False
        self.teacher.eval()
        if self.hparams.head_type == 'deformable_detr':
            self.detr_image_processor = DeformableDetrImageProcessor.from_pretrained("SenseTime/deformable-detr")

        self.train_aug = get_transforms(augment=True, image_size=self.hparams.image_size)
        self.val_aug = get_transforms(augment=False, image_size=self.hparams.image_size)
        self.validation_step_outputs = []; self.validation_step_losses = []
    def configure_optimizers(self): return torch.optim.AdamW([p for p in self.student.parameters() if p.requires_grad], lr=self.hparams.lr)
    @torch.no_grad()
    def _update_teacher(self):
        for t_param, s_param in zip(self.teacher.parameters(), self.student.parameters()):
            t_param.data.mul_(self.hparams.ema_decay).add_(s_param.data, alpha=1 - self.hparams.ema_decay)
    def on_before_optimizer_step(self, optimizer): self._update_teacher()
    
    def training_step(self, batch, batch_idx):
        if self.hparams.head_type == 'maskrcnn': return self._training_step_mrcnn(batch, batch_idx)
        elif self.hparams.head_type == 'contourformer': return self._training_step_cf(batch, batch_idx)
        elif self.hparams.head_type == 'deformable_detr': return self._training_step_detr(batch, batch_idx)
    
    def validation_step(self, batch, batch_idx):
        if self.hparams.head_type == 'maskrcnn': self._validation_step_mrcnn(batch, batch_idx)
        elif self.hparams.head_type == 'contourformer': self._validation_step_cf(batch, batch_idx)
        elif self.hparams.head_type == 'deformable_detr': return self._validation_step_detr(batch, batch_idx)

    def _training_step_detr(self, batch, batch_idx):
        # ---------------------------
        # 1. Labeled data (supervised)
        # ---------------------------
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
    
        # Ensure loss_sup is always a tensor
        if isinstance(losses_sup, dict) and losses_sup:
            loss_sup = sum(losses_sup.values())
            if not torch.is_tensor(loss_sup):
                loss_sup = torch.tensor(loss_sup, device=self.device, requires_grad=True)
        else:
            loss_sup = torch.tensor(0.0, device=self.device, requires_grad=True)
    
        # ---------------------------
        # 2. Warmup handling
        # ---------------------------
        if self.global_step < self.hparams.warmup_steps:
            self.log_dict({
                "train_loss": loss_sup,
                "sup_loss": loss_sup,
                "unsup_loss": torch.tensor(0.0, device=self.device, dtype=loss_sup.dtype)
            }, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            return {"loss": loss_sup}
    
        # ---------------------------
        # 3. Unlabeled data (unsupervised)
        # ---------------------------
        images_u, _ = batch["unlabeled"]
        images_u_weak = torch.stack([self.val_aug(img, None)[0] for img in images_u]).to(device=self.device, dtype=self.dtype)
    
        with torch.no_grad():
            teacher_preds = self.teacher(pixel_values=images_u_weak)
    
        pseudo_keep_threshold = 0.5
        student_input_images, pseudo_targets = [], []
    
        for i, img in enumerate(images_u):
            scores = teacher_preds['pred_logits'][i].softmax(-1)
            max_scores, labels = scores[..., :-1].max(-1)
            keep = max_scores > pseudo_keep_threshold
            if keep.sum() > 0:
                pseudo_target = {
                    "labels": labels[keep],
                    "boxes": teacher_preds['pred_boxes'][i][keep]
                }
                student_img, _ = self.train_aug(img, None)
                student_input_images.append(student_img)
                pseudo_targets.append({k: v.to(self.device) for k, v in pseudo_target.items()})
    
        # Default unsupervised loss tensor
        loss_unsup = torch.tensor(0.0, device=self.device, dtype=loss_sup.dtype, requires_grad=True)
    
        if student_input_images:
            pixel_values_u = torch.stack(student_input_images).to(device=self.device, dtype=self.dtype)
    
            # Provide dummy masks to match DETR input
            mask_h = self.hparams.image_size // 4
            mask_w = self.hparams.image_size // 4
            _, losses_unsup = self.student(
                pixel_values=pixel_values_u,
                targets=[{
                    **t,
                    'masks': torch.zeros(t['labels'].shape[0], mask_h, mask_w, device=self.device)
                } for t in pseudo_targets]
            )
            if losses_unsup:
                loss_unsup = sum(losses_unsup.values())
                if not torch.is_tensor(loss_unsup):
                    loss_unsup = torch.tensor(loss_unsup, device=self.device, dtype=loss_sup.dtype, requires_grad=True)
    
        # ---------------------------
        # 4. Combine losses with ramp-up
        # ---------------------------
        steps_after_warmup = self.global_step - self.hparams.warmup_steps
        unsup_weight = min(1.0, steps_after_warmup / self.hparams.unsup_rampup_steps)
        total_loss = loss_sup + unsup_weight * loss_unsup
    
        # ---------------------------
        # 5. Logging
        # ---------------------------
        self.log_dict({
            "train_loss": total_loss,
            "sup_loss": loss_sup,
            "unsup_loss": loss_unsup,
            "unsup_w": unsup_weight
        }, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
    
        # ---------------------------
        # 6. Return for Lightning
        # ---------------------------
        return {"loss": total_loss}



    def _validation_step_detr(self, batch, batch_idx):
        images, targets = batch
    
        # apply validation transforms + batch
        pixel_values = torch.stack([
            self.val_aug(img) for img in images
        ]).to(device=self.device, dtype=self.dtype)
    
        # forward pass
        outputs = self.student(pixel_values)
    
        # --- Resize pred masks to target mask size ---
        # DETR-style mask heads usually produce (B, num_queries, H_pred, W_pred)
        pred_masks = outputs["pred_masks"]
    
        # ensure correct shape for interpolate: (B*num_queries, 1, H_pred, W_pred)
        B, Q, H, W = pred_masks.shape
        pred_masks = pred_masks.reshape(B * Q, 1, H, W)
    
        # resize to ground truth resolution
        target_size = targets[0]['masks'].shape[-2:]  # (H, W)
        pred_masks = F.interpolate(
            pred_masks, size=target_size, mode="bilinear", align_corners=False
        )
    
        # back to (B, Q, H, W)
        pred_masks = pred_masks.reshape(B, Q, *target_size)
    
        # extract target info
        orig_sizes = [tuple(t['masks'].shape[1:]) for t in targets]  # (h, w)
        img_ids = [t['image_id'] for t in targets]
    
        # return what your criterion / evaluator needs
        return {
            "outputs": outputs,
            "targets": targets,
            "pred_masks": pred_masks,
            "orig_sizes": orig_sizes,
            "img_ids": img_ids,
        }



    
    def _training_step_cf(self, batch, batch_idx):
        images_l, targets_l = batch["labeled"]
        images_l_strong, targets_l_cf = [], []
        for img, target in zip(images_l, targets_l):
            contours, valid_indices = masks_to_contours(target['masks'], num_points=self.student.num_points)
            if contours is None: continue
            target_cf = {'labels': target['labels'][valid_indices], 'coords': contours}
            img_aug, _ = self.train_aug(img, None)
            coords_aug = target_cf['coords'].clone()
            if random.random() < 0.5: coords_aug[:, :, 0] = 1 - coords_aug[:, :, 0]
            images_l_strong.append(img_aug)
            targets_l_cf.append({k: v.to(self.device) for k, v in {'labels': target_cf['labels'], 'coords': coords_aug}.items()})
        loss_sup = torch.tensor(0.0, device=self.device)
        if images_l_strong:
            pixel_values_l = torch.stack(images_l_strong).to(device=self.device, dtype=self.dtype)
            _, losses_sup = self.student(pixel_values=pixel_values_l, targets=targets_l_cf)
            loss_sup = sum(losses_sup.values())

        if self.global_step < self.hparams.warmup_steps:
            self.log_dict({"train_loss": loss_sup, "sup_loss": loss_sup, "unsup_loss": 0}, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=len(images_l), sync_dist=True)
            return loss_sup
        images_u, _ = batch["unlabeled"]
        images_u_weak = torch.stack([self.val_aug(img, None)[0] for img in images_u]).to(device=self.device, dtype=self.dtype)
        with torch.no_grad(): teacher_preds = self.teacher(pixel_values=images_u_weak)
        pseudo_keep_threshold = min(0.7, 0.2 + (self.global_step - self.hparams.warmup_steps) / 20000)
        student_input_images, pseudo_targets = [], []
        scores = teacher_preds['pred_logits'].softmax(-1)
        max_scores, labels = scores[..., :-1].max(-1)
        for i in range(len(images_u)):
            keep = max_scores[i] > pseudo_keep_threshold
            if keep.sum() == 0: continue
            pseudo_target = {'labels': labels[i][keep], 'coords': teacher_preds['pred_coords'][i][keep]}
            img_aug, _ = self.train_aug(images_u[i], None)
            coords_aug = pseudo_target['coords'].clone()
            if random.random() < 0.5: coords_aug[:, :, 0] = 1 - coords_aug[:, :, 0]
            student_input_images.append(img_aug)
            pseudo_targets.append({k: v.to(self.device) for k, v in {'labels': pseudo_target['labels'], 'coords': coords_aug}.items()})
        loss_unsup = torch.tensor(0.0, device=self.device)
        if student_input_images:
            pixel_values_u = torch.stack(student_input_images).to(device=self.device, dtype=self.dtype)
            _, losses_unsup = self.student(pixel_values=pixel_values_u, targets=pseudo_targets)
            loss_unsup = sum(losses_unsup.values())

        steps_after_warmup = self.global_step - self.hparams.warmup_steps
        unsup_weight = min(1.0, steps_after_warmup / self.hparams.unsup_rampup_steps)
        total_loss = loss_sup + unsup_weight * loss_unsup
        self.log_dict({"train_loss": total_loss, "sup_loss": loss_sup, "unsup_loss": loss_unsup, "unsup_w": unsup_weight}, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=len(images_l), sync_dist=True)
        return total_loss

    def _validation_step_cf(self, batch, batch_idx):
        images, targets = batch; image, target = images[0], targets[0]
        pixel_values = self.val_aug(image, None)[0].unsqueeze(0).to(device=self.device, dtype=self.dtype)
        outputs = self.student(pixel_values=pixel_values)
        with torch.no_grad():
            contours, valid_indices = masks_to_contours(target['masks'], num_points=self.student.num_points)
            if contours is not None:
                target_cf = [{'labels': target['labels'][valid_indices].to(self.device), 'coords': contours.to(self.device)}]
                outputs_for_loss = {k: v.float() for k, v in outputs.items() if 'pred' in k}
                loss_dict = self.student.criterion(outputs_for_loss, target_cf)
                val_loss = sum(loss_dict.values())
                self.validation_step_losses.append(val_loss)

        scores = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        max_scores, labels = scores.max(-1)
        keep = max_scores > 0.05
        pred_coords = outputs['pred_coords'][0][keep]; pred_labels = labels[keep]; pred_scores = max_scores[keep]
        img_info = self.trainer.datamodule.coco_gt_val.loadImgs(target['image_id'])[0]
        original_size = (img_info['height'], img_info['width'])
        if pred_coords.shape[0] > 0:
            pred_masks = contours_to_masks(pred_coords, (self.hparams.image_size, self.hparams.image_size))
            pred_masks = F.interpolate(pred_masks.unsqueeze(1).float(), size=original_size, mode='bilinear', align_corners=False)[:, 0].bool()
        else: pred_masks = torch.empty(0, *original_size, dtype=torch.bool)
        coco_formatted_results = []
        for i in range(len(pred_scores)):
            category_id = self.trainer.datamodule.label2cat[pred_labels[i].item()]
            mask = pred_masks[i].cpu().numpy(); rle = mask_utils.encode(np.asfortranarray(mask))
            rle['counts'] = rle['counts'].decode('utf-8')
            coco_formatted_results.append({"image_id": target['image_id'], "category_id": category_id, "segmentation": rle, "score": pred_scores[i].item()})
        self.validation_step_outputs.append(coco_formatted_results)

    def _training_step_mrcnn(self, batch, batch_idx):
        images_l, targets_l = batch["labeled"]; images_l_aug, targets_l_aug = [], []
        for img, target in zip(images_l, targets_l):
            target_mrcnn = {"boxes": masks_to_boxes(target["masks"]), "labels": target["labels"] + 1, "masks": target["masks"].to(torch.uint8)}
            img_aug, target_aug = self.train_aug(img, target_mrcnn)
            images_l_aug.append(img_aug.to(device=self.device, dtype=self.dtype)); targets_l_aug.append({k: v.to(self.device) for k, v in target_aug.items()})
        loss_dict_sup = self.student(images_l_aug, targets_l_aug); loss_sup = sum(loss for loss in loss_dict_sup.values())
        if self.global_step < self.hparams.warmup_steps:
            self.log_dict({"train_loss": loss_sup, "sup_loss": loss_sup, "unsup_loss": 0}, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=len(images_l), sync_dist=True)
            return loss_sup
        images_u, _ = batch["unlabeled"]; device = next(self.student.parameters()).device; dtype = next(self.student.parameters()).dtype
        images_u_weak_aug = [self.val_aug(img, None)[0].to(device=device, dtype=dtype) for img in images_u]
        with torch.no_grad(): teacher_preds = self.teacher(images_u_weak_aug)
        pseudo_keep_threshold = min(0.7, 0.2 + (self.global_step - self.hparams.warmup_steps) / 20000)
        images_u_strong, pseudo_targets_strong = [], []
        for i, pred in enumerate(teacher_preds):
            if "scores" in pred and pred["scores"].numel() > 0:
                keep = pred["scores"] > pseudo_keep_threshold
                if keep.sum() == 0: continue
                pseudo_target = {"boxes": pred["boxes"][keep], "labels": pred["labels"][keep], "masks": (pred["masks"][keep, 0] > 0.5).to(torch.uint8)}
                if pseudo_target["labels"].numel() > 0:
                    img_strong, target_strong = self.train_aug(images_u[i], pseudo_target)
                    images_u_strong.append(img_strong.to(device=device, dtype=dtype)); pseudo_targets_strong.append({k: v.to(device=device, dtype=v.dtype) for k, v in target_strong.items()})
        loss_unsup = torch.tensor(0.0, device=device)
        if images_u_strong:
            loss_dict_unsup = self.student(images_u_strong, pseudo_targets_strong)
            loss_unsup = sum(loss for loss in loss_dict_unsup.values())
        steps_after_warmup = self.global_step - self.hparams.warmup_steps; unsup_weight = min(1.0, steps_after_warmup / self.hparams.unsup_rampup_steps)
        total_loss = loss_sup + unsup_weight * loss_unsup
        self.log_dict({"train_loss": total_loss, "sup_loss": loss_sup, "unsup_loss": loss_unsup, "unsup_w": unsup_weight}, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=len(images_l), sync_dist=True)
        return total_loss

    def _validation_step_mrcnn(self, batch, batch_idx):
        images, targets = batch; image, target = images[0], targets[0]
        pixel_values = self.val_aug(image, None)[0].to(device=self.device, dtype=self.dtype)
        self.student.train()
        with torch.no_grad():
            target_mrcnn = {"boxes": masks_to_boxes(target["masks"]).to(self.device), "labels": (target["labels"] + 1).to(self.device), "masks": target["masks"].to(torch.uint8).to(self.device)}
            loss_dict = self.student([pixel_values], [target_mrcnn]); val_loss = sum(loss for loss in loss_dict.values())
            self.validation_step_losses.append(val_loss)
        self.student.eval()
        predictions = self.student([pixel_values])[0]; coco_formatted_results = []
        for i in range(len(predictions["scores"])):
            score = predictions["scores"][i].item()
            if score < 0.05: continue
            model_label_id = predictions["labels"][i].item() - 1; category_id = self.trainer.datamodule.label2cat[model_label_id]
            mask = (predictions["masks"][i, 0] > 0.5).cpu().numpy(); rle = mask_utils.encode(np.asfortranarray(mask))
            rle['counts'] = rle['counts'].decode('utf-8')
            coco_formatted_results.append({"image_id": target['image_id'], "category_id": category_id, "segmentation": rle, "score": score})
        self.validation_step_outputs.append(coco_formatted_results)

    def on_validation_epoch_end(self):
        if self.validation_step_losses:
            avg_val_loss = torch.stack(self.validation_step_losses).mean()
            self.log("val_loss", avg_val_loss, prog_bar=True, sync_dist=True)
            self.validation_step_losses.clear()
        all_results = [item for sublist in self.validation_step_outputs for item in sublist]
        if not all_results:
            print("No valid detections during validation."); self.log("val_mAP", 0.0, prog_bar=True, sync_dist=True)
            self.validation_step_outputs.clear(); return
        coco_gt = self.trainer.datamodule.coco_gt_val; coco_dt = coco_gt.loadRes(all_results)
        coco_eval = COCOeval(coco_gt, coco_dt, 'segm'); coco_eval.evaluate(); coco_eval.accumulate(); coco_eval.summarize()
        self.log("val_mAP", coco_eval.stats[0], prog_bar=True, sync_dist=True)
        self.validation_step_outputs.clear()

    def on_train_epoch_end(self):
        # Step the Hungarian matcher epoch counter for ContourFormer
        if self.hparams.head_type == 'contourformer':
            self.student.criterion.matcher.step_epoch()

# --- Step 5: Main Execution Block ---
def main():
    parser = argparse.ArgumentParser(description="Train a Semi-Supervised Instance Segmentation Model.")
    parser.add_argument('--backbone', type=str, default='dino', choices=['dino', 'sam', 'swin', 'convnext', 'repvgg', 'resnet'], help="Choose the backbone model.")
    parser.add_argument('--head', type=str, default='maskrcnn', choices=['maskrcnn', 'contourformer', 'deformable_detr'], help="Choose the segmentation head.")
    parser.add_argument('--learning_rate', type=float, default=5e-5, help="Learning rate for the optimizer.")
    parser.add_argument('--image_size', type=int, default=None, help="Custom image size for resizing. Overrides backbone-specific defaults.")
    parser.add_argument('--batch_size', type=int, default=None, help="Override the default batch size.")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of workers for data loading.")
    parser.add_argument('--fast_dev_run', action='store_true', help="Run a single batch for training and validation to check for errors.")
    parser.add_argument('--max_steps', type=int, default=-1, help="Total number of training steps to perform. Overrides max_epochs.")
    parser.add_argument('--num_labeled_images', type=int, default=-1, help="Number of labeled images to use for training. -1 for all.")
    parser.add_argument('--num_unlabeled_images', type=int, default=-1, help="Number of unlabeled images to use for training. -1 for all.")
    parser.add_argument('--accumulate_grad_batches', type=int, default=1, help="Accumulate gradients over N batches.")
    parser.add_argument('--find_unused_parameters', action='store_true', help="Enable 'find_unused_parameters' for DDP. May slightly slow down training.")
    parser.add_argument('--warmup_steps', type=int, default=500, help="Number of initial steps to train on labeled data only.")
    parser.add_argument('--unsup_rampup_steps', type=int, default=5000, help="Number of steps to ramp up unsupervised loss weight after warmup.")
    parser.add_argument('--val_every_n_epoch', type=int, default=1, help="Run validation every N epochs.")
    args = parser.parse_args()

    if args.backbone == 'sam':
        default_image_size = 512
        default_batch_size = 4
    else:
        default_image_size = 224
        default_batch_size = 8

    image_size = args.image_size if args.image_size is not None else default_image_size
    batch_size = args.batch_size if args.batch_size is not None else default_batch_size

    if args.backbone == 'swin':
        swin_model_name = 'microsoft/swin-base-patch4-window7-224-in22k'
        try:
            config = AutoConfig.from_pretrained(swin_model_name)
            patch_size, window_size = config.patch_size, config.window_size
            divisor = patch_size * window_size
            if image_size % divisor != 0:
                print(f"ERROR: For the Swin Transformer backbone, 'image_size' must be a multiple of (patch_size * window_size) = {divisor}.")
                sys.exit(1)
        except Exception as e:
            print(f"Could not perform Swin Transformer validation check. Error: {e}")

    precision_setting = "32" if args.head in ['contourformer', 'deformable_detr'] or args.backbone in ['repvgg', 'resnet'] else "16-mixed"

    print("\n--- Configuration ---")
    print(f"  Backbone:        {args.backbone}")
    print(f"  Head:            {args.head}")
    print(f"  Image Size:      {image_size}x{image_size}")
    print(f"  Batch Size:      {batch_size}")
    print(f"  Learning Rate:   {args.learning_rate}")
    print(f"  Precision:       {precision_setting}")
    print("---------------------\n")

    pl.seed_everything(42)
    run_dir = os.path.join(PROJECT_DIR, f"{args.backbone}-{args.head}")
    os.makedirs(run_dir, exist_ok=True)
    coco_datamodule = COCODataModule(project_dir=PROJECT_DIR, batch_size=batch_size, image_size=image_size, num_workers=args.num_workers, num_labeled_images=args.num_labeled_images, num_unlabeled_images=args.num_unlabeled_images)
    model = SSLSegmentationLightning(num_classes=NUM_CLASSES, lr=args.learning_rate, backbone_type=args.backbone, head_type=args.head, image_size=image_size, warmup_steps=args.warmup_steps, unsup_rampup_steps=args.unsup_rampup_steps)
    checkpoint_callback = ModelCheckpoint(dirpath=run_dir, filename='best-model-{epoch:02d}-{val_mAP:.4f}', save_top_k=1, verbose=True, monitor='val_mAP', mode='max', save_last=True)
    strategy = "ddp_find_unused_parameters_true" if args.find_unused_parameters else "ddp"

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        max_steps=args.max_steps,
        accelerator="auto",
        devices="auto",
        strategy=strategy,
        accumulate_grad_batches=args.accumulate_grad_batches,
        check_val_every_n_epoch=args.val_every_n_epoch,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10,
        precision=precision_setting,
        default_root_dir=run_dir,
        fast_dev_run=args.fast_dev_run,
        logger=True
    )

    last_ckpt_path = os.path.join(run_dir, "last.ckpt")
    resume_path = last_ckpt_path if os.path.exists(last_ckpt_path) else None
    trainer.fit(model, datamodule=coco_datamodule, ckpt_path=resume_path)

    if not args.fast_dev_run:
        print("\n--- Running final validation on the best checkpoint ---")
        trainer.validate(model, datamodule=coco_datamodule, ckpt_path='best')

if __name__ == '__main__':
    main()
