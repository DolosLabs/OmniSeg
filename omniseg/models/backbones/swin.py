"""Swin Transformer backbone."""

from typing import Dict
import torch
import torch.nn as nn
from transformers import AutoModel

from ..base import BaseBackbone
from ...config import BACKBONE_CONFIGS


class SwinTransformerBackbone(BaseBackbone, nn.Module):
    def __init__(self, model_name: str = None, freeze_encoder: bool = True):
        BaseBackbone.__init__(self, freeze_encoder)
        nn.Module.__init__(self)
        
        if model_name is None:
            model_name = BACKBONE_CONFIGS['swin']['model_name']
            
        print(f"Loading backbone: {model_name}")
        self.swin = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32, out_indices=(0, 1, 2, 3))
        
        if self.freeze_encoder:
            for p in self.swin.parameters():
                p.requires_grad = False
            self.swin.eval()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = self.swin(pixel_values=x)
        return {f"res{i+2}": hs for i, hs in enumerate(outputs.hidden_states)}

    @property
    def output_channels(self) -> Dict[str, int]:
        # Swin output channels depend on the specific model, using reasonable defaults
        return {"res2": 128, "res3": 256, "res4": 512, "res5": 1024}