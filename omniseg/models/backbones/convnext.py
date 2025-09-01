"""ConvNeXt backbone."""

from typing import Dict
import torch
import torch.nn as nn
import timm

from ..base import BaseBackbone
from ...config import BACKBONE_CONFIGS


class ConvNeXtBackbone(BaseBackbone, nn.Module):
    def __init__(self, model_name: str = None, freeze_encoder: bool = True):
        BaseBackbone.__init__(self, freeze_encoder)
        nn.Module.__init__(self)
        
        if model_name is None:
            model_name = BACKBONE_CONFIGS['convnext']['model_name']
            
        print(f"Loading backbone: {model_name}")
        self.convnext = timm.create_model(model_name, pretrained=True, features_only=True, out_indices=(0, 1, 2, 3))
        
        if self.freeze_encoder:
            for p in self.convnext.parameters():
                p.requires_grad = False
            self.convnext.eval()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = self.convnext(x)
        return {f"res{i+2}": out for i, out in enumerate(outputs)}

    @property
    def output_channels(self) -> Dict[str, int]:
        # ConvNeXt output channels depend on the specific model
        return {"res2": 128, "res3": 256, "res4": 512, "res5": 1024}