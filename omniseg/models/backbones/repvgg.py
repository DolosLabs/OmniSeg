"""RepVGG backbone."""

from typing import Dict
import torch
import torch.nn as nn
import timm

from ..base import BaseBackbone
from ...config import BACKBONE_CONFIGS


class RepVGGBackbone(BaseBackbone, nn.Module):
    def __init__(self, model_name: str = None, freeze_encoder: bool = True):
        BaseBackbone.__init__(self, freeze_encoder)
        nn.Module.__init__(self)
        
        if model_name is None:
            model_name = BACKBONE_CONFIGS['repvgg']['model_name']
            
        print(f"Loading backbone: {model_name}")
        self.repvgg = timm.create_model(model_name, pretrained=True, features_only=True, out_indices=(0, 1, 2, 3))
        
        if self.freeze_encoder:
            for p in self.repvgg.parameters():
                p.requires_grad = False
            self.repvgg.eval()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = self.repvgg(x)
        return {f"res{i+2}": out for i, out in enumerate(outputs)}

    @property
    def output_channels(self) -> Dict[str, int]:
        # RepVGG output channels
        return {"res2": 64, "res3": 128, "res4": 256, "res5": 512}