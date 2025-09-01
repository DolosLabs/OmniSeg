"""SAM Vision Transformer backbone."""

import math
from typing import Dict
import torch
import torch.nn as nn
from transformers import AutoModel

from ..base import BaseBackbone
from ...config import BACKBONE_CONFIGS


class SamVisionTransformerBackbone(BaseBackbone, nn.Module):
    def __init__(self, model_name: str = None, freeze_encoder: bool = True):
        BaseBackbone.__init__(self, freeze_encoder)
        nn.Module.__init__(self)
        
        if model_name is None:
            model_name = BACKBONE_CONFIGS['sam']['model_name']
            
        print(f"Loading backbone: {model_name}")
        self.sam = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
        
        if self.freeze_encoder:
            for p in self.sam.parameters():
                p.requires_grad = False
            self.sam.eval()
            
        embed_dim = self.sam.config.vision_config.hidden_size
        decoder_channels = [128, 256, 512, 1024]
        self.proj = nn.Conv2d(embed_dim, decoder_channels[0], kernel_size=1)
        self.pyramid_layers = nn.ModuleList()
        
        for i in range(1, len(decoder_channels)):
            self.pyramid_layers.append(nn.Sequential(
                nn.Conv2d(decoder_channels[i-1], decoder_channels[i], kernel_size=3, stride=2, padding=1), 
                nn.ReLU()
            ))
        self._decoder_channels = decoder_channels

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = self.sam.vision_encoder(pixel_values=x)
        last_hidden_state = outputs.last_hidden_state
        B, N, D = last_hidden_state.shape
        side_len = int(math.sqrt(N))
        
        if side_len * side_len != N:
            raise ValueError("Patch tokens do not form a perfect square.")
            
        fmap = last_hidden_state.permute(0, 2, 1).reshape(B, D, side_len, side_len)
        features = {}
        current_fmap = self.proj(fmap)
        features["res2"] = current_fmap
        
        for i, layer in enumerate(self.pyramid_layers):
            current_fmap = layer(current_fmap)
            features[f"res{i+3}"] = current_fmap
        return features

    @property
    def output_channels(self) -> Dict[str, int]:
        return {f"res{i+2}": ch for i, ch in enumerate(self._decoder_channels)}