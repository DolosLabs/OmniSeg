"""DINO Vision Transformer backbone."""

import math
from typing import Dict
import torch
import torch.nn as nn
from transformers import AutoModel

from ..base import BaseBackbone
from ...config import BACKBONE_CONFIGS


class DinoVisionTransformerBackbone(BaseBackbone, nn.Module):
    """DINO Vision Transformer backbone."""
    
    def __init__(self, model_name: str = None, feature_layer: int = 8, freeze_encoder: bool = True):
        BaseBackbone.__init__(self, freeze_encoder)
        nn.Module.__init__(self)
        
        if model_name is None:
            model_name = BACKBONE_CONFIGS['dino']['model_name']
            
        print(f"Loading backbone: {model_name}")
        self.dino = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
        
        if self.freeze_encoder:
            for p in self.dino.parameters(): 
                p.requires_grad = False
            self.dino.eval()
            
        self.feature_layer_index = feature_layer
        self.num_register_tokens = self.dino.config.num_register_tokens
        embed_dim = self.dino.config.hidden_size
        decoder_channels = [128, 256, 512, 1024]
        
        self.pyramid_layers = nn.ModuleList()
        self.pyramid_layers.append(nn.Conv2d(embed_dim, decoder_channels[0], kernel_size=1))
        for i in range(1, len(decoder_channels)):
            self.pyramid_layers.append(nn.Sequential(
                nn.Conv2d(decoder_channels[i-1], decoder_channels[i], kernel_size=3, stride=2, padding=1), 
                nn.ReLU()
            ))
        self._decoder_channels = decoder_channels

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = self.dino(pixel_values=x, output_hidden_states=True)
        hidden_state = outputs.hidden_states[self.feature_layer_index]
        B, N, D = hidden_state.shape
        tokens_to_skip = 1 + self.num_register_tokens
        patch_tokens = hidden_state[:, tokens_to_skip:, :]
        seq_len_patches = N - tokens_to_skip
        side_len = int(math.sqrt(seq_len_patches))
        
        if side_len * side_len != seq_len_patches:
            raise ValueError("Patch tokens do not form a perfect square.")
            
        fmap = patch_tokens.permute(0, 2, 1).reshape(B, D, side_len, side_len)
        features = {}
        current_fmap = fmap
        
        for i, layer in enumerate(self.pyramid_layers):
            current_fmap = layer(current_fmap)
            features[f"res{i+2}"] = current_fmap
        return features

    @property
    def output_channels(self) -> Dict[str, int]:
        return {f"res{i+2}": ch for i, ch in enumerate(self._decoder_channels)}