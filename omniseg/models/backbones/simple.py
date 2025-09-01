"""Simple backbone for testing purposes that doesn't require external downloads."""

import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Dict
from ..base import BaseBackbone


class SimpleTestBackbone(BaseBackbone, nn.Module):
    """Simple CNN backbone for testing that doesn't require pretrained weights."""
    
    def __init__(self, freeze_encoder: bool = False, **kwargs):
        super().__init__(freeze_encoder)
        nn.Module.__init__(self)
        self.freeze_encoder = freeze_encoder
        
        # Simple CNN architecture
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Second block 
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Third block
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256), 
            nn.ReLU(inplace=True),
            
            # Fourth block
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        if freeze_encoder:
            for param in self.parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        """Forward pass returning features at multiple scales."""
        features = OrderedDict()
        
        # Pass through each block and save features
        x = self.features[:4](x)  # First block
        features['0'] = x
        
        x = self.features[4:7](x)  # Second block  
        features['1'] = x
        
        x = self.features[7:10](x)  # Third block
        features['2'] = x
        
        x = self.features[10:](x)  # Fourth block
        features['3'] = x
        
        return features
    
    @property
    def output_channels(self) -> Dict[str, int]:
        """Return the number of output channels for each feature level."""
        return {
            '0': 64,
            '1': 128,
            '2': 256,
            '3': 512
        }