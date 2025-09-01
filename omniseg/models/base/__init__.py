"""Base classes for OmniSeg models."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import torch


class BaseBackbone(ABC):
    """Abstract base class for all backbone architectures."""
    
    def __init__(self, freeze_encoder: bool = True):
        super().__init__()
        self.freeze_encoder = freeze_encoder
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass returning feature maps with consistent naming."""
        pass
    
    @property
    @abstractmethod
    def output_channels(self) -> Dict[str, int]:
        """Return the number of output channels for each feature level."""
        pass


class BaseHead(ABC):
    """Abstract base class for all segmentation heads."""
    
    def __init__(self, num_classes: int, backbone_type: str = 'dino', image_size: int = 224):
        super().__init__()
        self.num_classes = num_classes
        self.backbone_type = backbone_type
        self.image_size = image_size
    
    @abstractmethod
    def forward(self, pixel_values: torch.Tensor, targets: Optional[List[Dict]] = None) -> Dict[str, torch.Tensor]:
        """Forward pass for training/inference."""
        pass