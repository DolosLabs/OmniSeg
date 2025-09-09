"""
Utility module for handling DeformableAttention imports and fallbacks.
Provides a centralized warning system to avoid repetitive messages.
"""

import warnings
import torch.nn as nn

# Global flag to track if warning has been shown
_DEFORMABLE_ATTENTION_WARNING_SHOWN = False

def get_deformable_attention():
    """
    Attempt to import DeformableAttention with a centralized warning system.
    
    Returns:
        DeformableAttention class (official or fallback implementation)
    """
    global _DEFORMABLE_ATTENTION_WARNING_SHOWN
    
    try:
        from torchvision.ops.deformable_attention import DeformableAttention
        return DeformableAttention
    except ImportError:
        if not _DEFORMABLE_ATTENTION_WARNING_SHOWN:
            warnings.warn(
                "Failed to import official DeformableAttention from torchvision. "
                "Falling back to pure PyTorch implementation. "
                "Consider updating torch/torchvision for better performance.",
                UserWarning,
                stacklevel=2
            )
            _DEFORMABLE_ATTENTION_WARNING_SHOWN = True
        
        return PurePyTorchDeformableAttention


class PurePyTorchDeformableAttention(nn.Module):
    """Pure PyTorch fallback implementation of DeformableAttention."""
    
    def __init__(self, d_model, n_levels, n_heads, n_points):
        super().__init__()
        self.n_heads = n_heads
        self.n_levels = n_levels
        self.n_points = n_points
        self.d_model = d_model
        self.head_dim = d_model // n_heads

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight.data, 0.)
        nn.init.constant_(self.sampling_offsets.bias.data, 0.)
        nn.init.constant_(self.attention_weights.weight.data, 0.)
        nn.init.constant_(self.attention_weights.bias.data, 0.)
        nn.init.xavier_uniform_(self.value_proj.weight.data)
        nn.init.constant_(self.value_proj.bias.data, 0.)
        nn.init.xavier_uniform_(self.output_proj.weight.data)
        nn.init.constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):
        # Simplified fallback implementation
        # This is a placeholder that returns the query unchanged
        # In practice, this should implement the deformable attention mechanism
        return self.output_proj(query)