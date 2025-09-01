"""Mask utility functions."""

import torch
import numpy as np
from typing import List, Tuple
from skimage.measure import find_contours
from scipy.interpolate import interp1d

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