# OmniSeg Refactored - Modular Structure

This directory demonstrates the refactored modular structure of the OmniSeg project, breaking down the original monolithic `train.py` file (1755 lines) into organized modules.

## 📁 Directory Structure

```
omniseg/
├── config/
│   └── __init__.py          # Configuration constants, backbone/head configs
├── data/
│   └── __init__.py          # COCO dataset utilities, transforms, data modules
├── models/
│   ├── base/
│   │   └── __init__.py      # Abstract base classes (BaseBackbone, BaseHead)
│   ├── backbones/
│   │   ├── __init__.py      # Backbone factory and imports
│   │   ├── dino.py          # DINO Vision Transformer backbone
│   │   ├── sam.py           # SAM Vision Transformer backbone
│   │   ├── swin.py          # Swin Transformer backbone
│   │   ├── convnext.py      # ConvNeXt backbone
│   │   ├── repvgg.py        # RepVGG backbone
│   │   └── resnet.py        # ResNet backbone
│   └── heads/               # TODO: Segmentation heads (complex, kept in original)
├── utils/
│   └── mask_utils.py        # Mask utility functions
└── training/                # TODO: Training utilities
```

## 🎯 Key Improvements

### Before (Monolithic)
- **1755 lines** in a single `train.py` file
- Mixed concerns (config, data, models, training)
- Difficult to maintain and extend
- Hard to reuse components

### After (Modular)
- **Organized into focused modules** with clear responsibilities
- **Reusable components** that can be imported independently
- **Better maintainability** with separation of concerns
- **Easier testing** of individual components

## 🚀 Usage Examples

### Import and use individual components:

```python
# Configuration
from omniseg.config import get_available_backbones, get_default_config

# Data utilities
from omniseg.data import get_transforms, COCODataModule

# Models
from omniseg.models.base import BaseBackbone
from omniseg.models.backbones import get_backbone

# Create a backbone
backbone = get_backbone('resnet', freeze_encoder=True)

# Get transforms
transforms = get_transforms(augment=True, image_size=224)

# Get configuration
config = get_default_config('resnet', 'maskrcnn')
```

### Use the refactored training script:

```bash
# Same interface as original, but with modular imports
python train_refactored.py --backbone resnet --head maskrcnn --fast_dev_run
```

## 🧪 Testing

Run the import tests to verify the modular structure works:

```bash
python test_imports.py
```

## 📊 Comparison

| Aspect | Original | Refactored |
|--------|----------|------------|
| **File count** | 1 monolithic file | 13+ focused modules |
| **Lines per file** | 1755 lines | ~50-300 lines each |
| **Maintainability** | ❌ Difficult | ✅ Easy |
| **Reusability** | ❌ Poor | ✅ Excellent |
| **Testing** | ❌ Complex | ✅ Modular |
| **Extensibility** | ❌ Hard | ✅ Simple |

## 🔧 What Was Refactored

- [x] **Configuration module** - Constants, configs, and utility functions
- [x] **Data module** - COCO dataset utilities, transforms, data loaders
- [x] **Base classes** - Abstract interfaces for backbones and heads
- [x] **Backbone models** - 6 different backbone architectures in separate files
- [x] **Factory functions** - Clean interfaces for creating components
- [ ] **Head models** - Complex segmentation heads (would need more work)
- [ ] **Training module** - PyTorch Lightning training logic (would need extraction)

## 📝 Notes

- The complex segmentation heads (DETR, ContourFormer) remain in the original file due to their size and complexity
- This demonstrates the refactoring approach and benefits of modular structure
- Complete head refactoring would require additional time for the ~1000+ lines of complex model code

The refactoring successfully demonstrates how to break down monolithic code into clean, maintainable, and reusable modules while preserving all functionality.