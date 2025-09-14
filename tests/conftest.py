"""
Pytest configuration and fixtures for OmniSeg tests
"""
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Generator
import pytest
import torch
import numpy as np
from PIL import Image


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(scope="session")
def sample_image() -> torch.Tensor:
    """Create a sample image tensor for testing."""
    return torch.randn(3, 224, 224)


@pytest.fixture(scope="session") 
def sample_batch() -> torch.Tensor:
    """Create a sample batch of images for testing."""
    return torch.randn(2, 3, 224, 224)


@pytest.fixture(scope="session")
def sample_pil_image() -> Image.Image:
    """Create a sample PIL Image for testing."""
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(img_array)


@pytest.fixture(scope="session")
def sample_coco_annotation() -> Dict[str, Any]:
    """Create a sample COCO-style annotation for testing."""
    return {
        "image_id": 1,
        "id": 1,
        "category_id": 1,
        "bbox": [10, 10, 50, 50],  # x, y, width, height
        "area": 2500,
        "iscrowd": 0,
        "segmentation": [[10, 10, 60, 10, 60, 60, 10, 60]]  # polygon
    }


@pytest.fixture(scope="session")
def sample_targets() -> list:
    """Create sample target annotations for testing."""
    return [
        {
            "boxes": torch.tensor([[10, 10, 60, 60]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.int64),
            "masks": torch.ones(1, 224, 224, dtype=torch.float32),
            "image_id": torch.tensor([1])
        }
    ]


@pytest.fixture(scope="function")
def mock_huggingface_token(monkeypatch) -> None:
    """Mock HuggingFace token for testing."""
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")


@pytest.fixture(scope="session")
def test_config() -> Dict[str, Any]:
    """Basic test configuration."""
    return {
        "image_size": 224,
        "num_classes": 3,
        "batch_size": 2,
        "learning_rate": 1e-4,
        "max_epochs": 1
    }


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch, tmp_path):
    """Setup test environment with temporary directories."""
    # Set up temporary directories for cache and outputs
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(exist_ok=True)
    
    output_dir = tmp_path / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    # Set environment variables
    monkeypatch.setenv("TORCH_HOME", str(cache_dir))
    monkeypatch.setenv("TRANSFORMERS_CACHE", str(cache_dir))
    monkeypatch.setenv("HF_HOME", str(cache_dir))
    
    # Disable wandb for testing
    monkeypatch.setenv("WANDB_DISABLED", "true")
    
    # Set torch to CPU only for faster testing
    if torch.cuda.is_available():
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "security: mark test as security test"
    )
    config.addinivalue_line(
        "markers", "model: mark test as requiring model downloads"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on file/function names."""
    for item in items:
        # Add unit test marker to all tests by default
        if not any(mark.name in ['integration', 'slow', 'security', 'model'] 
                  for mark in item.iter_markers()):
            item.add_marker(pytest.mark.unit)
            
        # Mark slow tests
        if "slow" in item.nodeid or "test_training" in item.nodeid:
            item.add_marker(pytest.mark.slow)
            
        # Mark integration tests
        if "integration" in item.nodeid or "test_end_to_end" in item.nodeid:
            item.add_marker(pytest.mark.integration)
            
        # Mark security tests  
        if "security" in item.nodeid or "test_security" in item.nodeid:
            item.add_marker(pytest.mark.security)
            
        # Mark model tests
        if "model" in item.nodeid or any(backbone in item.nodeid 
                                       for backbone in ['dino', 'sam', 'swin', 'convnext']):
            item.add_marker(pytest.mark.model)