"""
Integration tests for OmniSeg end-to-end functionality
"""
import pytest
import torch
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import json

from omniseg.config import get_default_config
from omniseg.models.backbones import get_backbone
from omniseg.models.heads import get_head
from omniseg.training import SSLSegmentationLightning


class TestEndToEndIntegration:
    """Test suite for end-to-end integration scenarios."""
    
    @pytest.mark.integration
    def test_full_model_pipeline_simple(self, test_config, sample_batch, sample_targets):
        """Test complete model pipeline with simple backbone and head."""
        # Get configuration
        config = get_default_config('simple', 'lw_detr')
        config.update(test_config)
        
        # Create backbone
        backbone = get_backbone(config['backbone_type'], image_size=config['image_size'])
        
        # Create head  
        head = get_head(
            config['head_type'],
            num_classes=config['num_classes'],
            backbone_type=config['backbone_type'],
            image_size=config['image_size']
        )
        
        # Test forward pass
        with torch.no_grad():
            features = backbone(sample_batch)
            outputs = head(sample_batch)
        
        assert isinstance(features, dict)
        assert isinstance(outputs, dict)
        assert 'logits' in outputs
        assert 'pred_boxes' in outputs
    
    @pytest.mark.integration
    def test_training_module_initialization(self, test_config):
        """Test PyTorch Lightning training module initialization."""
        config = get_default_config('simple', 'lw_detr')
        config.update(test_config)
        
        # Create lightning module
        model = SSLSegmentationLightning(
            backbone_type=config['backbone_type'],
            head_type=config['head_type'],
            num_classes=config['num_classes'],
            image_size=config['image_size'],
            learning_rate=config['learning_rate']
        )
        
        assert model is not None
        assert hasattr(model, 'training_step')
        assert hasattr(model, 'validation_step')
        assert hasattr(model, 'configure_optimizers')
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_training_step_execution(self, test_config, sample_batch, sample_targets):
        """Test that training step executes without errors."""
        config = get_default_config('simple', 'lw_detr')
        config.update(test_config)
        
        model = SSLSegmentationLightning(
            backbone_type=config['backbone_type'],
            head_type=config['head_type'], 
            num_classes=config['num_classes'],
            image_size=config['image_size'],
            learning_rate=config['learning_rate']
        )
        
        # Prepare batch data
        batch = (sample_batch, sample_targets)
        
        # Execute training step
        loss = model.training_step(batch, batch_idx=0)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert torch.isfinite(loss).all()
    
    @pytest.mark.integration
    def test_validation_step_execution(self, test_config, sample_batch, sample_targets):
        """Test that validation step executes without errors."""
        config = get_default_config('simple', 'lw_detr')
        config.update(test_config)
        
        model = SSLSegmentationLightning(
            backbone_type=config['backbone_type'],
            head_type=config['head_type'],
            num_classes=config['num_classes'],
            image_size=config['image_size'],
            learning_rate=config['learning_rate']
        )
        
        batch = (sample_batch, sample_targets)
        
        # Execute validation step
        result = model.validation_step(batch, batch_idx=0)
        
        # Should complete without errors
        assert result is None or isinstance(result, (dict, torch.Tensor))
    
    @pytest.mark.integration
    def test_optimizer_configuration(self, test_config):
        """Test optimizer configuration."""
        config = get_default_config('simple', 'lw_detr')
        config.update(test_config)
        
        model = SSLSegmentationLightning(
            backbone_type=config['backbone_type'],
            head_type=config['head_type'],
            num_classes=config['num_classes'],
            image_size=config['image_size'],
            learning_rate=config['learning_rate']
        )
        
        optimizer_config = model.configure_optimizers()
        
        assert optimizer_config is not None
        # Should be optimizer or dict with optimizer
        if isinstance(optimizer_config, dict):
            assert 'optimizer' in optimizer_config
        else:
            assert hasattr(optimizer_config, 'step')
    
    @pytest.mark.integration
    @pytest.mark.parametrize("backbone,head", [
        ('simple', 'lw_detr'),
        ('simple', 'maskrcnn'),
        ('simple', 'sparrow_seg'),
    ])
    def test_backbone_head_compatibility(self, backbone, head, test_config):
        """Test compatibility between different backbones and heads."""
        try:
            config = get_default_config(backbone, head)
            config.update(test_config)
            
            # Test model creation
            backbone_model = get_backbone(backbone, image_size=config['image_size'])
            head_model = get_head(
                head,
                num_classes=config['num_classes'],
                backbone_type=backbone,
                image_size=config['image_size']
            )
            
            # Test forward pass
            sample_input = torch.randn(1, 3, config['image_size'], config['image_size'])
            
            with torch.no_grad():
                backbone_model.eval()
                head_model.eval()
                
                features = backbone_model(sample_input)
                
                if head == 'maskrcnn':
                    # Mask R-CNN expects list of images
                    outputs = head_model([sample_input[0]])
                else:
                    outputs = head_model(sample_input)
            
            assert features is not None
            assert outputs is not None
            
        except ValueError as e:
            if "Unknown" in str(e):
                pytest.skip(f"Combination {backbone}+{head} not supported")
            else:
                raise
    
    @pytest.mark.integration
    def test_model_state_dict_save_load(self, test_config):
        """Test model state dict saving and loading."""
        config = get_default_config('simple', 'lw_detr')
        config.update(test_config)
        
        # Create model
        model = SSLSegmentationLightning(
            backbone_type=config['backbone_type'],
            head_type=config['head_type'],
            num_classes=config['num_classes'],
            image_size=config['image_size'],
            learning_rate=config['learning_rate']
        )
        
        # Save state dict
        state_dict = model.state_dict()
        
        # Create new model and load state dict
        model2 = SSLSegmentationLightning(
            backbone_type=config['backbone_type'],
            head_type=config['head_type'],
            num_classes=config['num_classes'],
            image_size=config['image_size'],
            learning_rate=config['learning_rate']
        )
        
        model2.load_state_dict(state_dict)
        
        # Test that models produce same output
        sample_input = torch.randn(1, 3, config['image_size'], config['image_size'])
        sample_targets = [{
            "boxes": torch.tensor([[10, 10, 60, 60]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.int64),
            "masks": torch.ones(1, config['image_size'], config['image_size'], dtype=torch.float32),
            "image_id": torch.tensor([1])
        }]
        
        batch = (sample_input, sample_targets)
        
        with torch.no_grad():
            model.eval()
            model2.eval()
            
            loss1 = model.training_step(batch, 0)
            loss2 = model2.training_step(batch, 0)
            
            # Should produce identical results
            torch.testing.assert_close(loss1, loss2, rtol=1e-5, atol=1e-8)
    
    @pytest.mark.integration
    def test_config_validation_integration(self):
        """Test that config validation works with real model creation."""
        from omniseg.config import validate_config
        
        # Valid config should work
        valid_config = {
            'backbone_type': 'simple',
            'head_type': 'lw_detr',
            'num_classes': 3,
            'image_size': 224,
            'learning_rate': 1e-4,
            'batch_size': 2
        }
        
        validate_config(valid_config)
        
        # Should be able to create model with valid config
        model = SSLSegmentationLightning(**valid_config)
        assert model is not None
    
    @pytest.mark.integration
    def test_data_preprocessing_integration(self, sample_pil_image):
        """Test data preprocessing pipeline integration."""
        from omniseg.data import get_transforms
        
        # Get transforms
        transforms = get_transforms(augment=True, image_size=224)
        
        # Apply transforms
        transformed = transforms(sample_pil_image)
        
        assert isinstance(transformed, torch.Tensor)
        assert transformed.shape == (3, 224, 224)
        assert transformed.dtype == torch.float32
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_memory_usage_integration(self, test_config):
        """Test memory usage during training integration."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        config = get_default_config('simple', 'lw_detr')
        config.update(test_config)
        
        model = SSLSegmentationLightning(**config)
        
        # Simulate training steps
        for _ in range(3):
            sample_input = torch.randn(config['batch_size'], 3, config['image_size'], config['image_size'])
            sample_targets = [{
                "boxes": torch.tensor([[10, 10, 60, 60]], dtype=torch.float32),
                "labels": torch.tensor([1], dtype=torch.int64),
                "masks": torch.ones(1, config['image_size'], config['image_size'], dtype=torch.float32),
                "image_id": torch.tensor([i])
            } for i in range(config['batch_size'])]
            
            batch = (sample_input, sample_targets)
            loss = model.training_step(batch, 0)
            loss.backward()
            
            # Clear gradients
            model.zero_grad()
            del loss, batch, sample_input, sample_targets
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 1GB)
        assert memory_increase < 1000, f"Memory usage increased by {memory_increase:.2f} MB"
    
    @pytest.mark.integration
    def test_error_handling_integration(self, test_config):
        """Test error handling in integration scenarios."""
        config = get_default_config('simple', 'lw_detr')
        config.update(test_config)
        
        model = SSLSegmentationLightning(**config)
        
        # Test with malformed batch data
        with pytest.raises(Exception):
            bad_batch = ("not_a_tensor", "not_targets")
            model.training_step(bad_batch, 0)
        
        # Test with wrong input shape
        with pytest.raises(Exception):
            wrong_shape_input = torch.randn(2, 4, 224, 224)  # Wrong number of channels
            wrong_targets = [{
                "boxes": torch.tensor([[10, 10, 60, 60]], dtype=torch.float32),
                "labels": torch.tensor([1], dtype=torch.int64),
                "masks": torch.ones(1, 224, 224, dtype=torch.float32),
                "image_id": torch.tensor([1])
            }]
            batch = (wrong_shape_input, wrong_targets)
            model.training_step(batch, 0)
    
    @pytest.mark.integration
    def test_reproducibility_integration(self, test_config):
        """Test that training is reproducible with same seed."""
        import random
        import numpy as np
        
        def set_seed(seed=42):
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
        
        config = get_default_config('simple', 'lw_detr')
        config.update(test_config)
        
        # First run
        set_seed(42)
        model1 = SSLSegmentationLightning(**config)
        sample_input = torch.randn(1, 3, config['image_size'], config['image_size'])
        sample_targets = [{
            "boxes": torch.tensor([[10, 10, 60, 60]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.int64),
            "masks": torch.ones(1, config['image_size'], config['image_size'], dtype=torch.float32),
            "image_id": torch.tensor([1])
        }]
        batch1 = (sample_input, sample_targets)
        loss1 = model1.training_step(batch1, 0)
        
        # Second run with same seed
        set_seed(42)
        model2 = SSLSegmentationLightning(**config)
        loss2 = model2.training_step(batch1, 0)
        
        # Should produce identical results
        torch.testing.assert_close(loss1, loss2, rtol=1e-5, atol=1e-8)