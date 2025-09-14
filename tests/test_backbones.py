"""
Unit tests for omniseg.models.backbones module
"""
import pytest
import torch
from unittest.mock import patch, MagicMock

from omniseg.models.backbones import get_backbone
from omniseg.models.base import BaseBackbone


class TestBackboneFactory:
    """Test suite for backbone factory functions."""
    
    def test_get_backbone_simple(self):
        """Test getting simple backbone."""
        backbone = get_backbone('simple', freeze_encoder=False, image_size=224)
        
        assert isinstance(backbone, BaseBackbone)
        assert hasattr(backbone, 'forward')
        assert hasattr(backbone, 'get_feature_dims')
    
    def test_get_backbone_simple_forward_pass(self, sample_batch):
        """Test simple backbone forward pass."""
        backbone = get_backbone('simple', freeze_encoder=False, image_size=224)
        
        with torch.no_grad():
            features = backbone(sample_batch)
        
        assert isinstance(features, dict)
        assert len(features) > 0
        
        # Check feature dimensions
        for level, feat in features.items():
            assert isinstance(feat, torch.Tensor)
            assert feat.dim() == 4  # B, C, H, W
            assert feat.size(0) == sample_batch.size(0)  # Batch size preserved
    
    def test_get_backbone_invalid_type(self):
        """Test that invalid backbone type raises error."""
        with pytest.raises(ValueError, match="Unknown backbone type"):
            get_backbone('invalid_backbone', image_size=224)
    
    def test_get_backbone_freeze_encoder(self):
        """Test backbone with frozen encoder."""
        backbone = get_backbone('simple', freeze_encoder=True, image_size=224)
        
        # Check that some parameters are frozen
        frozen_params = [p for p in backbone.parameters() if not p.requires_grad]
        total_params = list(backbone.parameters())
        
        # Should have at least some frozen parameters
        assert len(frozen_params) > 0
        assert len(frozen_params) <= len(total_params)
    
    @pytest.mark.parametrize("image_size", [64, 128, 224, 256, 512])
    def test_get_backbone_different_sizes(self, image_size):
        """Test backbone with different image sizes."""
        backbone = get_backbone('simple', freeze_encoder=False, image_size=image_size)
        
        # Test with appropriate batch size
        batch = torch.randn(1, 3, image_size, image_size)
        
        with torch.no_grad():
            features = backbone(batch)
        
        assert isinstance(features, dict)
        
        # Check that output features have reasonable dimensions
        for level, feat in features.items():
            assert feat.size(2) <= image_size  # Height should be <= input
            assert feat.size(3) <= image_size  # Width should be <= input
            assert feat.size(2) > 0  # Should have positive dimensions
            assert feat.size(3) > 0
    
    def test_backbone_feature_dims(self):
        """Test that backbone returns consistent feature dimensions."""
        backbone = get_backbone('simple', freeze_encoder=False, image_size=224)
        
        feature_dims = backbone.get_feature_dims()
        
        assert isinstance(feature_dims, dict)
        assert len(feature_dims) > 0
        
        # All dimensions should be positive integers
        for level, dim in feature_dims.items():
            assert isinstance(dim, int)
            assert dim > 0
    
    def test_backbone_eval_mode(self, sample_batch):
        """Test backbone in evaluation mode."""
        backbone = get_backbone('simple', freeze_encoder=False, image_size=224)
        backbone.eval()
        
        with torch.no_grad():
            features1 = backbone(sample_batch)
            features2 = backbone(sample_batch)
        
        # In eval mode, outputs should be deterministic
        for level in features1.keys():
            torch.testing.assert_close(features1[level], features2[level])
    
    def test_backbone_train_mode(self, sample_batch):
        """Test backbone in training mode."""
        backbone = get_backbone('simple', freeze_encoder=False, image_size=224)
        backbone.train()
        
        # Should be able to compute gradients
        features = backbone(sample_batch)
        
        # Verify gradients can be computed
        for level, feat in features.items():
            if feat.requires_grad:
                loss = feat.sum()
                loss.backward(retain_graph=True)
                break
    
    @pytest.mark.model
    @patch('omniseg.models.backbones.transformers')
    def test_get_backbone_dino_mocked(self, mock_transformers):
        """Test DINO backbone with mocked transformers (avoiding download)."""
        # Mock the transformers module
        mock_model = MagicMock()
        mock_model.config.hidden_size = 768
        mock_transformers.AutoModel.from_pretrained.return_value = mock_model
        
        backbone = get_backbone('dino', freeze_encoder=False, image_size=224)
        
        assert isinstance(backbone, BaseBackbone)
        mock_transformers.AutoModel.from_pretrained.assert_called_once()
    
    def test_backbone_parameter_count(self):
        """Test that backbone has reasonable parameter count."""
        backbone = get_backbone('simple', freeze_encoder=False, image_size=224)
        
        total_params = sum(p.numel() for p in backbone.parameters())
        trainable_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        
        assert total_params > 0
        assert trainable_params > 0
        assert trainable_params <= total_params
        
        # Simple backbone should have reasonable param count (not too large)
        assert total_params < 100_000_000  # Less than 100M parameters
    
    def test_backbone_device_consistency(self, sample_batch):
        """Test that backbone handles device consistency."""
        backbone = get_backbone('simple', freeze_encoder=False, image_size=224)
        
        # Test CPU
        features = backbone(sample_batch)
        for feat in features.values():
            assert feat.device == sample_batch.device
    
    def test_backbone_batch_size_consistency(self):
        """Test backbone with different batch sizes."""
        backbone = get_backbone('simple', freeze_encoder=False, image_size=224)
        
        for batch_size in [1, 2, 4, 8]:
            batch = torch.randn(batch_size, 3, 224, 224)
            
            with torch.no_grad():
                features = backbone(batch)
            
            for feat in features.values():
                assert feat.size(0) == batch_size
    
    def test_backbone_memory_efficiency(self, sample_batch):
        """Test that backbone doesn't leak memory."""
        backbone = get_backbone('simple', freeze_encoder=False, image_size=224)
        
        # Multiple forward passes shouldn't accumulate memory significantly
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        for _ in range(5):
            with torch.no_grad():
                features = backbone(sample_batch)
                del features  # Explicit cleanup
        
        # Should complete without memory errors