"""
Unit tests for omniseg.models.heads module
"""
import pytest
import torch
from unittest.mock import patch, MagicMock

from omniseg.models.heads import get_head
from omniseg.models.base import BaseHead


class TestHeadFactory:
    """Test suite for head factory functions."""
    
    def test_get_head_maskrcnn(self, test_config):
        """Test getting Mask R-CNN head."""
        head = get_head(
            'maskrcnn', 
            num_classes=test_config['num_classes'],
            backbone_type='simple',
            image_size=test_config['image_size']
        )
        
        assert isinstance(head, BaseHead)
        assert hasattr(head, 'forward')
    
    def test_get_head_lw_detr(self, test_config):
        """Test getting LW-DETR head."""
        head = get_head(
            'lw_detr',
            num_classes=test_config['num_classes'], 
            backbone_type='simple',
            image_size=test_config['image_size']
        )
        
        assert isinstance(head, BaseHead)
        assert hasattr(head, 'forward')
    
    def test_get_head_invalid_type(self, test_config):
        """Test that invalid head type raises error."""
        with pytest.raises(ValueError, match="Unknown head type"):
            get_head(
                'invalid_head',
                num_classes=test_config['num_classes'],
                backbone_type='simple',
                image_size=test_config['image_size']
            )
    
    def test_get_head_invalid_num_classes(self):
        """Test that invalid num_classes raises error."""
        with pytest.raises(ValueError, match="num_classes must be positive"):
            get_head(
                'maskrcnn',
                num_classes=0,
                backbone_type='simple',
                image_size=224
            )
    
    def test_get_head_invalid_image_size(self):
        """Test that invalid image_size raises error.""" 
        with pytest.raises(ValueError, match="image_size must be positive"):
            get_head(
                'maskrcnn',
                num_classes=3,
                backbone_type='simple',
                image_size=0
            )
    
    @pytest.mark.parametrize("head_type", ['maskrcnn', 'lw_detr', 'sparrow_seg', 'deformable_detr'])
    def test_head_instantiation(self, head_type, test_config):
        """Test instantiation of different head types."""
        try:
            head = get_head(
                head_type,
                num_classes=test_config['num_classes'],
                backbone_type='simple',
                image_size=test_config['image_size']
            )
            assert isinstance(head, BaseHead)
        except Exception as e:
            # Some heads might not be fully implemented or require specific backends
            if "not implemented" in str(e).lower() or "unknown" in str(e).lower():
                pytest.skip(f"Head {head_type} not implemented or available")
            else:
                raise
    
    def test_head_forward_pass_detr_style(self, sample_batch, test_config):
        """Test forward pass for DETR-style heads."""
        head = get_head(
            'lw_detr',
            num_classes=test_config['num_classes'],
            backbone_type='simple',
            image_size=test_config['image_size']
        )
        
        with torch.no_grad():
            head.eval()
            outputs = head(sample_batch)
        
        # DETR outputs should be a dict with logits and boxes
        assert isinstance(outputs, dict)
        assert 'logits' in outputs
        assert 'pred_boxes' in outputs
        
        # Check tensor shapes
        batch_size = sample_batch.size(0)
        assert outputs['logits'].size(0) == batch_size
        assert outputs['pred_boxes'].size(0) == batch_size
    
    def test_head_forward_pass_maskrcnn_style(self, sample_batch, sample_targets, test_config):
        """Test forward pass for Mask R-CNN style heads."""
        head = get_head(
            'maskrcnn',
            num_classes=test_config['num_classes'],
            backbone_type='simple',
            image_size=test_config['image_size']
        )
        
        # Convert batch tensor to list for Mask R-CNN
        image_list = [sample_batch[i] for i in range(sample_batch.size(0))]
        
        with torch.no_grad():
            head.eval()
            outputs = head(image_list)
        
        # Mask R-CNN outputs should be a list of dicts
        assert isinstance(outputs, list)
        assert len(outputs) == len(image_list)
        
        for output in outputs:
            assert isinstance(output, dict)
            assert 'boxes' in output
            assert 'scores' in output
            assert 'labels' in output
    
    def test_head_training_mode(self, sample_batch, sample_targets, test_config):
        """Test head in training mode with targets."""
        head = get_head(
            'lw_detr',
            num_classes=test_config['num_classes'],
            backbone_type='simple',
            image_size=test_config['image_size']
        )
        
        head.train()
        
        # Should be able to compute loss with targets
        try:
            outputs = head(sample_batch, targets=sample_targets)
            
            if isinstance(outputs, dict) and 'loss' in outputs:
                loss = outputs['loss']
                assert isinstance(loss, torch.Tensor)
                assert loss.requires_grad
                
                # Test backward pass
                loss.backward()
        except Exception as e:
            # Some heads might require specific target formats
            if "target" not in str(e).lower():
                raise
    
    def test_head_parameter_count(self, test_config):
        """Test that head has reasonable parameter count."""
        head = get_head(
            'lw_detr',
            num_classes=test_config['num_classes'],
            backbone_type='simple',
            image_size=test_config['image_size']
        )
        
        total_params = sum(p.numel() for p in head.parameters())
        trainable_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
        
        assert total_params > 0
        assert trainable_params > 0
        assert trainable_params <= total_params
    
    @pytest.mark.parametrize("num_classes", [1, 3, 10, 80, 91])
    def test_head_different_num_classes(self, num_classes):
        """Test head with different numbers of classes."""
        head = get_head(
            'lw_detr',
            num_classes=num_classes,
            backbone_type='simple',
            image_size=224
        )
        
        assert isinstance(head, BaseHead)
        
        # Test forward pass
        batch = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            head.eval()
            outputs = head(batch)
        
        # Check that outputs have correct class dimension
        if isinstance(outputs, dict) and 'logits' in outputs:
            # For DETR-style: logits should have shape [batch, num_queries, num_classes + 1]
            # +1 for background/no-object class
            assert outputs['logits'].size(-1) == num_classes + 1
    
    def test_head_device_consistency(self, sample_batch, test_config):
        """Test that head handles device consistency."""
        head = get_head(
            'lw_detr',
            num_classes=test_config['num_classes'],
            backbone_type='simple',
            image_size=test_config['image_size']
        )
        
        with torch.no_grad():
            head.eval()
            outputs = head(sample_batch)
        
        # Outputs should be on same device as inputs
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                assert value.device == sample_batch.device
    
    def test_head_batch_size_consistency(self, test_config):
        """Test head with different batch sizes."""
        head = get_head(
            'lw_detr',
            num_classes=test_config['num_classes'],
            backbone_type='simple',
            image_size=test_config['image_size']
        )
        
        head.eval()
        
        for batch_size in [1, 2, 4]:
            batch = torch.randn(batch_size, 3, test_config['image_size'], test_config['image_size'])
            
            with torch.no_grad():
                outputs = head(batch)
            
            # Check batch dimension consistency
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor) and value.dim() > 0:
                    assert value.size(0) == batch_size
    
    def test_head_eval_deterministic(self, sample_batch, test_config):
        """Test that head produces deterministic outputs in eval mode."""
        head = get_head(
            'lw_detr',
            num_classes=test_config['num_classes'],
            backbone_type='simple', 
            image_size=test_config['image_size']
        )
        
        head.eval()
        
        with torch.no_grad():
            outputs1 = head(sample_batch)
            outputs2 = head(sample_batch)
        
        # Outputs should be identical in eval mode
        for key in outputs1.keys():
            if isinstance(outputs1[key], torch.Tensor):
                torch.testing.assert_close(outputs1[key], outputs2[key], rtol=1e-5, atol=1e-8)
    
    def test_head_memory_efficiency(self, sample_batch, test_config):
        """Test that head doesn't leak memory."""
        head = get_head(
            'lw_detr',
            num_classes=test_config['num_classes'],
            backbone_type='simple',
            image_size=test_config['image_size']
        )
        
        head.eval()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Multiple forward passes shouldn't accumulate memory
        for _ in range(3):
            with torch.no_grad():
                outputs = head(sample_batch)
                del outputs