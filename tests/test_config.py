"""
Unit tests for omniseg.config module
"""
import pytest
from typing import Dict, Any
from unittest.mock import patch, MagicMock

from omniseg.config import (
    get_available_backbones,
    get_available_heads,
    get_default_config,
    validate_config
)


class TestConfigModule:
    """Test suite for configuration module."""
    
    def test_get_available_backbones(self):
        """Test that available backbones are returned correctly."""
        backbones = get_available_backbones()
        
        assert isinstance(backbones, list)
        assert len(backbones) > 0
        assert 'simple' in backbones  # Basic backbone should always be available
        
        # Check expected backbones are present
        expected_backbones = ['simple', 'dino', 'sam', 'swin', 'convnext', 'repvgg', 'resnet']
        for backbone in expected_backbones:
            assert backbone in backbones
    
    def test_get_available_heads(self):
        """Test that available heads are returned correctly."""
        heads = get_available_heads()
        
        assert isinstance(heads, list)
        assert len(heads) > 0
        
        # Check expected heads are present
        expected_heads = ['maskrcnn', 'contourformer', 'deformable_detr', 'lw_detr', 'sparrow_seg']
        for head in expected_heads:
            assert head in heads
    
    def test_get_default_config_valid_combination(self):
        """Test getting default config for valid backbone-head combination."""
        config = get_default_config('simple', 'maskrcnn')
        
        assert isinstance(config, dict)
        assert 'backbone_type' in config
        assert 'head_type' in config
        assert config['backbone_type'] == 'simple'
        assert config['head_type'] == 'maskrcnn'
        
        # Check required config fields are present
        required_fields = ['num_classes', 'image_size', 'learning_rate']
        for field in required_fields:
            assert field in config
    
    def test_get_default_config_invalid_backbone(self):
        """Test that invalid backbone raises appropriate error."""
        with pytest.raises(ValueError, match="Unknown backbone"):
            get_default_config('invalid_backbone', 'maskrcnn')
    
    def test_get_default_config_invalid_head(self):
        """Test that invalid head raises appropriate error."""
        with pytest.raises(ValueError, match="Unknown head"):
            get_default_config('simple', 'invalid_head')
    
    def test_validate_config_valid(self):
        """Test config validation with valid configuration."""
        valid_config = {
            'backbone_type': 'simple',
            'head_type': 'maskrcnn',
            'num_classes': 3,
            'image_size': 224,
            'learning_rate': 1e-4,
            'batch_size': 16
        }
        
        # Should not raise any exception
        validate_config(valid_config)
    
    def test_validate_config_missing_required_field(self):
        """Test config validation with missing required field."""
        invalid_config = {
            'backbone_type': 'simple',
            'head_type': 'maskrcnn',
            # Missing num_classes
            'image_size': 224,
            'learning_rate': 1e-4
        }
        
        with pytest.raises(ValueError, match="Missing required field"):
            validate_config(invalid_config)
    
    def test_validate_config_invalid_values(self):
        """Test config validation with invalid values."""
        # Test negative num_classes
        invalid_config = {
            'backbone_type': 'simple',
            'head_type': 'maskrcnn',
            'num_classes': -1,
            'image_size': 224,
            'learning_rate': 1e-4
        }
        
        with pytest.raises(ValueError, match="num_classes must be positive"):
            validate_config(invalid_config)
        
        # Test invalid image size
        invalid_config['num_classes'] = 3
        invalid_config['image_size'] = 0
        
        with pytest.raises(ValueError, match="image_size must be positive"):
            validate_config(invalid_config)
    
    @pytest.mark.parametrize("backbone,head", [
        ('simple', 'maskrcnn'),
        ('simple', 'lw_detr'),
        ('resnet', 'maskrcnn'),
        ('dino', 'deformable_detr'),
    ])
    def test_config_backbone_head_combinations(self, backbone, head):
        """Test various backbone-head combinations."""
        try:
            config = get_default_config(backbone, head)
            assert config['backbone_type'] == backbone
            assert config['head_type'] == head
            validate_config(config)
        except ValueError as e:
            # Some combinations might not be supported
            if "Unknown" not in str(e):
                raise
    
    def test_config_immutability(self):
        """Test that returned config objects are independent."""
        config1 = get_default_config('simple', 'maskrcnn')
        config2 = get_default_config('simple', 'maskrcnn')
        
        # Modify one config
        config1['learning_rate'] = 999
        
        # Other config should be unchanged
        assert config2['learning_rate'] != 999
    
    def test_config_default_values(self):
        """Test that default config values are reasonable."""
        config = get_default_config('simple', 'maskrcnn')
        
        # Check reasonable defaults
        assert config['num_classes'] >= 1
        assert config['image_size'] > 0
        assert config['learning_rate'] > 0
        assert config['learning_rate'] < 1  # Should be a small value
        
        if 'batch_size' in config:
            assert config['batch_size'] > 0
        
        if 'max_epochs' in config:
            assert config['max_epochs'] > 0