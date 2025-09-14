"""
Performance and benchmark tests for OmniSeg
"""
import pytest
import torch
import time
import psutil
import os
from typing import Dict, Any
from unittest.mock import patch

from omniseg.config import get_default_config
from omniseg.models.backbones import get_backbone
from omniseg.models.heads import get_head
from omniseg.training import SSLSegmentationLightning


class TestPerformanceBenchmarks:
    """Test suite for performance benchmarks and monitoring."""
    
    @pytest.mark.benchmark
    def test_backbone_inference_speed(self, benchmark, test_config):
        """Benchmark backbone inference speed."""
        backbone = get_backbone('simple', image_size=test_config['image_size'])
        backbone.eval()
        
        sample_input = torch.randn(1, 3, test_config['image_size'], test_config['image_size'])
        
        def run_inference():
            with torch.no_grad():
                return backbone(sample_input)
        
        result = benchmark(run_inference)
        
        # Verify output is correct
        assert isinstance(result, dict)
        assert len(result) > 0
    
    @pytest.mark.benchmark
    def test_head_inference_speed(self, benchmark, test_config):
        """Benchmark head inference speed."""
        head = get_head(
            'lw_detr',
            num_classes=test_config['num_classes'],
            backbone_type='simple',
            image_size=test_config['image_size']
        )
        head.eval()
        
        sample_input = torch.randn(1, 3, test_config['image_size'], test_config['image_size'])
        
        def run_inference():
            with torch.no_grad():
                return head(sample_input)
        
        result = benchmark(run_inference)
        
        assert isinstance(result, dict)
        assert 'logits' in result
    
    @pytest.mark.benchmark
    def test_full_model_inference_speed(self, benchmark, test_config):
        """Benchmark full model inference speed."""
        model = SSLSegmentationLightning(
            backbone_type='simple',
            head_type='lw_detr',
            num_classes=test_config['num_classes'],
            image_size=test_config['image_size'],
            learning_rate=test_config['learning_rate']
        )
        model.eval()
        
        sample_input = torch.randn(1, 3, test_config['image_size'], test_config['image_size'])
        sample_targets = [{
            "boxes": torch.tensor([[10, 10, 60, 60]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.int64),
            "masks": torch.ones(1, test_config['image_size'], test_config['image_size'], dtype=torch.float32),
            "image_id": torch.tensor([1])
        }]
        
        batch = (sample_input, sample_targets)
        
        def run_validation():
            with torch.no_grad():
                return model.validation_step(batch, 0)
        
        benchmark(run_validation)
    
    @pytest.mark.slow
    def test_memory_usage_during_training(self, test_config):
        """Test memory usage during training."""
        process = psutil.Process(os.getpid())
        
        # Measure initial memory
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        model = SSLSegmentationLightning(
            backbone_type='simple',
            head_type='lw_detr',
            num_classes=test_config['num_classes'],
            image_size=test_config['image_size'],
            learning_rate=test_config['learning_rate']
        )
        
        # Measure memory after model creation
        model_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        max_memory = model_memory
        
        # Simulate training steps and monitor memory
        for i in range(5):
            sample_input = torch.randn(test_config['batch_size'], 3, test_config['image_size'], test_config['image_size'])
            sample_targets = [{
                "boxes": torch.tensor([[10, 10, 60, 60]], dtype=torch.float32),
                "labels": torch.tensor([1], dtype=torch.int64),
                "masks": torch.ones(1, test_config['image_size'], test_config['image_size'], dtype=torch.float32),
                "image_id": torch.tensor([j])
            } for j in range(test_config['batch_size'])]
            
            batch = (sample_input, sample_targets)
            
            loss = model.training_step(batch, i)
            loss.backward()
            model.zero_grad()
            
            # Measure memory after training step
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            max_memory = max(max_memory, current_memory)
            
            del loss, batch, sample_input, sample_targets
        
        # Memory usage should be reasonable
        memory_increase = max_memory - initial_memory
        model_size = model_memory - initial_memory
        
        print(f"Initial memory: {initial_memory:.2f} MB")
        print(f"Model size: {model_size:.2f} MB")
        print(f"Max memory: {max_memory:.2f} MB")
        print(f"Memory increase: {memory_increase:.2f} MB")
        
        # Assertions
        assert model_size < 500, f"Model size too large: {model_size:.2f} MB"
        assert memory_increase < 1000, f"Memory increase too large: {memory_increase:.2f} MB"
    
    def test_batch_size_scaling(self, test_config):
        """Test performance scaling with different batch sizes."""
        model = SSLSegmentationLightning(
            backbone_type='simple',
            head_type='lw_detr',
            num_classes=test_config['num_classes'],
            image_size=test_config['image_size'],
            learning_rate=test_config['learning_rate']
        )
        model.eval()
        
        times = {}
        
        for batch_size in [1, 2, 4, 8]:
            sample_input = torch.randn(batch_size, 3, test_config['image_size'], test_config['image_size'])
            sample_targets = [{
                "boxes": torch.tensor([[10, 10, 60, 60]], dtype=torch.float32),
                "labels": torch.tensor([1], dtype=torch.int64),
                "masks": torch.ones(1, test_config['image_size'], test_config['image_size'], dtype=torch.float32),
                "image_id": torch.tensor([i])
            } for i in range(batch_size)]
            
            batch = (sample_input, sample_targets)
            
            # Warm up
            with torch.no_grad():
                model.validation_step(batch, 0)
            
            # Measure time
            start_time = time.time()
            for _ in range(3):
                with torch.no_grad():
                    model.validation_step(batch, 0)
            end_time = time.time()
            
            times[batch_size] = (end_time - start_time) / 3  # Average time per iteration
        
        # Check that time scales reasonably with batch size
        # Time per sample should decrease as batch size increases (efficiency)
        time_per_sample = {bs: times[bs] / bs for bs in times.keys()}
        
        print("Batch size scaling results:")
        for bs, total_time in times.items():
            print(f"Batch size {bs}: {total_time:.4f}s total, {time_per_sample[bs]:.4f}s per sample")
        
        # Time per sample should be lower for larger batch sizes
        assert time_per_sample[8] <= time_per_sample[1] * 1.5, "Batch processing not efficient"
    
    def test_image_size_scaling(self, test_config):
        """Test performance scaling with different image sizes."""
        times = {}
        
        for image_size in [64, 128, 224]:
            model = SSLSegmentationLightning(
                backbone_type='simple',
                head_type='lw_detr',
                num_classes=test_config['num_classes'],
                image_size=image_size,
                learning_rate=test_config['learning_rate']
            )
            model.eval()
            
            sample_input = torch.randn(1, 3, image_size, image_size)
            sample_targets = [{
                "boxes": torch.tensor([[10, 10, 30, 30]], dtype=torch.float32),
                "labels": torch.tensor([1], dtype=torch.int64),
                "masks": torch.ones(1, image_size, image_size, dtype=torch.float32),
                "image_id": torch.tensor([1])
            }]
            
            batch = (sample_input, sample_targets)
            
            # Warm up
            with torch.no_grad():
                model.validation_step(batch, 0)
            
            # Measure time
            start_time = time.time()
            for _ in range(3):
                with torch.no_grad():
                    model.validation_step(batch, 0)
            end_time = time.time()
            
            times[image_size] = (end_time - start_time) / 3
        
        print("Image size scaling results:")
        for size, time_taken in times.items():
            print(f"Image size {size}: {time_taken:.4f}s")
        
        # Larger images should take more time (roughly quadratic relationship)
        assert times[224] > times[128] > times[64], "Time should increase with image size"
    
    @pytest.mark.parametrize("backbone_type", ['simple'])
    @pytest.mark.parametrize("head_type", ['lw_detr', 'sparrow_seg'])
    def test_model_combination_performance(self, backbone_type, head_type, test_config):
        """Test performance of different model combinations."""
        try:
            model = SSLSegmentationLightning(
                backbone_type=backbone_type,
                head_type=head_type,
                num_classes=test_config['num_classes'],
                image_size=test_config['image_size'],
                learning_rate=test_config['learning_rate']
            )
            model.eval()
            
            sample_input = torch.randn(1, 3, test_config['image_size'], test_config['image_size'])
            sample_targets = [{
                "boxes": torch.tensor([[10, 10, 60, 60]], dtype=torch.float32),
                "labels": torch.tensor([1], dtype=torch.int64),
                "masks": torch.ones(1, test_config['image_size'], test_config['image_size'], dtype=torch.float32),
                "image_id": torch.tensor([1])
            }]
            
            batch = (sample_input, sample_targets)
            
            # Measure inference time
            start_time = time.time()
            with torch.no_grad():
                result = model.validation_step(batch, 0)
            end_time = time.time()
            
            inference_time = end_time - start_time
            
            print(f"{backbone_type} + {head_type}: {inference_time:.4f}s")
            
            # Should complete in reasonable time
            assert inference_time < 5.0, f"Inference too slow: {inference_time:.4f}s"
            
        except ValueError as e:
            if "Unknown" in str(e):
                pytest.skip(f"Combination {backbone_type}+{head_type} not supported")
            else:
                raise
    
    def test_cpu_vs_gpu_performance(self, test_config):
        """Test CPU vs GPU performance if available."""
        model = SSLSegmentationLightning(
            backbone_type='simple',
            head_type='lw_detr',
            num_classes=test_config['num_classes'],
            image_size=test_config['image_size'],
            learning_rate=test_config['learning_rate']
        )
        
        sample_input = torch.randn(1, 3, test_config['image_size'], test_config['image_size'])
        sample_targets = [{
            "boxes": torch.tensor([[10, 10, 60, 60]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.int64),
            "masks": torch.ones(1, test_config['image_size'], test_config['image_size'], dtype=torch.float32),
            "image_id": torch.tensor([1])
        }]
        
        batch = (sample_input, sample_targets)
        
        # CPU performance
        model.cpu()
        model.eval()
        
        start_time = time.time()
        with torch.no_grad():
            model.validation_step(batch, 0)
        cpu_time = time.time() - start_time
        
        print(f"CPU inference time: {cpu_time:.4f}s")
        
        # GPU performance (if available)
        if torch.cuda.is_available():
            device = torch.device('cuda')
            model.to(device)
            batch_gpu = (sample_input.to(device), [{k: v.to(device) if torch.is_tensor(v) else v 
                                                   for k, v in target.items()} for target in sample_targets])
            
            start_time = time.time()
            with torch.no_grad():
                model.validation_step(batch_gpu, 0)
            gpu_time = time.time() - start_time
            
            print(f"GPU inference time: {gpu_time:.4f}s")
            print(f"GPU speedup: {cpu_time / gpu_time:.2f}x")
        else:
            pytest.skip("CUDA not available for GPU testing")
    
    def test_model_parameter_count(self, test_config):
        """Test and report model parameter counts."""
        model = SSLSegmentationLightning(
            backbone_type='simple',
            head_type='lw_detr',
            num_classes=test_config['num_classes'],
            image_size=test_config['image_size'],
            learning_rate=test_config['learning_rate']
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Breakdown by component
        backbone_params = sum(p.numel() for p in model.student.backbone.parameters())
        head_params = sum(p.numel() for p in model.student.head.parameters())
        
        print(f"Model parameter breakdown:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Backbone parameters: {backbone_params:,}")
        print(f"  Head parameters: {head_params:,}")
        print(f"  Trainable ratio: {trainable_params/total_params:.2%}")
        
        # Assertions
        assert total_params > 0, "Model should have parameters"
        assert trainable_params > 0, "Model should have trainable parameters"
        assert trainable_params <= total_params, "Trainable params should not exceed total"
        
        # Simple model should have reasonable parameter count
        assert total_params < 50_000_000, f"Model too large: {total_params:,} parameters"
    
    @pytest.mark.slow
    def test_training_convergence_speed(self, test_config):
        """Test training convergence speed."""
        model = SSLSegmentationLightning(
            backbone_type='simple',
            head_type='lw_detr',
            num_classes=test_config['num_classes'],
            image_size=test_config['image_size'],
            learning_rate=1e-3  # Higher LR for faster convergence test
        )
        
        # Create consistent training data
        torch.manual_seed(42)
        sample_input = torch.randn(test_config['batch_size'], 3, test_config['image_size'], test_config['image_size'])
        sample_targets = [{
            "boxes": torch.tensor([[20, 20, 80, 80]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.int64),
            "masks": torch.ones(1, test_config['image_size'], test_config['image_size'], dtype=torch.float32),
            "image_id": torch.tensor([i])
        } for i in range(test_config['batch_size'])]
        
        batch = (sample_input, sample_targets)
        
        losses = []
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Train for a few steps
        for step in range(10):
            optimizer.zero_grad()
            loss = model.training_step(batch, step)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if step > 0:
                print(f"Step {step}: Loss = {loss.item():.4f}")
        
        # Loss should generally decrease
        initial_loss = losses[0]
        final_loss = losses[-1]
        
        print(f"Initial loss: {initial_loss:.4f}")
        print(f"Final loss: {final_loss:.4f}")
        print(f"Loss reduction: {(initial_loss - final_loss) / initial_loss:.2%}")
        
        # Allow for some variation but expect general improvement
        assert final_loss < initial_loss * 1.1, "Model should show learning progress"