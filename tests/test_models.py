"""Test suite for hardware-aware neural networks."""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from src.models import HardwareAwareModel, QuantizedModel, create_hardware_aware_model
from src.data import HardwareAwareDataset, create_synthetic_dataset
from src.training import HardwareAwareTrainer, set_seed
from src.evaluation import HardwareAwareEvaluator
from src.export import ModelExporter


class TestHardwareAwareModel:
    """Test cases for HardwareAwareModel class."""
    
    def test_model_creation(self):
        """Test model creation with different configurations."""
        model = create_hardware_aware_model(
            num_classes=5,
            model_type="mobilenet_v2",
            width_multiplier=0.35,
            input_size=(96, 96)
        )
        
        assert isinstance(model, HardwareAwareModel)
        assert model.num_classes == 5
        assert model.input_size == (96, 96)
        assert model.model_type == "mobilenet_v2"
        assert model.width_multiplier == 0.35
    
    def test_model_forward(self):
        """Test model forward pass."""
        model = create_hardware_aware_model(num_classes=3)
        model.eval()
        
        # Create dummy input
        x = torch.randn(2, 3, 96, 96)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (2, 3)
        assert torch.all(torch.isfinite(output))
    
    def test_model_size_calculation(self):
        """Test model size calculation."""
        model = create_hardware_aware_model()
        size_info = model.get_model_size()
        
        assert "parameters_mb" in size_info
        assert "buffers_mb" in size_info
        assert "total_mb" in size_info
        assert "num_parameters" in size_info
        
        assert size_info["num_parameters"] > 0
        assert size_info["total_mb"] > 0
    
    def test_unsupported_model_type(self):
        """Test error handling for unsupported model types."""
        with pytest.raises(ValueError):
            create_hardware_aware_model(model_type="unsupported_model")


class TestQuantizedModel:
    """Test cases for QuantizedModel class."""
    
    def test_quantized_model_creation(self):
        """Test quantized model creation."""
        model = create_hardware_aware_model()
        quantized_model = QuantizedModel(model)
        
        assert isinstance(quantized_model, QuantizedModel)
        assert quantized_model.model is model
    
    def test_dynamic_quantization(self):
        """Test dynamic quantization."""
        model = create_hardware_aware_model()
        quantized_model = QuantizedModel(model)
        
        quantized = quantized_model.quantize_dynamic()
        
        assert quantized is not None
        assert quantized_model.quantized_model is quantized


class TestDataLoading:
    """Test cases for data loading utilities."""
    
    def test_synthetic_dataset_creation(self):
        """Test synthetic dataset creation."""
        output_dir = Path("test_data")
        
        try:
            create_synthetic_dataset(
                output_dir=output_dir,
                num_classes=3,
                samples_per_class=10,
                image_size=(32, 32)
            )
            
            assert output_dir.exists()
            assert len(list(output_dir.iterdir())) == 3  # 3 classes
            
            for class_dir in output_dir.iterdir():
                if class_dir.is_dir():
                    assert len(list(class_dir.glob("*.jpg"))) == 10
        
        finally:
            # Cleanup
            import shutil
            if output_dir.exists():
                shutil.rmtree(output_dir)
    
    def test_dataset_loading(self):
        """Test dataset loading."""
        output_dir = Path("test_data")
        
        try:
            # Create synthetic dataset
            create_synthetic_dataset(
                output_dir=output_dir,
                num_classes=2,
                samples_per_class=5,
                image_size=(32, 32)
            )
            
            # Test dataset loading
            dataset = HardwareAwareDataset(
                data_dir=output_dir,
                input_size=(32, 32),
                is_training=True
            )
            
            assert len(dataset) == 10  # 2 classes * 5 samples
            assert len(dataset.class_names) == 2
            
            # Test data loading
            image, label = dataset[0]
            assert isinstance(image, torch.Tensor)
            assert isinstance(label, int)
            assert image.shape == (3, 32, 32)
            assert 0 <= label < 2
        
        finally:
            # Cleanup
            import shutil
            if output_dir.exists():
                shutil.rmtree(output_dir)


class TestTraining:
    """Test cases for training utilities."""
    
    def test_seed_setting(self):
        """Test random seed setting."""
        set_seed(42)
        
        # Generate some random numbers
        torch_rand = torch.rand(5)
        np_rand = np.random.rand(5)
        
        # Set seed again and generate same numbers
        set_seed(42)
        torch_rand2 = torch.rand(5)
        np_rand2 = np.random.rand(5)
        
        assert torch.allclose(torch_rand, torch_rand2)
        assert np.allclose(np_rand, np_rand2)
    
    def test_trainer_creation(self):
        """Test trainer creation."""
        model = create_hardware_aware_model()
        device = torch.device("cpu")
        
        trainer = HardwareAwareTrainer(
            model=model,
            device=device,
            learning_rate=0.001,
            weight_decay=1e-4
        )
        
        assert isinstance(trainer, HardwareAwareTrainer)
        assert trainer.device == device
        assert trainer.learning_rate == 0.001
        assert trainer.weight_decay == 1e-4


class TestEvaluation:
    """Test cases for evaluation utilities."""
    
    def test_evaluator_creation(self):
        """Test evaluator creation."""
        model = create_hardware_aware_model()
        device = torch.device("cpu")
        
        evaluator = HardwareAwareEvaluator(model, device)
        
        assert isinstance(evaluator, HardwareAwareEvaluator)
        assert evaluator.device == device
    
    def test_model_efficiency_calculation(self):
        """Test model efficiency calculation."""
        model = create_hardware_aware_model()
        device = torch.device("cpu")
        
        evaluator = HardwareAwareEvaluator(model, device)
        efficiency = evaluator.get_model_efficiency()
        
        assert "model_size_mb" in efficiency
        assert "total_parameters" in efficiency
        assert "estimated_flops" in efficiency
        
        assert efficiency["total_parameters"] > 0
        assert efficiency["model_size_mb"] > 0


class TestExport:
    """Test cases for model export utilities."""
    
    def test_exporter_creation(self):
        """Test exporter creation."""
        model = create_hardware_aware_model()
        device = torch.device("cpu")
        
        exporter = ModelExporter(model, device)
        
        assert isinstance(exporter, ModelExporter)
        assert exporter.device == device
    
    def test_onnx_export(self):
        """Test ONNX export."""
        model = create_hardware_aware_model()
        device = torch.device("cpu")
        
        exporter = ModelExporter(model, device)
        output_path = Path("test_model.onnx")
        
        try:
            exported_path = exporter.export_onnx(
                output_path=output_path,
                input_size=(1, 3, 96, 96)
            )
            
            assert Path(exported_path).exists()
            assert exported_path.endswith(".onnx")
        
        finally:
            # Cleanup
            if output_path.exists():
                output_path.unlink()
    
    def test_torchscript_export(self):
        """Test TorchScript export."""
        model = create_hardware_aware_model()
        device = torch.device("cpu")
        
        exporter = ModelExporter(model, device)
        output_path = Path("test_model.pt")
        
        try:
            exported_path = exporter.export_torchscript(
                output_path=output_path,
                input_size=(1, 3, 96, 96)
            )
            
            assert Path(exported_path).exists()
            assert exported_path.endswith(".pt")
        
        finally:
            # Cleanup
            if output_path.exists():
                output_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__])
