"""Model export and deployment utilities for edge devices."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import numpy as np


class ModelExporter:
    """Exporter for converting PyTorch models to edge deployment formats."""
    
    def __init__(self, model: nn.Module, device: torch.device) -> None:
        """Initialize exporter.
        
        Args:
            model: PyTorch model to export
            device: Device to run export on
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def export_onnx(
        self,
        output_path: Union[str, Path],
        input_size: tuple = (1, 3, 96, 96),
        opset_version: int = 11
    ) -> str:
        """Export model to ONNX format.
        
        Args:
            output_path: Path to save ONNX model
            input_size: Input tensor size (batch, channels, height, width)
            opset_version: ONNX opset version
            
        Returns:
            Path to exported ONNX model
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create dummy input
        dummy_input = torch.randn(input_size).to(self.device)
        
        # Export to ONNX
        torch.onnx.export(
            self.model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        print(f"ONNX model exported to: {output_path}")
        return str(output_path)
    
    def export_torchscript(
        self,
        output_path: Union[str, Path],
        input_size: tuple = (1, 3, 96, 96)
    ) -> str:
        """Export model to TorchScript format.
        
        Args:
            output_path: Path to save TorchScript model
            input_size: Input tensor size
            
        Returns:
            Path to exported TorchScript model
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create dummy input
        dummy_input = torch.randn(input_size).to(self.device)
        
        # Trace the model
        traced_model = torch.jit.trace(self.model, dummy_input)
        
        # Save traced model
        traced_model.save(str(output_path))
        
        print(f"TorchScript model exported to: {output_path}")
        return str(output_path)
    
    def export_quantized_onnx(
        self,
        output_path: Union[str, Path],
        calibration_data: torch.Tensor,
        input_size: tuple = (1, 3, 96, 96)
    ) -> str:
        """Export quantized model to ONNX format.
        
        Args:
            output_path: Path to save quantized ONNX model
            calibration_data: Data for quantization calibration
            input_size: Input tensor size
            
        Returns:
            Path to exported quantized ONNX model
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create quantized model
        quantized_model = torch.quantization.quantize_dynamic(
            self.model, 
            {nn.Linear, nn.Conv2d}, 
            dtype=torch.qint8
        )
        
        # Create dummy input
        dummy_input = torch.randn(input_size).to(self.device)
        
        # Export quantized model
        torch.onnx.export(
            quantized_model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output']
        )
        
        print(f"Quantized ONNX model exported to: {output_path}")
        return str(output_path)


def create_deployment_config(
    model_path: Union[str, Path],
    model_type: str = "onnx",
    device_config: Optional[Dict] = None
) -> Dict:
    """Create deployment configuration for edge devices.
    
    Args:
        model_path: Path to exported model
        model_type: Type of model (onnx, torchscript, etc.)
        device_config: Device-specific configuration
        
    Returns:
        Deployment configuration dictionary
    """
    if device_config is None:
        device_config = {
            "raspberry_pi": {
                "cpu_cores": 4,
                "memory_mb": 1024,
                "storage_mb": 8192,
                "power_consumption_w": 3.5
            },
            "jetson_nano": {
                "cpu_cores": 4,
                "memory_mb": 4096,
                "storage_mb": 16384,
                "power_consumption_w": 10.0
            },
            "android": {
                "cpu_cores": 8,
                "memory_mb": 6144,
                "storage_mb": 32768,
                "power_consumption_w": 5.0
            }
        }
    
    config = {
        "model": {
            "path": str(model_path),
            "type": model_type,
            "size_mb": Path(model_path).stat().st_size / (1024 * 1024)
        },
        "devices": device_config,
        "runtime": {
            "onnx": "onnxruntime",
            "torchscript": "torch",
            "tflite": "tflite_runtime"
        },
        "optimization": {
            "quantization": True,
            "pruning": False,
            "distillation": False
        }
    }
    
    return config


def generate_deployment_script(
    config: Dict,
    output_path: Union[str, Path]
) -> str:
    """Generate deployment script for edge devices.
    
    Args:
        config: Deployment configuration
        output_path: Path to save deployment script
        
    Returns:
        Path to generated script
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    script_content = f'''#!/usr/bin/env python3
"""
Edge deployment script for hardware-aware neural network.
Generated automatically from deployment configuration.
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Add model path to Python path
sys.path.append(str(Path(__file__).parent))

def main():
    parser = argparse.ArgumentParser(description='Edge AI Model Deployment')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to the model file')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda', 'mps'],
                       help='Device to run inference on')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for inference')
    parser.add_argument('--num-runs', type=int, default=100,
                       help='Number of inference runs for benchmarking')
    
    args = parser.parse_args()
    
    print("Edge AI Model Deployment")
    print("=" * 40)
    print(f"Model: {{args.model_path}}")
    print(f"Device: {{args.device}}")
    print(f"Batch Size: {{args.batch_size}}")
    print(f"Runs: {{args.num_runs}}")
    print("=" * 40)
    
    # Load model based on type
    model_type = config["model"]["type"]
    
    if model_type == "onnx":
        import onnxruntime as ort
        session = ort.InferenceSession(args.model_path)
        print("ONNX Runtime session created")
        
    elif model_type == "torchscript":
        import torch
        model = torch.jit.load(args.model_path)
        model.eval()
        print("TorchScript model loaded")
        
    else:
        raise ValueError(f"Unsupported model type: {{model_type}}")
    
    # Benchmark inference
    print("\\nStarting inference benchmark...")
    times = []
    
    for i in range(args.num_runs):
        # Create dummy input
        dummy_input = np.random.randn(args.batch_size, 3, 96, 96).astype(np.float32)
        
        start_time = time.time()
        
        if model_type == "onnx":
            outputs = session.run(None, {{"input": dummy_input}})
        elif model_type == "torchscript":
            with torch.no_grad():
                outputs = model(torch.from_numpy(dummy_input))
        
        end_time = time.time()
        times.append(end_time - start_time)
        
        if (i + 1) % 10 == 0:
            print(f"Completed {{i + 1}}/{{args.num_runs}} runs")
    
    # Calculate statistics
    times = np.array(times)
    mean_time = np.mean(times)
    std_time = np.std(times)
    
    print("\\nBenchmark Results:")
    print(f"Mean inference time: {{mean_time * 1000:.2f}} ms")
    print(f"Std inference time: {{std_time * 1000:.2f}} ms")
    print(f"Throughput: {{1.0 / mean_time:.2f}} FPS")
    print(f"Model size: {{config['model']['size_mb']:.2f}} MB")

if __name__ == "__main__":
    main()
'''
    
    with open(output_path, 'w') as f:
        f.write(script_content)
    
    # Make script executable
    os.chmod(output_path, 0o755)
    
    print(f"Deployment script generated: {output_path}")
    return str(output_path)
