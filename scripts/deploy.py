#!/usr/bin/env python3
"""Deployment script for hardware-aware neural networks on edge devices."""

import os
import sys
import time
import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import numpy as np


def load_model(model_path: str, model_type: str = "onnx") -> Any:
    """Load model based on type.
    
    Args:
        model_path: Path to model file
        model_type: Type of model (onnx, torchscript, etc.)
        
    Returns:
        Loaded model object
    """
    if model_type == "onnx":
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(model_path)
            print(f"ONNX Runtime session created for: {model_path}")
            return session
        except ImportError:
            print("ONNX Runtime not available. Install with: pip install onnxruntime")
            return None
    
    elif model_type == "torchscript":
        try:
            model = torch.jit.load(model_path)
            model.eval()
            print(f"TorchScript model loaded: {model_path}")
            return model
        except Exception as e:
            print(f"Error loading TorchScript model: {e}")
            return None
    
    else:
        print(f"Unsupported model type: {model_type}")
        return None


def benchmark_inference(
    model: Any,
    model_type: str,
    input_size: tuple = (1, 3, 96, 96),
    num_runs: int = 100,
    num_warmup: int = 10
) -> Dict[str, float]:
    """Benchmark model inference performance.
    
    Args:
        model: Loaded model object
        model_type: Type of model
        input_size: Input tensor size
        num_runs: Number of benchmark runs
        num_warmup: Number of warmup runs
        
    Returns:
        Performance metrics dictionary
    """
    if model is None:
        return {}
    
    # Create dummy input
    dummy_input = np.random.randn(*input_size).astype(np.float32)
    
    # Warmup runs
    print(f"Running {num_warmup} warmup iterations...")
    for i in range(num_warmup):
        if model_type == "onnx":
            _ = model.run(None, {"input": dummy_input})
        elif model_type == "torchscript":
            with torch.no_grad():
                _ = model(torch.from_numpy(dummy_input))
    
    # Benchmark runs
    print(f"Running {num_runs} benchmark iterations...")
    times = []
    
    for i in range(num_runs):
        start_time = time.time()
        
        if model_type == "onnx":
            outputs = model.run(None, {"input": dummy_input})
        elif model_type == "torchscript":
            with torch.no_grad():
                outputs = model(torch.from_numpy(dummy_input))
        
        end_time = time.time()
        times.append(end_time - start_time)
        
        if (i + 1) % 20 == 0:
            print(f"Completed {i + 1}/{num_runs} runs")
    
    # Calculate statistics
    times = np.array(times)
    
    metrics = {
        "mean_latency_ms": np.mean(times) * 1000,
        "std_latency_ms": np.std(times) * 1000,
        "p50_latency_ms": np.percentile(times, 50) * 1000,
        "p95_latency_ms": np.percentile(times, 95) * 1000,
        "p99_latency_ms": np.percentile(times, 99) * 1000,
        "min_latency_ms": np.min(times) * 1000,
        "max_latency_ms": np.max(times) * 1000,
        "throughput_fps": 1.0 / np.mean(times)
    }
    
    return metrics


def get_system_info() -> Dict[str, Any]:
    """Get system information.
    
    Returns:
        System information dictionary
    """
    import platform
    import psutil
    
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": psutil.cpu_count(),
        "memory_total_gb": psutil.virtual_memory().total / (1024**3),
        "memory_available_gb": psutil.virtual_memory().available / (1024**3)
    }
    
    # Add PyTorch info if available
    try:
        info["pytorch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_count"] = torch.cuda.device_count()
    except:
        pass
    
    return info


def main():
    """Main deployment script."""
    parser = argparse.ArgumentParser(description="Edge AI Model Deployment")
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to the model file")
    parser.add_argument("--model-type", type=str, default="onnx",
                       choices=["onnx", "torchscript"],
                       help="Type of model file")
    parser.add_argument("--input-size", type=int, nargs=4, default=[1, 3, 96, 96],
                       help="Input tensor size (batch, channels, height, width)")
    parser.add_argument("--num-runs", type=int, default=100,
                       help="Number of inference runs for benchmarking")
    parser.add_argument("--num-warmup", type=int, default=10,
                       help="Number of warmup runs")
    parser.add_argument("--output-file", type=str,
                       help="Output file to save benchmark results")
    parser.add_argument("--config", type=str,
                       help="Deployment configuration file")
    
    args = parser.parse_args()
    
    print("Edge AI Model Deployment")
    print("=" * 50)
    print(f"Model: {args.model_path}")
    print(f"Type: {args.model_type}")
    print(f"Input Size: {args.input_size}")
    print(f"Runs: {args.num_runs}")
    print("=" * 50)
    
    # Check if model file exists
    if not Path(args.model_path).exists():
        print(f"Error: Model file not found: {args.model_path}")
        sys.exit(1)
    
    # Get system information
    print("System Information:")
    system_info = get_system_info()
    for key, value in system_info.items():
        print(f"  {key}: {value}")
    print()
    
    # Load model
    print("Loading model...")
    model = load_model(args.model_path, args.model_type)
    
    if model is None:
        print("Failed to load model. Exiting.")
        sys.exit(1)
    
    # Run benchmark
    print("Starting inference benchmark...")
    metrics = benchmark_inference(
        model=model,
        model_type=args.model_type,
        input_size=tuple(args.input_size),
        num_runs=args.num_runs,
        num_warmup=args.num_warmup
    )
    
    # Display results
    print("\nBenchmark Results:")
    print("-" * 30)
    for key, value in metrics.items():
        if "latency" in key:
            print(f"{key}: {value:.2f} ms")
        elif "fps" in key:
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    
    # Save results
    if args.output_file:
        results = {
            "model_path": args.model_path,
            "model_type": args.model_type,
            "input_size": args.input_size,
            "system_info": system_info,
            "metrics": metrics,
            "timestamp": time.time()
        }
        
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
    
    print("\nDeployment benchmark completed!")


if __name__ == "__main__":
    main()
