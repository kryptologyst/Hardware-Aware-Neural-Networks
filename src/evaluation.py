"""Evaluation and benchmarking utilities for hardware-aware models."""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class HardwareAwareEvaluator:
    """Evaluator for hardware-aware neural networks.
    
    Provides comprehensive evaluation including accuracy, efficiency, and hardware metrics.
    """
    
    def __init__(self, model: nn.Module, device: torch.device) -> None:
        """Initialize evaluator.
        
        Args:
            model: Neural network model to evaluate
            device: Device to run evaluation on
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def evaluate_accuracy(
        self, 
        data_loader: DataLoader,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Evaluate model accuracy.
        
        Args:
            data_loader: Data loader for evaluation
            class_names: List of class names for reporting
            
        Returns:
            Dictionary containing accuracy metrics
        """
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1)
                
                correct += pred.eq(target).sum().item()
                total += target.size(0)
                
                all_predictions.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        accuracy = 100. * correct / total
        
        # Calculate additional metrics
        metrics = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total
        }
        
        # Classification report
        if class_names:
            report = classification_report(
                all_targets, 
                all_predictions, 
                target_names=class_names,
                output_dict=True
            )
            metrics.update(report)
        
        return metrics
    
    def benchmark_inference(
        self,
        data_loader: DataLoader,
        num_warmup: int = 10,
        num_runs: int = 100
    ) -> Dict[str, float]:
        """Benchmark inference performance.
        
        Args:
            data_loader: Data loader for benchmarking
            num_warmup: Number of warmup runs
            num_runs: Number of benchmark runs
            
        Returns:
            Dictionary containing performance metrics
        """
        # Warmup
        with torch.no_grad():
            for i, (data, _) in enumerate(data_loader):
                if i >= num_warmup:
                    break
                data = data.to(self.device)
                _ = self.model(data)
        
        # Benchmark
        times = []
        memory_usage = []
        
        with torch.no_grad():
            for i, (data, _) in enumerate(data_loader):
                if i >= num_runs:
                    break
                
                data = data.to(self.device)
                
                # Measure time
                start_time = time.time()
                _ = self.model(data)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                
                times.append(end_time - start_time)
                
                # Measure memory (if CUDA available)
                if torch.cuda.is_available():
                    memory_usage.append(torch.cuda.max_memory_allocated() / 1024**2)  # MB
        
        # Calculate statistics
        times = np.array(times)
        memory_usage = np.array(memory_usage) if memory_usage else np.array([0])
        
        metrics = {
            "mean_latency_ms": np.mean(times) * 1000,
            "std_latency_ms": np.std(times) * 1000,
            "p50_latency_ms": np.percentile(times, 50) * 1000,
            "p95_latency_ms": np.percentile(times, 95) * 1000,
            "p99_latency_ms": np.percentile(times, 99) * 1000,
            "min_latency_ms": np.min(times) * 1000,
            "max_latency_ms": np.max(times) * 1000,
            "throughput_fps": 1.0 / np.mean(times),
            "mean_memory_mb": np.mean(memory_usage),
            "max_memory_mb": np.max(memory_usage)
        }
        
        return metrics
    
    def get_model_efficiency(self) -> Dict[str, float]:
        """Calculate model efficiency metrics.
        
        Returns:
            Dictionary containing model efficiency metrics
        """
        # Model size
        param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
        total_size = param_size + buffer_size
        
        # Parameter count
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # FLOPs estimation (approximate)
        def count_flops(model, input_size=(1, 3, 96, 96)):
            flops = 0
            for module in model.modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    if isinstance(module, nn.Conv2d):
                        # Approximate FLOPs for Conv2d
                        output_elements = module.out_channels * (input_size[2] // module.stride[0]) * (input_size[3] // module.stride[1])
                        kernel_flops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels
                        flops += output_elements * kernel_flops
                    else:
                        # FLOPs for Linear
                        flops += module.in_features * module.out_features
            return flops
        
        try:
            estimated_flops = count_flops(self.model)
        except:
            estimated_flops = 0
        
        return {
            "model_size_mb": total_size / (1024 * 1024),
            "parameters_mb": param_size / (1024 * 1024),
            "buffers_mb": buffer_size / (1024 * 1024),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "estimated_flops": estimated_flops,
            "parameters_per_mb": total_params / (total_size / (1024 * 1024)) if total_size > 0 else 0
        }
    
    def create_confusion_matrix(
        self,
        data_loader: DataLoader,
        class_names: List[str],
        save_path: Optional[Union[str, Path]] = None
    ) -> np.ndarray:
        """Create and optionally save confusion matrix.
        
        Args:
            data_loader: Data loader for evaluation
            class_names: List of class names
            save_path: Path to save confusion matrix plot
            
        Returns:
            Confusion matrix array
        """
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1)
                
                all_predictions.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        cm = confusion_matrix(all_targets, all_predictions)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return cm
    
    def comprehensive_evaluation(
        self,
        data_loader: DataLoader,
        class_names: List[str],
        benchmark_runs: int = 100
    ) -> Dict[str, Union[float, Dict]]:
        """Perform comprehensive model evaluation.
        
        Args:
            data_loader: Data loader for evaluation
            class_names: List of class names
            benchmark_runs: Number of runs for benchmarking
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        print("Starting comprehensive evaluation...")
        
        # Accuracy evaluation
        print("Evaluating accuracy...")
        accuracy_metrics = self.evaluate_accuracy(data_loader, class_names)
        
        # Performance benchmarking
        print("Benchmarking inference performance...")
        performance_metrics = self.benchmark_inference(data_loader, num_runs=benchmark_runs)
        
        # Model efficiency
        print("Calculating model efficiency...")
        efficiency_metrics = self.get_model_efficiency()
        
        # Combine all metrics
        results = {
            "accuracy": accuracy_metrics,
            "performance": performance_metrics,
            "efficiency": efficiency_metrics
        }
        
        # Print summary
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Accuracy: {accuracy_metrics['accuracy']:.2f}%")
        print(f"Mean Latency: {performance_metrics['mean_latency_ms']:.2f} ms")
        print(f"Throughput: {performance_metrics['throughput_fps']:.2f} FPS")
        print(f"Model Size: {efficiency_metrics['model_size_mb']:.2f} MB")
        print(f"Parameters: {efficiency_metrics['total_parameters']:,}")
        print("="*60)
        
        return results
