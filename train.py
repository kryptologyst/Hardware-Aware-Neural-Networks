"""Main training script for hardware-aware neural networks."""

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models import create_hardware_aware_model, QuantizedModel
from src.data import get_data_loaders, create_synthetic_dataset
from src.training import HardwareAwareTrainer, set_seed, get_device
from src.evaluation import HardwareAwareEvaluator
from src.export import ModelExporter, create_deployment_config


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Hardware-Aware Neural Network Training")
    
    # Data arguments
    parser.add_argument("--data-dir", type=str, default="data/synthetic",
                       help="Path to dataset directory")
    parser.add_argument("--create-synthetic", action="store_true",
                       help="Create synthetic dataset if data directory doesn't exist")
    parser.add_argument("--num-classes", type=int, default=5,
                       help="Number of classes")
    parser.add_argument("--samples-per-class", type=int, default=100,
                       help="Number of samples per class for synthetic data")
    
    # Model arguments
    parser.add_argument("--model-type", type=str, default="mobilenet_v2",
                       choices=["mobilenet_v2", "efficientnet_b0"],
                       help="Type of model architecture")
    parser.add_argument("--width-multiplier", type=float, default=0.35,
                       help="Width multiplier for efficient models")
    parser.add_argument("--input-size", type=int, nargs=2, default=[96, 96],
                       help="Input image size (height width)")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                       help="Weight decay")
    parser.add_argument("--num-workers", type=int, default=4,
                       help="Number of data loader workers")
    
    # Hardware arguments
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cpu", "cuda", "mps"],
                       help="Device to train on")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    # Export arguments
    parser.add_argument("--export-formats", type=str, nargs="+", 
                       default=["onnx", "torchscript"],
                       choices=["onnx", "torchscript", "quantized_onnx"],
                       help="Formats to export model to")
    parser.add_argument("--export-dir", type=str, default="exports",
                       help="Directory to save exported models")
    
    # Evaluation arguments
    parser.add_argument("--benchmark-runs", type=int, default=100,
                       help="Number of runs for performance benchmarking")
    
    return parser.parse_args()


def main() -> None:
    """Main training function."""
    args = parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Get device
    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)
    
    print("Hardware-Aware Neural Network Training")
    print("=" * 50)
    print(f"Device: {device}")
    print(f"Model: {args.model_type} (width_mult={args.width_multiplier})")
    print(f"Input size: {args.input_size}")
    print(f"Classes: {args.num_classes}")
    print("=" * 50)
    
    # Create data directory and synthetic dataset if needed
    data_dir = Path(args.data_dir)
    if not data_dir.exists() and args.create_synthetic:
        print("Creating synthetic dataset...")
        create_synthetic_dataset(
            output_dir=data_dir,
            num_classes=args.num_classes,
            samples_per_class=args.samples_per_class,
            image_size=tuple(args.input_size)
        )
        print(f"Synthetic dataset created at: {data_dir}")
    
    # Load data
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    print("Loading data...")
    train_loader, val_loader = get_data_loaders(
        data_dir=data_dir,
        batch_size=args.batch_size,
        input_size=tuple(args.input_size),
        num_workers=args.num_workers
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    print("Creating model...")
    model = create_hardware_aware_model(
        num_classes=args.num_classes,
        model_type=args.model_type,
        width_multiplier=args.width_multiplier,
        input_size=tuple(args.input_size)
    )
    
    # Print model info
    model_size = model.get_model_size()
    print(f"Model parameters: {model_size['num_parameters']:,}")
    print(f"Model size: {model_size['total_mb']:.2f} MB")
    
    # Create trainer
    trainer = HardwareAwareTrainer(
        model=model,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Train model
    print("Starting training...")
    save_path = Path("models") / f"best_model_{args.model_type}.pth"
    save_path.parent.mkdir(exist_ok=True)
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        save_path=save_path
    )
    
    # Load best model for evaluation
    model.load_state_dict(torch.load(save_path))
    
    # Evaluate model
    print("Evaluating model...")
    evaluator = HardwareAwareEvaluator(model, device)
    
    # Get class names
    class_names = [f"class_{i}" for i in range(args.num_classes)]
    
    # Comprehensive evaluation
    results = evaluator.comprehensive_evaluation(
        data_loader=val_loader,
        class_names=class_names,
        benchmark_runs=args.benchmark_runs
    )
    
    # Save evaluation results
    results_path = Path("results") / f"evaluation_{args.model_type}.json"
    results_path.parent.mkdir(exist_ok=True)
    
    # Convert numpy types to Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    def recursive_convert(obj):
        if isinstance(obj, dict):
            return {k: recursive_convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [recursive_convert(item) for item in obj]
        else:
            return convert_numpy(obj)
    
    with open(results_path, 'w') as f:
        json.dump(recursive_convert(results), f, indent=2)
    
    print(f"Evaluation results saved to: {results_path}")
    
    # Export models
    if args.export_formats:
        print("Exporting models...")
        export_dir = Path(args.export_dir)
        export_dir.mkdir(exist_ok=True)
        
        exporter = ModelExporter(model, device)
        
        for format_type in args.export_formats:
            if format_type == "onnx":
                onnx_path = export_dir / f"model_{args.model_type}.onnx"
                exporter.export_onnx(onnx_path, input_size=(1, 3, *args.input_size))
                
            elif format_type == "torchscript":
                torchscript_path = export_dir / f"model_{args.model_type}.pt"
                exporter.export_torchscript(torchscript_path, input_size=(1, 3, *args.input_size))
                
            elif format_type == "quantized_onnx":
                # Use a subset of validation data for calibration
                calibration_data = []
                for i, (data, _) in enumerate(val_loader):
                    if i >= 10:  # Use first 10 batches for calibration
                        break
                    calibration_data.append(data)
                
                quantized_onnx_path = export_dir / f"model_{args.model_type}_quantized.onnx"
                exporter.export_quantized_onnx(
                    quantized_onnx_path, 
                    calibration_data,
                    input_size=(1, 3, *args.input_size)
                )
        
        # Create deployment configuration
        config = create_deployment_config(
            model_path=export_dir / f"model_{args.model_type}.onnx",
            model_type="onnx"
        )
        
        config_path = export_dir / "deployment_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Models exported to: {export_dir}")
        print(f"Deployment config saved to: {config_path}")
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
