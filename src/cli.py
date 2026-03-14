"""Command-line interface for hardware-aware neural networks."""

import argparse
import sys
from pathlib import Path

from src.config import Config, create_training_config
from src.models import create_hardware_aware_model
from src.data import create_synthetic_dataset, get_data_loaders
from src.training import HardwareAwareTrainer, set_seed, get_device
from src.evaluation import HardwareAwareEvaluator
from src.export import ModelExporter


def train_command(args):
    """Train a hardware-aware model."""
    # Load configuration
    if args.config:
        config = Config(args.config)
    else:
        config = create_training_config(
            model_type=args.model_type,
            device_type=args.device_type,
            num_classes=args.num_classes
        )
    
    # Set random seed
    set_seed(config.get("hardware.seed", 42))
    
    # Get device
    device = get_device()
    
    # Create data
    data_dir = Path(config.get("data.data_dir", "data/synthetic"))
    if not data_dir.exists() and config.get("data.create_synthetic", True):
        print("Creating synthetic dataset...")
        create_synthetic_dataset(
            output_dir=data_dir,
            num_classes=config.get("model.num_classes", 5),
            samples_per_class=config.get("data.samples_per_class", 100),
            image_size=tuple(config.get("model.input_size", [96, 96]))
        )
    
    # Load data
    train_loader, val_loader = get_data_loaders(
        data_dir=data_dir,
        batch_size=config.get("training.batch_size", 32),
        input_size=tuple(config.get("model.input_size", [96, 96])),
        num_workers=config.get("training.num_workers", 4)
    )
    
    # Create model
    model = create_hardware_aware_model(
        num_classes=config.get("model.num_classes", 5),
        model_type=config.get("model.type", "mobilenet_v2"),
        width_multiplier=config.get("model.width_multiplier", 0.35),
        input_size=tuple(config.get("model.input_size", [96, 96]))
    )
    
    # Create trainer
    trainer = HardwareAwareTrainer(
        model=model,
        device=device,
        learning_rate=config.get("training.learning_rate", 0.001),
        weight_decay=config.get("training.weight_decay", 1e-4)
    )
    
    # Train model
    save_path = Path("models") / f"best_model_{config.get('model.type', 'mobilenet_v2')}.pth"
    save_path.parent.mkdir(exist_ok=True)
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.get("training.epochs", 10),
        save_path=save_path,
        early_stopping_patience=config.get("training.early_stopping_patience", 5)
    )
    
    print(f"Training completed! Model saved to: {save_path}")


def export_command(args):
    """Export a trained model."""
    # Load model
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        sys.exit(1)
    
    # Create model and load weights
    model = create_hardware_aware_model()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    
    # Get device
    device = get_device()
    
    # Create exporter
    exporter = ModelExporter(model, device)
    
    # Export models
    export_dir = Path(args.output_dir)
    export_dir.mkdir(exist_ok=True)
    
    for format_type in args.formats:
        if format_type == "onnx":
            output_path = export_dir / f"model.onnx"
            exporter.export_onnx(output_path, input_size=(1, 3, 96, 96))
        elif format_type == "torchscript":
            output_path = export_dir / f"model.pt"
            exporter.export_torchscript(output_path, input_size=(1, 3, 96, 96))
        else:
            print(f"Unsupported format: {format_type}")
    
    print(f"Models exported to: {export_dir}")


def evaluate_command(args):
    """Evaluate a trained model."""
    # Load model
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        sys.exit(1)
    
    # Create model and load weights
    model = create_hardware_aware_model()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    
    # Get device
    device = get_device()
    
    # Load data
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        sys.exit(1)
    
    _, val_loader = get_data_loaders(
        data_dir=data_dir,
        batch_size=32,
        input_size=(96, 96),
        num_workers=4
    )
    
    # Create evaluator
    evaluator = HardwareAwareEvaluator(model, device)
    
    # Run evaluation
    class_names = [f"class_{i}" for i in range(5)]
    results = evaluator.comprehensive_evaluation(
        data_loader=val_loader,
        class_names=class_names,
        benchmark_runs=args.benchmark_runs
    )
    
    print("Evaluation completed!")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Hardware-Aware Neural Networks CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--config", type=str, help="Configuration file path")
    train_parser.add_argument("--model-type", type=str, default="mobilenet_v2",
                            choices=["mobilenet_v2", "efficientnet_b0"])
    train_parser.add_argument("--device-type", type=str, default="raspberry_pi",
                            choices=["raspberry_pi", "jetson_nano", "android", "mcu"])
    train_parser.add_argument("--num-classes", type=int, default=5)
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export a trained model")
    export_parser.add_argument("--model-path", type=str, required=True,
                              help="Path to trained model")
    export_parser.add_argument("--output-dir", type=str, default="exports",
                              help="Output directory for exported models")
    export_parser.add_argument("--formats", type=str, nargs="+",
                              default=["onnx", "torchscript"],
                              choices=["onnx", "torchscript", "quantized_onnx"])
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    eval_parser.add_argument("--model-path", type=str, required=True,
                           help="Path to trained model")
    eval_parser.add_argument("--data-dir", type=str, required=True,
                           help="Path to evaluation data")
    eval_parser.add_argument("--benchmark-runs", type=int, default=100,
                           help="Number of benchmark runs")
    
    args = parser.parse_args()
    
    if args.command == "train":
        train_command(args)
    elif args.command == "export":
        export_command(args)
    elif args.command == "evaluate":
        evaluate_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
