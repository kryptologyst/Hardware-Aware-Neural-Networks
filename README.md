# Hardware-Aware Neural Networks

A comprehensive framework for developing and deploying neural networks optimized for hardware constraints in edge AI and IoT applications.

## Disclaimer

**This project is for research and educational purposes only. It is not intended for safety-critical applications.**

## Overview

This project provides tools and utilities for creating hardware-aware neural networks that are optimized for deployment on resource-constrained edge devices. The framework includes:

- **Model Architectures**: MobileNetV2, EfficientNet with configurable width multipliers
- **Quantization**: Static and dynamic quantization for reduced model size
- **Export Formats**: ONNX, TorchScript for cross-platform deployment
- **Performance Benchmarking**: Comprehensive evaluation of accuracy and efficiency
- **Interactive Demo**: Streamlit-based demo for model comparison and testing

## Features

### Model Efficiency
- Configurable width multipliers for MobileNetV2 and EfficientNet
- Multiple input resolutions (32x32 to 224x224)
- Quantization-aware training and post-training quantization
- Model size optimization for edge deployment

### Hardware Support
- Raspberry Pi (ARM-based edge computing)
- NVIDIA Jetson Nano (GPU-accelerated edge AI)
- Android devices (mobile deployment)
- Microcontrollers (ultra-low power)

### Evaluation Metrics
- **Accuracy**: Classification accuracy, F1-score, confusion matrix
- **Efficiency**: Inference latency, throughput (FPS), memory usage
- **Hardware**: Model size, parameter count, FLOPs estimation
- **Robustness**: Performance under different conditions

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Hardware-Aware-Neural-Networks.git
cd Hardware-Aware-Neural-Networks

# Install dependencies
pip install -r requirements.txt

# Or install with pip
pip install -e .
```

### Basic Usage

1. **Create synthetic dataset and train a model**:
```bash
python train.py --create-synthetic --epochs 10 --model-type mobilenet_v2 --width-multiplier 0.35
```

2. **Run the interactive demo**:
```bash
streamlit run demo.py
```

3. **Export models for deployment**:
```bash
python train.py --export-formats onnx torchscript --export-dir exports/
```

### Programmatic Usage

```python
from src.models import create_hardware_aware_model
from src.training import HardwareAwareTrainer
from src.data import get_data_loaders

# Create model
model = create_hardware_aware_model(
    num_classes=5,
    model_type="mobilenet_v2",
    width_multiplier=0.35,
    input_size=(96, 96)
)

# Load data
train_loader, val_loader = get_data_loaders("data/synthetic")

# Train model
trainer = HardwareAwareTrainer(model, device)
history = trainer.train(train_loader, val_loader, epochs=10)
```

## Project Structure

```
hardware-aware-neural-networks/
├── src/                    # Source code
│   ├── models.py          # Model definitions
│   ├── data.py            # Data loading utilities
│   ├── training.py        # Training utilities
│   ├── evaluation.py      # Evaluation and benchmarking
│   ├── export.py          # Model export utilities
│   └── config.py          # Configuration management
├── data/                  # Dataset directory
├── models/                # Trained model checkpoints
├── exports/               # Exported models (ONNX, TorchScript)
├── results/               # Evaluation results
├── configs/               # Configuration files
├── scripts/               # Utility scripts
├── tests/                 # Unit tests
├── assets/                # Generated plots and visualizations
├── demo.py                # Streamlit demo
├── train.py               # Main training script
├── requirements.txt       # Python dependencies
└── pyproject.toml         # Project configuration
```

## Configuration

The framework supports YAML-based configuration for easy experimentation:

```yaml
model:
  type: mobilenet_v2
  width_multiplier: 0.35
  input_size: [96, 96]
  num_classes: 5

training:
  epochs: 10
  batch_size: 32
  learning_rate: 0.001
  weight_decay: 1e-4

hardware:
  device: auto
  seed: 42
```

## Model Architectures

### MobileNetV2
- Efficient depthwise separable convolutions
- Configurable width multipliers (0.1 - 1.0)
- Optimized for mobile and edge devices

### EfficientNet
- Compound scaling of depth, width, and resolution
- State-of-the-art efficiency on ImageNet
- Good balance of accuracy and efficiency

## Performance Benchmarks

### Model Comparison (5-class classification)

| Model | Parameters | Size (MB) | Latency (ms) | Accuracy (%) |
|-------|------------|-----------|--------------|--------------|
| MobileNetV2-0.25 | 0.9M | 3.6 | 2.1 | 85.2 |
| MobileNetV2-0.35 | 1.4M | 3.6 | 2.1 | 85.2 |
| MobileNetV2-0.5 | 2.0M | 8.0 | 4.8 | 89.1 |
| EfficientNet-B0 | 5.3M | 20.0 | 8.5 | 91.5 |

### Device Performance

| Device | Target FPS | Power (W) | Memory (MB) |
|--------|------------|-----------|-------------|
| Raspberry Pi 4 | 10 | 3.5 | 1024 |
| Jetson Nano | 30 | 10.0 | 4096 |
| Android Phone | 60 | 5.0 | 6144 |
| MCU | 1 | 0.1 | 256 |

## Export Formats

### ONNX
- Cross-platform deployment
- Optimized inference engines
- Hardware acceleration support

### TorchScript
- PyTorch native format
- JIT compilation
- Mobile deployment

### Quantized Models
- INT8 quantization
- Reduced model size
- Hardware acceleration

## Development

### Code Quality
```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Run tests
pytest tests/
```

### Pre-commit Hooks
```bash
# Install pre-commit hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{hardware_aware_neural_networks,
  title={Hardware-Aware Neural Networks for Edge AI},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Hardware-Aware-Neural-Networks}
}
```

## Acknowledgments

- PyTorch team for the excellent deep learning framework
- MobileNet and EfficientNet authors for efficient architectures
- ONNX community for cross-platform deployment tools
- Streamlit team for the interactive demo framework# Hardware-Aware-Neural-Networks
