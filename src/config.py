"""Configuration files for hardware-aware neural networks."""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml


class Config:
    """Configuration management for hardware-aware models."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_default_config()
        
        if config_path and Path(config_path).exists():
            self._load_config(config_path)
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration."""
        return {
            "model": {
                "type": "mobilenet_v2",
                "width_multiplier": 0.35,
                "input_size": [96, 96],
                "num_classes": 5
            },
            "training": {
                "epochs": 10,
                "batch_size": 32,
                "learning_rate": 0.001,
                "weight_decay": 1e-4,
                "num_workers": 4,
                "early_stopping_patience": 5
            },
            "data": {
                "data_dir": "data/synthetic",
                "create_synthetic": True,
                "samples_per_class": 100,
                "train_split": 0.8,
                "augment": True
            },
            "hardware": {
                "device": "auto",
                "seed": 42,
                "deterministic": True
            },
            "export": {
                "formats": ["onnx", "torchscript"],
                "export_dir": "exports",
                "opset_version": 11
            },
            "evaluation": {
                "benchmark_runs": 100,
                "num_warmup": 10,
                "save_plots": True
            },
            "devices": {
                "raspberry_pi": {
                    "cpu_cores": 4,
                    "memory_mb": 1024,
                    "storage_mb": 8192,
                    "power_consumption_w": 3.5,
                    "target_fps": 10
                },
                "jetson_nano": {
                    "cpu_cores": 4,
                    "memory_mb": 4096,
                    "storage_mb": 16384,
                    "power_consumption_w": 10.0,
                    "target_fps": 30
                },
                "android": {
                    "cpu_cores": 8,
                    "memory_mb": 6144,
                    "storage_mb": 32768,
                    "power_consumption_w": 5.0,
                    "target_fps": 60
                },
                "mcu": {
                    "cpu_cores": 1,
                    "memory_mb": 256,
                    "storage_mb": 1024,
                    "power_consumption_w": 0.1,
                    "target_fps": 1
                }
            }
        }
    
    def _load_config(self, config_path: str) -> None:
        """Load configuration from file.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            file_config = yaml.safe_load(f)
        
        # Merge with default config
        self._merge_config(self.config, file_config)
    
    def _merge_config(self, default: Dict, override: Dict) -> None:
        """Recursively merge configuration dictionaries.
        
        Args:
            default: Default configuration
            override: Override configuration
        """
        for key, value in override.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                self._merge_config(default[key], value)
            else:
                default[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, output_path: str) -> None:
        """Save configuration to file.
        
        Args:
            output_path: Path to save configuration
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Configuration dictionary
        """
        return self.config.copy()


def create_device_config(device_type: str) -> Dict[str, Any]:
    """Create device-specific configuration.
    
    Args:
        device_type: Type of target device
        
    Returns:
        Device configuration dictionary
    """
    device_configs = {
        "raspberry_pi": {
            "model": {
                "width_multiplier": 0.25,
                "input_size": [64, 64]
            },
            "training": {
                "batch_size": 16,
                "learning_rate": 0.0005
            },
            "export": {
                "formats": ["onnx", "tflite"]
            }
        },
        "jetson_nano": {
            "model": {
                "width_multiplier": 0.5,
                "input_size": [128, 128]
            },
            "training": {
                "batch_size": 32,
                "learning_rate": 0.001
            },
            "export": {
                "formats": ["onnx", "tensorrt"]
            }
        },
        "android": {
            "model": {
                "width_multiplier": 0.75,
                "input_size": [224, 224]
            },
            "training": {
                "batch_size": 64,
                "learning_rate": 0.001
            },
            "export": {
                "formats": ["onnx", "coreml"]
            }
        },
        "mcu": {
            "model": {
                "width_multiplier": 0.1,
                "input_size": [32, 32]
            },
            "training": {
                "batch_size": 1,
                "learning_rate": 0.0001
            },
            "export": {
                "formats": ["tflite"]
            }
        }
    }
    
    return device_configs.get(device_type, {})


def create_training_config(
    model_type: str = "mobilenet_v2",
    device_type: str = "raspberry_pi",
    num_classes: int = 5
) -> Config:
    """Create training configuration for specific device.
    
    Args:
        model_type: Type of model architecture
        device_type: Target device type
        num_classes: Number of classes
        
    Returns:
        Configuration object
    """
    config = Config()
    
    # Set basic parameters
    config.set("model.type", model_type)
    config.set("model.num_classes", num_classes)
    
    # Apply device-specific configuration
    device_config = create_device_config(device_type)
    for section, values in device_config.items():
        for key, value in values.items():
            config.set(f"{section}.{key}", value)
    
    return config
