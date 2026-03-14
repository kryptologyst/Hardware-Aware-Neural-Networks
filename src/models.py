"""Hardware-aware neural network models and utilities."""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torchvision.models as models
from torch.quantization import quantize_dynamic, quantize_static
import numpy as np


class HardwareAwareModel(nn.Module):
    """Base class for hardware-aware neural network models.
    
    This class provides common functionality for models designed to run
    efficiently on edge devices with constrained resources.
    
    Args:
        num_classes: Number of output classes
        input_size: Input image size (height, width)
        model_type: Type of base model architecture
        width_multiplier: Width multiplier for efficient models
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        input_size: Tuple[int, int] = (96, 96),
        model_type: str = "mobilenet_v2",
        width_multiplier: float = 0.35,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.model_type = model_type
        self.width_multiplier = width_multiplier
        
        self._build_model()
    
    def _build_model(self) -> None:
        """Build the neural network model based on configuration."""
        if self.model_type == "mobilenet_v2":
            self.backbone = models.mobilenet_v2(
                pretrained=True,
                width_mult=self.width_multiplier
            )
            # Replace classifier for custom number of classes
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(self.backbone.last_channel, self.num_classes)
            )
        elif self.model_type == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(pretrained=True)
            self.backbone.classifier = nn.Linear(
                self.backbone.classifier[1].in_features, 
                self.num_classes
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        return self.backbone(x)
    
    def get_model_size(self) -> Dict[str, float]:
        """Calculate model size metrics.
        
        Returns:
            Dictionary containing model size information in MB
        """
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        total_size = param_size + buffer_size
        
        return {
            "parameters_mb": param_size / (1024 * 1024),
            "buffers_mb": buffer_size / (1024 * 1024),
            "total_mb": total_size / (1024 * 1024),
            "num_parameters": sum(p.numel() for p in self.parameters())
        }


class QuantizedModel:
    """Quantized model wrapper for hardware-aware deployment.
    
    Provides static and dynamic quantization capabilities for edge deployment.
    """
    
    def __init__(self, model: HardwareAwareModel) -> None:
        self.model = model
        self.quantized_model: Optional[nn.Module] = None
    
    def quantize_static(
        self, 
        calibration_data: torch.Tensor,
        backend: str = "qnnpack"
    ) -> nn.Module:
        """Apply static quantization to the model.
        
        Args:
            calibration_data: Representative data for calibration
            backend: Quantization backend ('qnnpack' or 'fbgemm')
            
        Returns:
            Quantized model
        """
        self.model.eval()
        
        # Set quantization configuration
        self.model.qconfig = torch.quantization.get_default_qconfig(backend)
        
        # Prepare model for quantization
        torch.quantization.prepare(self.model, inplace=True)
        
        # Calibrate with representative data
        with torch.no_grad():
            for batch in calibration_data:
                self.model(batch)
        
        # Convert to quantized model
        self.quantized_model = torch.quantization.convert(self.model, inplace=False)
        
        return self.quantized_model
    
    def quantize_dynamic(self) -> nn.Module:
        """Apply dynamic quantization to the model.
        
        Returns:
            Dynamically quantized model
        """
        self.quantized_model = quantize_dynamic(
            self.model, 
            {nn.Linear, nn.Conv2d}, 
            dtype=torch.qint8
        )
        
        return self.quantized_model


def create_hardware_aware_model(
    num_classes: int = 5,
    model_type: str = "mobilenet_v2",
    width_multiplier: float = 0.35,
    input_size: Tuple[int, int] = (96, 96)
) -> HardwareAwareModel:
    """Factory function to create hardware-aware models.
    
    Args:
        num_classes: Number of output classes
        model_type: Type of model architecture
        width_multiplier: Width multiplier for efficient models
        input_size: Input image size
        
    Returns:
        Configured hardware-aware model
    """
    return HardwareAwareModel(
        num_classes=num_classes,
        input_size=input_size,
        model_type=model_type,
        width_multiplier=width_multiplier
    )
