"""Data loading and preprocessing utilities for hardware-aware models."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import numpy as np
from PIL import Image


class HardwareAwareDataset(Dataset):
    """Dataset class optimized for hardware-aware training.
    
    Provides efficient data loading with hardware-constrained preprocessing.
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        input_size: Tuple[int, int] = (96, 96),
        is_training: bool = True,
        augment: bool = True
    ) -> None:
        """Initialize dataset.
        
        Args:
            data_dir: Path to dataset directory
            input_size: Target input size (height, width)
            is_training: Whether this is training data
            augment: Whether to apply data augmentation
        """
        self.data_dir = Path(data_dir)
        self.input_size = input_size
        self.is_training = is_training
        self.augment = augment
        
        # Get class names and create mapping
        self.class_names = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}
        
        # Load image paths and labels
        self.samples = self._load_samples()
        
        # Define transforms
        self.transform = self._get_transforms()
    
    def _load_samples(self) -> List[Tuple[str, int]]:
        """Load image paths and corresponding labels.
        
        Returns:
            List of (image_path, label) tuples
        """
        samples = []
        for class_name in self.class_names:
            class_dir = self.data_dir / class_name
            for img_path in class_dir.glob("*.jpg"):
                samples.append((str(img_path), self.class_to_idx[class_name]))
        return samples
    
    def _get_transforms(self) -> transforms.Compose:
        """Get data transformation pipeline.
        
        Returns:
            Composed transforms
        """
        transform_list = []
        
        if self.is_training and self.augment:
            # Training augmentations
            transform_list.extend([
                transforms.RandomResizedCrop(self.input_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ])
        else:
            # Validation/test transforms
            transform_list.append(transforms.Resize(self.input_size))
        
        # Common transforms
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return transforms.Compose(transform_list)
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get item by index.
        
        Args:
            idx: Item index
            
        Returns:
            Tuple of (image_tensor, label)
        """
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy image if loading fails
            image = torch.zeros(3, *self.input_size)
        
        return image, label


def create_synthetic_dataset(
    output_dir: Union[str, Path],
    num_classes: int = 5,
    samples_per_class: int = 100,
    image_size: Tuple[int, int] = (96, 96)
) -> None:
    """Create a synthetic dataset for testing and demonstration.
    
    Args:
        output_dir: Directory to save synthetic data
        num_classes: Number of classes to generate
        samples_per_class: Number of samples per class
        image_size: Size of generated images
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    class_names = [f"class_{i}" for i in range(num_classes)]
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = output_dir / class_name
        class_dir.mkdir(exist_ok=True)
        
        # Generate synthetic images with different patterns
        for sample_idx in range(samples_per_class):
            # Create synthetic image with class-specific patterns
            img_array = np.random.randint(0, 256, (*image_size, 3), dtype=np.uint8)
            
            # Add class-specific patterns
            if class_idx == 0:  # Vertical stripes
                img_array[:, ::10] = [255, 0, 0]
            elif class_idx == 1:  # Horizontal stripes
                img_array[::10, :] = [0, 255, 0]
            elif class_idx == 2:  # Diagonal pattern
                for i in range(image_size[0]):
                    for j in range(image_size[1]):
                        if (i + j) % 20 < 10:
                            img_array[i, j] = [0, 0, 255]
            elif class_idx == 3:  # Checkerboard
                img_array[::20, ::20] = [255, 255, 0]
                img_array[10::20, 10::20] = [255, 255, 0]
            else:  # Random noise with bias
                img_array = np.random.randint(50, 200, (*image_size, 3), dtype=np.uint8)
            
            # Save image
            img = Image.fromarray(img_array)
            img_path = class_dir / f"sample_{sample_idx:03d}.jpg"
            img.save(img_path)


def get_data_loaders(
    data_dir: Union[str, Path],
    batch_size: int = 32,
    input_size: Tuple[int, int] = (96, 96),
    num_workers: int = 4,
    train_split: float = 0.8
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders.
    
    Args:
        data_dir: Path to dataset directory
        batch_size: Batch size for data loaders
        input_size: Input image size
        num_workers: Number of worker processes
        train_split: Fraction of data to use for training
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    data_dir = Path(data_dir)
    
    # Create datasets
    train_dataset = HardwareAwareDataset(
        data_dir, 
        input_size=input_size, 
        is_training=True, 
        augment=True
    )
    
    val_dataset = HardwareAwareDataset(
        data_dir, 
        input_size=input_size, 
        is_training=False, 
        augment=False
    )
    
    # Split dataset
    train_size = int(train_split * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    train_dataset, _ = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    _, val_dataset = torch.utils.data.random_split(
        val_dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
