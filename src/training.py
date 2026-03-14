"""Training utilities for hardware-aware neural networks."""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm


class HardwareAwareTrainer:
    """Trainer class for hardware-aware neural networks.
    
    Provides training functionality with hardware constraints and efficiency metrics.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4
    ) -> None:
        """Initialize trainer.
        
        Args:
            model: Neural network model to train
            device: Device to train on (CPU/GPU)
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
        """
        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Move model to device
        self.model.to(device)
        
        # Initialize optimizer and loss function
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "epoch_times": []
        }
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Acc": f"{100. * correct / total:.2f}%"
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc="Validation"):
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        save_path: Optional[Union[str, Path]] = None,
        early_stopping_patience: int = 5
    ) -> Dict[str, List[float]]:
        """Train the model for multiple epochs.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs to train
            save_path: Path to save best model
            early_stopping_patience: Patience for early stopping
            
        Returns:
            Training history dictionary
        """
        best_val_acc = 0.0
        patience_counter = 0
        
        print(f"Starting training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Calculate epoch time
            epoch_time = time.time() - start_time
            
            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["epoch_times"].append(epoch_time)
            
            # Print progress
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"  Time: {epoch_time:.2f}s")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                if save_path:
                    torch.save(self.model.state_dict(), save_path)
                    print(f"  New best model saved! Val Acc: {val_acc:.2f}%")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
            
            print("-" * 50)
        
        print(f"Training completed! Best validation accuracy: {best_val_acc:.2f}%")
        return self.history


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device.
    
    Returns:
        PyTorch device (CUDA if available, otherwise CPU)
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    
    return device
