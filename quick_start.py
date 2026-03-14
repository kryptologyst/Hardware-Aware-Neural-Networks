#!/usr/bin/env python3
"""Quick start script for hardware-aware neural networks."""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    print(f"Running: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("SUCCESS!")
        if result.stdout:
            print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("FAILED!")
        print("Error:", e.stderr)
        return False


def main():
    """Main quick start function."""
    print("Hardware-Aware Neural Networks - Quick Start")
    print("=" * 60)
    print("This script will demonstrate the key features of the project.")
    print("DISCLAIMER: This is for research/education purposes only.")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("src").exists():
        print("Error: Please run this script from the project root directory")
        sys.exit(1)
    
    steps = [
        ("python -c \"from src.models import create_hardware_aware_model; print('Model creation: OK')\"", 
         "Test model creation"),
        
        ("python -c \"from src.data import create_synthetic_dataset; create_synthetic_dataset('demo_data', 3, 10, (32, 32)); print('Synthetic data: OK')\"", 
         "Create synthetic dataset"),
        
        ("python -c \"from src.training import set_seed; set_seed(42); print('Reproducibility: OK')\"", 
         "Test reproducibility"),
        
        ("python -c \"from src.evaluation import HardwareAwareEvaluator; print('Evaluation: OK')\"", 
         "Test evaluation utilities"),
        
        ("python -c \"from src.export import ModelExporter; print('Export: OK')\"", 
         "Test export utilities"),
        
        ("python -c \"from src.config import Config; config = Config(); print('Configuration: OK')\"", 
         "Test configuration system"),
    ]
    
    success_count = 0
    total_steps = len(steps)
    
    for cmd, description in steps:
        if run_command(cmd, description):
            success_count += 1
    
    print(f"\n{'='*60}")
    print("QUICK START SUMMARY")
    print(f"{'='*60}")
    print(f"Completed: {success_count}/{total_steps} steps")
    
    if success_count == total_steps:
        print("\n✅ All tests passed! The project is ready to use.")
        print("\nNext steps:")
        print("1. Train a model: python train.py --create-synthetic --epochs 5")
        print("2. Run the demo: streamlit run demo.py")
        print("3. Export models: python train.py --export-formats onnx torchscript")
        print("4. Deploy: python scripts/deploy.py --model-path exports/model.onnx")
    else:
        print(f"\n❌ {total_steps - success_count} tests failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
