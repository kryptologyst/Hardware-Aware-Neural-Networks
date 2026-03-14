"""Streamlit demo for hardware-aware neural networks."""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import time
from pathlib import Path
import json

from src.models import create_hardware_aware_model
from src.data import HardwareAwareDataset
from src.evaluation import HardwareAwareEvaluator
from src.export import ModelExporter


def load_model(model_path: str, device: torch.device) -> nn.Module:
    """Load trained model."""
    model = create_hardware_aware_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def preprocess_image(image: Image.Image, input_size: tuple = (96, 96)) -> torch.Tensor:
    """Preprocess image for inference."""
    import torchvision.transforms as transforms
    
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform(image).unsqueeze(0)


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Hardware-Aware Neural Networks Demo",
        page_icon="🔧",
        layout="wide"
    )
    
    st.title("🔧 Hardware-Aware Neural Networks Demo")
    st.markdown("""
    This demo showcases hardware-aware neural networks optimized for edge devices.
    **DISCLAIMER: This is for research/education purposes only. Not for safety-critical deployment.**
    """)
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    model_type = st.sidebar.selectbox(
        "Model Architecture",
        ["mobilenet_v2", "efficientnet_b0"],
        index=0
    )
    
    width_multiplier = st.sidebar.slider(
        "Width Multiplier",
        min_value=0.1,
        max_value=1.0,
        value=0.35,
        step=0.05,
        help="Controls model size and efficiency"
    )
    
    input_size = st.sidebar.selectbox(
        "Input Size",
        [(64, 64), (96, 96), (128, 128), (224, 224)],
        index=1,
        format_func=lambda x: f"{x[0]}x{x[1]}"
    )
    
    device_type = st.sidebar.selectbox(
        "Target Device",
        ["raspberry_pi", "jetson_nano", "android", "mcu"],
        index=0
    )
    
    # Device information
    device_info = {
        "raspberry_pi": {
            "cpu_cores": 4,
            "memory_mb": 1024,
            "power_w": 3.5,
            "target_fps": 10
        },
        "jetson_nano": {
            "cpu_cores": 4,
            "memory_mb": 4096,
            "power_w": 10.0,
            "target_fps": 30
        },
        "android": {
            "cpu_cores": 8,
            "memory_mb": 6144,
            "power_w": 5.0,
            "target_fps": 60
        },
        "mcu": {
            "cpu_cores": 1,
            "memory_mb": 256,
            "power_w": 0.1,
            "target_fps": 1
        }
    }
    
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = create_hardware_aware_model(
        num_classes=5,
        model_type=model_type,
        width_multiplier=width_multiplier,
        input_size=input_size
    )
    model.to(device)
    
    # Model information
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Information")
        model_size = model.get_model_size()
        
        st.metric("Parameters", f"{model_size['num_parameters']:,}")
        st.metric("Model Size", f"{model_size['total_mb']:.2f} MB")
        st.metric("Architecture", model_type)
        st.metric("Width Multiplier", width_multiplier)
    
    with col2:
        st.subheader("Target Device")
        info = device_info[device_type]
        
        st.metric("CPU Cores", info["cpu_cores"])
        st.metric("Memory", f"{info['memory_mb']} MB")
        st.metric("Power", f"{info['power_w']} W")
        st.metric("Target FPS", info["target_fps"])
    
    # Inference demo
    st.subheader("Inference Demo")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=['png', 'jpg', 'jpeg'],
        help="Upload an image to test inference"
    )
    
    if uploaded_file is not None:
        # Load and display image
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Input Image", use_column_width=True)
        
        with col2:
            # Preprocess image
            input_tensor = preprocess_image(image, input_size)
            input_tensor = input_tensor.to(device)
            
            # Run inference
            start_time = time.time()
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
            inference_time = time.time() - start_time
            
            # Display results
            class_names = [f"Class {i}" for i in range(5)]
            
            st.write("**Inference Results:**")
            st.write(f"Predicted Class: **{class_names[predicted_class]}**")
            st.write(f"Inference Time: **{inference_time*1000:.2f} ms**")
            st.write(f"Throughput: **{1.0/inference_time:.1f} FPS**")
            
            # Probability distribution
            probs = probabilities[0].cpu().numpy()
            
            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.bar(class_names, probs)
            ax.set_ylabel('Probability')
            ax.set_title('Class Probabilities')
            ax.set_ylim(0, 1)
            
            # Color bars by probability
            for bar, prob in zip(bars, probs):
                bar.set_color(plt.cm.viridis(prob))
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
    
    # Performance benchmarking
    st.subheader("Performance Benchmarking")
    
    if st.button("Run Benchmark"):
        with st.spinner("Running benchmark..."):
            # Create dummy data for benchmarking
            dummy_data = torch.randn(1, 3, *input_size).to(device)
            
            # Warmup
            for _ in range(10):
                with torch.no_grad():
                    _ = model(dummy_data)
            
            # Benchmark
            times = []
            for _ in range(100):
                start_time = time.time()
                with torch.no_grad():
                    _ = model(dummy_data)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                times.append(time.time() - start_time)
            
            times = np.array(times)
            
            # Display results
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Mean Latency", f"{np.mean(times)*1000:.2f} ms")
            with col2:
                st.metric("P95 Latency", f"{np.percentile(times, 95)*1000:.2f} ms")
            with col3:
                st.metric("Throughput", f"{1.0/np.mean(times):.1f} FPS")
            with col4:
                st.metric("Std Dev", f"{np.std(times)*1000:.2f} ms")
            
            # Latency distribution
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.hist(times*1000, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_xlabel('Latency (ms)')
            ax.set_ylabel('Frequency')
            ax.set_title('Inference Latency Distribution')
            ax.axvline(np.mean(times)*1000, color='red', linestyle='--', label=f'Mean: {np.mean(times)*1000:.2f} ms')
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
    
    # Model comparison
    st.subheader("Model Comparison")
    
    if st.button("Compare Models"):
        with st.spinner("Comparing models..."):
            models_to_compare = [
                ("MobileNetV2-0.25", "mobilenet_v2", 0.25),
                ("MobileNetV2-0.35", "mobilenet_v2", 0.35),
                ("MobileNetV2-0.5", "mobilenet_v2", 0.5),
                ("EfficientNet-B0", "efficientnet_b0", 1.0)
            ]
            
            results = []
            
            for name, model_type, width_mult in models_to_compare:
                # Create model
                comp_model = create_hardware_aware_model(
                    num_classes=5,
                    model_type=model_type,
                    width_multiplier=width_mult,
                    input_size=input_size
                )
                comp_model.to(device)
                
                # Get model size
                model_size = comp_model.get_model_size()
                
                # Benchmark
                dummy_data = torch.randn(1, 3, *input_size).to(device)
                
                # Warmup
                for _ in range(5):
                    with torch.no_grad():
                        _ = comp_model(dummy_data)
                
                # Benchmark
                times = []
                for _ in range(50):
                    start_time = time.time()
                    with torch.no_grad():
                        _ = comp_model(dummy_data)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    times.append(time.time() - start_time)
                
                results.append({
                    "name": name,
                    "parameters": model_size["num_parameters"],
                    "size_mb": model_size["total_mb"],
                    "latency_ms": np.mean(times) * 1000,
                    "fps": 1.0 / np.mean(times)
                })
            
            # Display comparison table
            st.dataframe(results, use_container_width=True)
            
            # Visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Model size vs latency
            names = [r["name"] for r in results]
            sizes = [r["size_mb"] for r in results]
            latencies = [r["latency_ms"] for r in results]
            
            scatter = ax1.scatter(sizes, latencies, s=100, alpha=0.7)
            ax1.set_xlabel('Model Size (MB)')
            ax1.set_ylabel('Latency (ms)')
            ax1.set_title('Model Size vs Latency')
            
            for i, name in enumerate(names):
                ax1.annotate(name, (sizes[i], latencies[i]), 
                           xytext=(5, 5), textcoords='offset points')
            
            # Parameters vs FPS
            params = [r["parameters"] for r in results]
            fps = [r["fps"] for r in results]
            
            ax2.scatter(params, fps, s=100, alpha=0.7, color='orange')
            ax2.set_xlabel('Parameters')
            ax2.set_ylabel('FPS')
            ax2.set_title('Parameters vs FPS')
            
            for i, name in enumerate(names):
                ax2.annotate(name, (params[i], fps[i]), 
                           xytext=(5, 5), textcoords='offset points')
            
            plt.tight_layout()
            st.pyplot(fig)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Disclaimer**: This demo is for research and educational purposes only. 
    The models and results shown here are not intended for safety-critical applications.
    """)


if __name__ == "__main__":
    main()
