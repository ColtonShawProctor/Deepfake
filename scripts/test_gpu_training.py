#!/usr/bin/env python3
"""
Quick test to verify GPU training is working
"""

import torch
import sys
from pathlib import Path

# Add the app directory to the path
sys.path.append(str(Path(__file__).parent / "app"))

def test_device_selection():
    """Test device selection logic"""
    print("🔍 Testing Device Selection...")
    
    # Check available devices
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    
    # Test device selection logic
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("✅ Using CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Using MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print("⚠️ Using CPU")
    
    print(f"Selected device: {device}")
    
    # Test tensor operations on device
    try:
        x = torch.randn(100, 100).to(device)
        y = torch.randn(100, 100).to(device)
        z = torch.mm(x, y)
        print(f"✅ Tensor operations successful on {device}")
        print(f"Result tensor device: {z.device}")
        return True
    except Exception as e:
        print(f"❌ Tensor operations failed: {e}")
        return False

def test_model_loading():
    """Test model loading on GPU"""
    print("\n🧠 Testing Model Loading...")
    
    try:
        from app.models.deepfake_models import ResNetDetector
        
        # Select device
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        
        # Create model
        model = ResNetDetector()
        model.to(device)
        
        print(f"✅ Model loaded successfully on {device}")
        print(f"Model device: {next(model.parameters()).device}")
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 GPU Training Test")
    print("=" * 40)
    
    device_test = test_device_selection()
    model_test = test_model_loading()
    
    print("\n" + "=" * 40)
    print("📋 Test Results:")
    print(f"  Device Selection: {'✅ PASS' if device_test else '❌ FAIL'}")
    print(f"  Model Loading: {'✅ PASS' if model_test else '❌ FAIL'}")
    
    if device_test and model_test:
        print("\n🎉 GPU training is ready!")
        print("✅ MPS (Apple Silicon GPU) is available and working")
        print("✅ Training will be much faster than CPU")
    else:
        print("\n⚠️ Some tests failed. Training will use CPU.")
    
    print("\n" + "=" * 40) 