#!/usr/bin/env python3
"""
Test script to investigate ResNet overfitting issue
"""

import sys
import os
sys.path.append('/Users/crus/deepfake')

from app.models.deepfake_models import ResNetDetector
from PIL import Image
import torch
import numpy as np

def test_resnet_predictions():
    """Test ResNet with known samples to identify overfitting"""
    
    print("🔍 Testing ResNet model for overfitting issues...")
    
    # Initialize ResNet detector
    detector = ResNetDetector(device="auto")
    detector.load_model("/Users/crus/deepfake/models/resnet_weights.pth")
    
    if not detector.is_loaded():
        print("❌ Failed to load ResNet model")
        return
    
    print("✅ ResNet model loaded successfully")
    
    # Test with demo images
    demo_images = [
        "/Users/crus/deepfake/demo_images/01_fake_low_confidence.jpg",
        "/Users/crus/deepfake/demo_images/02_fake_medium_confidence.jpg", 
        "/Users/crus/deepfake/demo_images/03_fake_high_confidence.jpg",
        "/Users/crus/deepfake/demo_images/04_real_medium_confidence.jpg",
        "/Users/crus/deepfake/demo_images/05_real_medium_confidence.jpg",
        "/Users/crus/deepfake/demo_images/06_real_low_confidence.jpg"
    ]
    
    print("\n📊 Testing ResNet predictions on demo images:")
    print("-" * 60)
    
    for img_path in demo_images:
        if os.path.exists(img_path):
            try:
                image = Image.open(img_path)
                result = detector.predict(image)
                
                filename = os.path.basename(img_path)
                print(f"{filename:40} | Confidence: {result.confidence_score:6.2f}% | Deepfake: {result.is_deepfake}")
                
            except Exception as e:
                print(f"❌ Error processing {img_path}: {e}")
        else:
            print(f"⚠️  File not found: {img_path}")
    
    # Test with synthetic data to check for overfitting
    print("\n🧪 Testing ResNet with synthetic data:")
    print("-" * 60)
    
    # Create synthetic images (random noise)
    for i in range(5):
        # Create random RGB image
        random_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        result = detector.predict(random_image)
        print(f"Random noise {i+1:2d}                    | Confidence: {result.confidence_score:6.2f}% | Deepfake: {result.is_deepfake}")
    
    # Test with uniform color images
    print("\n🎨 Testing ResNet with uniform color images:")
    print("-" * 60)
    
    colors = ['red', 'green', 'blue', 'white', 'black']
    for color in colors:
        # Create uniform color image
        if color == 'red':
            color_image = Image.new('RGB', (224, 224), (255, 0, 0))
        elif color == 'green':
            color_image = Image.new('RGB', (224, 224), (0, 255, 0))
        elif color == 'blue':
            color_image = Image.new('RGB', (224, 224), (0, 0, 255))
        elif color == 'white':
            color_image = Image.new('RGB', (224, 224), (255, 255, 255))
        else:  # black
            color_image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        result = detector.predict(color_image)
        print(f"Uniform {color:5s}                   | Confidence: {result.confidence_score:6.2f}% | Deepfake: {result.is_deepfake}")
    
    # Check model architecture
    print("\n🏗️  ResNet Model Architecture Analysis:")
    print("-" * 60)
    
    # Print model structure
    total_params = sum(p.numel() for p in detector.model.parameters())
    trainable_params = sum(p.numel() for p in detector.model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model device: {next(detector.model.parameters()).device}")
    
    # Check if model is in eval mode
    print(f"Model in eval mode: {not detector.model.training}")
    
    # Check the final layer weights
    final_layer = detector.model.fc[-1]  # Last linear layer
    if hasattr(final_layer, 'weight'):
        weights = final_layer.weight.data
        bias = final_layer.bias.data
        print(f"Final layer weights shape: {weights.shape}")
        print(f"Final layer bias shape: {bias.shape}")
        print(f"Final layer weight range: [{weights.min().item():.4f}, {weights.max().item():.4f}]")
        print(f"Final layer bias range: [{bias.min().item():.4f}, {bias.max().item():.4f}]")
    
    print("\n🔍 Analysis complete!")

if __name__ == "__main__":
    test_resnet_predictions()
