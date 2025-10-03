#!/usr/bin/env python3
"""
Debug EfficientNet Model

This script helps debug what's happening with the EfficientNet model
and why it's giving similar confidence scores for all images.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
from pathlib import Path

# Add the app directory to the path
sys.path.append('app')

def debug_efficientnet():
    """Debug the EfficientNet model behavior"""
    
    print("🔍 Debugging EfficientNet Model")
    print("=" * 50)
    
    try:
        from models.optimized_efficientnet_detector import OptimizedEfficientNetDetector
        print("✅ Successfully imported EfficientNet detector")
        
        # Initialize detector with trained weights
        detector = OptimizedEfficientNetDetector("models/efficientnet_weights.pth")
        print("✅ EfficientNet detector initialized")
        
        # Check if model is loaded
        if detector.is_model_loaded():
            print("✅ Model is loaded and ready")
        else:
            print("❌ Model is not loaded")
            return
        
        # Get detector info
        info = detector.get_detector_info()
        print(f"📊 Detector Info:")
        print(f"   Name: {info['name']}")
        print(f"   Version: {info['version']}")
        print(f"   Description: {info['description']}")
        print(f"   Confidence Threshold: {info['confidence_threshold']}")
        
        # Test with a simple image to see raw outputs
        print(f"\n🧪 Testing with simple synthetic image...")
        
        # Create a very simple test image
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        test_image_path = "debug_test_image.jpg"
        Image.fromarray(test_image).save(test_image_path)
        
        # Test prediction
        result = detector.predict(test_image_path)
        print(f"   Raw result: {result}")
        
        # Test with the real image
        print(f"\n🧪 Testing with real image...")
        real_image_path = "test_data/real/images/real_celebrity_1.jpg"
        if Path(real_image_path).exists():
            result = detector.predict(real_image_path)
            print(f"   Real image result: {result}")
        else:
            print(f"   ❌ Real image not found: {real_image_path}")
        
        # Test with a fake image
        print(f"\n🧪 Testing with fake image...")
        fake_image_path = "test_data/fake/images/deepfake_face_1.jpg"
        if Path(real_image_path).exists():
            result = detector.predict(fake_image_path)
            print(f"   Fake image result: {result}")
        else:
            print(f"   ❌ Fake image not found: {fake_image_path}")
        
        # Check the underlying model
        print(f"\n🔍 Checking underlying model...")
        underlying_detector = detector.detector
        
        if hasattr(underlying_detector, 'model'):
            model = underlying_detector.model
            print(f"   Model type: {type(model)}")
            print(f"   Model device: {next(model.parameters()).device}")
            print(f"   Model training mode: {model.training}")
            
            # Check if model has the expected structure
            if hasattr(model, 'efficientnet'):
                print(f"   ✅ Model has EfficientNet structure")
                classifier = model.efficientnet.classifier
                print(f"   Classifier layers: {len(classifier)}")
                for i, layer in enumerate(classifier):
                    print(f"     Layer {i}: {type(layer).__name__}")
                    if hasattr(layer, 'weight'):
                        print(f"       Weight shape: {layer.weight.shape}")
            else:
                print(f"   ❌ Model doesn't have expected EfficientNet structure")
        
        # Check confidence threshold
        print(f"\n🎯 Checking confidence threshold...")
        if hasattr(underlying_detector, 'config'):
            config = underlying_detector.config
            print(f"   Config: {config}")
            if 'confidence_threshold' in config:
                print(f"   Confidence threshold: {config['confidence_threshold']}")
        
        # Clean up
        if Path(test_image_path).exists():
            os.remove(test_image_path)
        
    except Exception as e:
        print(f"❌ Error during debugging: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_efficientnet()





