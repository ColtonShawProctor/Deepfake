#!/usr/bin/env python3
"""
Test script for the new Hugging Face deepfake detector.
This script tests the detector with a sample image to ensure it works correctly.
"""

import sys
import os
from pathlib import Path

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "app"))

try:
    from models.huggingface_detector import HuggingFaceDetectorWrapper
    from PIL import Image
    import numpy as np
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all dependencies are installed:")
    print("pip install transformers torch torchvision pillow")
    sys.exit(1)

def create_test_image():
    """Create a simple test image for testing"""
    # Create a 224x224 RGB test image
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    return img

def test_detector():
    """Test the Hugging Face detector"""
    print("Testing Hugging Face Deepfake Detector...")
    print("=" * 50)
    
    try:
        # Initialize detector
        print("1. Initializing detector...")
        detector = HuggingFaceDetectorWrapper()
        print("✅ Detector initialized successfully!")
        
        # Get detector info
        print("\n2. Getting detector info...")
        info = detector.get_detector_info()
        print(f"✅ Detector info: {info}")
        
        # Create test image
        print("\n3. Creating test image...")
        test_image = create_test_image()
        print("✅ Test image created (224x224 RGB)")
        
        # Save test image temporarily
        test_image_path = "test_image.jpg"
        test_image.save(test_image_path)
        print(f"✅ Test image saved to {test_image_path}")
        
        # Test prediction
        print("\n4. Testing prediction...")
        result = detector.predict(test_image_path)
        print(f"✅ Prediction result: {result}")
        
        # Clean up
        if os.path.exists(test_image_path):
            os.remove(test_image_path)
            print("✅ Test image cleaned up")
        
        print("\n" + "=" * 50)
        print("🎉 All tests passed! Hugging Face detector is working correctly.")
        print(f"Model: {info.get('name', 'Unknown')}")
        print(f"Architecture: {info.get('architecture', 'Unknown')}")
        print(f"Accuracy: {info.get('accuracy', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Ensure transformers is installed: pip install transformers")
        print("2. Check internet connection (model will be downloaded)")
        print("3. Verify PyTorch installation: pip install torch torchvision")
        return False

if __name__ == "__main__":
    success = test_detector()
    sys.exit(0 if success else 1)





