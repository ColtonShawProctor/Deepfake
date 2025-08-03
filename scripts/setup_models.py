#!/usr/bin/env python3
"""
Setup script for multi-model deepfake detection system

This script downloads pre-trained weights and sets up the model architecture
for the enhanced deepfake detection system.
"""

import os
import sys
import logging
import requests
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional

# Add the app directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from app.models.deepfake_models import (
    ResNetDetector,
    EfficientNetDetector,
    F3NetDetector,
    ModelManager
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelSetup:
    """Setup class for downloading and configuring models"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Model URLs (placeholder URLs - replace with actual pre-trained weights)
        self.model_urls = {
            "resnet_weights.pth": "https://example.com/resnet_deepfake_weights.pth",
            "efficientnet_weights.pth": "https://example.com/efficientnet_deepfake_weights.pth",
            "f3net_weights.pth": "https://example.com/f3net_deepfake_weights.pth"
        }
        
    def download_weights(self, model_name: str, force_download: bool = False) -> bool:
        """Download pre-trained weights for a specific model"""
        weight_file = self.models_dir / f"{model_name}_weights.pth"
        
        if weight_file.exists() and not force_download:
            logger.info(f"Weights for {model_name} already exist: {weight_file}")
            return True
        
        url = self.model_urls.get(f"{model_name}_weights.pth")
        if not url:
            logger.warning(f"No URL found for {model_name} weights")
            return False
        
        try:
            logger.info(f"Downloading {model_name} weights from {url}")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(weight_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Successfully downloaded {model_name} weights")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {model_name} weights: {str(e)}")
            return False
    
    def create_initial_weights(self, model_name: str) -> bool:
        """Create initial weights for models (for development/testing)"""
        weight_file = self.models_dir / f"{model_name}_weights.pth"
        
        if weight_file.exists():
            logger.info(f"Weights for {model_name} already exist: {weight_file}")
            return True
        
        try:
            logger.info(f"Creating initial weights for {model_name}")
            
            if model_name == "resnet":
                detector = ResNetDetector()
                detector.load_model()  # Load with pre-trained ImageNet weights
                
            elif model_name == "efficientnet":
                detector = EfficientNetDetector()
                detector.load_model()  # Load with pre-trained ImageNet weights
                
            elif model_name == "f3net":
                detector = F3NetDetector()
                detector.load_model()  # Load with random initialization
                
            else:
                logger.error(f"Unknown model: {model_name}")
                return False
            
            # Save the model weights
            torch.save(detector.model.state_dict(), weight_file)
            logger.info(f"Created initial weights for {model_name}: {weight_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create initial weights for {model_name}: {str(e)}")
            return False
    
    def setup_all_models(self, use_pretrained: bool = True) -> bool:
        """Setup all models with weights"""
        logger.info("Setting up all deepfake detection models...")
        
        models = ["resnet", "efficientnet", "f3net"]
        success_count = 0
        
        for model_name in models:
            if use_pretrained:
                success = self.download_weights(model_name)
            else:
                success = self.create_initial_weights(model_name)
            
            if success:
                success_count += 1
            else:
                logger.warning(f"Failed to setup {model_name}, will use random initialization")
        
        logger.info(f"Successfully setup {success_count}/{len(models)} models")
        return success_count > 0
    
    def test_models(self) -> bool:
        """Test that all models can be loaded and run inference"""
        logger.info("Testing model loading and inference...")
        
        try:
            # Create model manager
            manager = ModelManager(str(self.models_dir))
            manager.load_all_models()
            
            # Test with a dummy image
            from PIL import Image
            import numpy as np
            
            # Create a dummy image
            dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            
            # Test ensemble prediction
            result = manager.predict(dummy_image)
            logger.info(f"Ensemble test successful: confidence={result.confidence_score:.2f}%")
            
            # Test individual models
            for model_name in ["ResNet", "EfficientNet", "F3Net"]:
                try:
                    result = manager.predict_single_model(model_name, dummy_image)
                    logger.info(f"{model_name} test successful: confidence={result.confidence_score:.2f}%")
                except Exception as e:
                    logger.error(f"{model_name} test failed: {str(e)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Model testing failed: {str(e)}")
            return False
    
    def get_setup_info(self) -> dict:
        """Get information about the current setup"""
        info = {
            "models_dir": str(self.models_dir),
            "models": {}
        }
        
        for model_name in ["resnet", "efficientnet", "f3net"]:
            weight_file = self.models_dir / f"{model_name}_weights.pth"
            info["models"][model_name] = {
                "weights_file": str(weight_file),
                "exists": weight_file.exists(),
                "size_mb": weight_file.stat().st_size / (1024 * 1024) if weight_file.exists() else 0
            }
        
        return info

def main():
    """Main setup function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup multi-model deepfake detection system")
    parser.add_argument("--models-dir", default="models", help="Directory to store model weights")
    parser.add_argument("--use-pretrained", action="store_true", help="Download pre-trained weights")
    parser.add_argument("--force-download", action="store_true", help="Force download even if weights exist")
    parser.add_argument("--test", action="store_true", help="Test models after setup")
    parser.add_argument("--info", action="store_true", help="Show setup information")
    
    args = parser.parse_args()
    
    # Create setup instance
    setup = ModelSetup(args.models_dir)
    
    if args.info:
        info = setup.get_setup_info()
        print("Setup Information:")
        print(f"Models directory: {info['models_dir']}")
        print("\nModel status:")
        for model_name, model_info in info["models"].items():
            status = "✓" if model_info["exists"] else "✗"
            size = f"{model_info['size_mb']:.1f}MB" if model_info["exists"] else "Not found"
            print(f"  {status} {model_name}: {size}")
        return
    
    # Setup models
    logger.info("Starting model setup...")
    success = setup.setup_all_models(use_pretrained=args.use_pretrained)
    
    if success:
        logger.info("Model setup completed successfully")
        
        if args.test:
            logger.info("Testing models...")
            test_success = setup.test_models()
            if test_success:
                logger.info("All tests passed!")
            else:
                logger.error("Some tests failed")
                sys.exit(1)
    else:
        logger.error("Model setup failed")
        sys.exit(1)

if __name__ == "__main__":
    main() 