"""
Hugging Face Deepfake Detector

Replaces the broken EfficientNet with the prithivMLmods/deepfake-detector-model-v1
Vision Transformer model (94.4% accuracy).
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForImageClassification
import numpy as np

from .base_detector import BaseDetector, DetectionResult


class HuggingFaceDetector(BaseDetector):
    """
    Hugging Face deepfake detector using prithivMLmods/deepfake-detector-model-v1
    
    Performance: 94.4% accuracy on standard benchmarks
    Input: 224x224 RGB images (automatically resized)
    Architecture: Vision Transformer (ViT) based
    """
    
    def __init__(self, device: str = "auto"):
        super().__init__("HuggingFaceDetector", device)
        self.input_size = (224, 224)
        self.model_name = "prithivMLmods/deepfake-detector-model-v1"
        
        # Initialize model and processor as None
        self.model = None
        self.processor = None
        
        # Model configuration
        self.confidence_threshold = 0.5
        self.logger = logging.getLogger(f"{__name__}.HuggingFaceDetector")
        
    def load_model(self, weights_path: Optional[str] = None) -> None:
        """Load Hugging Face deepfake detection model"""
        try:
            self.logger.info(f"Loading Hugging Face model: {self.model_name}")
            
            # Load the model and processor
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForImageClassification.from_pretrained(self.model_name)
            
            # Move model to device
            self.model = self.model.to(self.device)
            self.model.eval()
            
            self.logger.info(f"HuggingFaceDetector loaded successfully on {self.device}")
            
        except Exception as e:
            self.logger.error(f"Failed to load Hugging Face model: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")
    
    def preprocess(self, image: Image.Image) -> Dict[str, torch.Tensor]:
        """Preprocess image for Hugging Face model input"""
        # Ensure RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Use the model's processor for preprocessing
        inputs = self.processor(images=image, return_tensors="pt")
        
        # Move inputs to device
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                inputs[key] = value.to(self.device)
        
        return inputs
    
    def predict(self, image: Image.Image) -> DetectionResult:
        """Perform deepfake detection using Hugging Face model"""
        start_time = time.time()
        
        try:
            # Check if model is loaded
            if self.model is None or self.processor is None:
                raise RuntimeError("Model not loaded. Call load_model() first.")
            
            # Preprocess image
            inputs = self.preprocess(image)
            
            # Perform inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Apply softmax to get probabilities
                probabilities = F.softmax(logits, dim=-1)
                
                # Get prediction and confidence
                predicted_class = torch.argmax(probabilities, dim=-1).item()
                confidence_score = probabilities[0, predicted_class].item()
                
                # Convert to percentage
                confidence_percentage = confidence_score * 100.0
                
                # Determine if it's a deepfake based on the model's output
                # The model outputs: [real, fake] where index 0 is real, index 1 is fake
                is_deepfake = predicted_class == 1  # Class 1 is fake/deepfake
                
                # For confidence, use the probability of the predicted class
                # If real (class 0), use that probability; if fake (class 1), use that probability
                final_confidence = confidence_percentage
            
            processing_time = time.time() - start_time
            
            return DetectionResult(
                confidence=final_confidence / 100.0,  # Convert to 0-1 scale
                is_deepfake=is_deepfake,
                model_name=self.model_name,
                inference_time=processing_time,
                metadata={
                    "input_size": self.input_size,
                    "model_architecture": "Vision Transformer (ViT)",
                    "confidence_threshold": self.confidence_threshold,
                    "predicted_class": predicted_class,
                    "class_names": ["real", "fake"],  # Class 0 is real, Class 1 is fake
                    "raw_probabilities": {
                        "real": probabilities[0, 0].item(),
                        "fake": probabilities[0, 1].item()
                    }
                }
            )
            
        except Exception as e:
            self.logger.error(f"Hugging Face prediction failed: {str(e)}")
            return DetectionResult(
                confidence=0.0,
                is_deepfake=False,
                model_name=self.model_name,
                inference_time=time.time() - start_time,
                metadata={"error": str(e)}
            )
    
    def predict_batch(self, images: list[Image.Image]) -> list[DetectionResult]:
        """Perform batch prediction on multiple images"""
        results = []
        for image in images:
            result = self.predict(image)
            results.append(result)
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "name": self.model_name,
            "architecture": "Vision Transformer (ViT)",
            "input_size": self.input_size,
            "accuracy": "94.4%",
            "device": self.device,
            "status": "loaded" if self.model is not None else "not_loaded",
            "model_type": "huggingface",
            "supported_formats": ["jpg", "jpeg", "png", "bmp", "tiff"]
        }


class HuggingFaceDetectorWrapper:
    """
    Wrapper class that provides the same interface as the old EfficientNet detector
    for drop-in replacement compatibility.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the Hugging Face detector wrapper.
        
        Args:
            model_path: Ignored for Hugging Face models (kept for compatibility)
        """
        self.logger = logging.getLogger(__name__)
        
        # Load the model with better error handling
        try:
            self.logger.info("Initializing Hugging Face detector...")
            # Force CPU usage to avoid MPS/GPU issues
            self.detector = HuggingFaceDetector(device="cpu")
            self.logger.info("Hugging Face detector created, loading model...")
            self.detector.load_model()
            self.logger.info("Hugging Face detector loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load Hugging Face detector: {str(e)}")
            self.logger.error(f"Error type: {type(e).__name__}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Detector initialization failed: {str(e)}")
    
    def predict(self, file_path: str) -> Dict[str, Any]:
        """
        Predict whether an image is a deepfake.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Dictionary with prediction results in the expected format
        """
        try:
            # Load image
            image = Image.open(file_path).convert('RGB')
            
            # Perform prediction
            result = self.detector.predict(image)
            
            # Get the raw probabilities for proper confidence interpretation
            probabilities = result.metadata.get("raw_probabilities", {})
            predicted_class = result.metadata.get("predicted_class", 0)
            
            # The confidence should be the probability of the predicted class
            # This ensures that high confidence in "real" shows as high confidence, not low
            # Class 0 is real, Class 1 is fake
            confidence = probabilities.get("real" if predicted_class == 0 else "fake", 0.0)
            
            # Convert to expected format
            return {
                "confidence": confidence,  # Probability of the predicted class
                "is_deepfake": result.is_deepfake,  # This is already correct from the core detector
                "model": "huggingface_detector",
                "inference_time": result.inference_time,
                "method": "huggingface_vit",
                "device": self.detector.device,
                "input_size": self.detector.input_size,
                "predicted_class": predicted_class,
                "probabilities": probabilities,
                "real_confidence": probabilities.get("real", 0.0),
                "fake_confidence": probabilities.get("fake", 0.0)
            }
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            return {
                "confidence": 0.0,
                "is_deepfake": False,
                "model": "huggingface_detector",
                "inference_time": 0.0,
                "method": "huggingface_vit",
                "device": "cpu",
                "error": str(e)
            }
    
    def get_detector_info(self) -> Dict[str, Any]:
        """Get detector information for API responses"""
        model_info = self.detector.get_model_info()
        
        # Convert to the format expected by DetectorInfo schema
        return {
            "name": model_info.get("name", "Hugging Face Deepfake Detector"),
            "version": "1.0.0",
            "description": f"{model_info.get('architecture', 'Vision Transformer')} deepfake detector with 94.4% accuracy",
            "capabilities": ["image_analysis", "deepfake_detection", "confidence_scoring", "real_time_inference"],
            "supported_formats": model_info.get("supported_formats", ["jpg", "jpeg", "png", "bmp", "tiff"]),
            "max_file_size_mb": 100,  # 100MB limit
            "confidence_threshold": 0.5
        }
