"""
Optimized EfficientNet detector wrapper that provides a simplified interface
for the detection routes.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image

from .efficientnet_detector import EfficientNetDetector
from .base_detector import DetectionResult


class OptimizedEfficientNetDetector:
    """
    Wrapper for EfficientNetDetector that provides a simplified interface
    expected by the detection routes.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the optimized EfficientNet detector.
        
        Args:
            model_path: Path to the model weights file
        """
        self.logger = logging.getLogger(__name__)
        self.detector = EfficientNetDetector(
            model_name="OptimizedEfficientNet",
            device="auto",
            config={
                "dropout_rate": 0.3,
                "confidence_threshold": 0.5,
                "enable_attention": True,
                "mobile_optimized": True
            }
        )
        
        # Load the model if path is provided
        if model_path:
            success = self.load_model(model_path)
        else:
            # Load without weights (will use ImageNet pre-trained)
            success = self.detector.load_model()
        
        if not success:
            self.logger.error("Failed to load EfficientNet model during initialization")
            raise RuntimeError("Model initialization failed")
    
    def load_model(self, model_path: str) -> bool:
        """
        Load the model from the specified path.
        
        Args:
            model_path: Path to the model weights file
            
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            if not Path(model_path).exists():
                self.logger.warning(f"Model weights file not found: {model_path}")
                self.logger.info("Loading model with ImageNet pre-trained weights")
                return self.detector.load_model()
            
            return self.detector.load_model(model_path)
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            # Fallback to ImageNet pre-trained weights
            return self.detector.load_model()
    
    def predict(self, file_path: str) -> Dict[str, Any]:
        """
        Perform deepfake detection on a file.
        
        Args:
            file_path: Path to the image file to analyze
            
        Returns:
            Dictionary containing detection results with keys:
            - confidence: float (0-1)
            - is_deepfake: bool
            - inference_time: float
            - model: str
            - method: str
            - device: str
            - input_size: list[int]
        """
        try:
            # Check if model is loaded
            if not self.is_model_loaded():
                raise RuntimeError("Model not loaded. Call load_model() first.")
            
            # Load and validate image
            image = Image.open(file_path).convert('RGB')
            
            # Perform detection
            result: DetectionResult = self.detector.predict(image)
            
            # Convert to expected dictionary format
            return {
                "confidence": result.confidence,
                "is_deepfake": result.is_deepfake,
                "inference_time": result.inference_time,
                "model": result.model_name,
                "method": "single",
                "device": self.detector.device,
                "input_size": self.detector.model_info.input_size
            }
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            raise
    
    def is_model_loaded(self) -> bool:
        """
        Check if the model is loaded and ready for inference.
        
        Returns:
            True if model is loaded, False otherwise
        """
        try:
            return self.detector.is_model_loaded
        except Exception as e:
            self.logger.error(f"Error checking model status: {str(e)}")
            return False
    
    def get_detector_info(self) -> Dict[str, Any]:
        """
        Get information about the detector.
        
        Returns:
            Dictionary containing detector information that matches DetectorInfo schema
        """
        return {
            "name": self.detector.model_name,
            "version": self.detector.model_info.version,
            "description": f"{self.detector.model_info.architecture} deepfake detector with optimized performance",
            "capabilities": ["image_analysis", "deepfake_detection", "confidence_scoring", "attention_visualization"],
            "supported_formats": self.detector.model_info.supported_formats,
            "max_file_size_mb": 100,  # 100MB limit
            "confidence_threshold": 0.5
        }
