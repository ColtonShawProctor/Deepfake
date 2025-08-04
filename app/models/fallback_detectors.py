"""
Fallback detector implementations for the multi-model system.
These provide basic functionality when the full model implementations are not available.
"""

import logging
import time
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass
from PIL import Image

logger = logging.getLogger(__name__)

@dataclass
class DetectionResult:
    """Result of deepfake detection analysis."""
    confidence_score: float
    is_deepfake: bool
    model_name: str
    processing_time: float
    uncertainty: Optional[float] = None
    attention_weights: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None

class BaseDetector:
    """Base detector class for fallback implementations."""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    def predict(self, image: Image.Image) -> DetectionResult:
        """Basic prediction method."""
        start_time = time.time()
        
        # Simple mock prediction based on image properties
        confidence = self._mock_prediction(image)
        processing_time = time.time() - start_time
        
        return DetectionResult(
            confidence_score=confidence,
            is_deepfake=confidence > 50.0,
            model_name=self.__class__.__name__,
            processing_time=processing_time,
            metadata={"status": "fallback_implementation"}
        )
    
    def _mock_prediction(self, image: Image.Image) -> float:
        """Generate a mock prediction based on image characteristics."""
        # Use image properties to generate consistent results
        width, height = image.size
        area = width * height
        
        # Simple heuristic based on image size and randomness
        base_score = 45.0 + (area % 20)  # Will be between 45-65
        return min(95.0, max(5.0, base_score))

class ResNetDetector(BaseDetector):
    """Fallback ResNet detector."""
    
    def __init__(self, device: str = "cpu"):
        super().__init__(device)
        self.logger.info("Using fallback ResNet detector")

class EfficientNetDetector(BaseDetector):
    """Fallback EfficientNet detector."""
    
    def __init__(self, device: str = "cpu"):
        super().__init__(device)
        self.logger.info("Using fallback EfficientNet detector")

class F3NetDetector(BaseDetector):
    """Fallback F3Net detector."""
    
    def __init__(self, device: str = "cpu"):
        super().__init__(device)
        self.logger.info("Using fallback F3Net detector")

class ModelManager:
    """Fallback model manager for basic ensemble functionality."""
    
    def __init__(self, models_dir: str, device: str = "cpu"):
        self.models_dir = models_dir
        self.device = device
        self.models = {}
        self.logger = logging.getLogger(f"{__name__.ModelManager}")
        self.logger.info("Using fallback ModelManager")
    
    def load_all_models(self) -> bool:
        """Load all available models."""
        try:
            # Initialize fallback models
            self.models = {
                "resnet": ResNetDetector(self.device),
                "efficientnet": EfficientNetDetector(self.device),
                "f3net": F3NetDetector(self.device)
            }
            
            self.logger.info(f"Loaded {len(self.models)} fallback models")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load fallback models: {str(e)}")
            return False
    
    def predict(self, image: Image.Image) -> DetectionResult:
        """Perform ensemble prediction using fallback models."""
        if not self.models:
            self.load_all_models()
        
        predictions = []
        total_time = 0
        
        # Get predictions from all models
        for name, model in self.models.items():
            try:
                result = model.predict(image)
                predictions.append(result)
                total_time += result.processing_time
            except Exception as e:
                self.logger.error(f"Model {name} failed: {str(e)}")
        
        if not predictions:
            # Return neutral prediction if all models fail
            return DetectionResult(
                confidence_score=50.0,
                is_deepfake=False,
                model_name="FallbackEnsemble",
                processing_time=0.1,
                metadata={"error": "All models failed"}
            )
        
        # Simple ensemble - average confidence
        avg_confidence = np.mean([pred.confidence_score for pred in predictions])
        
        # Calculate uncertainty as variance
        uncertainty = np.var([pred.confidence_score for pred in predictions])
        
        return DetectionResult(
            confidence_score=float(avg_confidence),
            is_deepfake=avg_confidence > 50.0,
            model_name="FallbackEnsemble",
            processing_time=total_time,
            uncertainty=float(uncertainty),
            metadata={
                "individual_predictions": {
                    pred.model_name: {
                        "confidence": pred.confidence_score,
                        "is_deepfake": pred.is_deepfake
                    }
                    for pred in predictions
                },
                "num_models": len(predictions),
                "fallback": True
            }
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        return {
            "loaded_models": list(self.models.keys()),
            "total_models": len(self.models),
            "device": self.device,
            "status": "fallback_implementation"
        }