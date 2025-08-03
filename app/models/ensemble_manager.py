"""
Ensemble prediction infrastructure for multi-model deepfake detection framework.
This module provides ensemble management and prediction fusion capabilities.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .base_detector import BaseDetector, DetectionResult, ModelInfo


class FusionMethod(str, Enum):
    """Available ensemble fusion methods."""
    WEIGHTED_AVERAGE = "weighted_average"
    MAJORITY_VOTING = "majority_voting"
    SOFT_VOTING = "soft_voting"
    ATTENTION_FUSION = "attention_fusion"
    MAX_CONFIDENCE = "max_confidence"
    MIN_CONFIDENCE = "min_confidence"


@dataclass
class EnsembleConfig:
    """Configuration for ensemble prediction."""
    fusion_method: FusionMethod = FusionMethod.WEIGHTED_AVERAGE
    default_weights: Optional[Dict[str, float]] = None
    temperature: float = 1.0
    min_models: int = 1
    max_models: int = 10
    confidence_threshold: float = 0.5
    enable_uncertainty: bool = True
    enable_attention: bool = False
    attention_dim: int = 128


@dataclass
class EnsembleResult:
    """Result of ensemble prediction."""
    is_deepfake: bool
    confidence: float
    fusion_method: str
    individual_predictions: Dict[str, DetectionResult]
    ensemble_confidence: float
    uncertainty: Optional[float] = None
    attention_weights: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class EnsembleManager:
    """
    Manages ensemble of multiple detectors.
    
    Provides functionality to combine predictions from multiple models
    using various fusion strategies.
    """
    
    def __init__(self, config: Optional[EnsembleConfig] = None):
        """
        Initialize the ensemble manager.
        
        Args:
            config: Ensemble configuration
        """
        self.config = config or EnsembleConfig()
        self.models: Dict[str, BaseDetector] = {}
        self.weights: Dict[str, float] = {}
        self.attention_weights: Optional[torch.Tensor] = None
        
        # Initialize default weights if provided
        if self.config.default_weights:
            self.weights.update(self.config.default_weights)
        
        self.logger = logging.getLogger(f"{__name__}.EnsembleManager")
        self.logger.info(f"Ensemble manager initialized with {self.config.fusion_method} fusion")
    
    def add_model(self, name: str, detector: BaseDetector, weight: float = 1.0) -> bool:
        """
        Add a model to the ensemble.
        
        Args:
            name: Unique name for the model
            detector: Detector instance
            weight: Weight for this model in ensemble fusion
            
        Returns:
            True if added successfully, False otherwise
        """
        try:
            if not detector.is_loaded():
                self.logger.warning(f"Model '{name}' is not loaded, attempting to load")
                if not detector.load_model():
                    self.logger.error(f"Failed to load model '{name}'")
                    return False
            
            self.models[name] = detector
            self.weights[name] = weight
            
            self.logger.info(f"Added model '{name}' to ensemble with weight {weight}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add model '{name}' to ensemble: {str(e)}")
            return False
    
    def remove_model(self, name: str) -> bool:
        """
        Remove a model from the ensemble.
        
        Args:
            name: Name of the model to remove
            
        Returns:
            True if removed successfully, False otherwise
        """
        try:
            if name in self.models:
                del self.models[name]
                self.weights.pop(name, None)
                self.logger.info(f"Removed model '{name}' from ensemble")
                return True
            else:
                self.logger.warning(f"Model '{name}' not found in ensemble")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to remove model '{name}' from ensemble: {str(e)}")
            return False
    
    def predict_ensemble(self, image) -> EnsembleResult:
        """
        Perform ensemble prediction on an image.
        
        Args:
            image: PIL Image to analyze
            
        Returns:
            EnsembleResult containing combined prediction
        """
        start_time = time.time()
        
        try:
            # Validate minimum number of models
            if len(self.models) < self.config.min_models:
                raise ValueError(f"Not enough models loaded. Need at least {self.config.min_models}")
            
            # Get individual predictions
            individual_predictions = {}
            for name, model in self.models.items():
                try:
                    result = model.predict(image)
                    individual_predictions[name] = result
                except Exception as e:
                    self.logger.error(f"Model '{name}' prediction failed: {str(e)}")
                    continue
            
            if not individual_predictions:
                raise ValueError("No models produced valid predictions")
            
            # Apply fusion method
            if self.config.fusion_method == FusionMethod.WEIGHTED_AVERAGE:
                ensemble_prediction = self._weighted_average(individual_predictions)
            elif self.config.fusion_method == FusionMethod.MAJORITY_VOTING:
                ensemble_prediction = self._majority_voting(individual_predictions)
            elif self.config.fusion_method == FusionMethod.SOFT_VOTING:
                ensemble_prediction = self._soft_voting(individual_predictions)
            elif self.config.fusion_method == FusionMethod.ATTENTION_FUSION:
                ensemble_prediction = self._attention_fusion(individual_predictions)
            elif self.config.fusion_method == FusionMethod.MAX_CONFIDENCE:
                ensemble_prediction = self._max_confidence(individual_predictions)
            elif self.config.fusion_method == FusionMethod.MIN_CONFIDENCE:
                ensemble_prediction = self._min_confidence(individual_predictions)
            else:
                raise ValueError(f"Unknown fusion method: {self.config.fusion_method}")
            
            # Calculate uncertainty if enabled
            uncertainty = None
            if self.config.enable_uncertainty:
                uncertainty = self._calculate_uncertainty(individual_predictions)
            
            # Create ensemble result
            ensemble_result = EnsembleResult(
                is_deepfake=ensemble_prediction["is_deepfake"],
                confidence=ensemble_prediction["confidence"],
                fusion_method=self.config.fusion_method.value,
                individual_predictions=individual_predictions,
                ensemble_confidence=ensemble_prediction["confidence"],
                uncertainty=uncertainty,
                attention_weights=ensemble_prediction.get("attention_weights"),
                metadata={
                    "num_models": len(individual_predictions),
                    "inference_time": time.time() - start_time,
                    "fusion_config": self.config.__dict__
                }
            )
            
            self.logger.info(f"Ensemble prediction completed in {ensemble_result.metadata['inference_time']:.3f}s")
            return ensemble_result
            
        except Exception as e:
            self.logger.error(f"Ensemble prediction failed: {str(e)}")
            raise
    
    def _weighted_average(self, predictions: Dict[str, DetectionResult]) -> Dict[str, Any]:
        """Apply weighted average fusion."""
        total_weight = 0.0
        weighted_sum = 0.0
        
        for name, result in predictions.items():
            weight = self.weights.get(name, 1.0)
            total_weight += weight
            weighted_sum += result.confidence * weight
        
        if total_weight == 0:
            # Fallback to simple average
            avg_confidence = np.mean([r.confidence for r in predictions.values()])
        else:
            avg_confidence = weighted_sum / total_weight
        
        return {
            "is_deepfake": avg_confidence > self.config.confidence_threshold,
            "confidence": avg_confidence
        }
    
    def _majority_voting(self, predictions: Dict[str, DetectionResult]) -> Dict[str, Any]:
        """Apply majority voting fusion."""
        votes = [1 if r.is_deepfake else 0 for r in predictions.values()]
        majority_vote = sum(votes) > len(votes) / 2
        
        # Calculate average confidence
        avg_confidence = np.mean([r.confidence for r in predictions.values()])
        
        return {
            "is_deepfake": majority_vote,
            "confidence": avg_confidence
        }
    
    def _soft_voting(self, predictions: Dict[str, DetectionResult]) -> Dict[str, Any]:
        """Apply soft voting fusion."""
        confidences = [r.confidence for r in predictions.values()]
        avg_confidence = np.mean(confidences)
        
        return {
            "is_deepfake": avg_confidence > self.config.confidence_threshold,
            "confidence": avg_confidence
        }
    
    def _attention_fusion(self, predictions: Dict[str, DetectionResult]) -> Dict[str, Any]:
        """Apply attention-based fusion."""
        if not self.config.enable_attention:
            return self._weighted_average(predictions)
        
        # Extract features and confidences
        confidences = torch.tensor([r.confidence for r in predictions.values()])
        model_names = list(predictions.keys())
        
        # Simple attention mechanism based on confidence
        attention_weights = F.softmax(confidences / self.config.temperature, dim=0)
        
        # Apply attention weights
        weighted_confidence = torch.sum(confidences * attention_weights).item()
        
        attention_dict = {name: weight.item() for name, weight in zip(model_names, attention_weights)}
        
        return {
            "is_deepfake": weighted_confidence > self.config.confidence_threshold,
            "confidence": weighted_confidence,
            "attention_weights": attention_dict
        }
    
    def _max_confidence(self, predictions: Dict[str, DetectionResult]) -> Dict[str, Any]:
        """Apply max confidence fusion."""
        max_confidence = max(r.confidence for r in predictions.values())
        max_model = max(predictions.keys(), key=lambda k: predictions[k].confidence)
        
        return {
            "is_deepfake": predictions[max_model].is_deepfake,
            "confidence": max_confidence
        }
    
    def _min_confidence(self, predictions: Dict[str, DetectionResult]) -> Dict[str, Any]:
        """Apply min confidence fusion."""
        min_confidence = min(r.confidence for r in predictions.values())
        min_model = min(predictions.keys(), key=lambda k: predictions[k].confidence)
        
        return {
            "is_deepfake": predictions[min_model].is_deepfake,
            "confidence": min_confidence
        }
    
    def _calculate_uncertainty(self, predictions: Dict[str, DetectionResult]) -> float:
        """Calculate prediction uncertainty."""
        confidences = [r.confidence for r in predictions.values()]
        
        if len(confidences) < 2:
            return 0.0
        
        # Calculate standard deviation as uncertainty measure
        uncertainty = np.std(confidences)
        
        return uncertainty
    
    def optimize_weights(self, validation_data: List[Tuple[Any, bool]]) -> bool:
        """
        Optimize ensemble weights using validation data.
        
        Args:
            validation_data: List of (image, ground_truth) tuples
            
        Returns:
            True if optimization successful, False otherwise
        """
        try:
            if len(validation_data) == 0:
                self.logger.warning("No validation data provided for weight optimization")
                return False
            
            # Simple grid search for weight optimization
            best_accuracy = 0.0
            best_weights = self.weights.copy()
            
            # Test different weight combinations
            weight_values = [0.5, 1.0, 1.5, 2.0]
            
            for weight1 in weight_values:
                for weight2 in weight_values:
                    if len(self.models) >= 2:
                        test_weights = {
                            list(self.models.keys())[0]: weight1,
                            list(self.models.keys())[1]: weight2
                        }
                        
                        # Add remaining models with default weight
                        for i, name in enumerate(list(self.models.keys())[2:], 2):
                            test_weights[name] = 1.0
                        
                        # Test accuracy
                        correct = 0
                        for image, ground_truth in validation_data:
                            try:
                                # Temporarily set weights
                                original_weights = self.weights.copy()
                                self.weights = test_weights
                                
                                result = self.predict_ensemble(image)
                                if result.is_deepfake == ground_truth:
                                    correct += 1
                                
                                # Restore original weights
                                self.weights = original_weights
                                
                            except Exception as e:
                                self.logger.warning(f"Error during weight optimization: {str(e)}")
                                continue
                        
                        accuracy = correct / len(validation_data)
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_weights = test_weights.copy()
            
            # Apply best weights
            self.weights = best_weights
            self.logger.info(f"Weight optimization completed. Best accuracy: {best_accuracy:.3f}")
            return True
            
        except Exception as e:
            self.logger.error(f"Weight optimization failed: {str(e)}")
            return False
    
    def get_ensemble_info(self) -> Dict[str, Any]:
        """Get information about the ensemble."""
        return {
            "num_models": len(self.models),
            "fusion_method": self.config.fusion_method.value,
            "models": list(self.models.keys()),
            "weights": self.weights.copy(),
            "config": self.config.__dict__
        }
    
    def set_fusion_method(self, method: FusionMethod) -> bool:
        """Change the fusion method."""
        try:
            self.config.fusion_method = method
            self.logger.info(f"Fusion method changed to {method.value}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to change fusion method: {str(e)}")
            return False
    
    def update_weights(self, weights: Dict[str, float]) -> bool:
        """Update model weights."""
        try:
            # Validate weights
            for name in weights.keys():
                if name not in self.models:
                    self.logger.warning(f"Weight provided for unknown model '{name}'")
            
            self.weights.update(weights)
            self.logger.info("Model weights updated successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to update weights: {str(e)}")
            return False 