"""
Spatial-Frequency Ensemble Manager for multi-domain deepfake detection.
This module implements intelligent fusion of spatial and frequency domain detectors.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .base_detector import BaseDetector, DetectionResult
from .ensemble_manager import EnsembleManager, EnsembleConfig, EnsembleResult, FusionMethod


class AnalysisDomain(str, Enum):
    """Analysis domains for deepfake detection."""
    SPATIAL = "spatial"
    FREQUENCY = "frequency"
    HYBRID = "hybrid"


@dataclass
class DomainWeights:
    """Weights for different analysis domains."""
    spatial_weight: float = 0.6
    frequency_weight: float = 0.4
    confidence_threshold: float = 0.5
    uncertainty_penalty: float = 0.1


@dataclass
class SpatialFrequencyResult:
    """Result of spatial-frequency ensemble analysis."""
    final_prediction: bool
    confidence_score: float
    spatial_confidence: float
    frequency_confidence: float
    domain_agreement: float
    uncertainty: float
    dominant_domain: AnalysisDomain
    individual_results: Dict[str, DetectionResult]
    fusion_metadata: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)


class AdaptiveWeightingModule(nn.Module):
    """
    Neural module for adaptive weighting of spatial and frequency predictions.
    Learns optimal fusion weights based on input characteristics.
    """
    
    def __init__(self, input_dim: int = 6, hidden_dim: int = 32):
        super(AdaptiveWeightingModule, self).__init__()
        
        self.weight_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim//2, 2),  # spatial_weight, frequency_weight
            nn.Softmax(dim=-1)
        )
        
        self.confidence_calibrator = nn.Sequential(
            nn.Linear(input_dim + 2, 16),  # input + weights
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute adaptive weights and calibrated confidence.
        
        Args:
            features: Input features [batch_size, feature_dim]
            
        Returns:
            Tuple of (domain_weights, calibration_factor)
        """
        weights = self.weight_network(features)
        
        # Combine features and weights for confidence calibration
        combined = torch.cat([features, weights], dim=-1)
        calibration = self.confidence_calibrator(combined)
        
        return weights, calibration


class SpatialFrequencyEnsemble(EnsembleManager):
    """
    Enhanced ensemble manager specifically designed for spatial-frequency fusion.
    
    Implements intelligent weighting strategies that consider:
    - Spatial artifacts (Xception, EfficientNet)
    - Frequency artifacts (F3Net)
    - Cross-domain agreement
    - Adaptive fusion based on input characteristics
    """
    
    def __init__(
        self,
        config: Optional[EnsembleConfig] = None,
        domain_weights: Optional[DomainWeights] = None,
        enable_adaptive_weighting: bool = True
    ):
        super().__init__(config)
        
        self.domain_weights = domain_weights or DomainWeights()
        self.enable_adaptive_weighting = enable_adaptive_weighting
        
        # Model domain mapping
        self.spatial_models = set()  # Xception, EfficientNet
        self.frequency_models = set()  # F3Net
        
        # Adaptive weighting module
        self.adaptive_weighter = None
        if enable_adaptive_weighting:
            self.adaptive_weighter = AdaptiveWeightingModule()
            self.adaptive_weighter.eval()
        
        self.logger = logging.getLogger(f"{__name__}.SpatialFrequencyEnsemble")
        self.logger.info("Spatial-frequency ensemble initialized")
    
    def add_spatial_model(self, name: str, detector: BaseDetector, weight: float = 1.0) -> bool:
        """Add a spatial domain model to the ensemble."""
        success = self.add_model(name, detector, weight)
        if success:
            self.spatial_models.add(name)
            self.logger.info(f"Added spatial model '{name}' to ensemble")
        return success
    
    def add_frequency_model(self, name: str, detector: BaseDetector, weight: float = 1.0) -> bool:
        """Add a frequency domain model to the ensemble."""
        success = self.add_model(name, detector, weight)
        if success:
            self.frequency_models.add(name)
            self.logger.info(f"Added frequency model '{name}' to ensemble")
        return success
    
    def predict_spatial_frequency(self, image: Image.Image) -> SpatialFrequencyResult:
        """
        Perform spatial-frequency ensemble prediction.
        
        Args:
            image: PIL Image to analyze
            
        Returns:
            SpatialFrequencyResult with comprehensive analysis
        """
        start_time = time.time()
        
        try:
            # Get individual predictions
            individual_predictions = {}
            spatial_predictions = {}
            frequency_predictions = {}
            
            # Collect spatial domain predictions
            for name in self.spatial_models:
                if name in self.models:
                    try:
                        result = self.models[name].predict(image)
                        individual_predictions[name] = result
                        spatial_predictions[name] = result
                    except Exception as e:
                        self.logger.error(f"Spatial model '{name}' failed: {str(e)}")
            
            # Collect frequency domain predictions
            for name in self.frequency_models:
                if name in self.models:
                    try:
                        result = self.models[name].predict(image)
                        individual_predictions[name] = result
                        frequency_predictions[name] = result
                    except Exception as e:
                        self.logger.error(f"Frequency model '{name}' failed: {str(e)}")
            
            # Ensure we have predictions from both domains
            if not spatial_predictions:
                raise ValueError("No spatial domain predictions available")
            if not frequency_predictions:
                raise ValueError("No frequency domain predictions available")
            
            # Calculate domain-specific confidences
            spatial_confidence = np.mean([pred.confidence for pred in spatial_predictions.values()])
            frequency_confidence = np.mean([pred.confidence for pred in frequency_predictions.values()])
            
            # Calculate domain agreement
            spatial_votes = [pred.is_deepfake for pred in spatial_predictions.values()]
            frequency_votes = [pred.is_deepfake for pred in frequency_predictions.values()]
            
            spatial_consensus = sum(spatial_votes) / len(spatial_votes)
            frequency_consensus = sum(frequency_votes) / len(frequency_votes)
            
            domain_agreement = 1.0 - abs(spatial_consensus - frequency_consensus)
            
            # Apply fusion strategy
            if self.enable_adaptive_weighting and self.adaptive_weighter:
                final_confidence, fusion_weights = self._adaptive_fusion(
                    spatial_confidence, frequency_confidence, domain_agreement, individual_predictions
                )
            else:
                final_confidence, fusion_weights = self._static_fusion(
                    spatial_confidence, frequency_confidence
                )
            
            # Determine final prediction
            final_prediction = final_confidence > 50.0
            
            # Calculate uncertainty
            uncertainty = self._calculate_cross_domain_uncertainty(
                spatial_predictions, frequency_predictions
            )
            
            # Determine dominant domain
            if fusion_weights['spatial'] > fusion_weights['frequency']:
                dominant_domain = AnalysisDomain.SPATIAL
            elif fusion_weights['frequency'] > fusion_weights['spatial']:
                dominant_domain = AnalysisDomain.FREQUENCY
            else:
                dominant_domain = AnalysisDomain.HYBRID
            
            # Create comprehensive result
            result = SpatialFrequencyResult(
                final_prediction=final_prediction,
                confidence_score=final_confidence,
                spatial_confidence=spatial_confidence,
                frequency_confidence=frequency_confidence,
                domain_agreement=domain_agreement,
                uncertainty=uncertainty,
                dominant_domain=dominant_domain,
                individual_results=individual_predictions,
                fusion_metadata={
                    'fusion_weights': fusion_weights,
                    'spatial_models': list(self.spatial_models),
                    'frequency_models': list(self.frequency_models),
                    'domain_consensus': {
                        'spatial': spatial_consensus,
                        'frequency': frequency_consensus
                    },
                    'processing_time': time.time() - start_time,
                    'adaptive_weighting_used': self.enable_adaptive_weighting
                }
            )
            
            self.logger.info(
                f"Spatial-frequency prediction: {final_confidence:.1f}% "
                f"(spatial: {spatial_confidence:.1f}%, frequency: {frequency_confidence:.1f}%, "
                f"agreement: {domain_agreement:.3f})"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Spatial-frequency prediction failed: {str(e)}")
            raise
    
    def _adaptive_fusion(
        self,
        spatial_conf: float,
        frequency_conf: float,
        agreement: float,
        predictions: Dict[str, DetectionResult]
    ) -> Tuple[float, Dict[str, float]]:
        """Apply adaptive fusion using learned weights."""
        try:
            # Extract features for adaptive weighting
            features = self._extract_fusion_features(spatial_conf, frequency_conf, agreement, predictions)
            
            with torch.no_grad():
                weights, calibration = self.adaptive_weighter(features)
                spatial_weight = weights[0, 0].item()
                frequency_weight = weights[0, 1].item()
                calibration_factor = calibration[0, 0].item()
            
            # Apply adaptive weights
            fused_confidence = (spatial_conf * spatial_weight + frequency_conf * frequency_weight)
            
            # Apply calibration
            fused_confidence = fused_confidence * calibration_factor
            
            fusion_weights = {
                'spatial': spatial_weight,
                'frequency': frequency_weight,
                'calibration': calibration_factor
            }
            
            return fused_confidence, fusion_weights
            
        except Exception as e:
            self.logger.warning(f"Adaptive fusion failed, falling back to static: {str(e)}")
            return self._static_fusion(spatial_conf, frequency_conf)
    
    def _static_fusion(self, spatial_conf: float, frequency_conf: float) -> Tuple[float, Dict[str, float]]:
        """Apply static fusion using predefined weights."""
        # Use configured domain weights
        spatial_weight = self.domain_weights.spatial_weight
        frequency_weight = self.domain_weights.frequency_weight
        
        # Normalize weights
        total_weight = spatial_weight + frequency_weight
        spatial_weight /= total_weight
        frequency_weight /= total_weight
        
        # Apply fusion
        fused_confidence = spatial_conf * spatial_weight + frequency_conf * frequency_weight
        
        fusion_weights = {
            'spatial': spatial_weight,
            'frequency': frequency_weight,
            'calibration': 1.0
        }
        
        return fused_confidence, fusion_weights
    
    def _extract_fusion_features(
        self,
        spatial_conf: float,
        frequency_conf: float,
        agreement: float,
        predictions: Dict[str, DetectionResult]
    ) -> torch.Tensor:
        """Extract features for adaptive fusion."""
        # Calculate additional statistics
        all_confidences = [pred.confidence for pred in predictions.values()]
        conf_variance = np.var(all_confidences)
        conf_mean = np.mean(all_confidences)
        
        # Extract uncertainty information
        uncertainties = [pred.uncertainty for pred in predictions.values() if pred.uncertainty is not None]
        avg_uncertainty = np.mean(uncertainties) if uncertainties else 0.0
        
        # Create feature vector
        features = torch.tensor([
            spatial_conf / 100.0,  # Normalize to [0, 1]
            frequency_conf / 100.0,
            agreement,
            conf_variance / 10000.0,  # Normalize variance
            conf_mean / 100.0,
            avg_uncertainty
        ], dtype=torch.float32).unsqueeze(0)
        
        return features
    
    def _calculate_cross_domain_uncertainty(
        self,
        spatial_predictions: Dict[str, DetectionResult],
        frequency_predictions: Dict[str, DetectionResult]
    ) -> float:
        """Calculate uncertainty considering cross-domain disagreement."""
        # Calculate within-domain uncertainties
        spatial_confidences = [pred.confidence for pred in spatial_predictions.values()]
        frequency_confidences = [pred.confidence for pred in frequency_predictions.values()]
        
        spatial_var = np.var(spatial_confidences) if len(spatial_confidences) > 1 else 0.0
        frequency_var = np.var(frequency_confidences) if len(frequency_confidences) > 1 else 0.0
        
        # Calculate cross-domain disagreement
        spatial_mean = np.mean(spatial_confidences)
        frequency_mean = np.mean(frequency_confidences)
        cross_domain_disagreement = abs(spatial_mean - frequency_mean)
        
        # Combine uncertainties
        total_uncertainty = (spatial_var + frequency_var) / 2 + cross_domain_disagreement
        
        return float(total_uncertainty)
    
    def optimize_domain_weights(
        self,
        validation_data: List[Tuple[Image.Image, bool]],
        spatial_models: List[str],
        frequency_models: List[str]
    ) -> Dict[str, float]:
        """
        Optimize domain weights using validation data.
        
        Args:
            validation_data: List of (image, ground_truth) tuples
            spatial_models: List of spatial model names
            frequency_models: List of frequency model names
            
        Returns:
            Optimized domain weights
        """
        try:
            if not validation_data:
                self.logger.warning("No validation data provided")
                return {'spatial': 0.6, 'frequency': 0.4}
            
            # Set model domains
            self.spatial_models.update(spatial_models)
            self.frequency_models.update(frequency_models)
            
            best_accuracy = 0.0
            best_weights = {'spatial': 0.6, 'frequency': 0.4}
            
            # Grid search for optimal weights
            for spatial_weight in np.arange(0.1, 1.0, 0.1):
                frequency_weight = 1.0 - spatial_weight
                
                # Test weights
                original_weights = self.domain_weights
                self.domain_weights = DomainWeights(
                    spatial_weight=spatial_weight,
                    frequency_weight=frequency_weight
                )
                
                correct = 0
                for image, ground_truth in validation_data:
                    try:
                        result = self.predict_spatial_frequency(image)
                        if result.final_prediction == ground_truth:
                            correct += 1
                    except Exception as e:
                        self.logger.warning(f"Validation prediction failed: {str(e)}")
                        continue
                
                accuracy = correct / len(validation_data)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_weights = {'spatial': spatial_weight, 'frequency': frequency_weight}
                
                # Restore original weights
                self.domain_weights = original_weights
            
            # Apply best weights
            self.domain_weights.spatial_weight = best_weights['spatial']
            self.domain_weights.frequency_weight = best_weights['frequency']
            
            self.logger.info(
                f"Domain weight optimization complete. Best accuracy: {best_accuracy:.3f} "
                f"(spatial: {best_weights['spatial']:.2f}, frequency: {best_weights['frequency']:.2f})"
            )
            
            return best_weights
            
        except Exception as e:
            self.logger.error(f"Domain weight optimization failed: {str(e)}")
            return {'spatial': 0.6, 'frequency': 0.4}
    
    def get_domain_analysis(self) -> Dict[str, Any]:
        """Get analysis of domain-specific performance."""
        return {
            'spatial_models': {
                'count': len(self.spatial_models),
                'models': list(self.spatial_models),
                'weight': self.domain_weights.spatial_weight,
                'specialization': 'facial_inconsistencies'
            },
            'frequency_models': {
                'count': len(self.frequency_models),
                'models': list(self.frequency_models),
                'weight': self.domain_weights.frequency_weight,
                'specialization': 'compression_artifacts'
            },
            'fusion_strategy': {
                'adaptive_weighting': self.enable_adaptive_weighting,
                'confidence_threshold': self.domain_weights.confidence_threshold,
                'uncertainty_penalty': self.domain_weights.uncertainty_penalty
            },
            'ensemble_info': self.get_ensemble_info()
        }
    
    def visualize_domain_contributions(
        self,
        image: Image.Image
    ) -> Dict[str, Any]:
        """
        Generate visualizations showing spatial vs frequency contributions.
        
        Args:
            image: Image to analyze
            
        Returns:
            Dictionary containing visualization data
        """
        try:
            result = self.predict_spatial_frequency(image)
            
            # Get frequency visualizations if F3Net is available
            frequency_viz = {}
            for name in self.frequency_models:
                if name in self.models and hasattr(self.models[name], 'get_frequency_visualization'):
                    frequency_viz[name] = self.models[name].get_frequency_visualization(image)
            
            return {
                'prediction_result': {
                    'final_confidence': result.confidence_score,
                    'spatial_confidence': result.spatial_confidence,
                    'frequency_confidence': result.frequency_confidence,
                    'domain_agreement': result.domain_agreement,
                    'dominant_domain': result.dominant_domain.value
                },
                'fusion_weights': result.fusion_metadata['fusion_weights'],
                'frequency_visualizations': frequency_viz,
                'domain_contributions': {
                    'spatial_weight': result.fusion_metadata['fusion_weights']['spatial'],
                    'frequency_weight': result.fusion_metadata['fusion_weights']['frequency'],
                    'spatial_models_used': result.fusion_metadata['spatial_models'],
                    'frequency_models_used': result.fusion_metadata['frequency_models']
                }
            }
            
        except Exception as e:
            self.logger.error(f"Domain visualization failed: {str(e)}")
            return {}