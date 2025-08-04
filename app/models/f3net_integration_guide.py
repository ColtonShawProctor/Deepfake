"""
F3Net Integration Guide and Usage Examples
This module provides comprehensive examples and guidance for integrating F3Net
frequency-domain analysis into the existing spatial ensemble system.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
from PIL import Image

from .f3net_detector import F3NetDetector
from .xception_detector import XceptionDetector
from .efficientnet_detector import EfficientNetDetector
from .spatial_frequency_ensemble import SpatialFrequencyEnsemble, DomainWeights
from .ensemble_manager import EnsembleConfig, FusionMethod


class F3NetIntegrationManager:
    """
    Manager class for integrating F3Net into existing spatial detection systems.
    Provides high-level interface for setting up and running spatial-frequency ensembles.
    """
    
    def __init__(self, models_dir: str = "models", device: str = "auto"):
        self.models_dir = Path(models_dir)
        self.device = device
        self.logger = logging.getLogger(f"{__name__}.F3NetIntegrationManager")
        
        # Initialize detectors
        self.spatial_detectors = {}
        self.frequency_detectors = {}
        self.ensemble = None
        
        self._setup_directories()
    
    def _setup_directories(self):
        """Create necessary directories for model weights and outputs."""
        self.models_dir.mkdir(exist_ok=True)
        (self.models_dir / "weights").mkdir(exist_ok=True)
        (self.models_dir / "visualizations").mkdir(exist_ok=True)
    
    def setup_comprehensive_ensemble(self) -> bool:
        """
        Set up a comprehensive spatial-frequency ensemble with all three pillars.
        
        Returns:
            True if setup successful, False otherwise
        """
        try:
            self.logger.info("Setting up comprehensive spatial-frequency ensemble...")
            
            # Initialize spatial detectors
            self.logger.info("Initializing spatial detectors...")
            
            # Xception detector (high accuracy spatial analysis)
            xception = XceptionDetector(device=self.device)
            xception_path = self.models_dir / "weights" / "xception_weights.pth"
            if xception.load_model(str(xception_path) if xception_path.exists() else None):
                self.spatial_detectors["xception"] = xception
                self.logger.info("Xception detector loaded successfully")
            else:
                self.logger.warning("Xception detector failed to load")
            
            # EfficientNet detector (mobile-optimized spatial analysis)
            efficientnet = EfficientNetDetector(device=self.device)
            efficientnet_path = self.models_dir / "weights" / "efficientnet_weights.pth"
            if efficientnet.load_model(str(efficientnet_path) if efficientnet_path.exists() else None):
                self.spatial_detectors["efficientnet"] = efficientnet
                self.logger.info("EfficientNet detector loaded successfully")
            else:
                self.logger.warning("EfficientNet detector failed to load")
            
            # Initialize frequency detector
            self.logger.info("Initializing frequency detector...")
            
            # F3Net detector (frequency-domain analysis)
            f3net = F3NetDetector(device=self.device)
            f3net_path = self.models_dir / "weights" / "f3net_weights.pth"
            if f3net.load_model(str(f3net_path) if f3net_path.exists() else None):
                self.frequency_detectors["f3net"] = f3net
                self.logger.info("F3Net detector loaded successfully")
            else:
                self.logger.warning("F3Net detector failed to load")
            
            # Setup ensemble with optimized domain weights
            self.logger.info("Configuring spatial-frequency ensemble...")
            
            ensemble_config = EnsembleConfig(
                fusion_method=FusionMethod.ATTENTION_FUSION,
                temperature=1.2,  # Slightly higher temperature for better calibration
                min_models=2,
                confidence_threshold=0.5,
                enable_uncertainty=True,
                enable_attention=True
            )
            
            domain_weights = DomainWeights(
                spatial_weight=0.65,  # Slight preference for spatial due to Xception's high accuracy
                frequency_weight=0.35,  # F3Net provides unique frequency perspective
                confidence_threshold=0.5,
                uncertainty_penalty=0.1
            )
            
            self.ensemble = SpatialFrequencyEnsemble(
                config=ensemble_config,
                domain_weights=domain_weights,
                enable_adaptive_weighting=True
            )
            
            # Add spatial models to ensemble
            for name, detector in self.spatial_detectors.items():
                if detector.is_loaded():
                    weight = 1.2 if name == "xception" else 1.0  # Higher weight for Xception
                    self.ensemble.add_spatial_model(name, detector, weight)
            
            # Add frequency models to ensemble
            for name, detector in self.frequency_detectors.items():
                if detector.is_loaded():
                    self.ensemble.add_frequency_model(name, detector, 1.0)
            
            loaded_models = len(self.spatial_detectors) + len(self.frequency_detectors)
            self.logger.info(f"Ensemble setup complete with {loaded_models} models")
            
            return loaded_models >= 2  # Need at least 2 models for meaningful ensemble
            
        except Exception as e:
            self.logger.error(f"Ensemble setup failed: {str(e)}")
            return False
    
    def analyze_with_frequency_insights(self, image: Image.Image) -> Dict[str, Any]:
        """
        Perform comprehensive analysis with frequency-domain insights.
        
        Args:
            image: PIL Image to analyze
            
        Returns:
            Comprehensive analysis results with frequency insights
        """
        if not self.ensemble:
            raise RuntimeError("Ensemble not initialized. Call setup_comprehensive_ensemble() first.")
        
        try:
            # Perform spatial-frequency ensemble prediction
            result = self.ensemble.predict_spatial_frequency(image)
            
            # Generate frequency visualizations
            frequency_viz = self.ensemble.visualize_domain_contributions(image)
            
            # Analyze frequency characteristics
            frequency_analysis = self._analyze_frequency_characteristics(image)
            
            # Create comprehensive report
            analysis_report = {
                'detection_result': {
                    'is_deepfake': result.final_prediction,
                    'confidence_score': result.confidence_score,
                    'uncertainty': result.uncertainty,
                    'processing_time': result.fusion_metadata['processing_time']
                },
                'domain_analysis': {
                    'spatial_confidence': result.spatial_confidence,
                    'frequency_confidence': result.frequency_confidence,
                    'domain_agreement': result.domain_agreement,
                    'dominant_domain': result.dominant_domain.value,
                    'fusion_weights': result.fusion_metadata['fusion_weights']
                },
                'frequency_insights': frequency_analysis,
                'individual_models': {
                    name: {
                        'confidence': pred.confidence,
                        'prediction': pred.is_deepfake,
                        'processing_time': pred.inference_time,
                        'uncertainty': pred.uncertainty
                    }
                    for name, pred in result.individual_results.items()
                },
                'visualizations': frequency_viz,
                'recommendations': self._generate_recommendations(result, frequency_analysis)
            }
            
            return analysis_report
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            raise
    
    def _analyze_frequency_characteristics(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze frequency characteristics of the input image."""
        if "f3net" not in self.frequency_detectors:
            return {"error": "F3Net detector not available"}
        
        try:
            f3net = self.frequency_detectors["f3net"]
            
            # Get frequency analysis capabilities
            capabilities = f3net.get_frequency_analysis_capabilities()
            
            # Perform frequency analysis
            if hasattr(f3net, 'get_frequency_visualization'):
                freq_viz = f3net.get_frequency_visualization(image)
                
                return {
                    'capabilities': capabilities,
                    'dct_analysis': 'available' if 'dct_visualization' in freq_viz else 'unavailable',
                    'attention_patterns': 'available' if 'attention_heatmap' in freq_viz else 'unavailable',
                    'frequency_spectrum': 'available' if 'frequency_spectrum' in freq_viz else 'unavailable',
                    'compression_artifacts': self._detect_compression_artifacts(image),
                    'frequency_domain_score': self._calculate_frequency_domain_score(freq_viz)
                }
            else:
                return {
                    'capabilities': capabilities,
                    'analysis': 'basic_frequency_analysis_only'
                }
                
        except Exception as e:
            self.logger.error(f"Frequency analysis failed: {str(e)}")
            return {"error": str(e)}
    
    def _detect_compression_artifacts(self, image: Image.Image) -> Dict[str, Any]:
        """Detect JPEG compression artifacts that may indicate manipulation."""
        try:
            import numpy as np
            from scipy.fft import dct, fft2
            
            # Convert to numpy array
            img_array = np.array(image.convert('L'))
            
            # Analyze DCT coefficients for compression patterns
            dct_coeffs = dct(dct(img_array.T, norm='ortho').T, norm='ortho')
            
            # Check for typical JPEG quantization patterns
            quantization_artifacts = np.sum(np.abs(dct_coeffs) < 1.0) / dct_coeffs.size
            
            # Analyze frequency domain for block artifacts
            freq_domain = np.abs(fft2(img_array))
            block_artifacts = self._detect_block_artifacts(freq_domain)
            
            return {
                'quantization_artifacts': float(quantization_artifacts),
                'block_artifacts': block_artifacts,
                'compression_score': (quantization_artifacts + block_artifacts) / 2,
                'analysis': 'completed'
            }
            
        except Exception as e:
            self.logger.error(f"Compression artifact detection failed: {str(e)}")
            return {'error': str(e)}
    
    def _detect_block_artifacts(self, freq_domain: np.ndarray) -> float:
        """Detect blocking artifacts in frequency domain."""
        try:
            # Look for periodic patterns that indicate 8x8 blocking
            h, w = freq_domain.shape
            
            # Check for peaks at multiples of 8 in frequency domain
            block_indicators = 0
            for i in range(8, min(h, w), 8):
                if i < h:
                    block_indicators += freq_domain[i, :].mean()
                if i < w:
                    block_indicators += freq_domain[:, i].mean()
            
            # Normalize by image size and frequency magnitude
            normalized_score = block_indicators / (freq_domain.mean() * min(h, w))
            return min(1.0, normalized_score)
            
        except Exception:
            return 0.0
    
    def _calculate_frequency_domain_score(self, freq_viz: Dict[str, Any]) -> float:
        """Calculate overall frequency domain anomaly score."""
        if not freq_viz:
            return 0.0
        
        try:
            score = 0.0
            components = 0
            
            # Score based on available visualizations
            if 'frequency_spectrum' in freq_viz:
                spectrum = freq_viz['frequency_spectrum']
                if isinstance(spectrum, np.ndarray):
                    # High-frequency component analysis
                    high_freq_energy = np.mean(spectrum[-len(spectrum)//4:])
                    score += high_freq_energy
                    components += 1
            
            if 'attention_heatmap' in freq_viz:
                # Attention concentration indicates anomalies
                score += 0.3  # Base score for attention-based detection
                components += 1
            
            # Normalize score
            return score / components if components > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _generate_recommendations(
        self,
        result: Any,
        frequency_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate analysis recommendations based on results."""
        recommendations = []
        
        # Confidence-based recommendations
        if result.confidence_score > 80:
            recommendations.append("High confidence detection - strong evidence of manipulation")
        elif result.confidence_score > 60:
            recommendations.append("Moderate confidence - additional verification recommended")
        elif result.confidence_score < 40:
            recommendations.append("Low confidence - image likely authentic")
        
        # Domain agreement recommendations
        if result.domain_agreement < 0.5:
            recommendations.append("Spatial and frequency domains disagree - manual review advised")
        elif result.domain_agreement > 0.8:
            recommendations.append("High domain agreement - reliable detection")
        
        # Uncertainty recommendations
        if result.uncertainty > 20:
            recommendations.append("High uncertainty detected - consider additional analysis")
        
        # Frequency-specific recommendations
        if frequency_analysis.get('compression_artifacts', {}).get('compression_score', 0) > 0.7:
            recommendations.append("Strong compression artifacts detected - possible post-processing")
        
        # Dominant domain recommendations
        if result.dominant_domain.value == "frequency":
            recommendations.append("Frequency domain provides strongest evidence - focus on compression analysis")
        elif result.dominant_domain.value == "spatial":
            recommendations.append("Spatial domain provides strongest evidence - focus on facial inconsistencies")
        
        return recommendations
    
    def optimize_for_dataset(
        self,
        validation_images: List[Tuple[Image.Image, bool]],
        optimization_target: str = "accuracy"
    ) -> Dict[str, Any]:
        """
        Optimize ensemble parameters for a specific dataset.
        
        Args:
            validation_images: List of (image, is_deepfake) tuples
            optimization_target: Target metric ("accuracy", "precision", "recall")
            
        Returns:
            Optimization results and new parameters
        """
        if not self.ensemble:
            raise RuntimeError("Ensemble not initialized")
        
        try:
            self.logger.info(f"Optimizing ensemble for {len(validation_images)} validation samples...")
            
            # Optimize domain weights
            spatial_models = list(self.spatial_detectors.keys())
            frequency_models = list(self.frequency_detectors.keys())
            
            optimized_weights = self.ensemble.optimize_domain_weights(
                validation_images, spatial_models, frequency_models
            )
            
            # Test optimized performance
            correct = 0
            total_spatial_conf = 0
            total_frequency_conf = 0
            total_agreement = 0
            
            for image, ground_truth in validation_images:
                try:
                    result = self.ensemble.predict_spatial_frequency(image)
                    if result.final_prediction == ground_truth:
                        correct += 1
                    
                    total_spatial_conf += result.spatial_confidence
                    total_frequency_conf += result.frequency_confidence
                    total_agreement += result.domain_agreement
                    
                except Exception as e:
                    self.logger.warning(f"Validation prediction failed: {str(e)}")
            
            accuracy = correct / len(validation_images)
            avg_spatial_conf = total_spatial_conf / len(validation_images)
            avg_frequency_conf = total_frequency_conf / len(validation_images)
            avg_agreement = total_agreement / len(validation_images)
            
            optimization_results = {
                'optimized_weights': optimized_weights,
                'performance_metrics': {
                    'accuracy': accuracy,
                    'average_spatial_confidence': avg_spatial_conf,
                    'average_frequency_confidence': avg_frequency_conf,
                    'average_domain_agreement': avg_agreement
                },
                'recommendations': [
                    f"Achieved {accuracy:.2%} accuracy on validation set",
                    f"Optimal spatial weight: {optimized_weights['spatial']:.2f}",
                    f"Optimal frequency weight: {optimized_weights['frequency']:.2f}",
                    f"Average domain agreement: {avg_agreement:.3f}"
                ]
            }
            
            self.logger.info(f"Optimization complete. Accuracy: {accuracy:.2%}")
            return optimization_results
            
        except Exception as e:
            self.logger.error(f"Optimization failed: {str(e)}")
            return {'error': str(e)}
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get comprehensive status of F3Net integration."""
        return {
            'spatial_detectors': {
                name: {
                    'loaded': detector.is_loaded(),
                    'model_info': detector.get_model_info().__dict__
                }
                for name, detector in self.spatial_detectors.items()
            },
            'frequency_detectors': {
                name: {
                    'loaded': detector.is_loaded(),
                    'model_info': detector.get_model_info().__dict__,
                    'capabilities': detector.get_frequency_analysis_capabilities()
                }
                for name, detector in self.frequency_detectors.items()
            },
            'ensemble_status': {
                'initialized': self.ensemble is not None,
                'domain_analysis': self.ensemble.get_domain_analysis() if self.ensemble else None
            },
            'integration_health': {
                'total_models': len(self.spatial_detectors) + len(self.frequency_detectors),
                'loaded_models': sum(1 for d in self.spatial_detectors.values() if d.is_loaded()) +
                               sum(1 for d in self.frequency_detectors.values() if d.is_loaded()),
                'spatial_frequency_balance': len(self.spatial_detectors) > 0 and len(self.frequency_detectors) > 0
            }
        }


# Usage Examples and Integration Guide

def example_basic_integration():
    """Example: Basic F3Net integration with existing spatial models."""
    
    # Initialize integration manager
    manager = F3NetIntegrationManager(models_dir="./models", device="cuda")
    
    # Setup comprehensive ensemble
    if manager.setup_comprehensive_ensemble():
        print("✅ Ensemble setup successful")
        
        # Load test image
        test_image = Image.open("test_deepfake.jpg")
        
        # Perform analysis
        results = manager.analyze_with_frequency_insights(test_image)
        
        # Print results
        print(f"Detection: {'DEEPFAKE' if results['detection_result']['is_deepfake'] else 'AUTHENTIC'}")
        print(f"Confidence: {results['detection_result']['confidence_score']:.1f}%")
        print(f"Domain Agreement: {results['domain_analysis']['domain_agreement']:.3f}")
        print(f"Dominant Domain: {results['domain_analysis']['dominant_domain']}")
        
        return results
    else:
        print("❌ Ensemble setup failed")
        return None


def example_optimization_workflow():
    """Example: Optimize ensemble for specific dataset."""
    
    manager = F3NetIntegrationManager()
    manager.setup_comprehensive_ensemble()
    
    # Load validation dataset
    validation_data = [
        (Image.open(f"val_{i}.jpg"), i % 2 == 0)  # Mock validation data
        for i in range(100)
    ]
    
    # Optimize ensemble
    optimization_results = manager.optimize_for_dataset(validation_data)
    
    print("Optimization Results:")
    print(f"Accuracy: {optimization_results['performance_metrics']['accuracy']:.2%}")
    print(f"Optimal Weights: {optimization_results['optimized_weights']}")
    
    return optimization_results


def example_frequency_visualization():
    """Example: Generate frequency-domain visualizations."""
    
    manager = F3NetIntegrationManager()
    manager.setup_comprehensive_ensemble()
    
    test_image = Image.open("suspicious_image.jpg")
    
    # Get comprehensive analysis with visualizations
    results = manager.analyze_with_frequency_insights(test_image)
    
    # Extract frequency visualizations
    freq_viz = results['visualizations']['frequency_visualizations']
    
    if 'f3net' in freq_viz:
        f3net_viz = freq_viz['f3net']
        
        # Save visualizations
        if 'dct_visualization' in f3net_viz:
            dct_viz = Image.fromarray(f3net_viz['dct_visualization'])
            dct_viz.save("dct_analysis.png")
            
        if 'attention_heatmap' in f3net_viz:
            attention_viz = Image.fromarray(f3net_viz['attention_heatmap'])
            attention_viz.save("frequency_attention.png")
        
        print("✅ Frequency visualizations saved")
    
    return results


if __name__ == "__main__":
    # Run integration examples
    print("🚀 F3Net Integration Examples")
    print("=" * 50)
    
    print("\n1. Basic Integration Test")
    example_basic_integration()
    
    print("\n2. Optimization Workflow")
    example_optimization_workflow()
    
    print("\n3. Frequency Visualization")
    example_frequency_visualization()