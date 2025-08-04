"""
Optimized Ensemble Detector

Integrates advanced ensemble techniques with the existing deepfake detection system.
Provides a unified interface that maintains compatibility while leveraging
state-of-the-art optimization strategies.
"""

import os
import logging
import time
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

# Import existing components
from .deepfake_models import ModelManager, DetectionResult
from .advanced_ensemble import HierarchicalEnsemble
from .cross_dataset_optimizer import CrossDatasetOptimizer

logger = logging.getLogger(__name__)

class OptimizedEnsembleDetector:
    """
    Advanced ensemble detector with state-of-the-art optimization techniques
    
    Features:
    - Hierarchical ensemble with attention-based merging
    - Confidence calibration (temperature scaling, Platt scaling)
    - Uncertainty quantification with Monte Carlo dropout
    - Adaptive weighting based on input characteristics
    - Cross-dataset generalization optimization
    - Model disagreement resolution
    """
    
    def __init__(self, models_dir: str = "models", device: str = "auto", 
                 optimization_level: str = "advanced"):
        """
        Initialize optimized ensemble detector
        
        Args:
            models_dir: Directory containing model weights
            device: Computing device ('auto', 'cpu', 'cuda')
            optimization_level: 'basic', 'advanced', 'research'
        """
        self.models_dir = Path(models_dir)
        self.device = device
        self.optimization_level = optimization_level
        self.logger = logging.getLogger(__name__)
        
        # Initialize base model manager
        try:
            self.base_model_manager = ModelManager(str(models_dir), device)
            self.base_model_manager.load_all_models()
            self.logger.info("Base model manager initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize base models: {e}")
            self.base_model_manager = None
        
        # Initialize advanced components based on optimization level
        self.advanced_ensemble = None
        self.cross_dataset_optimizer = None
        
        if optimization_level in ['advanced', 'research']:
            self._initialize_advanced_components()
        
        self.logger.info(f"OptimizedEnsembleDetector initialized with {optimization_level} optimization")
    
    def _initialize_advanced_components(self):
        """Initialize advanced optimization components"""
        try:
            # Get models from base manager
            if self.base_model_manager:
                models_dict = self.base_model_manager.ensemble.models
                
                # Initialize hierarchical ensemble
                self.advanced_ensemble = HierarchicalEnsemble(models_dict, self.device)
                
                # Initialize cross-dataset optimizer
                self.cross_dataset_optimizer = CrossDatasetOptimizer(models_dict)
                
                self.logger.info("Advanced optimization components initialized")
            else:
                self.logger.warning("Cannot initialize advanced components without base models")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize advanced components: {e}")
    
    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze image with optimized ensemble techniques
        
        Args:
            image_path: Path to image file
            
        Returns:
            Enhanced detection result with optimization details
        """
        start_time = time.time()
        
        try:
            # Validate input
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            self.logger.info(f"Starting optimized analysis for: {image_path}")
            
            # Load image
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Convert to numpy for analysis
                image_array = np.array(img)
                
                # Perform analysis based on optimization level
                if self.optimization_level == 'basic' or not self.advanced_ensemble:
                    return self._basic_analysis(img, image_path, start_time)
                else:
                    return self._advanced_analysis(image_array, img, image_path, start_time)
                    
        except Exception as e:
            self.logger.error(f"Analysis failed for {image_path}: {str(e)}")
            return self._create_error_result(str(e), time.time() - start_time)
    
    def _basic_analysis(self, img: Image.Image, image_path: str, start_time: float) -> Dict[str, Any]:
        """Perform basic ensemble analysis"""
        if not self.base_model_manager:
            raise RuntimeError("Base model manager not available")
        
        result = self.base_model_manager.predict(img)
        processing_time = time.time() - start_time
        
        return {
            "confidence_score": result.confidence_score,
            "is_deepfake": result.is_deepfake,
            "analysis_metadata": {
                "optimization_level": "basic",
                "model_predictions": result.metadata.get("individual_predictions", {}),
                "ensemble_method": result.metadata.get("ensemble_method", "basic"),
                "processing_time": processing_time,
                "models_used": list(self.base_model_manager.ensemble.models.keys())
            },
            "analysis_time": datetime.utcnow().isoformat(),
            "processing_time_seconds": processing_time,
            "error": None
        }
    
    def _advanced_analysis(self, image_array: np.ndarray, img: Image.Image, 
                          image_path: str, start_time: float) -> Dict[str, Any]:
        """Perform advanced optimized ensemble analysis"""
        
        # Step 1: Get base predictions from individual models
        base_predictions = {}
        if self.base_model_manager:
            for model_name, model in self.base_model_manager.ensemble.models.items():
                try:
                    model_result = model.predict(img)
                    base_predictions[model_name] = model_result.confidence_score / 100.0
                except Exception as e:
                    self.logger.warning(f"Model {model_name} failed: {e}")
                    base_predictions[model_name] = 0.5
        
        # Step 2: Advanced ensemble prediction
        ensemble_result = self.advanced_ensemble.predict(image_array, return_detailed=True)
        
        # Step 3: Cross-dataset optimization (if available)
        optimization_result = None
        if self.cross_dataset_optimizer:
            # Analyze image characteristics for dataset identification
            image_characteristics = self._analyze_image_characteristics(image_array)
            
            optimization_result = self.cross_dataset_optimizer.optimize_prediction(
                image_array, base_predictions, image_characteristics
            )
            
            # Use optimized prediction if available
            final_confidence = optimization_result['optimized_prediction'] * 100.0
        else:
            final_confidence = ensemble_result.confidence_score
        
        processing_time = time.time() - start_time
        
        # Create comprehensive result
        result = {
            "confidence_score": final_confidence,
            "is_deepfake": final_confidence > 50.0,
            "analysis_metadata": {
                "optimization_level": self.optimization_level,
                "ensemble_details": {
                    "base_predictions": {k: v * 100 for k, v in base_predictions.items()},
                    "attention_weights": ensemble_result.attention_weights,
                    "disagreement_score": ensemble_result.disagreement_score,
                    "uncertainty": ensemble_result.uncertainty,
                    "confidence_interval": ensemble_result.confidence_interval,
                    "calibrated_confidence": ensemble_result.calibrated_confidence * 100.0,
                    "resolution_strategy": ensemble_result.explanation.get('resolution_strategy', 'unknown')
                },
                "cross_dataset_optimization": optimization_result,
                "image_analysis": {
                    "characteristics": self._analyze_image_characteristics(image_array),
                    "adaptive_weights": ensemble_result.explanation.get('adaptive_weights', {}),
                    "identified_dataset": optimization_result.get('source_dataset', 'unknown') if optimization_result else 'unknown'
                },
                "processing_breakdown": {
                    "ensemble_time": ensemble_result.processing_time,
                    "optimization_time": processing_time - ensemble_result.processing_time,
                    "total_time": processing_time
                },
                "models_used": list(base_predictions.keys()),
                "optimization_stages": ensemble_result.explanation.get('ensemble_stages', [])
            },
            "analysis_time": datetime.utcnow().isoformat(),
            "processing_time_seconds": processing_time,
            "error": None
        }
        
        self.logger.info(
            f"Advanced analysis complete for {image_path}: "
            f"confidence={final_confidence:.1f}%, "
            f"uncertainty={ensemble_result.uncertainty:.3f}, "
            f"disagreement={ensemble_result.disagreement_score:.3f}, "
            f"processing_time={processing_time:.3f}s"
        )
        
        return result
    
    def _analyze_image_characteristics(self, image_array: np.ndarray) -> Dict[str, float]:
        """Analyze image characteristics for optimization"""
        try:
            import cv2
            
            # Basic image characteristics
            height, width = image_array.shape[:2]
            
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            
            characteristics = {
                'resolution': min(height * width / (224 * 224), 2.0),
                'aspect_ratio': width / height if height > 0 else 1.0,
            }
            
            # Compression analysis (simplified)
            try:
                # DCT-based compression estimation
                dct = cv2.dct(gray.astype(np.float32))
                high_freq = np.sum(np.abs(dct[4:, 4:]))
                total_energy = np.sum(np.abs(dct))
                characteristics['compression'] = 1.0 - (high_freq / total_energy) if total_energy > 0 else 0.5
            except:
                characteristics['compression'] = 0.5
            
            # Blur analysis
            try:
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                characteristics['blur'] = max(0.0, min(1.0, 1.0 - laplacian_var / 1000.0))
            except:
                characteristics['blur'] = 0.5
            
            # Noise analysis
            try:
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                noise = np.std(gray - blurred)
                characteristics['noise'] = min(noise / 50.0, 1.0)
            except:
                characteristics['noise'] = 0.5
            
            # Face quality (edge density proxy)
            try:
                edges = cv2.Canny(gray, 50, 150)
                edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
                characteristics['face_quality'] = min(edge_density * 10, 1.0)
            except:
                characteristics['face_quality'] = 0.5
            
            return characteristics
            
        except Exception as e:
            self.logger.warning(f"Failed to analyze image characteristics: {e}")
            return {
                'resolution': 1.0,
                'aspect_ratio': 1.0,
                'compression': 0.5,
                'blur': 0.5,
                'noise': 0.5,
                'face_quality': 0.5
            }
    
    def _create_error_result(self, error_msg: str, processing_time: float) -> Dict[str, Any]:
        """Create standardized error result"""
        return {
            "confidence_score": 0.0,
            "is_deepfake": False,
            "analysis_metadata": {
                "error": error_msg,
                "optimization_level": self.optimization_level,
                "processing_time": processing_time
            },
            "analysis_time": datetime.utcnow().isoformat(),
            "processing_time_seconds": processing_time,
            "error": error_msg
        }
    
    def train_calibration(self, validation_data: List[tuple]):
        """Train confidence calibration on validation data"""
        if self.advanced_ensemble and validation_data:
            try:
                # Convert validation data format
                cal_data = []
                for image_path, label in validation_data:
                    try:
                        with Image.open(image_path) as img:
                            if img.mode != 'RGB':
                                img = img.convert('RGB')
                            image_array = np.array(img)
                            cal_data.append((image_array, label))
                    except Exception as e:
                        self.logger.warning(f"Failed to load validation image {image_path}: {e}")
                
                if cal_data:
                    self.advanced_ensemble.train_calibration(cal_data)
                    self.logger.info(f"Calibration training completed with {len(cal_data)} samples")
                else:
                    self.logger.warning("No valid calibration data available")
                    
            except Exception as e:
                self.logger.error(f"Calibration training failed: {e}")
    
    def update_performance_tracking(self, dataset: str, accuracy: float):
        """Update performance tracking for optimization"""
        if self.cross_dataset_optimizer:
            self.cross_dataset_optimizer.update_performance(dataset, accuracy)
        
        if self.advanced_ensemble:
            # Update ensemble performance tracking
            for model_name in self.advanced_ensemble.models.keys():
                self.advanced_ensemble.update_performance(model_name, accuracy)
    
    def get_detector_info(self) -> Dict[str, Any]:
        """Get comprehensive detector information"""
        base_info = {
            "name": "Optimized Multi-Model Ensemble Detector",
            "version": "2.0.0",
            "optimization_level": self.optimization_level,
            "device": self.device
        }
        
        if self.base_model_manager:
            base_info.update(self.base_model_manager.get_model_info())
        
        if self.advanced_ensemble:
            base_info["advanced_features"] = self.advanced_ensemble.get_ensemble_info()
        
        if self.cross_dataset_optimizer:
            base_info["cross_dataset_optimization"] = self.cross_dataset_optimizer.get_optimization_info()
        
        return base_info
    
    def set_optimization_level(self, level: str):
        """Change optimization level dynamically"""
        if level in ['basic', 'advanced', 'research']:
            self.optimization_level = level
            if level in ['advanced', 'research'] and not self.advanced_ensemble:
                self._initialize_advanced_components()
            self.logger.info(f"Optimization level changed to: {level}")
        else:
            self.logger.warning(f"Invalid optimization level: {level}")

def create_optimized_detector(models_dir: str = "models", device: str = "auto",
                            optimization_level: str = "advanced") -> OptimizedEnsembleDetector:
    """Factory function to create optimized ensemble detector"""
    return OptimizedEnsembleDetector(models_dir, device, optimization_level)