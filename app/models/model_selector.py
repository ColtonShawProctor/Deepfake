"""
Intelligent Model Selection for Advanced Ensemble Optimization

This module implements dynamic model selection based on input characteristics,
performance requirements, and resource constraints to optimize the ensemble
for both accuracy and efficiency.
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from PIL import Image
import cv2

from .base_detector import BaseDetector, DetectionResult


class InputComplexity(str, Enum):
    """Input complexity levels for model selection."""
    SIMPLE = "simple"      # High-quality, clear images
    MEDIUM = "medium"      # Moderate quality with some artifacts
    COMPLEX = "complex"    # Low-quality, heavily processed images
    UNKNOWN = "unknown"    # Cannot determine complexity


class ModelPerformanceTier(str, Enum):
    """Model performance tiers based on speed vs accuracy trade-offs."""
    SPEED_OPTIMIZED = "speed"      # Fast inference, good accuracy
    BALANCED = "balanced"          # Balanced speed and accuracy
    ACCURACY_OPTIMIZED = "accuracy" # High accuracy, slower inference


@dataclass
class ModelProfile:
    """Profile defining model characteristics and performance."""
    name: str
    performance_tier: ModelPerformanceTier
    base_accuracy: float
    base_inference_time: float
    memory_usage: float
    input_size: Tuple[int, int]
    complexity_threshold: float  # Minimum complexity to activate this model
    confidence_threshold: float  # Minimum confidence to trust this model
    priority: int  # Higher number = higher priority


@dataclass
class InputAnalysis:
    """Analysis of input image characteristics."""
    complexity: InputComplexity
    face_confidence: float
    image_quality: float
    noise_level: float
    resolution: Tuple[int, int]
    estimated_processing_time: float
    recommended_models: List[str]


class ModelSelector:
    """
    Intelligent model selection based on input analysis and performance requirements.
    
    Analyzes input characteristics and selects the optimal subset of models
    to achieve the best accuracy/efficiency trade-off.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.ModelSelector")
        
        # Initialize model profiles with current performance data
        self.model_profiles = {
            "EfficientNet": ModelProfile(
                name="EfficientNet",
                performance_tier=ModelPerformanceTier.SPEED_OPTIMIZED,
                base_accuracy=0.8935,
                base_inference_time=0.080,
                memory_usage=512.0,
                input_size=(224, 224),
                complexity_threshold=0.3,  # Good for simple to medium complexity
                confidence_threshold=0.7,
                priority=3
            ),
            "Xception": ModelProfile(
                name="Xception",
                performance_tier=ModelPerformanceTier.ACCURACY_OPTIMIZED,
                base_accuracy=0.966,
                base_inference_time=0.150,
                memory_usage=2048.0,
                input_size=(299, 299),
                complexity_threshold=0.6,  # Best for complex images
                confidence_threshold=0.8,
                priority=4
            ),
            "F3Net": ModelProfile(
                name="F3Net",
                performance_tier=ModelPerformanceTier.BALANCED,
                base_accuracy=0.945,
                base_inference_time=0.120,
                memory_usage=1024.0,
                input_size=(224, 224),
                complexity_threshold=0.4,  # Good for medium complexity
                confidence_threshold=0.75,
                priority=2
            ),
            "MesoNet": ModelProfile(
                name="MesoNet",
                performance_tier=ModelPerformanceTier.SPEED_OPTIMIZED,
                base_accuracy=0.87,
                base_inference_time=0.060,
                memory_usage=256.0,
                input_size=(256, 256),
                complexity_threshold=0.2,  # Good for simple images
                confidence_threshold=0.65,
                priority=1
            )
        }
        
        # Performance requirements
        self.max_inference_time = 0.200  # 200ms target
        self.max_memory_usage = 2500.0   # 2.5GB target
        self.min_accuracy_threshold = 0.95  # 95% minimum accuracy
        
        # Initialize face detection for complexity analysis
        self.face_cascade = None
        self._initialize_face_detection()
    
    def _initialize_face_detection(self):
        """Initialize face detection for input analysis."""
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if self.face_cascade.empty():
                self.logger.warning("Failed to load face detection cascade")
                self.face_cascade = None
        except Exception as e:
            self.logger.warning(f"Face detection initialization failed: {str(e)}")
            self.face_cascade = None
    
    def analyze_input(self, image: Image.Image) -> InputAnalysis:
        """
        Analyze input image to determine complexity and characteristics.
        
        Args:
            image: PIL Image to analyze
            
        Returns:
            InputAnalysis with complexity assessment and recommendations
        """
        start_time = time.time()
        
        try:
            # Convert to numpy array for analysis
            img_array = np.array(image)
            height, width = img_array.shape[:2]
            
            # Face detection confidence
            face_confidence = self._analyze_face_confidence(img_array)
            
            # Image quality assessment
            image_quality = self._analyze_image_quality(img_array)
            
            # Noise level estimation
            noise_level = self._analyze_noise_level(img_array)
            
            # Determine overall complexity
            complexity = self._determine_complexity(face_confidence, image_quality, noise_level)
            
            # Estimate processing time based on complexity
            estimated_time = self._estimate_processing_time(complexity, (height, width))
            
            # Get recommended models
            recommended_models = self._get_recommended_models(complexity, face_confidence, image_quality)
            
            analysis = InputAnalysis(
                complexity=complexity,
                face_confidence=face_confidence,
                image_quality=image_quality,
                noise_level=noise_level,
                resolution=(height, width),
                estimated_processing_time=estimated_time,
                recommended_models=recommended_models
            )
            
            self.logger.info(f"Input analysis completed in {time.time() - start_time:.3f}s: {complexity.value}")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Input analysis failed: {str(e)}")
            # Return default analysis for unknown complexity
            return InputAnalysis(
                complexity=InputComplexity.UNKNOWN,
                face_confidence=0.5,
                image_quality=0.5,
                noise_level=0.5,
                resolution=image.size[::-1],
                estimated_processing_time=0.200,
                recommended_models=["EfficientNet", "F3Net"]  # Safe defaults
            )
    
    def _analyze_face_confidence(self, img_array: np.ndarray) -> float:
        """Analyze face detection confidence."""
        if self.face_cascade is None:
            return 0.5  # Default if face detection unavailable
        
        try:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            if len(faces) > 0:
                # Calculate confidence based on face size and number of faces
                largest_face = max(faces, key=lambda x: x[2] * x[3])
                face_area = largest_face[2] * largest_face[3]
                total_area = img_array.shape[0] * img_array.shape[1]
                area_ratio = face_area / total_area
                
                # Higher confidence for larger, single faces
                confidence = min(1.0, area_ratio * 2.0) * (1.0 / len(faces))
                return confidence
            else:
                return 0.0  # No face detected
                
        except Exception as e:
            self.logger.warning(f"Face analysis failed: {str(e)}")
            return 0.5
    
    def _analyze_image_quality(self, img_array: np.ndarray) -> float:
        """Analyze image quality using various metrics."""
        try:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Laplacian variance (sharpness)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Contrast analysis
            contrast = gray.std()
            
            # Brightness analysis
            brightness = gray.mean()
            
            # Normalize metrics (0-1 scale)
            sharpness_score = min(1.0, laplacian_var / 1000.0)
            contrast_score = min(1.0, contrast / 100.0)
            brightness_score = 1.0 - abs(brightness - 128) / 128.0
            
            # Combined quality score
            quality = (sharpness_score * 0.4 + contrast_score * 0.3 + brightness_score * 0.3)
            return min(1.0, max(0.0, quality))
            
        except Exception as e:
            self.logger.warning(f"Quality analysis failed: {str(e)}")
            return 0.5
    
    def _analyze_noise_level(self, img_array: np.ndarray) -> float:
        """Analyze noise level in the image."""
        try:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Use Laplacian to detect edges and noise
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            noise_level = laplacian.var()
            
            # Normalize to 0-1 scale (higher = more noise)
            normalized_noise = min(1.0, noise_level / 500.0)
            return normalized_noise
            
        except Exception as e:
            self.logger.warning(f"Noise analysis failed: {str(e)}")
            return 0.5
    
    def _determine_complexity(self, face_confidence: float, image_quality: float, noise_level: float) -> InputComplexity:
        """Determine input complexity based on analysis metrics."""
        # Simple: High face confidence, high quality, low noise
        if face_confidence > 0.7 and image_quality > 0.7 and noise_level < 0.3:
            return InputComplexity.SIMPLE
        
        # Complex: Low face confidence, low quality, high noise
        elif face_confidence < 0.3 or image_quality < 0.4 or noise_level > 0.7:
            return InputComplexity.COMPLEX
        
        # Medium: Everything else
        else:
            return InputComplexity.MEDIUM
    
    def _estimate_processing_time(self, complexity: InputComplexity, resolution: Tuple[int, int]) -> float:
        """Estimate processing time based on complexity and resolution."""
        base_time = {
            InputComplexity.SIMPLE: 0.100,
            InputComplexity.MEDIUM: 0.150,
            InputComplexity.COMPLEX: 0.250,
            InputComplexity.UNKNOWN: 0.200
        }[complexity]
        
        # Adjust for resolution (larger images take longer)
        height, width = resolution
        resolution_factor = min(2.0, (height * width) / (224 * 224))
        
        return base_time * resolution_factor
    
    def _get_recommended_models(self, complexity: InputComplexity, face_confidence: float, image_quality: float) -> List[str]:
        """Get recommended models based on complexity analysis."""
        recommendations = []
        
        if complexity == InputComplexity.SIMPLE:
            # For simple images, use fast models
            recommendations = ["MesoNet", "EfficientNet"]
            
        elif complexity == InputComplexity.MEDIUM:
            # For medium complexity, use balanced models
            recommendations = ["EfficientNet", "F3Net"]
            
        elif complexity == InputComplexity.COMPLEX:
            # For complex images, use high-accuracy models
            recommendations = ["Xception", "F3Net", "EfficientNet"]
            
        else:  # UNKNOWN
            # Default to safe, balanced selection
            recommendations = ["EfficientNet", "F3Net"]
        
        # Add Xception for high-confidence cases regardless of complexity
        if face_confidence > 0.8 and image_quality > 0.8:
            if "Xception" not in recommendations:
                recommendations.append("Xception")
        
        return recommendations
    
    def select_models(self, analysis: InputAnalysis, available_models: Dict[str, BaseDetector], 
                     max_models: int = 3) -> List[str]:
        """
        Select optimal models based on input analysis and constraints.
        
        Args:
            analysis: Input analysis results
            available_models: Dictionary of available model instances
            max_models: Maximum number of models to select
            
        Returns:
            List of selected model names
        """
        try:
            # Start with recommended models
            selected = analysis.recommended_models.copy()
            
            # Filter to only available models
            selected = [model for model in selected if model in available_models]
            
            # If we have too many models, prioritize by performance tier and accuracy
            if len(selected) > max_models:
                # Sort by priority and accuracy
                model_scores = []
                for model_name in selected:
                    profile = self.model_profiles.get(model_name)
                    if profile:
                        # Score based on priority, accuracy, and suitability for complexity
                        score = (
                            profile.priority * 0.4 +
                            profile.base_accuracy * 0.4 +
                            (1.0 - profile.base_inference_time) * 0.2
                        )
                        model_scores.append((model_name, score))
                
                # Sort by score and take top models
                model_scores.sort(key=lambda x: x[1], reverse=True)
                selected = [model for model, _ in model_scores[:max_models]]
            
            # Ensure we have at least one model
            if not selected:
                selected = ["EfficientNet"]  # Safe fallback
            
            self.logger.info(f"Selected models for {analysis.complexity.value} complexity: {selected}")
            return selected
            
        except Exception as e:
            self.logger.error(f"Model selection failed: {str(e)}")
            return ["EfficientNet"]  # Safe fallback
    
    def get_selection_rationale(self, analysis: InputAnalysis, selected_models: List[str]) -> Dict[str, Any]:
        """Get detailed rationale for model selection."""
        rationale = {
            "input_analysis": {
                "complexity": analysis.complexity.value,
                "face_confidence": analysis.face_confidence,
                "image_quality": analysis.image_quality,
                "noise_level": analysis.noise_level,
                "resolution": analysis.resolution
            },
            "selected_models": selected_models,
            "model_profiles": {},
            "selection_criteria": {
                "max_inference_time": self.max_inference_time,
                "max_memory_usage": self.max_memory_usage,
                "min_accuracy_threshold": self.min_accuracy_threshold
            }
        }
        
        # Add profile details for selected models
        for model_name in selected_models:
            profile = self.model_profiles.get(model_name)
            if profile:
                rationale["model_profiles"][model_name] = {
                    "performance_tier": profile.performance_tier.value,
                    "base_accuracy": profile.base_accuracy,
                    "base_inference_time": profile.base_inference_time,
                    "memory_usage": profile.memory_usage,
                    "priority": profile.priority
                }
        
        return rationale
