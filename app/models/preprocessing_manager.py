"""
Unified Preprocessing Manager for Advanced Ensemble Optimization

This module implements a unified preprocessing pipeline that eliminates
redundant processing across multiple models by sharing intermediate results
and caching preprocessing outputs.
"""

import logging
import time
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from PIL import Image
import cv2
from functools import lru_cache

from .preprocessing import UnifiedPreprocessor, PreprocessingConfig


@dataclass
class PreprocessingResult:
    """Result of unified preprocessing with shared intermediate data."""
    original_image: Image.Image
    processed_images: Dict[str, np.ndarray]  # Model name -> processed array
    intermediate_data: Dict[str, Any]  # Shared intermediate results
    preprocessing_time: float
    cache_key: str
    metadata: Dict[str, Any]


@dataclass
class ModelPreprocessingConfig:
    """Configuration for model-specific preprocessing requirements."""
    model_name: str
    input_size: Tuple[int, int]
    normalization: Optional[Tuple[float, float]] = None  # mean, std
    color_space: str = "RGB"
    requires_face_detection: bool = True
    requires_noise_reduction: bool = False
    requires_histogram_equalization: bool = False
    requires_sharpening: bool = False


class UnifiedPreprocessingManager:
    """
    Manages unified preprocessing for multiple models with shared intermediate results.
    
    Eliminates redundant preprocessing by:
    1. Performing common preprocessing steps once
    2. Sharing intermediate results across models
    3. Caching preprocessing results for reuse
    4. Optimizing memory usage through shared data structures
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self.logger = logging.getLogger(f"{__name__}.UnifiedPreprocessingManager")
        
        # Initialize unified preprocessor
        self.preprocessor = UnifiedPreprocessor(self.config)
        
        # Model-specific configurations
        self.model_configs = {
            "EfficientNet": ModelPreprocessingConfig(
                model_name="EfficientNet",
                input_size=(224, 224),
                normalization=(0.485, 0.229),  # ImageNet normalization
                color_space="RGB",
                requires_face_detection=True,
                requires_noise_reduction=False,
                requires_histogram_equalization=False,
                requires_sharpening=False
            ),
            "Xception": ModelPreprocessingConfig(
                model_name="Xception",
                input_size=(299, 299),
                normalization=(0.5, 0.5),  # Xception normalization
                color_space="RGB",
                requires_face_detection=True,
                requires_noise_reduction=True,
                requires_histogram_equalization=False,
                requires_sharpening=True
            ),
            "F3Net": ModelPreprocessingConfig(
                model_name="F3Net",
                input_size=(224, 224),
                normalization=(0.5, 0.5),  # F3Net normalization
                color_space="RGB",
                requires_face_detection=True,
                requires_noise_reduction=True,
                requires_histogram_equalization=True,
                requires_sharpening=False
            ),
            "MesoNet": ModelPreprocessingConfig(
                model_name="MesoNet",
                input_size=(256, 256),
                normalization=None,  # No normalization
                color_space="RGB",
                requires_face_detection=True,
                requires_noise_reduction=False,
                requires_histogram_equalization=False,
                requires_sharpening=False
            )
        }
        
        # Preprocessing cache
        self.cache: Dict[str, PreprocessingResult] = {}
        self.cache_max_size = 100  # Maximum cached results
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Performance tracking
        self.total_preprocessing_time = 0.0
        self.total_images_processed = 0
        self.shared_processing_savings = 0.0
    
    def _generate_cache_key(self, image: Image.Image, model_names: List[str]) -> str:
        """Generate cache key for image and model combination."""
        # Create hash based on image content and model requirements
        image_bytes = image.tobytes()
        model_hash = hashlib.md5(','.join(sorted(model_names)).encode()).hexdigest()
        image_hash = hashlib.md5(image_bytes).hexdigest()
        return f"{image_hash}_{model_hash}"
    
    def _get_shared_requirements(self, model_names: List[str]) -> Dict[str, bool]:
        """Determine shared preprocessing requirements across models."""
        requirements = {
            "face_detection": False,
            "noise_reduction": False,
            "histogram_equalization": False,
            "sharpening": False,
            "resize": True,  # Always needed
            "normalize": False
        }
        
        for model_name in model_names:
            if model_name in self.model_configs:
                config = self.model_configs[model_name]
                requirements["face_detection"] |= config.requires_face_detection
                requirements["noise_reduction"] |= config.requires_noise_reduction
                requirements["histogram_equalization"] |= config.requires_histogram_equalization
                requirements["sharpening"] |= config.requires_sharpening
                requirements["normalize"] |= (config.normalization is not None)
        
        return requirements
    
    def _perform_shared_preprocessing(self, image: Image.Image, requirements: Dict[str, bool]) -> Dict[str, Any]:
        """Perform shared preprocessing steps that all models need."""
        intermediate_data = {}
        
        # Convert to numpy array for processing
        img_array = np.array(image)
        
        # Face detection (if any model needs it)
        if requirements["face_detection"]:
            try:
                face_data = self.preprocessor._detect_faces(img_array)
                intermediate_data["face_data"] = face_data
                intermediate_data["face_detected"] = len(face_data) > 0
            except Exception as e:
                self.logger.warning(f"Face detection failed: {str(e)}")
                intermediate_data["face_data"] = []
                intermediate_data["face_detected"] = False
        
        # Noise reduction (if any model needs it)
        if requirements["noise_reduction"]:
            try:
                denoised = self.preprocessor._reduce_noise(img_array)
                intermediate_data["denoised_image"] = denoised
            except Exception as e:
                self.logger.warning(f"Noise reduction failed: {str(e)}")
                intermediate_data["denoised_image"] = img_array
        
        # Histogram equalization (if any model needs it)
        if requirements["histogram_equalization"]:
            try:
                equalized = self.preprocessor._equalize_histogram(img_array)
                intermediate_data["equalized_image"] = equalized
            except Exception as e:
                self.logger.warning(f"Histogram equalization failed: {str(e)}")
                intermediate_data["equalized_image"] = img_array
        
        # Sharpening (if any model needs it)
        if requirements["sharpening"]:
            try:
                sharpened = self.preprocessor._sharpen_image(img_array)
                intermediate_data["sharpened_image"] = sharpened
            except Exception as e:
                self.logger.warning(f"Sharpening failed: {str(e)}")
                intermediate_data["sharpened_image"] = img_array
        
        return intermediate_data
    
    def _process_for_model(self, image: Image.Image, model_name: str, 
                          intermediate_data: Dict[str, Any]) -> np.ndarray:
        """Process image for specific model using shared intermediate data."""
        if model_name not in self.model_configs:
            raise ValueError(f"Unknown model: {model_name}")
        
        config = self.model_configs[model_name]
        
        # Start with original image or appropriate intermediate result
        if config.requires_noise_reduction and "denoised_image" in intermediate_data:
            base_image = intermediate_data["denoised_image"]
        elif config.requires_histogram_equalization and "equalized_image" in intermediate_data:
            base_image = intermediate_data["equalized_image"]
        elif config.requires_sharpening and "sharpened_image" in intermediate_data:
            base_image = intermediate_data["sharpened_image"]
        else:
            base_image = np.array(image)
        
        # Resize to model-specific input size
        resized = cv2.resize(base_image, config.input_size, interpolation=cv2.INTER_LANCZOS4)
        
        # Convert color space if needed
        if config.color_space == "RGB" and len(resized.shape) == 3 and resized.shape[2] == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize if required
        if config.normalization is not None:
            mean, std = config.normalization
            resized = resized.astype(np.float32) / 255.0
            resized = (resized - mean) / std
        
        # Ensure correct data type and shape
        if config.normalization is None:
            resized = resized.astype(np.uint8)
        
        return resized
    
    def preprocess_for_models(self, image: Image.Image, model_names: List[str], 
                            use_cache: bool = True) -> PreprocessingResult:
        """
        Preprocess image for multiple models with shared intermediate results.
        
        Args:
            image: PIL Image to preprocess
            model_names: List of model names to preprocess for
            use_cache: Whether to use cached results if available
            
        Returns:
            PreprocessingResult with processed images and shared data
        """
        start_time = time.time()
        
        # Generate cache key
        cache_key = self._generate_cache_key(image, model_names)
        
        # Check cache first
        if use_cache and cache_key in self.cache:
            self.cache_hits += 1
            self.logger.debug(f"Cache hit for models: {model_names}")
            return self.cache[cache_key]
        
        self.cache_misses += 1
        
        try:
            # Get shared requirements
            requirements = self._get_shared_requirements(model_names)
            
            # Perform shared preprocessing
            intermediate_data = self._perform_shared_preprocessing(image, requirements)
            
            # Process for each model
            processed_images = {}
            for model_name in model_names:
                try:
                    processed_img = self._process_for_model(image, model_name, intermediate_data)
                    processed_images[model_name] = processed_img
                except Exception as e:
                    self.logger.error(f"Failed to process for {model_name}: {str(e)}")
                    # Use fallback processing
                    processed_images[model_name] = np.array(image.resize(
                        self.model_configs[model_name].input_size
                    ))
            
            # Calculate processing time
            preprocessing_time = time.time() - start_time
            
            # Create result
            result = PreprocessingResult(
                original_image=image,
                processed_images=processed_images,
                intermediate_data=intermediate_data,
                preprocessing_time=preprocessing_time,
                cache_key=cache_key,
                metadata={
                    "model_names": model_names,
                    "requirements": requirements,
                    "cache_used": False,
                    "shared_processing": True
                }
            )
            
            # Cache result if enabled
            if use_cache:
                self._cache_result(cache_key, result)
            
            # Update performance metrics
            self.total_preprocessing_time += preprocessing_time
            self.total_images_processed += 1
            
            # Calculate savings from shared processing
            individual_time = sum(
                self._estimate_individual_processing_time(model_name) 
                for model_name in model_names
            )
            self.shared_processing_savings += max(0, individual_time - preprocessing_time)
            
            self.logger.info(f"Preprocessed for {len(model_names)} models in {preprocessing_time:.3f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Unified preprocessing failed: {str(e)}")
            # Return fallback result
            return PreprocessingResult(
                original_image=image,
                processed_images={},
                intermediate_data={},
                preprocessing_time=time.time() - start_time,
                cache_key=cache_key,
                metadata={"error": str(e), "fallback": True}
            )
    
    def _estimate_individual_processing_time(self, model_name: str) -> float:
        """Estimate time for individual model preprocessing."""
        # Base processing time estimates (in seconds)
        base_times = {
            "EfficientNet": 0.020,
            "Xception": 0.030,
            "F3Net": 0.025,
            "MesoNet": 0.015
        }
        return base_times.get(model_name, 0.025)
    
    def _cache_result(self, cache_key: str, result: PreprocessingResult):
        """Cache preprocessing result with size management."""
        # Remove oldest entries if cache is full
        if len(self.cache) >= self.cache_max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[cache_key] = result
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0.0
        
        return {
            "cache_size": len(self.cache),
            "max_cache_size": self.cache_max_size,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "total_requests": total_requests
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get preprocessing performance statistics."""
        avg_time = self.total_preprocessing_time / self.total_images_processed if self.total_images_processed > 0 else 0.0
        
        return {
            "total_preprocessing_time": self.total_preprocessing_time,
            "total_images_processed": self.total_images_processed,
            "average_preprocessing_time": avg_time,
            "shared_processing_savings": self.shared_processing_savings,
            "cache_stats": self.get_cache_stats()
        }
    
    def clear_cache(self):
        """Clear preprocessing cache."""
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        self.logger.info("Preprocessing cache cleared")
    
    def add_model_config(self, model_name: str, config: ModelPreprocessingConfig):
        """Add configuration for a new model."""
        self.model_configs[model_name] = config
        self.logger.info(f"Added preprocessing config for {model_name}")
    
    def update_model_config(self, model_name: str, **kwargs):
        """Update configuration for existing model."""
        if model_name in self.model_configs:
            for key, value in kwargs.items():
                if hasattr(self.model_configs[model_name], key):
                    setattr(self.model_configs[model_name], key, value)
            self.logger.info(f"Updated preprocessing config for {model_name}")
        else:
            self.logger.warning(f"Model {model_name} not found for config update")
