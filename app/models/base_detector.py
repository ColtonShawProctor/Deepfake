"""
Base detector interface and abstract classes for multi-model deepfake detection framework.
This module provides the foundational classes that all detector models will inherit from.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pydantic import BaseModel


class DeviceType(str, Enum):
    """Supported device types for model inference."""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon
    AUTO = "auto"


class ModelStatus(str, Enum):
    """Model loading and operational status."""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    ERROR = "error"
    INFERENCE = "inference"


@dataclass
class ModelInfo:
    """Information about a detector model."""
    name: str
    version: str
    architecture: str
    input_size: Tuple[int, int]
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    num_classes: int = 1
    model_size_mb: Optional[float] = None
    parameters_count: Optional[int] = None
    supported_formats: List[str] = field(default_factory=lambda: ["jpg", "jpeg", "png", "bmp"])
    device: str = "cpu"
    status: ModelStatus = ModelStatus.UNLOADED
    last_loaded: Optional[float] = None
    inference_count: int = 0
    total_inference_time: float = 0.0
    average_inference_time: float = 0.0


@dataclass
class DetectionResult:
    """Result of a deepfake detection analysis."""
    is_deepfake: bool
    confidence: float
    model_name: str
    inference_time: float
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    uncertainty: Optional[float] = None
    attention_maps: Optional[np.ndarray] = None
    preprocessing_info: Optional[Dict[str, Any]] = None


@dataclass
class PreprocessingConfig:
    """Configuration for image preprocessing."""
    input_size: Tuple[int, int] = (224, 224)
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    normalize: bool = True
    augment: bool = False
    preserve_aspect_ratio: bool = True
    interpolation: str = "bilinear"
    color_mode: str = "RGB"


class BaseDetector(ABC):
    """
    Abstract base class for all deepfake detectors.
    
    This class defines the interface that all detector models must implement.
    It provides common functionality for model loading, preprocessing, and inference.
    """
    
    def __init__(
        self,
        model_name: str,
        device: Union[str, DeviceType] = DeviceType.AUTO,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the base detector.
        
        Args:
            model_name: Name of the detector model
            device: Device to run inference on (cpu, cuda, mps, auto)
            config: Optional configuration dictionary
        """
        self.model_name = model_name
        self.device = self._get_device(device)
        self.config = config or {}
        
        # Model state
        self.model: Optional[nn.Module] = None
        self.transform = None
        self.is_model_loaded = False
        
        # Performance tracking
        self.inference_count = 0
        self.total_inference_time = 0.0
        self.average_inference_time = 0.0
        
        # Logging
        self.logger = logging.getLogger(f"{__name__}.{model_name}")
        
        # Model information
        self.model_info = ModelInfo(
            name=model_name,
            version="1.0.0",
            architecture="base",
            input_size=(224, 224),
            device=self.device,
            status=ModelStatus.UNLOADED
        )
    
    def _get_device(self, device: Union[str, DeviceType]) -> str:
        """Determine the appropriate device for model inference."""
        if device == DeviceType.AUTO:
            if torch.cuda.is_available():
                return DeviceType.CUDA
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return DeviceType.MPS
            else:
                return DeviceType.CPU
        return str(device)
    
    @abstractmethod
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Load the model from file or initialize it.
        
        Args:
            model_path: Path to the model weights file
            
        Returns:
            True if model loaded successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess an image for model inference.
        
        Args:
            image: PIL Image to preprocess
            
        Returns:
            Preprocessed tensor ready for model input
        """
        pass
    
    @abstractmethod
    def predict(self, image: Image.Image) -> DetectionResult:
        """
        Perform deepfake detection on an image.
        
        Args:
            image: PIL Image to analyze
            
        Returns:
            DetectionResult containing the analysis results
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> ModelInfo:
        """
        Get information about the model.
        
        Returns:
            ModelInfo object containing model details
        """
        pass
    
    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready for inference."""
        return self.is_model_loaded and self.model is not None
    
    def __call__(self, image: Image.Image) -> DetectionResult:
        """Convenience method to call predict directly."""
        return self.predict(image)
    
    def _update_performance_metrics(self, inference_time: float):
        """Update performance tracking metrics."""
        self.inference_count += 1
        self.total_inference_time += inference_time
        self.average_inference_time = self.total_inference_time / self.inference_count
        
        # Update model info
        self.model_info.inference_count = self.inference_count
        self.model_info.total_inference_time = self.total_inference_time
        self.model_info.average_inference_time = self.average_inference_time
    
    def _validate_image(self, image: Image.Image) -> bool:
        """Validate that the image is suitable for processing."""
        if image is None:
            self.logger.error("Image is None")
            return False
        
        if image.size[0] == 0 or image.size[1] == 0:
            self.logger.error("Image has zero dimensions")
            return False
        
        if image.mode not in ['RGB', 'L']:
            self.logger.warning(f"Converting image from {image.mode} to RGB")
            image = image.convert('RGB')
        
        return True
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the model."""
        return {
            "inference_count": self.inference_count,
            "total_inference_time": self.total_inference_time,
            "average_inference_time": self.average_inference_time,
            "throughput_fps": 1.0 / self.average_inference_time if self.average_inference_time > 0 else 0
        }
    
    def reset_performance_stats(self):
        """Reset performance tracking statistics."""
        self.inference_count = 0
        self.total_inference_time = 0.0
        self.average_inference_time = 0.0
        self.model_info.inference_count = 0
        self.model_info.total_inference_time = 0.0
        self.model_info.average_inference_time = 0.0


class BasePreprocessor:
    """
    Base class for image preprocessing operations.
    
    Provides common preprocessing functionality that can be extended
    by specific model preprocessors.
    """
    
    def __init__(self, config: PreprocessingConfig):
        """
        Initialize the preprocessor.
        
        Args:
            config: Preprocessing configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.BasePreprocessor")
    
    def resize_image(self, image: Image.Image) -> Image.Image:
        """Resize image to target size while preserving aspect ratio if requested."""
        if self.config.preserve_aspect_ratio:
            # Calculate new size maintaining aspect ratio
            target_ratio = self.config.input_size[0] / self.config.input_size[1]
            current_ratio = image.size[0] / image.size[1]
            
            if current_ratio > target_ratio:
                # Image is wider than target
                new_width = int(self.config.input_size[1] * current_ratio)
                new_height = self.config.input_size[1]
            else:
                # Image is taller than target
                new_width = self.config.input_size[0]
                new_height = int(self.config.input_size[0] / current_ratio)
            
            image = image.resize((new_width, new_height), getattr(Image, self.config.interpolation.upper()))
            
            # Center crop to target size
            left = (new_width - self.config.input_size[0]) // 2
            top = (new_height - self.config.input_size[1]) // 2
            right = left + self.config.input_size[0]
            bottom = top + self.config.input_size[1]
            
            return image.crop((left, top, right, bottom))
        else:
            return image.resize(self.config.input_size, getattr(Image, self.config.interpolation.upper()))
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image using mean and std values."""
        if not self.config.normalize:
            return image
        
        # Convert to float32 and normalize
        image = image.astype(np.float32) / 255.0
        
        # Apply normalization
        for i in range(3):
            image[:, :, i] = (image[:, :, i] - self.config.mean[i]) / self.config.std[i]
        
        return image
    
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """
        Apply complete preprocessing pipeline.
        
        Args:
            image: PIL Image to preprocess
            
        Returns:
            Preprocessed tensor
        """
        # Validate image
        if not self._validate_image(image):
            raise ValueError("Invalid image provided for preprocessing")
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize
        image = self.resize_image(image)
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Normalize
        image_array = self.normalize_image(image_array)
        
        # Convert to tensor and add batch dimension
        tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)
        
        return tensor
    
    def _validate_image(self, image: Image.Image) -> bool:
        """Validate input image."""
        if image is None:
            self.logger.error("Image is None")
            return False
        
        if image.size[0] == 0 or image.size[1] == 0:
            self.logger.error("Image has zero dimensions")
            return False
        
        return True 