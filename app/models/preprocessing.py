"""
Unified preprocessing pipeline for multi-model deepfake detection framework.
This module provides standardized image preprocessing operations for all models.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter
from torchvision import transforms


class InterpolationMethod(str, Enum):
    """Available interpolation methods for image resizing."""
    NEAREST = "nearest"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    LANCZOS = "lanczos"


class AugmentationType(str, Enum):
    """Available augmentation types."""
    NONE = "none"
    BASIC = "basic"
    ADVANCED = "advanced"
    CUSTOM = "custom"


@dataclass
class PreprocessingConfig:
    """Configuration for image preprocessing."""
    input_size: Tuple[int, int] = (224, 224)
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    normalize: bool = True
    augment: bool = False
    augmentation_type: AugmentationType = AugmentationType.NONE
    preserve_aspect_ratio: bool = True
    interpolation: InterpolationMethod = InterpolationMethod.BILINEAR
    color_mode: str = "RGB"
    enable_face_detection: bool = False
    face_crop_margin: float = 0.1
    enable_noise_reduction: bool = False
    noise_reduction_strength: float = 0.1
    enable_histogram_equalization: bool = False
    enable_sharpening: bool = False
    sharpening_strength: float = 1.5


class UnifiedPreprocessor:
    """
    Unified preprocessing pipeline for all deepfake detection models.
    
    Provides standardized preprocessing operations including resizing, normalization,
    augmentation, and various image enhancement techniques.
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Initialize the preprocessor.
        
        Args:
            config: Preprocessing configuration
        """
        self.config = config or PreprocessingConfig()
        self.logger = logging.getLogger(f"{__name__}.UnifiedPreprocessor")
        
        # Initialize face detection if enabled
        self.face_cascade = None
        if self.config.enable_face_detection:
            self._initialize_face_detection()
        
        # Initialize transforms
        self._initialize_transforms()
        
        self.logger.info(f"Unified preprocessor initialized with config: {self.config}")
    
    def _initialize_face_detection(self):
        """Initialize face detection cascade classifier."""
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if self.face_cascade.empty():
                self.logger.warning("Failed to load face detection cascade, disabling face detection")
                self.config.enable_face_detection = False
            else:
                self.logger.info("Face detection initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to initialize face detection: {str(e)}")
            self.config.enable_face_detection = False
    
    def _initialize_transforms(self):
        """Initialize torchvision transforms based on configuration."""
        transform_list = []
        
        # Resize transform
        if self.config.preserve_aspect_ratio:
            transform_list.append(transforms.Resize(
                self.config.input_size,
                interpolation=getattr(transforms.InterpolationMode, self.config.interpolation.upper())
            ))
            transform_list.append(transforms.CenterCrop(self.config.input_size))
        else:
            transform_list.append(transforms.Resize(
                self.config.input_size,
                interpolation=getattr(transforms.InterpolationMode, self.config.interpolation.upper())
            ))
        
        # Color mode conversion
        if self.config.color_mode == "RGB":
            transform_list.append(transforms.Lambda(lambda img: img.convert('RGB')))
        elif self.config.color_mode == "GRAY":
            transform_list.append(transforms.Grayscale(num_output_channels=3))
        
        # Convert to tensor
        transform_list.append(transforms.ToTensor())
        
        # Normalization
        if self.config.normalize:
            transform_list.append(transforms.Normalize(self.config.mean, self.config.std))
        
        self.transform = transforms.Compose(transform_list)
        
        # Augmentation transforms
        if self.config.augment:
            self._initialize_augmentation_transforms()
    
    def _initialize_augmentation_transforms(self):
        """Initialize augmentation transforms based on type."""
        if self.config.augmentation_type == AugmentationType.BASIC:
            self.augment_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomResizedCrop(self.config.input_size, scale=(0.8, 1.0))
            ])
        elif self.config.augmentation_type == AugmentationType.ADVANCED:
            self.augment_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
                transforms.RandomResizedCrop(self.config.input_size, scale=(0.7, 1.0)),
                transforms.RandomGrayscale(p=0.1),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ])
        else:
            self.augment_transform = None
    
    def preprocess(self, image: Union[Image.Image, str, np.ndarray]) -> torch.Tensor:
        """
        Apply complete preprocessing pipeline.
        
        Args:
            image: Input image (PIL Image, file path, or numpy array)
            
        Returns:
            Preprocessed tensor ready for model input
        """
        try:
            # Convert to PIL Image if needed
            if isinstance(image, str):
                image = Image.open(image)
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            # Validate image
            if not self._validate_image(image):
                raise ValueError("Invalid image provided for preprocessing")
            
            # Apply face detection if enabled
            if self.config.enable_face_detection:
                image = self._detect_and_crop_face(image)
            
            # Apply image enhancements
            image = self._apply_enhancements(image)
            
            # Apply augmentation if enabled
            if self.config.augment and self.augment_transform:
                image = self.augment_transform(image)
            
            # Apply standard transforms
            tensor = self.transform(image)
            
            # Add batch dimension if needed
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
            
            return tensor
            
        except Exception as e:
            self.logger.error(f"Preprocessing failed: {str(e)}")
            raise
    
    def _validate_image(self, image: Image.Image) -> bool:
        """Validate input image."""
        if image is None:
            self.logger.error("Image is None")
            return False
        
        if image.size[0] == 0 or image.size[1] == 0:
            self.logger.error("Image has zero dimensions")
            return False
        
        return True
    
    def _detect_and_crop_face(self, image: Image.Image) -> Image.Image:
        """Detect and crop face from image."""
        try:
            # Convert PIL to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            if len(faces) > 0:
                # Use the largest face
                largest_face = max(faces, key=lambda x: x[2] * x[3])
                x, y, w, h = largest_face
                
                # Add margin
                margin_x = int(w * self.config.face_crop_margin)
                margin_y = int(h * self.config.face_crop_margin)
                
                # Calculate crop coordinates
                x1 = max(0, x - margin_x)
                y1 = max(0, y - margin_y)
                x2 = min(cv_image.shape[1], x + w + margin_x)
                y2 = min(cv_image.shape[0], y + h + margin_y)
                
                # Crop and convert back to PIL
                cropped = cv_image[y1:y2, x1:x2]
                cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                return Image.fromarray(cropped_rgb)
            
            # No face detected, return original image
            return image
            
        except Exception as e:
            self.logger.warning(f"Face detection failed: {str(e)}, using original image")
            return image
    
    def _apply_enhancements(self, image: Image.Image) -> Image.Image:
        """Apply various image enhancements."""
        try:
            # Noise reduction
            if self.config.enable_noise_reduction:
                image = self._reduce_noise(image)
            
            # Histogram equalization
            if self.config.enable_histogram_equalization:
                image = self._equalize_histogram(image)
            
            # Sharpening
            if self.config.enable_sharpening:
                image = self._sharpen_image(image)
            
            return image
            
        except Exception as e:
            self.logger.warning(f"Image enhancement failed: {str(e)}")
            return image
    
    def _reduce_noise(self, image: Image.Image) -> Image.Image:
        """Apply noise reduction to image."""
        # Convert to numpy array
        img_array = np.array(image)
        
        # Apply bilateral filter for noise reduction
        denoised = cv2.bilateralFilter(
            img_array,
            d=9,
            sigmaColor=self.config.noise_reduction_strength * 75,
            sigmaSpace=self.config.noise_reduction_strength * 75
        )
        
        return Image.fromarray(denoised)
    
    def _equalize_histogram(self, image: Image.Image) -> Image.Image:
        """Apply histogram equalization to image."""
        # Convert to numpy array
        img_array = np.array(image)
        
        # Convert to LAB color space
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Convert back to RGB
        equalized = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return Image.fromarray(equalized)
    
    def _sharpen_image(self, image: Image.Image) -> Image.Image:
        """Apply sharpening to image."""
        # Create sharpening filter
        sharpener = ImageEnhance.Sharpness(image)
        sharpened = sharpener.enhance(self.config.sharpening_strength)
        
        return sharpened
    
    def preprocess_batch(self, images: List[Union[Image.Image, str, np.ndarray]]) -> torch.Tensor:
        """
        Preprocess a batch of images.
        
        Args:
            images: List of input images
            
        Returns:
            Batch tensor ready for model input
        """
        tensors = []
        
        for image in images:
            tensor = self.preprocess(image)
            tensors.append(tensor)
        
        return torch.cat(tensors, dim=0)
    
    def get_preprocessing_info(self) -> Dict[str, Any]:
        """Get information about the preprocessing configuration."""
        return {
            "input_size": self.config.input_size,
            "mean": self.config.mean,
            "std": self.config.std,
            "normalize": self.config.normalize,
            "augment": self.config.augment,
            "augmentation_type": self.config.augmentation_type.value,
            "preserve_aspect_ratio": self.config.preserve_aspect_ratio,
            "interpolation": self.config.interpolation.value,
            "color_mode": self.config.color_mode,
            "enable_face_detection": self.config.enable_face_detection,
            "enable_noise_reduction": self.config.enable_noise_reduction,
            "enable_histogram_equalization": self.config.enable_histogram_equalization,
            "enable_sharpening": self.config.enable_sharpening
        }
    
    def update_config(self, new_config: PreprocessingConfig) -> bool:
        """Update preprocessing configuration."""
        try:
            self.config = new_config
            self._initialize_transforms()
            
            if self.config.enable_face_detection and self.face_cascade is None:
                self._initialize_face_detection()
            
            self.logger.info("Preprocessing configuration updated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update preprocessing config: {str(e)}")
            return False


class PreprocessingPipeline:
    """
    High-level preprocessing pipeline that manages multiple preprocessors.
    
    Provides a unified interface for preprocessing images for different models
    with different requirements.
    """
    
    def __init__(self):
        """Initialize the preprocessing pipeline."""
        self.preprocessors: Dict[str, UnifiedPreprocessor] = {}
        self.logger = logging.getLogger(f"{__name__}.PreprocessingPipeline")
    
    def add_preprocessor(self, name: str, config: PreprocessingConfig) -> bool:
        """
        Add a preprocessor with specific configuration.
        
        Args:
            name: Unique name for the preprocessor
            config: Preprocessing configuration
            
        Returns:
            True if added successfully, False otherwise
        """
        try:
            preprocessor = UnifiedPreprocessor(config)
            self.preprocessors[name] = preprocessor
            self.logger.info(f"Added preprocessor '{name}' with config: {config}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add preprocessor '{name}': {str(e)}")
            return False
    
    def get_preprocessor(self, name: str) -> Optional[UnifiedPreprocessor]:
        """Get a preprocessor by name."""
        return self.preprocessors.get(name)
    
    def preprocess_for_model(self, image: Union[Image.Image, str, np.ndarray], model_name: str) -> torch.Tensor:
        """
        Preprocess image for a specific model.
        
        Args:
            image: Input image
            model_name: Name of the model (used to select appropriate preprocessor)
            
        Returns:
            Preprocessed tensor
        """
        preprocessor = self.preprocessors.get(model_name)
        if preprocessor is None:
            # Use default preprocessor
            preprocessor = self.preprocessors.get("default")
            if preprocessor is None:
                raise ValueError(f"No preprocessor found for model '{model_name}' and no default preprocessor")
        
        return preprocessor.preprocess(image)
    
    def list_preprocessors(self) -> List[str]:
        """Get list of available preprocessors."""
        return list(self.preprocessors.keys())
    
    def remove_preprocessor(self, name: str) -> bool:
        """Remove a preprocessor."""
        if name in self.preprocessors:
            del self.preprocessors[name]
            self.logger.info(f"Removed preprocessor '{name}'")
            return True
        return False 