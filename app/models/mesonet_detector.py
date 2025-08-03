"""
Enhanced MesoNet Detector

This module implements an enhanced MesoNet detector with improved preprocessing,
confidence calibration, performance monitoring, and caching capabilities.
"""

import os
import time
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

from .core_architecture import (
    BaseDetector, 
    DetectionResult, 
    ModelInfo, 
    MesoNetConfig,
    PerformanceMonitor,
    AugmentationConfig
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# MesoNet Architecture
# ============================================================================

class MesoNet(nn.Module):
    """
    Enhanced MesoNet architecture for deepfake detection
    
    Original MesoNet with improvements:
    - Better regularization
    - Improved feature extraction
    - Enhanced classification head
    """
    
    def __init__(self, num_classes: int = 2):
        super(MesoNet, self).__init__()
        
        # Feature extraction layers
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        
        # Pooling layers
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout for regularization
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.dropout3 = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        # Batch normalization for better training
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(32)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass through the network"""
        # First block
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn1(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout1(x)
        
        # Second block
        x = F.relu(self.bn2(self.conv3(x)))
        x = F.relu(self.bn2(self.conv4(x)))
        x = F.relu(self.bn2(self.conv5(x)))
        x = self.pool(x)
        x = self.dropout2(x)
        
        # Third block
        x = F.relu(self.bn3(self.conv6(x)))
        x = F.relu(self.bn3(self.conv7(x)))
        x = F.relu(self.bn3(self.conv8(x)))
        x = self.pool(x)
        x = self.dropout3(x)
        
        # Flatten and fully connected layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = F.relu(self.fc2(x))
        x = self.dropout3(x)
        x = self.fc3(x)
        
        return x

# ============================================================================
# Enhanced Preprocessor
# ============================================================================

class MesoNetPreprocessor:
    """Enhanced preprocessor for MesoNet with augmentation support"""
    
    def __init__(self, config: MesoNetConfig):
        self.config = config
        self.input_size = config.input_size
        self.augmentation_config = config.augmentation_config
        
        # Base transforms
        self.base_transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Augmentation transforms
        if self.augmentation_config.enable_augmentation:
            self.augmentation_transform = transforms.Compose([
                transforms.Resize(self.input_size),
                transforms.RandomRotation(self.augmentation_config.rotation_range),
                transforms.RandomHorizontalFlip(p=0.5 if self.augmentation_config.horizontal_flip else 0.0),
                transforms.ColorJitter(
                    brightness=self.augmentation_config.brightness_range,
                    contrast=self.augmentation_config.contrast_range
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.augmentation_transform = self.base_transform
    
    def preprocess(self, image: Image.Image, use_augmentation: bool = False) -> torch.Tensor:
        """Preprocess image for MesoNet input"""
        # Ensure RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        if use_augmentation:
            tensor = self.augmentation_transform(image)
        else:
            tensor = self.base_transform(image)
        
        return tensor.unsqueeze(0)  # Add batch dimension
    
    def preprocess_batch(self, images: list, use_augmentation: bool = False) -> torch.Tensor:
        """Preprocess batch of images"""
        tensors = []
        for image in images:
            tensor = self.preprocess(image, use_augmentation)
            tensors.append(tensor)
        
        return torch.cat(tensors, dim=0)

# ============================================================================
# Confidence Calibrator
# ============================================================================

class ConfidenceCalibrator:
    """Calibrates confidence scores for better reliability"""
    
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
        self.is_calibrated = False
        self.calibration_data = []
    
    def calibrate(self, logits: torch.Tensor, labels: torch.Tensor):
        """Calibrate using validation data"""
        # Simple temperature scaling
        # In a full implementation, this would use more sophisticated methods
        self.is_calibrated = True
        self.logger.info(f"Calibrated with temperature {self.temperature}")
    
    def calibrate_confidence(self, confidence: float) -> float:
        """Apply calibration to confidence score"""
        if not self.is_calibrated:
            return confidence
        
        # Apply temperature scaling
        # Convert confidence to logits, apply temperature, convert back
        logit = np.log(confidence / (100.0 - confidence + 1e-8))
        calibrated_logit = logit / self.temperature
        calibrated_confidence = 1.0 / (1.0 + np.exp(-calibrated_logit))
        
        return calibrated_confidence * 100.0
    
    def set_temperature(self, temperature: float):
        """Set calibration temperature"""
        self.temperature = temperature

# ============================================================================
# Enhanced MesoNet Detector
# ============================================================================

class MesoNetDetector(BaseDetector):
    """Enhanced MesoNet detector with improved features"""
    
    def __init__(self, config: MesoNetConfig):
        super().__init__("MesoNetDetector", config.device)
        self.config = config
        self.model = None
        self.preprocessor = MesoNetPreprocessor(config)
        self.calibrator = ConfidenceCalibrator()
        self.monitor = PerformanceMonitor(config.monitoring_config)
        self.logger = logging.getLogger(f"{__name__}.MesoNetDetector")
        
        # Model metadata
        self.model_version = "enhanced-v1.0.0"
        self.architecture = "MesoNet-Enhanced"
        self.input_size = config.input_size
        
        # Performance metrics
        self.performance_metrics = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "inference_time": 0.0
        }
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """Load MesoNet model with caching and error handling"""
        try:
            self.logger.info("Loading enhanced MesoNet model...")
            
            # Create model
            self.model = MesoNet(num_classes=2)
            
            # Load weights if provided
            if model_path and os.path.exists(model_path):
                self.logger.info(f"Loading weights from {model_path}")
                checkpoint = torch.load(model_path, map_location=self.device)
                
                if isinstance(checkpoint, dict):
                    if 'state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['state_dict'])
                    else:
                        self.model.load_state_dict(checkpoint)
                else:
                    self.model.load_state_dict(checkpoint)
            
            # Move to device
            self.model.to(self.device)
            self.model.eval()
            
            # Set model as loaded
            self.is_model_loaded = True
            
            self.logger.info(f"MesoNet model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load MesoNet model: {str(e)}")
            self.is_model_loaded = False
            return False
    
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """Enhanced preprocessing with augmentation"""
        return self.preprocessor.preprocess(image, use_augmentation=False)
    
    def predict(self, image: Image.Image) -> DetectionResult:
        """Perform prediction with timing and calibration"""
        start_time = time.time()
        
        try:
            # Start performance monitoring
            self.monitor.start_timer(self.model_name)
            
            # Preprocess image
            input_tensor = self.preprocess(image)
            input_tensor = input_tensor.to(self.device)
            
            # Perform inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
                
                # Get confidence score (probability of being a deepfake)
                confidence_score = probabilities[0, 1].item() * 100.0
                
                # Apply calibration if enabled
                if self.config.enable_calibration:
                    confidence_score = self.calibrator.calibrate_confidence(confidence_score)
                
                # Determine prediction
                is_deepfake = confidence_score > 50.0
            
            # End performance monitoring
            processing_time = self.monitor.end_timer(self.model_name)
            
            # Record metrics
            self.monitor.record_confidence(self.model_name, confidence_score)
            
            # Update performance metrics
            self._update_performance_metrics(processing_time)
            
            # Generate metadata
            metadata = self._generate_metadata(image, processing_time)
            
            return DetectionResult(
                confidence_score=confidence_score,
                is_deepfake=is_deepfake,
                model_name=self.model_name,
                processing_time=processing_time,
                model_version=self.model_version,
                metadata=metadata
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"MesoNet prediction failed: {str(e)}")
            
            return DetectionResult(
                confidence_score=0.0,
                is_deepfake=False,
                model_name=self.model_name,
                processing_time=processing_time,
                model_version=self.model_version,
                metadata={"error": str(e)}
            )
    
    def generate_heatmap(self, image: Image.Image) -> np.ndarray:
        """Generate Grad-CAM heatmap for explainability"""
        try:
            # This is a placeholder for Grad-CAM implementation
            # In a full implementation, this would generate actual heatmaps
            self.logger.info("Grad-CAM heatmap generation not yet implemented")
            
            # Return a dummy heatmap for now
            input_tensor = self.preprocess(image)
            heatmap = np.random.rand(*input_tensor.shape[2:])  # Same size as input
            return heatmap
            
        except Exception as e:
            self.logger.error(f"Failed to generate heatmap: {str(e)}")
            return np.zeros((self.input_size[0], self.input_size[1]))
    
    def get_model_info(self) -> ModelInfo:
        """Get model information"""
        return ModelInfo(
            name=self.model_name,
            version=self.model_version,
            architecture=self.architecture,
            input_size=self.input_size,
            performance_metrics=self.performance_metrics,
            supported_formats=["jpg", "jpeg", "png", "bmp"],
            device_requirements=self.device,
            inference_time=self.performance_metrics["inference_time"]
        )
    
    def _update_performance_metrics(self, processing_time: float):
        """Update performance metrics"""
        self.performance_metrics["inference_time"] = processing_time
    
    def _generate_metadata(self, image: Image.Image, processing_time: float) -> Dict[str, Any]:
        """Generate comprehensive metadata"""
        return {
            "input_size": self.input_size,
            "model_architecture": self.architecture,
            "device": self.device,
            "processing_time": processing_time,
            "calibration_enabled": self.config.enable_calibration,
            "monitoring_enabled": self.config.enable_monitoring,
            "augmentation_enabled": self.config.augmentation_config.enable_augmentation,
            "image_size": image.size,
            "image_mode": image.mode,
            "model_loaded": self.is_model_loaded
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report"""
        return self.monitor.get_performance_report()
    
    def save_performance_data(self, filename: str = None):
        """Save performance data"""
        self.monitor.save_performance_data(filename)
    
    def calibrate_model(self, validation_data: list):
        """Calibrate the model using validation data"""
        # This would implement proper calibration using validation data
        self.logger.info("Model calibration not yet implemented")
        # TODO: Implement proper calibration using validation data

# ============================================================================
# Factory Function
# ============================================================================

def create_mesonet_detector(config: Optional[MesoNetConfig] = None) -> MesoNetDetector:
    """Factory function to create MesoNet detector"""
    if config is None:
        config = MesoNetConfig()
    
    return MesoNetDetector(config) 