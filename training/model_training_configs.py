#!/usr/bin/env python3
"""
Model-specific training configurations for deepfake detection ensemble

This module provides detailed training configurations for each model in the ensemble:
- ResNet Detector
- EfficientNet-B4 Detector  
- F3Net Detector
- Ensemble optimization

Based on best practices from DeepfakeBench and production requirements.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, ReduceLROnPlateau
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import numpy as np

@dataclass
class TrainingConfig:
    """Base configuration for model training"""
    model_name: str
    architecture: str
    pretrained: bool
    
    # Optimization
    optimizer: str
    learning_rate: float
    weight_decay: float
    scheduler: str
    warmup_epochs: int
    
    # Training
    batch_size: int
    epochs: int
    gradient_clipping: float
    early_stopping_patience: int
    
    # Loss function
    loss_function: str
    auxiliary_loss: Optional[str] = None
    loss_weights: Optional[List[float]] = None
    
    # Regularization
    dropout: float
    label_smoothing: float
    use_mixup: bool = False
    use_cutmix: bool = False
    
    # Data strategy
    dataset_mix: Dict[str, float] = None
    augmentation_strength: str = "medium"

class ResNetTrainingConfig:
    """Training configuration for ResNet Detector"""
    
    @staticmethod
    def get_config() -> TrainingConfig:
        return TrainingConfig(
            model_name="ResNetDetector",
            architecture="ResNet-50",
            pretrained=True,
            
            # Optimization - Conservative approach for stability
            optimizer="AdamW",
            learning_rate=1e-4,
            weight_decay=1e-4,
            scheduler="CosineAnnealingLR",
            warmup_epochs=5,
            
            # Training parameters
            batch_size=32,
            epochs=50,
            gradient_clipping=1.0,
            early_stopping_patience=10,
            
            # Loss function - BCEWithLogits + Focal for class imbalance
            loss_function="BCEWithLogitsLoss",
            auxiliary_loss="FocalLoss",
            loss_weights=[0.8, 0.2],
            
            # Regularization
            dropout=0.5,
            label_smoothing=0.1,
            use_mixup=True,
            use_cutmix=True,
            
            # Data strategy - Balanced across datasets
            dataset_mix={
                "FaceForensics++": 0.60,
                "DFDC": 0.25,
                "CelebDF": 0.15
            },
            augmentation_strength="medium"
        )
    
    @staticmethod
    def get_optimizer(model_parameters, config: TrainingConfig):
        """Get optimizer for ResNet training"""
        return optim.AdamW(
            model_parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    
    @staticmethod
    def get_scheduler(optimizer, config: TrainingConfig):
        """Get learning rate scheduler"""
        return CosineAnnealingLR(
            optimizer,
            T_max=config.epochs,
            eta_min=1e-6
        )
    
    @staticmethod
    def get_loss_function():
        """Get combined loss function"""
        class CombinedLoss(nn.Module):
            def __init__(self):
                super().__init__()
                self.bce_loss = nn.BCEWithLogitsLoss()
                self.focal_loss = FocalLoss(alpha=0.7, gamma=2.0)
            
            def forward(self, predictions, targets):
                bce = self.bce_loss(predictions, targets)
                focal = self.focal_loss(predictions, targets)
                return 0.8 * bce + 0.2 * focal
        
        return CombinedLoss()

class EfficientNetTrainingConfig:
    """Training configuration for EfficientNet-B4 Detector"""
    
    @staticmethod
    def get_config() -> TrainingConfig:
        return TrainingConfig(
            model_name="EfficientNetDetector",
            architecture="EfficientNet-B4",
            pretrained=True,
            
            # Optimization - EfficientNet specific parameters
            optimizer="RMSprop",
            learning_rate=0.016,  # Scaled for batch size
            weight_decay=1e-5,
            scheduler="ExponentialLR",
            warmup_epochs=3,
            
            # Training parameters - Smaller batch size due to memory
            batch_size=16,
            epochs=60,
            gradient_clipping=0.5,
            early_stopping_patience=15,
            
            # Loss function - BCEWithLogits + ArcFace for better feature learning
            loss_function="BCEWithLogitsLoss",
            auxiliary_loss="ArcFaceLoss",
            loss_weights=[0.7, 0.3],
            
            # Regularization - EfficientNet specific
            dropout=0.4,
            label_smoothing=0.05,
            use_mixup=True,
            use_cutmix=False,
            
            # Data strategy - Focus on diverse datasets
            dataset_mix={
                "FaceForensics++": 0.50,
                "DFDC": 0.30,
                "CelebDF": 0.20
            },
            augmentation_strength="strong"
        )
    
    @staticmethod
    def get_optimizer(model_parameters, config: TrainingConfig):
        """Get RMSprop optimizer for EfficientNet"""
        return optim.RMSprop(
            model_parameters,
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=config.weight_decay,
            eps=0.0316,
            alpha=0.9
        )
    
    @staticmethod
    def get_scheduler(optimizer, config: TrainingConfig):
        """Get exponential decay scheduler"""
        return ExponentialLR(
            optimizer,
            gamma=0.97
        )

class F3NetTrainingConfig:
    """Training configuration for F3Net Frequency-Domain Detector"""
    
    @staticmethod
    def get_config() -> TrainingConfig:
        return TrainingConfig(
            model_name="F3NetDetector",
            architecture="F3Net",
            pretrained=False,  # Custom architecture
            
            # Optimization - Adam for frequency domain
            optimizer="Adam",
            learning_rate=5e-4,
            weight_decay=1e-4,
            scheduler="ReduceLROnPlateau",
            warmup_epochs=0,
            
            # Training parameters - Longer training for frequency domain
            batch_size=24,
            epochs=80,
            gradient_clipping=0.5,
            early_stopping_patience=20,
            
            # Loss function - BCEWithLogits + Frequency domain loss
            loss_function="BCEWithLogitsLoss",
            auxiliary_loss="DCTFrequencyLoss",
            loss_weights=[1.0, 0.3],
            
            # Regularization - Frequency specific
            dropout=0.3,
            label_smoothing=0.0,  # No label smoothing for frequency domain
            use_mixup=False,
            use_cutmix=False,
            
            # Data strategy - Equal focus on all datasets
            dataset_mix={
                "FaceForensics++": 0.4,
                "DFDC": 0.3,
                "CelebDF": 0.3
            },
            augmentation_strength="light"  # Preserve frequency characteristics
        )
    
    @staticmethod
    def get_optimizer(model_parameters, config: TrainingConfig):
        """Get Adam optimizer for F3Net"""
        return optim.Adam(
            model_parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    
    @staticmethod
    def get_scheduler(optimizer, config: TrainingConfig):
        """Get ReduceLROnPlateau scheduler"""
        return ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True,
            min_lr=1e-6
        )

class EnsembleTrainingConfig:
    """Configuration for ensemble optimization and training"""
    
    ENSEMBLE_CONFIG = {
        "attention_weights_learning": {
            "method": "gradient_based",
            "initial_weights": [0.4, 0.35, 0.25],  # ResNet, EfficientNet, F3Net
            "learning_rate": 0.01,
            "epochs": 20,
            "weight_decay": 1e-4
        },
        
        "confidence_calibration": {
            "method": "temperature_scaling",
            "validation_split": 0.2,
            "cross_validation_folds": 5,
            "initial_temperature": 1.0
        },
        
        "uncertainty_quantification": {
            "method": "monte_carlo_dropout",
            "dropout_samples": 100,
            "uncertainty_threshold": 0.3,
            "ensemble_samples": 10
        },
        
        "meta_learning": {
            "method": "MAML",
            "inner_lr": 0.01,
            "outer_lr": 0.001,
            "adaptation_steps": 5,
            "meta_batch_size": 4
        }
    }
    
    @staticmethod
    def get_attention_optimizer(attention_weights):
        """Get optimizer for attention weight learning"""
        return optim.Adam(
            [attention_weights],
            lr=0.01,
            weight_decay=1e-4
        )

# Custom loss functions
class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class ArcFaceLoss(nn.Module):
    """ArcFace loss for better feature discrimination"""
    
    def __init__(self, feature_dim=512, num_classes=2, margin=0.5, scale=64):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        
        # Create weight matrix
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, feature_dim))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, features, targets):
        # Normalize features and weights
        features = nn.functional.normalize(features, p=2, dim=1)
        weight = nn.functional.normalize(self.weight, p=2, dim=1)
        
        # Calculate cosine similarity
        cosine = nn.functional.linear(features, weight)
        
        # Apply margin
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        target_theta = theta + self.margin
        target_cosine = torch.cos(target_theta)
        
        # Create one-hot targets
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, targets.view(-1, 1).long(), 1)
        
        # Apply margin only to target class
        output = (one_hot * target_cosine) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        
        return nn.functional.cross_entropy(output, targets.long())

class DCTFrequencyLoss(nn.Module):
    """Custom loss for frequency domain analysis in F3Net"""
    
    def __init__(self, lambda_freq=0.3):
        super().__init__()
        self.lambda_freq = lambda_freq
    
    def forward(self, features, targets):
        # Apply DCT to features (simplified implementation)
        batch_size = features.size(0)
        
        # Reshape features for 2D DCT
        if len(features.shape) == 4:
            # If features are 4D (batch, channels, height, width)
            features_2d = features.view(batch_size, -1)
        else:
            features_2d = features
        
        # Compute frequency domain loss (placeholder implementation)
        # In practice, this would involve actual DCT computation and frequency analysis
        freq_loss = torch.mean(torch.abs(features_2d)) * self.lambda_freq
        
        return freq_loss

# Data augmentation configurations
class AugmentationConfig:
    """Augmentation strategies for different models"""
    
    RESNET_AUGMENTATIONS = {
        'spatial': [
            'RandomHorizontalFlip(p=0.5)',
            'RandomRotation(degrees=15)',
            'RandomResizedCrop(224, scale=(0.8, 1.0))',
            'ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)'
        ],
        'advanced': [
            'MixUp(alpha=0.2, p=0.5)',
            'CutMix(alpha=1.0, p=0.3)',
            'GridMask(num_grid=3, p=0.2)'
        ]
    }
    
    EFFICIENTNET_AUGMENTATIONS = {
        'spatial': [
            'RandomHorizontalFlip(p=0.5)',
            'RandomRotation(degrees=10)',
            'RandomResizedCrop(224, scale=(0.85, 1.0))',
            'ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)'
        ],
        'advanced': [
            'AutoAugment(policy="imagenet")',
            'RandAugment(n=2, m=9)',
            'TrivialAugmentWide()',
            'MixUp(alpha=0.2, p=0.7)'
        ]
    }
    
    F3NET_AUGMENTATIONS = {
        'spatial': [
            'RandomHorizontalFlip(p=0.3)',
            'RandomRotation(degrees=5)',
            'ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)'
        ],
        'frequency_preserving': [
            'JPEG compression simulation',
            'Gaussian noise (σ=0.01)',
            'Quantization noise'
        ]
    }

# Training utilities
class TrainingUtils:
    """Utility functions for training"""
    
    @staticmethod
    def get_model_config(model_name: str) -> TrainingConfig:
        """Get training configuration for specific model"""
        configs = {
            "ResNetDetector": ResNetTrainingConfig.get_config(),
            "EfficientNetDetector": EfficientNetTrainingConfig.get_config(),
            "F3NetDetector": F3NetTrainingConfig.get_config()
        }
        
        if model_name not in configs:
            raise ValueError(f"Unknown model: {model_name}")
        
        return configs[model_name]
    
    @staticmethod
    def get_optimizer_and_scheduler(model_name: str, model_parameters, config: TrainingConfig):
        """Get optimizer and scheduler for specific model"""
        optimizers = {
            "ResNetDetector": ResNetTrainingConfig.get_optimizer,
            "EfficientNetDetector": EfficientNetTrainingConfig.get_optimizer,
            "F3NetDetector": F3NetTrainingConfig.get_optimizer
        }
        
        schedulers = {
            "ResNetDetector": ResNetTrainingConfig.get_scheduler,
            "EfficientNetDetector": EfficientNetTrainingConfig.get_scheduler,
            "F3NetDetector": F3NetTrainingConfig.get_scheduler
        }
        
        optimizer = optimizers[model_name](model_parameters, config)
        scheduler = schedulers[model_name](optimizer, config)
        
        return optimizer, scheduler
    
    @staticmethod
    def calculate_effective_batch_size(model_name: str, available_memory_gb: float) -> int:
        """Calculate effective batch size based on available GPU memory"""
        memory_requirements = {
            "ResNetDetector": 0.8,  # GB per sample
            "EfficientNetDetector": 1.2,
            "F3NetDetector": 0.6
        }
        
        memory_per_sample = memory_requirements.get(model_name, 1.0)
        effective_batch_size = int(available_memory_gb * 0.8 / memory_per_sample)
        
        # Ensure batch size is reasonable
        return max(4, min(64, effective_batch_size))

# Training schedule
TRAINING_SCHEDULE = {
    "phase_1_individual_training": {
        "duration_weeks": 8,
        "models": ["ResNetDetector", "EfficientNetDetector", "F3NetDetector"],
        "parallel_training": True,
        "checkpoints": "every_5_epochs"
    },
    
    "phase_2_ensemble_optimization": {
        "duration_weeks": 4,
        "focus": "attention_weights_and_calibration",
        "validation_frequency": "every_epoch"
    },
    
    "phase_3_advanced_techniques": {
        "duration_weeks": 4,
        "techniques": ["adversarial_training", "knowledge_distillation"],
        "experimental": True
    },
    
    "phase_4_production_preparation": {
        "duration_weeks": 2,
        "focus": "optimization_and_deployment",
        "model_compression": True,
        "inference_optimization": True
    }
}

if __name__ == "__main__":
    # Example usage
    print("Training Configurations for Deepfake Detection Ensemble")
    print("=" * 60)
    
    for model_name in ["ResNetDetector", "EfficientNetDetector", "F3NetDetector"]:
        config = TrainingUtils.get_model_config(model_name)
        print(f"\n{model_name}:")
        print(f"  Architecture: {config.architecture}")
        print(f"  Learning Rate: {config.learning_rate}")
        print(f"  Batch Size: {config.batch_size}")
        print(f"  Epochs: {config.epochs}")
        print(f"  Optimizer: {config.optimizer}")
        print(f"  Scheduler: {config.scheduler}")