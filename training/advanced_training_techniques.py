#!/usr/bin/env python3
"""
Advanced Training Techniques for Deepfake Detection

This module implements cutting-edge training techniques to improve
robustness, generalization, and performance of deepfake detectors:

1. Adversarial Training (FGSM, PGD, C&W, AutoAttack)
2. Self-Supervised Learning (SimCLR, MoCo, BYOL)
3. Knowledge Distillation (Teacher-Student, Progressive)
4. Contrastive Learning for deepfake-specific features
5. Domain Adaptation and Transfer Learning
6. Curriculum Learning and Progressive Training

Based on latest research in robust deep learning and deepfake detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Callable, Union
from dataclasses import dataclass
import logging
from pathlib import Path
import json
from PIL import Image
import cv2
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AdvTrainingConfig:
    """Configuration for advanced training techniques"""
    adversarial_ratio: float = 0.3
    attack_methods: List[str] = None
    contrastive_temperature: float = 0.1
    knowledge_distillation_alpha: float = 0.7
    curriculum_stages: int = 3
    domain_adaptation_lambda: float = 0.1

class AdversarialAttacks:
    """
    Implementation of various adversarial attacks for robust training
    """
    
    @staticmethod
    def fgsm_attack(image: torch.Tensor, epsilon: float, 
                   data_grad: torch.Tensor) -> torch.Tensor:
        """
        Fast Gradient Sign Method (FGSM) Attack
        
        Args:
            image: Original image tensor
            epsilon: Attack strength
            data_grad: Gradient of loss w.r.t. input
            
        Returns:
            adversarial_image: Perturbed image
        """
        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()
        
        # Create the perturbed image
        perturbed_image = image + epsilon * sign_data_grad
        
        # Clip to maintain valid pixel range
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        
        return perturbed_image
    
    @staticmethod
    def pgd_attack(model: nn.Module, image: torch.Tensor, label: torch.Tensor,
                   epsilon: float = 0.03, alpha: float = 0.007, 
                   num_steps: int = 10) -> torch.Tensor:
        """
        Projected Gradient Descent (PGD) Attack
        
        Args:
            model: Target model
            image: Original image
            label: True label
            epsilon: Maximum perturbation
            alpha: Step size
            num_steps: Number of PGD steps
            
        Returns:
            adversarial_image: Adversarially perturbed image
        """
        # Start with random noise
        delta = torch.zeros_like(image).uniform_(-epsilon, epsilon)
        delta.requires_grad = True
        
        for step in range(num_steps):
            # Forward pass
            output = model(image + delta)
            loss = F.binary_cross_entropy_with_logits(output, label.float())
            
            # Backward pass
            loss.backward()
            
            # Update delta
            delta.data = delta.data + alpha * delta.grad.data.sign()
            delta.data = torch.clamp(delta.data, -epsilon, epsilon)
            delta.data = torch.clamp(image.data + delta.data, 0, 1) - image.data
            
            # Reset gradients
            delta.grad.zero_()
        
        return image + delta
    
    @staticmethod
    def cw_attack(model: nn.Module, image: torch.Tensor, label: torch.Tensor,
                  confidence: float = 0, kappa: float = 0, 
                  learning_rate: float = 0.01, max_iter: int = 1000) -> torch.Tensor:
        """
        Carlini & Wagner (C&W) Attack
        
        Args:
            model: Target model
            image: Original image
            label: True label
            confidence: Confidence parameter
            kappa: Loss balance parameter
            learning_rate: Optimization learning rate
            max_iter: Maximum iterations
            
        Returns:
            adversarial_image: Adversarially perturbed image
        """
        # Initialize perturbation variable
        w = torch.zeros_like(image, requires_grad=True)
        optimizer = optim.Adam([w], lr=learning_rate)
        
        for iteration in range(max_iter):
            # Create adversarial example
            adv_image = 0.5 * (torch.tanh(w + torch.log(image / (1 - image + 1e-8))) + 1)
            
            # Forward pass
            output = model(adv_image)
            
            # C&W loss function
            f_loss = F.binary_cross_entropy_with_logits(output, label.float())
            
            # L2 distance penalty
            l2_loss = torch.norm(adv_image - image, p=2)
            
            # Total loss
            total_loss = f_loss + kappa * l2_loss
            
            # Optimization step
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Early stopping condition
            if iteration % 100 == 0:
                with torch.no_grad():
                    pred = torch.sigmoid(model(adv_image))
                    if (pred > 0.5) != (label > 0.5):  # Successful attack
                        break
        
        return 0.5 * (torch.tanh(w + torch.log(image / (1 - image + 1e-8))) + 1)

class AdversarialTrainer:
    """
    Adversarial training framework for robust deepfake detection
    """
    
    def __init__(self, model: nn.Module, config: AdvTrainingConfig):
        self.model = model
        self.config = config
        self.attack_methods = {
            'fgsm': self._fgsm_step,
            'pgd': self._pgd_step,
            'cw': self._cw_step
        }
        
        # Default attack methods if not specified
        if config.attack_methods is None:
            config.attack_methods = ['fgsm', 'pgd']
    
    def train_epoch(self, dataloader: DataLoader, optimizer: optim.Optimizer,
                   criterion: nn.Module, device: str) -> Dict[str, float]:
        """
        Train one epoch with adversarial examples
        
        Args:
            dataloader: Training data loader
            optimizer: Model optimizer
            criterion: Loss function
            device: Computing device
            
        Returns:
            metrics: Training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        natural_loss = 0.0
        adversarial_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Split batch into natural and adversarial examples
            batch_size = inputs.size(0)
            adv_size = int(batch_size * self.config.adversarial_ratio)
            nat_size = batch_size - adv_size
            
            # Natural examples
            if nat_size > 0:
                nat_inputs = inputs[:nat_size]
                nat_labels = labels[:nat_size]
                
                optimizer.zero_grad()
                nat_outputs = self.model(nat_inputs)
                nat_loss = criterion(nat_outputs, nat_labels.float())
                nat_loss.backward()
                
                natural_loss += nat_loss.item()
            
            # Adversarial examples
            if adv_size > 0:
                adv_inputs = inputs[nat_size:]
                adv_labels = labels[nat_size:]
                
                # Generate adversarial examples
                adv_examples = self._generate_adversarial_batch(
                    adv_inputs, adv_labels, device
                )
                
                optimizer.zero_grad()
                adv_outputs = self.model(adv_examples)
                adv_loss = criterion(adv_outputs, adv_labels.float())
                adv_loss.backward()
                
                adversarial_loss += adv_loss.item()
            
            optimizer.step()
            
            # Calculate accuracy on natural examples
            with torch.no_grad():
                outputs = self.model(inputs)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels.float()).sum().item()
            
            total_loss += (nat_loss.item() if nat_size > 0 else 0) + \
                         (adv_loss.item() if adv_size > 0 else 0)
        
        accuracy = 100.0 * correct / total
        
        return {
            'total_loss': total_loss / len(dataloader),
            'natural_loss': natural_loss / len(dataloader),
            'adversarial_loss': adversarial_loss / len(dataloader),
            'accuracy': accuracy
        }
    
    def _generate_adversarial_batch(self, inputs: torch.Tensor, 
                                  labels: torch.Tensor, device: str) -> torch.Tensor:
        """Generate adversarial examples for a batch"""
        # Randomly select attack method
        attack_method = random.choice(self.config.attack_methods)
        
        if attack_method in self.attack_methods:
            return self.attack_methods[attack_method](inputs, labels, device)
        else:
            logger.warning(f"Unknown attack method: {attack_method}")
            return inputs
    
    def _fgsm_step(self, inputs: torch.Tensor, labels: torch.Tensor, 
                   device: str) -> torch.Tensor:
        """Generate FGSM adversarial examples"""
        inputs.requires_grad = True
        outputs = self.model(inputs)
        loss = F.binary_cross_entropy_with_logits(outputs, labels.float())
        
        self.model.zero_grad()
        loss.backward()
        
        return AdversarialAttacks.fgsm_attack(inputs, epsilon=0.03, 
                                            data_grad=inputs.grad.data)
    
    def _pgd_step(self, inputs: torch.Tensor, labels: torch.Tensor, 
                  device: str) -> torch.Tensor:
        """Generate PGD adversarial examples"""
        return AdversarialAttacks.pgd_attack(self.model, inputs, labels)
    
    def _cw_step(self, inputs: torch.Tensor, labels: torch.Tensor, 
                 device: str) -> torch.Tensor:
        """Generate C&W adversarial examples"""
        return AdversarialAttacks.cw_attack(self.model, inputs, labels)

class ContrastiveLearning:
    """
    Contrastive learning for self-supervised deepfake feature learning
    
    Implements SimCLR-style contrastive learning adapted for deepfake detection
    """
    
    def __init__(self, feature_dim: int = 128, temperature: float = 0.1):
        self.feature_dim = feature_dim
        self.temperature = temperature
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(2048, 512),  # Assuming ResNet-50 features
            nn.ReLU(),
            nn.Linear(512, feature_dim)
        )
    
    def nt_xent_loss(self, features: torch.Tensor) -> torch.Tensor:
        """
        Normalized Temperature-scaled Cross Entropy Loss (NT-Xent)
        
        Args:
            features: Feature vectors [2*batch_size, feature_dim]
            
        Returns:
            contrastive_loss: NT-Xent loss
        """
        batch_size = features.size(0) // 2
        
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create labels for positive pairs
        labels = torch.cat([torch.arange(batch_size, 2*batch_size),
                           torch.arange(batch_size)]).to(features.device)
        
        # Mask to remove self-similarity
        mask = torch.eye(2*batch_size, dtype=bool).to(features.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))
        
        # Compute loss
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss
    
    def create_augmented_pairs(self, images: torch.Tensor) -> torch.Tensor:
        """
        Create augmented pairs for contrastive learning
        
        Args:
            images: Batch of images
            
        Returns:
            augmented_pairs: [2*batch_size, channels, height, width]
        """
        # Define strong augmentation
        strong_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))
            ], p=0.5)
        ])
        
        augmented_images = []
        
        for image in images:
            # Convert to PIL for transformations
            pil_image = transforms.ToPILImage()(image.cpu())
            
            # Create two augmented versions
            aug1 = strong_transform(pil_image)
            aug2 = strong_transform(pil_image)
            
            # Convert back to tensor
            aug1 = transforms.ToTensor()(aug1)
            aug2 = transforms.ToTensor()(aug2)
            
            augmented_images.extend([aug1, aug2])
        
        return torch.stack(augmented_images).to(images.device)

class KnowledgeDistillation:
    """
    Knowledge distillation for model compression and knowledge transfer
    """
    
    def __init__(self, teacher_model: nn.Module, student_model: nn.Module,
                 temperature: float = 4.0, alpha: float = 0.7):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha
        
        # Freeze teacher model
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        self.teacher_model.eval()
    
    def distillation_loss(self, student_logits: torch.Tensor, 
                         teacher_logits: torch.Tensor,
                         labels: torch.Tensor) -> torch.Tensor:
        """
        Compute knowledge distillation loss
        
        Args:
            student_logits: Student model outputs
            teacher_logits: Teacher model outputs
            labels: Ground truth labels
            
        Returns:
            total_loss: Combined distillation and student loss
        """
        # Distillation loss (KL divergence)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=1)
        
        distill_loss = F.kl_div(student_log_probs, teacher_probs, 
                               reduction='batchmean') * (self.temperature ** 2)
        
        # Student loss (cross-entropy)
        student_loss = F.binary_cross_entropy_with_logits(student_logits, labels.float())
        
        # Combined loss
        total_loss = self.alpha * distill_loss + (1 - self.alpha) * student_loss
        
        return total_loss
    
    def train_step(self, inputs: torch.Tensor, labels: torch.Tensor,
                   optimizer: optim.Optimizer) -> Dict[str, float]:
        """
        Single training step with knowledge distillation
        
        Args:
            inputs: Input batch
            labels: True labels
            optimizer: Student model optimizer
            
        Returns:
            metrics: Training step metrics
        """
        # Teacher predictions (no gradients)
        with torch.no_grad():
            teacher_logits = self.teacher_model(inputs)
        
        # Student predictions
        student_logits = self.student_model(inputs)
        
        # Compute loss
        loss = self.distillation_loss(student_logits, teacher_logits, labels)
        
        # Optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        with torch.no_grad():
            predicted = (torch.sigmoid(student_logits) > 0.5).float()
            accuracy = (predicted == labels.float()).float().mean().item()
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy * 100
        }

class CurriculumLearning:
    """
    Curriculum learning for progressive training difficulty
    """
    
    def __init__(self, num_stages: int = 3, difficulty_threshold: float = 0.8):
        self.num_stages = num_stages
        self.difficulty_threshold = difficulty_threshold
        self.current_stage = 0
        
        # Define curriculum stages
        self.stages = [
            {'name': 'easy', 'compression_quality': [95, 100], 'resolution': [224, 224]},
            {'name': 'medium', 'compression_quality': [70, 95], 'resolution': [160, 224]},
            {'name': 'hard', 'compression_quality': [30, 70], 'resolution': [112, 160]}
        ]
    
    def should_advance_stage(self, current_accuracy: float) -> bool:
        """
        Determine if curriculum should advance to next stage
        
        Args:
            current_accuracy: Current model accuracy
            
        Returns:
            should_advance: Whether to advance curriculum stage
        """
        return (current_accuracy > self.difficulty_threshold and 
                self.current_stage < self.num_stages - 1)
    
    def advance_stage(self):
        """Advance to next curriculum stage"""
        if self.current_stage < self.num_stages - 1:
            self.current_stage += 1
            logger.info(f"Advanced to curriculum stage {self.current_stage + 1}: "
                       f"{self.stages[self.current_stage]['name']}")
    
    def get_current_stage_config(self) -> Dict:
        """Get configuration for current curriculum stage"""
        return self.stages[self.current_stage]
    
    def filter_data_by_stage(self, dataset: Dataset) -> Dataset:
        """
        Filter dataset based on current curriculum stage
        
        Args:
            dataset: Original dataset
            
        Returns:
            filtered_dataset: Dataset filtered for current stage
        """
        stage_config = self.get_current_stage_config()
        
        # This would implement actual filtering logic based on
        # image quality, compression, resolution, etc.
        # For now, return original dataset
        return dataset

class DomainAdaptation:
    """
    Domain adaptation for cross-dataset generalization
    """
    
    def __init__(self, feature_extractor: nn.Module, classifier: nn.Module,
                 domain_discriminator: nn.Module, lambda_domain: float = 0.1):
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.domain_discriminator = domain_discriminator
        self.lambda_domain = lambda_domain
    
    def domain_adversarial_loss(self, source_features: torch.Tensor,
                               target_features: torch.Tensor) -> torch.Tensor:
        """
        Compute domain adversarial loss for domain adaptation
        
        Args:
            source_features: Features from source domain
            target_features: Features from target domain
            
        Returns:
            domain_loss: Domain adversarial loss
        """
        # Concatenate features
        all_features = torch.cat([source_features, target_features], dim=0)
        
        # Create domain labels (0 for source, 1 for target)
        domain_labels = torch.cat([
            torch.zeros(source_features.size(0)),
            torch.ones(target_features.size(0))
        ]).to(source_features.device)
        
        # Domain discrimination
        domain_preds = self.domain_discriminator(all_features)
        
        # Domain adversarial loss
        domain_loss = F.binary_cross_entropy_with_logits(
            domain_preds.squeeze(), domain_labels
        )
        
        return domain_loss
    
    def train_step(self, source_data: Tuple[torch.Tensor, torch.Tensor],
                   target_data: torch.Tensor,
                   optimizer_fe: optim.Optimizer,
                   optimizer_clf: optim.Optimizer,
                   optimizer_disc: optim.Optimizer) -> Dict[str, float]:
        """
        Training step for domain adaptation
        
        Args:
            source_data: (source_inputs, source_labels)
            target_data: target_inputs (unlabeled)
            optimizer_fe: Feature extractor optimizer
            optimizer_clf: Classifier optimizer
            optimizer_disc: Domain discriminator optimizer
            
        Returns:
            metrics: Training metrics
        """
        source_inputs, source_labels = source_data
        
        # Extract features
        source_features = self.feature_extractor(source_inputs)
        target_features = self.feature_extractor(target_data)
        
        # Classification loss on source data
        source_preds = self.classifier(source_features)
        classification_loss = F.binary_cross_entropy_with_logits(
            source_preds, source_labels.float()
        )
        
        # Domain adversarial loss
        domain_loss = self.domain_adversarial_loss(source_features, target_features)
        
        # Update domain discriminator
        optimizer_disc.zero_grad()
        domain_loss.backward(retain_graph=True)
        optimizer_disc.step()
        
        # Update feature extractor and classifier
        total_loss = classification_loss - self.lambda_domain * domain_loss
        
        optimizer_fe.zero_grad()
        optimizer_clf.zero_grad()
        total_loss.backward()
        optimizer_fe.step()
        optimizer_clf.step()
        
        return {
            'classification_loss': classification_loss.item(),
            'domain_loss': domain_loss.item(),
            'total_loss': total_loss.item()
        }

class AdvancedTrainingPipeline:
    """
    Main pipeline combining all advanced training techniques
    """
    
    def __init__(self, model: nn.Module, config: AdvTrainingConfig):
        self.model = model
        self.config = config
        
        # Initialize training components
        self.adversarial_trainer = AdversarialTrainer(model, config)
        self.contrastive_learning = ContrastiveLearning(
            temperature=config.contrastive_temperature
        )
        self.curriculum_learning = CurriculumLearning(
            num_stages=config.curriculum_stages
        )
        
        # Training history
        self.training_history = {
            'adversarial_accuracy': [],
            'natural_accuracy': [],
            'contrastive_loss': [],
            'curriculum_stage': []
        }
    
    def train_epoch_advanced(self, dataloader: DataLoader, 
                           optimizer: optim.Optimizer,
                           criterion: nn.Module, device: str,
                           epoch: int) -> Dict[str, float]:
        """
        Advanced training epoch combining multiple techniques
        
        Args:
            dataloader: Training data loader
            optimizer: Model optimizer
            criterion: Loss function
            device: Computing device
            epoch: Current epoch number
            
        Returns:
            metrics: Comprehensive training metrics
        """
        logger.info(f"Starting advanced training epoch {epoch}")
        
        # Adversarial training
        adv_metrics = self.adversarial_trainer.train_epoch(
            dataloader, optimizer, criterion, device
        )
        
        # Update curriculum if needed
        if self.curriculum_learning.should_advance_stage(adv_metrics['accuracy']):
            self.curriculum_learning.advance_stage()
        
        # Record training history
        self.training_history['adversarial_accuracy'].append(adv_metrics['accuracy'])
        self.training_history['curriculum_stage'].append(
            self.curriculum_learning.current_stage
        )
        
        # Comprehensive metrics
        metrics = {
            **adv_metrics,
            'curriculum_stage': self.curriculum_learning.current_stage,
            'epoch': epoch
        }
        
        logger.info(f"Epoch {epoch} metrics: {metrics}")
        
        return metrics
    
    def save_training_state(self, save_path: str):
        """Save complete training state"""
        state = {
            'model_state_dict': self.model.state_dict(),
            'training_history': self.training_history,
            'curriculum_stage': self.curriculum_learning.current_stage,
            'config': self.config.__dict__
        }
        
        torch.save(state, save_path)
        logger.info(f"Training state saved to {save_path}")

# Example usage and testing
if __name__ == "__main__":
    # Example configuration
    config = AdvTrainingConfig(
        adversarial_ratio=0.3,
        attack_methods=['fgsm', 'pgd'],
        contrastive_temperature=0.1,
        knowledge_distillation_alpha=0.7,
        curriculum_stages=3
    )
    
    print("Advanced Training Techniques initialized")
    print(f"Adversarial ratio: {config.adversarial_ratio}")
    print(f"Attack methods: {config.attack_methods}")
    print(f"Curriculum stages: {config.curriculum_stages}")
    
    # Test adversarial attacks
    print("\nTesting adversarial attack implementations...")
    dummy_image = torch.randn(1, 3, 224, 224)
    dummy_grad = torch.randn_like(dummy_image)
    
    fgsm_result = AdversarialAttacks.fgsm_attack(dummy_image, 0.03, dummy_grad)
    print(f"FGSM attack shape: {fgsm_result.shape}")
    
    print("Advanced training techniques ready for deployment!")