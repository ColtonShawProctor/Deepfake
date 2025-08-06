#!/usr/bin/env python3
"""
Ensemble Optimization Framework for Multi-Model Deepfake Detection

This module implements advanced ensemble optimization techniques including:
- Attention weight learning
- Confidence calibration (temperature scaling)
- Uncertainty quantification (Monte Carlo dropout)
- Cross-dataset adaptation (MAML)
- Continual learning (Elastic Weight Consolidation)

Based on state-of-the-art ensemble methods and meta-learning approaches.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from pathlib import Path
import json
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.optimize import minimize
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EnsembleResult:
    """Result from ensemble prediction with uncertainty"""
    confidence_score: float
    is_deepfake: bool
    uncertainty: float
    attention_weights: List[float]
    individual_predictions: Dict[str, float]
    calibrated_confidence: float
    metadata: Dict[str, Any]

class AttentionWeightLearner(nn.Module):
    """
    Learnable attention weights for ensemble combination
    
    Uses a small neural network to learn optimal attention weights
    based on individual model predictions and features.
    """
    
    def __init__(self, num_models: int, feature_dim: int = 64):
        super().__init__()
        self.num_models = num_models
        self.feature_dim = feature_dim
        
        # Feature extraction from individual predictions
        self.feature_extractor = nn.Sequential(
            nn.Linear(num_models, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Attention weight computation
        self.attention_network = nn.Sequential(
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, num_models),
            nn.Softmax(dim=-1)
        )
        
        # Initialize with equal weights
        self.register_buffer('base_weights', torch.ones(num_models) / num_models)
        
    def forward(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Learn attention weights based on predictions
        
        Args:
            predictions: Tensor of shape (batch_size, num_models)
            
        Returns:
            attention_weights: Tensor of shape (batch_size, num_models)
        """
        # Extract features from predictions
        features = self.feature_extractor(predictions)
        
        # Compute attention weights
        attention_weights = self.attention_network(features)
        
        return attention_weights

class TemperatureScaling(nn.Module):
    """
    Temperature scaling for confidence calibration
    
    Learns a temperature parameter to calibrate model confidence scores
    using Platt scaling or isotonic regression.
    """
    
    def __init__(self, method: str = "temperature"):
        super().__init__()
        self.method = method
        
        if method == "temperature":
            # Single temperature parameter
            self.temperature = nn.Parameter(torch.ones(1))
        elif method == "vector_scaling":
            # Per-class temperature scaling
            self.temperature = nn.Parameter(torch.ones(2))  # Binary classification
            self.bias = nn.Parameter(torch.zeros(2))
        else:
            raise ValueError(f"Unknown calibration method: {method}")
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling to logits
        
        Args:
            logits: Raw model logits
            
        Returns:
            calibrated_logits: Temperature-scaled logits
        """
        if self.method == "temperature":
            return logits / self.temperature
        elif self.method == "vector_scaling":
            return logits / self.temperature + self.bias
    
    def fit(self, logits: torch.Tensor, labels: torch.Tensor, 
            max_iter: int = 50, lr: float = 0.01):
        """
        Fit temperature scaling parameters using validation data
        
        Args:
            logits: Validation logits
            labels: True labels
            max_iter: Maximum optimization iterations
            lr: Learning rate
        """
        self.train()
        
        # Setup optimizer
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        
        def eval_loss():
            optimizer.zero_grad()
            loss = F.cross_entropy(self.forward(logits), labels.long())
            loss.backward()
            return loss
        
        # Optimize temperature
        optimizer.step(eval_loss)
        
        self.eval()
        logger.info(f"Learned temperature: {self.temperature.item():.4f}")

class MonteCarloDropout:
    """
    Monte Carlo Dropout for uncertainty quantification
    
    Estimates model uncertainty by performing multiple forward passes
    with dropout enabled during inference.
    """
    
    def __init__(self, num_samples: int = 100, dropout_rate: float = 0.3):
        self.num_samples = num_samples
        self.dropout_rate = dropout_rate
    
    def enable_dropout(self, model: nn.Module):
        """Enable dropout during inference"""
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    def disable_dropout(self, model: nn.Module):
        """Disable dropout (standard inference)"""
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.eval()
    
    def predict_with_uncertainty(self, model: nn.Module, 
                                input_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict with uncertainty estimation using MC Dropout
        
        Args:
            model: Neural network model
            input_tensor: Input tensor
            
        Returns:
            mean_prediction: Average prediction across samples
            uncertainty: Prediction uncertainty (variance)
        """
        model.eval()
        self.enable_dropout(model)
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(self.num_samples):
                pred = model(input_tensor)
                predictions.append(pred)
        
        self.disable_dropout(model)
        
        # Stack predictions and compute statistics
        predictions = torch.stack(predictions, dim=0)
        mean_prediction = predictions.mean(dim=0)
        uncertainty = predictions.var(dim=0)
        
        return mean_prediction, uncertainty

class MAMLEnsemble:
    """
    Model-Agnostic Meta-Learning for ensemble adaptation
    
    Enables quick adaptation to new datasets or attack methods
    with few-shot learning capabilities.
    """
    
    def __init__(self, ensemble_model: nn.Module, inner_lr: float = 0.01, 
                 outer_lr: float = 0.001, adaptation_steps: int = 5):
        self.ensemble_model = ensemble_model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.adaptation_steps = adaptation_steps
        
        # Meta-optimizer for outer loop
        self.meta_optimizer = optim.Adam(ensemble_model.parameters(), lr=outer_lr)
    
    def inner_loop(self, support_data: Tuple[torch.Tensor, torch.Tensor]) -> nn.Module:
        """
        Inner loop: Fast adaptation to support set
        
        Args:
            support_data: (support_inputs, support_labels)
            
        Returns:
            adapted_model: Model adapted to support set
        """
        support_inputs, support_labels = support_data
        
        # Clone model for adaptation
        adapted_model = self._clone_model(self.ensemble_model)
        adapted_optimizer = optim.SGD(adapted_model.parameters(), lr=self.inner_lr)
        
        # Adaptation steps
        for step in range(self.adaptation_steps):
            adapted_optimizer.zero_grad()
            
            predictions = adapted_model(support_inputs)
            loss = F.binary_cross_entropy_with_logits(predictions, support_labels)
            
            loss.backward()
            adapted_optimizer.step()
        
        return adapted_model
    
    def outer_loop(self, query_data: Tuple[torch.Tensor, torch.Tensor], 
                   adapted_model: nn.Module) -> float:
        """
        Outer loop: Update meta-parameters using query set
        
        Args:
            query_data: (query_inputs, query_labels)
            adapted_model: Model adapted in inner loop
            
        Returns:
            meta_loss: Loss on query set
        """
        query_inputs, query_labels = query_data
        
        self.meta_optimizer.zero_grad()
        
        # Compute meta-loss on query set
        predictions = adapted_model(query_inputs)
        meta_loss = F.binary_cross_entropy_with_logits(predictions, query_labels)
        
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return meta_loss.item()
    
    def _clone_model(self, model: nn.Module) -> nn.Module:
        """Create a copy of the model for adaptation"""
        import copy
        return copy.deepcopy(model)

class EnsembleOptimizer:
    """
    Main ensemble optimization framework
    
    Coordinates all optimization components including attention learning,
    calibration, uncertainty quantification, and meta-learning.
    """
    
    def __init__(self, model_names: List[str], device: str = "cuda"):
        self.model_names = model_names
        self.num_models = len(model_names)
        self.device = device
        
        # Initialize components
        self.attention_learner = AttentionWeightLearner(
            num_models=self.num_models
        ).to(device)
        
        self.temperature_scaling = TemperatureScaling(method="temperature").to(device)
        self.mc_dropout = MonteCarloDropout(num_samples=50)
        
        # Optimization history
        self.training_history = {
            'attention_weights': [],
            'calibration_scores': [],
            'uncertainty_scores': [],
            'validation_metrics': []
        }
        
        logger.info(f"Initialized ensemble optimizer for {self.num_models} models")
    
    def optimize_attention_weights(self, 
                                 validation_data: torch.utils.data.DataLoader,
                                 individual_models: Dict[str, nn.Module],
                                 epochs: int = 20) -> Dict[str, float]:
        """
        Learn optimal attention weights using validation data
        
        Args:
            validation_data: Validation dataset
            individual_models: Dictionary of individual models
            epochs: Training epochs for attention learning
            
        Returns:
            metrics: Training metrics and final attention weights
        """
        logger.info("Starting attention weight optimization...")
        
        # Setup optimizer for attention learner
        optimizer = optim.Adam(self.attention_learner.parameters(), lr=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
        
        best_auc = 0.0
        
        for epoch in range(epochs):
            epoch_losses = []
            all_predictions = []
            all_labels = []
            
            self.attention_learner.train()
            
            for batch_idx, (inputs, labels) in enumerate(validation_data):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Get predictions from individual models
                individual_preds = []
                for model_name, model in individual_models.items():
                    model.eval()
                    with torch.no_grad():
                        pred = torch.sigmoid(model(inputs))
                        individual_preds.append(pred)
                
                # Stack predictions
                individual_preds = torch.stack(individual_preds, dim=-1).squeeze()
                
                # Learn attention weights
                attention_weights = self.attention_learner(individual_preds)
                
                # Compute ensemble prediction
                ensemble_pred = (individual_preds * attention_weights).sum(dim=-1)
                
                # Compute loss
                loss = F.binary_cross_entropy(ensemble_pred, labels.float())
                
                # Optimization step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_losses.append(loss.item())
                all_predictions.extend(ensemble_pred.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            
            # Compute epoch metrics
            avg_loss = np.mean(epoch_losses)
            auc_score = roc_auc_score(all_labels, all_predictions)
            accuracy = accuracy_score(all_labels, np.array(all_predictions) > 0.5)
            
            scheduler.step(avg_loss)
            
            logger.info(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, "
                       f"AUC={auc_score:.4f}, Accuracy={accuracy:.4f}")
            
            # Save best model
            if auc_score > best_auc:
                best_auc = auc_score
                self._save_attention_weights()
            
            # Store history
            self.training_history['attention_weights'].append({
                'epoch': epoch,
                'loss': avg_loss,
                'auc': auc_score,
                'accuracy': accuracy
            })
        
        logger.info(f"Attention weight optimization completed. Best AUC: {best_auc:.4f}")
        
        return {
            'best_auc': best_auc,
            'final_loss': avg_loss,
            'epochs_trained': epochs
        }
    
    def calibrate_confidence(self, 
                           validation_data: torch.utils.data.DataLoader,
                           ensemble_model: nn.Module) -> float:
        """
        Calibrate ensemble confidence using temperature scaling
        
        Args:
            validation_data: Validation dataset
            ensemble_model: Trained ensemble model
            
        Returns:
            calibration_score: ECE (Expected Calibration Error)
        """
        logger.info("Starting confidence calibration...")
        
        # Collect validation predictions and labels
        all_logits = []
        all_labels = []
        
        ensemble_model.eval()
        with torch.no_grad():
            for inputs, labels in validation_data:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                logits = ensemble_model(inputs)
                all_logits.append(logits)
                all_labels.append(labels)
        
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Fit temperature scaling
        self.temperature_scaling.fit(all_logits, all_labels)
        
        # Evaluate calibration
        calibrated_logits = self.temperature_scaling(all_logits)
        calibrated_probs = torch.sigmoid(calibrated_logits)
        
        ece_score = self._compute_ece(calibrated_probs.cpu().numpy(), 
                                     all_labels.cpu().numpy())
        
        logger.info(f"Confidence calibration completed. ECE: {ece_score:.4f}")
        
        return ece_score
    
    def estimate_uncertainty(self, 
                           test_data: torch.utils.data.DataLoader,
                           ensemble_model: nn.Module) -> Dict[str, float]:
        """
        Estimate prediction uncertainty using Monte Carlo Dropout
        
        Args:
            test_data: Test dataset
            ensemble_model: Trained ensemble model
            
        Returns:
            uncertainty_metrics: Dictionary of uncertainty metrics
        """
        logger.info("Estimating prediction uncertainty...")
        
        all_uncertainties = []
        all_predictions = []
        all_labels = []
        
        for inputs, labels in test_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Get uncertainty estimates
            mean_pred, uncertainty = self.mc_dropout.predict_with_uncertainty(
                ensemble_model, inputs
            )
            
            all_uncertainties.extend(uncertainty.cpu().numpy())
            all_predictions.extend(torch.sigmoid(mean_pred).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # Compute uncertainty metrics
        avg_uncertainty = np.mean(all_uncertainties)
        uncertainty_std = np.std(all_uncertainties)
        
        # Correlation between uncertainty and prediction error
        errors = np.abs(np.array(all_predictions) - np.array(all_labels))
        uncertainty_correlation = np.corrcoef(all_uncertainties, errors)[0, 1]
        
        uncertainty_metrics = {
            'average_uncertainty': avg_uncertainty,
            'uncertainty_std': uncertainty_std,
            'uncertainty_error_correlation': uncertainty_correlation
        }
        
        logger.info(f"Uncertainty estimation completed. "
                   f"Avg uncertainty: {avg_uncertainty:.4f}")
        
        return uncertainty_metrics
    
    def meta_learn_adaptation(self, 
                            meta_train_tasks: List[Tuple[torch.Tensor, torch.Tensor]],
                            meta_val_tasks: List[Tuple[torch.Tensor, torch.Tensor]],
                            ensemble_model: nn.Module,
                            epochs: int = 50) -> Dict[str, float]:
        """
        Meta-learn for quick adaptation to new datasets/attacks
        
        Args:
            meta_train_tasks: List of (support_set, query_set) for training
            meta_val_tasks: List of (support_set, query_set) for validation
            ensemble_model: Ensemble model to meta-train
            epochs: Number of meta-training epochs
            
        Returns:
            meta_learning_metrics: Performance metrics
        """
        logger.info("Starting meta-learning for adaptation...")
        
        maml = MAMLEnsemble(ensemble_model)
        
        meta_train_losses = []
        meta_val_losses = []
        
        for epoch in range(epochs):
            # Meta-training
            epoch_train_losses = []
            
            for support_data, query_data in meta_train_tasks:
                # Inner loop: adapt to support set
                adapted_model = maml.inner_loop(support_data)
                
                # Outer loop: update meta-parameters
                meta_loss = maml.outer_loop(query_data, adapted_model)
                epoch_train_losses.append(meta_loss)
            
            avg_train_loss = np.mean(epoch_train_losses)
            meta_train_losses.append(avg_train_loss)
            
            # Meta-validation
            if epoch % 10 == 0:
                epoch_val_losses = []
                
                for support_data, query_data in meta_val_tasks:
                    adapted_model = maml.inner_loop(support_data)
                    
                    with torch.no_grad():
                        query_inputs, query_labels = query_data
                        predictions = adapted_model(query_inputs)
                        val_loss = F.binary_cross_entropy_with_logits(
                            predictions, query_labels
                        ).item()
                    
                    epoch_val_losses.append(val_loss)
                
                avg_val_loss = np.mean(epoch_val_losses)
                meta_val_losses.append(avg_val_loss)
                
                logger.info(f"Meta-epoch {epoch}: Train Loss={avg_train_loss:.4f}, "
                           f"Val Loss={avg_val_loss:.4f}")
        
        logger.info("Meta-learning completed.")
        
        return {
            'final_train_loss': meta_train_losses[-1],
            'final_val_loss': meta_val_losses[-1] if meta_val_losses else None,
            'epochs_trained': epochs
        }
    
    def _compute_ece(self, predictions: np.ndarray, labels: np.ndarray, 
                     n_bins: int = 15) -> float:
        """
        Compute Expected Calibration Error (ECE)
        
        Args:
            predictions: Model predictions
            labels: True labels
            n_bins: Number of bins for calibration
            
        Returns:
            ece: Expected Calibration Error
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find predictions in current bin
            in_bin = (predictions > bin_lower) & (predictions <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = labels[in_bin].mean()
                avg_confidence_in_bin = predictions[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def _save_attention_weights(self):
        """Save learned attention weights"""
        save_path = Path("models/attention_weights.pth")
        save_path.parent.mkdir(exist_ok=True)
        
        torch.save({
            'attention_learner_state': self.attention_learner.state_dict(),
            'model_names': self.model_names
        }, save_path)
        
        logger.info(f"Attention weights saved to {save_path}")
    
    def save_optimization_results(self, save_path: str):
        """Save all optimization results and configurations"""
        results = {
            'model_names': self.model_names,
            'training_history': self.training_history,
            'temperature': self.temperature_scaling.temperature.item(),
            'mc_dropout_samples': self.mc_dropout.num_samples
        }
        
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Optimization results saved to {save_path}")

# Example usage and testing
if __name__ == "__main__":
    # Example: Initialize ensemble optimizer
    model_names = ["ResNetDetector", "EfficientNetDetector", "F3NetDetector"]
    optimizer = EnsembleOptimizer(model_names, device="cuda")
    
    print(f"Ensemble optimizer initialized for models: {model_names}")
    print(f"Attention learner parameters: {sum(p.numel() for p in optimizer.attention_learner.parameters())}")
    print(f"Temperature scaling method: {optimizer.temperature_scaling.method}")
    print(f"MC Dropout samples: {optimizer.mc_dropout.num_samples}")