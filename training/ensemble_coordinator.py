"""
Ensemble Training Coordination System

This module provides comprehensive ensemble training coordination including:
- Ensemble training pipeline that coordinates all models
- Ensemble weight optimization during training
- Cross-validation for ensemble performance evaluation
- Model agreement analysis and disagreement handling
- Ensemble fine-tuning and calibration
- Multi-model training orchestration
"""

import os
import sys
import logging
import json
import time
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import yaml

# Add the app directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from training.model_trainer import ModelTrainer, TrainingConfig
from training.dataset_management import DatasetManager
from training.experiment_tracker import ExperimentTracker, ExperimentConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnsembleConfig:
    """Configuration for ensemble training"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load ensemble configuration"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        
        # Default configuration
        return {
            "ensemble": {
                "models": ["mesonet", "xception", "efficientnet", "f3net"],
                "fusion_method": "weighted_average",
                "weight_optimization": True,
                "cross_validation": True,
                "cv_folds": 5,
                "agreement_threshold": 0.7,
                "calibration": True
            },
            "training": {
                "parallel_training": False,
                "individual_training": True,
                "ensemble_training": True,
                "fine_tuning": True,
                "calibration_epochs": 10
            },
            "optimization": {
                "weight_optimization_method": "bayesian",
                "optimization_metric": "auc",
                "constraint_type": "sum_to_one",
                "initial_weights": "uniform"
            },
            "evaluation": {
                "cross_dataset_evaluation": True,
                "robustness_testing": True,
                "adversarial_testing": False,
                "performance_benchmarking": True
            }
        }

class EnsembleWeightOptimizer:
    """Optimizes ensemble weights for best performance"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.optimization_method = config["optimization"]["weight_optimization_method"]
        self.optimization_metric = config["optimization"]["optimization_metric"]
        self.constraint_type = config["optimization"]["constraint_type"]
    
    def optimize_weights(self, predictions: List[np.ndarray], labels: np.ndarray) -> np.ndarray:
        """Optimize ensemble weights given predictions and labels"""
        if self.optimization_method == "bayesian":
            return self._bayesian_optimization(predictions, labels)
        elif self.optimization_method == "grid_search":
            return self._grid_search_optimization(predictions, labels)
        elif self.optimization_method == "genetic":
            return self._genetic_optimization(predictions, labels)
        else:
            return self._uniform_weights(len(predictions))
    
    def _bayesian_optimization(self, predictions: List[np.ndarray], labels: np.ndarray) -> np.ndarray:
        """Bayesian optimization for weight optimization"""
        n_models = len(predictions)
        
        def objective(weights):
            # Ensure weights sum to 1
            weights = np.array(weights)
            weights = weights / np.sum(weights)
            
            # Calculate ensemble prediction
            ensemble_pred = np.zeros_like(predictions[0])
            for i, pred in enumerate(predictions):
                ensemble_pred += weights[i] * pred
            
            # Calculate metric
            if self.optimization_metric == "auc":
                return -roc_auc_score(labels, ensemble_pred)
            elif self.optimization_metric == "accuracy":
                binary_pred = (ensemble_pred > 0.5).astype(int)
                return -accuracy_score(labels, binary_pred)
            else:
                return -roc_auc_score(labels, ensemble_pred)
        
        # Initial weights
        if self.config["optimization"]["initial_weights"] == "uniform":
            initial_weights = np.ones(n_models) / n_models
        else:
            initial_weights = np.random.dirichlet(np.ones(n_models))
        
        # Constraints
        constraints = []
        if self.constraint_type == "sum_to_one":
            constraints.append({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        bounds = [(0, 1)] * n_models
        
        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            optimal_weights = result.x / np.sum(result.x)
            logger.info(f"Bayesian optimization successful. Optimal weights: {optimal_weights}")
            return optimal_weights
        else:
            logger.warning("Bayesian optimization failed, using uniform weights")
            return np.ones(n_models) / n_models
    
    def _grid_search_optimization(self, predictions: List[np.ndarray], labels: np.ndarray) -> np.ndarray:
        """Grid search optimization for weight optimization"""
        n_models = len(predictions)
        best_score = -1
        best_weights = np.ones(n_models) / n_models
        
        # Grid search with step size 0.1
        step = 0.1
        for w1 in np.arange(0, 1.1, step):
            for w2 in np.arange(0, 1.1, step):
                for w3 in np.arange(0, 1.1, step):
                    w4 = 1 - w1 - w2 - w3
                    if w4 >= 0 and w4 <= 1:
                        weights = np.array([w1, w2, w3, w4])
                        weights = weights / np.sum(weights)
                        
                        # Calculate ensemble prediction
                        ensemble_pred = np.zeros_like(predictions[0])
                        for i, pred in enumerate(predictions):
                            ensemble_pred += weights[i] * pred
                        
                        # Calculate score
                        score = roc_auc_score(labels, ensemble_pred)
                        
                        if score > best_score:
                            best_score = score
                            best_weights = weights
        
        logger.info(f"Grid search completed. Best weights: {best_weights}, Score: {best_score}")
        return best_weights
    
    def _genetic_optimization(self, predictions: List[np.ndarray], labels: np.ndarray) -> np.ndarray:
        """Genetic algorithm optimization for weight optimization"""
        n_models = len(predictions)
        population_size = 50
        generations = 100
        mutation_rate = 0.1
        
        # Initialize population
        population = []
        for _ in range(population_size):
            weights = np.random.dirichlet(np.ones(n_models))
            population.append(weights)
        
        def fitness(weights):
            ensemble_pred = np.zeros_like(predictions[0])
            for i, pred in enumerate(predictions):
                ensemble_pred += weights[i] * pred
            return roc_auc_score(labels, ensemble_pred)
        
        # Evolution
        for generation in range(generations):
            # Evaluate fitness
            fitness_scores = [fitness(weights) for weights in population]
            
            # Selection
            sorted_indices = np.argsort(fitness_scores)[::-1]
            population = [population[i] for i in sorted_indices[:population_size//2]]
            
            # Crossover and mutation
            new_population = []
            while len(new_population) < population_size:
                parent1, parent2 = np.random.choice(population, 2, replace=False)
                
                # Crossover
                crossover_point = np.random.randint(1, n_models)
                child = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
                
                # Mutation
                if np.random.random() < mutation_rate:
                    mutation_idx = np.random.randint(n_models)
                    child[mutation_idx] = np.random.random()
                
                # Normalize
                child = child / np.sum(child)
                new_population.append(child)
            
            population = new_population
        
        # Return best weights
        fitness_scores = [fitness(weights) for weights in population]
        best_idx = np.argmax(fitness_scores)
        best_weights = population[best_idx]
        
        logger.info(f"Genetic optimization completed. Best weights: {best_weights}, Score: {fitness_scores[best_idx]}")
        return best_weights
    
    def _uniform_weights(self, n_models: int) -> np.ndarray:
        """Return uniform weights"""
        return np.ones(n_models) / n_models

class ModelAgreementAnalyzer:
    """Analyzes agreement and disagreement between models"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agreement_threshold = config["ensemble"]["agreement_threshold"]
    
    def analyze_agreement(self, predictions: List[np.ndarray], labels: np.ndarray) -> Dict[str, Any]:
        """Analyze model agreement and disagreement"""
        n_models = len(predictions)
        n_samples = len(labels)
        
        # Convert to binary predictions
        binary_predictions = []
        for pred in predictions:
            binary_pred = (pred > 0.5).astype(int)
            binary_predictions.append(binary_pred)
        
        # Calculate agreement matrix
        agreement_matrix = np.zeros((n_samples, n_models))
        for i, pred in enumerate(binary_predictions):
            agreement_matrix[:, i] = pred
        
        # Calculate agreement statistics
        agreement_stats = {
            "total_samples": n_samples,
            "full_agreement": 0,
            "partial_agreement": 0,
            "no_agreement": 0,
            "agreement_rate": 0.0,
            "disagreement_samples": [],
            "model_agreement_pairs": {}
        }
        
        for i in range(n_samples):
            sample_predictions = agreement_matrix[i, :]
            unique_predictions = np.unique(sample_predictions)
            
            if len(unique_predictions) == 1:
                agreement_stats["full_agreement"] += 1
            elif len(unique_predictions) == 2:
                agreement_stats["partial_agreement"] += 1
                agreement_stats["disagreement_samples"].append(i)
            else:
                agreement_stats["no_agreement"] += 1
        
        agreement_stats["agreement_rate"] = agreement_stats["full_agreement"] / n_samples
        
        # Calculate pairwise agreement
        for i in range(n_models):
            for j in range(i + 1, n_models):
                pair_name = f"model_{i}_model_{j}"
                agreement = np.mean(binary_predictions[i] == binary_predictions[j])
                agreement_stats["model_agreement_pairs"][pair_name] = agreement
        
        return agreement_stats
    
    def identify_disagreement_cases(self, predictions: List[np.ndarray], labels: np.ndarray) -> List[int]:
        """Identify samples where models disagree"""
        n_models = len(predictions)
        n_samples = len(labels)
        
        # Convert to binary predictions
        binary_predictions = []
        for pred in predictions:
            binary_pred = (pred > 0.5).astype(int)
            binary_predictions.append(binary_pred)
        
        disagreement_cases = []
        
        for i in range(n_samples):
            sample_predictions = [pred[i] for pred in binary_predictions]
            unique_predictions = np.unique(sample_predictions)
            
            if len(unique_predictions) > 1:
                disagreement_cases.append(i)
        
        return disagreement_cases
    
    def resolve_disagreements(self, predictions: List[np.ndarray], 
                            disagreement_cases: List[int]) -> np.ndarray:
        """Resolve disagreements using various strategies"""
        n_models = len(predictions)
        n_samples = len(predictions[0])
        
        # Strategy 1: Confidence-weighted voting
        confidence_weights = []
        for pred in predictions:
            # Use prediction confidence as weight
            confidence = np.abs(pred - 0.5) * 2  # Convert to 0-1 scale
            confidence_weights.append(confidence)
        
        # Strategy 2: Majority voting with confidence threshold
        resolved_predictions = np.zeros(n_samples)
        
        for i in range(n_samples):
            if i in disagreement_cases:
                # Use confidence-weighted average
                weighted_sum = 0
                total_weight = 0
                
                for j, pred in enumerate(predictions):
                    weight = confidence_weights[j][i]
                    weighted_sum += pred[i] * weight
                    total_weight += weight
                
                if total_weight > 0:
                    resolved_predictions[i] = weighted_sum / total_weight
                else:
                    # Fallback to simple average
                    resolved_predictions[i] = np.mean([pred[i] for pred in predictions])
            else:
                # No disagreement, use simple average
                resolved_predictions[i] = np.mean([pred[i] for pred in predictions])
        
        return resolved_predictions

class EnsembleCalibrator:
    """Calibrates ensemble predictions for better calibration"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.calibration_method = "temperature_scaling"
        self.temperature = 1.0
    
    def calibrate_predictions(self, predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Calibrate ensemble predictions"""
        if self.calibration_method == "temperature_scaling":
            return self._temperature_scaling(predictions, labels)
        elif self.calibration_method == "platt_scaling":
            return self._platt_scaling(predictions, labels)
        else:
            return predictions
    
    def _temperature_scaling(self, predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Apply temperature scaling calibration"""
        # Optimize temperature parameter
        def objective(temperature):
            calibrated_pred = self._apply_temperature(predictions, temperature)
            return -roc_auc_score(labels, calibrated_pred)
        
        result = minimize(objective, [1.0], method='L-BFGS-B', bounds=[(0.1, 10.0)])
        
        if result.success:
            optimal_temperature = result.x[0]
            self.temperature = optimal_temperature
            logger.info(f"Temperature scaling optimized: T = {optimal_temperature:.3f}")
            return self._apply_temperature(predictions, optimal_temperature)
        else:
            logger.warning("Temperature scaling optimization failed")
            return predictions
    
    def _apply_temperature(self, predictions: np.ndarray, temperature: float) -> np.ndarray:
        """Apply temperature scaling to predictions"""
        # Apply temperature scaling: p_calibrated = sigmoid(logit(p) / T)
        logits = np.log(predictions / (1 - predictions + 1e-8))
        calibrated_logits = logits / temperature
        calibrated_pred = 1 / (1 + np.exp(-calibrated_logits))
        return calibrated_pred
    
    def _platt_scaling(self, predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Apply Platt scaling calibration"""
        from sklearn.linear_model import LogisticRegression
        
        # Fit logistic regression to calibrate
        lr = LogisticRegression()
        lr.fit(predictions.reshape(-1, 1), labels)
        
        # Apply calibration
        calibrated_pred = lr.predict_proba(predictions.reshape(-1, 1))[:, 1]
        return calibrated_pred

class EnsembleCoordinator:
    """Main ensemble training coordinator"""
    
    def __init__(self, config: EnsembleConfig, training_config: TrainingConfig,
                 experiment_config: ExperimentConfig, output_dir: str = "ensemble_outputs"):
        self.config = config
        self.training_config = training_config
        self.experiment_config = experiment_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.weight_optimizer = EnsembleWeightOptimizer(config.config)
        self.agreement_analyzer = ModelAgreementAnalyzer(config.config)
        self.calibrator = EnsembleCalibrator(config.config)
        
        # Training state
        self.models = {}
        self.ensemble_weights = None
        self.cross_validation_results = {}
        self.agreement_analysis = {}
        
        # Setup experiment tracking
        self.tracker = ExperimentTracker(experiment_config, str(self.output_dir))
    
    def train_individual_models(self, dataset_manager: DatasetManager, 
                              dataset_path: str) -> Dict[str, ModelTrainer]:
        """Train individual models in the ensemble"""
        models = {}
        
        for model_name in self.config.config["ensemble"]["models"]:
            logger.info(f"Training individual model: {model_name}")
            
            # Create trainer
            trainer = ModelTrainer(model_name, self.training_config, str(self.output_dir))
            
            # Get data loaders
            train_loader = dataset_manager.get_data_loader(
                dataset_path, "train", model_name=model_name
            )
            val_loader = dataset_manager.get_data_loader(
                dataset_path, "val", model_name=model_name, shuffle=False
            )
            
            # Train model
            results = trainer.train(train_loader, val_loader)
            models[model_name] = trainer
            
            # Log results
            self.tracker.log_metrics({
                f"{model_name}_final_accuracy": results["val_accuracies"][-1],
                f"{model_name}_best_accuracy": trainer.best_val_accuracy
            }, step=0, prefix="individual_training")
        
        self.models = models
        return models
    
    def perform_cross_validation(self, dataset_manager: DatasetManager, 
                               dataset_path: str) -> Dict[str, Any]:
        """Perform cross-validation for ensemble evaluation"""
        if not self.config.config["ensemble"]["cross_validation"]:
            return {}
        
        logger.info("Performing cross-validation...")
        
        # Load dataset
        dataset = dataset_manager.get_dataset(dataset_path)
        labels = [label for _, label in dataset.samples]
        
        # Setup cross-validation
        cv_folds = self.config.config["ensemble"]["cv_folds"]
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        cv_results = {
            "fold_results": [],
            "ensemble_weights": [],
            "agreement_analysis": []
        }
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(dataset)), labels)):
            logger.info(f"Cross-validation fold {fold + 1}/{cv_folds}")
            
            # Create fold-specific data loaders
            train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
            val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
            
            fold_results = {}
            
            # Train models on this fold
            for model_name in self.config.config["ensemble"]["models"]:
                trainer = ModelTrainer(model_name, self.training_config, str(self.output_dir))
                
                train_loader = DataLoader(
                    dataset,
                    batch_size=self.training_config.config["training"]["batch_size"],
                    sampler=train_sampler
                )
                val_loader = DataLoader(
                    dataset,
                    batch_size=self.training_config.config["training"]["batch_size"],
                    sampler=val_sampler
                )
                
                # Train model
                results = trainer.train(train_loader, val_loader)
                fold_results[model_name] = trainer
            
            # Evaluate ensemble on validation set
            ensemble_predictions = []
            val_labels = []
            
            for model_name, trainer in fold_results.items():
                model_predictions = []
                
                trainer.model.model.eval()
                with torch.no_grad():
                    for images, labels in val_loader:
                        images = images.to(self.training_config.device)
                        outputs = trainer.model.model(images)
                        model_predictions.extend(outputs.squeeze().cpu().numpy())
                        
                        if model_name == list(fold_results.keys())[0]:
                            val_labels.extend(labels.squeeze().cpu().numpy())
                
                ensemble_predictions.append(model_predictions)
            
            # Optimize ensemble weights
            optimal_weights = self.weight_optimizer.optimize_weights(
                ensemble_predictions, np.array(val_labels)
            )
            
            # Calculate ensemble performance
            ensemble_pred = np.zeros_like(ensemble_predictions[0])
            for i, pred in enumerate(ensemble_predictions):
                ensemble_pred += optimal_weights[i] * pred
            
            # Calculate metrics
            binary_pred = (ensemble_pred > 0.5).astype(int)
            accuracy = accuracy_score(val_labels, binary_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                val_labels, binary_pred, average='binary'
            )
            auc = roc_auc_score(val_labels, ensemble_pred)
            
            # Analyze agreement
            agreement_stats = self.agreement_analyzer.analyze_agreement(
                ensemble_predictions, np.array(val_labels)
            )
            
            # Store results
            cv_results["fold_results"].append({
                "fold": fold,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "auc": auc
            })
            
            cv_results["ensemble_weights"].append(optimal_weights)
            cv_results["agreement_analysis"].append(agreement_stats)
            
            # Log to experiment tracker
            self.tracker.log_metrics({
                f"cv_fold_{fold}_accuracy": accuracy,
                f"cv_fold_{fold}_auc": auc,
                f"cv_fold_{fold}_agreement_rate": agreement_stats["agreement_rate"]
            }, step=fold, prefix="cross_validation")
        
        self.cross_validation_results = cv_results
        
        # Calculate average results
        avg_accuracy = np.mean([r["accuracy"] for r in cv_results["fold_results"]])
        avg_auc = np.mean([r["auc"] for r in cv_results["fold_results"]])
        avg_agreement = np.mean([a["agreement_rate"] for a in cv_results["agreement_analysis"]])
        
        logger.info(f"Cross-validation completed:")
        logger.info(f"Average Accuracy: {avg_accuracy:.4f}")
        logger.info(f"Average AUC: {avg_auc:.4f}")
        logger.info(f"Average Agreement Rate: {avg_agreement:.4f}")
        
        return cv_results
    
    def optimize_ensemble_weights(self, test_loader: DataLoader) -> np.ndarray:
        """Optimize ensemble weights on test set"""
        logger.info("Optimizing ensemble weights...")
        
        # Get predictions from all models
        ensemble_predictions = []
        test_labels = []
        
        for model_name, trainer in self.models.items():
            model_predictions = []
            
            trainer.model.model.eval()
            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(self.training_config.device)
                    outputs = trainer.model.model(images)
                    model_predictions.extend(outputs.squeeze().cpu().numpy())
                    
                    if model_name == list(self.models.keys())[0]:
                        test_labels.extend(labels.squeeze().cpu().numpy())
            
            ensemble_predictions.append(model_predictions)
        
        # Optimize weights
        optimal_weights = self.weight_optimizer.optimize_weights(
            ensemble_predictions, np.array(test_labels)
        )
        
        self.ensemble_weights = optimal_weights
        
        # Log weights
        for i, model_name in enumerate(self.models.keys()):
            self.tracker.log_metrics({
                f"ensemble_weight_{model_name}": optimal_weights[i]
            }, step=0, prefix="ensemble_optimization")
        
        logger.info(f"Optimal ensemble weights: {optimal_weights}")
        return optimal_weights
    
    def analyze_model_agreement(self, test_loader: DataLoader) -> Dict[str, Any]:
        """Analyze model agreement and disagreement"""
        logger.info("Analyzing model agreement...")
        
        # Get predictions from all models
        ensemble_predictions = []
        test_labels = []
        
        for model_name, trainer in self.models.items():
            model_predictions = []
            
            trainer.model.model.eval()
            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(self.training_config.device)
                    outputs = trainer.model.model(images)
                    model_predictions.extend(outputs.squeeze().cpu().numpy())
                    
                    if model_name == list(self.models.keys())[0]:
                        test_labels.extend(labels.squeeze().cpu().numpy())
            
            ensemble_predictions.append(model_predictions)
        
        # Analyze agreement
        agreement_stats = self.agreement_analyzer.analyze_agreement(
            ensemble_predictions, np.array(test_labels)
        )
        
        # Identify disagreement cases
        disagreement_cases = self.agreement_analyzer.identify_disagreement_cases(
            ensemble_predictions, np.array(test_labels)
        )
        
        # Resolve disagreements
        resolved_predictions = self.agreement_analyzer.resolve_disagreements(
            ensemble_predictions, disagreement_cases
        )
        
        self.agreement_analysis = {
            "agreement_stats": agreement_stats,
            "disagreement_cases": disagreement_cases,
            "resolved_predictions": resolved_predictions
        }
        
        # Log agreement analysis
        self.tracker.log_metrics({
            "agreement_rate": agreement_stats["agreement_rate"],
            "full_agreement_rate": agreement_stats["full_agreement"] / agreement_stats["total_samples"],
            "disagreement_rate": len(disagreement_cases) / agreement_stats["total_samples"]
        }, step=0, prefix="agreement_analysis")
        
        logger.info(f"Agreement analysis completed:")
        logger.info(f"Agreement rate: {agreement_stats['agreement_rate']:.4f}")
        logger.info(f"Disagreement cases: {len(disagreement_cases)}")
        
        return self.agreement_analysis
    
    def calibrate_ensemble(self, val_loader: DataLoader) -> np.ndarray:
        """Calibrate ensemble predictions"""
        if not self.config.config["ensemble"]["calibration"]:
            return None
        
        logger.info("Calibrating ensemble predictions...")
        
        # Get ensemble predictions on validation set
        ensemble_predictions = []
        val_labels = []
        
        for model_name, trainer in self.models.items():
            model_predictions = []
            
            trainer.model.model.eval()
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(self.training_config.device)
                    outputs = trainer.model.model(images)
                    model_predictions.extend(outputs.squeeze().cpu().numpy())
                    
                    if model_name == list(self.models.keys())[0]:
                        val_labels.extend(labels.squeeze().cpu().numpy())
            
            ensemble_predictions.append(model_predictions)
        
        # Calculate weighted ensemble prediction
        if self.ensemble_weights is not None:
            ensemble_pred = np.zeros_like(ensemble_predictions[0])
            for i, pred in enumerate(ensemble_predictions):
                ensemble_pred += self.ensemble_weights[i] * pred
        else:
            ensemble_pred = np.mean(ensemble_predictions, axis=0)
        
        # Calibrate predictions
        calibrated_predictions = self.calibrator.calibrate_predictions(
            ensemble_pred, np.array(val_labels)
        )
        
        # Log calibration results
        original_auc = roc_auc_score(val_labels, ensemble_pred)
        calibrated_auc = roc_auc_score(val_labels, calibrated_predictions)
        
        self.tracker.log_metrics({
            "original_auc": original_auc,
            "calibrated_auc": calibrated_auc,
            "calibration_improvement": calibrated_auc - original_auc
        }, step=0, prefix="calibration")
        
        logger.info(f"Calibration completed:")
        logger.info(f"Original AUC: {original_auc:.4f}")
        logger.info(f"Calibrated AUC: {calibrated_auc:.4f}")
        
        return calibrated_predictions
    
    def evaluate_ensemble(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate final ensemble performance"""
        logger.info("Evaluating ensemble performance...")
        
        # Get predictions from all models
        ensemble_predictions = []
        test_labels = []
        
        for model_name, trainer in self.models.items():
            model_predictions = []
            
            trainer.model.model.eval()
            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(self.training_config.device)
                    outputs = trainer.model.model(images)
                    model_predictions.extend(outputs.squeeze().cpu().numpy())
                    
                    if model_name == list(self.models.keys())[0]:
                        test_labels.extend(labels.squeeze().cpu().numpy())
            
            ensemble_predictions.append(model_predictions)
        
        # Calculate ensemble prediction
        if self.ensemble_weights is not None:
            ensemble_pred = np.zeros_like(ensemble_predictions[0])
            for i, pred in enumerate(ensemble_predictions):
                ensemble_pred += self.ensemble_weights[i] * pred
        else:
            ensemble_pred = np.mean(ensemble_predictions, axis=0)
        
        # Calculate metrics
        binary_pred = (ensemble_pred > 0.5).astype(int)
        accuracy = accuracy_score(test_labels, binary_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_labels, binary_pred, average='binary'
        )
        auc = roc_auc_score(test_labels, ensemble_pred)
        
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc": auc
        }
        
        # Log final metrics
        self.tracker.log_metrics(metrics, step=0, prefix="final_evaluation")
        
        logger.info("Final ensemble evaluation:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def save_ensemble(self, save_path: str):
        """Save ensemble configuration and weights"""
        ensemble_config = {
            "models": list(self.models.keys()),
            "ensemble_weights": self.ensemble_weights.tolist() if self.ensemble_weights is not None else None,
            "cross_validation_results": self.cross_validation_results,
            "agreement_analysis": self.agreement_analysis,
            "calibration_temperature": self.calibrator.temperature,
            "config": self.config.config
        }
        
        with open(save_path, 'w') as f:
            json.dump(ensemble_config, f, indent=2)
        
        logger.info(f"Ensemble saved to: {save_path}")
    
    def close(self):
        """Close ensemble coordinator"""
        self.tracker.close()
        logger.info("Ensemble coordinator closed")

def main():
    """Main function for ensemble coordination"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ensemble Training Coordination")
    parser.add_argument("--config", help="Path to ensemble configuration file")
    parser.add_argument("--training-config", help="Path to training configuration file")
    parser.add_argument("--experiment-config", help="Path to experiment configuration file")
    parser.add_argument("--dataset", required=True, help="Path to dataset")
    parser.add_argument("--output-dir", default="ensemble_outputs", help="Output directory")
    parser.add_argument("--skip-individual", action="store_true", help="Skip individual model training")
    parser.add_argument("--skip-cv", action="store_true", help="Skip cross-validation")
    
    args = parser.parse_args()
    
    # Load configurations
    ensemble_config = EnsembleConfig(args.config)
    training_config = TrainingConfig(args.training_config)
    experiment_config = ExperimentConfig(args.experiment_config)
    
    # Setup dataset manager
    dataset_manager = DatasetManager()
    
    # Create ensemble coordinator
    coordinator = EnsembleCoordinator(
        ensemble_config, training_config, experiment_config, args.output_dir
    )
    
    try:
        # Train individual models
        if not args.skip_individual:
            coordinator.train_individual_models(dataset_manager, args.dataset)
        
        # Perform cross-validation
        if not args.skip_cv:
            coordinator.perform_cross_validation(dataset_manager, args.dataset)
        
        # Get test loader
        test_loader = dataset_manager.get_data_loader(
            args.dataset, "test", batch_size=training_config.config["training"]["batch_size"], 
            shuffle=False
        )
        
        # Optimize ensemble weights
        coordinator.optimize_ensemble_weights(test_loader)
        
        # Analyze model agreement
        coordinator.analyze_model_agreement(test_loader)
        
        # Calibrate ensemble
        val_loader = dataset_manager.get_data_loader(
            args.dataset, "val", batch_size=training_config.config["training"]["batch_size"], 
            shuffle=False
        )
        coordinator.calibrate_ensemble(val_loader)
        
        # Evaluate ensemble
        final_metrics = coordinator.evaluate_ensemble(test_loader)
        
        # Save ensemble
        ensemble_save_path = Path(args.output_dir) / "ensemble_config.json"
        coordinator.save_ensemble(str(ensemble_save_path))
        
        logger.info("Ensemble training and evaluation completed successfully!")
        
    finally:
        coordinator.close()

if __name__ == "__main__":
    main() 