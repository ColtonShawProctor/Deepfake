"""
Evaluation and Testing Framework for Deepfake Detection

This module provides comprehensive evaluation and testing including:
- Evaluation scripts for all models in the ensemble
- Cross-dataset evaluation protocols
- Performance benchmarking against existing models
- Model robustness testing with adversarial examples
- A/B testing framework for model comparisons
- Comprehensive evaluation metrics and reporting
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
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve,
    average_precision_score
)
from sklearn.model_selection import StratifiedKFold
import cv2
from PIL import Image, ImageFilter, ImageEnhance
import yaml
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add the app directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from training.dataset_management import DatasetManager
from training.model_trainer import ModelTrainer, TrainingConfig
from training.ensemble_coordinator import EnsembleCoordinator, EnsembleConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EvaluationConfig:
    """Configuration for evaluation framework"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load evaluation configuration"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        
        # Default configuration
        return {
            "evaluation": {
                "metrics": ["accuracy", "precision", "recall", "f1", "auc", "ap"],
                "confidence_thresholds": [0.1, 0.3, 0.5, 0.7, 0.9],
                "cross_validation_folds": 5,
                "random_seed": 42
            },
            "cross_dataset": {
                "enabled": True,
                "datasets": ["faceforensics", "celebdf", "dfdc"],
                "evaluation_metrics": ["accuracy", "auc", "f1"],
                "domain_adaptation": False
            },
            "robustness": {
                "enabled": True,
                "adversarial_attacks": ["fgsm", "pgd", "cw"],
                "noise_types": ["gaussian", "salt_pepper", "blur"],
                "compression_levels": [10, 30, 50, 70, 90],
                "brightness_levels": [0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
            },
            "benchmarking": {
                "enabled": True,
                "baseline_models": ["mesonet", "xception", "efficientnet"],
                "comparison_metrics": ["accuracy", "auc", "f1", "inference_time"],
                "statistical_significance": True
            },
            "ab_testing": {
                "enabled": True,
                "test_duration_days": 7,
                "sample_size": 1000,
                "confidence_level": 0.95,
                "min_detectable_effect": 0.05
            },
            "reporting": {
                "generate_reports": True,
                "save_plots": True,
                "export_formats": ["json", "csv", "html"],
                "report_template": "comprehensive"
            }
        }

class ModelEvaluator:
    """Evaluates individual models and ensembles"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.evaluation_results = {}
    
    def evaluate_model(self, model_trainer: ModelTrainer, test_loader: DataLoader,
                      model_name: str = "model") -> Dict[str, Any]:
        """Evaluate a single model"""
        logger.info(f"Evaluating model: {model_name}")
        
        model_trainer.model.model.eval()
        all_predictions = []
        all_labels = []
        all_confidences = []
        inference_times = []
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(test_loader):
                images = images.to(model_trainer.config.device)
                labels = labels.to(model_trainer.config.device)
                
                # Measure inference time
                start_time = time.time()
                outputs = model_trainer.model.model(images)
                inference_time = time.time() - start_time
                
                # Store results
                predictions = (outputs.squeeze() > 0.5).float().cpu().numpy()
                confidences = outputs.squeeze().cpu().numpy()
                
                all_predictions.extend(predictions)
                all_labels.extend(labels.squeeze().cpu().numpy())
                all_confidences.extend(confidences)
                inference_times.extend([inference_time / len(images)] * len(images))
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_labels, all_predictions, all_confidences)
        metrics["inference_time_mean"] = np.mean(inference_times)
        metrics["inference_time_std"] = np.std(inference_times)
        
        # Calculate metrics at different confidence thresholds
        threshold_metrics = self._calculate_threshold_metrics(all_labels, all_confidences)
        
        evaluation_result = {
            "model_name": model_name,
            "metrics": metrics,
            "threshold_metrics": threshold_metrics,
            "predictions": all_predictions,
            "confidences": all_confidences,
            "labels": all_labels,
            "inference_times": inference_times
        }
        
        self.evaluation_results[model_name] = evaluation_result
        return evaluation_result
    
    def _calculate_metrics(self, labels: List[float], predictions: List[float], 
                          confidences: List[float]) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics"""
        labels = np.array(labels)
        predictions = np.array(predictions)
        confidences = np.array(confidences)
        
        # Basic classification metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary'
        )
        
        # ROC and PR metrics
        auc = roc_auc_score(labels, confidences)
        ap = average_precision_score(labels, confidences)
        
        # Additional metrics
        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc": auc,
            "average_precision": ap,
            "specificity": specificity,
            "sensitivity": sensitivity,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp
        }
    
    def _calculate_threshold_metrics(self, labels: List[float], confidences: List[float]) -> Dict[str, Dict[str, float]]:
        """Calculate metrics at different confidence thresholds"""
        labels = np.array(labels)
        confidences = np.array(confidences)
        
        threshold_metrics = {}
        
        for threshold in self.config.config["evaluation"]["confidence_thresholds"]:
            predictions = (confidences > threshold).astype(int)
            
            accuracy = accuracy_score(labels, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, predictions, average='binary'
            )
            
            threshold_metrics[f"threshold_{threshold}"] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }
        
        return threshold_metrics

class CrossDatasetEvaluator:
    """Evaluates models across different datasets"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.cross_dataset_results = {}
    
    def evaluate_cross_dataset(self, model_trainer: ModelTrainer, 
                             dataset_manager: DatasetManager,
                             datasets: List[str]) -> Dict[str, Any]:
        """Evaluate model across multiple datasets"""
        logger.info("Starting cross-dataset evaluation")
        
        results = {}
        
        for dataset_name in datasets:
            logger.info(f"Evaluating on dataset: {dataset_name}")
            
            try:
                # Get dataset path
                dataset_path = f"datasets/{dataset_name}"
                
                # Create test loader
                test_loader = dataset_manager.get_data_loader(
                    dataset_path, "test", batch_size=32, shuffle=False
                )
                
                # Evaluate model
                evaluator = ModelEvaluator(self.config)
                evaluation_result = evaluator.evaluate_model(
                    model_trainer, test_loader, f"{model_trainer.model_name}_{dataset_name}"
                )
                
                results[dataset_name] = evaluation_result["metrics"]
                
            except Exception as e:
                logger.warning(f"Failed to evaluate on {dataset_name}: {e}")
                results[dataset_name] = {"error": str(e)}
        
        # Calculate cross-dataset statistics
        cross_dataset_stats = self._calculate_cross_dataset_stats(results)
        
        self.cross_dataset_results = {
            "dataset_results": results,
            "cross_dataset_stats": cross_dataset_stats
        }
        
        return self.cross_dataset_results
    
    def _calculate_cross_dataset_stats(self, dataset_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate statistics across datasets"""
        valid_results = {k: v for k, v in dataset_results.items() if "error" not in v}
        
        if not valid_results:
            return {"error": "No valid results"}
        
        stats = {}
        for metric in ["accuracy", "auc", "f1_score"]:
            values = [result[metric] for result in valid_results.values()]
            stats[f"{metric}_mean"] = np.mean(values)
            stats[f"{metric}_std"] = np.std(values)
            stats[f"{metric}_min"] = np.min(values)
            stats[f"{metric}_max"] = np.max(values)
        
        return stats

class RobustnessTester:
    """Tests model robustness against various perturbations"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.robustness_results = {}
    
    def test_robustness(self, model_trainer: ModelTrainer, test_loader: DataLoader) -> Dict[str, Any]:
        """Test model robustness against various perturbations"""
        logger.info("Starting robustness testing")
        
        results = {}
        
        # Test adversarial attacks
        if "fgsm" in self.config.config["robustness"]["adversarial_attacks"]:
            results["fgsm"] = self._test_fgsm_attack(model_trainer, test_loader)
        
        if "pgd" in self.config.config["robustness"]["adversarial_attacks"]:
            results["pgd"] = self._test_pgd_attack(model_trainer, test_loader)
        
        # Test noise perturbations
        for noise_type in self.config.config["robustness"]["noise_types"]:
            results[f"noise_{noise_type}"] = self._test_noise_robustness(
                model_trainer, test_loader, noise_type
            )
        
        # Test compression robustness
        results["compression"] = self._test_compression_robustness(model_trainer, test_loader)
        
        # Test brightness robustness
        results["brightness"] = self._test_brightness_robustness(model_trainer, test_loader)
        
        self.robustness_results = results
        return results
    
    def _test_fgsm_attack(self, model_trainer: ModelTrainer, test_loader: DataLoader) -> Dict[str, Any]:
        """Test FGSM adversarial attack"""
        logger.info("Testing FGSM attack")
        
        epsilon = 0.1
        results = {"epsilon": epsilon, "accuracy_drops": []}
        
        model_trainer.model.model.eval()
        
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(model_trainer.config.device)
            labels = labels.to(model_trainer.config.device)
            
            # Generate adversarial examples
            images.requires_grad = True
            
            outputs = model_trainer.model.model(images)
            loss = F.binary_cross_entropy(outputs.squeeze(), labels.squeeze())
            loss.backward()
            
            # FGSM attack
            perturbation = epsilon * images.grad.sign()
            adversarial_images = images + perturbation
            adversarial_images = torch.clamp(adversarial_images, 0, 1)
            
            # Test on adversarial examples
            with torch.no_grad():
                adv_outputs = model_trainer.model.model(adversarial_images)
                adv_predictions = (adv_outputs.squeeze() > 0.5).float()
                
                accuracy = (adv_predictions == labels.squeeze()).float().mean().item()
                results["accuracy_drops"].append(1.0 - accuracy)
        
        results["mean_accuracy_drop"] = np.mean(results["accuracy_drops"])
        return results
    
    def _test_pgd_attack(self, model_trainer: ModelTrainer, test_loader: DataLoader) -> Dict[str, Any]:
        """Test PGD adversarial attack"""
        logger.info("Testing PGD attack")
        
        epsilon = 0.1
        alpha = 0.01
        steps = 10
        results = {"epsilon": epsilon, "steps": steps, "accuracy_drops": []}
        
        model_trainer.model.model.eval()
        
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(model_trainer.config.device)
            labels = labels.to(model_trainer.config.device)
            
            # Initialize perturbation
            delta = torch.rand_like(images) * 2 * epsilon - epsilon
            
            for step in range(steps):
                delta.requires_grad = True
                
                outputs = model_trainer.model.model(images + delta)
                loss = F.binary_cross_entropy(outputs.squeeze(), labels.squeeze())
                loss.backward()
                
                # PGD update
                delta = delta + alpha * delta.grad.sign()
                delta = torch.clamp(delta, -epsilon, epsilon)
                delta = torch.clamp(images + delta, 0, 1) - images
            
            # Test on adversarial examples
            with torch.no_grad():
                adv_outputs = model_trainer.model.model(images + delta)
                adv_predictions = (adv_outputs.squeeze() > 0.5).float()
                
                accuracy = (adv_predictions == labels.squeeze()).float().mean().item()
                results["accuracy_drops"].append(1.0 - accuracy)
        
        results["mean_accuracy_drop"] = np.mean(results["accuracy_drops"])
        return results
    
    def _test_noise_robustness(self, model_trainer: ModelTrainer, test_loader: DataLoader,
                              noise_type: str) -> Dict[str, Any]:
        """Test robustness against noise"""
        logger.info(f"Testing {noise_type} noise robustness")
        
        noise_levels = [0.01, 0.05, 0.1, 0.15, 0.2]
        results = {"noise_type": noise_type, "levels": {}}
        
        model_trainer.model.model.eval()
        
        for noise_level in noise_levels:
            accuracies = []
            
            for batch_idx, (images, labels) in enumerate(test_loader):
                images = images.to(model_trainer.config.device)
                labels = labels.to(model_trainer.config.device)
                
                # Add noise
                if noise_type == "gaussian":
                    noise = torch.randn_like(images) * noise_level
                    noisy_images = torch.clamp(images + noise, 0, 1)
                elif noise_type == "salt_pepper":
                    noisy_images = images.clone()
                    mask = torch.rand_like(images) < noise_level
                    noisy_images[mask] = torch.randint(0, 2, (mask.sum(),), device=images.device).float()
                elif noise_type == "blur":
                    # Apply Gaussian blur
                    kernel_size = int(noise_level * 10) + 1
                    if kernel_size % 2 == 0:
                        kernel_size += 1
                    noisy_images = F.avg_pool2d(images, kernel_size, stride=1, padding=kernel_size//2)
                
                # Test on noisy images
                with torch.no_grad():
                    outputs = model_trainer.model.model(noisy_images)
                    predictions = (outputs.squeeze() > 0.5).float()
                    
                    accuracy = (predictions == labels.squeeze()).float().mean().item()
                    accuracies.append(accuracy)
            
            results["levels"][f"level_{noise_level}"] = np.mean(accuracies)
        
        return results
    
    def _test_compression_robustness(self, model_trainer: ModelTrainer, test_loader: DataLoader) -> Dict[str, Any]:
        """Test robustness against JPEG compression"""
        logger.info("Testing compression robustness")
        
        compression_levels = self.config.config["robustness"]["compression_levels"]
        results = {"compression_levels": {}}
        
        model_trainer.model.model.eval()
        
        for quality in compression_levels:
            accuracies = []
            
            for batch_idx, (images, labels) in enumerate(test_loader):
                images = images.to(model_trainer.config.device)
                labels = labels.to(model_trainer.config.device)
                
                # Apply JPEG compression simulation
                compressed_images = self._simulate_jpeg_compression(images, quality)
                
                # Test on compressed images
                with torch.no_grad():
                    outputs = model_trainer.model.model(compressed_images)
                    predictions = (outputs.squeeze() > 0.5).float()
                    
                    accuracy = (predictions == labels.squeeze()).float().mean().item()
                    accuracies.append(accuracy)
            
            results["compression_levels"][f"quality_{quality}"] = np.mean(accuracies)
        
        return results
    
    def _test_brightness_robustness(self, model_trainer: ModelTrainer, test_loader: DataLoader) -> Dict[str, Any]:
        """Test robustness against brightness changes"""
        logger.info("Testing brightness robustness")
        
        brightness_levels = self.config.config["robustness"]["brightness_levels"]
        results = {"brightness_levels": {}}
        
        model_trainer.model.model.eval()
        
        for level in brightness_levels:
            accuracies = []
            
            for batch_idx, (images, labels) in enumerate(test_loader):
                images = images.to(model_trainer.config.device)
                labels = labels.to(model_trainer.config.device)
                
                # Apply brightness adjustment
                bright_images = torch.clamp(images * level, 0, 1)
                
                # Test on brightness-adjusted images
                with torch.no_grad():
                    outputs = model_trainer.model.model(bright_images)
                    predictions = (outputs.squeeze() > 0.5).float()
                    
                    accuracy = (predictions == labels.squeeze()).float().mean().item()
                    accuracies.append(accuracy)
            
            results["brightness_levels"][f"level_{level}"] = np.mean(accuracies)
        
        return results
    
    def _simulate_jpeg_compression(self, images: torch.Tensor, quality: int) -> torch.Tensor:
        """Simulate JPEG compression"""
        # Convert to PIL images, compress, and convert back
        compressed_images = []
        
        for i in range(images.size(0)):
            # Convert to PIL
            img = images[i].cpu().numpy().transpose(1, 2, 0)
            img = (img * 255).astype(np.uint8)
            pil_img = Image.fromarray(img)
            
            # Simulate compression by saving and loading
            import io
            buffer = io.BytesIO()
            pil_img.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            compressed_pil = Image.open(buffer)
            
            # Convert back to tensor
            compressed_img = np.array(compressed_pil).astype(np.float32) / 255.0
            compressed_img = compressed_img.transpose(2, 0, 1)
            compressed_images.append(compressed_img)
        
        return torch.tensor(compressed_images, device=images.device)

class ABTestingFramework:
    """A/B testing framework for model comparison"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.ab_results = {}
    
    def run_ab_test(self, model_a: ModelTrainer, model_b: ModelTrainer,
                   test_loader: DataLoader, sample_size: int = None) -> Dict[str, Any]:
        """Run A/B test between two models"""
        logger.info("Starting A/B test")
        
        if sample_size is None:
            sample_size = self.config.config["ab_testing"]["sample_size"]
        
        # Sample data
        all_images = []
        all_labels = []
        
        for images, labels in test_loader:
            all_images.append(images)
            all_labels.append(labels)
            
            if len(all_images) * images.size(0) >= sample_size:
                break
        
        # Concatenate and sample
        all_images = torch.cat(all_images, dim=0)[:sample_size]
        all_labels = torch.cat(all_labels, dim=0)[:sample_size]
        
        # Evaluate both models
        evaluator = ModelEvaluator(self.config)
        
        # Create data loaders for sampled data
        sampled_dataset = torch.utils.data.TensorDataset(all_images, all_labels)
        sampled_loader = DataLoader(sampled_dataset, batch_size=32, shuffle=False)
        
        # Evaluate model A
        model_a.model.model.eval()
        results_a = evaluator.evaluate_model(model_a, sampled_loader, "model_a")
        
        # Evaluate model B
        model_b.model.model.eval()
        results_b = evaluator.evaluate_model(model_b, sampled_loader, "model_b")
        
        # Perform statistical significance test
        significance_results = self._statistical_significance_test(
            results_a["confidences"], results_b["confidences"], all_labels
        )
        
        # Calculate effect size
        effect_size = self._calculate_effect_size(
            results_a["metrics"]["accuracy"], results_b["metrics"]["accuracy"]
        )
        
        ab_result = {
            "model_a_results": results_a,
            "model_b_results": results_b,
            "statistical_significance": significance_results,
            "effect_size": effect_size,
            "sample_size": sample_size,
            "recommendation": self._generate_recommendation(significance_results, effect_size)
        }
        
        self.ab_results = ab_result
        return ab_result
    
    def _statistical_significance_test(self, confidences_a: List[float], 
                                     confidences_b: List[float], 
                                     labels: torch.Tensor) -> Dict[str, Any]:
        """Perform statistical significance test"""
        from scipy import stats
        
        # Convert to binary predictions
        predictions_a = (np.array(confidences_a) > 0.5).astype(int)
        predictions_b = (np.array(confidences_b) > 0.5).astype(int)
        
        # McNemar's test for paired samples
        mcnemar_stat, mcnemar_p = stats.mcnemar(
            confusion_matrix(labels, predictions_a, predictions_b)
        )
        
        # Paired t-test for confidence scores
        t_stat, t_p = stats.ttest_rel(confidences_a, confidences_b)
        
        return {
            "mcnemar_statistic": mcnemar_stat,
            "mcnemar_p_value": mcnemar_p,
            "t_statistic": t_stat,
            "t_p_value": t_p,
            "significant": t_p < (1 - self.config.config["ab_testing"]["confidence_level"])
        }
    
    def _calculate_effect_size(self, accuracy_a: float, accuracy_b: float) -> float:
        """Calculate Cohen's d effect size"""
        # Simplified effect size calculation
        return abs(accuracy_a - accuracy_b)
    
    def _generate_recommendation(self, significance_results: Dict[str, Any], 
                               effect_size: float) -> str:
        """Generate recommendation based on test results"""
        if significance_results["significant"]:
            if effect_size > self.config.config["ab_testing"]["min_detectable_effect"]:
                return "Model B shows significant improvement over Model A"
            else:
                return "Significant difference detected but effect size is small"
        else:
            return "No significant difference between models"

class EvaluationReporter:
    """Generates comprehensive evaluation reports"""
    
    def __init__(self, config: EvaluationConfig, output_dir: str = "evaluation_reports"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_comprehensive_report(self, evaluation_results: Dict[str, Any],
                                   cross_dataset_results: Dict[str, Any] = None,
                                   robustness_results: Dict[str, Any] = None,
                                   ab_test_results: Dict[str, Any] = None) -> str:
        """Generate comprehensive evaluation report"""
        logger.info("Generating comprehensive evaluation report")
        
        report_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"evaluation_report_{report_id}.html"
        
        # Generate HTML report
        html_content = self._generate_html_report(
            evaluation_results, cross_dataset_results, robustness_results, ab_test_results
        )
        
        with open(report_file, 'w') as f:
            f.write(html_content)
        
        # Generate plots
        self._generate_evaluation_plots(evaluation_results, report_id)
        
        # Export results in different formats
        self._export_results(evaluation_results, report_id)
        
        logger.info(f"Evaluation report generated: {report_file}")
        return str(report_file)
    
    def _generate_html_report(self, evaluation_results: Dict[str, Any],
                            cross_dataset_results: Dict[str, Any] = None,
                            robustness_results: Dict[str, Any] = None,
                            ab_test_results: Dict[str, Any] = None) -> str:
        """Generate HTML report"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Deepfake Detection Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background: #f5f5f5; }}
                .plot {{ margin: 20px 0; text-align: center; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Deepfake Detection Evaluation Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="section">
                <h2>Model Performance Summary</h2>
                {self._generate_performance_summary(evaluation_results)}
            </div>
        """
        
        if cross_dataset_results:
            html += f"""
            <div class="section">
                <h2>Cross-Dataset Evaluation</h2>
                {self._generate_cross_dataset_summary(cross_dataset_results)}
            </div>
            """
        
        if robustness_results:
            html += f"""
            <div class="section">
                <h2>Robustness Testing</h2>
                {self._generate_robustness_summary(robustness_results)}
            </div>
            """
        
        if ab_test_results:
            html += f"""
            <div class="section">
                <h2>A/B Testing Results</h2>
                {self._generate_ab_test_summary(ab_test_results)}
            </div>
            """
        
        html += """
        </body>
        </html>
        """
        
        return html
    
    def _generate_performance_summary(self, evaluation_results: Dict[str, Any]) -> str:
        """Generate performance summary HTML"""
        html = "<table><tr><th>Model</th><th>Accuracy</th><th>AUC</th><th>F1 Score</th><th>Precision</th><th>Recall</th></tr>"
        
        for model_name, results in evaluation_results.items():
            metrics = results["metrics"]
            html += f"""
            <tr>
                <td>{model_name}</td>
                <td>{metrics['accuracy']:.4f}</td>
                <td>{metrics['auc']:.4f}</td>
                <td>{metrics['f1_score']:.4f}</td>
                <td>{metrics['precision']:.4f}</td>
                <td>{metrics['recall']:.4f}</td>
            </tr>
            """
        
        html += "</table>"
        return html
    
    def _generate_cross_dataset_summary(self, cross_dataset_results: Dict[str, Any]) -> str:
        """Generate cross-dataset summary HTML"""
        html = "<h3>Cross-Dataset Performance</h3><table><tr><th>Dataset</th><th>Accuracy</th><th>AUC</th><th>F1 Score</th></tr>"
        
        for dataset_name, results in cross_dataset_results["dataset_results"].items():
            if "error" not in results:
                html += f"""
                <tr>
                    <td>{dataset_name}</td>
                    <td>{results['accuracy']:.4f}</td>
                    <td>{results['auc']:.4f}</td>
                    <td>{results['f1_score']:.4f}</td>
                </tr>
                """
        
        html += "</table>"
        return html
    
    def _generate_robustness_summary(self, robustness_results: Dict[str, Any]) -> str:
        """Generate robustness summary HTML"""
        html = "<h3>Robustness Test Results</h3>"
        
        for test_name, results in robustness_results.items():
            html += f"<h4>{test_name}</h4>"
            if "mean_accuracy_drop" in results:
                html += f"<p>Mean Accuracy Drop: {results['mean_accuracy_drop']:.4f}</p>"
            elif "levels" in results:
                html += "<ul>"
                for level, accuracy in results["levels"].items():
                    html += f"<li>{level}: {accuracy:.4f}</li>"
                html += "</ul>"
        
        return html
    
    def _generate_ab_test_summary(self, ab_test_results: Dict[str, Any]) -> str:
        """Generate A/B test summary HTML"""
        html = f"""
        <h3>A/B Test Results</h3>
        <p><strong>Recommendation:</strong> {ab_test_results['recommendation']}</p>
        <p><strong>Statistical Significance:</strong> {ab_test_results['statistical_significance']['significant']}</p>
        <p><strong>Effect Size:</strong> {ab_test_results['effect_size']:.4f}</p>
        <p><strong>Sample Size:</strong> {ab_test_results['sample_size']}</p>
        """
        return html
    
    def _generate_evaluation_plots(self, evaluation_results: Dict[str, Any], report_id: str):
        """Generate evaluation plots"""
        # ROC curves
        plt.figure(figsize=(10, 6))
        for model_name, results in evaluation_results.items():
            fpr, tpr, _ = roc_curve(results["labels"], results["confidences"])
            auc = results["metrics"]["auc"]
            plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.3f})")
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.output_dir / f"roc_curves_{report_id}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Precision-Recall curves
        plt.figure(figsize=(10, 6))
        for model_name, results in evaluation_results.items():
            precision, recall, _ = precision_recall_curve(results["labels"], results["confidences"])
            ap = results["metrics"]["average_precision"]
            plt.plot(recall, precision, label=f"{model_name} (AP = {ap:.3f})")
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.output_dir / f"pr_curves_{report_id}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _export_results(self, evaluation_results: Dict[str, Any], report_id: str):
        """Export results in different formats"""
        # JSON export
        json_file = self.output_dir / f"evaluation_results_{report_id}.json"
        with open(json_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        # CSV export
        csv_data = []
        for model_name, results in evaluation_results.items():
            row = {"model": model_name}
            row.update(results["metrics"])
            csv_data.append(row)
        
        df = pd.DataFrame(csv_data)
        csv_file = self.output_dir / f"evaluation_results_{report_id}.csv"
        df.to_csv(csv_file, index=False)

def main():
    """Main function for evaluation framework"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluation Framework for Deepfake Detection")
    parser.add_argument("--config", help="Path to evaluation configuration file")
    parser.add_argument("--model-path", required=True, help="Path to trained model")
    parser.add_argument("--dataset", required=True, help="Path to test dataset")
    parser.add_argument("--output-dir", default="evaluation_reports", help="Output directory")
    parser.add_argument("--cross-dataset", action="store_true", help="Run cross-dataset evaluation")
    parser.add_argument("--robustness", action="store_true", help="Run robustness testing")
    parser.add_argument("--ab-test", help="Path to second model for A/B testing")
    
    args = parser.parse_args()
    
    # Load configuration
    config = EvaluationConfig(args.config)
    
    # Setup components
    dataset_manager = DatasetManager()
    test_loader = dataset_manager.get_data_loader(args.dataset, "test", batch_size=32, shuffle=False)
    
    # Load model
    training_config = TrainingConfig()
    model_trainer = ModelTrainer("efficientnet", training_config)  # Adjust model type as needed
    model_trainer.load_checkpoint(args.model_path)
    
    # Run evaluations
    evaluator = ModelEvaluator(config)
    evaluation_results = evaluator.evaluate_model(model_trainer, test_loader, "test_model")
    
    cross_dataset_results = None
    if args.cross_dataset:
        cross_evaluator = CrossDatasetEvaluator(config)
        cross_dataset_results = cross_evaluator.evaluate_cross_dataset(
            model_trainer, dataset_manager, config.config["cross_dataset"]["datasets"]
        )
    
    robustness_results = None
    if args.robustness:
        robustness_tester = RobustnessTester(config)
        robustness_results = robustness_tester.test_robustness(model_trainer, test_loader)
    
    ab_test_results = None
    if args.ab_test:
        # Load second model for A/B testing
        model_trainer_b = ModelTrainer("efficientnet", training_config)
        model_trainer_b.load_checkpoint(args.ab_test)
        
        ab_tester = ABTestingFramework(config)
        ab_test_results = ab_tester.run_ab_test(model_trainer, model_trainer_b, test_loader)
    
    # Generate report
    reporter = EvaluationReporter(config, args.output_dir)
    report_file = reporter.generate_comprehensive_report(
        evaluation_results, cross_dataset_results, robustness_results, ab_test_results
    )
    
    logger.info(f"Evaluation completed. Report saved to: {report_file}")

if __name__ == "__main__":
    main() 