"""
Experiment Tracking and Monitoring System

This module provides comprehensive experiment tracking including:
- TensorBoard integration for real-time metrics visualization
- Model performance comparison tools
- Training progress monitoring and logging
- Validation metric tracking and best model selection
- Experiment configuration management
- Performance benchmarking and analysis
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
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import yaml
import hashlib
import uuid

# TensorBoard and MLflow imports
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ExperimentConfig:
    """Configuration for experiment tracking"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.experiment_id = self._generate_experiment_id()
        self.start_time = datetime.now()
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load experiment configuration"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        
        # Default configuration
        return {
            "experiment": {
                "name": "deepfake_detection_experiment",
                "description": "Multi-model deepfake detection ensemble training",
                "tags": ["deepfake", "ensemble", "computer_vision"],
                "enable_tensorboard": True,
                "enable_mlflow": False,
                "save_artifacts": True,
                "log_hyperparameters": True
            },
            "monitoring": {
                "log_interval": 10,
                "save_interval": 100,
                "validation_interval": 1,
                "early_stopping_patience": 10,
                "metric_tracking": ["loss", "accuracy", "precision", "recall", "f1", "auc"]
            },
            "visualization": {
                "plot_training_curves": True,
                "plot_confusion_matrix": True,
                "plot_roc_curve": True,
                "plot_precision_recall": True,
                "save_plots": True
            }
        }
    
    def _generate_experiment_id(self) -> str:
        """Generate unique experiment ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_id = str(uuid.uuid4())[:8]
        return f"{timestamp}_{random_id}"

class ExperimentTracker:
    """Main experiment tracking class"""
    
    def __init__(self, config: ExperimentConfig, output_dir: str = "experiments"):
        self.config = config
        self.output_dir = Path(output_dir) / config.experiment_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tracking components
        self.writer = None
        self.mlflow_run = None
        self.metrics_history = {}
        self.best_metrics = {}
        self.experiment_metadata = {}
        
        self._setup_tracking()
        self._save_experiment_config()
    
    def _setup_tracking(self):
        """Setup tracking components"""
        # Setup TensorBoard
        if self.config.config["experiment"]["enable_tensorboard"] and TENSORBOARD_AVAILABLE:
            log_dir = self.output_dir / "tensorboard_logs"
            self.writer = SummaryWriter(log_dir=str(log_dir))
            logger.info(f"TensorBoard logging enabled: {log_dir}")
        
        # Setup MLflow
        if self.config.config["experiment"]["enable_mlflow"] and MLFLOW_AVAILABLE:
            try:
                mlflow.set_experiment(self.config.config["experiment"]["name"])
                self.mlflow_run = mlflow.start_run()
                logger.info("MLflow tracking enabled")
            except Exception as e:
                logger.warning(f"Failed to setup MLflow: {e}")
    
    def _save_experiment_config(self):
        """Save experiment configuration"""
        config_file = self.output_dir / "experiment_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(self.config.config, f, default_flow_style=False)
        
        # Save metadata
        metadata = {
            "experiment_id": self.config.experiment_id,
            "start_time": self.config.start_time.isoformat(),
            "config_hash": self._get_config_hash()
        }
        
        metadata_file = self.output_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _get_config_hash(self) -> str:
        """Generate hash of configuration for reproducibility"""
        config_str = json.dumps(self.config.config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def log_hyperparameters(self, hyperparams: Dict[str, Any]):
        """Log hyperparameters"""
        self.experiment_metadata["hyperparameters"] = hyperparams
        
        # Log to TensorBoard
        if self.writer:
            for key, value in hyperparams.items():
                self.writer.add_text(f"Hyperparameters/{key}", str(value), 0)
        
        # Log to MLflow
        if self.mlflow_run:
            try:
                mlflow.log_params(hyperparams)
            except Exception as e:
                logger.warning(f"Failed to log hyperparameters to MLflow: {e}")
        
        # Save to file
        hyperparams_file = self.output_dir / "hyperparameters.json"
        with open(hyperparams_file, 'w') as f:
            json.dump(hyperparams, f, indent=2)
    
    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = ""):
        """Log metrics for current step"""
        # Store in history
        if prefix not in self.metrics_history:
            self.metrics_history[prefix] = {}
        
        for metric_name, value in metrics.items():
            full_name = f"{prefix}_{metric_name}" if prefix else metric_name
            
            if full_name not in self.metrics_history[prefix]:
                self.metrics_history[prefix][full_name] = []
            
            self.metrics_history[prefix][full_name].append({
                "step": step,
                "value": value,
                "timestamp": time.time()
            })
            
            # Update best metrics
            if full_name not in self.best_metrics:
                self.best_metrics[full_name] = {"value": value, "step": step}
            else:
                # For loss metrics, lower is better
                if "loss" in full_name.lower():
                    if value < self.best_metrics[full_name]["value"]:
                        self.best_metrics[full_name] = {"value": value, "step": step}
                else:
                    # For other metrics, higher is better
                    if value > self.best_metrics[full_name]["value"]:
                        self.best_metrics[full_name] = {"value": value, "step": step}
        
        # Log to TensorBoard
        if self.writer:
            for metric_name, value in metrics.items():
                full_name = f"{prefix}/{metric_name}" if prefix else metric_name
                self.writer.add_scalar(full_name, value, step)
        
        # Log to MLflow
        if self.mlflow_run:
            try:
                mlflow.log_metrics(metrics, step=step)
            except Exception as e:
                logger.warning(f"Failed to log metrics to MLflow: {e}")
    
    def log_model_weights(self, model, step: int, model_name: str = "model"):
        """Log model weights distribution"""
        if self.writer:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.writer.add_histogram(f"{model_name}/{name}", param.data, step)
                    if param.grad is not None:
                        self.writer.add_histogram(f"{model_name}/{name}_grad", param.grad, step)
    
    def log_images(self, images: torch.Tensor, step: int, tag: str = "images", max_images: int = 8):
        """Log images to TensorBoard"""
        if self.writer and images is not None:
            # Limit number of images
            if images.size(0) > max_images:
                images = images[:max_images]
            
            self.writer.add_images(tag, images, step)
    
    def log_text(self, text: str, step: int, tag: str = "text"):
        """Log text to TensorBoard"""
        if self.writer:
            self.writer.add_text(tag, text, step)
    
    def save_checkpoint(self, checkpoint: Dict[str, Any], step: int, is_best: bool = False):
        """Save model checkpoint with experiment tracking"""
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Add experiment metadata to checkpoint
        checkpoint["experiment_id"] = self.config.experiment_id
        checkpoint["step"] = step
        checkpoint["timestamp"] = time.time()
        checkpoint["best_metrics"] = self.best_metrics
        
        # Save checkpoint
        checkpoint_path = checkpoint_dir / f"checkpoint_step_{step}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_checkpoint_path = checkpoint_dir / "best_checkpoint.pth"
            shutil.copy(checkpoint_path, best_checkpoint_path)
        
        # Log to MLflow
        if self.mlflow_run:
            try:
                mlflow.log_artifact(str(checkpoint_path))
                if is_best:
                    mlflow.log_artifact(str(best_checkpoint_path))
            except Exception as e:
                logger.warning(f"Failed to log checkpoint to MLflow: {e}")
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """Plot training curves from logged metrics"""
        if not self.metrics_history:
            logger.warning("No metrics history to plot")
            return
        
        # Create subplots
        num_metrics = len(self.metrics_history.get("", {}))
        if num_metrics == 0:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        metric_idx = 0
        for prefix, metrics in self.metrics_history.items():
            for metric_name, history in metrics.items():
                if metric_idx >= len(axes):
                    break
                
                steps = [entry["step"] for entry in history]
                values = [entry["value"] for entry in history]
                
                axes[metric_idx].plot(steps, values, label=metric_name)
                axes[metric_idx].set_title(f"{metric_name}")
                axes[metric_idx].set_xlabel("Step")
                axes[metric_idx].set_ylabel("Value")
                axes[metric_idx].grid(True)
                axes[metric_idx].legend()
                
                metric_idx += 1
        
        # Hide unused subplots
        for i in range(metric_idx, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            save_path = self.output_dir / "training_curves.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
        
        # Log to TensorBoard
        if self.writer:
            self.writer.add_figure("Training_Curves", plt.gcf())
    
    def plot_confusion_matrix(self, y_true: List[int], y_pred: List[int], 
                            save_path: Optional[str] = None):
        """Plot confusion matrix"""
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            save_path = self.output_dir / "confusion_matrix.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
        
        # Log to TensorBoard
        if self.writer:
            self.writer.add_figure("Confusion_Matrix", plt.gcf())
    
    def plot_roc_curve(self, y_true: List[int], y_scores: List[float], 
                      save_path: Optional[str] = None):
        """Plot ROC curve"""
        from sklearn.metrics import roc_curve, auc
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            save_path = self.output_dir / "roc_curve.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
        
        # Log to TensorBoard
        if self.writer:
            self.writer.add_figure("ROC_Curve", plt.gcf())
    
    def plot_precision_recall_curve(self, y_true: List[int], y_scores: List[float], 
                                   save_path: Optional[str] = None):
        """Plot Precision-Recall curve"""
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        avg_precision = average_precision_score(y_true, y_scores)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, 
                label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            save_path = self.output_dir / "precision_recall_curve.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
        
        # Log to TensorBoard
        if self.writer:
            self.writer.add_figure("Precision_Recall_Curve", plt.gcf())
    
    def generate_experiment_report(self) -> Dict[str, Any]:
        """Generate comprehensive experiment report"""
        report = {
            "experiment_id": self.config.experiment_id,
            "start_time": self.config.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "duration": (datetime.now() - self.config.start_time).total_seconds(),
            "best_metrics": self.best_metrics,
            "config": self.config.config,
            "summary": {}
        }
        
        # Calculate summary statistics
        for prefix, metrics in self.metrics_history.items():
            for metric_name, history in metrics.items():
                values = [entry["value"] for entry in history]
                report["summary"][metric_name] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "final": values[-1] if values else None
                }
        
        # Save report
        report_file = self.output_dir / "experiment_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def compare_experiments(self, experiment_ids: List[str], 
                          metrics_to_compare: List[str] = None) -> pd.DataFrame:
        """Compare multiple experiments"""
        if metrics_to_compare is None:
            metrics_to_compare = ["accuracy", "precision", "recall", "f1_score", "auc"]
        
        comparison_data = []
        
        for exp_id in experiment_ids:
            exp_dir = Path("experiments") / exp_id
            report_file = exp_dir / "experiment_report.json"
            
            if report_file.exists():
                with open(report_file, 'r') as f:
                    report = json.load(f)
                
                row = {"experiment_id": exp_id}
                
                # Add best metrics
                for metric in metrics_to_compare:
                    if metric in report["best_metrics"]:
                        row[f"best_{metric}"] = report["best_metrics"][metric]["value"]
                
                # Add final metrics
                for metric in metrics_to_compare:
                    if metric in report["summary"]:
                        row[f"final_{metric}"] = report["summary"][metric]["final"]
                
                comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Save comparison
        comparison_file = self.output_dir / "experiment_comparison.csv"
        df.to_csv(comparison_file, index=False)
        
        return df
    
    def plot_experiment_comparison(self, df: pd.DataFrame, 
                                 metrics: List[str] = None) -> None:
        """Plot experiment comparison"""
        if metrics is None:
            metrics = ["accuracy", "precision", "recall", "f1_score", "auc"]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            if i >= len(axes):
                break
            
            best_col = f"best_{metric}"
            final_col = f"final_{metric}"
            
            if best_col in df.columns and final_col in df.columns:
                x = np.arange(len(df))
                width = 0.35
                
                axes[i].bar(x - width/2, df[best_col], width, label='Best', alpha=0.8)
                axes[i].bar(x + width/2, df[final_col], width, label='Final', alpha=0.8)
                
                axes[i].set_xlabel('Experiment')
                axes[i].set_ylabel(metric.capitalize())
                axes[i].set_title(f'{metric.capitalize()} Comparison')
                axes[i].set_xticks(x)
                axes[i].set_xticklabels(df['experiment_id'], rotation=45)
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(metrics), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        comparison_plot_file = self.output_dir / "experiment_comparison.png"
        plt.savefig(comparison_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def close(self):
        """Close experiment tracking"""
        if self.writer:
            self.writer.close()
        
        if self.mlflow_run:
            try:
                mlflow.end_run()
            except Exception as e:
                logger.warning(f"Failed to end MLflow run: {e}")
        
        # Generate final report
        self.generate_experiment_report()
        
        logger.info(f"Experiment {self.config.experiment_id} completed")

class PerformanceMonitor:
    """Real-time performance monitoring"""
    
    def __init__(self, tracker: ExperimentTracker):
        self.tracker = tracker
        self.monitoring_data = []
    
    def log_system_metrics(self, step: int):
        """Log system performance metrics"""
        import psutil
        
        metrics = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage_percent": psutil.disk_usage('/').percent
        }
        
        if torch.cuda.is_available():
            metrics["gpu_memory_used"] = torch.cuda.memory_allocated() / 1024**3  # GB
            metrics["gpu_memory_cached"] = torch.cuda.memory_reserved() / 1024**3  # GB
        
        self.tracker.log_metrics(metrics, step, prefix="system")
    
    def log_training_speed(self, step: int, batch_time: float, 
                          samples_per_sec: float):
        """Log training speed metrics"""
        metrics = {
            "batch_time": batch_time,
            "samples_per_sec": samples_per_sec
        }
        
        self.tracker.log_metrics(metrics, step, prefix="speed")

def main():
    """Main function for experiment tracking"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Experiment Tracking for Deepfake Detection")
    parser.add_argument("--action", choices=["track", "compare", "report"],
                       required=True, help="Action to perform")
    parser.add_argument("--config", help="Path to experiment configuration file")
    parser.add_argument("--experiment-id", help="Experiment ID for comparison/report")
    parser.add_argument("--experiment-ids", nargs="+", help="Experiment IDs to compare")
    parser.add_argument("--output-dir", default="experiments", help="Output directory")
    
    args = parser.parse_args()
    
    if args.action == "track":
        # Create new experiment tracker
        config = ExperimentConfig(args.config)
        tracker = ExperimentTracker(config, args.output_dir)
        
        logger.info(f"Created experiment tracker: {config.experiment_id}")
        logger.info(f"Output directory: {tracker.output_dir}")
        
        # Example usage
        tracker.log_hyperparameters({
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100
        })
        
        # Simulate some metrics
        for step in range(100):
            metrics = {
                "loss": 1.0 - step * 0.01,
                "accuracy": step * 0.01,
                "precision": 0.5 + step * 0.005
            }
            tracker.log_metrics(metrics, step)
        
        tracker.close()
    
    elif args.action == "compare":
        if not args.experiment_ids:
            logger.error("Experiment IDs required for comparison")
            return
        
        config = ExperimentConfig(args.config)
        tracker = ExperimentTracker(config, args.output_dir)
        
        df = tracker.compare_experiments(args.experiment_ids)
        tracker.plot_experiment_comparison(df)
        
        logger.info("Experiment comparison completed")
        logger.info(f"Results saved to: {tracker.output_dir}")
    
    elif args.action == "report":
        if not args.experiment_id:
            logger.error("Experiment ID required for report")
            return
        
        exp_dir = Path(args.output_dir) / args.experiment_id
        report_file = exp_dir / "experiment_report.json"
        
        if report_file.exists():
            with open(report_file, 'r') as f:
                report = json.load(f)
            
            print("Experiment Report:")
            print(f"ID: {report['experiment_id']}")
            print(f"Duration: {report['duration']:.2f} seconds")
            print("\nBest Metrics:")
            for metric, data in report['best_metrics'].items():
                print(f"  {metric}: {data['value']:.4f} (step {data['step']})")
        else:
            logger.error(f"Report not found: {report_file}")

if __name__ == "__main__":
    main() 