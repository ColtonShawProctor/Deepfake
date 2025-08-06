"""
Model Training Infrastructure for Deepfake Detection

This module provides comprehensive training infrastructure including:
- Training scripts for each model in the ensemble
- GPU/CPU detection and resource optimization
- Mixed precision training (FP16) for memory efficiency
- Checkpoint saving and resuming functionality
- Learning rate schedulers and optimization strategies
- Early stopping and model selection criteria
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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import yaml

# Add the app directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from app.models.deepfake_models import (
    ResNetDetector, EfficientNetDetector, F3NetDetector
)
from app.models.mesonet_detector import MesoNetDetector
from app.models.xception_detector import XceptionDetector
from training.dataset_management import DatasetManager, DeepfakeDataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrainingConfig:
    """Configuration for model training"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.device = self._setup_device()
        self.mixed_precision = self._check_mixed_precision()
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load training configuration"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        
        # Default configuration
        return {
            "training": {
                "epochs": 100,
                "batch_size": 32,
                "learning_rate": 0.001,
                "weight_decay": 1e-4,
                "gradient_clip": 1.0,
                "early_stopping_patience": 10,
                "save_best_only": True,
                "mixed_precision": True,
                "num_workers": 4,
                "pin_memory": True
            },
            "optimization": {
                "optimizer": "adam",
                "scheduler": "cosine_annealing",
                "warmup_epochs": 5,
                "lr_decay": 0.1,
                "min_lr": 1e-6
            },
            "models": {
                "mesonet": {
                    "input_size": (256, 256),
                    "batch_size": 64,
                    "learning_rate": 0.001
                },
                "xception": {
                    "input_size": (299, 299),
                    "batch_size": 16,
                    "learning_rate": 0.0001
                },
                "efficientnet": {
                    "input_size": (224, 224),
                    "batch_size": 32,
                    "learning_rate": 0.0001
                },
                "f3net": {
                    "input_size": (224, 224),
                    "batch_size": 16,
                    "learning_rate": 0.0001
                }
            }
        }
    
    def _setup_device(self) -> torch.device:
        """Setup training device"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU")
        
        return device
    
    def _check_mixed_precision(self) -> bool:
        """Check if mixed precision training is available"""
        if not torch.cuda.is_available():
            return False
        
        # Check if AMP is available
        try:
            from torch.cuda.amp import autocast
            return True
        except ImportError:
            return False

class ModelTrainer:
    """Base trainer class for deepfake detection models"""
    
    def __init__(self, model_name: str, config: TrainingConfig, 
                 output_dir: str = "training_outputs"):
        self.model_name = model_name
        self.config = config
        self.output_dir = Path(output_dir) / model_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging and tensorboard
        self.writer = SummaryWriter(log_dir=str(self.output_dir / "logs"))
        self.logger = self._setup_logger()
        
        # Training state
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.criterion = None
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0
        self.epochs_without_improvement = 0
        
        # Initialize model and training components
        self._setup_model()
        self._setup_training_components()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup model-specific logger"""
        logger = logging.getLogger(f"{__name__}.{self.model_name}")
        
        # Create file handler
        log_file = self.output_dir / "training.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)
        
        return logger
    
    def _setup_model(self):
        """Setup model architecture"""
        model_config = self.config.config["models"].get(self.model_name, {})
        
        if self.model_name == "mesonet":
            self.model = MesoNetDetector(device=str(self.config.device))
        elif self.model_name == "xception":
            self.model = XceptionDetector(device=str(self.config.device))
        elif self.model_name == "efficientnet":
            self.model = EfficientNetDetector(device=str(self.config.device))
        elif self.model_name == "f3net":
            self.model = F3NetDetector(device=str(self.config.device))
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        
        # Move model to device
        self.model.model = self.model.model.to(self.config.device)
        
        # Enable mixed precision if available
        if self.config.mixed_precision and self.config.config["training"]["mixed_precision"]:
            self.scaler = GradScaler()
        
        self.logger.info(f"Initialized {self.model_name} model")
    
    def _setup_training_components(self):
        """Setup optimizer, scheduler, and loss function"""
        training_config = self.config.config["training"]
        model_config = self.config.config["models"].get(self.model_name, {})
        
        # Setup optimizer
        lr = model_config.get("learning_rate", training_config["learning_rate"])
        weight_decay = training_config["weight_decay"]
        
        if self.config.config["optimization"]["optimizer"] == "adam":
            self.optimizer = optim.Adam(
                self.model.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif self.config.config["optimization"]["optimizer"] == "adamw":
            self.optimizer = optim.AdamW(
                self.model.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif self.config.config["optimization"]["optimizer"] == "sgd":
            self.optimizer = optim.SGD(
                self.model.model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay
            )
        
        # Setup scheduler
        scheduler_config = self.config.config["optimization"]["scheduler"]
        if scheduler_config == "cosine_annealing":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=training_config["epochs"],
                eta_min=self.config.config["optimization"]["min_lr"]
            )
        elif scheduler_config == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=self.config.config["optimization"]["lr_decay"]
            )
        elif scheduler_config == "reduce_on_plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.config["optimization"]["lr_decay"],
                patience=5,
                min_lr=self.config.config["optimization"]["min_lr"]
            )
        
        # Setup loss function
        self.criterion = nn.BCELoss()
        
        self.logger.info(f"Setup training components: optimizer={self.config.config['optimization']['optimizer']}, "
                        f"scheduler={scheduler_config}, lr={lr}")
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.config.device)
            labels = labels.to(self.config.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.scaler is not None:
                with autocast():
                    outputs = self.model.model(images)
                    loss = self.criterion(outputs.squeeze(), labels.squeeze())
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.config["training"]["gradient_clip"] > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.model.parameters(),
                        self.config.config["training"]["gradient_clip"]
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model.model(images)
                loss = self.criterion(outputs.squeeze(), labels.squeeze())
                loss.backward()
                
                # Gradient clipping
                if self.config.config["training"]["gradient_clip"] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.model.parameters(),
                        self.config.config["training"]["gradient_clip"]
                    )
                
                self.optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy
            predicted = (outputs.squeeze() > 0.5).float()
            correct += (predicted == labels.squeeze()).sum().item()
            total += labels.size(0)
            
            # Log progress
            if batch_idx % 10 == 0:
                self.logger.info(f"Batch {batch_idx}/{len(train_loader)}: "
                               f"Loss={loss.item():.4f}, "
                               f"Accuracy={100.0 * correct / total:.2f}%")
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float, Dict[str, float]]:
        """Validate for one epoch"""
        self.model.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.config.device)
                labels = labels.to(self.config.device)
                
                if self.scaler is not None:
                    with autocast():
                        outputs = self.model.model(images)
                        loss = self.criterion(outputs.squeeze(), labels.squeeze())
                else:
                    outputs = self.model.model(images)
                    loss = self.criterion(outputs.squeeze(), labels.squeeze())
                
                total_loss += loss.item()
                
                # Calculate accuracy
                predicted = (outputs.squeeze() > 0.5).float()
                correct += (predicted == labels.squeeze()).sum().item()
                total += labels.size(0)
                
                # Store predictions for metrics
                all_predictions.extend(outputs.squeeze().cpu().numpy())
                all_labels.extend(labels.squeeze().cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total
        
        # Calculate additional metrics
        metrics = self._calculate_metrics(all_predictions, all_labels)
        
        return avg_loss, accuracy, metrics
    
    def _calculate_metrics(self, predictions: List[float], labels: List[float]) -> Dict[str, float]:
        """Calculate additional evaluation metrics"""
        predictions = np.array(predictions)
        labels = np.array(labels)
        
        # Convert to binary predictions
        binary_predictions = (predictions > 0.5).astype(int)
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, binary_predictions, average='binary'
        )
        
        auc = roc_auc_score(labels, predictions)
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc": auc
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            "best_val_loss": self.best_val_loss,
            "best_val_accuracy": self.best_val_accuracy,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_accuracies": self.train_accuracies,
            "val_accuracies": self.val_accuracies,
            "config": self.config.config
        }
        
        # Save latest checkpoint
        checkpoint_path = self.output_dir / "latest_checkpoint.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint if requested
        if is_best:
            best_checkpoint_path = self.output_dir / "best_checkpoint.pth"
            shutil.copy(checkpoint_path, best_checkpoint_path)
            self.logger.info(f"Saved best checkpoint to {best_checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        
        self.model.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if checkpoint["scheduler_state_dict"] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        if checkpoint["scaler_state_dict"] and self.scaler:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        self.best_val_loss = checkpoint["best_val_loss"]
        self.best_val_accuracy = checkpoint["best_val_accuracy"]
        self.train_losses = checkpoint["train_losses"]
        self.val_losses = checkpoint["val_losses"]
        self.train_accuracies = checkpoint["train_accuracies"]
        self.val_accuracies = checkpoint["val_accuracies"]
        
        self.logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint["epoch"]
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              start_epoch: int = 0) -> Dict[str, List[float]]:
        """Main training loop"""
        training_config = self.config.config["training"]
        epochs = training_config["epochs"]
        
        self.logger.info(f"Starting training for {epochs} epochs")
        
        for epoch in range(start_epoch, epochs):
            epoch_start_time = time.time()
            
            # Training phase
            train_loss, train_accuracy = self.train_epoch(train_loader)
            
            # Validation phase
            val_loss, val_accuracy, val_metrics = self.validate_epoch(val_loader)
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Record metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_accuracy)
            self.val_accuracies.append(val_accuracy)
            
            # Log metrics
            current_lr = self.optimizer.param_groups[0]['lr']
            epoch_time = time.time() - epoch_start_time
            
            self.logger.info(f"Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s): "
                           f"Train Loss={train_loss:.4f}, Train Acc={train_accuracy:.2f}%, "
                           f"Val Loss={val_loss:.4f}, Val Acc={val_accuracy:.2f}%, "
                           f"LR={current_lr:.6f}")
            
            # Log to tensorboard
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Validation', val_loss, epoch)
            self.writer.add_scalar('Accuracy/Train', train_accuracy, epoch)
            self.writer.add_scalar('Accuracy/Validation', val_accuracy, epoch)
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            for metric_name, metric_value in val_metrics.items():
                self.writer.add_scalar(f'Metrics/{metric_name}', metric_value, epoch)
            
            # Check for improvement
            improved = False
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                improved = True
            
            if val_accuracy > self.best_val_accuracy:
                self.best_val_accuracy = val_accuracy
                improved = True
            
            # Save checkpoint
            if improved:
                self.epochs_without_improvement = 0
                self.save_checkpoint(epoch, is_best=True)
            else:
                self.epochs_without_improvement += 1
                if training_config["save_best_only"]:
                    self.save_checkpoint(epoch, is_best=False)
            
            # Early stopping
            if self.epochs_without_improvement >= training_config["early_stopping_patience"]:
                self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Save final checkpoint
        self.save_checkpoint(epochs - 1, is_best=False)
        
        # Close tensorboard writer
        self.writer.close()
        
        self.logger.info(f"Training completed. Best validation accuracy: {self.best_val_accuracy:.2f}%")
        
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_accuracies": self.train_accuracies,
            "val_accuracies": self.val_accuracies
        }
    
    def plot_training_curves(self):
        """Plot training curves"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curves
        ax2.plot(self.train_accuracies, label='Train Accuracy')
        ax2.plot(self.val_accuracies, label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        # Learning rate curve
        if self.scheduler:
            lrs = []
            for i in range(len(self.train_losses)):
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    lrs.append(self.optimizer.param_groups[0]['lr'])
                else:
                    # Simulate LR for other schedulers
                    lrs.append(self.optimizer.param_groups[0]['lr'])
            
            ax3.plot(lrs)
            ax3.set_title('Learning Rate Schedule')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Learning Rate')
            ax3.grid(True)
        
        # Loss distribution
        ax4.hist(self.train_losses, alpha=0.7, label='Train Loss', bins=20)
        ax4.hist(self.val_losses, alpha=0.7, label='Validation Loss', bins=20)
        ax4.set_title('Loss Distribution')
        ax4.set_xlabel('Loss')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "training_curves.png", dpi=300, bbox_inches='tight')
        plt.close()

class EnsembleTrainer:
    """Trainer for ensemble models"""
    
    def __init__(self, config: TrainingConfig, output_dir: str = "training_outputs"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.trainers = {}
        self.ensemble_results = {}
    
    def train_all_models(self, train_loader: DataLoader, val_loader: DataLoader,
                        models: List[str] = None) -> Dict[str, Dict[str, List[float]]]:
        """Train all models in the ensemble"""
        if models is None:
            models = ["mesonet", "xception", "efficientnet", "f3net"]
        
        results = {}
        
        for model_name in models:
            self.logger.info(f"Training {model_name}...")
            
            # Create trainer for this model
            trainer = ModelTrainer(model_name, self.config, str(self.output_dir))
            self.trainers[model_name] = trainer
            
            # Train the model
            model_results = trainer.train(train_loader, val_loader)
            results[model_name] = model_results
            
            # Plot training curves
            trainer.plot_training_curves()
        
        self.ensemble_results = results
        return results
    
    def evaluate_ensemble(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate ensemble performance"""
        ensemble_predictions = []
        true_labels = []
        
        for model_name, trainer in self.trainers.items():
            model_predictions = []
            
            trainer.model.model.eval()
            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(self.config.device)
                    outputs = trainer.model.model(images)
                    model_predictions.extend(outputs.squeeze().cpu().numpy())
                    
                    if model_name == list(self.trainers.keys())[0]:
                        true_labels.extend(labels.squeeze().cpu().numpy())
            
            ensemble_predictions.append(model_predictions)
        
        # Average ensemble predictions
        ensemble_predictions = np.array(ensemble_predictions)
        avg_predictions = np.mean(ensemble_predictions, axis=0)
        
        # Calculate ensemble metrics
        binary_predictions = (avg_predictions > 0.5).astype(int)
        true_labels = np.array(true_labels)
        
        accuracy = accuracy_score(true_labels, binary_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, binary_predictions, average='binary'
        )
        auc = roc_auc_score(true_labels, avg_predictions)
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc": auc
        }

def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train deepfake detection models")
    parser.add_argument("--model", choices=["mesonet", "xception", "efficientnet", "f3net", "all"],
                       default="all", help="Model to train")
    parser.add_argument("--config", help="Path to training configuration file")
    parser.add_argument("--dataset", required=True, help="Path to dataset")
    parser.add_argument("--output-dir", default="training_outputs", help="Output directory")
    parser.add_argument("--resume", help="Path to checkpoint to resume from")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    
    args = parser.parse_args()
    
    # Load configuration
    config = TrainingConfig(args.config)
    
    # Override batch size if specified
    if args.batch_size:
        config.config["training"]["batch_size"] = args.batch_size
    
    # Setup dataset
    dataset_manager = DatasetManager()
    
    # Create data loaders
    train_loader = dataset_manager.get_data_loader(
        args.dataset, "train", batch_size=config.config["training"]["batch_size"]
    )
    val_loader = dataset_manager.get_data_loader(
        args.dataset, "val", batch_size=config.config["training"]["batch_size"], shuffle=False
    )
    
    if args.model == "all":
        # Train ensemble
        ensemble_trainer = EnsembleTrainer(config, args.output_dir)
        results = ensemble_trainer.train_all_models(train_loader, val_loader)
        
        # Evaluate ensemble
        test_loader = dataset_manager.get_data_loader(
            args.dataset, "test", batch_size=config.config["training"]["batch_size"], shuffle=False
        )
        ensemble_metrics = ensemble_trainer.evaluate_ensemble(test_loader)
        
        logger.info("Ensemble training completed!")
        logger.info(f"Ensemble metrics: {ensemble_metrics}")
        
    else:
        # Train single model
        trainer = ModelTrainer(args.model, config, args.output_dir)
        
        # Resume from checkpoint if specified
        start_epoch = 0
        if args.resume:
            start_epoch = trainer.load_checkpoint(args.resume)
        
        # Train the model
        results = trainer.train(train_loader, val_loader, start_epoch)
        
        # Plot training curves
        trainer.plot_training_curves()
        
        logger.info(f"{args.model} training completed!")

if __name__ == "__main__":
    main() 