"""
Training script for EfficientNet-B4 model on deepfake detection datasets.
Optimized for mobile deployment with efficient augmentation and memory management.
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from .efficientnet_detector import EfficientNetDetector, EfficientNetPreprocessor


class EfficientNetDataset(Dataset):
    """
    Dataset class for EfficientNet-B4 deepfake detection training.
    
    Optimized for mobile deployment with efficient preprocessing and augmentation.
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        balance_classes: bool = True,
        mobile_optimized: bool = True
    ):
        """
        Initialize dataset.
        
        Args:
            data_dir: Directory containing dataset
            split: Dataset split ('train', 'val', 'test')
            transform: Image transformations
            balance_classes: Whether to balance real/fake classes
            mobile_optimized: Whether to use mobile-optimized preprocessing
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.mobile_optimized = mobile_optimized
        self.transform = transform or self._get_default_transform()
        self.balance_classes = balance_classes
        
        # Load dataset
        self.samples = self._load_dataset()
        
        if balance_classes:
            self.samples = self._balance_classes()
        
        logging.info(f"Loaded {len(self.samples)} samples for {split} split")
    
    def _get_default_transform(self) -> transforms.Compose:
        """Get default transformations optimized for EfficientNet-B4."""
        if self.mobile_optimized:
            # Mobile-optimized transforms (faster, less memory)
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=5),  # Smaller rotation for mobile
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            # Full augmentation transforms
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def _load_dataset(self) -> List[Tuple[str, int]]:
        """Load dataset samples."""
        samples = []
        
        # Check for different dataset formats
        if (self.data_dir / "real").exists() and (self.data_dir / "fake").exists():
            # FaceForensics++ format
            samples.extend(self._load_faceforensics_format())
        elif (self.data_dir / "train").exists():
            # DFDC format
            samples.extend(self._load_dfdc_format())
        else:
            # Generic format
            samples.extend(self._load_generic_format())
        
        return samples
    
    def _load_faceforensics_format(self) -> List[Tuple[str, int]]:
        """Load FaceForensics++ format dataset."""
        samples = []
        
        # Real images
        real_dir = self.data_dir / "real"
        if real_dir.exists():
            for img_path in real_dir.rglob("*.jpg"):
                samples.append((str(img_path), 0))  # 0 = real
            for img_path in real_dir.rglob("*.png"):
                samples.append((str(img_path), 0))
        
        # Fake images
        fake_dir = self.data_dir / "fake"
        if fake_dir.exists():
            for img_path in fake_dir.rglob("*.jpg"):
                samples.append((str(img_path), 1))  # 1 = fake
            for img_path in fake_dir.rglob("*.png"):
                samples.append((str(img_path), 1))
        
        return samples
    
    def _load_dfdc_format(self) -> List[Tuple[str, int]]:
        """Load DFDC format dataset."""
        samples = []
        
        split_dir = self.data_dir / self.split
        if not split_dir.exists():
            return samples
        
        # Load metadata if available
        metadata_file = split_dir / "metadata.json"
        if metadata_file.exists():
            import json
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            for video_id, info in metadata.items():
                label = 1 if info.get("label", "REAL") == "FAKE" else 0
                video_dir = split_dir / video_id
                if video_dir.exists():
                    for img_path in video_dir.glob("*.jpg"):
                        samples.append((str(img_path), label))
        else:
            # Fallback: assume directory structure
            for video_dir in split_dir.iterdir():
                if video_dir.is_dir():
                    for img_path in video_dir.glob("*.jpg"):
                        samples.append((str(img_path), 0))  # Default to real
        
        return samples
    
    def _load_generic_format(self) -> List[Tuple[str, int]]:
        """Load generic format dataset."""
        samples = []
        
        # Look for common patterns
        for img_path in self.data_dir.rglob("*.jpg"):
            # Try to infer label from filename or path
            label = self._infer_label_from_path(img_path)
            samples.append((str(img_path), label))
        
        for img_path in self.data_dir.rglob("*.png"):
            label = self._infer_label_from_path(img_path)
            samples.append((str(img_path), label))
        
        return samples
    
    def _infer_label_from_path(self, img_path: Path) -> int:
        """Infer label from image path."""
        path_str = str(img_path).lower()
        
        # Common patterns for fake images
        fake_keywords = ["fake", "deepfake", "synthetic", "generated", "manipulated"]
        for keyword in fake_keywords:
            if keyword in path_str:
                return 1
        
        # Common patterns for real images
        real_keywords = ["real", "authentic", "original", "genuine"]
        for keyword in real_keywords:
            if keyword in path_str:
                return 0
        
        # Default to real if uncertain
        return 0
    
    def _balance_classes(self) -> List[Tuple[str, int]]:
        """Balance real and fake classes."""
        real_samples = [s for s in self.samples if s[1] == 0]
        fake_samples = [s for s in self.samples if s[1] == 1]
        
        min_count = min(len(real_samples), len(fake_samples))
        
        # Randomly sample to balance
        np.random.seed(42)
        balanced_samples = []
        balanced_samples.extend(np.random.choice(real_samples, min_count, replace=False))
        balanced_samples.extend(np.random.choice(fake_samples, min_count, replace=False))
        
        # Shuffle
        np.random.shuffle(balanced_samples)
        
        return balanced_samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a sample from the dataset."""
        img_path, label = self.samples[idx]
        
        try:
            # Load image
            image = Image.open(img_path).convert('RGB')
            
            # Apply transformations
            if self.transform:
                image = self.transform(image)
            
            return image, label
            
        except Exception as e:
            logging.warning(f"Failed to load image {img_path}: {str(e)}")
            # Return a placeholder image
            placeholder = torch.zeros(3, 224, 224)
            return placeholder, label


class EfficientNetTrainer:
    """
    Trainer class for EfficientNet-B4 model on deepfake detection.
    
    Optimized for mobile deployment with efficient training strategies.
    """
    
    def __init__(
        self,
        model: EfficientNetDetector,
        train_dir: str,
        val_dir: str,
        output_dir: str = "checkpoints",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: EfficientNet detector instance
            train_dir: Training data directory
            val_dir: Validation data directory
            output_dir: Output directory for checkpoints
            config: Training configuration
        """
        self.model = model
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Training configuration
        self.config = config or {}
        self.batch_size = self.config.get("batch_size", 32)  # Larger batch size for EfficientNet
        self.learning_rate = self.config.get("learning_rate", 5e-5)  # Lower LR for EfficientNet
        self.weight_decay = self.config.get("weight_decay", 1e-4)
        self.num_epochs = self.config.get("num_epochs", 30)  # Fewer epochs for mobile
        self.save_interval = self.config.get("save_interval", 5)
        self.eval_interval = self.config.get("eval_interval", 1)
        self.mobile_optimized = self.config.get("mobile_optimized", True)
        
        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.EfficientNetTrainer")
        
        # Initialize training components
        self._setup_training()
        
        self.logger.info(f"EfficientNet trainer initialized with config: {self.config}")
    
    def _setup_training(self):
        """Setup training components."""
        # Setup fine-tuning
        training_components = self.model.fine_tune_setup(
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        self.optimizer = training_components["optimizer"]
        self.scheduler = training_components["scheduler"]
        self.criterion = training_components["criterion"]
        
        # Create datasets
        self.train_dataset = EfficientNetDataset(
            self.train_dir,
            split="train",
            balance_classes=True,
            mobile_optimized=self.mobile_optimized
        )
        
        self.val_dataset = EfficientNetDataset(
            self.val_dir,
            split="val",
            balance_classes=False,
            mobile_optimized=self.mobile_optimized
        )
        
        # Create data loaders with mobile optimization
        num_workers = 2 if self.mobile_optimized else 4  # Fewer workers for mobile
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False
        )
    
    def train(self) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Returns:
            Dictionary containing training history
        """
        self.logger.info("Starting EfficientNet-B4 training...")
        
        # Training history
        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1": [],
            "val_auroc": []
        }
        
        best_val_auroc = 0.0  # Track AUROC instead of accuracy for EfficientNet
        
        for epoch in range(self.num_epochs):
            self.logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")
            
            # Training phase
            train_loss, train_acc = self._train_epoch()
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            
            # Validation phase
            if (epoch + 1) % self.eval_interval == 0:
                val_metrics = self._validate_epoch()
                history["val_loss"].append(val_metrics["loss"])
                history["val_acc"].append(val_metrics["accuracy"])
                history["val_precision"].append(val_metrics["precision"])
                history["val_recall"].append(val_metrics["recall"])
                history["val_f1"].append(val_metrics["f1_score"])
                history["val_auroc"].append(val_metrics["auroc"])
                
                # Update learning rate
                self.scheduler.step(val_metrics["loss"])
                
                # Save best model based on AUROC
                if val_metrics["auroc"] > best_val_auroc:
                    best_val_auroc = val_metrics["auroc"]
                    self._save_checkpoint("best_model.pth", epoch, val_metrics)
                
                self.logger.info(
                    f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, "
                    f"Train Acc: {train_acc:.4f}, Val AUROC: {val_metrics['auroc']:.4f}"
                )
            
            # Save checkpoint periodically
            if (epoch + 1) % self.save_interval == 0:
                self._save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pth", epoch, val_metrics)
        
        # Save final model
        self._save_checkpoint("final_model.pth", self.num_epochs - 1, val_metrics)
        
        self.logger.info(f"Training completed. Best validation AUROC: {best_val_auroc:.4f}")
        return history
    
    def _train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images = images.to(self.model.device)
            labels = labels.float().to(self.model.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model.model(images).squeeze()
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Calculate accuracy
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{correct/total:.4f}'
            })
            
            # Memory optimization for mobile
            if self.mobile_optimized and batch_idx % 10 == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def _validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validation"):
                images = images.to(self.model.device)
                labels = labels.float().to(self.model.device)
                
                # Forward pass
                outputs = self.model.model(images).squeeze()
                loss = self.criterion(outputs, labels)
                
                # Store predictions and labels
                predictions = (outputs > 0.5).float()
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(outputs.cpu().numpy())
                
                total_loss += loss.item()
        
        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
        
        # Calculate precision, recall, F1
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
        
        precision = precision_score(all_labels, all_predictions, zero_division=0)
        recall = recall_score(all_labels, all_predictions, zero_division=0)
        f1 = f1_score(all_labels, all_predictions, zero_division=0)
        
        # Calculate AUROC
        try:
            auroc = roc_auc_score(all_labels, all_probabilities)
        except ValueError:
            auroc = 0.5  # Default value if AUROC cannot be calculated
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auroc": auroc
        }
    
    def _save_checkpoint(self, filename: str, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint."""
        checkpoint_path = self.output_dir / filename
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "config": self.config
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def evaluate_on_test_set(self, test_dir: str) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Args:
            test_dir: Test data directory
            
        Returns:
            Dictionary containing test metrics
        """
        self.logger.info("Evaluating on test set...")
        
        # Create test dataset
        test_dataset = EfficientNetDataset(
            test_dir, 
            split="test", 
            balance_classes=False,
            mobile_optimized=self.mobile_optimized
        )
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Evaluate
        test_metrics = self._validate_epoch()
        
        self.logger.info(f"Test Results: {test_metrics}")
        return test_metrics
    
    def benchmark_performance(self) -> Dict[str, Any]:
        """
        Benchmark model performance for mobile deployment.
        
        Returns:
            Dictionary containing performance benchmarks
        """
        self.logger.info("Running performance benchmarks...")
        
        # Test inference speed
        test_image = torch.randn(1, 3, 224, 224).to(self.model.device)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = self.model.model(test_image)
        
        # Benchmark inference time
        times = []
        for _ in range(100):
            start_time = time.time()
            with torch.no_grad():
                _ = self.model.model(test_image)
            times.append(time.time() - start_time)
        
        avg_inference_time = np.mean(times) * 1000  # Convert to milliseconds
        throughput_fps = 1.0 / np.mean(times)
        
        # Memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
            memory_reserved = torch.cuda.memory_reserved() / (1024 * 1024)  # MB
        else:
            memory_allocated = 0
            memory_reserved = 0
        
        return {
            "avg_inference_time_ms": avg_inference_time,
            "throughput_fps": throughput_fps,
            "memory_allocated_mb": memory_allocated,
            "memory_reserved_mb": memory_reserved,
            "model_size_mb": self.model.model_info.model_size_mb,
            "parameters_count": self.model.model_info.parameters_count
        }


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train EfficientNet-B4 model for deepfake detection")
    parser.add_argument("--train_dir", required=True, help="Training data directory")
    parser.add_argument("--val_dir", required=True, help="Validation data directory")
    parser.add_argument("--test_dir", help="Test data directory")
    parser.add_argument("--output_dir", default="checkpoints", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--device", default="auto", help="Device to use")
    parser.add_argument("--mobile_optimized", action="store_true", help="Enable mobile optimization")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize model
    model = EfficientNetDetector(
        model_name="EfficientNetDetector",
        device=args.device,
        config={
            "enable_attention": True,
            "mobile_optimized": args.mobile_optimized
        }
    )
    
    # Load model
    if not model.load_model():
        logging.error("Failed to load model")
        return
    
    # Initialize trainer
    trainer = EfficientNetTrainer(
        model=model,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        output_dir=args.output_dir,
        config={
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "num_epochs": args.num_epochs,
            "mobile_optimized": args.mobile_optimized
        }
    )
    
    # Train model
    history = trainer.train()
    
    # Run performance benchmarks
    benchmarks = trainer.benchmark_performance()
    logging.info(f"Performance benchmarks: {benchmarks}")
    
    # Evaluate on test set if provided
    if args.test_dir:
        test_metrics = trainer.evaluate_on_test_set(args.test_dir)
        logging.info(f"Final test metrics: {test_metrics}")


if __name__ == "__main__":
    main() 