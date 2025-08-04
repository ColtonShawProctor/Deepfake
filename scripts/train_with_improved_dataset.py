#!/usr/bin/env python3
"""
Train deepfake detection models with improved diverse dataset

This script trains the models using the improved dataset with much better
variety and realistic samples for superior performance.
"""

import os
import sys
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
import json
from typing import List, Tuple, Dict, Any

# Add the app directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from app.models.deepfake_models import (
    ResNetDetector,
    EfficientNetDetector,
    F3NetDetector,
    ModelManager
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImprovedDataset(Dataset):
    """Dataset using the improved training data"""
    
    def __init__(self, training_data_dir: str = "improved_training_data", transform=None):
        self.training_data_dir = Path(training_data_dir)
        self.transform = transform or self._get_default_transform()
        
        # Load metadata
        with open(self.training_data_dir / "improved_training_metadata.json", 'r') as f:
            self.metadata = json.load(f)
        
        # Create samples list
        self.samples = []
        
        # Add real samples
        for sample_info in self.metadata["real"]:
            image_path = Path(sample_info["path"])
            if image_path.exists():
                self.samples.append((str(image_path), 0))  # 0 = real
        
        # Add fake samples
        for sample_info in self.metadata["fake"]:
            image_path = Path(sample_info["path"])
            if image_path.exists():
                self.samples.append((str(image_path), 1))  # 1 = fake
        
        logger.info(f"Loaded {len(self.samples)} samples from improved dataset")
        logger.info(f"Real samples: {len([s for s in self.samples if s[1] == 0])}")
        logger.info(f"Fake samples: {len([s for s in self.samples if s[1] == 1])}")
        
        # Log dataset statistics
        logger.info(f"Dataset statistics:")
        logger.info(f"  Celebrities used: {len(self.metadata['statistics']['celebrities_used'])}")
        logger.info(f"  Techniques used: {len(self.metadata['statistics']['techniques_used'])}")
        logger.info(f"  Variations applied: {len(self.metadata['statistics']['variations_applied'])}")
    
    def _get_default_transform(self):
        """Default transformations for training"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.float32)

class ImprovedModelTrainer:
    """Improved trainer for deepfake detection models"""
    
    def __init__(self, model_name: str, models_dir: str = "models"):
        self.model_name = model_name
        self.models_dir = Path(models_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize model
        if model_name == "resnet":
            self.model = ResNetDetector(device=str(self.device))
        elif model_name == "efficientnet":
            self.model = EfficientNetDetector(device=str(self.device))
        elif model_name == "f3net":
            self.model = F3NetDetector(device=str(self.device))
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Load existing weights if available
        weights_path = self.models_dir / f"{model_name}_weights.pth"
        if weights_path.exists():
            logger.info(f"Loading existing weights from {weights_path}")
            self.model.load_model(str(weights_path))
        else:
            logger.info("Loading default weights")
            self.model.load_model()
        
        self.model.model.train()
    
    def train(self, dataset: ImprovedDataset, epochs: int = 100, lr: float = 0.0001, 
              batch_size: int = 32, val_split: float = 0.2):
        """Train the model with validation split"""
        logger.info(f"Training {self.model_name} for {epochs} epochs")
        logger.info(f"Dataset size: {len(dataset)} samples")
        
        # Split dataset into train and validation
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
        
        # Setup data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        # Setup training
        criterion = nn.BCELoss()
        optimizer = optim.AdamW(self.model.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
        
        # Training history
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        best_val_accuracy = 0.0
        patience_counter = 0
        max_patience = 20
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.model.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model.model(images)
                loss = criterion(outputs.squeeze(), labels.squeeze())
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                # Calculate accuracy
                predicted = (outputs.squeeze() > 0.5).float()
                train_correct += (predicted == labels.squeeze()).sum().item()
                train_total += labels.size(0)
                
                if batch_idx % 50 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, "
                              f"Loss: {loss.item():.4f}")
            
            # Validation phase
            self.model.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = self.model.model(images)
                    loss = criterion(outputs.squeeze(), labels.squeeze())
                    
                    val_loss += loss.item()
                    
                    predicted = (outputs.squeeze() > 0.5).float()
                    val_correct += (predicted == labels.squeeze()).sum().item()
                    val_total += labels.size(0)
            
            # Calculate metrics
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            train_accuracy = 100.0 * train_correct / train_total
            val_accuracy = 100.0 * val_correct / val_total
            
            # Update learning rate
            scheduler.step(avg_val_loss)
            
            # Save history
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)
            
            logger.info(f"Epoch {epoch+1}/{epochs}:")
            logger.info(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
            logger.info(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
            logger.info(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                patience_counter = 0
                weights_path = self.models_dir / f"{self.model_name}_weights_best.pth"
                torch.save(self.model.model.state_dict(), weights_path)
                logger.info(f"  New best model saved with validation accuracy: {val_accuracy:.2f}%")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= max_patience:
                logger.info(f"Early stopping at epoch {epoch+1} due to no improvement for {max_patience} epochs")
                break
        
        # Save final model
        weights_path = self.models_dir / f"{self.model_name}_weights.pth"
        torch.save(self.model.model.state_dict(), weights_path)
        logger.info(f"Final model saved to {weights_path}")
        
        # Save training history
        history = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_accuracies": train_accuracies,
            "val_accuracies": val_accuracies,
            "best_val_accuracy": best_val_accuracy,
            "final_epoch": len(train_losses)
        }
        
        history_path = self.models_dir / f"{self.model_name}_improved_training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"Training history saved to {history_path}")
        return history
    
    def evaluate(self, dataset: ImprovedDataset) -> Dict[str, float]:
        """Evaluate the trained model"""
        self.model.model.eval()
        
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model.model(images)
                predicted = (outputs.squeeze() > 0.5).float()
                
                correct += (predicted == labels.squeeze()).sum().item()
                total += labels.size(0)
        
        accuracy = 100.0 * correct / total
        logger.info(f"Final evaluation accuracy: {accuracy:.2f}%")
        
        return {"accuracy": accuracy}

def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train deepfake detection models with improved dataset")
    parser.add_argument("--model", choices=["resnet", "efficientnet", "f3net", "all"], 
                       default="all", help="Model to train")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--training-data-dir", default="improved_training_data", help="Training data directory")
    parser.add_argument("--models-dir", default="models", help="Models directory")
    
    args = parser.parse_args()
    
    # Load dataset
    dataset = ImprovedDataset(args.training_data_dir)
    
    if args.model == "all":
        models = ["resnet", "efficientnet", "f3net"]
    else:
        models = [args.model]
    
    # Train each model
    for model_name in models:
        logger.info(f"Training {model_name}...")
        trainer = ImprovedModelTrainer(model_name, args.models_dir)
        
        # Train the model
        history = trainer.train(
            dataset, 
            args.epochs, 
            args.lr, 
            args.batch_size, 
            args.val_split
        )
        
        # Evaluate
        results = trainer.evaluate(dataset)
        logger.info(f"{model_name} final accuracy: {results['accuracy']:.2f}%")
        logger.info(f"{model_name} best validation accuracy: {history['best_val_accuracy']:.2f}%")
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main() 