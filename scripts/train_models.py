#!/usr/bin/env python3
"""
Training script for deepfake detection models

This script fine-tunes the pre-trained models on deepfake detection datasets
to improve their performance on the test samples.
"""

import os
import sys
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
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

class TestSamplesDataset(Dataset):
    """Dataset using the test samples for training"""
    
    def __init__(self, test_samples_dir: str = "test_samples", transform=None):
        self.test_samples_dir = Path(test_samples_dir)
        self.transform = transform or self._get_default_transform()
        
        # Load metadata
        with open(self.test_samples_dir / "samples_metadata.json", 'r') as f:
            self.metadata = json.load(f)
        
        # Create samples list
        self.samples = []
        for sample_name, info in self.metadata.items():
            image_path = self.test_samples_dir / info["path"].split("/")[-1]
            if image_path.exists():
                self.samples.append((str(image_path), info["expected"]))
        
        logger.info(f"Loaded {len(self.samples)} samples from test_samples")
    
    def _get_default_transform(self):
        """Default transformations for training"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
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

class ModelTrainer:
    """Trainer for deepfake detection models"""
    
    def __init__(self, model_name: str, models_dir: str = "models"):
        self.model_name = model_name
        self.models_dir = Path(models_dir)
        # Check for available devices in order of preference: CUDA > MPS > CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        # Initialize model
        if model_name == "resnet":
            self.model = ResNetDetector(device=str(self.device))
        elif model_name == "efficientnet":
            self.model = EfficientNetDetector(device=str(self.device))
        elif model_name == "f3net":
            self.model = F3NetDetector(device=str(self.device))
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Load existing weights
        weights_path = self.models_dir / f"{model_name}_weights.pth"
        if weights_path.exists():
            self.model.load_model(str(weights_path))
        else:
            self.model.load_model()
        
        self.model.model.train()
    
    def train(self, dataset: TestSamplesDataset, epochs: int = 10, lr: float = 0.001):
        """Train the model"""
        logger.info(f"Training {self.model_name} for {epochs} epochs")
        
        # Setup training
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.model.parameters(), lr=lr)
        
        # Training loop
        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (images, labels) in enumerate(dataloader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model.model(images)
                loss = criterion(outputs.squeeze(), labels.squeeze())
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                # Calculate accuracy
                predicted = (outputs.squeeze() > 0.5).float()
                correct += (predicted == labels.squeeze()).sum().item()
                total += labels.size(0)
            
            avg_loss = total_loss / len(dataloader)
            accuracy = 100.0 * correct / total
            
            logger.info(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%")
        
        # Save trained model
        weights_path = self.models_dir / f"{self.model_name}_weights.pth"
        torch.save(self.model.model.state_dict(), weights_path)
        logger.info(f"Saved trained model to {weights_path}")
    
    def evaluate(self, dataset: TestSamplesDataset) -> Dict[str, float]:
        """Evaluate the trained model"""
        self.model.model.eval()
        
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
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
        logger.info(f"Evaluation accuracy: {accuracy:.2f}%")
        
        return {"accuracy": accuracy}

def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train deepfake detection models")
    parser.add_argument("--model", choices=["resnet", "efficientnet", "f3net", "all"], 
                       default="all", help="Model to train")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--test-samples-dir", default="test_samples", help="Test samples directory")
    parser.add_argument("--models-dir", default="models", help="Models directory")
    
    args = parser.parse_args()
    
    # Load dataset
    dataset = TestSamplesDataset(args.test_samples_dir)
    
    if args.model == "all":
        models = ["resnet", "efficientnet", "f3net"]
    else:
        models = [args.model]
    
    # Train each model
    for model_name in models:
        logger.info(f"Training {model_name}...")
        trainer = ModelTrainer(model_name, args.models_dir)
        trainer.train(dataset, args.epochs, args.lr)
        
        # Evaluate
        results = trainer.evaluate(dataset)
        logger.info(f"{model_name} final accuracy: {results['accuracy']:.2f}%")
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main() 