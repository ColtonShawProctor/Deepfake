#!/usr/bin/env python3
"""
Simple, Fast, and Efficient Training Script for Deepfake Detection Models
Target: Base level testing accuracy (80-85%) with minimal complexity
"""

import os
import sys
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import numpy as np
from PIL import Image
import random
from pathlib import Path
import logging
from torch.cuda.amp import autocast, GradScaler

# Add the app directory to the path
sys.path.append('app')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleCelebDFDataset(Dataset):
    """Simple dataset for Celeb-DF-v2 videos with efficient frame extraction"""
    
    def __init__(self, video_dir, transform=None, max_frames_per_video=5, is_training=True):
        self.video_dir = Path(video_dir)
        self.transform = transform
        self.max_frames_per_video = max_frames_per_video
        self.is_training = is_training
        
        # Get all video files
        self.video_files = []
        self.labels = []
        for video_file in self.video_dir.rglob("*.mp4"):
            self.video_files.append(video_file)
            # Determine label based on directory
            if "real" in str(video_file):
                self.labels.append(0)  # Real
            else:
                self.labels.append(1)  # Fake
        
        logger.info(f"Found {len(self.video_files)} videos in {video_dir}")
        logger.info(f"Real: {sum(1 for l in self.labels if l == 0)}, Fake: {sum(1 for l in self.labels if l == 1)}")
    
    def __len__(self):
        return len(self.video_files)
    
    def extract_frames(self, video_path, num_frames=5):
        """Extract frames efficiently"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            return []
        
        # Sample frames evenly
        if total_frames <= num_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
        
        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
        
        cap.release()
        return frames
    
    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        label = self.labels[idx]
        
        # Extract frames
        frames = self.extract_frames(video_path, self.max_frames_per_video)
        
        if not frames:
            # Return a dummy frame if extraction fails
            frames = [np.zeros((224, 224, 3), dtype=np.uint8)]
        
        # Use random frame for training, first frame for validation
        if self.is_training:
            frame = random.choice(frames)
        else:
            frame = frames[0]
        
        # Convert to PIL Image
        image = Image.fromarray(frame)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)

def get_simple_transforms(model_name, is_training=True):
    """Get simple but effective transforms for each model"""
    if model_name == "mesonet":
        input_size = (256, 256)
    elif model_name == "xception":
        input_size = (299, 299)  # InceptionV3 uses 299x299
    else:
        input_size = (224, 224)
    
    if is_training:
        return transforms.Compose([
            transforms.Resize(input_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

class SimpleModelFactory:
    """Factory for creating simple but effective models"""
    
    @staticmethod
    def create_model(model_name, num_classes=2):
        """Create model with appropriate architecture"""
        if model_name == "resnet":
            return SimpleModelFactory._create_resnet(num_classes)
        elif model_name == "efficientnet":
            return SimpleModelFactory._create_efficientnet(num_classes)
        elif model_name == "f3net":
            return SimpleModelFactory._create_f3net(num_classes)
        elif model_name == "mesonet":
            return SimpleModelFactory._create_mesonet(num_classes)
        elif model_name == "xception":
            return SimpleModelFactory._create_xception(num_classes)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    @staticmethod
    def _create_resnet(num_classes):
        """Create ResNet with transfer learning"""
        model = models.resnet50(pretrained=True)
        
        # Freeze early layers
        for param in list(model.parameters())[:-20]:
            param.requires_grad = False
        
        # Replace classifier
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        return model
    
    @staticmethod
    def _create_efficientnet(num_classes):
        """Create EfficientNet with transfer learning"""
        model = models.efficientnet_b4(pretrained=True)
        
        # Freeze early layers
        for param in list(model.parameters())[:-10]:
            param.requires_grad = False
        
        # Replace classifier
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        return model
    
    @staticmethod
    def _create_f3net(num_classes):
        """Create simple F3Net"""
        class SimpleF3Net(nn.Module):
            def __init__(self, num_classes=2):
                super(SimpleF3Net, self).__init__()
                
                # Feature extraction
                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    
                    nn.Conv2d(128, 256, 3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((1, 1))
                )
                
                # Classifier
                self.classifier = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, num_classes)
                )
            
            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                return x
        
        return SimpleF3Net(num_classes)
    
    @staticmethod
    def _create_mesonet(num_classes):
        """Create MesoNet"""
        class SimpleMesoNet(nn.Module):
            def __init__(self, num_classes=2):
                super(SimpleMesoNet, self).__init__()
                
                self.features = nn.Sequential(
                    # Block 1
                    nn.Conv2d(3, 8, 3, padding=1),
                    nn.BatchNorm2d(8),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    
                    # Block 2
                    nn.Conv2d(8, 16, 5, padding=2),
                    nn.BatchNorm2d(16),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    
                    # Block 3
                    nn.Conv2d(16, 32, 5, padding=2),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    
                    # Block 4
                    nn.Conv2d(32, 64, 5, padding=2),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((1, 1))
                )
                
                self.classifier = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(64, 32),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                    nn.Linear(32, num_classes)
                )
            
            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                return x
        
        return SimpleMesoNet(num_classes)
    
    @staticmethod
    def _create_xception(num_classes):
        """Create Xception-like model using InceptionV3 as base"""
        class InceptionWrapper(nn.Module):
            def __init__(self, num_classes):
                super(InceptionWrapper, self).__init__()
                self.backbone = models.inception_v3(pretrained=True)
                
                # Freeze early layers
                for param in list(self.backbone.parameters())[:-20]:
                    param.requires_grad = False
                
                # Replace classifier
                num_features = self.backbone.fc.in_features
                self.backbone.fc = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(num_features, 512),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(512, 128),
                    nn.ReLU(),
                    nn.Linear(128, num_classes)
                )
            
            def forward(self, x):
                output = self.backbone(x)
                # InceptionV3 returns a tuple, we want the main output
                if isinstance(output, tuple):
                    return output[0]
                return output
        
        return InceptionWrapper(num_classes)

class SimpleTrainer:
    """Simple but effective trainer"""
    
    def __init__(self, model_name, device, num_classes=2):
        self.model_name = model_name
        self.device = device
        self.num_classes = num_classes
        self.scaler = GradScaler() if device.type == 'cuda' else None
        self.use_amp = device.type == 'cuda'
    
    def train(self, train_loader, val_loader, epochs=25):
        """Train model with simple but effective approach"""
        model = SimpleModelFactory.create_model(self.model_name, self.num_classes)
        model.to(self.device)
        
        # Setup optimizer with different learning rates
        backbone_params = []
        classifier_params = []
        
        for name, param in model.named_parameters():
            if 'classifier' in name or 'fc' in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)
        
        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': 1e-5},
            {'params': classifier_params, 'lr': 1e-4}
        ], weight_decay=0.01)
        
        # Simple scheduler
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0
        patience = 8
        patience_counter = 0
        
        logger.info(f"Training {self.model_name} for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                
                if self.use_amp and self.scaler:
                    with autocast():
                        output = model(data)
                        loss = criterion(output, target)
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                
                train_loss += loss.item()
                pred = output.argmax(dim=1)
                train_correct += pred.eq(target).sum().item()
                train_total += target.size(0)
                
                if batch_idx % 20 == 0:
                    logger.info(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    
                    if self.use_amp and self.scaler:
                        with autocast():
                            output = model(data)
                            loss = criterion(output, target)
                    else:
                        output = model(data)
                        loss = criterion(output, target)
                    
                    val_loss += loss.item()
                    pred = output.argmax(dim=1)
                    val_correct += pred.eq(target).sum().item()
                    val_total += target.size(0)
            
            train_acc = 100. * train_correct / train_total
            val_acc = 100. * val_correct / val_total
            val_loss /= len(val_loader)
            
            logger.info(f'Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, Val Loss: {val_loss:.4f}')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(model.state_dict(), f'models/{self.model_name}_weights.pth')
                logger.info(f'New best model saved for {self.model_name} with acc: {val_acc:.2f}%')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f'Early stopping at epoch {epoch+1}')
                    break
            
            scheduler.step()
        
        return best_val_acc

def main():
    """Main training function"""
    logger.info("Starting simple, fast, and efficient training...")
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    logger.info(f"Using device: {device}")
    
    # Dataset paths
    real_videos = "Celeb-DF-v2/Celeb-real"
    fake_videos = "Celeb-DF-v2/Celeb-synthesis"
    
    # Check if datasets exist
    if not os.path.exists(real_videos) or not os.path.exists(fake_videos):
        logger.error("Celeb-DF-v2 dataset not found.")
        return
    
    # Check which models are missing and only train those
    models_to_train = []
    available_models = ["resnet", "efficientnet", "f3net", "mesonet", "xception"]
    
    for model_name in available_models:
        model_path = f"models/{model_name}_weights.pth"
        if not os.path.exists(model_path):
            models_to_train.append(model_name)
            logger.info(f"{model_name} model not found, will train")
        else:
            logger.info(f"{model_name} model already exists, skipping")
    
    if not models_to_train:
        logger.info("All models already trained!")
        return
    
    for model_name in models_to_train:
        logger.info(f"Training {model_name}...")
        
        # Get transforms
        train_transform = get_simple_transforms(model_name, is_training=True)
        val_transform = get_simple_transforms(model_name, is_training=False)
        
        # Create datasets
        real_train_dataset = SimpleCelebDFDataset(real_videos, transform=train_transform, is_training=True)
        fake_train_dataset = SimpleCelebDFDataset(fake_videos, transform=train_transform, is_training=True)
        real_val_dataset = SimpleCelebDFDataset(real_videos, transform=val_transform, is_training=False)
        fake_val_dataset = SimpleCelebDFDataset(fake_videos, transform=val_transform, is_training=False)
        
        # Combine datasets
        from torch.utils.data import ConcatDataset
        train_dataset = ConcatDataset([real_train_dataset, fake_train_dataset])
        val_dataset = ConcatDataset([real_val_dataset, fake_val_dataset])
        
        # Create data loaders
        pin_memory = device.type == 'cuda'
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=pin_memory)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=pin_memory)
        
        # Train model
        trainer = SimpleTrainer(model_name, device)
        best_acc = trainer.train(train_loader, val_loader, epochs=25)
        logger.info(f"{model_name} training completed with best accuracy: {best_acc:.2f}%")
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main()
