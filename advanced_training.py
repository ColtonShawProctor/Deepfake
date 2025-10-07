#!/usr/bin/env python3
"""
Advanced training script with modern deep learning best practices
"""

import os
import sys
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
import numpy as np
from PIL import Image
import random
from pathlib import Path
import logging
from torch.cuda.amp import autocast, GradScaler
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add the app directory to the path
sys.path.append('app')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedCelebDFDataset(Dataset):
    """Advanced dataset with proper augmentation and balancing"""
    
    def __init__(self, video_dir, transform=None, max_frames_per_video=8, is_training=True):
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
    
    def extract_frames(self, video_path, num_frames=8):
        """Extract frames with better sampling strategy"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            return []
        
        # Better frame sampling strategy
        if total_frames <= num_frames:
            frame_indices = list(range(total_frames))
        else:
            # Sample more frames from the middle where faces are more stable
            start = total_frames // 4
            end = 3 * total_frames // 4
            middle_frames = np.linspace(start, end, num_frames, dtype=int)
            frame_indices = middle_frames.tolist()
        
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
        
        # Apply transforms (Albumentations expects numpy array)
        if self.transform:
            image = self.transform(image=frame)['image']
        else:
            # Convert to tensor if no transform
            image = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        
        return image, torch.tensor(label, dtype=torch.long)

def get_advanced_transforms(is_training=True):
    """Get advanced transforms with proper augmentation"""
    if is_training:
        return A.Compose([
            A.Resize(256, 256),
            A.RandomCrop(224, 224),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.GaussianBlur(blur_limit=3, p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class AdvancedModelTrainer:
    """Advanced trainer with modern techniques"""
    
    def __init__(self, model_name, device, num_classes=2):
        self.model_name = model_name
        self.device = device
        self.num_classes = num_classes
        self.scaler = GradScaler() if device.type == 'cuda' else None
        self.use_amp = device.type == 'cuda'  # Only use AMP for CUDA
        
    def create_model(self):
        """Create model with proper architecture"""
        if self.model_name == "mesonet":
            return self._create_mesonet()
        elif self.model_name == "xception":
            return self._create_xception()
        elif self.model_name == "efficientnet":
            return self._create_efficientnet()
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
    
    def _create_mesonet(self):
        """Create improved MesoNet"""
        class ImprovedMesoNet(nn.Module):
            def __init__(self, num_classes=2):
                super(ImprovedMesoNet, self).__init__()
                
                # Feature extraction layers
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
                    nn.MaxPool2d(2, 2),
                    
                    # Block 5
                    nn.Conv2d(64, 128, 5, padding=2),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((1, 1))
                )
                
                # Classifier
                self.classifier = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(128, 64),
                    nn.BatchNorm1d(64),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                    nn.Linear(64, num_classes)
                )
            
            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                return x
        
        return ImprovedMesoNet(self.num_classes)
    
    def _create_xception(self):
        """Create Xception with proper transfer learning"""
        model = models.xception(pretrained=True)
        
        # Freeze early layers
        for param in list(model.parameters())[:-20]:
            param.requires_grad = False
        
        # Replace classifier
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.num_classes)
        )
        
        return model
    
    def _create_efficientnet(self):
        """Create EfficientNet with proper transfer learning"""
        model = models.efficientnet_b4(pretrained=True)
        
        # Freeze early layers
        for param in list(model.parameters())[:-10]:
            param.requires_grad = False
        
        # Replace classifier
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, self.num_classes)
        )
        
        return model
    
    def train(self, train_loader, val_loader, epochs=50):
        """Advanced training with modern techniques"""
        model = self.create_model().to(self.device)
        
        # Use different optimizers for different parts
        backbone_params = []
        classifier_params = []
        
        for name, param in model.named_parameters():
            if 'classifier' in name or 'fc' in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)
        
        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': 1e-5},  # Lower LR for backbone
            {'params': classifier_params, 'lr': 1e-4}  # Higher LR for classifier
        ], weight_decay=0.01)
        
        # Advanced scheduler
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=1e-3, epochs=epochs, 
            steps_per_epoch=len(train_loader)
        )
        
        # Loss function with class balancing
        criterion = FocalLoss(alpha=1, gamma=2)
        
        best_val_acc = 0
        patience = 15
        patience_counter = 0
        
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
                
                scheduler.step()
                
                train_loss += loss.item()
                pred = output.argmax(dim=1)
                train_correct += pred.eq(target).sum().item()
                train_total += target.size(0)
                
                if batch_idx % 20 == 0:
                    logger.info(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
            
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
            
            logger.info(f'Epoch {epoch}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, Val Loss: {val_loss:.4f}')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(model.state_dict(), f'models/{self.model_name}_weights_advanced.pth')
                logger.info(f'New best model saved for {self.model_name} with acc: {val_acc:.2f}%')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f'Early stopping at epoch {epoch}')
                    break
        
        return best_val_acc

def main():
    """Main training function with advanced techniques"""
    logger.info("Starting advanced training with modern techniques...")
    
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
    
    # Create datasets
    train_transform = get_advanced_transforms(is_training=True)
    val_transform = get_advanced_transforms(is_training=False)
    
    real_train_dataset = AdvancedCelebDFDataset(real_videos, transform=train_transform, is_training=True)
    fake_train_dataset = AdvancedCelebDFDataset(fake_videos, transform=train_transform, is_training=True)
    real_val_dataset = AdvancedCelebDFDataset(real_videos, transform=val_transform, is_training=False)
    fake_val_dataset = AdvancedCelebDFDataset(fake_videos, transform=val_transform, is_training=False)
    
    # Combine datasets
    from torch.utils.data import ConcatDataset
    train_dataset = ConcatDataset([real_train_dataset, fake_train_dataset])
    val_dataset = ConcatDataset([real_val_dataset, fake_val_dataset])
    
    # Create data loaders with proper balancing
    pin_memory = device.type == 'cuda'  # Only pin memory for CUDA
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=pin_memory)
    
    # Train models
    models_to_train = ["mesonet", "xception", "efficientnet"]
    
    for model_name in models_to_train:
        logger.info(f"Training {model_name} with advanced techniques...")
        trainer = AdvancedModelTrainer(model_name, device)
        best_acc = trainer.train(train_loader, val_loader, epochs=30)
        logger.info(f"{model_name} training completed with best accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()
