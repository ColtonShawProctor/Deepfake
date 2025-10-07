#!/usr/bin/env python3
"""
Simple but effective GPU training script
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

# Add the app directory to the path
sys.path.append('app')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleCelebDFDataset(Dataset):
    """Simple dataset for Celeb-DF-v2 videos"""
    
    def __init__(self, video_dir, transform=None, max_frames_per_video=5):
        self.video_dir = Path(video_dir)
        self.transform = transform
        self.max_frames_per_video = max_frames_per_video
        
        # Get all video files
        self.video_files = []
        for video_file in self.video_dir.rglob("*.mp4"):
            self.video_files.append(video_file)
        
        logger.info(f"Found {len(self.video_files)} videos in {video_dir}")
    
    def __len__(self):
        return len(self.video_files)
    
    def extract_frames(self, video_path, num_frames=5):
        """Extract frames from video"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            return []
        
        # Sample frames evenly
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
        
        # Determine label based on directory
        if "real" in str(video_path):
            label = 0  # Real
        else:
            label = 1  # Fake
        
        # Extract frames
        frames = self.extract_frames(video_path, self.max_frames_per_video)
        
        if not frames:
            # Return a dummy frame if extraction fails
            frames = [np.zeros((224, 224, 3), dtype=np.uint8)]
        
        # Use random frame
        frame = random.choice(frames)
        
        # Convert to PIL Image
        image = Image.fromarray(frame)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)

def get_simple_transforms():
    """Get simple but effective transforms"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_val_transforms():
    """Get validation transforms"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

class SimpleMesoNet(nn.Module):
    """Simple but effective MesoNet"""
    
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

class SimpleXception(nn.Module):
    """Simple Xception with transfer learning"""
    
    def __init__(self, num_classes=2):
        super(SimpleXception, self).__init__()
        
        # Load pre-trained Xception
        self.backbone = models.xception(pretrained=True)
        
        # Freeze early layers
        for param in list(self.backbone.parameters())[:-10]:
            param.requires_grad = False
        
        # Replace classifier
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

def train_model(model, train_loader, val_loader, model_name, device, epochs=20):
    """Train a model with GPU acceleration"""
    model.to(device)
    
    # Use different learning rates for different parts
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
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
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
                data, target = data.to(device), target.to(device)
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
            torch.save(model.state_dict(), f'models/{model_name}_weights_simple.pth')
            logger.info(f'New best model saved for {model_name} with acc: {val_acc:.2f}%')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f'Early stopping at epoch {epoch}')
                break
        
        scheduler.step()
    
    return best_val_acc

def main():
    """Main training function"""
    logger.info("Starting simple GPU training...")
    
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
    train_transform = get_simple_transforms()
    val_transform = get_val_transforms()
    
    real_train_dataset = SimpleCelebDFDataset(real_videos, transform=train_transform)
    fake_train_dataset = SimpleCelebDFDataset(fake_videos, transform=train_transform)
    real_val_dataset = SimpleCelebDFDataset(real_videos, transform=val_transform)
    fake_val_dataset = SimpleCelebDFDataset(fake_videos, transform=val_transform)
    
    # Combine datasets
    from torch.utils.data import ConcatDataset
    train_dataset = ConcatDataset([real_train_dataset, fake_train_dataset])
    val_dataset = ConcatDataset([real_val_dataset, fake_val_dataset])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # Train models
    models_to_train = [
        ("mesonet", SimpleMesoNet),
        ("xception", SimpleXception)
    ]
    
    for model_name, model_class in models_to_train:
        logger.info(f"Training {model_name}...")
        model = model_class()
        best_acc = train_model(model, train_loader, val_loader, model_name, device, epochs=20)
        logger.info(f"{model_name} training completed with best accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()
