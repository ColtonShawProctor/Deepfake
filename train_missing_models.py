#!/usr/bin/env python3
"""
Train missing MesoNet and Xception models using Celeb-DF-v2 dataset
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

class CelebDFDataset(Dataset):
    """Dataset for Celeb-DF-v2 videos"""
    
    def __init__(self, video_dir, transform=None, max_frames_per_video=10):
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
    
    def extract_frames(self, video_path, num_frames=10):
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
        
        # Use the first frame for now (can be extended to use multiple frames)
        frame = frames[0]
        
        # Convert to PIL Image
        image = Image.fromarray(frame)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.float32)

class MesoNet(nn.Module):
    """MesoNet architecture for deepfake detection"""
    
    def __init__(self, num_classes=1):
        super(MesoNet, self).__init__()
        
        # First block
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Second block
        self.conv2 = nn.Conv2d(8, 8, 5, padding=2)
        self.bn2 = nn.BatchNorm2d(8)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Third block
        self.conv3 = nn.Conv2d(8, 8, 5, padding=2)
        self.bn3 = nn.BatchNorm2d(8)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Fourth block
        self.conv4 = nn.Conv2d(8, 8, 5, padding=2)
        self.bn4 = nn.BatchNorm2d(8)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.fc1 = nn.Linear(8, 16)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(16, num_classes)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
        
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        
        return x

class XceptionDetector(nn.Module):
    """Xception-based deepfake detector"""
    
    def __init__(self, num_classes=1):
        super(XceptionDetector, self).__init__()
        
        # Load pre-trained Xception
        self.backbone = models.xception(pretrained=True)
        
        # Modify the classifier
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.backbone(x)

def get_transforms(model_name):
    """Get appropriate transforms for each model"""
    if model_name == "mesonet":
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif model_name == "xception":
        return transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def train_model(model, train_loader, val_loader, model_name, epochs=50):
    """Train a model"""
    # Use MPS (Apple Silicon GPU) if available, otherwise CUDA, otherwise CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    model.to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    best_val_loss = float('inf')
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
            loss = criterion(output.squeeze(), target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = (output.squeeze() > 0.5).float()
            train_correct += pred.eq(target).sum().item()
            train_total += target.size(0)
            
            if batch_idx % 10 == 0:
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
                loss = criterion(output.squeeze(), target)
                
                val_loss += loss.item()
                pred = (output.squeeze() > 0.5).float()
                val_correct += pred.eq(target).sum().item()
                val_total += target.size(0)
        
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        val_loss /= len(val_loader)
        
        logger.info(f'Epoch {epoch}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, Val Loss: {val_loss:.4f}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), f'models/{model_name}_weights.pth')
            logger.info(f'New best model saved for {model_name}')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f'Early stopping for {model_name} at epoch {epoch}')
                break
        
        scheduler.step()

def main():
    """Main training function"""
    logger.info("Starting training of missing models...")
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Dataset paths
    real_videos = "Celeb-DF-v2/Celeb-real"
    fake_videos = "Celeb-DF-v2/Celeb-synthesis"
    
    # Check if datasets exist
    if not os.path.exists(real_videos) or not os.path.exists(fake_videos):
        logger.error("Celeb-DF-v2 dataset not found. Please ensure the dataset is available.")
        return
    
    # Create datasets
    models_to_train = ["mesonet", "xception"]
    
    for model_name in models_to_train:
        logger.info(f"Training {model_name}...")
        
        # Get transforms
        transform = get_transforms(model_name)
        
        # Create datasets
        real_dataset = CelebDFDataset(real_videos, transform=transform)
        fake_dataset = CelebDFDataset(fake_videos, transform=transform)
        
        # Combine datasets
        from torch.utils.data import ConcatDataset
        full_dataset = ConcatDataset([real_dataset, fake_dataset])
        
        # Split dataset
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        
        # Create data loaders - larger batch size for GPU
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
        
        # Create model
        if model_name == "mesonet":
            model = MesoNet()
        elif model_name == "xception":
            model = XceptionDetector()
        
        # Train model
        train_model(model, train_loader, val_loader, model_name, epochs=50)
        
        logger.info(f"Training completed for {model_name}")

if __name__ == "__main__":
    main()
