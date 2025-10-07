#!/usr/bin/env python3
"""
Fix ResNet overfitting by retraining with proper regularization
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
    """Dataset for Celeb-DF-v2 videos with proper regularization"""
    
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
        """Extract frames from video with better sampling"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            return []
        
        # Sample frames more strategically
        if total_frames <= num_frames:
            frame_indices = list(range(total_frames))
        else:
            # Sample from beginning, middle, and end
            frame_indices = []
            frame_indices.extend([0, total_frames//4, total_frames//2, 3*total_frames//4, total_frames-1])
            frame_indices = frame_indices[:num_frames]
        
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
        
        # Use random frame for variety
        frame = random.choice(frames)
        
        # Convert to PIL Image
        image = Image.fromarray(frame)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.float32)

class ImprovedResNetDetector(nn.Module):
    """Improved ResNet with better regularization"""
    
    def __init__(self, num_classes=1):
        super(ImprovedResNetDetector, self).__init__()
        
        # Load pre-trained ResNet-50
        self.backbone = models.resnet50(pretrained=True)
        
        # Freeze early layers to prevent overfitting
        for param in list(self.backbone.parameters())[:-10]:
            param.requires_grad = False
        
        # Modify the classifier with better regularization
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.7),  # Higher dropout
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.backbone(x)

def get_improved_transforms():
    """Get transforms with better augmentation to prevent overfitting"""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2)  # Random erasing for regularization
    ])

def train_improved_resnet():
    """Train improved ResNet with better regularization"""
    logger.info("Training improved ResNet with better regularization...")
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Dataset paths
    real_videos = "Celeb-DF-v2/Celeb-real"
    fake_videos = "Celeb-DF-v2/Celeb-synthesis"
    
    # Check if datasets exist
    if not os.path.exists(real_videos) or not os.path.exists(fake_videos):
        logger.error("Celeb-DF-v2 dataset not found.")
        return
    
    # Get transforms
    transform = get_improved_transforms()
    
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
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    
    # Create improved model
    model = ImprovedResNetDetector()
    
    # Training setup - Use MPS (Apple Silicon GPU) if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    model.to(device)
    
    # Use better loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)  # Lower LR, weight decay
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)
    
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    logger.info(f"Training on device: {device}")
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    for epoch in range(50):
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
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            pred = (output.squeeze() > 0.5).float()
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
            torch.save(model.state_dict(), 'models/resnet_weights_fixed.pth')
            logger.info(f'New best model saved at epoch {epoch}')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f'Early stopping at epoch {epoch}')
                break
        
        scheduler.step()
    
    logger.info("Improved ResNet training completed!")

def test_improved_resnet():
    """Test the improved ResNet model"""
    logger.info("Testing improved ResNet model...")
    
    # Load the improved model
    model = ImprovedResNetDetector()
    model.load_state_dict(torch.load('models/resnet_weights_fixed.pth', map_location='cpu'))
    model.eval()
    
    # Test with demo images
    demo_images = [
        "demo_images/01_fake_low_confidence.jpg",
        "demo_images/02_fake_medium_confidence.jpg", 
        "demo_images/03_fake_high_confidence.jpg",
        "demo_images/04_real_medium_confidence.jpg",
        "demo_images/05_real_medium_confidence.jpg",
        "demo_images/06_real_low_confidence.jpg"
    ]
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("\n📊 Testing Improved ResNet predictions:")
    print("-" * 60)
    
    for img_path in demo_images:
        if os.path.exists(img_path):
            try:
                image = Image.open(img_path)
                input_tensor = transform(image).unsqueeze(0)
                
                with torch.no_grad():
                    output = model(input_tensor)
                    confidence = output.squeeze().item() * 100.0
                
                filename = os.path.basename(img_path)
                print(f"{filename:40} | Confidence: {confidence:6.2f}% | Deepfake: {confidence > 50.0}")
                
            except Exception as e:
                print(f"❌ Error processing {img_path}: {e}")
        else:
            print(f"⚠️  File not found: {img_path}")

if __name__ == "__main__":
    train_improved_resnet()
    test_improved_resnet()
