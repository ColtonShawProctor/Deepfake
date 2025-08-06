"""
Dataset Management System for Deepfake Detection Training

This module provides comprehensive dataset management including:
- Dataset downloading and organization
- Data preprocessing pipelines for different model types
- Data augmentation specific to deepfake detection
- Train/validation/test splits with stratification
- Quality filtering and validation
"""

import os
import sys
import logging
import json
import shutil
import zipfile
import tarfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageEnhance
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import requests
from tqdm import tqdm
import face_recognition
from sklearn.model_selection import train_test_split
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetConfig:
    """Configuration for dataset management"""
    
    def __init__(self):
        self.datasets = {
            "faceforensics": {
                "name": "FaceForensics++",
                "url": "https://github.com/ondyari/FaceForensics",
                "size": "~1.7GB",
                "real_count": 1000,
                "fake_count": 4000,
                "manipulation_types": ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"],
                "quality_levels": ["c23", "c40"],
                "download_method": "git_clone"
            },
            "dfdc": {
                "name": "DeepFake Detection Challenge",
                "url": "https://ai.facebook.com/datasets/dfdc/",
                "size": "~100GB",
                "real_count": 100000,
                "fake_count": 100000,
                "download_method": "manual"
            },
            "celebdf": {
                "name": "Celeb-DF",
                "url": "https://github.com/yuezunli/celeb-deepfakeforensics",
                "size": "~1.5GB",
                "real_count": 590,
                "fake_count": 5639,
                "download_method": "git_clone"
            }
        }
        
        # Model-specific preprocessing configurations
        self.model_configs = {
            "mesonet": {
                "input_size": (256, 256),
                "normalization": "imagenet",
                "augmentation": "basic"
            },
            "xception": {
                "input_size": (299, 299),
                "normalization": "imagenet",
                "augmentation": "advanced"
            },
            "efficientnet": {
                "input_size": (224, 224),
                "normalization": "imagenet",
                "augmentation": "advanced"
            },
            "f3net": {
                "input_size": (224, 224),
                "normalization": "imagenet",
                "augmentation": "frequency_domain"
            }
        }

class DatasetDownloader:
    """Handles downloading and organizing datasets"""
    
    def __init__(self, datasets_dir: str = "datasets"):
        self.datasets_dir = Path(datasets_dir)
        self.datasets_dir.mkdir(exist_ok=True)
        self.config = DatasetConfig()
    
    def download_file(self, url: str, filename: str, chunk_size: int = 8192) -> bool:
        """Download a file with progress bar"""
        try:
            logger.info(f"Downloading {filename} from {url}")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            filepath = self.datasets_dir / filename
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            logger.info(f"Successfully downloaded {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {filename}: {str(e)}")
            return False
    
    def extract_archive(self, archive_path: Path, extract_dir: Path) -> bool:
        """Extract compressed archive"""
        try:
            logger.info(f"Extracting {archive_path}")
            
            if archive_path.suffix == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
            elif archive_path.suffix in ['.tar', '.tar.gz', '.tgz']:
                with tarfile.open(archive_path, 'r:*') as tar_ref:
                    tar_ref.extractall(extract_dir)
            else:
                logger.error(f"Unsupported archive format: {archive_path.suffix}")
                return False
            
            logger.info(f"Successfully extracted to {extract_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to extract {archive_path}: {str(e)}")
            return False
    
    def git_clone(self, repo_url: str, target_dir: str) -> bool:
        """Clone a git repository"""
        try:
            logger.info(f"Cloning {repo_url} to {target_dir}")
            
            import subprocess
            result = subprocess.run([
                'git', 'clone', repo_url, target_dir
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Successfully cloned to {target_dir}")
                return True
            else:
                logger.error(f"Git clone failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to clone repository: {str(e)}")
            return False
    
    def download_faceforensics(self) -> bool:
        """Download FaceForensics++ dataset"""
        logger.info("Setting up FaceForensics++ dataset...")
        
        ff_dir = self.datasets_dir / "faceforensics"
        ff_dir.mkdir(exist_ok=True)
        
        # Clone the repository
        if not self.git_clone(
            "https://github.com/ondyari/FaceForensics.git",
            str(ff_dir / "repo")
        ):
            return False
        
        logger.info("FaceForensics++ repository cloned. You'll need to:")
        logger.info("1. Follow the instructions in the README to download the actual dataset")
        logger.info("2. The dataset requires manual download due to size and licensing")
        
        return True
    
    def download_celebdf(self) -> bool:
        """Download Celeb-DF dataset"""
        logger.info("Setting up Celeb-DF dataset...")
        
        celebdf_dir = self.datasets_dir / "celebdf"
        celebdf_dir.mkdir(exist_ok=True)
        
        # Clone the repository
        if not self.git_clone(
            "https://github.com/yuezunli/celeb-deepfakeforensics.git",
            str(celebdf_dir / "repo")
        ):
            return False
        
        logger.info("Celeb-DF repository cloned. You'll need to:")
        logger.info("1. Follow the instructions in the README to download the actual dataset")
        logger.info("2. The dataset requires manual download due to size and licensing")
        
        return True
    
    def download_all(self) -> bool:
        """Download all available datasets"""
        logger.info("Starting download of all datasets...")
        
        success_count = 0
        total_count = len(self.config.datasets)
        
        for dataset_id, dataset_info in self.config.datasets.items():
            logger.info(f"Processing {dataset_info['name']}...")
            
            if dataset_info["download_method"] == "git_clone":
                if dataset_id == "faceforensics":
                    success = self.download_faceforensics()
                elif dataset_id == "celebdf":
                    success = self.download_celebdf()
                else:
                    success = False
            else:
                logger.info(f"Skipping {dataset_id} - requires manual download")
                success = True
        
            if success:
                success_count += 1
        
        logger.info(f"Download completed: {success_count}/{total_count} datasets processed")
        return success_count > 0

class DataPreprocessor:
    """Handles data preprocessing for different model types"""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.face_detector = None
        self._init_face_detector()
    
    def _init_face_detector(self):
        """Initialize face detector"""
        try:
            import face_recognition
            self.face_detector = "face_recognition"
        except ImportError:
            try:
                import cv2
                self.face_detector = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
            except:
                logger.warning("No face detection available")
                self.face_detector = None
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in image and return bounding boxes"""
        if self.face_detector == "face_recognition":
            face_locations = face_recognition.face_locations(image)
            return [(top, right, bottom, left) for top, right, bottom, left in face_locations]
        elif self.face_detector is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            faces = self.face_detector.detectMultiScale(gray, 1.1, 4)
            return [(y, x + w, y + h, x) for x, y, w, h in faces]
        else:
            # Return full image if no face detector available
            h, w = image.shape[:2]
            return [(0, w, h, 0)]
    
    def crop_face(self, image: np.ndarray, bbox: Tuple[int, int, int, int], 
                  target_size: Tuple[int, int]) -> np.ndarray:
        """Crop and resize face region"""
        top, right, bottom, left = bbox
        face_crop = image[top:bottom, left:right]
        
        if face_crop.size == 0:
            # If no face detected, use center crop
            h, w = image.shape[:2]
            center_h, center_w = h // 2, w // 2
            crop_size = min(h, w) // 2
            face_crop = image[
                center_h - crop_size:center_h + crop_size,
                center_w - crop_size:center_w + crop_size
            ]
        
        # Resize to target size
        face_crop = cv2.resize(face_crop, target_size)
        return face_crop
    
    def get_preprocessing_pipeline(self, model_type: str) -> Dict[str, Any]:
        """Get preprocessing pipeline for specific model type"""
        if model_type not in self.config.model_configs:
            raise ValueError(f"Unknown model type: {model_type}")
        
        config = self.config.model_configs[model_type]
        
        # Base transforms
        base_transforms = [
            A.Resize(config["input_size"][0], config["input_size"][1]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ]
        
        # Add augmentation based on model type
        if config["augmentation"] == "basic":
            augmentation = [
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.Rotate(limit=15, p=0.3)
            ]
        elif config["augmentation"] == "advanced":
            augmentation = [
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.Rotate(limit=15, p=0.3),
                A.ShiftScaleRotate(p=0.3),
                A.OneOf([
                    A.GaussNoise(),
                    A.GaussianBlur(),
                    A.MotionBlur()
                ], p=0.3),
                A.OneOf([
                    A.RandomGamma(),
                    A.RandomBrightnessContrast(),
                    A.HueSaturationValue()
                ], p=0.3)
            ]
        elif config["augmentation"] == "frequency_domain":
            augmentation = [
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.OneOf([
                    A.GaussNoise(),
                    A.GaussianBlur(),
                    A.MotionBlur()
                ], p=0.3),
                # Frequency domain specific augmentations
                A.OneOf([
                    A.JpegCompression(quality_lower=70, quality_upper=95),
                    A.ISONoise(),
                    A.MultiplicativeNoise()
                ], p=0.3)
            ]
        else:
            augmentation = []
        
        return {
            "input_size": config["input_size"],
            "transforms": A.Compose(augmentation + base_transforms),
            "target_transforms": A.Compose(base_transforms)
        }
    
    def preprocess_image(self, image_path: str, model_type: str, 
                        detect_faces: bool = True) -> List[np.ndarray]:
        """Preprocess image for specific model type"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return []
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get preprocessing pipeline
        pipeline = self.get_preprocessing_pipeline(model_type)
        
        # Detect faces if requested
        if detect_faces:
            face_bboxes = self.detect_faces(image)
        else:
            face_bboxes = [(0, image.shape[1], image.shape[0], 0)]
        
        processed_images = []
        for bbox in face_bboxes:
            # Crop face
            face_crop = self.crop_face(image, bbox, pipeline["input_size"])
            
            # Apply transforms
            transformed = pipeline["transforms"](image=face_crop)
            processed_images.append(transformed["image"])
        
        return processed_images

class DeepfakeDataset(Dataset):
    """Custom dataset for deepfake detection"""
    
    def __init__(self, data_dir: str, transform=None, model_type: str = "efficientnet"):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.model_type = model_type
        
        # Load dataset metadata
        self.samples = self._load_samples()
        
        logger.info(f"Loaded {len(self.samples)} samples from {data_dir}")
    
    def _load_samples(self) -> List[Tuple[str, int]]:
        """Load dataset samples and labels"""
        samples = []
        
        # Look for metadata file
        metadata_file = self.data_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            for sample_info in metadata:
                image_path = self.data_dir / sample_info["path"]
                if image_path.exists():
                    samples.append((str(image_path), sample_info["label"]))
        else:
            # Fallback: scan directory structure
            real_dir = self.data_dir / "real"
            fake_dir = self.data_dir / "fake"
            
            if real_dir.exists():
                for img_path in real_dir.glob("*.jpg"):
                    samples.append((str(img_path), 0))
            
            if fake_dir.exists():
                for img_path in fake_dir.glob("*.jpg"):
                    samples.append((str(img_path), 1))
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.float32)

class DatasetManager:
    """Main dataset management class"""
    
    def __init__(self, datasets_dir: str = "datasets"):
        self.datasets_dir = Path(datasets_dir)
        self.config = DatasetConfig()
        self.downloader = DatasetDownloader(datasets_dir)
        self.preprocessor = DataPreprocessor(self.config)
    
    def create_train_val_test_splits(self, dataset_path: str, 
                                   train_ratio: float = 0.7,
                                   val_ratio: float = 0.15,
                                   test_ratio: float = 0.15,
                                   random_state: int = 42) -> Dict[str, List[str]]:
        """Create train/validation/test splits with stratification"""
        dataset = DeepfakeDataset(dataset_path)
        
        # Extract labels for stratification
        labels = [label for _, label in dataset.samples]
        
        # Create splits
        train_indices, temp_indices = train_test_split(
            range(len(dataset)), 
            test_size=(1 - train_ratio),
            stratify=labels,
            random_state=random_state
        )
        
        val_size = val_ratio / (val_ratio + test_ratio)
        val_indices, test_indices = train_test_split(
            temp_indices,
            test_size=(1 - val_size),
            stratify=[labels[i] for i in temp_indices],
            random_state=random_state
        )
        
        # Create split datasets
        splits = {
            "train": [dataset.samples[i] for i in train_indices],
            "val": [dataset.samples[i] for i in val_indices],
            "test": [dataset.samples[i] for i in test_indices]
        }
        
        # Save splits
        splits_file = Path(dataset_path) / "splits.json"
        with open(splits_file, 'w') as f:
            json.dump(splits, f, indent=2)
        
        logger.info(f"Created splits: train={len(splits['train'])}, "
                   f"val={len(splits['val'])}, test={len(splits['test'])}")
        
        return splits
    
    def get_data_loader(self, dataset_path: str, split: str = "train",
                       model_type: str = "efficientnet", batch_size: int = 32,
                       shuffle: bool = True, num_workers: int = 4) -> DataLoader:
        """Get data loader for specific split and model type"""
        # Get preprocessing pipeline
        pipeline = self.preprocessor.get_preprocessing_pipeline(model_type)
        
        # Create dataset
        dataset = DeepfakeDataset(dataset_path, transform=pipeline["transforms"], 
                                model_type=model_type)
        
        # Create data loader
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return loader
    
    def validate_dataset_quality(self, dataset_path: str) -> Dict[str, Any]:
        """Validate dataset quality and return statistics"""
        dataset = DeepfakeDataset(dataset_path)
        
        stats = {
            "total_samples": len(dataset),
            "real_samples": 0,
            "fake_samples": 0,
            "image_sizes": [],
            "corrupted_images": 0
        }
        
        for image_path, label in dataset.samples:
            # Count labels
            if label == 0:
                stats["real_samples"] += 1
            else:
                stats["fake_samples"] += 1
            
            # Check image quality
            try:
                image = Image.open(image_path)
                stats["image_sizes"].append(image.size)
            except Exception as e:
                stats["corrupted_images"] += 1
                logger.warning(f"Corrupted image: {image_path}")
        
        # Calculate additional statistics
        if stats["image_sizes"]:
            sizes = np.array(stats["image_sizes"])
            stats["avg_width"] = np.mean(sizes[:, 0])
            stats["avg_height"] = np.mean(sizes[:, 1])
            stats["min_size"] = sizes.min()
            stats["max_size"] = sizes.max()
        
        stats["real_ratio"] = stats["real_samples"] / stats["total_samples"]
        stats["fake_ratio"] = stats["fake_samples"] / stats["total_samples"]
        
        return stats

def main():
    """Main function for dataset management"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Dataset Management for Deepfake Detection")
    parser.add_argument("--action", choices=["download", "preprocess", "split", "validate"],
                       required=True, help="Action to perform")
    parser.add_argument("--datasets-dir", default="datasets", help="Datasets directory")
    parser.add_argument("--dataset", help="Specific dataset to process")
    parser.add_argument("--model-type", default="efficientnet", 
                       choices=["mesonet", "xception", "efficientnet", "f3net"],
                       help="Model type for preprocessing")
    
    args = parser.parse_args()
    
    manager = DatasetManager(args.datasets_dir)
    
    if args.action == "download":
        success = manager.downloader.download_all()
        if success:
            logger.info("Dataset download completed successfully!")
        else:
            logger.error("Dataset download failed!")
    
    elif args.action == "preprocess":
        if not args.dataset:
            logger.error("Dataset path required for preprocessing")
            return
        
        # Preprocess dataset for specific model type
        logger.info(f"Preprocessing {args.dataset} for {args.model_type}")
        # Implementation would go here
    
    elif args.action == "split":
        if not args.dataset:
            logger.error("Dataset path required for splitting")
            return
        
        splits = manager.create_train_val_test_splits(args.dataset)
        logger.info("Dataset splits created successfully!")
    
    elif args.action == "validate":
        if not args.dataset:
            logger.error("Dataset path required for validation")
            return
        
        stats = manager.validate_dataset_quality(args.dataset)
        logger.info("Dataset validation completed!")
        logger.info(f"Statistics: {json.dumps(stats, indent=2)}")

if __name__ == "__main__":
    main() 