#!/usr/bin/env python3
"""
Generate synthetic training data for deepfake detection

This script creates synthetic deepfake images and real images for training
the deepfake detection models with more data.
"""

import os
import sys
import logging
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import cv2
from pathlib import Path
import random
from typing import List, Tuple
import json
import io

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SyntheticDataGenerator:
    """Generate synthetic training data for deepfake detection"""
    
    def __init__(self, output_dir: str = "training_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "real").mkdir(exist_ok=True)
        (self.output_dir / "fake").mkdir(exist_ok=True)
        
        self.metadata = {
            "real": [],
            "fake": []
        }
    
    def generate_synthetic_face(self, size: Tuple[int, int] = (224, 224)) -> Image.Image:
        """Generate a synthetic face-like image"""
        # Create base face structure
        img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
        
        # Add skin tone
        skin_color = random.choice([
            [255, 220, 177],  # Light skin
            [255, 205, 148],  # Medium light
            [234, 192, 134],  # Medium
            [213, 170, 115],  # Medium dark
            [184, 151, 120],  # Dark
        ])
        img[:, :] = skin_color
        
        # Add face features (simplified)
        center_x, center_y = size[1] // 2, size[0] // 2
        
        # Eyes
        eye_color = [50, 50, 50]
        img[center_y-20:center_y-10, center_x-30:center_x-10] = eye_color
        img[center_y-20:center_y-10, center_x+10:center_x+30] = eye_color
        
        # Nose
        nose_color = [200, 180, 160]
        img[center_y-5:center_y+15, center_x-5:center_x+5] = nose_color
        
        # Mouth
        mouth_color = [150, 100, 100]
        img[center_y+20:center_y+30, center_x-15:center_x+15] = mouth_color
        
        # Add some noise and blur for realism
        img = cv2.GaussianBlur(img, (3, 3), 0.5)
        noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
        img = np.clip(img + noise, 0, 255)
        
        return Image.fromarray(img)
    
    def apply_deepfake_artifacts(self, image: Image.Image) -> Image.Image:
        """Apply common deepfake artifacts to an image"""
        artifacts = random.choice([
            "compression",
            "blending",
            "lighting",
            "texture",
            "edge",
            "color"
        ])
        
        if artifacts == "compression":
            # Simulate compression artifacts
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=random.randint(30, 70))
            buffer.seek(0)
            return Image.open(buffer)
        
        elif artifacts == "blending":
            # Simulate poor face blending
            mask = Image.new('L', image.size, 128)
            mask = mask.filter(ImageFilter.GaussianBlur(radius=5))
            return Image.composite(image, image.filter(ImageFilter.BLUR), mask)
        
        elif artifacts == "lighting":
            # Simulate lighting inconsistencies
            enhancer = ImageEnhance.Brightness(image)
            factor = random.uniform(0.8, 1.2)
            return enhancer.enhance(factor)
        
        elif artifacts == "texture":
            # Simulate texture artifacts
            return image.filter(ImageFilter.EDGE_ENHANCE_MORE)
        
        elif artifacts == "edge":
            # Simulate edge artifacts
            return image.filter(ImageFilter.FIND_EDGES)
        
        elif artifacts == "color":
            # Simulate color inconsistencies
            enhancer = ImageEnhance.Color(image)
            factor = random.uniform(0.7, 1.3)
            return enhancer.enhance(factor)
        
        return image
    
    def generate_real_image(self, size: Tuple[int, int] = (224, 224)) -> Image.Image:
        """Generate a realistic-looking image"""
        # Create a more realistic face
        img = self.generate_synthetic_face(size)
        
        # Add natural variations
        if random.random() > 0.5:
            # Add slight blur
            img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        if random.random() > 0.5:
            # Add slight brightness variation
            enhancer = ImageEnhance.Brightness(img)
            factor = random.uniform(0.9, 1.1)
            img = enhancer.enhance(factor)
        
        return img
    
    def generate_fake_image(self, size: Tuple[int, int] = (224, 224)) -> Image.Image:
        """Generate a deepfake image with artifacts"""
        # Start with a realistic face
        img = self.generate_synthetic_face(size)
        
        # Apply deepfake artifacts
        img = self.apply_deepfake_artifacts(img)
        
        # Add additional artifacts
        if random.random() > 0.7:
            # Add more severe artifacts
            img = self.apply_deepfake_artifacts(img)
        
        return img
    
    def generate_dataset(self, num_real: int = 1000, num_fake: int = 1000):
        """Generate a complete training dataset"""
        logger.info(f"Generating {num_real} real images and {num_fake} fake images...")
        
        # Generate real images
        for i in range(num_real):
            img = self.generate_real_image()
            filename = f"real_{i:06d}.jpg"
            filepath = self.output_dir / "real" / filename
            img.save(filepath, "JPEG", quality=95)
            
            self.metadata["real"].append({
                "filename": filename,
                "path": str(filepath),
                "expected": False,
                "description": "Synthetic real face",
                "source": "Generated",
                "size": [224, 224]
            })
            
            if (i + 1) % 100 == 0:
                logger.info(f"Generated {i + 1} real images")
        
        # Generate fake images
        for i in range(num_fake):
            img = self.generate_fake_image()
            filename = f"fake_{i:06d}.jpg"
            filepath = self.output_dir / "fake" / filename
            img.save(filepath, "JPEG", quality=95)
            
            self.metadata["fake"].append({
                "filename": filename,
                "path": str(filepath),
                "expected": True,
                "description": "Synthetic deepfake with artifacts",
                "source": "Generated",
                "size": [224, 224]
            })
            
            if (i + 1) % 100 == 0:
                logger.info(f"Generated {i + 1} fake images")
        
        # Save metadata
        metadata_file = self.output_dir / "training_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        logger.info(f"Dataset generated successfully!")
        logger.info(f"Real images: {len(self.metadata['real'])}")
        logger.info(f"Fake images: {len(self.metadata['fake'])}")
        logger.info(f"Metadata saved to: {metadata_file}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic training data")
    parser.add_argument("--output-dir", default="training_data", help="Output directory")
    parser.add_argument("--num-real", type=int, default=1000, help="Number of real images")
    parser.add_argument("--num-fake", type=int, default=1000, help="Number of fake images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Generate dataset
    generator = SyntheticDataGenerator(args.output_dir)
    generator.generate_dataset(args.num_real, args.num_fake)

if __name__ == "__main__":
    main() 