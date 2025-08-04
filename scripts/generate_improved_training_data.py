#!/usr/bin/env python3
"""
Generate improved training data for deepfake detection

This script creates a much more diverse and realistic training dataset
with better variety in both real and fake samples.
"""

import os
import sys
import logging
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw, ImageFont
import cv2
from pathlib import Path
import random
from typing import List, Tuple
import json
import requests
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImprovedDataGenerator:
    """Generate improved training data for deepfake detection"""
    
    def __init__(self, output_dir: str = "improved_training_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "real").mkdir(exist_ok=True)
        (self.output_dir / "fake").mkdir(exist_ok=True)
        
        # Celebrities for more realistic training
        self.celebrities = [
            "tom_cruise", "angelina_jolie", "brad_pitt", "jennifer_lawrence",
            "leonardo_dicaprio", "scarlett_johansson", "johnny_depp", "emma_watson",
            "chris_hemsworth", "natalie_portman", "robert_downey_jr", "margot_robbie"
        ]
        
        # Deepfake techniques to simulate
        self.deepfake_techniques = [
            "face_swap", "deepfake_gan", "style_transfer", "face_morphing",
            "expression_transfer", "age_progression", "gender_swap", "ethnicity_change"
        ]
    
    def download_celebrity_image(self, celebrity: str) -> Image.Image:
        """Download a celebrity image (simulated)"""
        # Create a realistic celebrity-like image
        img = Image.new('RGB', (512, 512), color='white')
        draw = ImageDraw.Draw(img)
        
        # Simulate a face with basic shapes
        # Face outline
        draw.ellipse([100, 100, 412, 412], outline='black', width=3)
        
        # Eyes
        draw.ellipse([150, 200, 200, 250], fill='blue')
        draw.ellipse([312, 200, 362, 250], fill='blue')
        
        # Nose
        draw.polygon([(256, 250), (246, 300), (266, 300)], fill='pink')
        
        # Mouth
        draw.arc([200, 320, 312, 380], start=0, end=180, fill='red', width=3)
        
        # Add celebrity name
        try:
            font = ImageFont.load_default()
            draw.text((200, 450), f"{celebrity.replace('_', ' ').title()}", 
                     fill='black', font=font)
        except:
            draw.text((200, 450), f"{celebrity.replace('_', ' ').title()}", fill='black')
        
        return img
    
    def create_realistic_fake(self, base_image: Image.Image, technique: str) -> Image.Image:
        """Create a realistic deepfake using various techniques"""
        img = base_image.copy()
        
        if technique == "face_swap":
            # Simulate face swap artifacts
            # Add slight color mismatch
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(1.2)
            
            # Add slight blur around edges
            img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
            
            # Add subtle artifacts
            draw = ImageDraw.Draw(img)
            for _ in range(5):
                x = random.randint(0, img.width)
                y = random.randint(0, img.height)
                draw.point((x, y), fill='white')
        
        elif technique == "deepfake_gan":
            # Simulate GAN artifacts
            # Add checkerboard pattern
            draw = ImageDraw.Draw(img)
            for i in range(0, img.width, 20):
                for j in range(0, img.height, 20):
                    if (i + j) % 40 == 0:
                        draw.rectangle([i, j, i+10, j+10], fill='gray')
            
            # Add color artifacts
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.3)
        
        elif technique == "style_transfer":
            # Simulate style transfer
            # Add painterly effect
            img = img.filter(ImageFilter.EDGE_ENHANCE)
            img = img.filter(ImageFilter.SMOOTH)
            
            # Adjust colors
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(0.8)
        
        elif technique == "face_morphing":
            # Simulate face morphing
            # Add warping effect
            img = img.transform(img.size, Image.AFFINE, 
                              (1, 0.1, 0, 0, 1, 0), Image.BICUBIC)
            
            # Add blending artifacts
            draw = ImageDraw.Draw(img)
            for _ in range(10):
                x = random.randint(0, img.width)
                y = random.randint(0, img.height)
                draw.ellipse([x-5, y-5, x+5, y+5], fill='white', outline='gray')
        
        return img
    
    def generate_diverse_real_samples(self, num_samples: int = 2000) -> List[dict]:
        """Generate diverse real samples"""
        logger.info(f"Generating {num_samples} diverse real samples...")
        
        samples = []
        for i in range(num_samples):
            # Choose random celebrity
            celebrity = random.choice(self.celebrities)
            
            # Create base image
            img = self.download_celebrity_image(celebrity)
            
            # Apply random variations
            variations = [
                "brightness", "contrast", "saturation", "rotation",
                "crop", "noise", "blur", "sharpness"
            ]
            
            for variation in random.sample(variations, random.randint(1, 3)):
                if variation == "brightness":
                    enhancer = ImageEnhance.Brightness(img)
                    img = enhancer.enhance(random.uniform(0.8, 1.2))
                elif variation == "contrast":
                    enhancer = ImageEnhance.Contrast(img)
                    img = enhancer.enhance(random.uniform(0.8, 1.3))
                elif variation == "saturation":
                    enhancer = ImageEnhance.Color(img)
                    img = enhancer.enhance(random.uniform(0.7, 1.2))
                elif variation == "rotation":
                    angle = random.uniform(-15, 15)
                    img = img.rotate(angle, expand=True)
                elif variation == "crop":
                    # Random crop
                    width, height = img.size
                    crop_size = min(width, height) - 50
                    left = random.randint(0, width - crop_size)
                    top = random.randint(0, height - crop_size)
                    img = img.crop((left, top, left + crop_size, top + crop_size))
                elif variation == "noise":
                    # Add noise
                    img_array = np.array(img)
                    noise = np.random.normal(0, 10, img_array.shape).astype(np.uint8)
                    img_array = np.clip(img_array + noise, 0, 255)
                    img = Image.fromarray(img_array)
                elif variation == "blur":
                    img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2)))
                elif variation == "sharpness":
                    img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150))
            
            # Resize to standard size
            img = img.resize((224, 224), Image.LANCZOS)
            
            # Save image
            filename = f"real_{i:06d}.jpg"
            filepath = self.output_dir / "real" / filename
            img.save(filepath, "JPEG", quality=95)
            
            samples.append({
                "path": str(filepath),
                "type": "real",
                "celebrity": celebrity,
                "variations": variations[:3],
                "filename": filename
            })
            
            if (i + 1) % 100 == 0:
                logger.info(f"Generated {i + 1} real samples")
        
        return samples
    
    def generate_diverse_fake_samples(self, num_samples: int = 2000) -> List[dict]:
        """Generate diverse fake samples"""
        logger.info(f"Generating {num_samples} diverse fake samples...")
        
        samples = []
        for i in range(num_samples):
            # Choose random celebrity as base
            celebrity = random.choice(self.celebrities)
            
            # Create base image
            base_img = self.download_celebrity_image(celebrity)
            
            # Choose random deepfake technique
            technique = random.choice(self.deepfake_techniques)
            
            # Create fake image
            fake_img = self.create_realistic_fake(base_img, technique)
            
            # Apply additional variations
            variations = ["brightness", "contrast", "saturation", "rotation", "crop"]
            for variation in random.sample(variations, random.randint(1, 2)):
                if variation == "brightness":
                    enhancer = ImageEnhance.Brightness(fake_img)
                    fake_img = enhancer.enhance(random.uniform(0.7, 1.3))
                elif variation == "contrast":
                    enhancer = ImageEnhance.Contrast(fake_img)
                    fake_img = enhancer.enhance(random.uniform(0.8, 1.4))
                elif variation == "saturation":
                    enhancer = ImageEnhance.Color(fake_img)
                    fake_img = enhancer.enhance(random.uniform(0.6, 1.3))
                elif variation == "rotation":
                    angle = random.uniform(-10, 10)
                    fake_img = fake_img.rotate(angle, expand=True)
                elif variation == "crop":
                    width, height = fake_img.size
                    crop_size = min(width, height) - 30
                    left = random.randint(0, width - crop_size)
                    top = random.randint(0, height - crop_size)
                    fake_img = fake_img.crop((left, top, left + crop_size, top + crop_size))
            
            # Resize to standard size
            fake_img = fake_img.resize((224, 224), Image.LANCZOS)
            
            # Save image
            filename = f"fake_{i:06d}.jpg"
            filepath = self.output_dir / "fake" / filename
            fake_img.save(filepath, "JPEG", quality=95)
            
            samples.append({
                "path": str(filepath),
                "type": "fake",
                "base_celebrity": celebrity,
                "technique": technique,
                "filename": filename
            })
            
            if (i + 1) % 100 == 0:
                logger.info(f"Generated {i + 1} fake samples")
        
        return samples
    
    def generate_dataset(self, num_real: int = 2000, num_fake: int = 2000):
        """Generate the complete improved dataset"""
        logger.info(f"Generating improved dataset: {num_real} real, {num_fake} fake")
        
        # Generate real samples
        real_samples = self.generate_diverse_real_samples(num_real)
        
        # Generate fake samples
        fake_samples = self.generate_diverse_fake_samples(num_fake)
        
        # Create metadata
        metadata = {
            "dataset_info": {
                "name": "Improved Deepfake Detection Dataset",
                "description": "Diverse training dataset with realistic variations",
                "total_samples": num_real + num_fake,
                "real_samples": num_real,
                "fake_samples": num_fake,
                "generated_at": str(Path().absolute()),
                "version": "2.0"
            },
            "real": real_samples,
            "fake": fake_samples,
            "statistics": {
                "celebrities_used": list(set(s["celebrity"] for s in real_samples)),
                "techniques_used": list(set(s["technique"] for s in fake_samples)),
                "variations_applied": list(set(var for s in real_samples for var in s.get("variations", [])))
            }
        }
        
        # Save metadata
        metadata_path = self.output_dir / "improved_training_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Dataset generated successfully!")
        logger.info(f"Real samples: {len(real_samples)}")
        logger.info(f"Fake samples: {len(fake_samples)}")
        logger.info(f"Metadata saved to: {metadata_path}")
        
        return metadata

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate improved training data")
    parser.add_argument("--num-real", type=int, default=2000, help="Number of real samples")
    parser.add_argument("--num-fake", type=int, default=2000, help="Number of fake samples")
    parser.add_argument("--output-dir", default="improved_training_data", help="Output directory")
    
    args = parser.parse_args()
    
    generator = ImprovedDataGenerator(args.output_dir)
    generator.generate_dataset(args.num_real, args.num_fake)

if __name__ == "__main__":
    main() 