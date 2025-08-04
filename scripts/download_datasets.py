#!/usr/bin/env python3
"""
Download public deepfake detection datasets

This script downloads and prepares public datasets for training deepfake detection models.
"""

import os
import sys
import logging
import requests
import zipfile
import tarfile
from pathlib import Path
import subprocess
from typing import Optional, Dict, List
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetDownloader:
    """Download and prepare deepfake detection datasets"""
    
    def __init__(self, datasets_dir: str = "datasets"):
        self.datasets_dir = Path(datasets_dir)
        self.datasets_dir.mkdir(exist_ok=True)
        
        # Dataset configurations
        self.datasets = {
            "celebdf": {
                "name": "Celeb-DF",
                "description": "Celeb-DF: Large-scale Celebrities DeepFake Detection Dataset",
                "url": "https://github.com/yuezunli/celeb-deepfakeforensics",
                "size": "~1.5GB",
                "type": "video",
                "real_count": 590,
                "fake_count": 5639,
                "download_method": "git_clone"
            },
            "faceforensics": {
                "name": "FaceForensics++",
                "description": "FaceForensics++: Learning to Detect Manipulated Facial Images",
                "url": "https://github.com/ondyari/FaceForensics",
                "size": "~1.7GB",
                "type": "video",
                "real_count": 1000,
                "fake_count": 4000,
                "download_method": "git_clone"
            },
            "dfdc": {
                "name": "DeepFake Detection Challenge",
                "description": "Facebook's DeepFake Detection Challenge Dataset",
                "url": "https://ai.facebook.com/datasets/dfdc/",
                "size": "~100GB",
                "type": "video",
                "real_count": 100000,
                "fake_count": 100000,
                "download_method": "manual"
            },
            "ffiw": {
                "name": "Face Forensics in the Wild",
                "description": "Face Forensics in the Wild Dataset",
                "url": "https://github.com/ondyari/FaceForensics",
                "size": "~10GB",
                "type": "video",
                "real_count": 10000,
                "fake_count": 10000,
                "download_method": "git_clone"
            }
        }
    
    def download_file(self, url: str, filename: str) -> bool:
        """Download a file from URL"""
        try:
            logger.info(f"Downloading {filename} from {url}")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            filepath = self.datasets_dir / filename
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
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
    
    def download_synthetic_samples(self) -> bool:
        """Download additional synthetic samples"""
        logger.info("Downloading synthetic deepfake samples...")
        
        # Create synthetic samples directory
        synthetic_dir = self.datasets_dir / "synthetic"
        synthetic_dir.mkdir(exist_ok=True)
        
        # Download some sample images from public sources
        sample_urls = [
            "https://raw.githubusercontent.com/ondyari/FaceForensics/master/images/real.jpg",
            "https://raw.githubusercontent.com/ondyari/FaceForensics/master/images/fake.jpg"
        ]
        
        success_count = 0
        for i, url in enumerate(sample_urls):
            filename = f"sample_{i}.jpg"
            if self.download_file(url, f"synthetic/{filename}"):
                success_count += 1
        
        logger.info(f"Downloaded {success_count}/{len(sample_urls)} synthetic samples")
        return success_count > 0
    
    def create_dataset_info(self) -> Dict:
        """Create information about available datasets"""
        info = {
            "datasets_dir": str(self.datasets_dir),
            "available_datasets": {},
            "total_real": 0,
            "total_fake": 0
        }
        
        for dataset_id, dataset_info in self.datasets.items():
            dataset_path = self.datasets_dir / dataset_id
            info["available_datasets"][dataset_id] = {
                "name": dataset_info["name"],
                "description": dataset_info["description"],
                "path": str(dataset_path),
                "exists": dataset_path.exists(),
                "real_count": dataset_info["real_count"],
                "fake_count": dataset_info["fake_count"],
                "size": dataset_info["size"],
                "type": dataset_info["type"]
            }
            
            if dataset_path.exists():
                info["total_real"] += dataset_info["real_count"]
                info["total_fake"] += dataset_info["fake_count"]
        
        return info
    
    def download_all(self) -> bool:
        """Download all available datasets"""
        logger.info("Starting download of all datasets...")
        
        success_count = 0
        total_count = len(self.datasets)
        
        # Download each dataset
        for dataset_id, dataset_info in self.datasets.items():
            logger.info(f"Processing {dataset_info['name']}...")
            
            if dataset_info["download_method"] == "git_clone":
                if dataset_id == "celebdf":
                    success = self.download_celebdf()
                elif dataset_id == "faceforensics":
                    success = self.download_faceforensics()
                else:
                    success = False
            else:
                logger.info(f"Skipping {dataset_id} - requires manual download")
                success = True  # Not a failure, just manual process
            
            if success:
                success_count += 1
        
        # Download synthetic samples
        if self.download_synthetic_samples():
            success_count += 1
        
        logger.info(f"Download completed: {success_count}/{total_count + 1} datasets processed")
        
        # Save dataset information
        info = self.create_dataset_info()
        info_file = self.datasets_dir / "dataset_info.json"
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)
        
        logger.info(f"Dataset information saved to {info_file}")
        return success_count > 0

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download deepfake detection datasets")
    parser.add_argument("--datasets-dir", default="datasets", help="Directory to store datasets")
    parser.add_argument("--dataset", choices=["celebdf", "faceforensics", "all"], 
                       default="all", help="Specific dataset to download")
    parser.add_argument("--info", action="store_true", help="Show dataset information")
    
    args = parser.parse_args()
    
    downloader = DatasetDownloader(args.datasets_dir)
    
    if args.info:
        info = downloader.create_dataset_info()
        print("Dataset Information:")
        print(f"Datasets directory: {info['datasets_dir']}")
        print(f"Total real images: {info['total_real']:,}")
        print(f"Total fake images: {info['total_fake']:,}")
        print("\nAvailable datasets:")
        for dataset_id, dataset_info in info["available_datasets"].items():
            status = "✓" if dataset_info["exists"] else "✗"
            print(f"  {status} {dataset_info['name']}: {dataset_info['size']}")
        return
    
    if args.dataset == "all":
        success = downloader.download_all()
    elif args.dataset == "celebdf":
        success = downloader.download_celebdf()
    elif args.dataset == "faceforensics":
        success = downloader.download_faceforensics()
    else:
        logger.error(f"Unknown dataset: {args.dataset}")
        return
    
    if success:
        logger.info("Dataset download completed successfully!")
        logger.info("Note: Some datasets require manual download due to licensing restrictions.")
        logger.info("Please check the README files in the downloaded repositories for instructions.")
    else:
        logger.error("Dataset download failed!")

if __name__ == "__main__":
    main() 