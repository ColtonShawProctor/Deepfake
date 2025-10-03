#!/usr/bin/env python3
"""
Create Test Dataset for Deepfake Detection Demo (Simplified Version)

This script creates a comprehensive test dataset without requiring OpenCV.
It focuses on image creation and provides placeholder video files.
"""

import os
import sys
import time
import requests
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import json
import random

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

class SimpleTestDatasetCreator:
    def __init__(self, base_dir: str = "test_data"):
        self.base_dir = Path(base_dir)
        self.real_images_dir = self.base_dir / "real" / "images"
        self.real_videos_dir = self.base_dir / "real" / "videos"
        self.fake_images_dir = self.base_dir / "fake" / "images"
        self.fake_videos_dir = self.base_dir / "fake" / "videos"
        
        # Create directories
        for dir_path in [self.real_images_dir, self.real_videos_dir, 
                        self.fake_images_dir, self.fake_videos_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.downloaded_samples = {}
        self.synthetic_samples = {}
        
    def create_real_content(self):
        """Download and create real content samples"""
        print("📸 Creating Real Content Collection...")
        print("=" * 50)
        
        # Real Images
        real_images = self._download_real_images()
        self.downloaded_samples.update(real_images)
        
        # Create synthetic real-looking samples
        synthetic_real = self._create_synthetic_real_samples()
        self.synthetic_samples.update(synthetic_real)
        
        # Create placeholder video files
        self._create_video_placeholders(self.real_videos_dir, "real", 2)
        
    def create_fake_content(self):
        """Download and create fake content samples"""
        print("\n🎭 Creating Fake Content Collection...")
        print("=" * 50)
        
        # Fake Images
        fake_images = self._download_fake_images()
        self.downloaded_samples.update(fake_images)
        
        # Create synthetic deepfake samples
        synthetic_fake = self._create_synthetic_fake_samples()
        self.synthetic_samples.update(synthetic_fake)
        
        # Create placeholder video files
        self._create_video_placeholders(self.fake_videos_dir, "fake", 2)
        
    def _download_real_images(self):
        """Download real images from verified sources"""
        real_image_sources = {
            "real_celebrity_1": {
                "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/85/Elon_Musk_Royal_Society_%28crop1%29.jpg/512px-Elon_Musk_Royal_Society_%28crop1%29.jpg",
                "description": "Real celebrity photo - Elon Musk",
                "expected": False,
                "source": "Wikipedia Commons",
                "category": "real"
            },
            "real_celebrity_2": {
                "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1f/Mark_Zuckerberg_F8_2019_Keynote_%2832830578717%29_%28cropped%29.jpg/512px-Mark_Zuckerberg_F8_2019_Keynote_%2832830578717%29_%28cropped%29.jpg",
                "description": "Real celebrity photo - Mark Zuckerberg",
                "expected": False,
                "source": "Wikipedia Commons",
                "category": "real"
            },
            "real_person_1": {
                "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2f/Barack_Obama_2012_portrait.jpg/512px-Barack_Obama_2012_portrait.jpg",
                "description": "Real person photo - Barack Obama",
                "expected": False,
                "source": "Wikipedia Commons",
                "category": "real"
            }
        }
        
        downloaded = {}
        for name, info in real_image_sources.items():
            try:
                print(f"   📥 Downloading {name}...")
                file_path = self._download_image(info["url"], name, self.real_images_dir)
                
                if file_path:
                    downloaded[name] = {
                        "path": str(file_path),
                        "expected": info["expected"],
                        "description": info["description"],
                        "source": info["source"],
                        "category": info["category"],
                        "file_type": "image"
                    }
                    print(f"      ✅ Downloaded: {file_path.name}")
                    
            except Exception as e:
                print(f"      ❌ Failed to download {name}: {e}")
                
        return downloaded
    
    def _download_fake_images(self):
        """Download fake images from research datasets"""
        print("   🎨 Creating synthetic fake image samples...")
        
        fake_image_samples = {
            "deepfake_face_1": {
                "description": "Synthetic deepfake with artifacts",
                "expected": True,
                "source": "Generated",
                "category": "fake",
                "artifact_type": "high_frequency"
            },
            "deepfake_face_2": {
                "description": "Synthetic deepfake with compression",
                "expected": True,
                "source": "Generated",
                "category": "fake",
                "artifact_type": "compression"
            },
            "ai_generated_face": {
                "description": "AI-generated face (StyleGAN-like)",
                "expected": True,
                "source": "Generated",
                "category": "fake",
                "artifact_type": "ai_generation"
            }
        }
        
        downloaded = {}
        for name, info in fake_image_samples.items():
            try:
                file_path = self._create_synthetic_fake_image(
                    name, info["artifact_type"], self.fake_images_dir
                )
                
                downloaded[name] = {
                    "path": str(file_path),
                    "expected": info["expected"],
                    "description": info["description"],
                    "source": info["source"],
                    "category": info["category"],
                    "file_type": "image",
                    "artifact_type": info["artifact_type"]
                }
                print(f"      ✅ Created: {file_path.name}")
                
            except Exception as e:
                print(f"      ❌ Failed to create {name}: {e}")
                
        return downloaded
    
    def _download_image(self, url: str, name: str, target_dir: Path):
        """Download an image from URL"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Determine file extension
            content_type = response.headers.get('content-type', '')
            if 'jpeg' in content_type or 'jpg' in content_type:
                ext = '.jpg'
            elif 'png' in content_type:
                ext = '.png'
            else:
                ext = '.jpg'  # Default
            
            filename = f"{name}{ext}"
            file_path = target_dir / filename
            
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            # Verify and resize if needed
            with Image.open(file_path) as img:
                if img.size != (512, 512):
                    img = img.resize((512, 512), Image.Resampling.LANCZOS)
                    img.save(file_path, quality=95)
            
            return file_path
            
        except Exception as e:
            print(f"      Error downloading {url}: {e}")
            return None
    
    def _create_synthetic_real_samples(self):
        """Create synthetic real-looking samples"""
        print("   🎨 Creating synthetic real samples...")
        
        samples = {}
        
        # Natural face-like image
        natural_face = self._create_natural_face()
        nat_path = self.real_images_dir / "synthetic_natural_face.jpg"
        Image.fromarray(natural_face).save(nat_path, quality=95)
        
        samples["synthetic_natural_face"] = {
            "path": str(nat_path),
            "expected": False,
            "description": "Synthetic natural-looking face",
            "source": "Generated",
            "category": "real",
            "file_type": "image"
        }
        
        return samples
    
    def _create_synthetic_fake_samples(self):
        """Create synthetic deepfake samples"""
        print("   🎭 Creating synthetic deepfake samples...")
        
        samples = {}
        
        # High-frequency artifact sample
        hf_face = self._create_high_frequency_face()
        hf_path = self.fake_images_dir / "synthetic_hf_face.jpg"
        Image.fromarray(hf_face).save(hf_path, quality=85)
        
        samples["synthetic_hf_face"] = {
            "path": str(hf_path),
            "expected": True,
            "description": "Synthetic face with high-frequency artifacts",
            "source": "Generated",
            "category": "fake",
            "file_type": "image"
        }
        
        return samples
    
    def _create_synthetic_fake_image(self, name: str, artifact_type: str, target_dir: Path) -> Path:
        """Create a synthetic fake image with specific artifacts"""
        if artifact_type == "high_frequency":
            img_array = self._create_high_frequency_face()
        elif artifact_type == "compression":
            img_array = self._create_compression_artifact_face()
        elif artifact_type == "ai_generation":
            img_array = self._create_ai_generated_face()
        else:
            img_array = self._create_high_frequency_face()
        
        filename = f"{name}.jpg"
        file_path = target_dir / filename
        Image.fromarray(img_array).save(file_path, quality=85)
        
        return file_path
    
    def _create_video_placeholders(self, target_dir: Path, category: str, count: int):
        """Create placeholder video files (text-based)"""
        print(f"   🎥 Creating {category} video placeholders...")
        
        for i in range(count):
            filename = f"{category}_video_{i+1}.txt"
            file_path = target_dir / filename
            
            # Create a text file that describes the video
            content = f"""This is a placeholder for a {category} video file.
            
For actual testing, you should replace this with a real video file.
The video should be:
- Format: MP4
- Duration: 10-30 seconds
- Size: Under 50MB
- Content: {'Real person speaking/interacting' if category == 'real' else 'Deepfake/manipulated content'}

This placeholder file is created because the full video creation requires additional dependencies.
You can manually add real video files to this directory for testing.
"""
            
            with open(file_path, 'w') as f:
                f.write(content)
            
            # Add to samples
            sample_name = f"{category}_video_{i+1}"
            self.downloaded_samples[sample_name] = {
                "path": str(file_path),
                "expected": category == "fake",
                "description": f"Placeholder for {category} video {i+1}",
                "source": "Placeholder",
                "category": category,
                "file_type": "video",
                "is_placeholder": True
            }
            
            print(f"      ✅ Created placeholder: {file_path.name}")
    
    def _create_natural_face(self) -> np.ndarray:
        """Create a natural-looking face image"""
        img = np.random.randint(100, 180, (512, 512, 3), dtype=np.uint8)
        
        # Add face-like structure
        center_y, center_x = 256, 256
        for y in range(512):
            for x in range(512):
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                if dist < 150:  # Face region
                    # Natural skin tone
                    img[y, x] = [180 + int(20 * np.sin(dist/20)), 
                               150 + int(15 * np.cos(dist/16)),
                               130 + int(10 * np.sin(dist/24))]
        
        return img
    
    def _create_high_frequency_face(self) -> np.ndarray:
        """Create a face with high-frequency artifacts typical of deepfakes"""
        img = self._create_natural_face()
        
        # Add high-frequency artifacts
        noise_pattern = np.random.randint(-40, 40, (512, 512, 3))
        high_freq_mask = np.zeros((512, 512, 3))
        
        # Create checkerboard pattern (deepfake artifact)
        for y in range(0, 512, 4):
            for x in range(0, 512, 4):
                if (x + y) % 8 == 0:
                    high_freq_mask[y:y+4, x:x+4] = 1
        
        # Apply artifacts
        img = img + (noise_pattern * high_freq_mask * 0.3).astype(np.uint8)
        img = np.clip(img, 0, 255)
        
        return img
    
    def _create_compression_artifact_face(self) -> np.ndarray:
        """Create a face with compression artifacts"""
        img = self._create_natural_face()
        
        # Add compression artifacts
        for y in range(0, 512, 8):
            for x in range(0, 512, 8):
                block = img[y:y+8, x:x+8]
                # Simulate block compression
                block_mean = np.mean(block, axis=(0, 1))
                img[y:y+8, x:x+8] = block_mean + np.random.randint(-10, 10, 3)
        
        return img
    
    def _create_ai_generated_face(self) -> np.ndarray:
        """Create an AI-generated looking face"""
        img = np.random.randint(80, 200, (512, 512, 3), dtype=np.uint8)
        
        # Add smooth, artificial-looking features
        center_y, center_x = 256, 256
        for y in range(512):
            for x in range(512):
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                if dist < 180:
                    # Smooth gradients (AI-like)
                    angle = np.arctan2(y - center_y, x - center_x)
                    radius = dist / 180
                    
                    # Create smooth color transitions
                    r = int(180 + 40 * np.cos(angle * 3) * (1 - radius))
                    g = int(150 + 30 * np.sin(angle * 2) * (1 - radius))
                    b = int(130 + 20 * np.cos(angle * 4) * (1 - radius))
                    
                    img[y, x] = [r, g, b]
        
        return img
    
    def create_metadata(self):
        """Create comprehensive metadata for all samples"""
        print("\n📋 Creating Metadata...")
        
        all_samples = {**self.downloaded_samples, **self.synthetic_samples}
        
        metadata = {
            "dataset_info": {
                "name": "Deepfake Detection Test Dataset",
                "description": "Test dataset for demonstrating deepfake detection capabilities",
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_samples": len(all_samples),
                "categories": {
                    "real": len([s for s in all_samples.values() if s["category"] == "real"]),
                    "fake": len([s for s in all_samples.values() if s["category"] == "fake"])
                },
                "note": "Some video files are placeholders. Replace with actual video files for full testing."
            },
            "samples": all_samples,
            "expected_results": self._generate_expected_results(all_samples)
        }
        
        metadata_file = self.base_dir / "dataset_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   ✅ Metadata saved: {metadata_file}")
        return metadata
    
    def _generate_expected_results(self, samples):
        """Generate expected detection results for each sample"""
        expected = {}
        
        for name, info in samples.items():
            expected[name] = {
                "expected_classification": "FAKE" if info["expected"] else "REAL",
                "expected_confidence": {
                    "min": 0.7 if info["expected"] else 0.8,
                    "max": 0.95 if info["expected"] else 0.98
                },
                "notes": f"Should be classified as {'FAKE' if info['expected'] else 'REAL'} based on {info['description']}"
            }
        
        return expected
    
    def create_batch_testing_script(self):
        """Create a script for batch testing all samples"""
        print("\n🧪 Creating Batch Testing Script...")
        
        script_content = '''#!/usr/bin/env python3
"""
Batch Testing Script for Deepfake Detection Test Dataset

This script tests all samples in the test dataset against your deepfake detection system.
"""

import os
import sys
import json
import time
from pathlib import Path
import requests

# Configuration
API_BASE_URL = "http://localhost:8000"  # Adjust to your API URL
TEST_DATA_DIR = "test_data"
METADATA_FILE = "dataset_metadata.json"

def load_test_metadata():
    """Load test dataset metadata"""
    metadata_path = Path(TEST_DATA_DIR) / METADATA_FILE
    with open(metadata_path, 'r') as f:
        return json.load(f)

def test_image_sample(image_path: str, api_url: str) -> dict:
    """Test a single image sample"""
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{api_url}/detect", files=files, timeout=30)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        return {"error": str(e)}

def test_video_sample(video_path: str, api_url: str) -> dict:
    """Test a single video sample"""
    try:
        with open(video_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{api_url}/detect-video", files=files, timeout=60)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        return {"error": str(e)}

def run_batch_tests():
    """Run tests on all samples"""
    print("🧪 Running Batch Tests...")
    print("=" * 50)
    
    # Load metadata
    metadata = load_test_metadata()
    samples = metadata["samples"]
    expected_results = metadata["expected_results"]
    
    results = []
    total_samples = len(samples)
    
    for i, (sample_name, sample_info) in enumerate(samples.items(), 1):
        print(f"\\n[{i}/{total_samples}] Testing {sample_name}...")
        
        # Skip placeholder files
        if sample_info.get("is_placeholder", False):
            print(f"   ⏭️  Skipping placeholder file")
            continue
        
        # Determine API endpoint based on file type
        if sample_info["file_type"] == "image":
            result = test_image_sample(sample_info["path"], f"{API_BASE_URL}/api")
        else:
            result = test_video_sample(sample_info["path"], f"{API_BASE_URL}/api")
        
        # Store result
        test_result = {
            "sample_name": sample_name,
            "file_path": sample_info["path"],
            "expected": expected_results[sample_name],
            "actual_result": result,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        results.append(test_result)
        
        # Print result summary
        if "error" in result:
            print(f"   ❌ Error: {result['error']}")
        else:
            print(f"   ✅ Test completed")
    
    # Generate report
    generate_test_report(results, metadata)
    
    return results

def generate_test_report(results: list, metadata: dict):
    """Generate a comprehensive test report"""
    print("\\n📊 Generating Test Report...")
    
    report = {
        "test_summary": {
            "total_samples": len(results),
            "successful_tests": len([r for r in results if "error" not in r["actual_result"]]),
            "failed_tests": len([r for r in results if "error" in r["actual_result"]]),
            "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "results": results
    }
    
    # Save report
    report_file = Path(TEST_DATA_DIR) / "test_results.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"   ✅ Test report saved: {report_file}")
    
    # Print summary
    print(f"\\n📈 Test Summary:")
    print(f"   Total samples: {report['test_summary']['total_samples']}")
    print(f"   Successful tests: {report['test_summary']['successful_tests']}")
    print(f"   Failed tests: {report['test_summary']['failed_tests']}")

if __name__ == "__main__":
    run_batch_tests()
'''
        
        script_path = self.base_dir / "batch_test.py"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make executable
        os.chmod(script_path, 0o755)
        
        print(f"   ✅ Batch testing script created: {script_path}")
    
    def create_readme(self):
        """Create a comprehensive README for the test dataset"""
        print("\n📖 Creating README...")
        
        readme_content = f"""# Deepfake Detection Test Dataset

This directory contains a comprehensive test dataset for demonstrating deepfake detection capabilities.

## Dataset Structure

```
{self.base_dir}/
├── real/
│   ├── images/          # Real person photos
│   └── videos/          # Real video clips (placeholders)
├── fake/
│   ├── images/          # Deepfake/fake images
│   └── videos/          # Deepfake/fake videos (placeholders)
├── dataset_metadata.json # Complete dataset information
├── batch_test.py        # Batch testing script
└── README.md            # This file
```

## Content Overview

### Real Content ({len([s for s in self.downloaded_samples.values() if s["category"] == "real"])} samples)
- **Images**: High-quality photos of real people from verified sources
- **Videos**: Placeholder files (replace with actual video content)
- **Sources**: Wikipedia Commons, official portraits, verified content

### Fake Content ({len([s for s in self.downloaded_samples.values() if s["category"] == "fake"])} samples)
- **Images**: Synthetic deepfake images with various artifacts
- **Videos**: Placeholder files (replace with actual deepfake videos)
- **Sources**: Generated samples, research dataset simulations

## Expected Results

Each sample includes expected classification results:
- **REAL**: Should be classified as genuine content
- **FAKE**: Should be classified as deepfake/manipulated content

Confidence scores are provided as ranges to account for model variations.

## Usage

### 1. Manual Testing
Upload individual files through your deepfake detection system's interface.

### 2. Batch Testing
Run the automated batch testing script:
```bash
cd {self.base_dir}
python batch_test.py
```

### 3. Integration Testing
Use the metadata file to integrate with your testing framework.

## File Requirements

- **Images**: JPG format, 512x512 resolution
- **Videos**: MP4 format, 10-30 seconds duration (replace placeholders)
- **Size**: All files under 50MB for easy testing

## Important Notes

- **Video Placeholders**: Some video files are text placeholders. Replace them with actual video files for full testing.
- **Real Videos**: Use clips from news interviews, official statements, or verified content.
- **Fake Videos**: Use samples from deepfake detection challenges or research datasets.

## Getting Real Video Content

For real video content, consider:
1. News interview clips (YouTube, news websites)
2. Official statements from verified sources
3. Public domain educational content

For fake video content, consider:
1. Deepfake detection challenge datasets
2. Academic research paper supplementary materials
3. Publicly available deepfake examples

## Support

For questions about this test dataset, refer to your deepfake detection system documentation.
"""
        
        readme_path = self.base_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        print(f"   ✅ README created: {readme_path}")
    
    def run(self):
        """Execute the complete dataset creation process"""
        print("🚀 Creating Deepfake Detection Test Dataset (Simplified)")
        print("=" * 60)
        
        # Create content
        self.create_real_content()
        self.create_fake_content()
        
        # Create metadata and documentation
        self.create_metadata()
        self.create_batch_testing_script()
        self.create_readme()
        
        # Summary
        total_samples = len(self.downloaded_samples) + len(self.synthetic_samples)
        real_count = len([s for s in self.downloaded_samples.values() if s["category"] == "real"])
        fake_count = len([s for s in self.downloaded_samples.values() if s["category"] == "fake"])
        
        print(f"\n🎉 Dataset Creation Complete!")
        print("=" * 60)
        print(f"📊 Summary:")
        print(f"   Total samples: {total_samples}")
        print(f"   Real content: {real_count}")
        print(f"   Fake content: {fake_count}")
        print(f"   Dataset location: {self.base_dir}")
        print(f"   Metadata: {self.base_dir}/dataset_metadata.json")
        print(f"   Batch test script: {self.base_dir}/batch_test.py")
        print(f"   README: {self.base_dir}/README.md")
        
        print(f"\n🚀 Next steps:")
        print(f"   1. Review the dataset in {self.base_dir}")
        print(f"   2. Replace video placeholders with actual video files")
        print(f"   3. Run batch tests: cd {self.base_dir} && python batch_test.py")
        print(f"   4. Integrate with your deepfake detection system")
        
        print(f"\n⚠️  Note: Some video files are placeholders. Replace them with actual video content for full testing.")

def main():
    """Main execution function"""
    creator = SimpleTestDatasetCreator()
    creator.run()

if __name__ == "__main__":
    main()
