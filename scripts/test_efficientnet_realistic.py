#!/usr/bin/env python3
"""
Realistic Test for EfficientNet Model

This script creates more realistic test cases that better match what the
EfficientNet model was trained on (FaceForensics++, DFDC, CelebDF datasets).
"""

import os
import sys
import json
import time
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
from pathlib import Path
import cv2

# Add the app directory to the path so we can import the detector
sys.path.append('app')

def create_realistic_test_images():
    """Create more realistic test images that better simulate training data"""
    
    print("🎨 Creating Realistic Test Images...")
    print("=" * 50)
    
    # Create test directory
    test_dir = Path("realistic_test_images")
    test_dir.mkdir(exist_ok=True)
    
    # Load a real image to work with
    real_image_path = Path("test_data/real/images/real_celebrity_1.jpg")
    if not real_image_path.exists():
        print("❌ Real image not found. Run the test dataset creation first.")
        return None
    
    # Load the real image
    real_image = Image.open(real_image_path).convert('RGB')
    real_image = real_image.resize((224, 224))  # Standard input size
    
    # Save the original real image
    real_path = test_dir / "real_original.jpg"
    real_image.save(real_path, quality=95)
    
    # Create realistic fake images that better simulate deepfake artifacts
    
    # 1. Face region manipulation (simulating face-swap artifacts)
    print("   Creating face-swap simulation...")
    face_swap = real_image.copy()
    # Apply subtle color inconsistencies in face region
    face_swap_array = np.array(face_swap)
    center_y, center_x = 112, 112
    for y in range(224):
        for x in range(224):
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            if dist < 60:  # Face region
                # Add subtle color shifts (common in face-swaps)
                face_swap_array[y, x, 0] = np.clip(face_swap_array[y, x, 0] * 1.1, 0, 255)  # Red shift
                face_swap_array[y, x, 2] = np.clip(face_swap_array[y, x, 2] * 0.9, 0, 255)  # Blue shift
    
    face_swap_path = test_dir / "fake_faceswap.jpg"
    Image.fromarray(face_swap_array).save(face_swap_path, quality=90)
    
    # 2. Frequency domain artifacts (common in deepfakes)
    print("   Creating frequency domain artifacts...")
    freq_fake = real_image.copy()
    # Apply high-pass filter to create frequency artifacts
    freq_fake = freq_fake.filter(ImageFilter.EDGE_ENHANCE_MORE)
    # Add subtle noise patterns
    freq_array = np.array(freq_fake)
    noise = np.random.normal(0, 5, freq_array.shape).astype(np.uint8)
    freq_array = np.clip(freq_array + noise, 0, 255)
    
    freq_fake_path = test_dir / "fake_frequency.jpg"
    Image.fromarray(freq_array).save(freq_fake_path, quality=85)
    
    # 3. Compression artifacts (common in video deepfakes)
    print("   Creating compression artifacts...")
    comp_fake = real_image.copy()
    # Save with heavy compression to create artifacts
    comp_fake_path = test_dir / "fake_compression.jpg"
    comp_fake.save(comp_fake_path, quality=30, optimize=True)
    
    # 4. Lighting inconsistencies (common in deepfake generation)
    print("   Creating lighting inconsistencies...")
    light_fake = real_image.copy()
    # Apply uneven lighting
    enhancer = ImageEnhance.Brightness(light_fake)
    # Create a mask for uneven lighting
    light_array = np.array(light_fake)
    for y in range(224):
        for x in range(224):
            # Add gradient lighting effect
            light_factor = 1.0 + 0.3 * np.sin(x / 224 * np.pi) * np.cos(y / 224 * np.pi)
            light_array[y, x] = np.clip(light_array[y, x] * light_factor, 0, 255)
    
    light_fake_path = test_dir / "fake_lighting.jpg"
    Image.fromarray(light_array.astype(np.uint8)).save(light_fake_path, quality=90)
    
    # 5. Edge artifacts (common in deepfake boundaries)
    print("   Creating edge artifacts...")
    edge_fake = real_image.copy()
    # Apply edge detection and blend
    edge_detected = edge_fake.filter(ImageFilter.FIND_EDGES)
    edge_array = np.array(edge_fake).astype(float)
    edge_detected_array = np.array(edge_detected).astype(float)
    
    # Blend edges with original (simulating boundary artifacts)
    blended = edge_array * 0.8 + edge_detected_array * 0.2
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    
    edge_fake_path = test_dir / "fake_edges.jpg"
    Image.fromarray(blended).save(edge_fake_path, quality=90)
    
    # Create metadata for the realistic test
    test_metadata = {
        "real_samples": {
            "real_original": {
                "path": str(real_path),
                "expected": False,
                "description": "Original real image (Elon Musk)",
                "category": "real"
            }
        },
        "fake_samples": {
            "fake_faceswap": {
                "path": str(face_swap_path),
                "expected": True,
                "description": "Face-swap simulation with color inconsistencies",
                "category": "fake"
            },
            "fake_frequency": {
                "path": str(freq_fake_path),
                "expected": True,
                "description": "Frequency domain artifacts and noise",
                "category": "fake"
            },
            "fake_compression": {
                "path": str(comp_fake_path),
                "expected": True,
                "description": "Heavy compression artifacts",
                "category": "fake"
            },
            "fake_lighting": {
                "path": str(light_fake_path),
                "expected": True,
                "description": "Uneven lighting and color shifts",
                "category": "fake"
            },
            "fake_edges": {
                "path": str(edge_fake_path),
                "expected": True,
                "description": "Edge artifacts and boundary inconsistencies",
                "category": "fake"
            }
        }
    }
    
    # Save metadata
    metadata_path = test_dir / "realistic_test_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(test_metadata, f, indent=2)
    
    print(f"   ✅ Created {len(test_metadata['fake_samples'])} realistic fake images")
    print(f"   ✅ Created {len(test_metadata['real_samples'])} real image")
    print(f"   ✅ Metadata saved to: {metadata_path}")
    
    return test_metadata

def test_efficientnet_realistic():
    """Test EfficientNet on realistic test images"""
    
    print("🧪 Testing EfficientNet on Realistic Test Images")
    print("=" * 60)
    
    # Create realistic test images
    test_metadata = create_realistic_test_images()
    if not test_metadata:
        return
    
    # Try to import and initialize the EfficientNet detector
    try:
        from models.optimized_efficientnet_detector import OptimizedEfficientNetDetector
        print("✅ Successfully imported EfficientNet detector")
        
        # Initialize detector with the trained weights
        detector = OptimizedEfficientNetDetector("models/efficientnet_weights.pth")
        print("✅ EfficientNet detector initialized with trained weights")
        
    except ImportError as e:
        print(f"❌ Failed to import EfficientNet detector: {e}")
        return
    except Exception as e:
        print(f"❌ Failed to initialize detector: {e}")
        return
    
    # Test all samples
    all_samples = {**test_metadata["real_samples"], **test_metadata["fake_samples"]}
    
    print(f"\n📸 Testing {len(all_samples)} realistic samples")
    print("=" * 60)
    
    results = []
    
    for i, (sample_name, sample_info) in enumerate(all_samples.items(), 1):
        print(f"\n[{i}/{len(all_samples)}] Testing: {sample_name}")
        print(f"   File: {sample_info['path']}")
        print(f"   Expected: {'FAKE' if sample_info['expected'] else 'REAL'}")
        print(f"   Description: {sample_info['description']}")
        
        # Check if file exists
        file_path = Path(sample_info['path'])
        if not file_path.exists():
            print(f"   ❌ File not found: {file_path}")
            continue
        
        try:
            # Test the image with EfficientNet
            start_time = time.time()
            result = detector.predict(str(file_path))
            inference_time = time.time() - start_time
            
            # Extract results
            confidence = result["confidence"]
            is_deepfake = result["is_deepfake"]
            model_name = result.get("model", "efficientnet")
            
            # Determine classification
            classification = "FAKE" if is_deepfake else "REAL"
            
            # Check if prediction matches expected
            expected_class = "FAKE" if sample_info['expected'] else "REAL"
            is_correct = classification == expected_class
            
            # Store result
            test_result = {
                "sample_name": sample_name,
                "file_path": str(file_path),
                "expected": expected_class,
                "predicted": classification,
                "confidence": confidence,
                "is_correct": is_correct,
                "inference_time": inference_time,
                "model": model_name
            }
            
            results.append(test_result)
            
            # Print result
            status_icon = "✅" if is_correct else "❌"
            print(f"   {status_icon} Prediction: {classification} (confidence: {confidence:.3f})")
            print(f"   ⏱️  Inference time: {inference_time:.3f}s")
            print(f"   🎯 Accuracy: {'Correct' if is_correct else 'Incorrect'}")
            
        except Exception as e:
            print(f"   ❌ Error testing {sample_name}: {e}")
            continue
    
    # Generate summary report
    print("\n" + "=" * 60)
    print("📊 REALISTIC TEST SUMMARY")
    print("=" * 60)
    
    if results:
        total_samples = len(results)
        correct_predictions = sum(1 for r in results if r["is_correct"])
        accuracy = (correct_predictions / total_samples) * 100
        
        print(f"Total samples tested: {total_samples}")
        print(f"Correct predictions: {correct_predictions}")
        print(f"Incorrect predictions: {total_samples - correct_predictions}")
        print(f"Accuracy: {accuracy:.1f}%")
        
        # Show confidence score statistics
        confidences = [r["confidence"] for r in results]
        avg_confidence = sum(confidences) / len(confidences)
        min_confidence = min(confidences)
        max_confidence = max(confidences)
        
        print(f"\nConfidence Score Statistics:")
        print(f"  Average: {avg_confidence:.3f}")
        print(f"  Range: {min_confidence:.3f} - {max_confidence:.3f}")
        
        # Show results by category
        print(f"\nResults by Category:")
        real_samples = [r for r in results if r["expected"] == "REAL"]
        fake_samples = [r for r in results if r["expected"] == "FAKE"]
        
        if real_samples:
            real_accuracy = sum(1 for r in real_samples if r["is_correct"]) / len(real_samples) * 100
            real_avg_conf = sum(r["confidence"] for r in real_samples) / len(real_samples)
            print(f"  REAL samples: {len(real_samples)} tested, {real_accuracy:.1f}% accuracy, avg confidence: {real_avg_conf:.3f}")
        
        if fake_samples:
            fake_accuracy = sum(1 for r in fake_samples if r["is_correct"]) / len(fake_samples) * 100
            fake_avg_conf = sum(r["confidence"] for r in fake_samples) / len(fake_samples)
            print(f"  FAKE samples: {len(fake_samples)} tested, {fake_accuracy:.1f}% accuracy, avg confidence: {fake_avg_conf:.3f}")
        
        # Save detailed results
        results_file = "realistic_test_images/realistic_test_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                "test_info": {
                    "model": "EfficientNet",
                    "test_type": "realistic",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "total_samples": total_samples,
                    "accuracy": accuracy
                },
                "results": results
            }, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: {results_file}")
        
    else:
        print("No samples were successfully tested.")
    
    print("\n🎯 Realistic Test Complete!")
    print("These results should better reflect your model's actual performance.")

if __name__ == "__main__":
    test_efficientnet_realistic()





