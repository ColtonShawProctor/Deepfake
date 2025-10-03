#!/usr/bin/env python3
"""
Test EfficientNet Model on Test Dataset

This script tests your EfficientNet deepfake detection model on the test dataset
files and shows confidence scores for each sample.
"""

import os
import sys
import json
import time
from pathlib import Path
from PIL import Image

# Add the app directory to the path so we can import the detector
sys.path.append('app')

def test_efficientnet_on_dataset():
    """Test EfficientNet model on all test dataset images"""
    
    print("🧪 Testing EfficientNet Model on Test Dataset")
    print("=" * 60)
    
    # Load the test dataset metadata
    metadata_path = Path("test_data/dataset_metadata.json")
    if not metadata_path.exists():
        print("❌ Test dataset metadata not found. Run the dataset creation script first.")
        return
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Try to import and initialize the EfficientNet detector
    try:
        from models.optimized_efficientnet_detector import OptimizedEfficientNetDetector
        print("✅ Successfully imported EfficientNet detector")
        
        # Initialize detector with the trained weights
        detector = OptimizedEfficientNetDetector("models/efficientnet_weights.pth")
        print("✅ EfficientNet detector initialized")
        
    except ImportError as e:
        print(f"❌ Failed to import EfficientNet detector: {e}")
        print("Make sure you're running this from the project root directory")
        return
    except Exception as e:
        print(f"❌ Failed to initialize detector: {e}")
        return
    
    # Get all image samples
    samples = metadata["samples"]
    image_samples = {name: info for name, info in samples.items() 
                    if info["file_type"] == "image"}
    
    print(f"\n📸 Found {len(image_samples)} image samples to test")
    print("=" * 60)
    
    results = []
    
    for i, (sample_name, sample_info) in enumerate(image_samples.items(), 1):
        print(f"\n[{i}/{len(image_samples)}] Testing: {sample_name}")
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
    print("📊 TEST SUMMARY")
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
        results_file = "test_data/efficientnet_test_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                "test_info": {
                    "model": "EfficientNet",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "total_samples": total_samples,
                    "accuracy": accuracy
                },
                "results": results
            }, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: {results_file}")
        
    else:
        print("No samples were successfully tested.")
    
    print("\n🎯 Analysis Complete!")
    print("Use these results to evaluate your EfficientNet model's performance on the test dataset.")

if __name__ == "__main__":
    test_efficientnet_on_dataset()
