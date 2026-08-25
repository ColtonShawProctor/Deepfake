#!/usr/bin/env python3
"""
Test EfficientNet on Celeb-DF-v2 Dataset

This script tests your fixed EfficientNet model on real deepfake videos
from the Celeb-DF-v2 dataset to see how it performs on professional deepfakes.
"""

import os
import sys
import time
import random
import tempfile
from pathlib import Path
from PIL import Image
import cv2

# Add the app directory to the path
sys.path.append('app')

def extract_frames_from_video(video_path, num_frames=3):
    """Extract frames from a video file and save as temporary files"""
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return []
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        # Extract frames at different timestamps
        frame_files = []
        if total_frames > 0:
            # Extract frames at 25%, 50%, and 75% of the video
            frame_positions = [
                int(total_frames * 0.25),
                int(total_frames * 0.50),
                int(total_frames * 0.75)
            ]
            
            for i, pos in enumerate(frame_positions):
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Save frame as temporary file
                    temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
                    pil_image = Image.fromarray(frame_rgb)
                    pil_image.save(temp_file.name, 'JPEG', quality=95)
                    temp_file.close()
                    
                    frame_files.append(temp_file.name)
        
        cap.release()
        return frame_files[:num_frames]
        
    except Exception as e:
        print(f"   Error extracting frames: {e}")
        return []

def test_celeb_df_v2():
    """Test EfficientNet on Celeb-DF-v2 dataset"""
    
    print("🎯 Testing EfficientNet on Celeb-DF-v2 Dataset")
    print("=" * 70)
    print("This will test your model on REAL deepfake videos from the dataset!")
    print("=" * 70)
    
    try:
        from models.optimized_efficientnet_detector import OptimizedEfficientNetDetector
        print("✅ Successfully imported EfficientNet detector")
        
        # Initialize detector with trained weights
        detector = OptimizedEfficientNetDetector("models/efficientnet_weights.pth")
        print("✅ EfficientNet detector initialized with trained weights")
        
        # Define test samples from Celeb-DF-v2
        test_samples = [
            {
                "name": "Real Celebrity Video",
                "path": "Celeb-DF-v2/Celeb-real/id0_0000.mp4",
                "expected": "REAL",
                "description": "Real celebrity video from Celeb-DF-v2"
            },
            {
                "name": "Deepfake Video 1",
                "path": "Celeb-DF-v2/Celeb-synthesis/id0_id16_0000.mp4",
                "expected": "FAKE",
                "description": "Deepfake video (id0 swapped to id16)"
            },
            {
                "name": "Deepfake Video 2",
                "path": "Celeb-DF-v2/Celeb-synthesis/id0_id17_0000.mp4",
                "expected": "FAKE",
                "description": "Deepfake video (id0 swapped to id17)"
            },
            {
                "name": "Deepfake Video 3",
                "path": "Celeb-DF-v2/Celeb-synthesis/id0_id16_0001.mp4",
                "expected": "FAKE",
                "description": "Deepfake video (id0 swapped to id16)"
            },
            {
                "name": "Real Celebrity Video 2",
                "path": "Celeb-DF-v2/Celeb-real/id10_0000.mp4",
                "expected": "REAL",
                "description": "Another real celebrity video from Celeb-DF-v2"
            }
        ]
        
        print(f"\n🧪 Testing Celeb-DF-v2 Samples")
        print("=" * 70)
        
        results = []
        temp_files_to_cleanup = []
        
        for i, sample in enumerate(test_samples, 1):
            print(f"\n[{i}/{len(test_samples)}] {sample['name']}")
            print(f"   Expected: {sample['expected']}")
            print(f"   Description: {sample['description']}")
            print(f"   File: {sample['path']}")
            
            # Check if file exists
            file_path = Path(sample['path'])
            if not file_path.exists():
                print(f"   ❌ File not found: {file_path}")
                continue
            
            try:
                # Extract frames from video
                print(f"   🎬 Extracting frames from video...")
                frame_files = extract_frames_from_video(file_path, num_frames=3)
                
                if not frame_files:
                    print(f"   ❌ Could not extract frames from video")
                    continue
                
                print(f"   ✅ Extracted {len(frame_files)} frames")
                
                # Track temp files for cleanup
                temp_files_to_cleanup.extend(frame_files)
                
                # Test each frame
                frame_results = []
                for j, frame_file in enumerate(frame_files):
                    # Test the frame file
                    start_time = time.time()
                    result = detector.predict(frame_file)
                    inference_time = time.time() - start_time
                    
                    # Extract results
                    confidence = result["confidence"]
                    is_deepfake = result["is_deepfake"]
                    
                    # Determine classification
                    classification = "FAKE" if is_deepfake else "REAL"
                    
                    frame_results.append({
                        "frame": j + 1,
                        "classification": classification,
                        "confidence": confidence,
                        "inference_time": inference_time
                    })
                
                # Determine overall prediction based on majority vote
                fake_votes = sum(1 for r in frame_results if r["classification"] == "FAKE")
                real_votes = len(frame_results) - fake_votes
                overall_classification = "FAKE" if fake_votes > real_votes else "REAL"
                
                # Calculate average confidence
                avg_confidence = sum(r["confidence"] for r in frame_results) / len(frame_results)
                
                # Check if prediction matches expected
                expected_class = sample['expected']
                is_correct = overall_classification == expected_class
                
                # Store result
                results.append({
                    "name": sample['name'],
                    "expected": expected_class,
                    "predicted": overall_classification,
                    "confidence": avg_confidence,
                    "is_correct": is_correct,
                    "frame_results": frame_results
                })
                
                # Print result with clear formatting
                status_icon = "✅" if is_correct else "❌"
                confidence_percent = avg_confidence * 100
                
                print(f"   {status_icon} Overall Prediction: {overall_classification}")
                print(f"   📊 Average Confidence: {confidence_percent:.1f}%")
                print(f"   🎯 Result: {'CORRECT' if is_correct else 'INCORRECT'}")
                
                # Show frame-by-frame results
                print(f"   📷 Frame-by-frame results:")
                for frame_result in frame_results:
                    frame_conf = frame_result["confidence"] * 100
                    print(f"      Frame {frame_result['frame']}: {frame_result['classification']} ({frame_conf:.1f}%)")
                
                # Add explanation
                if is_correct:
                    if overall_classification == "REAL":
                        print(f"   💡 Model correctly identified this as a REAL video with {confidence_percent:.1f}% confidence")
                    else:
                        print(f"   💡 Model correctly detected this as FAKE with {confidence_percent:.1f}% confidence")
                else:
                    print(f"   ⚠️  Model prediction doesn't match expected result")
                
            except Exception as e:
                print(f"   ❌ Error testing {sample['name']}: {e}")
                continue
        
        # Clean up temporary files
        print(f"\n🧹 Cleaning up temporary files...")
        for temp_file in temp_files_to_cleanup:
            try:
                os.unlink(temp_file)
            except:
                pass
        
        # Generate test summary
        print("\n" + "=" * 70)
        print("🎉 CELEB-DF-V2 TEST SUMMARY")
        print("=" * 70)
        
        if results:
            total_samples = len(results)
            correct_predictions = sum(1 for r in results if r["is_correct"])
            accuracy = (correct_predictions / total_samples) * 100
            
            print(f"Total samples tested: {total_samples}")
            print(f"Correct predictions: {correct_predictions}")
            print(f"Incorrect predictions: {total_samples - correct_predictions}")
            print(f"Test accuracy: {accuracy:.1f}%")
            
            # Show confidence score statistics
            confidences = [r["confidence"] for r in results]
            avg_confidence = sum(confidences) / len(confidences)
            min_confidence = min(confidences)
            max_confidence = max(confidences)
            
            print(f"\n📊 Confidence Score Analysis:")
            print(f"  Average confidence: {avg_confidence:.3f}")
            print(f"  Range: {min_confidence:.3f} - {max_confidence:.3f}")
            
            # Show results by category
            print(f"\n🎯 Results by Category:")
            real_samples = [r for r in results if r["expected"] == "REAL"]
            fake_samples = [r for r in results if r["expected"] == "FAKE"]
            
            if real_samples:
                real_accuracy = sum(1 for r in real_samples if r["is_correct"]) / len(real_samples) * 100
                real_avg_conf = sum(r["confidence"] for r in real_samples) / len(real_samples)
                print(f"  REAL samples: {len(real_samples)} tested, {real_accuracy:.1f}% accuracy")
                print(f"    Average confidence: {real_avg_conf:.3f}")
            
            if fake_samples:
                fake_accuracy = sum(1 for r in fake_samples if r["is_correct"]) / len(fake_samples) * 100
                fake_avg_conf = sum(r["confidence"] for r in fake_samples) / len(fake_samples)
                print(f"  FAKE samples: {len(fake_samples)} tested, {fake_accuracy:.1f}% accuracy")
                print(f"    Average confidence: {fake_avg_conf:.3f}")
            
            print(f"\n🚀 Your EfficientNet model tested on REAL deepfake data!")
            print(f"   - This is the same type of data it was trained on")
            print(f"   - Results should be much closer to your 88.5% training accuracy")
            
        else:
            print("No samples were successfully tested.")
        
        print("\n🎯 Celeb-DF-v2 Test Complete!")
        
    except Exception as e:
        print(f"❌ Error during Celeb-DF-v2 test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_celeb_df_v2()
