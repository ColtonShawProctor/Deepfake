#!/usr/bin/env python3
"""
Test the new Hugging Face deepfake detector on demo database images.
This script tests the model's performance on known real and fake images.
"""

import requests
import json
import time
from pathlib import Path
from PIL import Image
import os

def test_image_analysis(image_path: str, expected_type: str, auth_token: str):
    """Test image analysis for a single image"""
    print(f"\n🔍 Testing {expected_type.upper()} image: {os.path.basename(image_path)}")
    print("-" * 60)
    
    try:
        # Upload the image with authentication
        print("1. Uploading image...")
        with open(image_path, 'rb') as f:
            files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
            headers = {'Authorization': f'Bearer {auth_token}'}
            upload_response = requests.post('http://localhost:8000/api/upload', files=files, headers=headers)
        
        if upload_response.status_code != 200:
            print(f"❌ Upload failed: {upload_response.status_code}")
            print(f"   Response: {upload_response.text}")
            return None
        
        upload_data = upload_response.json()
        file_id = upload_data['file_id']
        print(f"✅ Upload successful! File ID: {file_id}")
        
        # Analyze the image with authentication
        print("2. Analyzing image...")
        headers = {'Authorization': f'Bearer {auth_token}'}
        analysis_response = requests.post(f'http://localhost:8000/api/detection/analyze/{file_id}', headers=headers)
        
        if analysis_response.status_code != 200:
            print(f"❌ Analysis failed: {analysis_response.status_code}")
            print(f"   Response: {analysis_response.text}")
            return None
        
        analysis_data = analysis_response.json()
        print("✅ Analysis completed!")
        
        # Get detailed results with authentication
        print("3. Getting detailed results...")
        results_response = requests.get(f'http://localhost:8000/api/detection/results/{file_id}', headers=headers)
        
        if results_response.status_code != 200:
            print(f"❌ Results retrieval failed: {analysis_response.status_code}")
            print(f"   Response: {analysis_response.text}")
            return None
        
        results_data = results_response.json()
        print("✅ Results retrieved!")
        
        # Display results
        detection_result = results_data['detection_result']
        confidence = detection_result['confidence_score']
        is_deepfake = detection_result['is_deepfake']
        processing_time = detection_result['processing_time_seconds']
        
        print(f"\n📊 RESULTS:")
        print(f"   Confidence: {confidence:.2f}%")
        print(f"   Prediction: {'FAKE' if is_deepfake else 'REAL'}")
        print(f"   Processing Time: {processing_time:.3f}s")
        print(f"   Expected: {expected_type.upper()}")
        
        # Evaluate prediction accuracy
        if expected_type == 'fake' and is_deepfake:
            print("   ✅ CORRECT - Fake image correctly identified as FAKE")
            accuracy = "CORRECT"
        elif expected_type == 'real' and not is_deepfake:
            print("   ✅ CORRECT - Real image correctly identified as REAL")
            accuracy = "CORRECT"
        else:
            print("   ❌ INCORRECT - Prediction doesn't match expected type")
            accuracy = "INCORRECT"
        
        return {
            'file_id': file_id,
            'filename': os.path.basename(image_path),
            'expected_type': expected_type,
            'predicted_type': 'FAKE' if is_deepfake else 'REAL',
            'confidence': confidence,
            'processing_time': processing_time,
            'accuracy': accuracy
        }
        
    except Exception as e:
        print(f"❌ Error testing image: {str(e)}")
        return None

def test_all_demo_images():
    """Test the model on all available demo images"""
    print("🧪 Testing Hugging Face Deepfake Detector on Demo Database")
    print("=" * 80)
    
    # First, authenticate to get a token
    print("🔐 Authenticating with test credentials...")
    login_data = {
        "username": "test@test.com",
        "password": "test1234"
    }
    
    try:
        login_response = requests.post(
            'http://localhost:8000/auth/login',
            data=login_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        
        if login_response.status_code != 200:
            print(f"❌ Authentication failed: {login_response.status_code}")
            print(f"Response: {login_response.text}")
            return None
        
        login_result = login_response.json()
        auth_token = login_result["access_token"]
        print(f"✅ Authentication successful! Token: {auth_token[:20]}...")
        
    except Exception as e:
        print(f"❌ Authentication error: {str(e)}")
        return None
    
    # Test images with their expected types
    test_images = [
        # Fake images
        ('test_data/fake/images/synthetic_hf_face.jpg', 'fake'),
        ('test_data/fake/images/ai_generated_face.jpg', 'fake'),
        ('test_data/fake/images/deepfake_face_1.jpg', 'fake'),
        ('test_data/fake/images/deepfake_face_2.jpg', 'fake'),
        
        # Real images
        ('test_data/real/images/real_celebrity_1.jpg', 'real'),
        ('test_data/real/images/synthetic_natural_face.jpg', 'real')
    ]
    
    results = []
    total_correct = 0
    total_images = 0
    
    for image_path, expected_type in test_images:
        if not os.path.exists(image_path):
            print(f"⚠️  Image not found: {image_path}")
            continue
        
        result = test_image_analysis(image_path, expected_type, auth_token)
        if result:
            results.append(result)
            if result['accuracy'] == 'CORRECT':
                total_correct += 1
            total_images += 1
        
        # Small delay between tests
        time.sleep(1)
    
    # Summary
    print("\n" + "=" * 80)
    print("📈 FINAL RESULTS SUMMARY")
    print("=" * 80)
    
    if total_images > 0:
        accuracy_percentage = (total_correct / total_images) * 100
        print(f"Total Images Tested: {total_images}")
        print(f"Correct Predictions: {total_correct}")
        print(f"Incorrect Predictions: {total_images - total_correct}")
        print(f"Accuracy: {accuracy_percentage:.1f}%")
        
        if accuracy_percentage >= 80:
            print("🎉 EXCELLENT performance!")
        elif accuracy_percentage >= 60:
            print("👍 GOOD performance!")
        else:
            print("⚠️  Performance needs improvement")
    else:
        print("❌ No images were successfully tested")
    
    # Detailed results table
    print(f"\n📋 DETAILED RESULTS:")
    print("-" * 80)
    print(f"{'Image':<25} {'Expected':<8} {'Predicted':<9} {'Confidence':<12} {'Accuracy':<10}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['filename']:<25} {result['expected_type']:<8} {result['predicted_type']:<9} "
              f"{result['confidence']:<12.1f}% {result['accuracy']:<10}")
    
    print("-" * 80)
    
    return results

if __name__ == "__main__":
    try:
        results = test_all_demo_images()
        if results:
            print(f"\n✅ Testing completed successfully!")
            print(f"Model: Hugging Face prithivMLmods/deepfake-detector-model-v1")
            print(f"Architecture: Vision Transformer (ViT)")
            print(f"Expected Accuracy: 94.4%")
        else:
            print("\n❌ Testing failed - no results obtained")
            exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Testing interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        exit(1)
