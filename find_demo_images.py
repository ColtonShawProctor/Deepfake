#!/usr/bin/env python3
"""
Find the best demonstration images for fake/real detection.
This script tests multiple images to find examples with clear, varied confidence scores.
"""

import requests
import json
import time
import os
from pathlib import Path

def authenticate():
    """Authenticate and get token"""
    print("🔐 Authenticating...")
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
            return None
        
        login_result = login_response.json()
        auth_token = login_result["access_token"]
        print(f"✅ Authentication successful!")
        return auth_token
        
    except Exception as e:
        print(f"❌ Authentication error: {str(e)}")
        return None

def test_single_image(image_path: str, expected_type: str, auth_token: str):
    """Test a single image and return results"""
    try:
        # Upload image
        with open(image_path, 'rb') as f:
            files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
            headers = {'Authorization': f'Bearer {auth_token}'}
            upload_response = requests.post('http://localhost:8000/api/upload', files=files, headers=headers)
        
        if upload_response.status_code != 200:
            return None
        
        upload_data = upload_response.json()
        file_id = upload_data['file_id']
        
        # Analyze image
        headers = {'Authorization': f'Bearer {auth_token}'}
        analysis_response = requests.post(f'http://localhost:8000/api/detection/analyze/{file_id}', headers=headers)
        
        if analysis_response.status_code != 200:
            return None
        
        # Get results
        results_response = requests.get(f'http://localhost:8000/api/detection/results/{file_id}', headers=headers)
        
        if results_response.status_code != 200:
            return None
        
        results_data = results_response.json()
        detection_result = results_data['detection_result']
        
        return {
            'path': image_path,
            'filename': os.path.basename(image_path),
            'expected_type': expected_type,
            'predicted_type': 'FAKE' if detection_result['is_deepfake'] else 'REAL',
            'confidence': detection_result['confidence_score'],
            'processing_time': detection_result['processing_time_seconds'],
            'correct': (expected_type == 'fake' and detection_result['is_deepfake']) or 
                      (expected_type == 'real' and not detection_result['is_deepfake'])
        }
        
    except Exception as e:
        print(f"Error testing {image_path}: {e}")
        return None

def find_demo_images():
    """Find the best demonstration images"""
    print("🔍 Finding Best Demonstration Images for Fake/Real Detection")
    print("=" * 80)
    
    # Authenticate
    auth_token = authenticate()
    if not auth_token:
        print("❌ Authentication failed, cannot proceed")
        return
    
    # Define test images from various sources
    test_images = [
        # Original test data
        ('test_data/fake/images/synthetic_hf_face.jpg', 'fake'),
        ('test_data/fake/images/ai_generated_face.jpg', 'fake'),
        ('test_data/fake/images/deepfake_face_1.jpg', 'fake'),
        ('test_data/fake/images/deepfake_face_2.jpg', 'fake'),
        ('test_data/real/images/real_celebrity_1.jpg', 'real'),
        ('test_data/real/images/synthetic_natural_face.jpg', 'real'),
        
        # Realistic test images
        ('realistic_test_images/fake_edges.jpg', 'fake'),
        ('realistic_test_images/fake_lighting.jpg', 'fake'),
        ('realistic_test_images/fake_compression.jpg', 'fake'),
        ('realistic_test_images/fake_frequency.jpg', 'fake'),
        ('realistic_test_images/fake_faceswap.jpg', 'fake'),
        ('realistic_test_images/real_original.jpg', 'real'),
        
        # Celeb-DF-v2 samples (if available)
        ('Celeb-DF-v2/Celeb-real/real_sample.jpg', 'real'),
        ('Celeb-DF-v2/Celeb-synthesis/fake_sample.jpg', 'fake'),
    ]
    
    results = []
    
    print(f"Testing {len(test_images)} images...")
    for i, (image_path, expected_type) in enumerate(test_images):
        if not os.path.exists(image_path):
            continue
            
        print(f"Testing {i+1}/{len(test_images)}: {os.path.basename(image_path)}")
        result = test_single_image(image_path, expected_type, auth_token)
        
        if result:
            results.append(result)
            print(f"  ✅ {result['predicted_type']} (Confidence: {result['confidence']:.1f}%)")
        else:
            print(f"  ❌ Failed")
        
        time.sleep(0.5)  # Small delay
    
    # Analyze results to find best examples
    print(f"\n📊 Analysis Complete! Found {len(results)} working images")
    
    # Separate by type and sort by confidence
    fake_results = [r for r in results if r['expected_type'] == 'fake']
    real_results = [r for r in results if r['expected_type'] == 'real']
    
    # Sort by confidence (ascending for fake, descending for real)
    fake_results.sort(key=lambda x: x['confidence'])
    real_results.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Find best examples
    print(f"\n🎯 BEST DEMONSTRATION IMAGES")
    print("=" * 80)
    
    print(f"\n🔴 FAKE IMAGES (3 best examples):")
    for i, result in enumerate(fake_results[:3]):
        status = "✅" if result['correct'] else "❌"
        print(f"{i+1}. {result['filename']}")
        print(f"   Confidence: {result['confidence']:.1f}% | Prediction: {result['predicted_type']} | {status}")
        print(f"   Path: {result['path']}")
    
    print(f"\n🟢 REAL IMAGES (3 best examples):")
    for i, result in enumerate(real_results[:3]):
        status = "✅" if result['correct'] else "❌"
        print(f"{i+1}. {result['filename']}")
        print(f"   Confidence: {result['confidence']:.1f}% | Prediction: {result['predicted_type']} | {status}")
        print(f"   Path: {result['path']}")
    
    # Summary statistics
    print(f"\n📈 SUMMARY STATISTICS")
    print("=" * 80)
    
    if fake_results:
        fake_confidences = [r['confidence'] for r in fake_results]
        print(f"Fake Images: {len(fake_results)} tested")
        print(f"  Confidence Range: {min(fake_confidences):.1f}% - {max(fake_confidences):.1f}%")
        print(f"  Average Confidence: {sum(fake_confidences)/len(fake_confidences):.1f}%")
    
    if real_results:
        real_confidences = [r['confidence'] for r in real_results]
        print(f"Real Images: {len(real_results)} tested")
        print(f"  Confidence Range: {min(real_confidences):.1f}% - {max(real_confidences):.1f}%")
        print(f"  Average Confidence: {sum(real_confidences)/len(real_confidences):.1f}%")
    
    # Save results to file
    output_file = "demo_image_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Detailed results saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    try:
        results = find_demo_images()
        if results:
            print(f"\n🎉 Demo image selection completed!")
            print(f"Use the top 3 fake and top 3 real images for your demonstration.")
        else:
            print("\n❌ No results obtained")
            exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Testing interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        exit(1)





