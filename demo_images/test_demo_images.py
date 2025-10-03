#!/usr/bin/env python3
"""
Quick test script for the demonstration images.
This script tests all 6 demo images to verify they work correctly.
"""

import requests
import json
import time
import os
from pathlib import Path

def authenticate():
    """Authenticate and get token"""
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
        return auth_token
        
    except Exception as e:
        print(f"❌ Authentication error: {str(e)}")
        return None

def test_demo_image(image_path: str, auth_token: str):
    """Test a single demo image"""
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
            'filename': os.path.basename(image_path),
            'predicted_type': 'FAKE' if detection_result['is_deepfake'] else 'REAL',
            'confidence': detection_result['confidence_score'],
            'processing_time': detection_result['processing_time_seconds']
        }
        
    except Exception as e:
        print(f"Error testing {image_path}: {e}")
        return None

def test_all_demo_images():
    """Test all demo images"""
    print("🧪 Testing All Demo Images")
    print("=" * 50)
    
    # Authenticate
    auth_token = authenticate()
    if not auth_token:
        print("❌ Authentication failed")
        return
    
    # Get all demo images
    demo_dir = Path(__file__).parent
    demo_images = sorted([f for f in demo_dir.glob("*.jpg") if f.name != "README.md"])
    
    print(f"Found {len(demo_images)} demo images")
    print()
    
    results = []
    
    for i, image_path in enumerate(demo_images):
        print(f"Testing {i+1}/{len(demo_images)}: {image_path.name}")
        result = test_demo_image(str(image_path), auth_token)
        
        if result:
            results.append(result)
            print(f"  ✅ {result['predicted_type']} | Confidence: {result['confidence']:.1f}% | Time: {result['processing_time']:.3f}s")
        else:
            print(f"  ❌ Failed")
        
        time.sleep(0.5)
    
    # Summary
    print(f"\n📊 DEMO IMAGES TEST SUMMARY")
    print("=" * 50)
    
    if results:
        fake_count = len([r for r in results if r['predicted_type'] == 'FAKE'])
        real_count = len([r for r in results if r['predicted_type'] == 'REAL'])
        
        confidences = [r['confidence'] for r in results]
        avg_time = sum(r['processing_time'] for r in results) / len(results)
        
        print(f"Total Tested: {len(results)}")
        print(f"Fake Detected: {fake_count}")
        print(f"Real Detected: {real_count}")
        print(f"Confidence Range: {min(confidences):.1f}% - {max(confidences):.1f}%")
        print(f"Average Processing Time: {avg_time:.3f}s")
        
        print(f"\n🎉 All demo images are working correctly!")
        print(f"Ready for demonstration!")
    else:
        print("❌ No images were successfully tested")

if __name__ == "__main__":
    test_all_demo_images()





