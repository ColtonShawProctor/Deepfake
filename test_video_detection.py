#!/usr/bin/env python3
"""
Test video detection with Hugging Face model and find best demo videos
"""

import os
import json
import time
import requests
from pathlib import Path
from typing import List, Dict, Any

# API configuration
BASE_URL = "http://localhost:8000"
LOGIN_URL = f"{BASE_URL}/auth/login"
UPLOAD_URL = f"{BASE_URL}/api/video/upload"
ANALYSIS_URL = f"{BASE_URL}/api/analysis/analyze"

# Test credentials
TEST_EMAIL = "test@test.com"
TEST_PASSWORD = "test1234"

def login_and_get_token() -> str:
    """Login and return JWT token"""
    # OAuth2PasswordRequestForm expects form data, not JSON
    login_data = {
        "username": TEST_EMAIL,  # OAuth2 form uses 'username' field
        "password": TEST_PASSWORD
    }
    
    try:
        response = requests.post(LOGIN_URL, data=login_data)
        response.raise_for_status()
        token = response.json()["access_token"]
        print(f"✅ Login successful, token obtained")
        return token
    except Exception as e:
        print(f"❌ Login failed: {e}")
        return None

def upload_video(video_path: str, token: str) -> Dict[str, Any]:
    """Upload a video file"""
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        with open(video_path, 'rb') as f:
            files = {'video_file': (os.path.basename(video_path), f, 'video/mp4')}
            response = requests.post(UPLOAD_URL, headers=headers, files=files)
            response.raise_for_status()
            result = response.json()
            print(f"✅ Uploaded {os.path.basename(video_path)}: {result.get('file_id')}")
            return result
    except Exception as e:
        print(f"❌ Upload failed for {os.path.basename(video_path)}: {e}")
        return None

def analyze_video(file_id: int, token: str) -> Dict[str, Any]:
    """Analyze a video file"""
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        response = requests.post(f"{ANALYSIS_URL}/{file_id}", headers=headers)
        response.raise_for_status()
        result = response.json()
        print(f"✅ Analysis completed for file {file_id}")
        return result
    except Exception as e:
        print(f"❌ Analysis failed for file {file_id}: {e}")
        return None

def test_video_detection(video_path: str, token: str) -> Dict[str, Any]:
    """Test a single video through the full pipeline"""
    print(f"\n🎬 Testing: {os.path.basename(video_path)}")
    
    # Upload video
    upload_result = upload_video(video_path, token)
    if not upload_result:
        return None
    
    file_id = upload_result['file_id']
    
    # Wait a moment for processing
    time.sleep(2)
    
    # Analyze video
    analysis_result = analyze_video(file_id, token)
    if not analysis_result:
        return None
    
    # Extract key results
    detection_result = analysis_result.get('detection_result', {})
    
    result = {
        'filename': os.path.basename(video_path),
        'file_path': video_path,
        'file_id': file_id,
        'confidence': detection_result.get('confidence_score', 0),
        'is_deepfake': detection_result.get('is_deepfake', False),
        'processing_time': detection_result.get('analysis_metadata', {}).get('processing_time', 0),
        'model': detection_result.get('analysis_metadata', {}).get('model', 'unknown')
    }
    
    print(f"   📊 Confidence: {result['confidence']:.1f}%")
    print(f"   🎭 Is Deepfake: {result['is_deepfake']}")
    print(f"   ⏱️  Processing Time: {result['processing_time']:.2f}s")
    
    return result

def find_best_demo_videos(video_results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Find the best 2 real and 2 fake videos for demo"""
    
    # Separate real and fake videos
    real_videos = [v for v in video_results if not v['is_deepfake']]
    fake_videos = [v for v in video_results if v['is_deepfake']]
    
    print(f"\n📊 Found {len(real_videos)} real videos and {len(fake_videos)} fake videos")
    
    # Sort by confidence (higher is better for real, lower is better for fake)
    real_videos.sort(key=lambda x: x['confidence'], reverse=True)
    fake_videos.sort(key=lambda x: x['confidence'])  # Lower confidence = more clearly fake
    
    # Select top 2 from each category
    best_real = real_videos[:2] if len(real_videos) >= 2 else real_videos
    best_fake = fake_videos[:2] if len(fake_videos) >= 2 else fake_videos
    
    return {
        'real': best_real,
        'fake': best_fake
    }

def main():
    """Main testing function"""
    print("🎬 Testing Video Detection with Hugging Face Model")
    print("=" * 60)
    
    # Check if backend is running
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code != 200:
            print("❌ Backend not responding. Please start the server first.")
            return
        print("✅ Backend is running")
    except:
        print("❌ Cannot connect to backend. Please start the server first.")
        return
    
    # Login
    token = login_and_get_token()
    if not token:
        return
    
    # Test videos from Celeb-DF-v2
    test_videos = [
        # Real videos
        "Celeb-DF-v2/Celeb-real/id5_0000.mp4",
        "Celeb-DF-v2/Celeb-real/id59_0000.mp4", 
        "Celeb-DF-v2/Celeb-real/id60_0000.mp4",
        "Celeb-DF-v2/YouTube-real/00250.mp4",
        
        # Fake videos
        "Celeb-DF-v2/Celeb-synthesis/id59_id61_0000.mp4",
        "Celeb-DF-v2/Celeb-synthesis/id5_id61_0008.mp4",
        "Celeb-DF-v2/Celeb-synthesis/id60_id59_0000.mp4",
        "Celeb-DF-v2/Celeb-synthesis/id59_id5_0000.mp4"
    ]
    
    # Filter to only existing files
    existing_videos = [v for v in test_videos if os.path.exists(v)]
    print(f"\n🎯 Testing {len(existing_videos)} videos...")
    
    # Test each video
    results = []
    for video_path in existing_videos:
        result = test_video_detection(video_path, token)
        if result:
            results.append(result)
        time.sleep(1)  # Small delay between tests
    
    if not results:
        print("❌ No videos were successfully tested")
        return
    
    # Find best demo videos
    best_videos = find_best_demo_videos(results)
    
    # Display results
    print(f"\n🏆 BEST DEMO VIDEOS SELECTED:")
    print("=" * 60)
    
    print(f"\n🔴 REAL VIDEOS (2):")
    for i, video in enumerate(best_videos['real'], 1):
        print(f"  {i}. {video['filename']}")
        print(f"     Confidence: {video['confidence']:.1f}%")
        print(f"     Path: {video['file_path']}")
    
    print(f"\n🟢 FAKE VIDEOS (2):")
    for i, video in enumerate(best_videos['fake'], 1):
        print(f"  {i}. {video['filename']}")
        print(f"     Confidence: {video['confidence']:.1f}%")
        print(f"     Path: {video['file_path']}")
    
    # Save results
    demo_results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_tested': len(results),
        'best_demo_videos': best_videos
    }
    
    with open('video_demo_results.json', 'w') as f:
        json.dump(demo_results, f, indent=2)
    
    print(f"\n💾 Results saved to video_demo_results.json")
    
    # Create demo directory and copy videos
    demo_dir = Path("demo_videos")
    demo_dir.mkdir(exist_ok=True)
    
    print(f"\n📁 Copying demo videos to demo_videos/ directory...")
    
    for category, videos in best_videos.items():
        for i, video in enumerate(videos, 1):
            src_path = Path(video['file_path'])
            dst_path = demo_dir / f"{category}_{i:02d}_{video['filename']}"
            
            try:
                import shutil
                shutil.copy2(src_path, dst_path)
                print(f"  ✅ Copied {src_path.name} → {dst_path.name}")
            except Exception as e:
                print(f"  ❌ Failed to copy {src_path.name}: {e}")
    
    print(f"\n🎉 Video demo setup complete!")
    print(f"   Demo videos are in: demo_videos/")
    print(f"   Results saved to: video_demo_results.json")

if __name__ == "__main__":
    main()
