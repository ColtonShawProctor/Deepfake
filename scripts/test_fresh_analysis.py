#!/usr/bin/env python3
"""
Test that the analysis endpoint runs fresh analysis
"""

import requests
import json

# API configuration
BASE_URL = "http://localhost:8000"
LOGIN_URL = f"{BASE_URL}/auth/login"
ANALYSIS_URL = f"{BASE_URL}/api/analysis/analyze"

# Test credentials
TEST_EMAIL = "test@test.com"
TEST_PASSWORD = "test1234"

def login_and_get_token():
    """Login and return JWT token"""
    login_data = {
        "username": TEST_EMAIL,
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

def test_analysis(file_id: int, token: str):
    """Test analysis endpoint on a specific file"""
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        response = requests.post(f"{ANALYSIS_URL}/{file_id}", headers=headers)
        response.raise_for_status()
        result = response.json()
        
        print(f"\n🎬 Analysis for file {file_id}:")
        print(f"   Message: {result.get('message', 'N/A')}")
        
        if 'detection_result' in result:
            detection = result['detection_result']
            print(f"   Confidence: {detection.get('confidence_score', 0):.1f}%")
            print(f"   Is Deepfake: {detection.get('is_deepfake', False)}")
            print(f"   Model: {detection.get('analysis_metadata', {}).get('model', 'N/A')}")
        elif 'confidence_score' in result:
            print(f"   Confidence: {result.get('confidence_score', 0):.1f}%")
            print(f"   Is Deepfake: {result.get('is_deepfake', False)}")
            print(f"   Status: {result.get('status', 'N/A')}")
        
        return result
    except Exception as e:
        print(f"❌ Analysis failed for file {file_id}: {e}")
        return None

def main():
    """Main test function"""
    print("🧪 Testing Fresh Analysis Endpoint")
    print("=" * 50)
    
    # Login
    token = login_and_get_token()
    if not token:
        return
    
    # Test analysis on a few recent files
    test_files = [90, 89, 88]  # Recent file IDs from the database
    
    for file_id in test_files:
        test_analysis(file_id, token)
        print("-" * 30)

if __name__ == "__main__":
    main()





