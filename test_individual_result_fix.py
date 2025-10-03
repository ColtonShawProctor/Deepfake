#!/usr/bin/env python3
"""
Test that the individual result endpoint now returns confidence scores as percentages
"""

import requests
import json

# API configuration
BASE_URL = "http://localhost:8000"
LOGIN_URL = f"{BASE_URL}/auth/login"
INDIVIDUAL_RESULT_URL = f"{BASE_URL}/api/detection/results"

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

def test_individual_result(token: str, file_id: int):
    """Test that individual result returns confidence score as percentage"""
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        response = requests.get(f"{INDIVIDUAL_RESULT_URL}/{file_id}", headers=headers)
        response.raise_for_status()
        result = response.json()
        
        print(f"\n📊 Testing Individual Result for File ID {file_id}:")
        print("=" * 50)
        
        filename = result.get('filename', 'Unknown')
        confidence = result.get('detection_result', {}).get('confidence_score', 0)
        
        print(f"   Filename: {filename}")
        print(f"   Confidence Score: {confidence}")
        print(f"   Type: {type(confidence)}")
        print(f"   Is Percentage (>1): {confidence > 1}")
        
        if confidence > 1:
            print(f"   ✅ CORRECT: {confidence:.1f}%")
        else:
            print(f"   ❌ INCORRECT: {confidence:.1f} (should be percentage)")
        
        return result
    except Exception as e:
        print(f"❌ Failed to get individual result: {e}")
        return None

def main():
    """Main test function"""
    print("🧪 Testing Individual Result Confidence Score Fix")
    print("=" * 50)
    
    # Login
    token = login_and_get_token()
    if not token:
        return
    
    # Test individual result for the image that was showing 0.6%
    # Based on the database, this should be file ID 101
    test_file_id = 101  # 06_real_low_confidence.jpg
    
    print(f"🎯 Testing file ID {test_file_id} (06_real_low_confidence.jpg)")
    
    # Test individual result
    result = test_individual_result(token, test_file_id)
    
    if result:
        print(f"\n📋 Summary: Individual result endpoint should now return percentage")
        print("   This should fix the right side display on the website!")

if __name__ == "__main__":
    main()





