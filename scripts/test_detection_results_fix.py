#!/usr/bin/env python3
"""
Test that the detection results endpoint now returns confidence scores as percentages
"""

import requests
import json

# API configuration
BASE_URL = "http://localhost:8000"
LOGIN_URL = f"{BASE_URL}/auth/login"
DETECTION_RESULTS_URL = f"{BASE_URL}/api/detection/results"

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

def test_detection_results(token: str):
    """Test that detection results return confidence scores as percentages"""
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        response = requests.get(DETECTION_RESULTS_URL, headers=headers)
        response.raise_for_status()
        results = response.json()
        
        print(f"\n📊 Testing Detection Results Confidence Score Format:")
        print("=" * 60)
        
        for i, result in enumerate(results[:5]):  # Check first 5 results
            filename = result.get('filename', 'Unknown')
            confidence = result.get('detection_result', {}).get('confidence_score', 0)
            
            print(f"\nResult {i+1}: {filename}")
            print(f"   Confidence Score: {confidence}")
            print(f"   Type: {type(confidence)}")
            print(f"   Is Percentage (>1): {confidence > 1}")
            
            if confidence > 1:
                print(f"   ✅ CORRECT: {confidence:.1f}%")
            else:
                print(f"   ❌ INCORRECT: {confidence:.1f} (should be percentage)")
        
        return results
    except Exception as e:
        print(f"❌ Failed to get detection results: {e}")
        return None

def main():
    """Main test function"""
    print("🧪 Testing Detection Results Confidence Score Fix")
    print("=" * 50)
    
    # Login
    token = login_and_get_token()
    if not token:
        return
    
    # Test detection results
    results = test_detection_results(token)
    
    if results:
        print(f"\n📋 Summary: Checked {len(results)} results")
        print("   Look for confidence scores > 1 (percentages) instead of < 1 (decimals)")
        print("   This should fix the right side results display!")

if __name__ == "__main__":
    main()





