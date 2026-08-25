#!/usr/bin/env python3
"""
Test script to verify authentication flow and dashboard endpoints
"""

import requests
import json

# Test configuration
BASE_URL = "http://localhost:8000"

def test_auth_flow():
    """Test complete authentication flow"""
    
    print("Testing authentication flow...")
    
    # 1. Login
    login_data = {
        "username": "test@test.com",
        "password": "test1234"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/auth/login", data=login_data)
        print(f"Login response: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            access_token = result.get("access_token")
            if access_token:
                print("✓ Login successful!")
                print(f"Token: {access_token[:20]}...")
                
                # 2. Test getting all results (dashboard endpoint)
                headers = {
                    "Authorization": f"Bearer {access_token}"
                }
                
                print("\nTesting dashboard endpoint...")
                results_response = requests.get(f"{BASE_URL}/api/detection/results", headers=headers)
                print(f"Dashboard response: {results_response.status_code}")
                
                if results_response.status_code == 200:
                    results = results_response.json()
                    print(f"✓ Dashboard working! Found {len(results)} results")
                    
                    # Show first few results
                    for i, result in enumerate(results[:3]):
                        print(f"  Result {i+1}: File {result['file_id']} - {result['filename']}")
                        print(f"    Confidence: {result['detection_result']['confidence_score']:.3f}")
                        print(f"    Deepfake: {result['detection_result']['is_deepfake']}")
                else:
                    print(f"✗ Dashboard failed: {results_response.text}")
                    
                # 3. Test individual result endpoint
                print("\nTesting individual result endpoint...")
                if results_response.status_code == 200 and len(results) > 0:
                    first_file_id = results[0]['file_id']
                    individual_response = requests.get(f"{BASE_URL}/api/detection/results/{first_file_id}", headers=headers)
                    print(f"Individual result response: {individual_response.status_code}")
                    
                    if individual_response.status_code == 200:
                        print("✓ Individual result endpoint working!")
                    else:
                        print(f"✗ Individual result failed: {individual_response.text}")
                
            else:
                print("✗ No access token in response")
                return
        else:
            print(f"✗ Login failed: {response.text}")
            return
            
    except Exception as e:
        print(f"Error during authentication test: {e}")

if __name__ == "__main__":
    test_auth_flow()
