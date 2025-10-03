#!/usr/bin/env python3
"""
Test script for backend authentication
"""

import requests
import json
import time

# Configuration
BASE_URL = "http://localhost:8000"
TEST_EMAIL = "demo@example.com"
TEST_PASSWORD = "password123"

def test_backend_health():
    """Test if backend is running"""
    try:
        response = requests.get(f"{BASE_URL}/docs")
        print(f"✅ Backend is running (status: {response.status_code})")
        return True
    except requests.exceptions.ConnectionError:
        print("❌ Backend is not running")
        return False

def test_login():
    """Test login endpoint"""
    print("\n🔐 Testing login endpoint...")
    
    # Prepare form data
    form_data = {
        'username': TEST_EMAIL,
        'password': TEST_PASSWORD
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/auth/login",
            data=form_data,
            headers={'Content-Type': 'application/x-www-form-urlencoded'}
        )
        
        print(f"Login response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Login successful!")
            print(f"Response: {json.dumps(data, indent=2)}")
            
            # Check if response has required fields
            required_fields = ['success', 'message', 'access_token', 'token_type', 'user']
            missing_fields = [field for field in required_fields if field not in data]
            
            if missing_fields:
                print(f"❌ Missing required fields: {missing_fields}")
                return None
            else:
                print("✅ All required fields present")
                return data['access_token']
        else:
            print(f"❌ Login failed: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Login error: {e}")
        return None

def test_auth_me(token):
    """Test /auth/me endpoint with token"""
    if not token:
        print("❌ No token provided for /auth/me test")
        return False
    
    print(f"\n👤 Testing /auth/me endpoint...")
    
    try:
        headers = {'Authorization': f'Bearer {token}'}
        response = requests.get(f"{BASE_URL}/auth/me", headers=headers)
        
        print(f"/auth/me response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ /auth/me successful!")
            print(f"User data: {json.dumps(data, indent=2)}")
            return True
        else:
            print(f"❌ /auth/me failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ /auth/me error: {e}")
        return False

def test_protected_endpoint(token):
    """Test a protected endpoint"""
    if not token:
        print("❌ No token provided for protected endpoint test")
        return False
    
    print(f"\n🔒 Testing protected endpoint...")
    
    try:
        headers = {'Authorization': f'Bearer {token}'}
        response = requests.get(f"{BASE_URL}/api/video/list", headers=headers)
        
        print(f"Protected endpoint response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Protected endpoint accessible!")
            print(f"Response: {json.dumps(data, indent=2)}")
            return True
        elif response.status_code == 401:
            print("❌ Protected endpoint returned 401 (unauthorized)")
            return False
        else:
            print(f"⚠️ Protected endpoint returned unexpected status: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Protected endpoint error: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Backend Authentication Test")
    print("=" * 40)
    
    # Test backend health
    if not test_backend_health():
        return
    
    # Test login
    token = test_login()
    
    if token:
        # Test /auth/me endpoint
        test_auth_me(token)
        
        # Test protected endpoint
        test_protected_endpoint(token)
        
        print(f"\n🎉 All tests completed!")
    else:
        print(f"\n❌ Login test failed, skipping other tests")

if __name__ == "__main__":
    main()
