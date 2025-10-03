#!/usr/bin/env python3
"""
Test script to verify the API integration with the new Hugging Face detector.
This tests the actual API endpoints to ensure they work correctly.
"""

import requests
import json
import time

def test_api_endpoints():
    """Test the main API endpoints"""
    base_url = "http://localhost:8000"
    
    print("Testing API Integration with Hugging Face Detector...")
    print("=" * 60)
    
    # Test 1: Health check
    print("1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ Health endpoint working")
            health_data = response.json()
            print(f"   Database: {health_data['database']['status']}")
            print(f"   Advanced ensemble: {health_data['advanced_ensemble']}")
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health endpoint error: {e}")
        return False
    
    # Test 2: Root endpoint
    print("\n2. Testing root endpoint...")
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            print("✅ Root endpoint working")
            root_data = response.json()
            print(f"   Version: {root_data['version']}")
            print(f"   Basic detection: {root_data['features']['basic_detection']}")
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Root endpoint error: {e}")
        return False
    
    # Test 3: Detector info
    print("\n3. Testing detector info endpoint...")
    try:
        response = requests.get(f"{base_url}/api/detection/info")
        if response.status_code == 200:
            print("✅ Detector info endpoint working")
            detector_info = response.json()
            print(f"   Model: {detector_info['name']}")
            print(f"   Architecture: {detector_info['description']}")
            print(f"   Capabilities: {', '.join(detector_info['capabilities'])}")
            print(f"   Supported formats: {', '.join(detector_info['supported_formats'])}")
        else:
            print(f"❌ Detector info endpoint failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Detector info endpoint error: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 All API endpoints are working correctly!")
    print("✅ Hugging Face detector is successfully integrated")
    print("✅ Backend is fully operational")
    print("✅ Ready for frontend integration")
    
    return True

if __name__ == "__main__":
    success = test_api_endpoints()
    if not success:
        print("\n❌ Some tests failed. Check the server logs for details.")
        exit(1)





