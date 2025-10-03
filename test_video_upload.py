#!/usr/bin/env python3
"""
Test script for video upload and analysis
"""

import requests
import json
import time

# Test configuration
BASE_URL = "http://localhost:8000"
TEST_VIDEO_PATH = "uploads/videos/9cd3bab5-1e39-4bf7-b1c9-8450c8e0b416.mp4"

def login_user():
    """Login with test user credentials"""
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
                return access_token
            else:
                print("✗ No access token in response")
                return None
        else:
            print(f"✗ Login failed: {response.text}")
            return None
            
    except Exception as e:
        print(f"Error during login: {e}")
        return None

def test_video_upload(access_token):
    """Test video upload and analysis with authentication"""
    
    print("\nTesting video upload and analysis...")
    
    # Set up headers with authentication
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    
    # Test video upload
    try:
        with open(TEST_VIDEO_PATH, "rb") as video_file:
            files = {
                "video_file": ("test_video.mp4", video_file, "video/mp4")
            }
            
            response = requests.post(
                f"{BASE_URL}/api/video/upload", 
                files=files,
                headers=headers
            )
            
            print(f"Video upload response: {response.status_code}")
            print(f"Response body: {response.text}")
            
            if response.status_code == 200:
                print("✓ Video upload successful!")
                result = response.json()
                file_id = result.get("file_id")
                if file_id:
                    print(f"File ID: {file_id}")
                    
                    # Wait a moment for analysis to complete
                    print("Waiting for analysis to complete...")
                    time.sleep(3)
                    
                    # Test getting results
                    results_response = requests.get(
                        f"{BASE_URL}/api/detection/results/{file_id}",
                        headers=headers
                    )
                    print(f"Results response: {results_response.status_code}")
                    print(f"Results body: {results_response.text}")
                    
                    if results_response.status_code == 200:
                        print("✓ Video analysis results retrieved successfully!")
                    else:
                        print(f"✗ Failed to get results: {results_response.text}")
                        
            else:
                print(f"✗ Video upload failed with status {response.status_code}")
                
    except Exception as e:
        print(f"Error testing video upload: {e}")

def main():
    """Main test function"""
    print("Starting video upload test...")
    
    # First, let's check if the server is running
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Server health check: {response.status_code}")
        if response.status_code == 200:
            print("Server is running!")
        else:
            print("Server health check failed")
            return
    except Exception as e:
        print(f"Server not accessible: {e}")
        return
    
    # Login with test user
    access_token = login_user()
    if not access_token:
        print("Cannot proceed without authentication")
        return
    
    # Test video upload with authentication
    test_video_upload(access_token)

if __name__ == "__main__":
    main()
