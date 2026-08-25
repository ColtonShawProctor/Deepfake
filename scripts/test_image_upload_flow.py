#!/usr/bin/env python3
"""
Test script to verify the complete image upload and analysis flow
"""

import requests
import json
import time

# Configuration
BASE_URL = "http://localhost:8000"
TEST_EMAIL = "test@example.com"
TEST_PASSWORD = "changeme123"

def test_complete_flow():
    """Test the complete image upload and analysis flow"""
    
    print("🧪 Testing complete image upload and analysis flow...")
    
    # Step 1: Login to get authentication token
    print("\n1️⃣ Logging in...")
    login_data = {
        "username": TEST_EMAIL,
        "password": TEST_PASSWORD
    }
    
    login_response = requests.post(
        f"{BASE_URL}/auth/login",
        data=login_data,
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    
    if login_response.status_code != 200:
        print(f"❌ Login failed: {login_response.status_code}")
        print(f"Response: {login_response.text}")
        return False
    
    login_result = login_response.json()
    token = login_result["access_token"]
    print(f"✅ Login successful, token: {token[:20]}...")
    
    # Step 2: Upload a test image
    print("\n2️⃣ Uploading test image...")
    
    # Use the existing test image
    with open("test_image.png", "rb") as f:
        files = {"file": ("test_image.png", f, "image/png")}
        headers = {"Authorization": f"Bearer {token}"}
        
        upload_response = requests.post(
            f"{BASE_URL}/api/upload",
            files=files,
            headers=headers
        )
    
    if upload_response.status_code != 200:
        print(f"❌ Upload failed: {upload_response.status_code}")
        print(f"Response: {upload_response.text}")
        return False
    
    upload_result = upload_response.json()
    file_id = upload_result["file_id"]
    print(f"✅ Upload successful, file ID: {file_id}")
    
    # Step 3: Analyze the uploaded image
    print("\n3️⃣ Analyzing image for deepfake detection...")
    
    analysis_response = requests.post(
        f"{BASE_URL}/api/detection/analyze/{file_id}",
        headers={"Authorization": f"Bearer {token}"}
    )
    
    if analysis_response.status_code != 200:
        print(f"❌ Analysis failed: {analysis_response.status_code}")
        print(f"Response: {analysis_response.text}")
        return False
    
    analysis_result = analysis_response.json()
    print(f"✅ Analysis successful!")
    print(f"   - Confidence: {analysis_result['detection_result']['confidence_score']:.3f}")
    print(f"   - Is Deepfake: {analysis_result['detection_result']['is_deepfake']}")
    print(f"   - Processing Time: {analysis_result['detection_result']['processing_time_seconds']:.3f}s")
    
    # Step 4: Get the analysis results
    print("\n4️⃣ Retrieving analysis results...")
    
    results_response = requests.get(
        f"{BASE_URL}/api/detection/results/{file_id}",
        headers={"Authorization": f"Bearer {token}"}
    )
    
    if results_response.status_code != 200:
        print(f"❌ Results retrieval failed: {results_response.status_code}")
        print(f"Response: {results_response.text}")
        return False
    
    results = results_response.json()
    print(f"✅ Results retrieved successfully!")
    print(f"   - File: {results['filename']}")
    print(f"   - Analysis Time: {results['created_at']}")
    
    # Step 5: Test file listing
    print("\n5️⃣ Testing file listing...")
    
    files_response = requests.get(
        f"{BASE_URL}/api/files",
        headers={"Authorization": f"Bearer {token}"}
    )
    
    if files_response.status_code != 200:
        print(f"❌ File listing failed: {files_response.status_code}")
        print(f"Response: {files_response.text}")
        return False
    
    files = files_response.json()
    print(f"✅ File listing successful! Found {len(files)} files")
    
    # Summary
    print("\n🎉 Complete flow test successful!")
    print("✅ Login: Working")
    print("✅ Upload: Working")
    print("✅ Analysis: Working")
    print("✅ Results: Working")
    print("✅ File Listing: Working")
    
    return True

def test_error_handling():
    """Test error handling scenarios"""
    
    print("\n🧪 Testing error handling...")
    
    # Test 1: Upload without authentication
    print("\n1️⃣ Testing upload without authentication...")
    with open("test_image.png", "rb") as f:
        files = {"file": ("test_image.png", f, "image/png")}
        
        response = requests.post(f"{BASE_URL}/api/upload", files=files)
    
    if response.status_code == 401:
        print("✅ Unauthorized access properly blocked")
    else:
        print(f"❌ Expected 401, got {response.status_code}")
    
    # Test 2: Upload invalid file type
    print("\n2️⃣ Testing upload with invalid file type...")
    
    # Create a text file
    with open("test.txt", "w") as f:
        f.write("This is not an image")
    
    # Login first
    login_data = {"username": TEST_EMAIL, "password": TEST_PASSWORD}
    login_response = requests.post(
        f"{BASE_URL}/auth/login",
        data=login_data,
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    token = login_response.json()["access_token"]
    
    # Try to upload text file
    with open("test.txt", "rb") as f:
        files = {"file": ("test.txt", f, "text/plain")}
        headers = {"Authorization": f"Bearer {token}"}
        
        response = requests.post(f"{BASE_URL}/api/upload", files=files, headers=headers)
    
    if response.status_code == 400:
        print("✅ Invalid file type properly rejected")
    else:
        print(f"❌ Expected 400, got {response.status_code}")
    
    # Clean up
    import os
    if os.path.exists("test.txt"):
        os.remove("test.txt")
    
    print("✅ Error handling tests completed")

if __name__ == "__main__":
    try:
        # Test the complete flow
        success = test_complete_flow()
        
        if success:
            # Test error handling
            test_error_handling()
        else:
            print("❌ Main flow failed, skipping error handling tests")
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()





