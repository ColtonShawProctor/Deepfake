#!/usr/bin/env python3
"""
Script to upload an image file using the API
"""
import requests
import os
from pathlib import Path

def upload_image():
    """Upload an image file to test the API"""
    
    # API endpoint
    base_url = "http://localhost:8000"
    upload_url = f"{base_url}/api/upload"
    
    # Authentication token (from previous login)
    token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxIiwiZXhwIjoxNzU1MjA0MzAyfQ.uHEi57p1nuT5FANW0RQ0_hQm5mg10sGZ-UNOQIVeKmU"
    
    headers = {
        "Authorization": f"Bearer {token}"
    }
    
    # Use an existing image file for testing
    image_path = "uploads/fb9b36e9-3ead-46cf-afc6-26d13ab92503.jpg"
    
    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        return
    
    print(f"Uploading image: {image_path}")
    
    try:
        with open(image_path, "rb") as image_file:
            files = {
                "file": ("test_image.jpg", image_file, "image/jpeg")
            }
            
            response = requests.post(
                upload_url,
                headers=headers,
                files=files
            )
            
            print(f"Response status: {response.status_code}")
            print(f"Response body: {response.text}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"Upload successful! File ID: {result.get('file_id')}")
                print(f"Message: {result.get('message')}")
                
                # Now try to get the results
                file_id = result.get('file_id')
                if file_id:
                    print(f"\nTesting detection results endpoint...")
                    results_url = f"{base_url}/api/detection/results/{file_id}"
                    results_response = requests.get(results_url, headers=headers)
                    print(f"Results status: {results_response.status_code}")
                    print(f"Results body: {results_response.text}")
            else:
                print(f"Upload failed: {response.text}")
                
    except Exception as e:
        print(f"Error uploading image: {e}")

if __name__ == "__main__":
    upload_image()
