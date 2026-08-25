#!/usr/bin/env python3
"""
Script to upload a video file using the API
"""
import requests
import os
from pathlib import Path

def upload_video():
    """Upload a video file to test the API"""
    
    # API endpoint
    base_url = "http://localhost:8000"
    upload_url = f"{base_url}/api/video/upload"
    
    # Authentication token (from previous login)
    token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxIiwiZXhwIjoxNzU1MjA0MzAyfQ.uHEi57p1nuT5FANW0RQ0_hQm5mg10sGZ-UNOQIVeKmU"
    
    headers = {
        "Authorization": f"Bearer {token}"
    }
    
    # Use an existing video file for testing
    video_path = "uploads/videos/154c2800-260d-4942-83a4-83495d3e8631.mp4"
    
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return
    
    print(f"Uploading video: {video_path}")
    
    try:
        with open(video_path, "rb") as video_file:
            files = {
                "video_file": ("test_video.mp4", video_file, "video/mp4")
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
        print(f"Error uploading video: {e}")

if __name__ == "__main__":
    upload_video()
