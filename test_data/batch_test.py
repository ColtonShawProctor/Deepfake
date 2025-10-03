#!/usr/bin/env python3
"""
Batch Testing Script for Deepfake Detection Test Dataset

This script tests all samples in the test dataset against your deepfake detection system.
"""

import os
import sys
import json
import time
from pathlib import Path
import requests

# Configuration
API_BASE_URL = "http://localhost:8000"  # Adjust to your API URL
TEST_DATA_DIR = "test_data"
METADATA_FILE = "dataset_metadata.json"

def load_test_metadata():
    """Load test dataset metadata"""
    metadata_path = Path(TEST_DATA_DIR) / METADATA_FILE
    with open(metadata_path, 'r') as f:
        return json.load(f)

def test_image_sample(image_path: str, api_url: str) -> dict:
    """Test a single image sample"""
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{api_url}/detect", files=files, timeout=30)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        return {"error": str(e)}

def test_video_sample(video_path: str, api_url: str) -> dict:
    """Test a single video sample"""
    try:
        with open(video_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{api_url}/detect-video", files=files, timeout=60)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        return {"error": str(e)}

def run_batch_tests():
    """Run tests on all samples"""
    print("🧪 Running Batch Tests...")
    print("=" * 50)
    
    # Load metadata
    metadata = load_test_metadata()
    samples = metadata["samples"]
    expected_results = metadata["expected_results"]
    
    results = []
    total_samples = len(samples)
    
    for i, (sample_name, sample_info) in enumerate(samples.items(), 1):
        print(f"\n[{i}/{total_samples}] Testing {sample_name}...")
        
        # Skip placeholder files
        if sample_info.get("is_placeholder", False):
            print(f"   ⏭️  Skipping placeholder file")
            continue
        
        # Determine API endpoint based on file type
        if sample_info["file_type"] == "image":
            result = test_image_sample(sample_info["path"], f"{API_BASE_URL}/api")
        else:
            result = test_video_sample(sample_info["path"], f"{API_BASE_URL}/api")
        
        # Store result
        test_result = {
            "sample_name": sample_name,
            "file_path": sample_info["path"],
            "expected": expected_results[sample_name],
            "actual_result": result,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        results.append(test_result)
        
        # Print result summary
        if "error" in result:
            print(f"   ❌ Error: {result['error']}")
        else:
            print(f"   ✅ Test completed")
    
    # Generate report
    generate_test_report(results, metadata)
    
    return results

def generate_test_report(results: list, metadata: dict):
    """Generate a comprehensive test report"""
    print("\n📊 Generating Test Report...")
    
    report = {
        "test_summary": {
            "total_samples": len(results),
            "successful_tests": len([r for r in results if "error" not in r["actual_result"]]),
            "failed_tests": len([r for r in results if "error" in r["actual_result"]]),
            "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "results": results
    }
    
    # Save report
    report_file = Path(TEST_DATA_DIR) / "test_results.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"   ✅ Test report saved: {report_file}")
    
    # Print summary
    print(f"\n📈 Test Summary:")
    print(f"   Total samples: {report['test_summary']['total_samples']}")
    print(f"   Successful tests: {report['test_summary']['successful_tests']}")
    print(f"   Failed tests: {report['test_summary']['failed_tests']}")

if __name__ == "__main__":
    run_batch_tests()
