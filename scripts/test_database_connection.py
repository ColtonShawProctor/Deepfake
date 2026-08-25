#!/usr/bin/env python3
"""
Test script to verify database connection and detection results endpoint
"""

import sqlite3
import requests
import json
from datetime import datetime

def test_database_connection():
    """Test direct database connection"""
    print("🔍 Testing database connection...")
    
    try:
        conn = sqlite3.connect('deepfake.db')
        cursor = conn.cursor()
        
        # Check tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"✅ Tables found: {[table[0] for table in tables]}")
        
        # Check detection results count
        cursor.execute("SELECT COUNT(*) FROM detection_results;")
        results_count = cursor.fetchone()[0]
        print(f"✅ Detection results count: {results_count}")
        
        # Check users count
        cursor.execute("SELECT COUNT(*) FROM users;")
        users_count = cursor.fetchone()[0]
        print(f"✅ Users count: {users_count}")
        
        # Check media files count
        cursor.execute("SELECT COUNT(*) FROM media_files;")
        files_count = cursor.fetchone()[0]
        print(f"✅ Media files count: {files_count}")
        
        # Check sample detection result
        if results_count > 0:
            cursor.execute("SELECT * FROM detection_results LIMIT 1;")
            sample_result = cursor.fetchone()
            print(f"✅ Sample result: ID={sample_result[0]}, confidence={sample_result[2]}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

def test_api_endpoints():
    """Test API endpoints"""
    print("\n🔍 Testing API endpoints...")
    
    base_url = "http://localhost:8000"
    
    try:
        # Test health endpoint
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            print("✅ Health endpoint working")
            health_data = response.json()
            print(f"   Database status: {health_data.get('database', {}).get('status', 'unknown')}")
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
            return False
            
        # Test root endpoint
        response = requests.get(f"{base_url}/", timeout=10)
        if response.status_code == 200:
            print("✅ Root endpoint working")
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
            
        # Test detection results endpoint (without auth - should fail with 401)
        response = requests.get(f"{base_url}/api/detection/results?page=1&limit=10", timeout=10)
        if response.status_code == 401:
            print("✅ Detection results endpoint exists (auth required)")
        else:
            print(f"⚠️  Detection results endpoint returned: {response.status_code}")
            
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API server. Is it running?")
        return False
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

def test_cors():
    """Test CORS headers"""
    print("\n🔍 Testing CORS configuration...")
    
    base_url = "http://localhost:8000"
    
    try:
        # Test OPTIONS request to check CORS headers
        response = requests.options(f"{base_url}/api/detection/results", timeout=10)
        
        cors_headers = {
            'Access-Control-Allow-Origin': response.headers.get('Access-Control-Allow-Origin'),
            'Access-Control-Allow-Methods': response.headers.get('Access-Control-Allow-Methods'),
            'Access-Control-Allow-Headers': response.headers.get('Access-Control-Allow-Headers'),
        }
        
        print("✅ CORS headers found:")
        for header, value in cors_headers.items():
            print(f"   {header}: {value}")
            
        # Check if localhost:3000 is allowed
        if any('localhost:3000' in str(value) or '*' in str(value) for value in cors_headers.values()):
            print("✅ CORS should allow localhost:3000")
        else:
            print("⚠️  CORS may not allow localhost:3000")
            
        return True
        
    except Exception as e:
        print(f"❌ CORS test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Starting database and API tests...\n")
    
    # Test database
    db_ok = test_database_connection()
    
    # Test API
    api_ok = test_api_endpoints()
    
    # Test CORS
    cors_ok = test_cors()
    
    print("\n" + "="*50)
    print("📊 Test Results Summary:")
    print(f"   Database: {'✅ OK' if db_ok else '❌ FAILED'}")
    print(f"   API: {'✅ OK' if api_ok else '❌ FAILED'}")
    print(f"   CORS: {'✅ OK' if cors_ok else '❌ FAILED'}")
    
    if all([db_ok, api_ok, cors_ok]):
        print("\n🎉 All tests passed! The system should be working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
        
        if not db_ok:
            print("   💡 Database issue: Check if deepfake.db exists and has correct schema")
        if not api_ok:
            print("   💡 API issue: Make sure the backend server is running on port 8000")
        if not cors_ok:
            print("   💡 CORS issue: Check CORS middleware configuration")

if __name__ == "__main__":
    main()
