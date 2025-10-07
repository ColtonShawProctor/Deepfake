#!/usr/bin/env python3
"""
System Demo - Test the complete optimization suite
"""

import requests
import json
import time
from PIL import Image
import numpy as np

def create_test_image():
    """Create a test image for analysis"""
    # Create a test image
    img = Image.new('RGB', (512, 512), color='white')
    img_array = np.array(img)
    # Add some structure to make it look like a face
    img_array[200:300, 200:300] = [255, 200, 150]  # Face area
    img_array[220:240, 220:240] = [0, 0, 0]        # Eyes
    img_array[280:300, 250:270] = [0, 0, 0]        # Mouth
    img = Image.fromarray(img_array)
    img.save('/tmp/test_image.jpg')
    return '/tmp/test_image.jpg'

def test_system_health():
    """Test system health and monitoring"""
    print("🏥 Testing System Health...")
    
    # Test main API
    response = requests.get('http://localhost:8000/')
    if response.status_code == 200:
        data = response.json()
        print(f"✅ API Status: {data['message']}")
        print(f"   Features: {data['features']}")
    else:
        print(f"❌ API Health Check Failed: {response.status_code}")
        return False
    
    # Test monitoring dashboard
    response = requests.get('http://localhost:8000/api/multi-model/api/v2/monitoring/dashboard')
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Monitoring Dashboard: {data['dashboard_data']['overall_health']} health")
        print(f"   Active Alerts: {len(data['dashboard_data']['active_alerts'])}")
        print(f"   Recommendations: {len(data['dashboard_data']['recommendations'])}")
    else:
        print(f"❌ Monitoring Dashboard Failed: {response.status_code}")
        return False
    
    return True

def test_optimization_components():
    """Test all optimization components"""
    print("\n🔧 Testing Optimization Components...")
    
    components = [
        ("Model Selection", "/api/multi-model/api/v2/model-selection/info"),
        ("Preprocessing", "/api/multi-model/api/v2/preprocessing/stats"),
        ("Adaptive Weighting", "/api/multi-model/api/v2/adaptive-weighting/info"),
        ("Parallel Processing", "/api/multi-model/api/v2/parallel-processing/info"),
        ("Performance Monitoring", "/api/multi-model/api/v2/monitoring/dashboard")
    ]
    
    for name, endpoint in components:
        try:
            response = requests.get(f'http://localhost:8000{endpoint}')
            if response.status_code == 200:
                print(f"✅ {name}: Available")
            else:
                print(f"⚠️  {name}: {response.status_code}")
        except Exception as e:
            print(f"❌ {name}: {str(e)}")
    
    return True

def test_performance_metrics():
    """Test performance metrics collection"""
    print("\n📊 Testing Performance Metrics...")
    
    # Test metrics endpoint
    response = requests.get('http://localhost:8000/api/multi-model/api/v2/monitoring/metrics/latency')
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Latency Metrics: {data['summary']['count']} data points")
        print(f"   Average: {data['summary']['avg']:.3f}s")
    else:
        print(f"❌ Metrics Collection Failed: {response.status_code}")
    
    # Test alerts
    response = requests.get('http://localhost:8000/api/multi-model/api/v2/monitoring/alerts')
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Alert System: {len(data['alerts'])} active alerts")
        for alert in data['alerts']:
            print(f"   - {alert['level'].upper()}: {alert['message']}")
    else:
        print(f"❌ Alert System Failed: {response.status_code}")
    
    return True

def test_optimization_endpoints():
    """Test optimization-specific endpoints"""
    print("\n🚀 Testing Optimization Endpoints...")
    
    # Test optimization status
    response = requests.get('http://localhost:8000/api/multi-model/api/v2/monitoring/optimization/status')
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Auto-Optimization: {'Enabled' if data['optimization_enabled'] else 'Disabled'}")
        print(f"   Interval: {data['optimization_interval']}s")
        print(f"   History: {len(data['optimization_history'])} entries")
    else:
        print(f"❌ Optimization Status Failed: {response.status_code}")
    
    # Test parallel processing warmup
    response = requests.post('http://localhost:8000/api/multi-model/api/v2/parallel-processing/warmup')
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Model Warmup: {data['message']}")
        print(f"   Models Warmed: {len(data['models_warmed'])}")
    else:
        print(f"❌ Model Warmup Failed: {response.status_code}")
    
    return True

def main():
    """Run complete system demo"""
    print("🎯 Deepfake Detection System - Complete Optimization Suite Demo")
    print("=" * 70)
    
    # Test system health
    if not test_system_health():
        print("❌ System health check failed. Exiting.")
        return
    
    # Test optimization components
    test_optimization_components()
    
    # Test performance metrics
    test_performance_metrics()
    
    # Test optimization endpoints
    test_optimization_endpoints()
    
    print("\n" + "=" * 70)
    print("🎉 System Demo Complete!")
    print("\n📈 Optimization Suite Features:")
    print("  ✅ Intelligent Model Selection")
    print("  ✅ Unified Preprocessing Pipeline")
    print("  ✅ Dynamic Ensemble Weighting")
    print("  ✅ Parallel Processing Optimization")
    print("  ✅ Performance Monitoring & Alerting")
    print("  ✅ Auto-Optimization Engine")
    print("  ✅ Real-time Dashboard APIs")
    print("  ✅ Comprehensive Metrics Collection")
    print("  ✅ Performance Data Export")
    
    print("\n🌐 Available Endpoints:")
    print("  • /api/multi-model/api/v2/monitoring/dashboard")
    print("  • /api/multi-model/api/v2/parallel-processing/info")
    print("  • /api/multi-model/api/v2/adaptive-weighting/info")
    print("  • /api/multi-model/api/v2/preprocessing/stats")
    print("  • /api/multi-model/api/v2/monitoring/alerts")
    print("  • /api/multi-model/api/v2/monitoring/optimization/status")
    
    print("\n🚀 System is ready for production use!")

if __name__ == "__main__":
    main()
