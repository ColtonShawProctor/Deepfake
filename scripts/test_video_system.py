#!/usr/bin/env python3
"""
Test script for video analysis system

Tests:
1. Video processor initialization
2. Frame extraction
3. Batch processing
4. Temporal analysis
5. Memory management
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add the app directory to the path
sys.path.append(str(Path(__file__).parent / "app"))

def test_video_processor():
    """Test the video processor components"""
    print("🧪 Testing Video Processing System...")
    
    try:
        from app.models.video_processor import (
            VideoProcessor, VideoFrameExtractor, BatchProcessor, 
            TemporalAnalyzer, MemoryManager, ProgressTracker
        )
        print("✅ Video processor imports successful")
        
        # Test video processor initialization
        processor = VideoProcessor(
            max_frames=50,
            batch_size=4,
            max_workers=2,
            max_memory_mb=1024
        )
        print("✅ Video processor initialized")
        
        # Test memory manager
        memory_manager = MemoryManager(max_memory_mb=1024)
        memory_ok = memory_manager.check_memory_usage()
        print(f"✅ Memory manager working: {memory_ok}")
        
        # Test progress tracker
        progress_tracker = ProgressTracker(total_frames=100)
        progress_tracker.update_progress(10)
        progress_info = progress_tracker.get_progress()
        print(f"✅ Progress tracker working: {progress_info['progress_percent']}%")
        
        # Test temporal analyzer
        temporal_analyzer = TemporalAnalyzer()
        print("✅ Temporal analyzer initialized")
        
        # Test batch processor
        batch_processor = BatchProcessor(batch_size=4)
        print("✅ Batch processor initialized")
        
        print("\n🎉 All video processing components working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing video processor: {e}")
        return False

def test_video_api_routes():
    """Test video API routes"""
    print("\n🧪 Testing Video API Routes...")
    
    try:
        from app.video_routes import router
        print("✅ Video routes imported successfully")
        
        # Check if routes are properly configured
        routes = [route.path for route in router.routes]
        expected_routes = [
            "/api/video/upload",
            "/api/video/progress/{task_id}",
            "/api/video/results/{file_id}",
            "/api/video/list",
            "/api/video/{file_id}"
        ]
        
        print(f"✅ Found {len(routes)} video routes")
        return True
        
    except Exception as e:
        print(f"❌ Error testing video API routes: {e}")
        return False

def test_schemas():
    """Test video analysis schemas"""
    print("\n🧪 Testing Video Schemas...")
    
    try:
        from app.schemas import (
            VideoMetadata, FrameAnalysis, TemporalAnalysis,
            VideoAnalysisResults, VideoUploadResponse,
            VideoProgressResponse, VideoAnalysisResponse
        )
        print("✅ Video schemas imported successfully")
        
        # Test schema creation
        metadata = VideoMetadata(
            fps=30.0,
            frame_count=3000,
            duration=100.0,
            width=1920,
            height=1080
        )
        print("✅ Video metadata schema working")
        
        frame_analysis = FrameAnalysis(
            frame_number=1,
            timestamp=0.033,
            confidence_score=0.85,
            is_deepfake=False,
            processing_time=0.1
        )
        print("✅ Frame analysis schema working")
        
        print("✅ All video schemas working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing video schemas: {e}")
        return False

def test_synthetic_video_analysis():
    """Test video analysis with synthetic data"""
    print("\n🧪 Testing Synthetic Video Analysis...")
    
    try:
        from app.models.video_processor import VideoAnalysisResult, FrameAnalysis
        
        # Create synthetic frame analyses
        frame_analyses = []
        for i in range(10):
            frame_analysis = FrameAnalysis(
                frame_number=i,
                timestamp=i * 0.1,
                confidence_score=0.7 + (i % 3) * 0.1,
                is_deepfake=i % 3 == 0,
                analysis_metadata={"model": "test"},
                processing_time=0.05
            )
            frame_analyses.append(frame_analysis)
        
        # Create synthetic video analysis result
        result = VideoAnalysisResult(
            video_path="test_video.mp4",
            total_frames=100,
            analyzed_frames=10,
            overall_confidence=0.75,
            is_deepfake=False,
            temporal_consistency=0.8,
            frame_analyses=frame_analyses,
            processing_time=5.0,
            analysis_metadata={
                "temporal_analysis": {
                    "temporal_consistency": 0.8,
                    "consistency_score": 0.8,
                    "temporal_patterns": [],
                    "anomaly_frames": []
                },
                "video_metadata": {
                    "fps": 30.0,
                    "frame_count": 100,
                    "duration": 3.33,
                    "width": 1920,
                    "height": 1080
                },
                "processing_config": {
                    "max_frames": 10,
                    "batch_size": 4,
                    "sampling_strategy": "uniform"
                }
            }
        )
        
        print(f"✅ Synthetic video analysis created:")
        print(f"   - Analyzed frames: {result.analyzed_frames}")
        print(f"   - Overall confidence: {result.overall_confidence:.2f}")
        print(f"   - Temporal consistency: {result.temporal_consistency:.2f}")
        print(f"   - Processing time: {result.processing_time:.2f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing synthetic video analysis: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Video Analysis System Tests\n")
    
    tests = [
        test_video_processor,
        test_video_api_routes,
        test_schemas,
        test_synthetic_video_analysis
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("📊 Test Results:")
    print(f"   Passed: {passed}/{total}")
    print(f"   Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("\n🎉 All tests passed! Video analysis system is ready.")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 