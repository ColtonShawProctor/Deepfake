#!/usr/bin/env python3
"""
Find the best demo videos that show clear real vs fake classification
"""

import sqlite3
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any

def get_video_analysis_results() -> List[Dict[str, Any]]:
    """Get video analysis results from database"""
    conn = sqlite3.connect('deepfake.db')
    cursor = conn.cursor()
    
    # Get all video analysis results with Hugging Face detector
    query = """
    SELECT 
        dr.id,
        dr.media_file_id,
        dr.confidence_score,
        dr.is_deepfake,
        dr.model_name,
        mf.filename,
        mf.file_path
    FROM detection_results dr
    JOIN media_files mf ON dr.media_file_id = mf.id
    WHERE mf.file_type = 'video' 
    AND dr.model_name = 'huggingface_detector'
    ORDER BY dr.confidence_score DESC
    """
    
    cursor.execute(query)
    results = cursor.fetchall()
    conn.close()
    
    videos = []
    for row in results:
        video = {
            'id': row[0],
            'media_file_id': row[1],
            'confidence_score': row[2],
            'confidence_percent': row[2] * 100.0,
            'is_deepfake': row[3],
            'model_name': row[4],
            'filename': row[5],
            'file_path': row[6]
        }
        videos.append(video)
    
    return videos

def analyze_classification_patterns(video_results: List[Dict[str, Any]]):
    """Analyze the classification patterns"""
    print("🔍 CLASSIFICATION PATTERN ANALYSIS:")
    print("=" * 60)
    
    # Group by classification
    fake_videos = [v for v in video_results if v['is_deepfake']]
    real_videos = [v for v in video_results if not v['is_deepfake']]
    
    print(f"📊 Total videos analyzed: {len(video_results)}")
    print(f"🟢 Classified as FAKE: {len(fake_videos)}")
    print(f"🔴 Classified as REAL: {len(real_videos)}")
    
    if fake_videos:
        print(f"\n🟢 FAKE VIDEOS - Confidence Range:")
        fake_confidences = [v['confidence_percent'] for v in fake_videos]
        print(f"   Highest: {max(fake_confidences):.1f}%")
        print(f"   Lowest: {min(fake_confidences):.1f}%")
        print(f"   Average: {sum(fake_confidences)/len(fake_confidences):.1f}%")
    
    if real_videos:
        print(f"\n🔴 REAL VIDEOS - Confidence Range:")
        real_confidences = [v['confidence_percent'] for v in real_videos]
        print(f"   Highest: {max(real_confidences):.1f}%")
        print(f"   Lowest: {min(real_confidences):.1f}%")
        print(f"   Average: {sum(real_confidences)/len(real_confidences):.1f}%")
    
    # Find the confidence threshold that separates real from fake
    if fake_videos and real_videos:
        min_fake_confidence = min([v['confidence_percent'] for v in fake_videos])
        max_real_confidence = max([v['confidence_percent'] for v in real_videos])
        
        if min_fake_confidence > max_real_confidence:
            print(f"\n✅ CLEAR SEPARATION: Fake videos start at {min_fake_confidence:.1f}%, Real videos max at {max_real_confidence:.1f}%")
            print(f"   🎯 Optimal threshold: {(min_fake_confidence + max_real_confidence) / 2:.1f}%")
        else:
            print(f"\n⚠️  OVERLAP: Some fake videos ({min_fake_confidence:.1f}%) have lower confidence than some real videos ({max_real_confidence:.1f}%)")

def find_best_demo_videos(video_results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Find the best videos for demo that show clear classification"""
    
    # Separate by classification
    fake_videos = [v for v in video_results if v['is_deepfake']]
    real_videos = [v for v in video_results if not v['is_deepfake']]
    
    print(f"\n🎯 SELECTING BEST DEMO VIDEOS:")
    print("=" * 60)
    
    # For fake videos: higher confidence = more clearly fake
    fake_videos.sort(key=lambda x: x['confidence_percent'], reverse=True)
    
    # For real videos: lower confidence = more clearly real (in this case)
    real_videos.sort(key=lambda x: x['confidence_percent'])
    
    # Select top examples from each category
    best_fake = fake_videos[:3] if len(fake_videos) >= 3 else fake_videos
    best_real = real_videos[:3] if len(real_videos) >= 3 else real_videos
    
    return {
        'fake': best_fake,
        'real': best_real
    }

def create_demo_directory(best_videos: Dict[str, List[Dict[str, Any]]]):
    """Create demo directory and copy selected videos"""
    demo_dir = Path("best_demo_videos")
    demo_dir.mkdir(exist_ok=True)
    
    print(f"\n📁 Creating demo directory: {demo_dir}")
    
    for category, videos in best_videos.items():
        print(f"\n{category.upper()} VIDEOS:")
        for i, video in enumerate(videos, 1):
            src_path = Path(video['file_path'])
            if src_path.exists():
                # Create descriptive filename
                confidence_str = f"{video['confidence_percent']:.0f}percent"
                dst_filename = f"{category}_{i:02d}_{confidence_str}_{video['filename']}"
                dst_path = demo_dir / dst_filename
                
                try:
                    shutil.copy2(src_path, dst_path)
                    print(f"  ✅ {src_path.name} → {dst_filename}")
                except Exception as e:
                    print(f"  ❌ Failed to copy {src_path.name}: {e}")
            else:
                print(f"  ⚠️  Source file not found: {src_path}")

def main():
    """Main function"""
    print("🎬 Finding Best Demo Videos for Hugging Face Model")
    print("=" * 60)
    
    # Get video analysis results
    video_results = get_video_analysis_results()
    
    if not video_results:
        print("❌ No video analysis results found")
        return
    
    # Display all results
    print(f"\n📋 ALL VIDEO RESULTS (Hugging Face Model):")
    print("-" * 60)
    for video in video_results:
        status = "🟢 FAKE" if video['is_deepfake'] else "🔴 REAL"
        print(f"{status} | {video['filename']} | Confidence: {video['confidence_percent']:.1f}%")
    
    # Analyze patterns
    analyze_classification_patterns(video_results)
    
    # Find best demo videos
    best_videos = find_best_demo_videos(video_results)
    
    # Display selected videos
    print(f"\n🏆 BEST DEMO VIDEOS SELECTED:")
    print("=" * 60)
    
    print(f"\n🟢 FAKE VIDEOS ({len(best_videos['fake'])}):")
    for i, video in enumerate(best_videos['fake'], 1):
        print(f"  {i}. {video['filename']}")
        print(f"     Confidence: {video['confidence_percent']:.1f}%")
        print(f"     Path: {video['file_path']}")
    
    print(f"\n🔴 REAL VIDEOS ({len(best_videos['real'])}):")
    for i, video in enumerate(best_videos['real'], 1):
        print(f"  {i}. {video['filename']}")
        print(f"     Confidence: {video['confidence_percent']:.1f}%")
        print(f"     Path: {video['file_path']}")
    
    # Create demo directory
    create_demo_directory(best_videos)
    
    # Save results
    demo_results = {
        'timestamp': '2024-12-19',
        'total_videos': len(video_results),
        'best_demo_videos': best_videos,
        'all_videos': video_results
    }
    
    with open('best_demo_videos_analysis.json', 'w') as f:
        json.dump(demo_results, f, indent=2)
    
    print(f"\n💾 Analysis saved to best_demo_videos_analysis.json")
    print(f"🎉 Demo setup complete! Check the 'best_demo_videos' directory")

if __name__ == "__main__":
    main()






