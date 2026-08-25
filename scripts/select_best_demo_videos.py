#!/usr/bin/env python3
"""
Select best demo videos by reading directly from the database
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
    
    # Get recent video analysis results
    query = """
    SELECT 
        dr.id,
        dr.media_file_id,
        dr.confidence_score,
        dr.is_deepfake,
        dr.model_name,
        dr.processing_time,
        mf.filename,
        mf.file_path,
        mf.file_metadata
    FROM detection_results dr
    JOIN media_files mf ON dr.media_file_id = mf.id
    WHERE mf.file_type = 'video'
    ORDER BY dr.id DESC
    LIMIT 20
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
            'is_deepfake': row[3],
            'model_name': row[4],
            'processing_time': row[5],
            'filename': row[6],
            'file_path': row[7],
            'file_metadata': row[8]
        }
        videos.append(video)
    
    return videos

def find_best_demo_videos(video_results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Find the best 2 real and 2 fake videos for demo"""
    
    # Separate real and fake videos
    real_videos = [v for v in video_results if not v['is_deepfake']]
    fake_videos = [v for v in video_results if v['is_deepfake']]
    
    print(f"\n📊 Found {len(real_videos)} real videos and {len(fake_videos)} fake videos")
    
    # Sort by confidence (higher is better for real, lower is better for fake)
    real_videos.sort(key=lambda x: x['confidence_score'], reverse=True)
    fake_videos.sort(key=lambda x: x['confidence_score'], reverse=True)  # Higher confidence = more clearly fake
    
    # Select top 2 from each category
    best_real = real_videos[:2] if len(real_videos) >= 2 else real_videos
    best_fake = fake_videos[:2] if len(fake_videos) >= 2 else fake_videos
    
    return {
        'real': best_real,
        'fake': best_fake
    }

def main():
    """Main function to select best demo videos"""
    print("🎬 Selecting Best Demo Videos from Database")
    print("=" * 60)
    
    # Get video analysis results
    video_results = get_video_analysis_results()
    
    if not video_results:
        print("❌ No video analysis results found in database")
        return
    
    print(f"📊 Found {len(video_results)} video analysis results")
    
    # Display all results
    print(f"\n📋 ALL VIDEO RESULTS:")
    print("-" * 60)
    for video in video_results:
        status = "🟢 FAKE" if video['is_deepfake'] else "🔴 REAL"
        print(f"{status} | {video['filename']} | Confidence: {video['confidence_score']:.1%} | Model: {video['model_name']}")
    
    # Find best demo videos
    best_videos = find_best_demo_videos(video_results)
    
    # Display selected demo videos
    print(f"\n🏆 BEST DEMO VIDEOS SELECTED:")
    print("=" * 60)
    
    print(f"\n🔴 REAL VIDEOS ({len(best_videos['real'])}):")
    for i, video in enumerate(best_videos['real'], 1):
        print(f"  {i}. {video['filename']}")
        print(f"     Confidence: {video['confidence_score']:.1%}")
        print(f"     Model: {video['model_name']}")
        print(f"     Path: {video['file_path']}")
    
    print(f"\n🟢 FAKE VIDEOS ({len(best_videos['fake'])}):")
    for i, video in enumerate(best_videos['fake'], 1):
        print(f"  {i}. {video['filename']}")
        print(f"     Confidence: {video['confidence_score']:.1%}")
        print(f"     Model: {video['model_name']}")
        print(f"     Path: {video['file_path']}")
    
    # Save results
    demo_results = {
        'timestamp': '2024-12-19',
        'total_videos': len(video_results),
        'best_demo_videos': best_videos,
        'all_videos': video_results
    }
    
    with open('best_demo_videos.json', 'w') as f:
        json.dump(demo_results, f, indent=2)
    
    print(f"\n💾 Results saved to best_demo_videos.json")
    
    # Create demo directory and copy videos
    demo_dir = Path("demo_videos_final")
    demo_dir.mkdir(exist_ok=True)
    
    print(f"\n📁 Copying demo videos to demo_videos_final/ directory...")
    
    for category, videos in best_videos.items():
        for i, video in enumerate(videos, 1):
            src_path = Path(video['file_path'])
            if src_path.exists():
                dst_path = demo_dir / f"{category}_{i:02d}_{video['filename']}"
                
                try:
                    shutil.copy2(src_path, dst_path)
                    print(f"  ✅ Copied {src_path.name} → {dst_path.name}")
                except Exception as e:
                    print(f"  ❌ Failed to copy {src_path.name}: {e}")
            else:
                print(f"  ⚠️  Source file not found: {src_path}")
    
    print(f"\n🎉 Final demo video selection complete!")
    print(f"   Demo videos are in: demo_videos_final/")
    print(f"   Results saved to: best_demo_videos.json")

if __name__ == "__main__":
    main()





