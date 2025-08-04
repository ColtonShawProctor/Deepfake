#!/usr/bin/env python3
"""
Video Analysis API Routes

Provides endpoints for:
- Video upload and processing
- Progress tracking
- Video analysis results
- Batch video processing
"""

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, BackgroundTasks
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import List, Optional
import json
import os
import uuid
from pathlib import Path
import logging
from datetime import datetime

from app.database import get_db
from app.auth import get_current_user
from app.models.user import User
from app.models.media_file import MediaFile
from app.models.detection_result import DetectionResult as DetectionResultModel
from app.schemas import VideoAnalysisResponse, VideoProgressResponse, VideoUploadResponse
from app.models.video_processor import VideoProcessor, VideoAnalysisResult
from app.utils.deepfake_detector import DeepfakeDetector

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/video", tags=["video analysis"])

# Initialize video processor and detector
video_processor = VideoProcessor(
    max_frames=100,
    batch_size=8,
    max_workers=4,
    max_memory_mb=2048
)
detector = DeepfakeDetector()

# Store active video processing tasks
active_tasks = {}

@router.post("/upload", response_model=VideoUploadResponse)
async def upload_video(
    background_tasks: BackgroundTasks,
    video_file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Upload a video file for deepfake analysis
    
    - **video_file**: Video file to upload (MP4, AVI, MOV supported)
    - **Authentication required**
    """
    
    # Validate file type
    allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
    file_extension = Path(video_file.filename).suffix.lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Validate file size (max 500MB)
    max_size = 500 * 1024 * 1024  # 500MB
    if video_file.size and video_file.size > max_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File too large. Maximum size: 500MB"
        )
    
    try:
        # Create uploads directory if it doesn't exist
        uploads_dir = Path("uploads/videos")
        uploads_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename
        file_id = str(uuid.uuid4())
        filename = f"{file_id}{file_extension}"
        file_path = uploads_dir / filename
        
        # Save video file
        with open(file_path, "wb") as buffer:
            content = await video_file.read()
            buffer.write(content)
        
        # Get video metadata
        try:
            metadata = video_processor.get_video_metadata(str(file_path))
        except Exception as e:
            # Clean up file if metadata extraction fails
            os.remove(file_path)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid video file: {str(e)}"
            )
        
        # Save to database
        media_file = MediaFile(
            user_id=current_user.id,
            filename=video_file.filename,
            file_path=str(file_path),
            file_type="video",
            file_size=len(content),
            metadata=json.dumps({
                "fps": metadata.fps,
                "frame_count": metadata.frame_count,
                "duration": metadata.duration,
                "width": metadata.width,
                "height": metadata.height,
                "codec": metadata.codec,
                "bitrate": metadata.bitrate
            })
        )
        
        db.add(media_file)
        db.commit()
        db.refresh(media_file)
        
        # Start background processing
        task_id = str(uuid.uuid4())
        active_tasks[task_id] = {
            "status": "processing",
            "progress": 0,
            "media_file_id": media_file.id,
            "start_time": datetime.utcnow()
        }
        
        background_tasks.add_task(
            process_video_background,
            task_id,
            media_file.id,
            str(file_path),
            current_user.id,
            db
        )
        
        return VideoUploadResponse(
            success=True,
            message="Video uploaded successfully. Processing started.",
            file_id=media_file.id,
            task_id=task_id,
            filename=video_file.filename,
            video_metadata={
                "fps": metadata.fps,
                "frame_count": metadata.frame_count,
                "duration": metadata.duration,
                "width": metadata.width,
                "height": metadata.height
            }
        )
        
    except Exception as e:
        logger.error(f"Error uploading video: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {str(e)}"
        )

@router.get("/progress/{task_id}", response_model=VideoProgressResponse)
async def get_video_progress(
    task_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get progress of video analysis
    
    - **task_id**: Task ID returned from upload
    - **Authentication required**
    """
    
    if task_id not in active_tasks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found"
        )
    
    task = active_tasks[task_id]
    
    return VideoProgressResponse(
        task_id=task_id,
        status=task["status"],
        progress_percent=task.get("progress", 0),
        processed_frames=task.get("processed_frames", 0),
        total_frames=task.get("total_frames", 0),
        elapsed_time=task.get("elapsed_time", 0),
        estimated_remaining=task.get("estimated_remaining", 0),
        frames_per_second=task.get("frames_per_second", 0)
    )

@router.get("/results/{file_id}", response_model=VideoAnalysisResponse)
async def get_video_results(
    file_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get video analysis results
    
    - **file_id**: ID of the analyzed video file
    - **Authentication required**
    """
    
    # Get the media file and verify ownership
    media_file = db.query(MediaFile).filter(
        MediaFile.id == file_id,
        MediaFile.user_id == current_user.id,
        MediaFile.file_type == "video"
    ).first()
    
    if not media_file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video file not found or access denied"
        )
    
    # Get detection results
    detection_results = db.query(DetectionResultModel).filter(
        DetectionResultModel.media_file_id == file_id
    ).all()
    
    if not detection_results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis results not found"
        )
    
    # Parse analysis metadata
    analysis_metadata = {}
    if detection_results:
        try:
            analysis_metadata = json.loads(detection_results[0].result_metadata)
        except:
            analysis_metadata = {}
    
    return VideoAnalysisResponse(
        success=True,
        file_id=media_file.id,
        filename=media_file.filename,
        video_metadata=json.loads(media_file.metadata) if media_file.metadata else {},
        analysis_results={
            "overall_confidence": detection_results[0].confidence_score,
            "is_deepfake": detection_results[0].is_deepfake,
            "temporal_consistency": analysis_metadata.get("temporal_analysis", {}).get("temporal_consistency", 0.0),
            "frame_analyses": analysis_metadata.get("frame_analyses", []),
            "temporal_analysis": analysis_metadata.get("temporal_analysis", {}),
            "processing_config": analysis_metadata.get("processing_config", {})
        },
        created_at=detection_results[0].analysis_time
    )

@router.get("/list", response_model=List[VideoAnalysisResponse])
async def list_user_videos(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    List all video files uploaded by the user
    
    - **Authentication required**
    """
    
    media_files = db.query(MediaFile).filter(
        MediaFile.user_id == current_user.id,
        MediaFile.file_type == "video"
    ).all()
    
    results = []
    for media_file in media_files:
        detection_results = db.query(DetectionResultModel).filter(
            DetectionResultModel.media_file_id == media_file.id
        ).all()
        
        if detection_results:
            analysis_metadata = {}
            try:
                analysis_metadata = json.loads(detection_results[0].result_metadata)
            except:
                analysis_metadata = {}
            
            results.append(VideoAnalysisResponse(
                success=True,
                file_id=media_file.id,
                filename=media_file.filename,
                video_metadata=json.loads(media_file.metadata) if media_file.metadata else {},
                analysis_results={
                    "overall_confidence": detection_results[0].confidence_score,
                    "is_deepfake": detection_results[0].is_deepfake,
                    "temporal_consistency": analysis_metadata.get("temporal_analysis", {}).get("temporal_consistency", 0.0),
                    "frame_analyses": analysis_metadata.get("frame_analyses", []),
                    "temporal_analysis": analysis_metadata.get("temporal_analysis", {}),
                    "processing_config": analysis_metadata.get("processing_config", {})
                },
                created_at=detection_results[0].analysis_time
            ))
    
    return results

@router.delete("/{file_id}")
async def delete_video(
    file_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Delete a video file and its analysis results
    
    - **file_id**: ID of the video file to delete
    - **Authentication required**
    """
    
    # Get the media file and verify ownership
    media_file = db.query(MediaFile).filter(
        MediaFile.id == file_id,
        MediaFile.user_id == current_user.id,
        MediaFile.file_type == "video"
    ).first()
    
    if not media_file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video file not found or access denied"
        )
    
    try:
        # Delete file from disk
        if os.path.exists(media_file.file_path):
            os.remove(media_file.file_path)
        
        # Delete detection results
        db.query(DetectionResultModel).filter(
            DetectionResultModel.media_file_id == file_id
        ).delete()
        
        # Delete media file record
        db.delete(media_file)
        db.commit()
        
        return {"success": True, "message": "Video file deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting video: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Delete failed: {str(e)}"
        )

async def process_video_background(task_id: str, media_file_id: int, file_path: str, user_id: int, db: Session):
    """Background task for video processing"""
    
    try:
        # Update task status
        active_tasks[task_id]["status"] = "processing"
        
        # Progress callback function
        def progress_callback(progress):
            active_tasks[task_id].update(progress)
        
        # Analyze video
        result = video_processor.analyze_video(file_path, detector, progress_callback)
        
        # Store results in database
        detection_result = DetectionResultModel(
            media_file_id=media_file_id,
            confidence_score=result.overall_confidence,
            is_deepfake=result.is_deepfake,
            analysis_time=datetime.utcnow(),
            result_metadata=json.dumps({
                "temporal_analysis": result.analysis_metadata["temporal_analysis"],
                "video_metadata": result.analysis_metadata["video_metadata"],
                "processing_config": result.analysis_metadata["processing_config"],
                "frame_analyses": [
                    {
                        "frame_number": fa.frame_number,
                        "timestamp": fa.timestamp,
                        "confidence_score": fa.confidence_score,
                        "is_deepfake": fa.is_deepfake,
                        "processing_time": fa.processing_time
                    }
                    for fa in result.frame_analyses
                ]
            })
        )
        
        db.add(detection_result)
        db.commit()
        
        # Update task status
        active_tasks[task_id]["status"] = "completed"
        active_tasks[task_id]["progress"] = 100
        
        logger.info(f"Video analysis completed for task {task_id}")
        
    except Exception as e:
        logger.error(f"Error processing video for task {task_id}: {str(e)}")
        active_tasks[task_id]["status"] = "failed"
        active_tasks[task_id]["error"] = str(e) 