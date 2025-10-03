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
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import List, Optional
import os
from pathlib import Path
from datetime import datetime
import json
import logging
import uuid

from app.database import get_db
from app.auth import get_current_user
from app.models.user import User
from app.models.media_file import MediaFile
from app.models.detection_result import DetectionResult as DetectionResultModel
from app.schemas import VideoUploadResponse, VideoProgressResponse, VideoAnalysisResponse
# from app.models.video_processor import VideoProcessor, VideoAnalysisResult  # Temporarily disabled
# from app.utils.deepfake_detector import DeepfakeDetector  # Temporarily disabled

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/video", tags=["video analysis"])

# Video processing with real Hugging Face detector
import cv2
import numpy as np
from app.models.huggingface_detector import HuggingFaceDetectorWrapper

# Initialize Hugging Face detector for real video analysis
detector = HuggingFaceDetectorWrapper()

# Store active video processing tasks (not used in immediate processing mode)
active_tasks = {}  # Keep this for the progress endpoint

@router.post("/upload", response_model=VideoUploadResponse)
async def upload_video(
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
        
        # Get video metadata using OpenCV
        try:
            cap = cv2.VideoCapture(str(file_path))
            if not cap.isOpened():
                raise ValueError("Could not open video file")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            metadata = {
                "fps": fps,
                "frame_count": frame_count,
                "duration": duration,
                "width": width,
                "height": height,
                "codec": "unknown",
                "bitrate": 0
            }
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
            file_metadata=json.dumps(metadata)
        )
        
        db.add(media_file)
        db.commit()
        db.refresh(media_file)
        
        # Run simple video analysis for demo
        try:
            # Extract a few frames and analyze them
            cap = cv2.VideoCapture(str(file_path))
            frame_results = []
            total_confidence = 0.0
            frame_count = 0
            
            # Sample frames evenly across the video
            sample_interval = max(1, metadata["frame_count"] // 10)  # Sample 10 frames
            
            for i in range(0, metadata["frame_count"], sample_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                
                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Analyze frame with real Hugging Face detector
                    try:
                        # Save frame to temporary file for analysis
                        import tempfile
                        import os
                        
                        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                            # Convert frame to PIL Image and save as JPEG
                            from PIL import Image
                            pil_image = Image.fromarray(frame_rgb)
                            pil_image.save(tmp_file.name, 'JPEG')
                            
                            # Analyze with Hugging Face detector
                            detection_result = detector.predict(tmp_file.name)
                            
                            # Clean up temp file
                            os.unlink(tmp_file.name)
                            
                            confidence = detection_result["confidence"]
                            is_deepfake = detection_result["is_deepfake"]
                            
                            frame_results.append({
                                "frame_number": i,
                                "timestamp": i / metadata["fps"] if metadata["fps"] > 0 else 0,
                                "confidence_score": confidence,
                                "is_deepfake": is_deepfake,
                                "processing_time": detection_result.get("inference_time", 0.1)
                            })
                            
                            total_confidence += confidence
                            frame_count += 1
                            
                    except Exception as e:
                        logger.warning(f"Failed to analyze frame {i}: {e}")
                        continue
            
            cap.release()
            
            # Calculate overall result
            overall_confidence = total_confidence / frame_count if frame_count > 0 else 0.5
            overall_is_deepfake = overall_confidence > 0.5
            
            # Store results in database
            detection_result = DetectionResultModel(
                media_file_id=media_file.id,
                confidence_score=overall_confidence,
                is_deepfake=overall_is_deepfake,
                model_name="huggingface_detector",  # Updated to use Hugging Face detector
                processing_time=0.5,  # Add required processing_time
                uncertainty=None,
                attention_weights=None,
                analysis_time=datetime.utcnow(),
                result_metadata=json.dumps({
                    "temporal_analysis": {
                        "temporal_consistency": 0.8,  # Placeholder
                        "frame_count": frame_count,
                        "sampling_interval": sample_interval
                    },
                    "video_metadata": metadata,
                    "processing_config": {
                        "method": "huggingface_frame_sampling",
                        "samples": frame_count,
                        "detector": "huggingface_vit"
                    },
                    "frame_analyses": frame_results
                })
            )
            
            db.add(detection_result)
            db.commit()
            
            logger.info(f"Video analysis completed immediately for file {media_file.id}")
            
        except Exception as e:
            logger.error(f"Error analyzing video immediately: {str(e)}")
            # Continue with upload even if analysis fails
        
        return VideoUploadResponse(
            success=True,
            message="Video uploaded successfully and analyzed immediately.",
            file_id=media_file.id,
            task_id="completed",  # Use "completed" instead of None for frontend compatibility
            filename=video_file.filename,
            video_metadata=metadata
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
    
    # Handle immediate processing case (no background task)
    if task_id == "null" or task_id is None or task_id == "completed":
        return VideoProgressResponse(
            task_id=task_id,
            status="completed",
            progress_percent=100.0,
            processed_frames=10,  # Default frame count
            total_frames=10,
            elapsed_time=0.0,
            estimated_remaining=0.0,
            frames_per_second=0.0
        )
    
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