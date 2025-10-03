#!/usr/bin/env python3
"""
Emergency Detection Routes - Fast Recovery Solution

These routes provide immediate deepfake detection functionality using
the emergency detector while the main system is being repaired.
"""

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.orm import Session
from typing import List, Optional
import json
import os
from pathlib import Path
from datetime import datetime

from app.database import get_db
from app.auth import get_current_user
from app.models.user import User
from app.models.media_file import MediaFile
from app.models.detection_result import DetectionResult as DetectionResultModel
from app.schemas import DetectionResponse, DetectorInfo, DetectionResult
from app.models.emergency_detector import emergency_detector

router = APIRouter(prefix="/api/emergency", tags=["emergency deepfake detection"])

@router.get("/info", response_model=DetectorInfo)
async def get_emergency_detector_info():
    """
    Get information about the emergency deepfake detector
    """
    info = emergency_detector.get_detector_info()
    return DetectorInfo(
        name=info["name"],
        status=info["status"],
        model=info["model"],
        accuracy=info["accuracy"],
        capabilities=info["capabilities"]
    )

@router.post("/analyze/image")
async def analyze_image_emergency(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Analyze an uploaded image for deepfake detection using emergency detector
    
    - **file**: Image file to analyze
    - **Authentication required**
    """
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image"
        )
    
    try:
        # Save uploaded file temporarily
        upload_dir = Path("uploads/images")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"emergency_{current_user.id}_{timestamp}_{file.filename}"
        file_path = upload_dir / filename
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Store file record in database
        media_file = MediaFile(
            user_id=current_user.id,
            filename=file.filename,
            file_path=str(file_path),
            file_type="image",
            file_size=len(content),
            upload_time=datetime.utcnow()
        )
        db.add(media_file)
        db.commit()
        db.refresh(media_file)
        
        # Perform deepfake detection
        detection_result = emergency_detector.predict(str(file_path))
        
        # Store result in database
        db_detection_result = DetectionResultModel(
            media_file_id=media_file.id,
            confidence_score=detection_result["confidence"],
            is_deepfake=detection_result["is_deepfake"],
            model_name=detection_result.get("model", "emergency_detector"),
            processing_time=detection_result.get("inference_time", 0.0),
            uncertainty=None,
            attention_weights=None,
            analysis_time=datetime.utcnow(),
            result_metadata=json.dumps({
                "method": detection_result.get("method", "emergency"),
                "device": detection_result.get("device", "cpu"),
                "predicted_class": detection_result.get("predicted_class", 0),
                "probabilities": detection_result.get("probabilities", [])
            })
        )
        
        db.add(db_detection_result)
        db.commit()
        
        # Return response
        return DetectionResponse(
            file_id=media_file.id,
            filename=file.filename,
            detection_result=DetectionResult(
                confidence_score=detection_result["confidence"],
                is_deepfake=detection_result["is_deepfake"],
                analysis_metadata={
                    "method": detection_result.get("method", "emergency"),
                    "model": detection_result.get("model", "emergency_detector"),
                    "device": detection_result.get("device", "cpu"),
                    "processing_time": detection_result.get("inference_time", 0.0)
                },
                analysis_time=datetime.utcnow().isoformat(),
                processing_time_seconds=detection_result.get("inference_time", 0.0),
                error=detection_result.get("error")
            ),
            message="Image analyzed successfully using emergency detector"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )

@router.post("/analyze/video")
async def analyze_video_emergency(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Analyze an uploaded video for deepfake detection using emergency detector
    
    - **file**: Video file to analyze
    - **Authentication required**
    """
    # Validate file type
    if not file.content_type.startswith('video/'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be a video"
        )
    
    try:
        # Save uploaded file temporarily
        upload_dir = Path("uploads/videos")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"emergency_{current_user.id}_{timestamp}_{file.filename}"
        file_path = upload_dir / filename
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Store file record in database
        media_file = MediaFile(
            user_id=current_user.id,
            filename=file.filename,
            file_path=str(file_path),
            file_type="video",
            file_size=len(content),
            upload_time=datetime.utcnow()
        )
        db.add(media_file)
        db.commit()
        db.refresh(media_file)
        
        # Perform deepfake detection on video
        detection_result = emergency_detector.predict_video(str(file_path))
        
        # Store result in database
        db_detection_result = DetectionResultModel(
            media_file_id=media_file.id,
            confidence_score=detection_result["confidence"],
            is_deepfake=detection_result["is_deepfake"],
            model_name=detection_result.get("model", "emergency_detector"),
            processing_time=detection_result.get("inference_time", 0.0),
            uncertainty=None,
            attention_weights=None,
            analysis_time=datetime.utcnow(),
            result_metadata=json.dumps({
                "method": detection_result.get("method", "emergency_video"),
                "device": detection_result.get("device", "cpu"),
                "total_frames": detection_result.get("total_frames", 0),
                "fake_votes": detection_result.get("fake_votes", 0),
                "real_votes": detection_result.get("real_votes", 0),
                "frame_results": detection_result.get("frame_results", [])
            })
        )
        
        db.add(db_detection_result)
        db.commit()
        
        # Return response
        return DetectionResponse(
            file_id=media_file.id,
            filename=file.filename,
            detection_result=DetectionResult(
                confidence_score=detection_result["confidence"],
                is_deepfake=detection_result["is_deepfake"],
                analysis_metadata={
                    "method": detection_result.get("method", "emergency_video"),
                    "model": detection_result.get("model", "emergency_detector"),
                    "device": detection_result.get("device", "cpu"),
                    "total_frames": detection_result.get("total_frames", 0),
                    "frame_analysis": detection_result.get("frame_results", []),
                    "processing_time": detection_result.get("inference_time", 0.0)
                },
                analysis_time=datetime.utcnow().isoformat(),
                processing_time_seconds=detection_result.get("inference_time", 0.0),
                error=detection_result.get("error")
            ),
            message="Video analyzed successfully using emergency detector"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )

@router.post("/analyze/{file_id}")
async def analyze_existing_file_emergency(
    file_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Analyze a previously uploaded file using emergency detector
    
    - **file_id**: ID of the uploaded file to analyze
    - **Authentication required**
    """
    # Get the media file and verify ownership
    media_file = db.query(MediaFile).filter(
        MediaFile.id == file_id,
        MediaFile.user_id == current_user.id
    ).first()
    
    if not media_file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found or access denied"
        )
    
    # Check if file exists on disk
    file_path = Path(media_file.file_path)
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found on disk"
        )
    
    try:
        # Perform detection based on file type
        if media_file.file_type == "video":
            detection_result = emergency_detector.predict_video(str(file_path))
        else:
            detection_result = emergency_detector.predict(str(file_path))
        
        # Store result in database
        db_detection_result = DetectionResultModel(
            media_file_id=media_file.id,
            confidence_score=detection_result["confidence"],
            is_deepfake=detection_result["is_deepfake"],
            model_name=detection_result.get("model", "emergency_detector"),
            processing_time=detection_result.get("inference_time", 0.0),
            uncertainty=None,
            attention_weights=None,
            analysis_time=datetime.utcnow(),
            result_metadata=json.dumps({
                "method": detection_result.get("method", "emergency"),
                "device": detection_result.get("device", "cpu"),
                "file_type": media_file.file_type,
                "processing_time": detection_result.get("inference_time", 0.0)
            })
        )
        
        db.add(db_detection_result)
        db.commit()
        
        # Return response
        return DetectionResponse(
            file_id=media_file.id,
            filename=media_file.filename,
            detection_result=DetectionResult(
                confidence_score=detection_result["confidence"],
                is_deepfake=detection_result["is_deepfake"],
                analysis_metadata={
                    "method": detection_result.get("method", "emergency"),
                    "model": detection_result.get("model", "emergency_detector"),
                    "device": detection_result.get("device", "cpu"),
                    "file_type": media_file.file_type,
                    "processing_time": detection_result.get("inference_time", 0.0)
                },
                analysis_time=datetime.utcnow().isoformat(),
                processing_time_seconds=detection_result.get("inference_time", 0.0),
                error=detection_result.get("error")
            ),
            message=f"{media_file.file_type.title()} analyzed successfully using emergency detector"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )

@router.get("/status")
async def get_emergency_status():
    """
    Get the current status of the emergency detection system
    """
    info = emergency_detector.get_detector_info()
    
    return {
        "status": "operational",
        "detector": info,
        "message": "Emergency deepfake detection system is operational",
        "timestamp": datetime.utcnow().isoformat(),
        "capabilities": {
            "image_detection": True,
            "video_analysis": True,
            "real_time": False,
            "batch_processing": False
        }
    }





