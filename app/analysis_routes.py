"""
Analysis endpoints for deepfake detection.

This module provides:
- POST /analyze/{file_id} - Trigger deepfake analysis for uploaded file
- GET /results/{file_id} - Get analysis results for specific file
- GET /history - Get user's analysis history
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import desc, func

from app.database import get_db
from app.auth import get_current_user
from app.models.user import User
from app.models.media_file import MediaFile
from app.models.detection_result import DetectionResult as DetectionResultModel
from app.schemas import DetectionResponse
from app.utils.deepfake_detector import DeepfakeDetector

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/analysis", tags=["analysis"])

# Initialize detector
detector = DeepfakeDetector()

@router.post("/analyze/{file_id}", response_model=Dict[str, Any])
async def analyze_file(
    file_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Trigger deepfake analysis for an uploaded file.
    
    - **file_id**: ID of the uploaded file to analyze
    - **Authentication required**
    - **File ownership verification required**
    
    Returns:
        Analysis result and metadata
    """
    # Check if file exists and user owns it
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
    
    # Check if analysis already completed
    existing_result = db.query(DetectionResultModel).filter(
        DetectionResultModel.media_file_id == file_id
    ).order_by(desc(DetectionResultModel.analysis_time)).first()
    
    if existing_result:
        return {
            "message": "Analysis already completed",
            "file_id": file_id,
            "status": "completed",
            "result_id": existing_result.id,
            "confidence_score": existing_result.confidence_score * 100.0,
            "is_deepfake": existing_result.is_deepfake,
            "completed_at": existing_result.analysis_time.isoformat()
        }
    
    try:
        # Run analysis synchronously
        logger.info(f"Starting analysis for file {file_id}")
        
        # Run analysis
        detection_result = detector.analyze_image(str(file_path))
        
        # Store result in database
        analysis_time = datetime.fromisoformat(detection_result["analysis_time"].replace('Z', '+00:00'))
        
        db_detection_result = DetectionResultModel(
            media_file_id=file_id,
            confidence_score=detection_result["confidence_score"] / 100.0,  # Convert to 0-1 scale
            is_deepfake=detection_result["is_deepfake"],
            analysis_time=analysis_time,
            result_metadata=json.dumps(detection_result["analysis_metadata"])
        )
        
        db.add(db_detection_result)
        db.commit()
        db.refresh(db_detection_result)
        
        logger.info(f"Analysis completed for file {file_id} with confidence {detection_result['confidence_score']}%")
        
        return {
            "message": "Analysis completed successfully",
            "file_id": file_id,
            "filename": media_file.filename,
            "status": "completed",
            "confidence_score": detection_result["confidence_score"],
            "is_deepfake": detection_result["is_deepfake"],
            "processing_time_seconds": detection_result["processing_time_seconds"],
            "result_id": db_detection_result.id,
            "completed_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Analysis failed for file {file_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )

@router.get("/results/{file_id}", response_model=DetectionResponse)
async def get_analysis_results(
    file_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get analysis results for a specific file.
    
    - **file_id**: ID of the file to get results for
    - **Authentication required**
    """
    # Verify file ownership
    media_file = db.query(MediaFile).filter(
        MediaFile.id == file_id,
        MediaFile.user_id == current_user.id
    ).first()
    
    if not media_file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found or access denied"
        )
    
    # Get the latest analysis result
    detection_result = db.query(DetectionResultModel).filter(
        DetectionResultModel.media_file_id == file_id
    ).order_by(desc(DetectionResultModel.analysis_time)).first()
    
    if not detection_result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No analysis results found for this file. Start an analysis first."
        )
    
    # Convert database result to API format
    try:
        metadata = json.loads(detection_result.result_metadata)
    except json.JSONDecodeError:
        metadata = {}
    
    api_detection_result = {
        "confidence_score": detection_result.confidence_score * 100.0,
        "is_deepfake": detection_result.is_deepfake,
        "analysis_metadata": metadata,
        "analysis_time": detection_result.analysis_time.isoformat(),
        "processing_time_seconds": 0.0,  # Not stored in DB
        "error": None
    }
    
    # Construct file URL for frontend (using public endpoint for image display)
    file_url = f"/api/files/public/{media_file.id}"
    
    return DetectionResponse(
        success=True,
        message="Analysis results retrieved successfully",
        file_id=media_file.id,
        filename=media_file.filename,
        file_url=file_url,
        file_size=media_file.file_size,
        file_type=media_file.file_type,
        detection_result=api_detection_result,
        created_at=detection_result.analysis_time
    )

@router.get("/history", response_model=List[DetectionResponse])
async def get_analysis_history(
    limit: int = 50,
    offset: int = 0,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get user's analysis history.
    
    - **limit**: Maximum number of results to return (default: 50, max: 100)
    - **offset**: Number of results to skip (default: 0)
    - **Authentication required**
    """
    # Validate parameters
    if limit > 100:
        limit = 100
    if limit < 1:
        limit = 10
    if offset < 0:
        offset = 0
    
    # Get all analysis results for the user's files
    results = db.query(DetectionResultModel).join(MediaFile).filter(
        MediaFile.user_id == current_user.id
    ).order_by(desc(DetectionResultModel.analysis_time)).offset(offset).limit(limit).all()
    
    response_list = []
    for result in results:
        # Get the associated media file
        media_file = db.query(MediaFile).filter(MediaFile.id == result.media_file_id).first()
        
        if media_file:
            try:
                metadata = json.loads(result.result_metadata)
            except json.JSONDecodeError:
                metadata = {}
            
            api_detection_result = {
                "confidence_score": result.confidence_score * 100.0,
                "is_deepfake": result.is_deepfake,
                "analysis_metadata": metadata,
                "analysis_time": result.analysis_time.isoformat(),
                "processing_time_seconds": 0.0,
                "error": None
            }
            
            # Construct file URL for frontend (using public endpoint for image display)
            file_url = f"/api/files/public/{media_file.id}"
            
            response_list.append(DetectionResponse(
                success=True,
                message="Analysis result retrieved successfully",
                file_id=media_file.id,
                filename=media_file.filename,
                file_url=file_url,
                file_size=media_file.file_size,
                file_type=media_file.file_type,
                detection_result=api_detection_result,
                created_at=result.analysis_time
            ))
    
    return response_list

@router.get("/stats", response_model=Dict[str, Any])
async def get_analysis_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get analysis statistics for the current user.
    
    - **Authentication required**
    """
    # Get total analysis count
    total_analyses = db.query(DetectionResultModel).join(MediaFile).filter(
        MediaFile.user_id == current_user.id
    ).count()
    
    # Get deepfake detection count
    deepfake_count = db.query(DetectionResultModel).join(MediaFile).filter(
        MediaFile.user_id == current_user.id,
        DetectionResultModel.is_deepfake == True
    ).count()
    
    # Get average confidence score
    avg_confidence = db.query(DetectionResultModel).join(MediaFile).filter(
        MediaFile.user_id == current_user.id
    ).with_entities(func.avg(DetectionResultModel.confidence_score)).scalar()
    
    # Get recent activity (last 7 days)
    week_ago = datetime.utcnow() - timedelta(days=7)
    recent_analyses = db.query(DetectionResultModel).join(MediaFile).filter(
        MediaFile.user_id == current_user.id,
        DetectionResultModel.analysis_time >= week_ago
    ).count()
    
    return {
        "total_analyses": total_analyses,
        "deepfake_detections": deepfake_count,
        "authentic_detections": total_analyses - deepfake_count,
        "average_confidence": round(avg_confidence * 100, 2) if avg_confidence else 0,
        "recent_analyses_7_days": recent_analyses,
        "deepfake_rate": round((deepfake_count / total_analyses * 100), 2) if total_analyses > 0 else 0
    } 