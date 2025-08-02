from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
import json

from app.database import get_db
from app.auth import get_current_user
from app.models.user import User
from app.models.media_file import MediaFile
from app.models.detection_result import DetectionResult as DetectionResultModel
from app.schemas import DetectionResponse, DetectorInfo
from app.utils.deepfake_detector import DeepfakeDetector

router = APIRouter(prefix="/api/detection", tags=["deepfake detection"])

# Initialize the detector
detector = DeepfakeDetector()

@router.get("/info", response_model=DetectorInfo)
async def get_detector_info():
    """
    Get information about the deepfake detector service
    """
    return DetectorInfo(**detector.get_detector_info())

@router.post("/analyze/{file_id}", response_model=DetectionResponse)
async def analyze_file(
    file_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Analyze a previously uploaded file for deepfake detection
    
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
    from pathlib import Path
    file_path = Path(media_file.file_path)
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found on disk"
        )
    
    try:
        # Perform deepfake detection
        detection_result = detector.analyze_image(str(file_path))
        
        # Store result in database
        from datetime import datetime
        analysis_time = datetime.fromisoformat(detection_result["analysis_time"].replace('Z', '+00:00'))
        
        db_detection_result = DetectionResultModel(
            media_file_id=media_file.id,
            confidence_score=detection_result["confidence_score"] / 100.0,  # Convert to 0-1 scale
            is_deepfake=detection_result["is_deepfake"],
            analysis_time=analysis_time,
            result_metadata=json.dumps(detection_result["analysis_metadata"])
        )
        
        db.add(db_detection_result)
        db.commit()
        db.refresh(db_detection_result)
        
        return DetectionResponse(
            success=True,
            message="Deepfake analysis completed successfully",
            file_id=media_file.id,
            filename=media_file.filename,
            detection_result=detection_result,
            created_at=db_detection_result.analysis_time
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )

@router.get("/results/{file_id}", response_model=DetectionResponse)
async def get_detection_result(
    file_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get the detection result for a specific file
    
    - **file_id**: ID of the file to get results for
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
    
    # Get the latest detection result
    detection_result = db.query(DetectionResultModel).filter(
        DetectionResultModel.media_file_id == file_id
    ).order_by(DetectionResultModel.analysis_time.desc()).first()
    
    if not detection_result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No detection results found for this file"
        )
    
    # Convert database result back to API format
    try:
        metadata = json.loads(detection_result.result_metadata)
    except json.JSONDecodeError:
        metadata = {}
    
    api_detection_result = {
        "confidence_score": detection_result.confidence_score * 100.0,  # Convert back to 0-100 scale
        "is_deepfake": detection_result.is_deepfake,
        "analysis_metadata": metadata,
        "analysis_time": detection_result.analysis_time.isoformat(),
        "processing_time_seconds": 0.0,  # Not stored in DB
        "error": None
    }
    
    return DetectionResponse(
        success=True,
        message="Detection result retrieved successfully",
        file_id=media_file.id,
        filename=media_file.filename,
        detection_result=api_detection_result,
        created_at=detection_result.analysis_time
    )

@router.get("/results", response_model=List[DetectionResponse])
async def get_user_detection_results(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get all detection results for the current user's files
    """
    # Get all files owned by the user that have detection results
    results = db.query(DetectionResultModel).join(MediaFile).filter(
        MediaFile.user_id == current_user.id
    ).order_by(DetectionResultModel.analysis_time.desc()).all()
    
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
            
            response_list.append(DetectionResponse(
                success=True,
                message="Detection result retrieved successfully",
                file_id=media_file.id,
                filename=media_file.filename,
                detection_result=api_detection_result,
                created_at=result.analysis_time
            ))
    
    return response_list

@router.delete("/results/{file_id}")
async def delete_detection_result(
    file_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Delete detection results for a specific file (only if owned by current user)
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
    
    # Delete all detection results for this file
    deleted_count = db.query(DetectionResultModel).filter(
        DetectionResultModel.media_file_id == file_id
    ).delete()
    
    db.commit()
    
    return {
        "message": f"Deleted {deleted_count} detection result(s) for file {file_id}",
        "deleted_count": deleted_count
    } 