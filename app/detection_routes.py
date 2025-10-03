from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
import json

from app.database import get_db
from app.auth import get_current_user
from app.models.user import User
from app.models.media_file import MediaFile
from app.models.detection_result import DetectionResult as DetectionResultModel
from app.schemas import DetectionResponse, DetectorInfo, DetectionResult
from app.models.huggingface_detector import HuggingFaceDetectorWrapper

router = APIRouter(prefix="/api/detection", tags=["deepfake detection"])

# Initialize the Hugging Face detector (replaces broken EfficientNet)
detector = HuggingFaceDetectorWrapper()

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
        # Perform deepfake detection with single EfficientNet model
        detection_result = detector.predict(str(file_path))
        
        # Store result in database
        from datetime import datetime
        analysis_time = datetime.utcnow()
        
        db_detection_result = DetectionResultModel(
            media_file_id=media_file.id,
            confidence_score=detection_result["confidence"],  # Already 0-1 scale
            is_deepfake=detection_result["is_deepfake"],
            model_name=detection_result.get("model", "efficientnet"),  # Single model name
            processing_time=detection_result.get("inference_time", 0.0),  # Inference time
            uncertainty=None,  # Single model doesn't have ensemble uncertainty
            attention_weights=None,  # Single model doesn't have ensemble weights
            analysis_time=analysis_time,
            result_metadata=json.dumps({
                "method": detection_result.get("method", "single"),
                "device": detection_result.get("device", "cpu"),
                "input_size": detection_result.get("input_size", [224, 224]),
                "tta_enabled": detection_result.get("method") == "tta"
            })
        )
        
        db.add(db_detection_result)
        db.commit()
        db.refresh(db_detection_result)
        
        # Create proper DetectionResult object from detector response
        from app.schemas import DetectionResult
        from datetime import datetime
        
        detection_result_obj = DetectionResult(
            confidence_score=detection_result["confidence"],
            is_deepfake=detection_result["is_deepfake"],
            analysis_metadata=None,  # Could be populated with additional metadata
            analysis_time=analysis_time.isoformat(),
            processing_time_seconds=detection_result.get("inference_time", 0.0),
            error=None
        )
        
        return DetectionResponse(
            success=True,
            message="Deepfake analysis completed successfully",
            file_id=media_file.id,
            filename=media_file.filename,
            detection_result=detection_result_obj,
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
            detail="File not found"
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
    
    # Create proper DetectionResult object
    from app.schemas import DetectionResult
    
    api_detection_result = DetectionResult(
        confidence_score=detection_result.confidence_score * 100.0,  # Convert from 0-1 to 0-100 scale
        is_deepfake=detection_result.is_deepfake,
        analysis_metadata=metadata,
        analysis_time=detection_result.analysis_time.isoformat(),
        processing_time_seconds=detection_result.processing_time,  # Use stored processing time
        error=None
    )
    
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
    page: int = 1,
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get all detection results for the current user's files with pagination
    """
    try:
        # Calculate offset for pagination
        offset = (page - 1) * limit
        
        # Get total count for pagination
        total_count = db.query(DetectionResultModel).join(MediaFile).filter(
            MediaFile.user_id == current_user.id
        ).count()
        
        # Get paginated results
        results = db.query(DetectionResultModel).join(MediaFile).filter(
            MediaFile.user_id == current_user.id
        ).order_by(DetectionResultModel.analysis_time.desc()).offset(offset).limit(limit).all()
        
        response_list = []
        for result in results:
            try:
                # Get the associated media file
                media_file = db.query(MediaFile).filter(MediaFile.id == result.media_file_id).first()
                
                if media_file:
                    try:
                        metadata = json.loads(result.result_metadata) if result.result_metadata else {}
                    except json.JSONDecodeError:
                        metadata = {}
                    
                    # Create proper DetectionResult object
                    api_detection_result = DetectionResult(
                        confidence_score=result.confidence_score * 100.0,  # Convert from 0-1 to 0-100 scale
                        is_deepfake=result.is_deepfake,
                        analysis_metadata=metadata,
                        analysis_time=result.analysis_time.isoformat(),
                        processing_time_seconds=result.processing_time,  # Map from database field
                        error=None
                    )
                    
                    response_list.append(DetectionResponse(
                        success=True,
                        message="Detection result retrieved successfully",
                        file_id=media_file.id,
                        filename=media_file.filename,
                        file_size=media_file.file_size,
                        file_type=media_file.file_type,
                        detection_result=api_detection_result,
                        created_at=result.analysis_time
                    ))
            except Exception as e:
                print(f"Error processing result {result.id}: {str(e)}")
                continue
        
        return response_list
        
    except Exception as e:
        print(f"Error in get_user_detection_results: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve results: {str(e)}"
        )

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