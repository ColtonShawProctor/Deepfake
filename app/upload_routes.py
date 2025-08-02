import os
import uuid
import shutil
from pathlib import Path
from typing import List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from fastapi.responses import JSONResponse, FileResponse
from sqlalchemy.orm import Session
from PIL import Image
import io

from app.database import get_db
from app.auth import get_current_user
from app.models.user import User
from app.models.media_file import MediaFile
from app.schemas import UploadResponse

router = APIRouter(prefix="/api", tags=["file upload"])

# Configuration
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB in bytes
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/jpg"}
UPLOAD_DIR = Path("uploads")

def create_upload_directory():
    """Create uploads directory if it doesn't exist"""
    UPLOAD_DIR.mkdir(exist_ok=True)

def generate_unique_filename(original_filename: str) -> str:
    """Generate a unique filename to prevent conflicts"""
    # Get file extension
    file_ext = Path(original_filename).suffix.lower()
    
    # Generate unique filename with UUID
    unique_id = str(uuid.uuid4())
    return f"{unique_id}{file_ext}"

def validate_image_file(file_content: bytes) -> bool:
    """Validate that the file is actually an image using Pillow"""
    try:
        # Try to open the image with Pillow
        image = Image.open(io.BytesIO(file_content))
        image.verify()  # Verify it's a valid image
        return True
    except Exception:
        return False

def get_file_type(file_content: bytes, filename: str) -> str:
    """Determine file type using Pillow and filename"""
    try:
        image = Image.open(io.BytesIO(file_content))
        format_name = image.format.lower()
        
        # Map format to MIME type
        format_to_mime = {
            'jpeg': 'image/jpeg',
            'jpg': 'image/jpeg', 
            'png': 'image/png'
        }
        
        return format_to_mime.get(format_name, 'image/unknown')
    except Exception:
        # Fallback to filename extension
        ext = Path(filename).suffix.lower()
        if ext in ['.jpg', '.jpeg']:
            return 'image/jpeg'
        elif ext == '.png':
            return 'image/png'
        else:
            return 'image/unknown'

@router.post("/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Upload an image file (JPEG, PNG only)
    
    - **file**: Image file to upload (max 10MB)
    - **Authentication required**
    """
    
    # Create uploads directory if it doesn't exist
    create_upload_directory()
    
    # Validate file type
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Allowed types: {', '.join(ALLOWED_IMAGE_TYPES)}"
        )
    
    # Validate file size
    file_content = await file.read()
    if len(file_content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB"
        )
    
    # Validate that it's actually an image
    if not validate_image_file(file_content):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image file. File must be a valid JPEG or PNG image."
        )
    
    # Generate unique filename
    unique_filename = generate_unique_filename(file.filename)
    file_path = UPLOAD_DIR / unique_filename
    
    try:
        # Save file to disk
        with open(file_path, "wb") as buffer:
            buffer.write(file_content)
        
        # Determine file type using Pillow
        file_type = get_file_type(file_content, file.filename)
        
        # Store metadata in database
        media_file = MediaFile(
            user_id=current_user.id,
            filename=file.filename,  # Original filename
            file_path=str(file_path),  # Full path to saved file
            file_size=len(file_content),
            file_type=file_type
        )
        
        db.add(media_file)
        db.commit()
        db.refresh(media_file)
        
        return UploadResponse(
            file_id=media_file.id,
            filename=file.filename,
            file_size=media_file.file_size,
            file_type=media_file.file_type,
            upload_time=media_file.upload_time,
            message="File uploaded successfully"
        )
        
    except Exception as e:
        # Clean up file if database operation fails
        if file_path.exists():
            file_path.unlink()
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload file: {str(e)}"
        )

@router.get("/files", response_model=List[UploadResponse])
async def get_user_files(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get all files uploaded by the current user
    """
    files = db.query(MediaFile).filter(MediaFile.user_id == current_user.id).all()
    
    return [
        UploadResponse(
            file_id=file.id,
            filename=file.filename,
            file_size=file.file_size,
            file_type=file.file_type,
            upload_time=file.upload_time,
            message="File retrieved successfully"
        )
        for file in files
    ]

@router.delete("/files/{file_id}")
async def delete_file(
    file_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Delete a file (only if owned by current user)
    """
    # Get file and verify ownership
    media_file = db.query(MediaFile).filter(
        MediaFile.id == file_id,
        MediaFile.user_id == current_user.id
    ).first()
    
    if not media_file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found or access denied"
        )
    
    try:
        # Delete file from disk
        file_path = Path(media_file.file_path)
        if file_path.exists():
            file_path.unlink()
        
        # Delete from database
        db.delete(media_file)
        db.commit()
        
        return {"message": "File deleted successfully"}
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete file: {str(e)}"
        )


@router.get("/uploads/{user_id}", response_model=List[UploadResponse])
async def list_user_files(
    user_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List files uploaded by a specific user (only if requesting user is the owner)"""
    if current_user.id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied. You can only view your own files."
        )
    
    files = db.query(MediaFile).filter(MediaFile.user_id == user_id).all()
    
    return [
        UploadResponse(
            file_id=file.id,
            filename=file.filename,
            file_size=file.file_size,
            file_type=file.file_type,
            upload_time=file.upload_time,
            message="File retrieved successfully"
        )
        for file in files
    ]

@router.get("/files/{file_id}")
async def serve_file(
    file_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Serve a specific file by ID (only if owned by current user)"""
    media_file = db.query(MediaFile).filter(
        MediaFile.id == file_id,
        MediaFile.user_id == current_user.id
    ).first()
    
    if not media_file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found or access denied"
        )
    
    file_path = Path(media_file.file_path)
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found on disk"
        )
    
    content_type = media_file.file_type or "application/octet-stream"
    
    return FileResponse(
        path=file_path,
        filename=media_file.filename,
        media_type=content_type,
        headers={
            "Content-Disposition": f"inline; filename={media_file.filename}",
            "Cache-Control": "public, max-age=3600"
        }
    )
