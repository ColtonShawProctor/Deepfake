from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import Optional, Dict, Any, List

class UserBase(BaseModel):
    email: EmailStr

class UserCreate(UserBase):
    password: str

class UserResponse(UserBase):
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    user_id: Optional[int] = None

class AuthResponse(BaseModel):
    success: bool
    message: str
    access_token: str
    token_type: str
    expires_in: int
    user: UserResponse

class UploadResponse(BaseModel):
    """Schema for file upload response"""
    file_id: int
    filename: str
    file_size: int
    file_type: str
    upload_time: datetime
    message: str
    
    class Config:
        from_attributes = True

# Deepfake Detection Schemas
class DetectionFeatures(BaseModel):
    """Schema for detection features"""
    face_detection: Optional[Dict[str, Any]] = None
    texture_analysis: Optional[Dict[str, Any]] = None
    color_analysis: Optional[Dict[str, Any]] = None
    edge_detection: Optional[Dict[str, Any]] = None

class ModelPredictions(BaseModel):
    """Schema for model predictions"""
    face_swap_probability: Optional[float] = None
    style_transfer_probability: Optional[float] = None
    gan_generated_probability: Optional[float] = None
    deepfake_indicators: Optional[int] = None

class ImageProperties(BaseModel):
    """Schema for image properties"""
    width: int
    height: int
    aspect_ratio: float
    file_size_bytes: int
    file_size_mb: float
    format: str
    color_mode: str
    filename: str

class AnalysisParameters(BaseModel):
    """Schema for analysis parameters"""
    model_version: Optional[str] = None
    analysis_method: Optional[str] = None
    confidence_threshold: Optional[float] = None
    processing_notes: Optional[str] = None

class ResultSummary(BaseModel):
    """Schema for result summary"""
    primary_indicator: Optional[str] = None
    secondary_indicators: Optional[List[str]] = None
    recommendation: Optional[str] = None

class AnalysisMetadata(BaseModel):
    """Schema for analysis metadata"""
    image_properties: Optional[ImageProperties] = None
    detection_features: Optional[DetectionFeatures] = None
    model_predictions: Optional[ModelPredictions] = None
    analysis_parameters: Optional[AnalysisParameters] = None
    result_summary: Optional[ResultSummary] = None

class DetectionResult(BaseModel):
    """Schema for deepfake detection result"""
    confidence_score: float
    is_deepfake: bool
    analysis_metadata: Optional[AnalysisMetadata] = None
    analysis_time: str
    processing_time_seconds: float
    error: Optional[str] = None

class DetectionResponse(BaseModel):
    """Schema for detection API response"""
    success: bool
    message: str
    file_id: int
    filename: str
    file_url: Optional[str] = None
    file_size: Optional[int] = None
    file_type: Optional[str] = None
    detection_result: DetectionResult
    created_at: datetime

class DetectorInfo(BaseModel):
    """Schema for detector information"""
    name: str
    version: str
    description: str
    capabilities: List[str]
    supported_formats: List[str]
    max_file_size_mb: int
    confidence_threshold: float
