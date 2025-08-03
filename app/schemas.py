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
    face_detection: Dict[str, Any]
    texture_analysis: Dict[str, Any]
    color_analysis: Dict[str, Any]
    edge_detection: Dict[str, Any]

class ModelPredictions(BaseModel):
    """Schema for model predictions"""
    face_swap_probability: float
    style_transfer_probability: float
    gan_generated_probability: float
    deepfake_indicators: int

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
    model_version: str
    analysis_method: str
    confidence_threshold: float
    processing_notes: str

class ResultSummary(BaseModel):
    """Schema for result summary"""
    primary_indicator: str
    secondary_indicators: List[str]
    recommendation: str

class AnalysisMetadata(BaseModel):
    """Schema for analysis metadata"""
    image_properties: ImageProperties
    detection_features: DetectionFeatures
    model_predictions: ModelPredictions
    analysis_parameters: AnalysisParameters
    result_summary: ResultSummary

class DetectionResult(BaseModel):
    """Schema for deepfake detection result"""
    confidence_score: float
    is_deepfake: bool
    analysis_metadata: AnalysisMetadata
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
