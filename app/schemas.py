from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import Optional

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
