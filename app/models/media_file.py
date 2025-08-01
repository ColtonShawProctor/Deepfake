from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from app.database import Base
from datetime import datetime

class MediaFile(Base):
    """MediaFile model for storing uploaded media files"""
    __tablename__ = "media_files"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    upload_time = Column(DateTime, default=datetime.utcnow, nullable=False)
    file_size = Column(Integer, nullable=False)  # Size in bytes
    file_type = Column(String(50), nullable=False)  # e.g., 'image/jpeg', 'video/mp4'
    
    # Relationships
    user = relationship("User", back_populates="media_files")
    detection_results = relationship("DetectionResult", back_populates="media_file", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<MediaFile(id={self.id}, filename='{self.filename}', user_id={self.user_id})>"

    def to_dict(self) -> dict:
        """Convert media file to dictionary"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "filename": self.filename,
            "file_path": self.file_path,
            "upload_time": self.upload_time.isoformat() if self.upload_time else None,
            "file_size": self.file_size,
            "file_type": self.file_type
        }

    @property
    def file_size_mb(self) -> float:
        """Get file size in megabytes"""
        return round(self.file_size / (1024 * 1024), 2)

    @property
    def is_image(self) -> bool:
        """Check if file is an image"""
        return self.file_type.startswith('image/')

    @property
    def is_video(self) -> bool:
        """Check if file is a video"""
        return self.file_type.startswith('video/')
