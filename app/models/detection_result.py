from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from app.database import Base
from datetime import datetime
import json

class DetectionResult(Base):
    """DetectionResult model for storing deepfake detection results"""
    __tablename__ = "detection_results"

    id = Column(Integer, primary_key=True, index=True)
    media_file_id = Column(Integer, ForeignKey("media_files.id"), nullable=False)
    confidence_score = Column(Float, nullable=False)  # 0.0 to 1.0
    is_deepfake = Column(Boolean, nullable=False)  # True if deepfake detected
    model_name = Column(String(100), nullable=False)  # Name of the model used
    processing_time = Column(Float, nullable=False)  # Processing time in seconds
    uncertainty = Column(Float, nullable=True)  # Uncertainty score
    attention_weights = Column(Text, nullable=True)  # Attention weights for ensemble
    result_metadata = Column(Text, nullable=True)  # JSON field for additional data
    analysis_time = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    media_file = relationship("MediaFile", back_populates="detection_results")

    def __repr__(self):
        return f"<DetectionResult(id={self.id}, media_file_id={self.media_file_id}, confidence={self.confidence_score})>"

    def to_dict(self) -> dict:
        """Convert detection result to dictionary"""
        return {
            "id": self.id,
            "media_file_id": self.media_file_id,
            "confidence_score": self.confidence_score,
            "is_deepfake": self.is_deepfake,
            "model_name": self.model_name,
            "processing_time": self.processing_time,
            "uncertainty": self.uncertainty,
            "attention_weights": self.attention_weights,
            "analysis_time": self.analysis_time.isoformat() if self.analysis_time else None,
            "result_metadata": self.get_metadata()
        }

    def get_metadata(self) -> dict:
        """Get metadata as dictionary"""
        if self.result_metadata:
            try:
                return json.loads(self.result_metadata)
            except json.JSONDecodeError:
                return {}
        return {}

    def set_metadata(self, metadata: dict):
        """Set metadata as JSON string"""
        self.result_metadata = json.dumps(metadata)

    @property
    def result_summary(self) -> str:
        """Get a human-readable result summary"""
        if self.is_deepfake:
            return f"Deepfake detected with {self.confidence_score:.2%} confidence"
        else:
            return f"Authentic content with {self.confidence_score:.2%} confidence"

    @property
    def confidence_percentage(self) -> float:
        """Get confidence score as percentage"""
        return round(self.confidence_score * 100, 2)
