from sqlalchemy import Column, Integer, String, DateTime, Boolean
from sqlalchemy.orm import relationship
from app.database import Base
from passlib.context import CryptContext
from datetime import datetime

# Password hashing context (same as auth.py)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class User(Base):
    """User model for authentication and user management"""
    __tablename__ = "users"
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(100), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Relationships
    media_files = relationship("MediaFile", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User(id={self.id}, email='{self.email}', is_active={self.is_active})>"

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using bcrypt (same as auth.py)"""
        return pwd_context.hash(password)

    def verify_password(self, password: str) -> bool:
        """Verify password against stored hash using bcrypt"""
        return pwd_context.verify(password, self.password_hash)

    @staticmethod
    def create_token() -> str:
        """Create a random token for authentication"""
        import secrets
        return secrets.token_urlsafe(32)

    def to_dict(self) -> dict:
        """Convert user to dictionary (excluding sensitive data)"""
        return {
            "id": self.id,
            "email": self.email,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "is_active": self.is_active
        }
