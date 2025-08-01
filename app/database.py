from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from contextlib import contextmanager
from typing import Generator
import os
import time

# Database configuration
DATABASE_URL = "sqlite:///./deepfake.db"

# Create SQLAlchemy engine
engine = create_engine(
    DATABASE_URL,
    connect_args={
        "check_same_thread": False,  # Needed for SQLite
    },
    poolclass=StaticPool,  # Use static pool for SQLite
    echo=False,  # Set to True for SQL query logging
)

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create Base class for models
Base = declarative_base()

# Database initialization function
def init_database():
    """Initialize the database by creating all tables"""
    try:
        # Import all models here to ensure they are registered with Base
        from app.models.user import User
        from app.models.detection_result import DetectionResult
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        print("✅ Database initialized successfully")
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        raise

# Database session dependency for FastAPI
def get_db() -> Generator[Session, None, None]:
    """Dependency to get database session for FastAPI"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Context manager for database sessions
@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Context manager for database sessions"""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

# Database health check function
def check_database_health() -> dict:
    """Check database connectivity and health"""
    try:
        with get_db_session() as db:
            # Try a simple query
            db.execute("SELECT 1")
            return {
                "status": "healthy",
                "database": "deepfake.db",
                "connection": "active"
            }
    except Exception as e:
        return {
            "status": "unhealthy",
            "database": "deepfake.db",
            "error": str(e)
        }

# Database cleanup function
def cleanup_database():
    """Clean up database connections"""
    try:
        engine.dispose()
        print("✅ Database connections cleaned up")
    except Exception as e:
        print(f"❌ Database cleanup failed: {e}")

# Database file management
def get_database_size() -> int:
    """Get the size of the database file in bytes"""
    try:
        if os.path.exists("deepfake.db"):
            return os.path.getsize("deepfake.db")
        return 0
    except Exception:
        return 0

def backup_database(backup_path: str = None) -> bool:
    """Create a backup of the database"""
    try:
        if not os.path.exists("deepfake.db"):
            return False
        
        if backup_path is None:
            backup_path = f"deepfake_backup_{int(time.time())}.db"
        
        import shutil
        shutil.copy2("deepfake.db", backup_path)
        print(f"✅ Database backed up to {backup_path}")
        return True
    except Exception as e:
        print(f"❌ Database backup failed: {e}")
        return False
