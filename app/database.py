from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
import os
from pathlib import Path

# Database configuration
DATABASE_URL = "sqlite:///./deepfake.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_database():
    """Initialize database and create tables"""
    try:
        # Import models to ensure they are registered
        from app.models.user import User
        from app.models.media_file import MediaFile
        from app.models.detection_result import DetectionResult
        
        # Create tables
        Base.metadata.create_all(bind=engine)
        print("✅ Database initialized successfully")
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        raise

def cleanup_database():
    """Clean up database connections"""
    try:
        engine.dispose()
        print("✅ Database connections cleaned up")
    except Exception as e:
        print(f"❌ Database cleanup failed: {e}")

def check_database_health():
    """Check database health and connectivity"""
    try:
        # Test database connection using SQLAlchemy 2.0 syntax
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            result.fetchone()
        
        # Test session creation and basic operations
        db = SessionLocal()
        try:
            # Test if we can query the database
            result = db.execute(text("SELECT COUNT(*) FROM users"))
            result.scalar()
            db.commit()
            
            return {
                "status": "healthy",
                "database": "deepfake.db",
                "connection": "successful",
                "tables": ["users", "media_files", "detection_results"]
            }
        finally:
            db.close()
            
    except SQLAlchemyError as e:
        return {
            "status": "unhealthy",
            "database": "deepfake.db",
            "error": str(e),
            "connection": "failed"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "database": "deepfake.db",
            "error": str(e),
            "connection": "unknown_error"
        }

def test_database_operations():
    """Test basic database operations"""
    try:
        db = SessionLocal()
        
        # Test user table
        result = db.execute(text("SELECT COUNT(*) FROM users"))
        user_count = result.scalar()
        print(f"Users in database: {user_count}")
        
        # Test media_files table
        result = db.execute(text("SELECT COUNT(*) FROM media_files"))
        file_count = result.scalar()
        print(f"Files in database: {file_count}")
        
        # Test detection_results table
        result = db.execute(text("SELECT COUNT(*) FROM detection_results"))
        result_count = result.scalar()
        print(f"Detection results in database: {result_count}")
        
        db.close()
        return True
    except Exception as e:
        print(f"Database operation test failed: {e}")
        return False
