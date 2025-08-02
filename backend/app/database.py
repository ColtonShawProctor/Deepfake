"""
Database configuration and setup for the Deepfake Detection API.

This module provides:
- SQLAlchemy database engine and session management
- Base class for all models
- Database initialization and cleanup functions
- Session dependency for FastAPI
- Health check and testing utilities
"""

import os
import logging
from pathlib import Path
from typing import Generator, Dict, Any, Optional
from contextlib import contextmanager

from sqlalchemy import create_engine, text, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, scoped_session
from sqlalchemy.exc import SQLAlchemyError, OperationalError, IntegrityError
from sqlalchemy.pool import StaticPool

# Configure logging
logger = logging.getLogger(__name__)

# Database configuration
DATABASE_URL = "sqlite:///./deepfake.db"
DATABASE_PATH = Path("./deepfake.db")

# Create database directory if it doesn't exist
DATABASE_PATH.parent.mkdir(exist_ok=True)

# Engine configuration with optimized settings for SQLite
engine = create_engine(
    DATABASE_URL,
    connect_args={
        "check_same_thread": False,  # Required for SQLite with multiple threads
        "timeout": 30,  # Connection timeout in seconds
    },
    poolclass=StaticPool,  # Use static pool for SQLite
    pool_pre_ping=True,  # Verify connections before use
    echo=False,  # Set to True for SQL query logging
)

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False,  # Prevent expired object issues
)

# Scoped session for thread safety
SessionScoped = scoped_session(SessionLocal)

# Base class for all models
Base = declarative_base()

# Database event listeners for better SQLite support
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """Configure SQLite pragmas for better performance and foreign key support"""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA cache_size=10000")
    cursor.execute("PRAGMA temp_store=MEMORY")
    cursor.close()

def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency to get database session.
    
    Yields:
        Session: SQLAlchemy database session
        
    Note:
        This function automatically handles session cleanup.
    """
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()

@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Context manager for database sessions.
    
    Yields:
        Session: SQLAlchemy database session
        
    Example:
        with get_db_session() as db:
            result = db.query(User).all()
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()

def init_database() -> bool:
    """
    Initialize database and create all tables.
    
    Returns:
        bool: True if initialization successful, False otherwise
    """
    try:
        logger.info("Initializing database...")
        
        # Import all models to ensure they are registered with Base
        from app.models.user import User
        from app.models.media_file import MediaFile
        from app.models.detection_result import DetectionResult
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        
        logger.info("✅ Database initialized successfully")
        logger.info(f"Database file: {DATABASE_PATH.absolute()}")
        
        # Verify tables were created
        with engine.connect() as connection:
            result = connection.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
            tables = [row[0] for row in result.fetchall()]
            logger.info(f"Created tables: {tables}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        raise

def reset_database() -> bool:
    """
    Reset database by dropping all tables and recreating them.
    
    Returns:
        bool: True if reset successful, False otherwise
    """
    try:
        logger.warning("Resetting database - this will delete all data!")
        
        # Drop all tables
        Base.metadata.drop_all(bind=engine)
        logger.info("Dropped all tables")
        
        # Recreate tables
        init_database()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Database reset failed: {e}")
        raise

def cleanup_database() -> None:
    """Clean up database connections and resources."""
    try:
        # Dispose of the engine
        engine.dispose()
        logger.info("✅ Database connections cleaned up")
    except Exception as e:
        logger.error(f"❌ Database cleanup failed: {e}")

def check_database_health() -> Dict[str, Any]:
    """
    Check database health and connectivity.
    
    Returns:
        Dict containing health status and diagnostic information
    """
    try:
        # Test basic connectivity
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            result.fetchone()
        
        # Test session creation and basic operations
        with get_db_session() as db:
            # Check if tables exist
            result = db.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
            tables = [row[0] for row in result.fetchall()]
            
            # Test basic queries on each table
            table_counts = {}
            for table in tables:
                try:
                    result = db.execute(text(f"SELECT COUNT(*) FROM {table}"))
                    count = result.scalar()
                    table_counts[table] = count
                except Exception as e:
                    table_counts[table] = f"Error: {str(e)}"
        
        return {
            "status": "healthy",
            "database": str(DATABASE_PATH.absolute()),
            "connection": "successful",
            "tables": tables,
            "table_counts": table_counts,
            "database_size_mb": round(DATABASE_PATH.stat().st_size / (1024 * 1024), 2) if DATABASE_PATH.exists() else 0
        }
        
    except SQLAlchemyError as e:
        return {
            "status": "unhealthy",
            "database": str(DATABASE_PATH.absolute()),
            "error": str(e),
            "connection": "failed",
            "tables": [],
            "table_counts": {},
            "database_size_mb": 0
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "database": str(DATABASE_PATH.absolute()),
            "error": str(e),
            "connection": "unknown_error",
            "tables": [],
            "table_counts": {},
            "database_size_mb": 0
        }

def test_database_operations() -> bool:
    """
    Test basic database operations.
    
    Returns:
        bool: True if all tests pass, False otherwise
    """
    try:
        logger.info("Testing database operations...")
        
        with get_db_session() as db:
            # Test user table
            result = db.execute(text("SELECT COUNT(*) FROM users"))
            user_count = result.scalar()
            logger.info(f"Users in database: {user_count}")
            
            # Test media_files table
            result = db.execute(text("SELECT COUNT(*) FROM media_files"))
            file_count = result.scalar()
            logger.info(f"Files in database: {file_count}")
            
            # Test detection_results table
            result = db.execute(text("SELECT COUNT(*) FROM detection_results"))
            result_count = result.scalar()
            logger.info(f"Detection results in database: {result_count}")
            
            # Test foreign key relationships
            result = db.execute(text("""
                SELECT COUNT(*) FROM media_files m 
                JOIN users u ON m.user_id = u.id
            """))
            relationship_count = result.scalar()
            logger.info(f"Valid relationships: {relationship_count}")
        
        logger.info("✅ All database operation tests passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Database operation test failed: {e}")
        return False

def get_database_info() -> Dict[str, Any]:
    """
    Get comprehensive database information.
    
    Returns:
        Dict containing database configuration and statistics
    """
    health = check_database_health()
    
    return {
        "database_url": DATABASE_URL,
        "database_path": str(DATABASE_PATH.absolute()),
        "engine_config": {
            "pool_class": engine.pool.__class__.__name__,
            "pool_size": getattr(engine.pool, 'size', 'N/A'),
            "echo": engine.echo,
        },
        "health": health,
        "sqlite_pragmas": {
            "foreign_keys": "ON",
            "journal_mode": "WAL",
            "synchronous": "NORMAL",
            "cache_size": "10000",
            "temp_store": "MEMORY"
        }
    }

def backup_database(backup_path: Optional[str] = None) -> str:
    """
    Create a backup of the database.
    
    Args:
        backup_path: Optional path for backup file. If None, uses timestamp.
        
    Returns:
        str: Path to the backup file
    """
    import shutil
    from datetime import datetime
    
    if backup_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"./backups/deepfake_backup_{timestamp}.db"
    
    backup_path = Path(backup_path)
    backup_path.parent.mkdir(exist_ok=True)
    
    try:
        shutil.copy2(DATABASE_PATH, backup_path)
        logger.info(f"✅ Database backed up to: {backup_path}")
        return str(backup_path)
    except Exception as e:
        logger.error(f"❌ Database backup failed: {e}")
        raise

# Export commonly used items
__all__ = [
    "Base",
    "engine",
    "SessionLocal",
    "get_db",
    "get_db_session",
    "init_database",
    "reset_database",
    "cleanup_database",
    "check_database_health",
    "test_database_operations",
    "get_database_info",
    "backup_database"
] 