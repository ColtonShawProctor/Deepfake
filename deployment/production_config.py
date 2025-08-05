"""
Production Configuration for Deepfake Detection System

This module contains production-specific configuration settings including
security, performance, and monitoring configurations.
"""

import os
from typing import List, Optional
from pydantic import BaseSettings, Field


class ProductionConfig(BaseSettings):
    """Production configuration settings."""
    
    # Environment
    ENVIRONMENT: str = "production"
    DEBUG: bool = False
    
    # Security
    SECRET_KEY: str = Field(..., description="Secret key for JWT tokens")
    JWT_SECRET: str = Field(..., description="JWT secret key")
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_HOURS: int = 24
    
    # Database
    DATABASE_URL: str = Field(..., description="PostgreSQL database URL")
    DATABASE_POOL_SIZE: int = 20
    DATABASE_MAX_OVERFLOW: int = 30
    DATABASE_POOL_TIMEOUT: int = 30
    DATABASE_POOL_RECYCLE: int = 3600
    
    # Redis
    REDIS_URL: str = Field(..., description="Redis connection URL")
    REDIS_MAX_CONNECTIONS: int = 50
    REDIS_SOCKET_TIMEOUT: int = 5
    REDIS_SOCKET_CONNECT_TIMEOUT: int = 5
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 4
    API_WORKER_CLASS: str = "uvicorn.workers.UvicornWorker"
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 100
    RATE_LIMIT_PER_HOUR: int = 1000
    RATE_LIMIT_BURST: int = 20
    
    # File Upload
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    ALLOWED_FILE_TYPES: List[str] = ["image/jpeg", "image/png", "image/gif", "video/mp4", "video/avi"]
    UPLOAD_DIR: str = "/app/uploads"
    TEMP_DIR: str = "/tmp"
    
    # Model Configuration
    MODEL_CACHE_SIZE: int = 1000
    MODEL_LOAD_TIMEOUT: int = 300
    MODEL_BATCH_SIZE: int = 32
    MODEL_DEVICE: str = "cuda" if os.getenv("CUDA_AVAILABLE", "false").lower() == "true" else "cpu"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE: str = "/app/logs/app.log"
    LOG_MAX_SIZE: int = 100 * 1024 * 1024  # 100MB
    LOG_BACKUP_COUNT: int = 5
    
    # Monitoring
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    HEALTH_CHECK_INTERVAL: int = 30
    HEALTH_CHECK_TIMEOUT: int = 10
    
    # CORS
    CORS_ORIGINS: List[str] = Field(
        default=["https://yourdomain.com", "https://www.yourdomain.com"],
        description="Allowed CORS origins"
    )
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]
    
    # Security Headers
    SECURITY_HEADERS: dict = {
        "X-Frame-Options": "DENY",
        "X-Content-Type-Options": "nosniff",
        "X-XSS-Protection": "1; mode=block",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline';"
    }
    
    # SSL/TLS
    SSL_CERT_FILE: Optional[str] = None
    SSL_KEY_FILE: Optional[str] = None
    
    # Performance
    WORKER_CONNECTIONS: int = 1024
    KEEPALIVE_TIMEOUT: int = 65
    CLIENT_MAX_BODY_SIZE: str = "100M"
    PROXY_READ_TIMEOUT: int = 300
    PROXY_SEND_TIMEOUT: int = 300
    
    # Cache Configuration
    CACHE_TTL: int = 3600  # 1 hour
    CACHE_MAX_SIZE: int = 1000
    CACHE_ENABLE: bool = True
    
    # Session Configuration
    SESSION_COOKIE_SECURE: bool = True
    SESSION_COOKIE_HTTPONLY: bool = True
    SESSION_COOKIE_SAMESITE: str = "Lax"
    SESSION_COOKIE_MAX_AGE: int = 86400  # 24 hours
    
    # Backup Configuration
    BACKUP_ENABLED: bool = True
    BACKUP_INTERVAL_HOURS: int = 24
    BACKUP_RETENTION_DAYS: int = 30
    BACKUP_PATH: str = "/app/backups"
    
    # Alerting
    ALERT_EMAIL_ENABLED: bool = True
    ALERT_EMAIL_SMTP_HOST: Optional[str] = None
    ALERT_EMAIL_SMTP_PORT: int = 587
    ALERT_EMAIL_USERNAME: Optional[str] = None
    ALERT_EMAIL_PASSWORD: Optional[str] = None
    ALERT_EMAIL_RECIPIENTS: List[str] = []
    
    class Config:
        env_file = ".env"
        case_sensitive = False


class SecurityConfig:
    """Security-specific configuration."""
    
    # Password Policy
    MIN_PASSWORD_LENGTH: int = 12
    REQUIRE_UPPERCASE: bool = True
    REQUIRE_LOWERCASE: bool = True
    REQUIRE_DIGITS: bool = True
    REQUIRE_SPECIAL_CHARS: bool = True
    PASSWORD_HISTORY_SIZE: int = 5
    
    # Account Lockout
    MAX_LOGIN_ATTEMPTS: int = 5
    LOCKOUT_DURATION_MINUTES: int = 30
    ACCOUNT_LOCKOUT_THRESHOLD: int = 10
    
    # Session Security
    SESSION_TIMEOUT_MINUTES: int = 60
    MAX_CONCURRENT_SESSIONS: int = 3
    FORCE_LOGOUT_ON_PASSWORD_CHANGE: bool = True
    
    # API Security
    API_KEY_LENGTH: int = 32
    API_KEY_EXPIRATION_DAYS: int = 365
    REQUIRE_API_KEY_FOR_UPLOADS: bool = True
    
    # File Security
    SCAN_UPLOADS_FOR_MALWARE: bool = True
    VALIDATE_FILE_SIGNATURES: bool = True
    MAX_FILENAME_LENGTH: int = 255
    SANITIZE_FILENAMES: bool = True


class PerformanceConfig:
    """Performance optimization configuration."""
    
    # Database Optimization
    DB_CONNECTION_POOL_SIZE: int = 20
    DB_CONNECTION_POOL_TIMEOUT: int = 30
    DB_CONNECTION_POOL_RECYCLE: int = 3600
    DB_STATEMENT_TIMEOUT: int = 30000  # 30 seconds
    
    # Cache Optimization
    REDIS_CONNECTION_POOL_SIZE: int = 50
    REDIS_SOCKET_TIMEOUT: int = 5
    REDIS_SOCKET_CONNECT_TIMEOUT: int = 5
    REDIS_RETRY_ON_TIMEOUT: bool = True
    
    # Model Optimization
    MODEL_PRECISION: str = "float16"  # Use mixed precision for faster inference
    MODEL_COMPILATION: bool = True
    MODEL_OPTIMIZATION_LEVEL: int = 2
    MODEL_MEMORY_EFFICIENT: bool = True
    
    # Async Configuration
    ASYNC_WORKER_POOL_SIZE: int = 10
    ASYNC_TASK_TIMEOUT: int = 300
    ASYNC_MAX_CONCURRENT_TASKS: int = 100
    
    # File Processing
    CHUNK_SIZE: int = 8192
    MAX_CONCURRENT_UPLOADS: int = 10
    UPLOAD_BUFFER_SIZE: int = 1024 * 1024  # 1MB


# Global configuration instances
config = ProductionConfig()
security_config = SecurityConfig()
performance_config = PerformanceConfig() 