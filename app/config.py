from typing import List

class Settings:
    DATABASE_URL: str = "sqlite:///./deepfake.db"
    SECRET_KEY: str = "your-secret-key-here"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 24 * 60  # 24 hours
    CORS_ORIGINS: List[str] = ["*"]
    MODEL_PATH: str = "./models"
    UPLOAD_DIR: str = "./uploads"

settings = Settings()
