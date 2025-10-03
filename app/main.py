from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.database import init_database, check_database_health, cleanup_database
from app.auth_routes import router as auth_router
from app.upload_routes import router as upload_router
from app.detection_routes import router as detection_router
from app.analysis_routes import router as analysis_router
from app.video_routes import router as video_router

# Import emergency detection routes
try:
    from app.emergency_detection_routes import router as emergency_detection_router
    EMERGENCY_DETECTION_AVAILABLE = True
except ImportError:
    EMERGENCY_DETECTION_AVAILABLE = False
    print("Warning: Emergency detection routes not available")

# Import advanced ensemble routes
try:
    from api.advanced_ensemble_api import router as advanced_ensemble_router
    ADVANCED_ENSEMBLE_AVAILABLE = True
except ImportError:
    ADVANCED_ENSEMBLE_AVAILABLE = False
    print("Warning: Advanced ensemble API not available")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_database()
    yield
    # Shutdown
    cleanup_database()

app = FastAPI(
    title="Deepfake Detection API",
    description="API for detecting deepfake images and videos with advanced ensemble methods",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
        "*"  # Keep wildcard for other origins
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router)
app.include_router(upload_router)
app.include_router(detection_router)
app.include_router(analysis_router, prefix="/api")
app.include_router(video_router)

# Include advanced ensemble routes if available
if ADVANCED_ENSEMBLE_AVAILABLE:
    app.include_router(advanced_ensemble_router)
    print("✓ Advanced ensemble API routes included")

@app.get("/")
async def root():
    return {
        "message": "Deepfake Detection API",
        "version": "1.0.0",
        "features": {
            "basic_detection": True,
            "advanced_ensemble": ADVANCED_ENSEMBLE_AVAILABLE,
            "multi_model": True,
            "analysis": True,
            "video_analysis": True
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    db_health = check_database_health()
    return {
        "status": "healthy",
        "database": db_health,
        "advanced_ensemble": ADVANCED_ENSEMBLE_AVAILABLE,
        "timestamp": "2024-08-01T00:00:00Z"
    }

@app.get("/models")
async def get_available_models():
    """Get information about available detection models"""
    return {
        "models": [
            {
                "name": "Xception",
                "type": "CNN",
                "description": "Xception-based deepfake detector",
                "available": True
            },
            {
                "name": "EfficientNet",
                "type": "CNN",
                "description": "EfficientNet-based deepfake detector",
                "available": True
            },
            {
                "name": "F3Net",
                "type": "Frequency",
                "description": "Frequency-based deepfake detector",
                "available": True
            }
        ],
        "ensemble_methods": [
            "attention_merge",
            "temperature_scaled",
            "monte_carlo_dropout",
            "adaptive_weighting",
            "agreement_resolution"
        ] if ADVANCED_ENSEMBLE_AVAILABLE else []
    }
