from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.database import init_database, check_database_health, cleanup_database
from app.auth_routes import router as auth_router
from app.upload_routes import router as upload_router
from app.detection_routes import router as detection_router
from app.analysis_routes import router as analysis_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_database()
    yield
    # Shutdown
    cleanup_database()

app = FastAPI(
    title="Deepfake Detection API",
    description="API for detecting deepfake images and videos",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router)
app.include_router(upload_router)
app.include_router(detection_router)
app.include_router(analysis_router, prefix="/api")

@app.get("/")
async def root():
    return {"message": "Deepfake Detection API"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    db_health = check_database_health()
    return {
        "status": "healthy",
        "database": db_health,
        "timestamp": "2024-08-01T00:00:00Z"
    }
