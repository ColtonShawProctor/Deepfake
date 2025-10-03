"""
Multi-Model Deepfake Detection API

This module implements the enhanced API endpoints for multi-model deepfake detection,
integrating all the architecture components with comprehensive error handling,
progress tracking, and advanced features.
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import uuid

from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..models.core_architecture import (
    ModelRegistry,
    EnsembleManager,
    AsyncProcessingManager,
    ResourceManager,
    ErrorHandler,
    MultiModelResult,
    ProgressInfo,
    EnsembleConfig,
    APIConfig,
    ResourceConfig,
    AsyncConfig,
    ErrorConfig
)
from ..models.mesonet_detector import MesoNetDetector, MesoNetConfig
from ..models.deepfake_models import ResNetDetector, EfficientNetDetector, F3NetDetector
from ..models.model_selector import ModelSelector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# API Models
# ============================================================================

class AnalysisRequest(BaseModel):
    """Request model for image analysis"""
    models: Optional[List[str]] = None  # List of model names to use
    use_ensemble: bool = True
    generate_heatmaps: bool = False
    timeout_seconds: Optional[int] = 300

class AnalysisResponse(BaseModel):
    """Response model for analysis results"""
    task_id: str
    status: str
    overall_confidence: float
    overall_verdict: str
    model_results: Dict[str, Dict]
    ensemble_result: Optional[Dict]
    processing_time: float
    metadata: Dict[str, Any]

class ProgressResponse(BaseModel):
    """Response model for progress tracking"""
    task_id: str
    status: str
    progress: float
    completed_models: List[str]
    remaining_models: List[str]
    estimated_time_remaining: Optional[float]

class ModelInfoResponse(BaseModel):
    """Response model for model information"""
    name: str
    version: str
    architecture: str
    input_size: List[int]
    performance_metrics: Dict[str, float]
    supported_formats: List[str]
    device_requirements: str
    inference_time: float

# ============================================================================
# Multi-Model API Class
# ============================================================================

class MultiModelAPI:
    """Enhanced API for multi-model deepfake detection"""
    
    def __init__(self, config: Optional[APIConfig] = None):
        self.config = config or APIConfig()
        self.registry = ModelRegistry()
        self.ensemble = EnsembleManager(EnsembleConfig())
        self.async_manager = AsyncProcessingManager(AsyncConfig())
        self.resource_manager = ResourceManager(ResourceConfig())
        self.error_handler = ErrorHandler(ErrorConfig())
        
        # Initialize intelligent model selector
        self.model_selector = ModelSelector()
        
        # Task tracking
        self.active_tasks: Dict[str, Dict] = {}
        
        # Initialize models
        self._initialize_models()
        
        self.logger = logging.getLogger(f"{__name__}.MultiModelAPI")
    
    def _initialize_models(self):
        """Initialize all available models"""
        try:
            # Register MesoNet
            mesonet_config = MesoNetConfig()
            mesonet = MesoNetDetector(mesonet_config)
            self.registry.register_model("MesoNet", mesonet, {"type": "enhanced"})
            
            # Register existing models
            resnet = ResNetDetector()
            self.registry.register_model("ResNet", resnet, {"type": "resnet"})
            
            efficientnet = EfficientNetDetector()
            self.registry.register_model("EfficientNet", efficientnet, {"type": "efficientnet"})
            
            f3net = F3NetDetector()
            self.registry.register_model("F3Net", f3net, {"type": "f3net"})
            
            # Add models to ensemble
            self.ensemble.add_model("MesoNet", mesonet, weight=0.3)
            self.ensemble.add_model("ResNet", resnet, weight=0.4)
            self.ensemble.add_model("EfficientNet", efficientnet, weight=0.2)
            self.ensemble.add_model("F3Net", f3net, weight=0.1)
            
            # Load all models
            load_results = self.registry.load_all_models()
            self.logger.info(f"Model loading results: {load_results}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize models: {str(e)}")
    
    async def analyze_image_multi_model(self, image, models: Optional[List[str]] = None) -> MultiModelResult:
        """Analyze image with intelligent model selection"""
        task_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # Analyze input to determine optimal model selection
            input_analysis = self.model_selector.analyze_input(image)
            
            # Get available models
            available_models = {name: model for name, model in self.registry.models.items() if model.is_loaded()}
            
            # Use intelligent model selection if no specific models requested
            if models is None:
                models = self.model_selector.select_models(input_analysis, available_models, max_models=3)
                self.logger.info(f"Intelligent model selection: {models} for {input_analysis.complexity.value} complexity")
            else:
                # Validate requested models
                invalid_models = [m for m in models if m not in available_models]
                if invalid_models:
                    raise ValueError(f"Invalid models: {invalid_models}")
            
            # Get model instances for selected models
            model_instances = {}
            for model_name in models:
                if model_name in available_models:
                    model_instances[model_name] = available_models[model_name]
            
            if not model_instances:
                raise ValueError("No valid models available")
            
            # Process with selected models
            model_results = await self.async_manager.process_image_async(image, model_instances)
            
            # Calculate overall results
            confidences = [result.confidence_score for result in model_results.values()]
            overall_confidence = sum(confidences) / len(confidences)
            overall_verdict = "DEEPFAKE" if overall_confidence > 50.0 else "AUTHENTIC"
            
            processing_time = time.time() - start_time
            
            # Get selection rationale for metadata
            selection_rationale = self.model_selector.get_selection_rationale(input_analysis, models)
            
            return MultiModelResult(
                task_id=task_id,
                overall_confidence=overall_confidence,
                overall_verdict=overall_verdict,
                model_results=model_results,
                ensemble_result=None,  # Will be calculated separately if needed
                processing_time=processing_time,
                metadata={
                    "models_used": list(model_results.keys()),
                    "total_models": len(model_results),
                    "success_rate": len(model_results) / len(models),
                    "input_analysis": {
                        "complexity": input_analysis.complexity.value,
                        "face_confidence": input_analysis.face_confidence,
                        "image_quality": input_analysis.image_quality,
                        "noise_level": input_analysis.noise_level
                    },
                    "selection_rationale": selection_rationale,
                    "optimization_enabled": True
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Multi-model analysis failed: {str(e)}")
            
            # Return error result
            return MultiModelResult(
                task_id=task_id,
                overall_confidence=0.0,
                overall_verdict="ERROR",
                model_results={},
                ensemble_result=None,
                processing_time=processing_time,
                metadata={"error": str(e)}
            )
    
    async def analyze_image_ensemble(self, image) -> MultiModelResult:
        """Analyze image with ensemble"""
        task_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # Perform ensemble prediction
            ensemble_result = self.ensemble.predict_ensemble(image)
            
            processing_time = time.time() - start_time
            
            return MultiModelResult(
                task_id=task_id,
                overall_confidence=ensemble_result.overall_confidence,
                overall_verdict=ensemble_result.overall_verdict,
                model_results=ensemble_result.individual_predictions,
                ensemble_result=ensemble_result,
                processing_time=processing_time,
                metadata={
                    "ensemble_method": ensemble_result.ensemble_method,
                    "uncertainty": ensemble_result.uncertainty,
                    "attention_weights": ensemble_result.attention_weights
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Ensemble analysis failed: {str(e)}")
            
            # Return error result
            return MultiModelResult(
                task_id=task_id,
                overall_confidence=0.0,
                overall_verdict="ERROR",
                model_results={},
                ensemble_result=None,
                processing_time=processing_time,
                metadata={"error": str(e)}
            )
    
    async def analyze_batch(self, images: List, models: Optional[List[str]] = None) -> List[MultiModelResult]:
        """Analyze batch of images"""
        results = []
        
        for i, image in enumerate(images):
            try:
                result = await self.analyze_image_multi_model(image, models)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Batch analysis failed for image {i}: {str(e)}")
                # Create error result for this image
                error_result = MultiModelResult(
                    task_id=str(uuid.uuid4()),
                    overall_confidence=0.0,
                    overall_verdict="ERROR",
                    model_results={},
                    ensemble_result=None,
                    processing_time=0.0,
                    metadata={"error": str(e), "image_index": i}
                )
                results.append(error_result)
        
        return results
    
    def get_analysis_progress(self, task_id: str) -> ProgressInfo:
        """Get progress of ongoing analysis"""
        if task_id not in self.active_tasks:
            return ProgressInfo(
                task_id=task_id,
                status="not_found",
                progress=0.0,
                completed_models=[],
                remaining_models=[],
                estimated_time_remaining=None
            )
        
        task_info = self.active_tasks[task_id]
        return ProgressInfo(
            task_id=task_id,
            status=task_info.get("status", "unknown"),
            progress=task_info.get("progress", 0.0),
            completed_models=task_info.get("completed_models", []),
            remaining_models=task_info.get("remaining_models", []),
            estimated_time_remaining=task_info.get("estimated_time_remaining")
        )
    
    def get_model_info(self) -> Dict[str, ModelInfoResponse]:
        """Get information about all models"""
        model_info = {}
        
        for name, model in self.registry.models.items():
            try:
                info = model.get_model_info()
                model_info[name] = ModelInfoResponse(
                    name=info.name,
                    version=info.version,
                    architecture=info.architecture,
                    input_size=list(info.input_size),
                    performance_metrics=info.performance_metrics,
                    supported_formats=info.supported_formats,
                    device_requirements=info.device_requirements,
                    inference_time=info.inference_time
                )
            except Exception as e:
                self.logger.error(f"Failed to get info for model {name}: {str(e)}")
        
        return model_info
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        return {
            "models": self.registry.get_model_status(),
            "ensemble": self.ensemble.get_ensemble_info(),
            "resources": self.resource_manager.get_device_usage(),
            "errors": self.error_handler.get_error_summary(),
            "active_tasks": len(self.active_tasks)
        }

# ============================================================================
# FastAPI Router
# ============================================================================

router = APIRouter(prefix="/api/v2", tags=["Multi-Model Detection"])

# Global API instance
api_instance = MultiModelAPI()

@router.post("/analyze/multi-model", response_model=AnalysisResponse)
async def analyze_image_multi_model(
    file: UploadFile = File(...),
    models: Optional[str] = None,
    use_ensemble: bool = True,
    generate_heatmaps: bool = False
):
    """Analyze image with multiple models"""
    try:
        # Validate file
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and validate image
        from PIL import Image
        import io
        
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Parse models parameter
        model_list = None
        if models:
            model_list = [m.strip() for m in models.split(',')]
        
        # Perform analysis
        if use_ensemble:
            result = await api_instance.analyze_image_ensemble(image)
        else:
            result = await api_instance.analyze_image_multi_model(image, model_list)
        
        # Convert to response format
        model_results_dict = {}
        for name, detection_result in result.model_results.items():
            model_results_dict[name] = {
                "confidence_score": detection_result.confidence_score,
                "is_deepfake": detection_result.is_deepfake,
                "processing_time": detection_result.processing_time,
                "model_version": detection_result.model_version,
                "metadata": detection_result.metadata
            }
        
        ensemble_result_dict = None
        if result.ensemble_result:
            ensemble_result_dict = {
                "overall_confidence": result.ensemble_result.overall_confidence,
                "overall_verdict": result.ensemble_result.overall_verdict,
                "ensemble_method": result.ensemble_result.ensemble_method,
                "uncertainty": result.ensemble_result.uncertainty,
                "attention_weights": result.ensemble_result.attention_weights
            }
        
        return AnalysisResponse(
            task_id=result.task_id,
            status="completed",
            overall_confidence=result.overall_confidence,
            overall_verdict=result.overall_verdict,
            model_results=model_results_dict,
            ensemble_result=ensemble_result_dict,
            processing_time=result.processing_time,
            metadata=result.metadata
        )
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze/batch")
async def analyze_batch(
    files: List[UploadFile] = File(...),
    models: Optional[str] = None
):
    """Analyze batch of images"""
    try:
        # Parse models parameter
        model_list = None
        if models:
            model_list = [m.strip() for m in models.split(',')]
        
        # Process images
        images = []
        for file in files:
            if not file.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail=f"File {file.filename} must be an image")
            
            from PIL import Image
            import io
            
            image_data = await file.read()
            image = Image.open(io.BytesIO(image_data))
            images.append(image)
        
        # Perform batch analysis
        results = await api_instance.analyze_batch(images, model_list)
        
        # Convert to response format
        response_results = []
        for result in results:
            model_results_dict = {}
            for name, detection_result in result.model_results.items():
                model_results_dict[name] = {
                    "confidence_score": detection_result.confidence_score,
                    "is_deepfake": detection_result.is_deepfake,
                    "processing_time": detection_result.processing_time,
                    "model_version": detection_result.model_version,
                    "metadata": detection_result.metadata
                }
            
            response_results.append({
                "task_id": result.task_id,
                "overall_confidence": result.overall_confidence,
                "overall_verdict": result.overall_verdict,
                "model_results": model_results_dict,
                "processing_time": result.processing_time,
                "metadata": result.metadata
            })
        
        return {"results": response_results}
        
    except Exception as e:
        logger.error(f"Batch analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models", response_model=Dict[str, ModelInfoResponse])
async def get_models():
    """Get information about all available models"""
    try:
        return api_instance.get_model_info()
    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_system_status():
    """Get overall system status"""
    try:
        return api_instance.get_system_status()
    except Exception as e:
        logger.error(f"Failed to get system status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/progress/{task_id}", response_model=ProgressResponse)
async def get_progress(task_id: str):
    """Get progress of ongoing analysis"""
    try:
        progress = api_instance.get_analysis_progress(task_id)
        return ProgressResponse(
            task_id=progress.task_id,
            status=progress.status,
            progress=progress.progress,
            completed_models=progress.completed_models,
            remaining_models=progress.remaining_models,
            estimated_time_remaining=progress.estimated_time_remaining
        )
    except Exception as e:
        logger.error(f"Failed to get progress for task {task_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/cancel/{task_id}")
async def cancel_analysis(task_id: str):
    """Cancel ongoing analysis"""
    try:
        api_instance.async_manager.cancel_processing(task_id)
        return {"message": f"Task {task_id} cancelled successfully"}
    except Exception as e:
        logger.error(f"Failed to cancel task {task_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Health Check Endpoint
# ============================================================================

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        status = api_instance.get_system_status()
        
        # Check if any models are loaded
        models_loaded = any(
            model_status["loaded"] 
            for model_status in status["models"].values()
        )
        
        if models_loaded:
            return {
                "status": "healthy",
                "models_loaded": models_loaded,
                "timestamp": time.time()
            }
        else:
            return {
                "status": "degraded",
                "models_loaded": models_loaded,
                "message": "No models are currently loaded",
                "timestamp": time.time()
            }
            
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }

# ============================================================================
# Optimization Endpoints
# ============================================================================

@router.post("/analyze/optimized")
async def analyze_image_optimized(file: UploadFile = File(...)):
    """Analyze image with intelligent model selection optimization"""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Analyze with intelligent model selection
        result = await api_instance.analyze_image_multi_model(image)
        
        return {
            "task_id": result.task_id,
            "overall_confidence": result.overall_confidence,
            "overall_verdict": result.overall_verdict,
            "processing_time": result.processing_time,
            "model_results": {
                name: {
                    "confidence": model_result.confidence_score,
                    "verdict": model_result.verdict,
                    "processing_time": model_result.processing_time
                } for name, model_result in result.model_results.items()
            },
            "optimization_info": result.metadata,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Optimized analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/model-selection/info")
async def get_model_selection_info():
    """Get information about intelligent model selection capabilities"""
    try:
        if not hasattr(api_instance, 'model_selector'):
            raise HTTPException(status_code=503, detail="Model selector not available")
        
        selector = api_instance.model_selector
        
        return {
            "available_models": list(selector.model_profiles.keys()),
            "model_profiles": {
                name: {
                    "performance_tier": profile.performance_tier.value,
                    "base_accuracy": profile.base_accuracy,
                    "base_inference_time": profile.base_inference_time,
                    "memory_usage": profile.memory_usage,
                    "priority": profile.priority
                } for name, profile in selector.model_profiles.items()
            },
            "selection_criteria": {
                "max_inference_time": selector.max_inference_time,
                "max_memory_usage": selector.max_memory_usage,
                "min_accuracy_threshold": selector.min_accuracy_threshold
            },
            "complexity_levels": [level.value for level in selector.InputComplexity],
            "performance_tiers": [tier.value for tier in selector.ModelPerformanceTier]
        }
        
    except Exception as e:
        logger.error(f"Model selection info failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model selection info: {str(e)}") 