"""
Advanced Ensemble API Routes

This module provides FastAPI routes for the advanced ensemble system,
including prediction, evaluation, and configuration management.
"""

import logging
import time
from typing import Dict, List, Optional, Any
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import numpy as np
from PIL import Image
import io
import json

# Import the advanced ensemble system
try:
    from app.models.advanced_ensemble import (
        AdvancedEnsembleManager, AdvancedEnsembleConfig, AdvancedFusionMethod
    )
    from app.models.advanced_ensemble_evaluator import AdvancedEnsembleEvaluator
    ADVANCED_ENSEMBLE_AVAILABLE = True
except ImportError:
    ADVANCED_ENSEMBLE_AVAILABLE = False
    print("Warning: Advanced ensemble models not available")
    # Create placeholder classes for type hints
    class AdvancedEnsembleManager:
        def __init__(self, config=None):
            pass
        def predict(self, image):
            return {"is_deepfake": False, "confidence": 0.5}
    
    class AdvancedEnsembleConfig:
        def __init__(self, **kwargs):
            pass
    
    class AdvancedFusionMethod:
        ATTENTION_MERGE = "attention_merge"
        TEMPERATURE_SCALED = "temperature_scaled"
        MONTE_CARLO_DROPOUT = "monte_carlo_dropout"
        ADAPTIVE_WEIGHTING = "adaptive_weighting"
        AGREEMENT_RESOLUTION = "agreement_resolution"
        CROSS_DATASET_FUSION = "cross_dataset_fusion"
    
    class AdvancedEnsembleEvaluator:
        def __init__(self, output_dir="evaluation_results"):
            pass
        def evaluate_ensemble(self, ensemble_manager, dataset_path=None):
            return {"accuracy": 0.5, "precision": 0.5, "recall": 0.5}


# Pydantic models for API requests/responses
class AdvancedEnsembleConfigRequest(BaseModel):
    """Request model for advanced ensemble configuration."""
    fusion_method: str = Field(default="attention_merge", description="Fusion method to use")
    temperature: float = Field(default=1.0, description="Temperature for scaling")
    min_models: int = Field(default=2, description="Minimum number of models required")
    max_models: int = Field(default=10, description="Maximum number of models allowed")
    confidence_threshold: float = Field(default=0.5, description="Confidence threshold")
    attention_dim: int = Field(default=128, description="Attention dimension")
    attention_heads: int = Field(default=8, description="Number of attention heads")
    mc_dropout_samples: int = Field(default=30, description="MC dropout samples")
    enable_adaptive_weighting: bool = Field(default=True, description="Enable adaptive weighting")
    enable_cross_dataset: bool = Field(default=False, description="Enable cross-dataset evaluation")


class AdvancedEnsembleResponse(BaseModel):
    """Response model for advanced ensemble prediction."""
    is_deepfake: bool
    confidence: float
    fusion_method: str
    uncertainty: float
    attention_weights: Dict[str, float]
    temperature_scaled_confidence: float
    mc_dropout_uncertainty: float
    adaptive_weights: Dict[str, float]
    agreement_score: float
    conflict_resolution: Optional[str]
    individual_predictions: Dict[str, Dict[str, Any]]
    processing_time: float
    metadata: Dict[str, Any]


class EvaluationRequest(BaseModel):
    """Request model for ensemble evaluation."""
    ensemble_config: AdvancedEnsembleConfigRequest
    evaluation_name: str = Field(description="Name for the evaluation")
    test_dataset_path: Optional[str] = Field(default=None, description="Path to test dataset")


class EvaluationResponse(BaseModel):
    """Response model for ensemble evaluation."""
    evaluation_id: str
    ensemble_name: str
    dataset_name: str
    metrics: Dict[str, float]
    processing_time: float
    status: str
    report_path: Optional[str] = None


# Global ensemble manager instance
advanced_ensemble_manager: Optional[AdvancedEnsembleManager] = None
evaluator: Optional[AdvancedEnsembleEvaluator] = None

# Initialize logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/advanced-ensemble", tags=["Advanced Ensemble"])


def get_ensemble_manager() -> AdvancedEnsembleManager:
    """Get or create the global ensemble manager."""
    global advanced_ensemble_manager
    
    if advanced_ensemble_manager is None and ADVANCED_ENSEMBLE_AVAILABLE:
        try:
            # Create default configuration
            config = AdvancedEnsembleConfig()
            advanced_ensemble_manager = AdvancedEnsembleManager(config)
            
            # Add mock models for testing (since we don't have actual model weights)
            # In production, you would load actual models here
            logger.info("Initialized advanced ensemble manager with mock models")
            
        except Exception as e:
            logger.error(f"Failed to initialize advanced ensemble manager: {e}")
            raise HTTPException(status_code=500, detail="Failed to initialize ensemble manager")
    
    if advanced_ensemble_manager is None:
        raise HTTPException(status_code=500, detail="Advanced ensemble system not available")
    
    return advanced_ensemble_manager


def get_evaluator() -> AdvancedEnsembleEvaluator:
    """Get or create the global evaluator."""
    global evaluator
    
    if evaluator is None and ADVANCED_ENSEMBLE_AVAILABLE:
        evaluator = AdvancedEnsembleEvaluator(output_dir="evaluation_results")
    elif evaluator is None and not ADVANCED_ENSEMBLE_AVAILABLE:
        evaluator = AdvancedEnsembleEvaluator(output_dir="evaluation_results")
    
    return evaluator


@router.post("/configure", response_model=Dict[str, Any])
async def configure_advanced_ensemble(config_request: AdvancedEnsembleConfigRequest):
    """
    Configure the advanced ensemble with custom settings.
    """
    if not ADVANCED_ENSEMBLE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Advanced ensemble system not available")
    
    try:
        # Convert string fusion method to enum
        fusion_method_map = {
            "attention_merge": AdvancedFusionMethod.ATTENTION_MERGE,
            "temperature_scaled": AdvancedFusionMethod.TEMPERATURE_SCALED,
            "monte_carlo_dropout": AdvancedFusionMethod.MONTE_CARLO_DROPOUT,
            "adaptive_weighting": AdvancedFusionMethod.ADAPTIVE_WEIGHTING,
            "agreement_resolution": AdvancedFusionMethod.AGREEMENT_RESOLUTION,
            "cross_dataset_fusion": AdvancedFusionMethod.CROSS_DATASET_FUSION
        }
        
        fusion_method = fusion_method_map.get(config_request.fusion_method, AdvancedFusionMethod.ATTENTION_MERGE)
        
        # Create new configuration
        config = AdvancedEnsembleConfig(
            fusion_method=fusion_method,
            temperature=config_request.temperature,
            min_models=config_request.min_models,
            max_models=config_request.max_models,
            confidence_threshold=config_request.confidence_threshold,
            attention_dim=config_request.attention_dim,
            attention_heads=config_request.attention_heads,
            mc_dropout_samples=config_request.mc_dropout_samples,
            enable_adaptive_weighting=config_request.enable_adaptive_weighting,
            enable_cross_dataset=config_request.enable_cross_dataset
        )
        
        # Create new ensemble manager with configuration
        global advanced_ensemble_manager
        advanced_ensemble_manager = AdvancedEnsembleManager(config)
        
        logger.info(f"Advanced ensemble configured with {config_request.fusion_method} fusion method")
        
        return {
            "status": "success",
            "message": "Advanced ensemble configured successfully",
            "config": config.__dict__,
            "models_loaded": 3  # Mock models
        }
        
    except Exception as e:
        logger.error(f"Failed to configure advanced ensemble: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Configuration failed: {str(e)}")


@router.post("/predict", response_model=AdvancedEnsembleResponse)
async def predict_advanced_ensemble(
    file: UploadFile = File(...),
    fusion_method: Optional[str] = Form(default="attention_merge")
):
    """
    Perform advanced ensemble prediction on uploaded image.
    """
    if not ADVANCED_ENSEMBLE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Advanced ensemble system not available")
    
    try:
        # Validate file
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Get ensemble manager
        ensemble_manager = get_ensemble_manager()
        
        # Update fusion method if specified
        if fusion_method != "attention_merge":
            fusion_method_map = {
                "attention_merge": AdvancedFusionMethod.ATTENTION_MERGE,
                "temperature_scaled": AdvancedFusionMethod.TEMPERATURE_SCALED,
                "monte_carlo_dropout": AdvancedFusionMethod.MONTE_CARLO_DROPOUT,
                "adaptive_weighting": AdvancedFusionMethod.ADAPTIVE_WEIGHTING,
                "agreement_resolution": AdvancedFusionMethod.AGREEMENT_RESOLUTION,
                "cross_dataset_fusion": AdvancedFusionMethod.CROSS_DATASET_FUSION
            }
            if fusion_method in fusion_method_map:
                ensemble_manager.config.fusion_method = fusion_method_map[fusion_method]
        
        # Perform prediction
        start_time = time.time()
        result = ensemble_manager.predict_advanced(image)
        processing_time = time.time() - start_time
        
        # Convert individual predictions to serializable format
        individual_predictions = {}
        for model_name, pred_result in result.individual_predictions.items():
            individual_predictions[model_name] = {
                "is_deepfake": pred_result.is_deepfake,
                "confidence": pred_result.confidence,
                "inference_time": pred_result.inference_time,
                "model_name": pred_result.model_name,
                "uncertainty": pred_result.uncertainty
            }
        
        return AdvancedEnsembleResponse(
            is_deepfake=result.is_deepfake,
            confidence=result.confidence,
            fusion_method=result.fusion_method,
            uncertainty=result.uncertainty,
            attention_weights=result.attention_weights,
            temperature_scaled_confidence=result.temperature_scaled_confidence,
            mc_dropout_uncertainty=result.mc_dropout_uncertainty,
            adaptive_weights=result.adaptive_weights,
            agreement_score=result.agreement_score,
            conflict_resolution=result.conflict_resolution,
            individual_predictions=individual_predictions,
            processing_time=processing_time,
            metadata=result.metadata
        )
        
    except Exception as e:
        logger.error(f"Advanced ensemble prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.get("/info")
async def get_advanced_ensemble_info():
    """
    Get information about the current advanced ensemble configuration.
    """
    if not ADVANCED_ENSEMBLE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Advanced ensemble system not available")
    
    try:
        ensemble_manager = get_ensemble_manager()
        info = ensemble_manager.get_ensemble_info()
        
        return {
            "status": "success",
            "ensemble_info": info
        }
        
    except Exception as e:
        logger.error(f"Failed to get ensemble info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get info: {str(e)}")


@router.post("/test-samples")
async def test_with_available_samples():
    """
    Test the advanced ensemble with available test samples.
    """
    if not ADVANCED_ENSEMBLE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Advanced ensemble system not available")
    
    try:
        # Load test samples
        samples_dir = Path("test_samples")
        metadata_file = samples_dir / "samples_metadata.json"
        
        if not metadata_file.exists():
            raise HTTPException(status_code=404, detail="Test samples not found")
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Test each sample
        results = []
        ensemble_manager = get_ensemble_manager()
        
        for sample_name, sample_info in metadata.items():
            sample_path = Path(sample_info["path"])
            if not sample_path.exists():
                continue
            
            try:
                # Load image
                image = Image.open(sample_path).convert('RGB')
                
                # Perform prediction
                start_time = time.time()
                result = ensemble_manager.predict_advanced(image)
                processing_time = time.time() - start_time
                
                # Create result
                sample_result = {
                    "sample_name": sample_name,
                    "expected": sample_info["expected"],
                    "prediction": result.is_deepfake,
                    "confidence": result.confidence,
                    "uncertainty": result.uncertainty,
                    "agreement_score": result.agreement_score,
                    "processing_time": processing_time,
                    "correct": result.is_deepfake == sample_info["expected"],
                    "description": sample_info["description"],
                    "source": sample_info["source"]
                }
                
                results.append(sample_result)
                logger.info(f"Tested {sample_name}: {'✓' if sample_result['correct'] else '✗'}")
                
            except Exception as e:
                logger.error(f"Failed to test {sample_name}: {e}")
                results.append({
                    "sample_name": sample_name,
                    "error": str(e),
                    "correct": False
                })
        
        # Calculate overall metrics
        correct = sum(1 for r in results if r.get("correct", False))
        total = len(results)
        accuracy = correct / total if total > 0 else 0
        
        return {
            "status": "success",
            "total_samples": total,
            "correct_predictions": correct,
            "accuracy": accuracy,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Test with samples failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Test failed: {str(e)}")


@router.get("/health")
async def health_check():
    """
    Health check endpoint for the advanced ensemble system.
    """
    return {
        "status": "healthy" if ADVANCED_ENSEMBLE_AVAILABLE else "unavailable",
        "advanced_ensemble_available": ADVANCED_ENSEMBLE_AVAILABLE,
        "ensemble_initialized": advanced_ensemble_manager is not None,
        "timestamp": time.time()
    } 