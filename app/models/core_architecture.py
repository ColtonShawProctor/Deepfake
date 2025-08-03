"""
Core Architecture for Multi-Model Deepfake Detection System

This module implements the foundational classes and interfaces for the multi-model
deepfake detection system, providing a clean, extensible architecture.
"""

import os
import time
import logging
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import uuid

import torch
import numpy as np
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Core Data Classes
# ============================================================================

@dataclass
class DetectionResult:
    """Standardized detection result format"""
    confidence_score: float
    is_deepfake: bool
    model_name: str
    processing_time: float
    model_version: str
    metadata: Dict[str, Any]
    heatmap: Optional[np.ndarray] = None
    uncertainty: Optional[float] = None
    attention_weights: Optional[List[float]] = None

@dataclass
class ModelInfo:
    """Model information and metadata"""
    name: str
    version: str
    architecture: str
    input_size: Tuple[int, int]
    performance_metrics: Dict[str, float]
    supported_formats: List[str]
    device_requirements: str
    inference_time: float

@dataclass
class EnsembleResult:
    """Result from ensemble prediction"""
    overall_confidence: float
    overall_verdict: str
    individual_predictions: Dict[str, DetectionResult]
    ensemble_method: str
    uncertainty: float
    attention_weights: List[float]
    processing_time: float
    metadata: Dict[str, Any]

@dataclass
class MultiModelResult:
    """Result from multi-model analysis"""
    task_id: str
    overall_confidence: float
    overall_verdict: str
    model_results: Dict[str, DetectionResult]
    ensemble_result: Optional[EnsembleResult]
    processing_time: float
    metadata: Dict[str, Any]
    heatmaps: Optional[Dict[str, np.ndarray]] = None
    uncertainty: Optional[float] = None

@dataclass
class ProgressInfo:
    """Progress information for async operations"""
    task_id: str
    status: str  # pending, processing, completed, failed
    progress: float  # 0.0 to 1.0
    completed_models: List[str]
    remaining_models: List[str]
    estimated_time_remaining: Optional[float] = None

# ============================================================================
# Configuration Classes
# ============================================================================

@dataclass
class AugmentationConfig:
    """Configuration for data augmentation"""
    enable_augmentation: bool = True
    rotation_range: float = 10.0
    zoom_range: float = 0.1
    horizontal_flip: bool = True
    brightness_range: float = 0.1
    contrast_range: float = 0.1

@dataclass
class MesoNetConfig:
    """Configuration for MesoNet detector"""
    input_size: Tuple[int, int] = (256, 256)
    batch_size: int = 1
    device: str = "auto"
    enable_cache: bool = True
    enable_calibration: bool = True
    enable_monitoring: bool = True
    augmentation_config: AugmentationConfig = field(default_factory=AugmentationConfig)

@dataclass
class EnsembleConfig:
    """Configuration for ensemble methods"""
    method: str = "weighted_average"  # weighted_average, voting, stacking
    enable_uncertainty: bool = True
    enable_calibration: bool = True
    fallback_strategy: str = "majority_vote"
    min_models_required: int = 1
    default_weights: Dict[str, float] = field(default_factory=dict)

@dataclass
class APIConfig:
    """Configuration for API endpoints"""
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    supported_formats: List[str] = field(default_factory=lambda: ["jpg", "jpeg", "png"])
    enable_async: bool = True
    max_concurrent_requests: int = 10
    timeout_seconds: int = 300

@dataclass
class ResourceConfig:
    """Configuration for resource management"""
    enable_gpu: bool = True
    max_gpu_memory: Optional[float] = None  # GB
    enable_memory_optimization: bool = True
    device_allocation_strategy: str = "round_robin"  # round_robin, priority, load_balanced

@dataclass
class MonitoringConfig:
    """Configuration for performance monitoring"""
    enable_monitoring: bool = True
    log_metrics: bool = True
    save_performance_data: bool = True
    performance_data_path: str = "performance_data"
    metrics_retention_days: int = 30

@dataclass
class CacheConfig:
    """Configuration for model caching"""
    enable_cache: bool = True
    max_cache_size: int = 5  # Number of models to cache
    cache_cleanup_interval: int = 3600  # seconds
    cache_persistence: bool = False

@dataclass
class AsyncConfig:
    """Configuration for async processing"""
    max_workers: int = 4
    enable_progress_tracking: bool = True
    task_timeout: int = 300  # seconds
    enable_cancellation: bool = True

@dataclass
class ErrorConfig:
    """Configuration for error handling"""
    enable_fallback: bool = True
    log_errors: bool = True
    error_retention_days: int = 90
    enable_error_recovery: bool = True

# ============================================================================
# Abstract Base Classes
# ============================================================================

class BaseDetector(ABC):
    """Abstract base class for all deepfake detectors"""
    
    def __init__(self, model_name: str, device: str = "auto"):
        self.model_name = model_name
        self.device = self._get_device(device)
        self.model = None
        self.transform = None
        self.logger = logging.getLogger(f"{__name__}.{model_name}")
        self.is_model_loaded = False
        
    def _get_device(self, device: str) -> str:
        """Determine the best available device"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    @abstractmethod
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """Load the model and weights"""
        pass
    
    @abstractmethod
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for model input"""
        pass
    
    @abstractmethod
    def predict(self, image: Image.Image) -> DetectionResult:
        """Perform prediction on image"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> ModelInfo:
        """Get model information"""
        pass
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.is_model_loaded
    
    def __call__(self, image: Image.Image) -> DetectionResult:
        """Convenience method for prediction"""
        return self.predict(image)

# ============================================================================
# Model Registry
# ============================================================================

class ModelRegistry:
    """Registry for managing multiple detector models"""
    
    def __init__(self):
        self.models: Dict[str, BaseDetector] = {}
        self.model_configs: Dict[str, Dict] = {}
        self.logger = logging.getLogger(f"{__name__}.ModelRegistry")
    
    def register_model(self, name: str, detector: BaseDetector, config: Dict = None):
        """Register a new detector model"""
        if name in self.models:
            self.logger.warning(f"Model {name} already registered, overwriting")
        
        self.models[name] = detector
        self.model_configs[name] = config or {}
        self.logger.info(f"Registered model: {name}")
    
    def get_model(self, name: str) -> Optional[BaseDetector]:
        """Get a registered model by name"""
        return self.models.get(name)
    
    def list_models(self) -> List[str]:
        """List all registered models"""
        return list(self.models.keys())
    
    def load_all_models(self) -> Dict[str, bool]:
        """Load all registered models"""
        results = {}
        for name, detector in self.models.items():
            try:
                # Load model if not already loaded
                if not detector.is_loaded():
                    success = detector.load_model()
                    results[name] = success
                    if success:
                        self.logger.info(f"Successfully loaded model: {name}")
                    else:
                        self.logger.error(f"Failed to load model: {name}")
                else:
                    results[name] = True
                    self.logger.info(f"Model {name} already loaded")
            except Exception as e:
                self.logger.error(f"Error loading model {name}: {str(e)}")
                results[name] = False
        
        return results
    
    def unregister_model(self, name: str) -> bool:
        """Unregister a model"""
        if name in self.models:
            del self.models[name]
            if name in self.model_configs:
                del self.model_configs[name]
            self.logger.info(f"Unregistered model: {name}")
            return True
        return False
    
    def get_model_status(self) -> Dict[str, Dict]:
        """Get status of all registered models"""
        status = {}
        for name, detector in self.models.items():
            status[name] = {
                "loaded": detector.is_loaded(),
                "device": getattr(detector, 'device', 'unknown'),
                "config": self.model_configs.get(name, {})
            }
        return status

# ============================================================================
# Model Factory
# ============================================================================

class ModelFactory:
    """Factory for creating detector instances"""
    
    @staticmethod
    def create_mesonet(config: MesoNetConfig) -> 'MesoNetDetector':
        """Create MesoNet detector instance"""
        from .mesonet_detector import MesoNetDetector
        return MesoNetDetector(config)
    
    @staticmethod
    def create_resnet(config: Dict) -> 'ResNetDetector':
        """Create ResNet detector instance"""
        from .deepfake_models import ResNetDetector
        return ResNetDetector(device=config.get('device', 'auto'))
    
    @staticmethod
    def create_efficientnet(config: Dict) -> 'EfficientNetDetector':
        """Create EfficientNet detector instance"""
        from .deepfake_models import EfficientNetDetector
        return EfficientNetDetector(device=config.get('device', 'auto'))
    
    @staticmethod
    def create_f3net(config: Dict) -> 'F3NetDetector':
        """Create F3Net detector instance"""
        from .deepfake_models import F3NetDetector
        return F3NetDetector(device=config.get('device', 'auto'))

# ============================================================================
# Ensemble Manager
# ============================================================================

class EnsembleManager:
    """Manages ensemble of multiple detectors"""
    
    def __init__(self, config: EnsembleConfig):
        self.config = config
        self.models: Dict[str, BaseDetector] = {}
        self.weights: Dict[str, float] = {}
        self.logger = logging.getLogger(f"{__name__}.EnsembleManager")
        
        # Initialize default weights if provided
        if config.default_weights:
            self.weights.update(config.default_weights)
    
    def add_model(self, name: str, detector: BaseDetector, weight: float = 1.0):
        """Add model to ensemble"""
        self.models[name] = detector
        self.weights[name] = weight
        self.logger.info(f"Added model {name} to ensemble with weight {weight}")
    
    def remove_model(self, name: str) -> bool:
        """Remove model from ensemble"""
        if name in self.models:
            del self.models[name]
            if name in self.weights:
                del self.weights[name]
            self.logger.info(f"Removed model {name} from ensemble")
            return True
        return False
    
    def predict_ensemble(self, image: Image.Image) -> EnsembleResult:
        """Perform ensemble prediction"""
        start_time = time.time()
        
        if not self.models:
            raise ValueError("No models in ensemble")
        
        # Get predictions from all models
        predictions = {}
        for name, model in self.models.items():
            try:
                result = model.predict(image)
                predictions[name] = result
            except Exception as e:
                self.logger.error(f"Model {name} prediction failed: {str(e)}")
                # Use fallback prediction
                predictions[name] = DetectionResult(
                    confidence_score=50.0,  # Neutral prediction
                    is_deepfake=False,
                    model_name=name,
                    processing_time=0.0,
                    model_version="fallback",
                    metadata={"error": str(e)}
                )
        
        # Apply ensemble method
        if self.config.method == "weighted_average":
            ensemble_result = self._weighted_average(predictions)
        elif self.config.method == "voting":
            ensemble_result = self._majority_voting(predictions)
        else:
            ensemble_result = self._weighted_average(predictions)  # Default
        
        # Calculate uncertainty if enabled
        uncertainty = None
        if self.config.enable_uncertainty:
            uncertainty = self._calculate_uncertainty(predictions)
        
        processing_time = time.time() - start_time
        
        return EnsembleResult(
            overall_confidence=ensemble_result["confidence"],
            overall_verdict=ensemble_result["verdict"],
            individual_predictions=predictions,
            ensemble_method=self.config.method,
            uncertainty=uncertainty,
            attention_weights=list(self.weights.values()),
            processing_time=processing_time,
            metadata=ensemble_result.get("metadata", {})
        )
    
    def _weighted_average(self, predictions: Dict[str, DetectionResult]) -> Dict:
        """Calculate weighted average of predictions"""
        total_weight = sum(self.weights.get(name, 1.0) for name in predictions.keys())
        weighted_sum = sum(
            predictions[name].confidence_score * self.weights.get(name, 1.0)
            for name in predictions.keys()
        )
        
        avg_confidence = weighted_sum / total_weight if total_weight > 0 else 50.0
        verdict = "DEEPFAKE" if avg_confidence > 50.0 else "AUTHENTIC"
        
        return {
            "confidence": avg_confidence,
            "verdict": verdict,
            "metadata": {"method": "weighted_average"}
        }
    
    def _majority_voting(self, predictions: Dict[str, DetectionResult]) -> Dict:
        """Calculate majority vote of predictions"""
        deepfake_votes = sum(1 for pred in predictions.values() if pred.is_deepfake)
        total_votes = len(predictions)
        
        is_deepfake = deepfake_votes > total_votes / 2
        confidence = (deepfake_votes / total_votes) * 100 if is_deepfake else (1 - deepfake_votes / total_votes) * 100
        verdict = "DEEPFAKE" if is_deepfake else "AUTHENTIC"
        
        return {
            "confidence": confidence,
            "verdict": verdict,
            "metadata": {"method": "majority_voting", "votes": deepfake_votes, "total": total_votes}
        }
    
    def _calculate_uncertainty(self, predictions: Dict[str, DetectionResult]) -> float:
        """Calculate prediction uncertainty"""
        if len(predictions) < 2:
            return 0.0
        
        confidences = [pred.confidence_score for pred in predictions.values()]
        return float(np.var(confidences))
    
    def optimize_weights(self, validation_data: List[Tuple[Image.Image, bool]]):
        """Optimize ensemble weights (placeholder for future implementation)"""
        self.logger.info("Weight optimization not yet implemented")
        # TODO: Implement weight optimization using validation data
    
    def get_ensemble_info(self) -> Dict[str, Any]:
        """Get information about the ensemble"""
        return {
            "num_models": len(self.models),
            "model_names": list(self.models.keys()),
            "weights": self.weights.copy(),
            "method": self.config.method,
            "enable_uncertainty": self.config.enable_uncertainty
        }

# ============================================================================
# Performance Monitor
# ============================================================================

class PerformanceMonitor:
    """Monitors and tracks model performance metrics"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.timers: Dict[str, float] = {}
        self.logger = logging.getLogger(f"{__name__}.PerformanceMonitor")
        
        # Create performance data directory if needed
        if config.save_performance_data:
            Path(config.performance_data_path).mkdir(exist_ok=True)
    
    def start_timer(self, model_name: str):
        """Start timing for a model"""
        self.timers[model_name] = time.time()
    
    def end_timer(self, model_name: str) -> float:
        """End timing and record duration"""
        if model_name in self.timers:
            duration = time.time() - self.timers[model_name]
            self.metrics[f"{model_name}_inference_time"].append(duration)
            del self.timers[model_name]
            return duration
        return 0.0
    
    def record_accuracy(self, model_name: str, accuracy: float):
        """Record accuracy metric"""
        self.metrics[f"{model_name}_accuracy"].append(accuracy)
    
    def record_confidence(self, model_name: str, confidence: float):
        """Record confidence metric"""
        self.metrics[f"{model_name}_confidence"].append(confidence)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report"""
        report = {}
        for metric_name, values in self.metrics.items():
            if values:
                report[metric_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "count": len(values)
                }
        return report
    
    def save_performance_data(self, filename: str = None):
        """Save performance data to file"""
        if not self.config.save_performance_data:
            return
        
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"performance_data_{timestamp}.json"
        
        filepath = Path(self.config.performance_data_path) / filename
        
        import json
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        self.logger.info(f"Performance data saved to {filepath}")

# ============================================================================
# Async Processing Manager
# ============================================================================

class AsyncProcessingManager:
    """Manages asynchronous processing of multiple models"""
    
    def __init__(self, config: AsyncConfig):
        self.config = config
        self.executor = ThreadPoolExecutor(max_workers=config.max_workers)
        self.results_cache = {}
        self.active_tasks = {}
        self.logger = logging.getLogger(f"{__name__}.AsyncProcessingManager")
    
    async def process_image_async(self, image: Image.Image, 
                                models: Dict[str, BaseDetector]) -> Dict[str, DetectionResult]:
        """Process image with multiple models asynchronously"""
        task_id = str(uuid.uuid4())
        
        # Create tasks for each model
        tasks = {}
        for name, model in models.items():
            task = asyncio.create_task(self._process_single_model(name, model, image))
            tasks[name] = task
        
        # Wait for all tasks to complete
        results = {}
        for name, task in tasks.items():
            try:
                result = await asyncio.wait_for(task, timeout=self.config.task_timeout)
                results[name] = result
            except asyncio.TimeoutError:
                self.logger.error(f"Task {name} timed out")
                results[name] = self._create_timeout_result(name)
            except Exception as e:
                self.logger.error(f"Task {name} failed: {str(e)}")
                results[name] = self._create_error_result(name, str(e))
        
        return results
    
    async def _process_single_model(self, name: str, model: BaseDetector, 
                                  image: Image.Image) -> DetectionResult:
        """Process image with a single model"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, model.predict, image)
    
    def _create_timeout_result(self, model_name: str) -> DetectionResult:
        """Create timeout result"""
        return DetectionResult(
            confidence_score=50.0,
            is_deepfake=False,
            model_name=model_name,
            processing_time=0.0,
            model_version="timeout",
            metadata={"error": "Task timed out"}
        )
    
    def _create_error_result(self, model_name: str, error: str) -> DetectionResult:
        """Create error result"""
        return DetectionResult(
            confidence_score=50.0,
            is_deepfake=False,
            model_name=model_name,
            processing_time=0.0,
            model_version="error",
            metadata={"error": error}
        )
    
    def cancel_processing(self, task_id: str):
        """Cancel ongoing processing task"""
        if task_id in self.active_tasks:
            self.active_tasks[task_id].cancel()
            del self.active_tasks[task_id]
            self.logger.info(f"Cancelled task {task_id}")

# ============================================================================
# Resource Manager
# ============================================================================

class ResourceManager:
    """Manages GPU/CPU resources for model inference"""
    
    def __init__(self, config: ResourceConfig):
        self.config = config
        self.gpu_available = torch.cuda.is_available()
        self.device_map: Dict[str, str] = {}
        self.device_usage: Dict[str, float] = {}
        self.logger = logging.getLogger(f"{__name__}.ResourceManager")
    
    def allocate_device(self, model_name: str, priority: int = 1) -> str:
        """Allocate device for model"""
        if self.config.device_allocation_strategy == "round_robin":
            device = self._allocate_round_robin()
        elif self.config.device_allocation_strategy == "priority":
            device = self._allocate_by_priority(priority)
        else:  # load_balanced
            device = self._allocate_load_balanced()
        
        self.device_map[model_name] = device
        self.logger.info(f"Allocated device {device} for model {model_name}")
        return device
    
    def _allocate_round_robin(self) -> str:
        """Allocate device using round-robin strategy"""
        devices = ["cpu"]
        if self.gpu_available:
            devices.append("cuda")
        
        # Simple round-robin allocation
        used_devices = set(self.device_map.values())
        for device in devices:
            if device not in used_devices:
                return device
        
        return devices[0]  # Fallback to first device
    
    def _allocate_by_priority(self, priority: int) -> str:
        """Allocate device by priority"""
        if priority > 5 and self.gpu_available:
            return "cuda"
        return "cpu"
    
    def _allocate_load_balanced(self) -> str:
        """Allocate device based on current load"""
        if not self.gpu_available:
            return "cpu"
        
        # Simple load balancing - prefer CPU if GPU is heavily used
        gpu_usage = self.device_usage.get("cuda", 0.0)
        if gpu_usage > 0.8:  # 80% usage threshold
            return "cpu"
        return "cuda"
    
    def release_device(self, model_name: str):
        """Release device allocation"""
        if model_name in self.device_map:
            device = self.device_map[model_name]
            del self.device_map[model_name]
            self.logger.info(f"Released device {device} for model {model_name}")
    
    def get_device_usage(self) -> Dict[str, float]:
        """Get current device usage statistics"""
        usage = {"cpu": 0.0, "cuda": 0.0}
        
        # Count models per device
        device_counts = defaultdict(int)
        for device in self.device_map.values():
            device_counts[device] += 1
        
        # Calculate usage based on number of models
        total_models = len(self.device_map)
        if total_models > 0:
            for device, count in device_counts.items():
                usage[device] = count / total_models
        
        return usage
    
    def optimize_device_allocation(self):
        """Optimize device allocation based on usage"""
        # Simple optimization - move models to less used devices
        usage = self.get_device_usage()
        
        if usage["cuda"] > 0.8 and usage["cpu"] < 0.5:
            # Move some models from GPU to CPU
            gpu_models = [name for name, device in self.device_map.items() if device == "cuda"]
            for model_name in gpu_models[:2]:  # Move up to 2 models
                self.device_map[model_name] = "cpu"
                self.logger.info(f"Optimized: moved {model_name} from GPU to CPU")

# ============================================================================
# Error Handler
# ============================================================================

class ErrorHandler:
    """Handles errors and implements fallback strategies"""
    
    def __init__(self, config: ErrorConfig):
        self.config = config
        self.error_log: List[Dict] = []
        self.logger = logging.getLogger(f"{__name__}.ErrorHandler")
    
    def handle_model_error(self, model_name: str, error: Exception) -> DetectionResult:
        """Handle model-specific errors"""
        error_record = {
            "timestamp": time.time(),
            "model_name": model_name,
            "error_type": type(error).__name__,
            "error_message": str(error)
        }
        
        if self.config.log_errors:
            self.error_log.append(error_record)
            self.logger.error(f"Model {model_name} error: {str(error)}")
        
        # Return fallback prediction
        return self._get_fallback_prediction(model_name, error)
    
    def handle_ensemble_error(self, error: Exception) -> DetectionResult:
        """Handle ensemble errors"""
        error_record = {
            "timestamp": time.time(),
            "error_type": type(error).__name__,
            "error_message": str(error)
        }
        
        if self.config.log_errors:
            self.error_log.append(error_record)
            self.logger.error(f"Ensemble error: {str(error)}")
        
        return self._get_fallback_prediction("ensemble", error)
    
    def _get_fallback_prediction(self, context: str, error: Exception) -> DetectionResult:
        """Get fallback prediction when models fail"""
        return DetectionResult(
            confidence_score=50.0,  # Neutral prediction
            is_deepfake=False,
            model_name=f"fallback_{context}",
            processing_time=0.0,
            model_version="fallback",
            metadata={
                "error": str(error),
                "fallback_reason": context,
                "timestamp": time.time()
            }
        )
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of recent errors"""
        if not self.error_log:
            return {"total_errors": 0, "recent_errors": []}
        
        # Get recent errors (last 24 hours)
        current_time = time.time()
        recent_errors = [
            error for error in self.error_log
            if current_time - error["timestamp"] < 86400  # 24 hours
        ]
        
        return {
            "total_errors": len(self.error_log),
            "recent_errors": len(recent_errors),
            "error_types": self._count_error_types(),
            "models_with_errors": self._get_models_with_errors()
        }
    
    def _count_error_types(self) -> Dict[str, int]:
        """Count occurrences of different error types"""
        error_counts = defaultdict(int)
        for error in self.error_log:
            error_counts[error["error_type"]] += 1
        return dict(error_counts)
    
    def _get_models_with_errors(self) -> List[str]:
        """Get list of models that have had errors"""
        models = set()
        for error in self.error_log:
            if "model_name" in error:
                models.add(error["model_name"])
        return list(models) 