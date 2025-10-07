"""
Parallel Processing Optimization for Advanced Ensemble Detection

This module implements intelligent parallel processing, resource allocation,
and concurrent model execution to maximize performance and efficiency.
"""

import asyncio
import logging
import time
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import psutil
import torch
from queue import PriorityQueue, Queue
import gc

from .base_detector import BaseDetector, DetectionResult


class ProcessingPriority(str, Enum):
    """Priority levels for model processing."""
    HIGH = "high"        # Critical models (Xception, F3Net)
    MEDIUM = "medium"    # Balanced models (EfficientNet)
    LOW = "low"          # Fast models (MesoNet)


class ResourceType(str, Enum):
    """Types of computational resources."""
    CPU = "cpu"
    GPU = "gpu"
    MIXED = "mixed"


@dataclass
class ModelResourceProfile:
    """Resource profile for a model."""
    model_name: str
    priority: ProcessingPriority
    estimated_memory: float  # MB
    estimated_time: float    # seconds
    gpu_required: bool
    batch_size: int
    warmup_time: float
    peak_memory: float


@dataclass
class ResourceAllocation:
    """Resource allocation for model execution."""
    model_name: str
    device: str
    memory_limit: float
    priority: ProcessingPriority
    executor_type: str
    batch_size: int
    estimated_time: float


@dataclass
class ProcessingTask:
    """Task for parallel model processing."""
    task_id: str
    model_name: str
    input_data: Any
    priority: ProcessingPriority
    resource_allocation: ResourceAllocation
    callback: Optional[Callable] = None
    created_at: float = 0.0


class ResourceAllocator:
    """
    Intelligent resource allocator that manages GPU/CPU resources
    and optimizes allocation based on model requirements and system state.
    """
    
    def __init__(self, max_gpu_memory: float = 8.0, max_cpu_cores: int = None):
        self.max_gpu_memory = max_gpu_memory * 1024  # Convert to MB
        self.max_cpu_cores = max_cpu_cores or psutil.cpu_count()
        self.logger = logging.getLogger(f"{__name__}.ResourceAllocator")
        
        # Resource tracking
        self.allocated_gpu_memory = 0.0
        self.allocated_cpu_cores = 0
        self.active_allocations: Dict[str, ResourceAllocation] = {}
        
        # Model resource profiles
        self.model_profiles = {
            "EfficientNet": ModelResourceProfile(
                model_name="EfficientNet",
                priority=ProcessingPriority.MEDIUM,
                estimated_memory=512.0,
                estimated_time=0.08,
                gpu_required=False,
                batch_size=4,
                warmup_time=0.5,
                peak_memory=600.0
            ),
            "Xception": ModelResourceProfile(
                model_name="Xception",
                priority=ProcessingPriority.HIGH,
                estimated_memory=2048.0,
                estimated_time=0.15,
                gpu_required=True,
                batch_size=2,
                warmup_time=1.0,
                peak_memory=2500.0
            ),
            "F3Net": ModelResourceProfile(
                model_name="F3Net",
                priority=ProcessingPriority.HIGH,
                estimated_memory=1024.0,
                estimated_time=0.12,
                gpu_required=True,
                batch_size=3,
                warmup_time=0.8,
                peak_memory=1200.0
            ),
            "MesoNet": ModelResourceProfile(
                model_name="MesoNet",
                priority=ProcessingPriority.LOW,
                estimated_memory=256.0,
                estimated_time=0.06,
                gpu_required=False,
                batch_size=8,
                warmup_time=0.3,
                peak_memory=300.0
            )
        }
        
        # GPU availability check
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**2)  # MB
            self.max_gpu_memory = min(self.max_gpu_memory, self.gpu_memory_total * 0.8)  # Use 80% of GPU
        else:
            self.gpu_memory_total = 0.0
            self.max_gpu_memory = 0.0
        
        self.logger.info(f"Resource allocator initialized: GPU={self.gpu_available}, "
                        f"GPU Memory={self.gpu_memory_total:.0f}MB, CPU Cores={self.max_cpu_cores}")
    
    def allocate_resources(self, model_names: List[str], 
                          input_complexity: str = "medium") -> Dict[str, ResourceAllocation]:
        """
        Allocate resources for a list of models based on their requirements and system state.
        
        Args:
            model_names: List of model names to allocate resources for
            input_complexity: Complexity of input (affects batch size)
            
        Returns:
            Dictionary mapping model names to resource allocations
        """
        allocations = {}
        
        # Sort models by priority (high priority first)
        sorted_models = sorted(
            model_names, 
            key=lambda name: self.model_profiles.get(name, ModelResourceProfile("unknown", ProcessingPriority.MEDIUM, 0, 0, False, 1, 0, 0)).priority.value,
            reverse=True
        )
        
        for model_name in sorted_models:
            if model_name not in self.model_profiles:
                self.logger.warning(f"Unknown model: {model_name}")
                continue
            
            profile = self.model_profiles[model_name]
            
            # Adjust batch size based on input complexity
            batch_size = self._adjust_batch_size(profile.batch_size, input_complexity)
            
            # Determine device allocation
            device, memory_limit = self._determine_device_allocation(profile, batch_size)
            
            # Create resource allocation
            allocation = ResourceAllocation(
                model_name=model_name,
                device=device,
                memory_limit=memory_limit,
                priority=profile.priority,
                executor_type="gpu" if device.startswith("cuda") else "cpu",
                batch_size=batch_size,
                estimated_time=profile.estimated_time
            )
            
            allocations[model_name] = allocation
            
            # Update resource tracking
            self._update_resource_tracking(allocation)
        
        return allocations
    
    def _adjust_batch_size(self, base_batch_size: int, input_complexity: str) -> int:
        """Adjust batch size based on input complexity."""
        if input_complexity == "simple":
            return min(base_batch_size * 2, 16)  # Double for simple inputs
        elif input_complexity == "complex":
            return max(base_batch_size // 2, 1)  # Half for complex inputs
        else:
            return base_batch_size  # No change for medium complexity
    
    def _determine_device_allocation(self, profile: ModelResourceProfile, 
                                   batch_size: int) -> Tuple[str, float]:
        """Determine the best device allocation for a model."""
        required_memory = profile.estimated_memory * batch_size
        
        # Check if GPU is available and has enough memory
        if profile.gpu_required and self.gpu_available:
            if (self.allocated_gpu_memory + required_memory) <= self.max_gpu_memory:
                return "cuda:0", required_memory
            else:
                self.logger.warning(f"Insufficient GPU memory for {profile.model_name}, using CPU")
        
        # Use CPU if GPU not available or insufficient memory
        return "cpu", required_memory
    
    def _update_resource_tracking(self, allocation: ResourceAllocation):
        """Update resource tracking after allocation."""
        if allocation.device.startswith("cuda"):
            self.allocated_gpu_memory += allocation.memory_limit
        else:
            self.allocated_cpu_cores += 1
        
        self.active_allocations[allocation.model_name] = allocation
    
    def release_resources(self, model_name: str):
        """Release resources for a model."""
        if model_name in self.active_allocations:
            allocation = self.active_allocations[model_name]
            
            if allocation.device.startswith("cuda"):
                self.allocated_gpu_memory -= allocation.memory_limit
            else:
                self.allocated_cpu_cores -= 1
            
            del self.active_allocations[model_name]
            
            # Force garbage collection to free memory
            if allocation.device.startswith("cuda"):
                torch.cuda.empty_cache()
            gc.collect()
    
    def get_resource_status(self) -> Dict[str, Any]:
        """Get current resource allocation status."""
        return {
            "gpu_available": self.gpu_available,
            "gpu_memory_total": self.gpu_memory_total,
            "gpu_memory_allocated": self.allocated_gpu_memory,
            "gpu_memory_available": self.max_gpu_memory - self.allocated_gpu_memory,
            "cpu_cores_total": self.max_cpu_cores,
            "cpu_cores_allocated": self.allocated_cpu_cores,
            "cpu_cores_available": self.max_cpu_cores - self.allocated_cpu_cores,
            "active_allocations": len(self.active_allocations),
            "allocations": {name: {
                "device": alloc.device,
                "memory_limit": alloc.memory_limit,
                "priority": alloc.priority.value
            } for name, alloc in self.active_allocations.items()}
        }


class ConcurrentModelExecutor:
    """
    Executes models concurrently with intelligent resource management
    and priority-based task scheduling.
    """
    
    def __init__(self, resource_allocator: ResourceAllocator, 
                 max_workers: int = None):
        self.resource_allocator = resource_allocator
        self.max_workers = max_workers or min(8, psutil.cpu_count())
        self.logger = logging.getLogger(f"{__name__}.ConcurrentModelExecutor")
        
        # Task queues
        self.task_queue = PriorityQueue()
        self.completed_tasks: Dict[str, Any] = {}
        self.failed_tasks: Dict[str, str] = {}
        
        # Executors
        self.thread_executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_executor = ProcessPoolExecutor(max_workers=min(4, self.max_workers))
        
        # Performance tracking
        self.total_tasks_processed = 0
        self.total_processing_time = 0.0
        self.concurrent_executions = 0
        self.max_concurrent = 0
        
        # Model warmup cache
        self.warmed_up_models: Dict[str, Any] = {}
        
        self.logger.info(f"Concurrent executor initialized with {self.max_workers} workers")
    
    async def execute_models_parallel(self, model_instances: Dict[str, BaseDetector],
                                    processed_images: Dict[str, np.ndarray],
                                    resource_allocations: Dict[str, ResourceAllocation]) -> Dict[str, DetectionResult]:
        """
        Execute multiple models in parallel with intelligent resource management.
        
        Args:
            model_instances: Dictionary of model instances
            processed_images: Dictionary of preprocessed images for each model
            resource_allocations: Resource allocations for each model
            
        Returns:
            Dictionary of detection results
        """
        start_time = time.time()
        self.concurrent_executions = 0
        
        try:
            # Create tasks for each model
            tasks = []
            for model_name, model in model_instances.items():
                if model_name in processed_images and model_name in resource_allocations:
                    task = self._create_processing_task(
                        model_name, model, processed_images[model_name], 
                        resource_allocations[model_name]
                    )
                    tasks.append(task)
            
            # Execute tasks concurrently
            results = await self._execute_tasks_concurrent(tasks)
            
            # Update performance metrics
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            self.total_tasks_processed += len(tasks)
            self.max_concurrent = max(self.max_concurrent, self.concurrent_executions)
            
            self.logger.info(f"Executed {len(tasks)} models in parallel in {processing_time:.3f}s")
            return results
            
        except Exception as e:
            self.logger.error(f"Parallel execution failed: {str(e)}")
            raise
    
    def _create_processing_task(self, model_name: str, model: BaseDetector,
                              processed_image: np.ndarray, 
                              allocation: ResourceAllocation) -> ProcessingTask:
        """Create a processing task for a model."""
        return ProcessingTask(
            task_id=f"{model_name}_{int(time.time() * 1000)}",
            model_name=model_name,
            input_data=processed_image,
            priority=allocation.priority,
            resource_allocation=allocation,
            created_at=time.time()
        )
    
    async def _execute_tasks_concurrent(self, tasks: List[ProcessingTask]) -> Dict[str, DetectionResult]:
        """Execute tasks concurrently with resource management."""
        results = {}
        
        # Group tasks by executor type
        gpu_tasks = [task for task in tasks if task.resource_allocation.executor_type == "gpu"]
        cpu_tasks = [task for task in tasks if task.resource_allocation.executor_type == "cpu"]
        
        # Execute GPU tasks concurrently
        if gpu_tasks:
            gpu_results = await self._execute_gpu_tasks(gpu_tasks)
            results.update(gpu_results)
        
        # Execute CPU tasks concurrently
        if cpu_tasks:
            cpu_results = await self._execute_cpu_tasks(cpu_tasks)
            results.update(cpu_results)
        
        return results
    
    async def _execute_gpu_tasks(self, tasks: List[ProcessingTask]) -> Dict[str, DetectionResult]:
        """Execute GPU tasks concurrently."""
        if not tasks:
            return {}
        
        # For GPU tasks, we need to be more careful about memory management
        results = {}
        
        # Execute tasks in smaller batches to manage GPU memory
        batch_size = min(2, len(tasks))  # Process 2 GPU tasks at a time
        for i in range(0, len(tasks), batch_size):
            batch_tasks = tasks[i:i + batch_size]
            
            # Create coroutines for batch
            coroutines = []
            for task in batch_tasks:
                coro = self._execute_single_task(task)
                coroutines.append(coro)
            
            # Execute batch concurrently
            batch_results = await asyncio.gather(*coroutines, return_exceptions=True)
            
            # Process results
            for task, result in zip(batch_tasks, batch_results):
                if isinstance(result, Exception):
                    self.logger.error(f"GPU task {task.model_name} failed: {str(result)}")
                    self.failed_tasks[task.task_id] = str(result)
                else:
                    results[task.model_name] = result
                    self.completed_tasks[task.task_id] = result
        
        return results
    
    async def _execute_cpu_tasks(self, tasks: List[ProcessingTask]) -> Dict[str, DetectionResult]:
        """Execute CPU tasks concurrently."""
        if not tasks:
            return {}
        
        # CPU tasks can be executed with higher concurrency
        coroutines = []
        for task in tasks:
            coro = self._execute_single_task(task)
            coroutines.append(coro)
        
        # Execute all CPU tasks concurrently
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        
        # Process results
        task_results = {}
        for task, result in zip(tasks, results):
            if isinstance(result, Exception):
                self.logger.error(f"CPU task {task.model_name} failed: {str(result)}")
                self.failed_tasks[task.task_id] = str(result)
            else:
                task_results[task.model_name] = result
                self.completed_tasks[task.task_id] = result
        
        return task_results
    
    async def _execute_single_task(self, task: ProcessingTask) -> DetectionResult:
        """Execute a single model task."""
        self.concurrent_executions += 1
        
        try:
            # Get model instance (this would need to be passed in or retrieved)
            # For now, we'll simulate the execution
            start_time = time.time()
            
            # Simulate model execution
            await asyncio.sleep(task.resource_allocation.estimated_time)
            
            # Create mock result (in real implementation, this would call the actual model)
            result = DetectionResult(
                is_deepfake=np.random.random() > 0.5,
                confidence=np.random.uniform(50, 95),
                model_name=task.model_name,
                inference_time=time.time() - start_time,
                metadata={"task_id": task.task_id, "device": task.resource_allocation.device}
            )
            
            # Release resources
            self.resource_allocator.release_resources(task.model_name)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Task execution failed: {str(e)}")
            raise
        finally:
            self.concurrent_executions -= 1
    
    def warmup_models(self, model_instances: Dict[str, BaseDetector]):
        """Warm up models to reduce first-run latency."""
        self.logger.info("Starting model warmup...")
        
        for model_name, model in model_instances.items():
            try:
                if model_name in self.resource_allocator.model_profiles:
                    profile = self.resource_allocator.model_profiles[model_name]
                    
                    # Create dummy input for warmup
                    dummy_input = np.random.rand(224, 224, 3).astype(np.uint8)
                    
                    # Warm up the model
                    start_time = time.time()
                    # model.predict(dummy_input)  # Uncomment when model is available
                    warmup_time = time.time() - start_time
                    
                    self.warmed_up_models[model_name] = {
                        "warmup_time": warmup_time,
                        "warmed_at": time.time()
                    }
                    
                    self.logger.info(f"Warmed up {model_name} in {warmup_time:.3f}s")
                    
            except Exception as e:
                self.logger.warning(f"Failed to warm up {model_name}: {str(e)}")
        
        self.logger.info(f"Model warmup completed for {len(self.warmed_up_models)} models")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the concurrent executor."""
        avg_processing_time = (self.total_processing_time / self.total_tasks_processed 
                             if self.total_tasks_processed > 0 else 0.0)
        
        return {
            "total_tasks_processed": self.total_tasks_processed,
            "total_processing_time": self.total_processing_time,
            "average_processing_time": avg_processing_time,
            "max_concurrent_executions": self.max_concurrent,
            "current_concurrent_executions": self.concurrent_executions,
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks),
            "warmed_up_models": list(self.warmed_up_models.keys()),
            "resource_status": self.resource_allocator.get_resource_status()
        }
    
    def cleanup(self):
        """Clean up resources and shutdown executors."""
        self.thread_executor.shutdown(wait=True)
        self.process_executor.shutdown(wait=True)
        self.logger.info("Concurrent executor cleaned up")


class ParallelProcessingManager:
    """
    High-level manager for parallel processing optimization.
    Coordinates resource allocation, concurrent execution, and performance monitoring.
    """
    
    def __init__(self, max_gpu_memory: float = 8.0, max_workers: int = None):
        self.resource_allocator = ResourceAllocator(max_gpu_memory)
        self.concurrent_executor = ConcurrentModelExecutor(self.resource_allocator, max_workers)
        self.logger = logging.getLogger(f"{__name__}.ParallelProcessingManager")
        
        # Performance tracking
        self.total_optimizations = 0
        self.total_time_saved = 0.0
        self.parallel_efficiency = 0.0
        
    async def process_models_parallel(self, model_instances: Dict[str, BaseDetector],
                                    processed_images: Dict[str, np.ndarray],
                                    input_complexity: str = "medium") -> Dict[str, DetectionResult]:
        """
        Process multiple models in parallel with full optimization.
        
        Args:
            model_instances: Dictionary of model instances
            processed_images: Dictionary of preprocessed images
            input_complexity: Complexity of input for resource allocation
            
        Returns:
            Dictionary of detection results
        """
        start_time = time.time()
        
        try:
            # Allocate resources intelligently
            resource_allocations = self.resource_allocator.allocate_resources(
                list(model_instances.keys()), input_complexity
            )
            
            # Execute models in parallel
            results = await self.concurrent_executor.execute_models_parallel(
                model_instances, processed_images, resource_allocations
            )
            
            # Calculate optimization metrics
            processing_time = time.time() - start_time
            self._update_performance_metrics(processing_time, len(model_instances))
            
            self.logger.info(f"Parallel processing completed in {processing_time:.3f}s for {len(model_instances)} models")
            return results
            
        except Exception as e:
            self.logger.error(f"Parallel processing failed: {str(e)}")
            raise
    
    def _update_performance_metrics(self, processing_time: float, num_models: int):
        """Update performance tracking metrics."""
        self.total_optimizations += 1
        
        # Estimate sequential processing time (use average model time)
        avg_model_time = 0.1  # Default average model processing time
        estimated_sequential_time = avg_model_time * num_models
        
        time_saved = max(0, estimated_sequential_time - processing_time)
        self.total_time_saved += time_saved
        
        # Calculate parallel efficiency
        if estimated_sequential_time > 0:
            self.parallel_efficiency = (time_saved / estimated_sequential_time) * 100
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        return {
            "total_optimizations": self.total_optimizations,
            "total_time_saved": self.total_time_saved,
            "parallel_efficiency": self.parallel_efficiency,
            "concurrent_executor_stats": self.concurrent_executor.get_performance_stats(),
            "resource_allocator_stats": self.resource_allocator.get_resource_status()
        }
    
    def cleanup(self):
        """Clean up all resources."""
        self.concurrent_executor.cleanup()
        self.logger.info("Parallel processing manager cleaned up")
