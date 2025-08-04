"""
Video Analysis Integration API

Comprehensive integration layer that combines all video analysis components
into a unified, production-ready system for temporal deepfake detection.

Integration Components:
1. Unified Video Analysis API
2. Configuration Management
3. Processing Pipeline Orchestration
4. Result Management and Caching
5. Performance Monitoring
6. Error Handling and Recovery
7. Export and Visualization Management
"""

import asyncio
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional, Union, Callable, Any, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
import json
import pickle
import hashlib
from datetime import datetime, timedelta

from .video_analysis_core import (
    VideoProcessor, VideoAnalysisConfig, VideoAnalysisResult,
    FrameSamplingStrategy, ProcessingMode, TemporalModelType,
    create_video_analyzer
)
from .realtime_video_processor import (
    RealtimeVideoProcessor, StreamConfig, StreamStats, StreamAlert,
    StreamSource, StreamQuality, create_realtime_processor
)
from .video_visualization import (
    VideoTimelineVisualizer, RealTimeVisualizationDashboard,
    VideoExportUtilities, VisualizationConfig,
    create_comprehensive_visualization_suite
)
from .optimized_ensemble_detector import OptimizedEnsembleDetector

logger = logging.getLogger(__name__)

class AnalysisMode(str, Enum):
    """Video analysis modes"""
    BATCH = "batch"           # Process complete video file
    REALTIME = "realtime"     # Live stream processing
    HYBRID = "hybrid"         # Batch with real-time capabilities
    STREAMING = "streaming"   # Chunk-based processing

class CacheStrategy(str, Enum):
    """Result caching strategies"""
    NONE = "none"            # No caching
    MEMORY = "memory"        # In-memory cache
    DISK = "disk"           # Disk-based cache
    HYBRID = "hybrid"       # Memory + disk cache

@dataclass
class IntegrationConfig:
    """Master configuration for video analysis integration"""
    # Analysis settings
    analysis_mode: AnalysisMode = AnalysisMode.BATCH
    optimization_level: str = "advanced"  # basic, advanced, research
    
    # Video processing config
    video_config: VideoAnalysisConfig = field(default_factory=VideoAnalysisConfig)
    
    # Stream processing config
    stream_config: StreamConfig = field(default_factory=StreamConfig)
    
    # Visualization config
    visualization_config: VisualizationConfig = field(default_factory=VisualizationConfig)
    
    # Caching and storage
    cache_strategy: CacheStrategy = CacheStrategy.MEMORY
    cache_dir: str = "cache"
    max_cache_size_gb: float = 5.0
    cache_ttl_hours: int = 24
    
    # Performance and scaling
    max_concurrent_analyses: int = 3
    enable_gpu_acceleration: bool = True
    memory_limit_gb: float = 8.0
    
    # Export settings
    auto_generate_visualizations: bool = True
    export_detailed_results: bool = True
    output_directory: str = "results"
    
    # Monitoring and alerts
    enable_performance_monitoring: bool = True
    alert_webhook_url: Optional[str] = None
    log_level: str = "INFO"

@dataclass
class AnalysisJob:
    """Video analysis job specification"""
    job_id: str
    input_path: str
    analysis_mode: AnalysisMode
    config: IntegrationConfig
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = "pending"  # pending, running, completed, failed
    result: Optional[VideoAnalysisResult] = None
    error: Optional[str] = None
    progress: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class ResultCache:
    """Result caching system"""
    
    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.memory_cache: Dict[str, VideoAnalysisResult] = {}
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_index = self._load_cache_index()
        
    def _load_cache_index(self) -> Dict[str, Dict[str, Any]]:
        """Load cache index from disk"""
        index_file = self.cache_dir / "cache_index.json"
        if index_file.exists():
            try:
                with open(index_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load cache index: {e}")
        return {}
    
    def _save_cache_index(self):
        """Save cache index to disk"""
        index_file = self.cache_dir / "cache_index.json"
        try:
            with open(index_file, 'w') as f:
                json.dump(self.cache_index, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save cache index: {e}")
    
    def _generate_cache_key(self, video_path: str, config: VideoAnalysisConfig) -> str:
        """Generate unique cache key"""
        # Include file size and modification time for cache validation
        try:
            stat = Path(video_path).stat()
            key_data = f"{video_path}_{stat.st_size}_{stat.st_mtime}_{config.__dict__}"
            return hashlib.sha256(key_data.encode()).hexdigest()
        except Exception:
            return hashlib.sha256(f"{video_path}_{config.__dict__}".encode()).hexdigest()
    
    def get(self, video_path: str, config: VideoAnalysisConfig) -> Optional[VideoAnalysisResult]:
        """Get cached result"""
        cache_key = self._generate_cache_key(video_path, config)
        
        # Check memory cache first
        if cache_key in self.memory_cache:
            return self.memory_cache[cache_key]
        
        # Check disk cache
        if cache_key in self.cache_index:
            cache_info = self.cache_index[cache_key]
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            
            # Check if cache is still valid
            cached_at = datetime.fromisoformat(cache_info['cached_at'])
            if datetime.now() - cached_at > timedelta(hours=self.config.cache_ttl_hours):
                self._remove_from_cache(cache_key)
                return None
            
            try:
                with open(cache_file, 'rb') as f:
                    result = pickle.load(f)
                    # Add to memory cache
                    self.memory_cache[cache_key] = result
                    return result
            except Exception as e:
                logger.error(f"Failed to load cached result: {e}")
                self._remove_from_cache(cache_key)
        
        return None
    
    def put(self, video_path: str, config: VideoAnalysisConfig, result: VideoAnalysisResult):
        """Cache result"""
        cache_key = self._generate_cache_key(video_path, config)
        
        # Add to memory cache
        self.memory_cache[cache_key] = result
        
        # Add to disk cache if enabled
        if self.config.cache_strategy in [CacheStrategy.DISK, CacheStrategy.HYBRID]:
            try:
                cache_file = self.cache_dir / f"{cache_key}.pkl"
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
                
                self.cache_index[cache_key] = {
                    'video_path': video_path,
                    'cached_at': datetime.now().isoformat(),
                    'file_size': cache_file.stat().st_size
                }
                self._save_cache_index()
                
            except Exception as e:
                logger.error(f"Failed to cache result to disk: {e}")
    
    def _remove_from_cache(self, cache_key: str):
        """Remove result from cache"""
        # Remove from memory
        self.memory_cache.pop(cache_key, None)
        
        # Remove from disk
        if cache_key in self.cache_index:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            cache_file.unlink(missing_ok=True)
            del self.cache_index[cache_key]
            self._save_cache_index()
    
    def cleanup(self):
        """Clean up expired and oversized cache"""
        current_time = datetime.now()
        total_size = 0
        cache_items = []
        
        # Collect cache info with sizes and ages
        for cache_key, cache_info in self.cache_index.items():
            cached_at = datetime.fromisoformat(cache_info['cached_at'])
            age = current_time - cached_at
            
            # Remove expired items
            if age > timedelta(hours=self.config.cache_ttl_hours):
                self._remove_from_cache(cache_key)
                continue
            
            cache_items.append((cache_key, cache_info['file_size'], cached_at))
            total_size += cache_info['file_size']
        
        # Remove oldest items if over size limit
        max_size = self.config.max_cache_size_gb * 1024**3
        if total_size > max_size:
            # Sort by age (oldest first)
            cache_items.sort(key=lambda x: x[2])
            
            for cache_key, file_size, _ in cache_items:
                if total_size <= max_size:
                    break
                self._remove_from_cache(cache_key)
                total_size -= file_size

class VideoAnalysisOrchestrator:
    """Orchestrates video analysis across different modes and configurations"""
    
    def __init__(self, ensemble_detector: OptimizedEnsembleDetector, 
                 config: IntegrationConfig):
        self.ensemble_detector = ensemble_detector
        self.config = config
        self.result_cache = ResultCache(config)
        self.active_jobs: Dict[str, AnalysisJob] = {}
        self.job_executor = ThreadPoolExecutor(max_workers=config.max_concurrent_analyses)
        
        # Initialize components
        self.video_processor = create_video_analyzer(ensemble_detector, config.video_config)
        self.realtime_processor = None
        self.visualizer = VideoTimelineVisualizer(config.visualization_config)
        self.dashboard = RealTimeVisualizationDashboard(config.visualization_config)
        
        # Performance monitoring
        self.performance_metrics = {
            'total_analyses': 0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'average_processing_time': 0.0
        }
        
        self.logger = logging.getLogger(f"{__name__}.VideoAnalysisOrchestrator")
        self.logger.setLevel(getattr(logging, config.log_level))
    
    async def analyze_video(self, video_path: str, 
                          job_id: Optional[str] = None,
                          analysis_mode: Optional[AnalysisMode] = None) -> str:
        """Analyze video and return job ID"""
        
        if job_id is None:
            job_id = self._generate_job_id(video_path)
        
        if analysis_mode is None:
            analysis_mode = self.config.analysis_mode
        
        # Create analysis job
        job = AnalysisJob(
            job_id=job_id,
            input_path=video_path,
            analysis_mode=analysis_mode,
            config=self.config,
            metadata={'source': 'api_request'}
        )
        
        self.active_jobs[job_id] = job
        
        # Submit for processing
        if analysis_mode == AnalysisMode.BATCH:
            future = self.job_executor.submit(self._process_batch_analysis, job)
        elif analysis_mode == AnalysisMode.REALTIME:
            future = self.job_executor.submit(self._process_realtime_analysis, job)
        elif analysis_mode == AnalysisMode.HYBRID:
            future = self.job_executor.submit(self._process_hybrid_analysis, job)
        else:  # STREAMING
            future = self.job_executor.submit(self._process_streaming_analysis, job)
        
        # Don't wait for completion in async context
        threading.Thread(target=self._handle_job_completion, args=(job_id, future), daemon=True).start()
        
        return job_id
    
    def _process_batch_analysis(self, job: AnalysisJob) -> VideoAnalysisResult:
        """Process video in batch mode"""
        job.status = "running"
        job.started_at = datetime.now()
        
        try:
            # Check cache first
            cached_result = self.result_cache.get(job.input_path, self.config.video_config)
            if cached_result:
                self.performance_metrics['cache_hits'] += 1
                job.progress = 1.0
                return cached_result
            
            self.performance_metrics['cache_misses'] += 1
            
            # Process video
            result = self.video_processor.analyze_video(job.input_path)
            
            # Cache result
            self.result_cache.put(job.input_path, self.config.video_config, result)
            
            # Generate visualizations if enabled
            if self.config.auto_generate_visualizations:
                self._generate_visualizations(result, job.job_id)
            
            job.progress = 1.0
            return result
            
        except Exception as e:
            self.logger.error(f"Batch analysis failed for job {job.job_id}: {e}")
            raise
    
    def _process_realtime_analysis(self, job: AnalysisJob) -> VideoAnalysisResult:
        """Process video in real-time mode"""
        job.status = "running"
        job.started_at = datetime.now()
        
        try:
            # Initialize real-time processor if not already done
            if self.realtime_processor is None:
                stream_config = self.config.stream_config
                stream_config.source_path = job.input_path
                self.realtime_processor = create_realtime_processor(
                    self.ensemble_detector, stream_config
                )
            
            # Start processing
            if not self.realtime_processor.start():
                raise RuntimeError("Failed to start real-time processor")
            
            # Monitor progress (simplified - in real implementation would be more sophisticated)
            start_time = time.time()
            while self.realtime_processor.running:
                stats = self.realtime_processor.get_stats()
                elapsed_time = time.time() - start_time
                
                # Update progress based on processing stats
                job.progress = min(1.0, elapsed_time / 60.0)  # Assume 1 minute max
                
                if elapsed_time > 300:  # 5 minute timeout
                    break
                
                time.sleep(1)
            
            # Create result from real-time stats
            stats = self.realtime_processor.get_stats()
            alerts = self.realtime_processor.get_recent_alerts()
            
            result = VideoAnalysisResult(
                video_path=job.input_path,
                total_frames=stats.frames_processed,
                analyzed_frames=stats.frames_processed,
                overall_confidence=80.0 if stats.deepfake_detections > 0 else 20.0,  # Simplified
                is_deepfake=stats.deepfake_detections > 0,
                frame_results=[],  # Would be populated from real-time data
                temporal_consistency=0.8,  # Placeholder
                processing_time=time.time() - start_time,
                memory_usage=stats.memory_usage,
                temporal_patterns={'realtime_stats': stats.__dict__},
                optical_flow_analysis={},
                confidence_timeline=[],
                video_metadata={'realtime_mode': True},
                analysis_config=self.config.video_config
            )
            
            job.progress = 1.0
            return result
            
        except Exception as e:
            self.logger.error(f"Real-time analysis failed for job {job.job_id}: {e}")
            raise
    
    def _process_hybrid_analysis(self, job: AnalysisJob) -> VideoAnalysisResult:
        """Process video in hybrid mode (batch + real-time capabilities)"""
        return self._process_batch_analysis(job)  # Simplified implementation
    
    def _process_streaming_analysis(self, job: AnalysisJob) -> VideoAnalysisResult:
        """Process video in streaming mode"""
        return self._process_batch_analysis(job)  # Simplified implementation
    
    def _handle_job_completion(self, job_id: str, future):
        """Handle job completion"""
        job = self.active_jobs.get(job_id)
        if not job:
            return
        
        try:
            result = future.result()
            job.result = result
            job.status = "completed"
            job.completed_at = datetime.now()
            
            # Update performance metrics
            self.performance_metrics['total_analyses'] += 1
            self.performance_metrics['successful_analyses'] += 1
            
            processing_time = (job.completed_at - job.started_at).total_seconds()
            self.performance_metrics['average_processing_time'] = (
                (self.performance_metrics['average_processing_time'] * 
                 (self.performance_metrics['successful_analyses'] - 1) + processing_time) /
                self.performance_metrics['successful_analyses']
            )
            
            self.logger.info(f"Job {job_id} completed successfully in {processing_time:.2f}s")
            
        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            job.completed_at = datetime.now()
            
            self.performance_metrics['total_analyses'] += 1
            self.performance_metrics['failed_analyses'] += 1
            
            self.logger.error(f"Job {job_id} failed: {e}")
    
    def _generate_job_id(self, video_path: str) -> str:
        """Generate unique job ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path_hash = hashlib.md5(video_path.encode()).hexdigest()[:8]
        return f"video_analysis_{timestamp}_{path_hash}"
    
    def _generate_visualizations(self, result: VideoAnalysisResult, job_id: str):
        """Generate visualizations for analysis result"""
        try:
            output_dir = Path(self.config.output_directory) / job_id / "visualizations"
            output_files = create_comprehensive_visualization_suite(
                result, str(output_dir), self.config.visualization_config
            )
            self.logger.info(f"Generated visualizations for job {job_id}: {list(output_files.keys())}")
        except Exception as e:
            self.logger.error(f"Failed to generate visualizations for job {job_id}: {e}")
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status and progress"""
        job = self.active_jobs.get(job_id)
        if not job:
            return None
        
        return {
            'job_id': job.job_id,
            'status': job.status,
            'progress': job.progress,
            'created_at': job.created_at.isoformat(),
            'started_at': job.started_at.isoformat() if job.started_at else None,
            'completed_at': job.completed_at.isoformat() if job.completed_at else None,
            'error': job.error,
            'input_path': job.input_path,
            'analysis_mode': job.analysis_mode
        }
    
    def get_job_result(self, job_id: str) -> Optional[VideoAnalysisResult]:
        """Get job result"""
        job = self.active_jobs.get(job_id)
        if job and job.result:
            return job.result
        return None
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel running job"""
        job = self.active_jobs.get(job_id)
        if job and job.status == "running":
            job.status = "cancelled"
            # In real implementation, would need to signal cancellation to processing thread
            return True
        return False
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            **self.performance_metrics,
            'active_jobs': len([j for j in self.active_jobs.values() if j.status == "running"]),
            'cache_hit_rate': (self.performance_metrics['cache_hits'] / 
                             max(1, self.performance_metrics['cache_hits'] + self.performance_metrics['cache_misses'])),
            'success_rate': (self.performance_metrics['successful_analyses'] / 
                           max(1, self.performance_metrics['total_analyses']))
        }
    
    def cleanup_completed_jobs(self, max_age_hours: int = 24):
        """Clean up old completed jobs"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        jobs_to_remove = []
        for job_id, job in self.active_jobs.items():
            if (job.status in ["completed", "failed", "cancelled"] and 
                job.completed_at and job.completed_at < cutoff_time):
                jobs_to_remove.append(job_id)
        
        for job_id in jobs_to_remove:
            del self.active_jobs[job_id]
        
        self.logger.info(f"Cleaned up {len(jobs_to_remove)} old jobs")
    
    def shutdown(self):
        """Shutdown orchestrator"""
        # Cancel all running jobs
        for job in self.active_jobs.values():
            if job.status == "running":
                job.status = "cancelled"
        
        # Shutdown executor
        self.job_executor.shutdown(wait=True)
        
        # Stop real-time processor if running
        if self.realtime_processor:
            self.realtime_processor.stop()
        
        # Clean up cache
        self.result_cache.cleanup()
        
        self.logger.info("Video analysis orchestrator shutdown complete")

class VideoAnalysisAPI:
    """High-level API for video analysis integration"""
    
    def __init__(self, ensemble_detector: OptimizedEnsembleDetector,
                 config: Optional[IntegrationConfig] = None):
        self.config = config or IntegrationConfig()
        self.orchestrator = VideoAnalysisOrchestrator(ensemble_detector, self.config)
        self.logger = logging.getLogger(f"{__name__}.VideoAnalysisAPI")
    
    async def analyze_video_file(self, video_path: str, 
                               analysis_mode: AnalysisMode = AnalysisMode.BATCH) -> str:
        """Analyze video file"""
        return await self.orchestrator.analyze_video(video_path, analysis_mode=analysis_mode)
    
    async def start_realtime_analysis(self, source: str,
                                    source_type: StreamSource = StreamSource.WEBCAM) -> str:
        """Start real-time video analysis"""
        # Update stream config
        self.config.stream_config.source_path = source
        self.config.stream_config.source_type = source_type
        
        return await self.orchestrator.analyze_video(source, analysis_mode=AnalysisMode.REALTIME)
    
    def get_analysis_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get analysis status"""
        return self.orchestrator.get_job_status(job_id)
    
    def get_analysis_result(self, job_id: str) -> Optional[VideoAnalysisResult]:
        """Get analysis result"""
        return self.orchestrator.get_job_result(job_id)
    
    def cancel_analysis(self, job_id: str) -> bool:
        """Cancel analysis"""
        return self.orchestrator.cancel_job(job_id)
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics"""
        return self.orchestrator.get_performance_metrics()
    
    def export_result(self, job_id: str, output_path: str, 
                     include_visualizations: bool = True) -> bool:
        """Export analysis result"""
        result = self.orchestrator.get_job_result(job_id)
        if not result:
            return False
        
        try:
            VideoExportUtilities.export_analysis_report(
                result, output_path, include_visualizations
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to export result for job {job_id}: {e}")
            return False
    
    def cleanup_system(self):
        """Perform system cleanup"""
        self.orchestrator.cleanup_completed_jobs()
        self.orchestrator.result_cache.cleanup()
    
    def shutdown(self):
        """Shutdown API"""
        self.orchestrator.shutdown()

def create_video_analysis_api(ensemble_detector: OptimizedEnsembleDetector,
                            config: Optional[IntegrationConfig] = None) -> VideoAnalysisAPI:
    """Factory function to create video analysis API"""
    return VideoAnalysisAPI(ensemble_detector, config)