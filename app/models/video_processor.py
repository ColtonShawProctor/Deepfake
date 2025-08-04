#!/usr/bin/env python3
"""
Advanced Video Processing System for Deepfake Detection

Features:
- Optimized frame extraction with intelligent sampling
- Batch processing for multiple frames
- Temporal consistency analysis
- Memory-optimized processing
- Progress tracking and resource management
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Generator
from pathlib import Path
import logging
import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue
import gc

logger = logging.getLogger(__name__)

@dataclass
class VideoMetadata:
    """Video metadata and properties"""
    path: str
    fps: float
    frame_count: int
    duration: float
    width: int
    height: int
    codec: str
    bitrate: int

@dataclass
class FrameAnalysis:
    """Individual frame analysis result"""
    frame_number: int
    timestamp: float
    confidence_score: float
    is_deepfake: bool
    analysis_metadata: Dict
    processing_time: float

@dataclass
class VideoAnalysisResult:
    """Complete video analysis result"""
    video_path: str
    total_frames: int
    analyzed_frames: int
    overall_confidence: float
    is_deepfake: bool
    temporal_consistency: float
    frame_analyses: List[FrameAnalysis]
    processing_time: float
    analysis_metadata: Dict

class VideoFrameExtractor:
    """Optimized video frame extraction with intelligent sampling"""
    
    def __init__(self, max_frames: int = 100, sampling_strategy: str = "uniform"):
        self.max_frames = max_frames
        self.sampling_strategy = sampling_strategy
        
    def extract_frames(self, video_path: str) -> Generator[Tuple[int, np.ndarray, float], None, None]:
        """Extract frames from video with optimized sampling"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            # Determine frame indices based on sampling strategy
            frame_indices = self._get_frame_indices(total_frames)
            
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    timestamp = frame_idx / fps if fps > 0 else 0
                    yield frame_idx, frame, timestamp
                    
        finally:
            cap.release()
    
    def _get_frame_indices(self, total_frames: int) -> List[int]:
        """Get frame indices based on sampling strategy"""
        if self.sampling_strategy == "uniform":
            # Uniform sampling across video
            step = max(1, total_frames // self.max_frames)
            return list(range(0, total_frames, step))[:self.max_frames]
        
        elif self.sampling_strategy == "adaptive":
            # Adaptive sampling - more frames at beginning and end
            indices = []
            # First 20% of frames
            start_frames = int(0.2 * self.max_frames)
            indices.extend(range(0, total_frames // 5, max(1, (total_frames // 5) // start_frames)))
            
            # Middle 60% of frames
            middle_frames = int(0.6 * self.max_frames)
            step = max(1, (total_frames * 3 // 5) // middle_frames)
            indices.extend(range(total_frames // 5, 4 * total_frames // 5, step))
            
            # Last 20% of frames
            end_frames = self.max_frames - len(indices)
            indices.extend(range(4 * total_frames // 5, total_frames, max(1, (total_frames // 5) // end_frames)))
            
            return sorted(indices[:self.max_frames])
        
        elif self.sampling_strategy == "scene_based":
            # Scene-based sampling (simplified)
            return list(range(0, total_frames, max(1, total_frames // self.max_frames)))[:self.max_frames]
        
        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")

class BatchProcessor:
    """Batch processing for multiple frames"""
    
    def __init__(self, batch_size: int = 8, max_workers: int = 4):
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def process_batch(self, frames: List[np.ndarray], detector) -> List[Dict]:
        """Process a batch of frames"""
        if not frames:
            return []
        
        # Preprocess frames
        processed_frames = self._preprocess_frames(frames)
        
        # Convert to tensor batch
        batch_tensor = torch.stack(processed_frames).to(self.device)
        
        # Process with detector
        with torch.no_grad():
            results = detector.analyze_batch(batch_tensor)
        
        return results
    
    def _preprocess_frames(self, frames: List[np.ndarray]) -> List[torch.Tensor]:
        """Preprocess frames for batch processing"""
        processed = []
        
        for frame in frames:
            # Resize to standard size
            frame_resized = cv2.resize(frame, (224, 224))
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            # Normalize
            frame_normalized = frame_rgb.astype(np.float32) / 255.0
            
            # Convert to tensor
            frame_tensor = torch.from_numpy(frame_normalized).permute(2, 0, 1)
            
            processed.append(frame_tensor)
        
        return processed

class TemporalAnalyzer:
    """Temporal consistency analysis across frames"""
    
    def __init__(self, window_size: int = 5, consistency_threshold: float = 0.7):
        self.window_size = window_size
        self.consistency_threshold = consistency_threshold
    
    def analyze_temporal_consistency(self, frame_analyses: List[FrameAnalysis]) -> Dict:
        """Analyze temporal consistency across frames"""
        if len(frame_analyses) < 2:
            return {
                "temporal_consistency": 1.0,
                "consistency_score": 1.0,
                "temporal_patterns": [],
                "anomaly_frames": []
            }
        
        # Extract confidence scores and timestamps
        confidences = [fa.confidence_score for fa in frame_analyses]
        timestamps = [fa.timestamp for fa in frame_analyses]
        
        # Calculate temporal consistency metrics
        consistency_score = self._calculate_consistency_score(confidences)
        temporal_patterns = self._detect_temporal_patterns(confidences, timestamps)
        anomaly_frames = self._detect_anomalies(frame_analyses)
        
        return {
            "temporal_consistency": consistency_score,
            "consistency_score": consistency_score,
            "temporal_patterns": temporal_patterns,
            "anomaly_frames": anomaly_frames,
            "confidence_variance": np.var(confidences),
            "confidence_trend": self._calculate_trend(confidences)
        }
    
    def _calculate_consistency_score(self, confidences: List[float]) -> float:
        """Calculate temporal consistency score"""
        if len(confidences) < 2:
            return 1.0
        
        # Calculate rolling variance
        variances = []
        for i in range(len(confidences) - self.window_size + 1):
            window = confidences[i:i + self.window_size]
            variances.append(np.var(window))
        
        if not variances:
            return 1.0
        
        # Convert variance to consistency score (lower variance = higher consistency)
        avg_variance = np.mean(variances)
        consistency = max(0.0, 1.0 - avg_variance)
        
        return consistency
    
    def _detect_temporal_patterns(self, confidences: List[float], timestamps: List[float]) -> List[Dict]:
        """Detect temporal patterns in confidence scores"""
        patterns = []
        
        # Detect sudden changes
        for i in range(1, len(confidences)):
            change = abs(confidences[i] - confidences[i-1])
            if change > 0.3:  # Significant change threshold
                patterns.append({
                    "type": "sudden_change",
                    "frame_index": i,
                    "timestamp": timestamps[i],
                    "magnitude": change
                })
        
        # Detect trends
        if len(confidences) >= 3:
            trend = self._calculate_trend(confidences)
            if abs(trend) > 0.1:
                patterns.append({
                    "type": "trend",
                    "direction": "increasing" if trend > 0 else "decreasing",
                    "magnitude": abs(trend)
                })
        
        return patterns
    
    def _detect_anomalies(self, frame_analyses: List[FrameAnalysis]) -> List[int]:
        """Detect anomalous frames"""
        confidences = [fa.confidence_score for fa in frame_analyses]
        mean_confidence = np.mean(confidences)
        std_confidence = np.std(confidences)
        
        anomalies = []
        for i, fa in enumerate(frame_analyses):
            # Detect frames with significantly different confidence
            if abs(fa.confidence_score - mean_confidence) > 2 * std_confidence:
                anomalies.append(i)
        
        return anomalies
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate linear trend of values"""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return slope

class MemoryManager:
    """Memory-optimized processing for large files"""
    
    def __init__(self, max_memory_mb: int = 2048):
        self.max_memory_mb = max_memory_mb
        self.current_memory = 0
        self.lock = threading.Lock()
    
    def check_memory_usage(self) -> bool:
        """Check if memory usage is within limits"""
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        return memory_mb < self.max_memory_mb
    
    def cleanup_memory(self):
        """Clean up memory"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def estimate_frame_memory(self, frame_count: int, frame_size: Tuple[int, int]) -> int:
        """Estimate memory usage for frames"""
        # Estimate memory per frame (RGB float32)
        frame_memory = frame_size[0] * frame_size[1] * 3 * 4  # bytes
        total_memory = frame_memory * frame_count / 1024 / 1024  # MB
        return total_memory

class ProgressTracker:
    """Progress tracking for long video analysis"""
    
    def __init__(self, total_frames: int):
        self.total_frames = total_frames
        self.processed_frames = 0
        self.start_time = time.time()
        self.lock = threading.Lock()
    
    def update_progress(self, frames_processed: int = 1):
        """Update progress"""
        with self.lock:
            self.processed_frames += frames_processed
    
    def get_progress(self) -> Dict:
        """Get current progress information"""
        with self.lock:
            elapsed_time = time.time() - self.start_time
            progress_percent = (self.processed_frames / self.total_frames) * 100 if self.total_frames > 0 else 0
            
            # Estimate remaining time
            if self.processed_frames > 0:
                frames_per_second = self.processed_frames / elapsed_time
                remaining_frames = self.total_frames - self.processed_frames
                estimated_remaining = remaining_frames / frames_per_second if frames_per_second > 0 else 0
            else:
                estimated_remaining = 0
            
            return {
                "processed_frames": self.processed_frames,
                "total_frames": self.total_frames,
                "progress_percent": progress_percent,
                "elapsed_time": elapsed_time,
                "estimated_remaining": estimated_remaining,
                "frames_per_second": self.processed_frames / elapsed_time if elapsed_time > 0 else 0
            }

class VideoProcessor:
    """Main video processing pipeline"""
    
    def __init__(self, 
                 max_frames: int = 100,
                 batch_size: int = 8,
                 max_workers: int = 4,
                 max_memory_mb: int = 2048):
        
        self.frame_extractor = VideoFrameExtractor(max_frames=max_frames)
        self.batch_processor = BatchProcessor(batch_size=batch_size, max_workers=max_workers)
        self.temporal_analyzer = TemporalAnalyzer()
        self.memory_manager = MemoryManager(max_memory_mb=max_memory_mb)
        
    def get_video_metadata(self, video_path: str) -> VideoMetadata:
        """Get video metadata"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        try:
            metadata = VideoMetadata(
                path=video_path,
                fps=cap.get(cv2.CAP_PROP_FPS),
                frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                duration=cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
                width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                codec=int(cap.get(cv2.CAP_PROP_FOURCC)),
                bitrate=int(cap.get(cv2.CAP_PROP_BITRATE))
            )
            return metadata
        finally:
            cap.release()
    
    def analyze_video(self, video_path: str, detector, progress_callback=None) -> VideoAnalysisResult:
        """Analyze video for deepfake detection"""
        start_time = time.time()
        
        # Get video metadata
        metadata = self.get_video_metadata(video_path)
        
        # Initialize progress tracker
        progress_tracker = ProgressTracker(metadata.frame_count)
        
        # Extract and analyze frames
        frame_analyses = []
        current_batch = []
        
        for frame_idx, frame, timestamp in self.frame_extractor.extract_frames(video_path):
            # Check memory usage
            if not self.memory_manager.check_memory_usage():
                logger.warning("Memory limit reached, cleaning up...")
                self.memory_manager.cleanup_memory()
            
            # Add frame to current batch
            current_batch.append((frame_idx, frame, timestamp))
            
            # Process batch when full
            if len(current_batch) >= self.batch_processor.batch_size:
                batch_results = self._process_batch(current_batch, detector)
                frame_analyses.extend(batch_results)
                current_batch = []
                
                # Update progress
                progress_tracker.update_progress(len(batch_results))
                if progress_callback:
                    progress_callback(progress_tracker.get_progress())
        
        # Process remaining frames
        if current_batch:
            batch_results = self._process_batch(current_batch, detector)
            frame_analyses.extend(batch_results)
            progress_tracker.update_progress(len(batch_results))
        
        # Analyze temporal consistency
        temporal_analysis = self.temporal_analyzer.analyze_temporal_consistency(frame_analyses)
        
        # Calculate overall result
        overall_confidence = np.mean([fa.confidence_score for fa in frame_analyses])
        is_deepfake = overall_confidence > 0.5
        
        processing_time = time.time() - start_time
        
        return VideoAnalysisResult(
            video_path=video_path,
            total_frames=metadata.frame_count,
            analyzed_frames=len(frame_analyses),
            overall_confidence=overall_confidence,
            is_deepfake=is_deepfake,
            temporal_consistency=temporal_analysis["temporal_consistency"],
            frame_analyses=frame_analyses,
            processing_time=processing_time,
            analysis_metadata={
                "temporal_analysis": temporal_analysis,
                "video_metadata": metadata.__dict__,
                "processing_config": {
                    "max_frames": self.frame_extractor.max_frames,
                    "batch_size": self.batch_processor.batch_size,
                    "sampling_strategy": self.frame_extractor.sampling_strategy
                }
            }
        )
    
    def _process_batch(self, batch: List[Tuple[int, np.ndarray, float]], detector) -> List[FrameAnalysis]:
        """Process a batch of frames"""
        frame_indices = [item[0] for item in batch]
        frames = [item[1] for item in batch]
        timestamps = [item[2] for item in batch]
        
        # Process batch
        batch_start_time = time.time()
        results = self.batch_processor.process_batch(frames, detector)
        batch_time = time.time() - batch_start_time
        
        # Convert results to FrameAnalysis objects
        frame_analyses = []
        for i, result in enumerate(results):
            frame_analysis = FrameAnalysis(
                frame_number=frame_indices[i],
                timestamp=timestamps[i],
                confidence_score=result.get("confidence_score", 0.0),
                is_deepfake=result.get("is_deepfake", False),
                analysis_metadata=result.get("metadata", {}),
                processing_time=batch_time / len(batch)
            )
            frame_analyses.append(frame_analysis)
        
        return frame_analyses 