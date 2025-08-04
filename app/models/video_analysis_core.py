"""
Video Analysis Core for Temporal Deepfake Detection

Advanced video analysis system that extends image-based ensemble detection
to handle temporal information and video-specific artifacts.

Architecture Components:
1. Frame Sampling Strategies (uniform, adaptive, key-frame based)
2. Temporal Model Integration (CNN-LSTM-Transformer hybrids)
3. Memory Management for large videos
4. Real-time Stream Processing
5. Temporal Consistency Analysis
6. Optical Flow Integration
7. AltFreezing for temporal weight management
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Generator, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
import threading
from pathlib import Path
from collections import deque
import psutil
import gc
from PIL import Image

logger = logging.getLogger(__name__)

class FrameSamplingStrategy(str, Enum):
    """Frame sampling strategies for video analysis"""
    UNIFORM = "uniform"              # Regular intervals
    ADAPTIVE = "adaptive"            # Based on motion/change detection
    KEYFRAME = "keyframe"           # I-frames and scene changes
    TEMPORAL_AWARE = "temporal_aware" # Content-aware sampling
    HYBRID = "hybrid"               # Combination of strategies

class ProcessingMode(str, Enum):
    """Video processing modes"""
    BATCH = "batch"                 # Full video processing
    REALTIME = "realtime"          # Live stream processing
    STREAMING = "streaming"        # Chunk-based processing
    MEMORY_OPTIMIZED = "memory_optimized" # Low memory usage

class TemporalModelType(str, Enum):
    """Temporal model architectures"""
    LSTM = "lstm"                   # LSTM for temporal modeling
    TRANSFORMER = "transformer"    # Transformer attention
    CNN_LSTM = "cnn_lstm"          # CNN + LSTM hybrid
    CNN_TRANSFORMER = "cnn_transformer" # CNN + Transformer hybrid
    OPTICAL_FLOW = "optical_flow"  # Optical flow based
    ALTFREEZING = "altfreezing"    # AltFreezing temporal weights

@dataclass
class VideoAnalysisConfig:
    """Configuration for video analysis system"""
    # Frame sampling
    sampling_strategy: FrameSamplingStrategy = FrameSamplingStrategy.HYBRID
    target_fps: float = 5.0                    # Target sampling rate
    max_frames: int = 300                      # Maximum frames to process
    min_frames: int = 10                       # Minimum frames required
    keyframe_threshold: float = 0.3            # Change threshold for keyframes
    
    # Processing mode
    processing_mode: ProcessingMode = ProcessingMode.BATCH
    chunk_size: int = 32                       # Frames per processing chunk
    overlap_frames: int = 4                    # Frame overlap between chunks
    
    # Temporal modeling
    temporal_model: TemporalModelType = TemporalModelType.CNN_TRANSFORMER
    sequence_length: int = 16                   # Temporal sequence length
    temporal_stride: int = 2                    # Stride for temporal sampling
    
    # Memory management
    max_memory_gb: float = 4.0                 # Maximum memory usage
    enable_memory_optimization: bool = True
    frame_cache_size: int = 100                # Number of frames to cache
    
    # Real-time processing
    realtime_buffer_size: int = 64             # Frame buffer for real-time
    processing_threads: int = 2                # Number of processing threads
    max_latency_ms: float = 500.0             # Maximum acceptable latency
    
    # Temporal consistency
    enable_temporal_smoothing: bool = True
    smoothing_window: int = 5                  # Frames for smoothing
    consistency_threshold: float = 0.15        # Temporal consistency threshold
    
    # Optical flow
    enable_optical_flow: bool = True
    flow_method: str = "farneback"            # Optical flow algorithm
    flow_threshold: float = 0.1               # Motion threshold
    
    # AltFreezing
    enable_altfreezing: bool = True
    freeze_layers: List[str] = field(default_factory=lambda: ["conv1", "conv2"])
    freeze_probability: float = 0.3

@dataclass
class FrameInfo:
    """Information about a video frame"""
    frame_idx: int
    timestamp: float
    frame_data: np.ndarray
    motion_score: float = 0.0
    is_keyframe: bool = False
    optical_flow: Optional[np.ndarray] = None
    features: Optional[np.ndarray] = None

@dataclass
class VideoAnalysisResult:
    """Result of video analysis"""
    video_path: str
    total_frames: int
    analyzed_frames: int
    overall_confidence: float
    is_deepfake: bool
    frame_results: List[Dict[str, Any]]
    temporal_consistency: float
    processing_time: float
    memory_usage: float
    
    # Temporal analysis
    temporal_patterns: Dict[str, Any]
    optical_flow_analysis: Dict[str, Any]
    confidence_timeline: List[float]
    
    # Metadata
    video_metadata: Dict[str, Any]
    analysis_config: VideoAnalysisConfig
    error_info: Optional[str] = None

class FrameSampler:
    """Intelligent frame sampling for video analysis"""
    
    def __init__(self, config: VideoAnalysisConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.FrameSampler")
    
    def sample_frames(self, video_path: str) -> List[FrameInfo]:
        """Sample frames from video based on configured strategy"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            self.logger.info(f"Video: {total_frames} frames, {fps:.2f} FPS, {duration:.2f}s")
            
            if self.config.sampling_strategy == FrameSamplingStrategy.UNIFORM:
                return self._uniform_sampling(cap, total_frames, fps)
            elif self.config.sampling_strategy == FrameSamplingStrategy.ADAPTIVE:
                return self._adaptive_sampling(cap, total_frames, fps)
            elif self.config.sampling_strategy == FrameSamplingStrategy.KEYFRAME:
                return self._keyframe_sampling(cap, total_frames, fps)
            elif self.config.sampling_strategy == FrameSamplingStrategy.TEMPORAL_AWARE:
                return self._temporal_aware_sampling(cap, total_frames, fps)
            else:  # HYBRID
                return self._hybrid_sampling(cap, total_frames, fps)
                
        finally:
            cap.release()
    
    def _uniform_sampling(self, cap: cv2.VideoCapture, total_frames: int, fps: float) -> List[FrameInfo]:
        """Uniform frame sampling at regular intervals"""
        target_fps = min(self.config.target_fps, fps)
        frame_interval = max(1, int(fps / target_fps))
        max_frames = min(self.config.max_frames, total_frames // frame_interval)
        
        frames = []
        for i in range(0, total_frames, frame_interval):
            if len(frames) >= max_frames:
                break
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                timestamp = i / fps if fps > 0 else 0
                frames.append(FrameInfo(
                    frame_idx=i,
                    timestamp=timestamp,
                    frame_data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ))
        
        self.logger.info(f"Uniform sampling: {len(frames)} frames")
        return frames
    
    def _adaptive_sampling(self, cap: cv2.VideoCapture, total_frames: int, fps: float) -> List[FrameInfo]:
        """Adaptive sampling based on motion detection"""
        frames = []
        prev_frame = None
        frame_count = 0
        
        while frame_count < total_frames and len(frames) < self.config.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            motion_score = 0.0
            if prev_frame is not None:
                # Calculate motion between frames
                motion_score = self._calculate_motion_score(prev_frame, gray)
            
            # Include frame if high motion or first frame
            if motion_score > self.config.keyframe_threshold or len(frames) == 0:
                timestamp = frame_count / fps if fps > 0 else 0
                frames.append(FrameInfo(
                    frame_idx=frame_count,
                    timestamp=timestamp,
                    frame_data=frame_rgb,
                    motion_score=motion_score,
                    is_keyframe=motion_score > self.config.keyframe_threshold
                ))
            
            prev_frame = gray
            frame_count += 1
        
        self.logger.info(f"Adaptive sampling: {len(frames)} frames")
        return frames
    
    def _keyframe_sampling(self, cap: cv2.VideoCapture, total_frames: int, fps: float) -> List[FrameInfo]:
        """Sample frames based on keyframe detection and scene changes"""
        frames = []
        prev_hist = None
        frame_count = 0
        
        while frame_count < total_frames and len(frames) < self.config.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Calculate histogram for scene change detection
            hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            
            is_keyframe = False
            if prev_hist is not None:
                # Compare histograms to detect scene changes
                correlation = cv2.compareHist(hist, prev_hist, cv2.HISTCMP_CORREL)
                if correlation < (1.0 - self.config.keyframe_threshold):
                    is_keyframe = True
            else:
                is_keyframe = True  # First frame
            
            if is_keyframe or len(frames) == 0:
                timestamp = frame_count / fps if fps > 0 else 0
                frames.append(FrameInfo(
                    frame_idx=frame_count,
                    timestamp=timestamp,
                    frame_data=frame_rgb,
                    is_keyframe=is_keyframe
                ))
            
            prev_hist = hist
            frame_count += 1
        
        self.logger.info(f"Keyframe sampling: {len(frames)} frames")
        return frames
    
    def _temporal_aware_sampling(self, cap: cv2.VideoCapture, total_frames: int, fps: float) -> List[FrameInfo]:
        """Content-aware temporal sampling"""
        # This would implement more sophisticated content analysis
        # For now, using adaptive sampling with enhanced motion detection
        return self._adaptive_sampling(cap, total_frames, fps)
    
    def _hybrid_sampling(self, cap: cv2.VideoCapture, total_frames: int, fps: float) -> List[FrameInfo]:
        """Hybrid sampling combining multiple strategies"""
        # Start with uniform base sampling
        uniform_frames = self._uniform_sampling(cap, total_frames, fps)
        
        # Add keyframes
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset position
        keyframes = self._keyframe_sampling(cap, total_frames, fps)
        
        # Merge and deduplicate
        all_frames = {}
        for frame in uniform_frames + keyframes:
            all_frames[frame.frame_idx] = frame
        
        # Sort by frame index
        sorted_frames = sorted(all_frames.values(), key=lambda x: x.frame_idx)
        
        # Limit to max frames
        if len(sorted_frames) > self.config.max_frames:
            # Keep evenly distributed subset
            indices = np.linspace(0, len(sorted_frames) - 1, self.config.max_frames, dtype=int)
            sorted_frames = [sorted_frames[i] for i in indices]
        
        self.logger.info(f"Hybrid sampling: {len(sorted_frames)} frames")
        return sorted_frames
    
    def _calculate_motion_score(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
        """Calculate motion score between two grayscale frames"""
        # Simple frame difference
        diff = cv2.absdiff(prev_gray, curr_gray)
        motion_score = np.mean(diff) / 255.0
        return motion_score

class OpticalFlowAnalyzer:
    """Optical flow analysis for temporal consistency"""
    
    def __init__(self, config: VideoAnalysisConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.OpticalFlowAnalyzer")
    
    def analyze_flow(self, frames: List[FrameInfo]) -> Dict[str, Any]:
        """Analyze optical flow between consecutive frames"""
        if len(frames) < 2:
            return {"error": "Need at least 2 frames for optical flow"}
        
        flow_vectors = []
        flow_magnitudes = []
        inconsistencies = []
        
        for i in range(len(frames) - 1):
            curr_frame = frames[i].frame_data
            next_frame = frames[i + 1].frame_data
            
            # Convert to grayscale
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)
            next_gray = cv2.cvtColor(next_frame, cv2.COLOR_RGB2GRAY)
            
            # Calculate optical flow
            if self.config.flow_method == "farneback":
                flow = cv2.calcOpticalFlowPyrLK(curr_gray, next_gray, None, None)
                if flow[0] is not None:
                    # Calculate flow magnitude
                    magnitude = np.sqrt(flow[0][:, :, 0]**2 + flow[0][:, :, 1]**2)
                    flow_magnitudes.append(np.mean(magnitude))
                    
                    # Store flow for frame
                    frames[i].optical_flow = flow[0]
            
        # Detect temporal inconsistencies
        if len(flow_magnitudes) > 1:
            flow_diff = np.diff(flow_magnitudes)
            inconsistencies = np.where(np.abs(flow_diff) > self.config.flow_threshold)[0]
        
        return {
            "flow_magnitudes": flow_magnitudes,
            "average_flow": np.mean(flow_magnitudes) if flow_magnitudes else 0.0,
            "flow_variance": np.var(flow_magnitudes) if flow_magnitudes else 0.0,
            "inconsistencies": inconsistencies.tolist(),
            "inconsistency_ratio": len(inconsistencies) / max(1, len(flow_magnitudes) - 1)
        }

class TemporalModel(nn.Module):
    """Temporal model for video analysis"""
    
    def __init__(self, config: VideoAnalysisConfig, input_dim: int = 512):
        super().__init__()
        self.config = config
        self.model_type = config.temporal_model
        self.input_dim = input_dim
        self.sequence_length = config.sequence_length
        
        if self.model_type == TemporalModelType.LSTM:
            self.temporal_layer = nn.LSTM(
                input_dim, 256, batch_first=True, num_layers=2, dropout=0.2
            )
            self.classifier = nn.Linear(256, 1)
            
        elif self.model_type == TemporalModelType.TRANSFORMER:
            self.pos_encoding = self._create_positional_encoding()
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=input_dim, nhead=8, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
            self.classifier = nn.Linear(input_dim, 1)
            
        elif self.model_type == TemporalModelType.CNN_LSTM:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(input_dim, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(256, 128, kernel_size=3, padding=1),
                nn.ReLU()
            )
            self.lstm = nn.LSTM(128, 128, batch_first=True, num_layers=2)
            self.classifier = nn.Linear(128, 1)
            
        elif self.model_type == TemporalModelType.CNN_TRANSFORMER:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(input_dim, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(256, 256, kernel_size=3, padding=1),
                nn.ReLU()
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=256, nhead=8, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
            self.classifier = nn.Linear(256, 1)
        
        # AltFreezing components
        if config.enable_altfreezing:
            self.frozen_layers = set()
            self.freeze_scheduler = self._create_freeze_scheduler()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through temporal model"""
        batch_size, seq_len, feat_dim = x.shape
        
        if self.model_type == TemporalModelType.LSTM:
            output, _ = self.temporal_layer(x)
            # Use last time step
            output = output[:, -1, :]
            
        elif self.model_type == TemporalModelType.TRANSFORMER:
            # Add positional encoding
            x = x + self.pos_encoding[:seq_len, :].unsqueeze(0)
            output = self.transformer(x)
            # Global average pooling
            output = torch.mean(output, dim=1)
            
        elif self.model_type == TemporalModelType.CNN_LSTM:
            # Reshape for 1D convolution
            x = x.transpose(1, 2)  # (batch, feat, seq)
            x = self.conv_layers(x)
            x = x.transpose(1, 2)  # Back to (batch, seq, feat)
            output, _ = self.lstm(x)
            output = output[:, -1, :]
            
        elif self.model_type == TemporalModelType.CNN_TRANSFORMER:
            x = x.transpose(1, 2)
            x = self.conv_layers(x)
            x = x.transpose(1, 2)
            output = self.transformer(x)
            output = torch.mean(output, dim=1)
        
        return torch.sigmoid(self.classifier(output))
    
    def _create_positional_encoding(self) -> torch.Tensor:
        """Create positional encoding for transformer"""
        pe = torch.zeros(self.sequence_length, self.input_dim)
        position = torch.arange(0, self.sequence_length).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, self.input_dim, 2).float() *
                           -(np.log(10000.0) / self.input_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    def _create_freeze_scheduler(self):
        """Create AltFreezing scheduler"""
        # Placeholder for AltFreezing implementation
        return None
    
    def apply_altfreezing(self, epoch: int):
        """Apply AltFreezing temporal weight management"""
        if not self.config.enable_altfreezing:
            return
        
        # Randomly freeze/unfreeze layers based on probability
        for name, param in self.named_parameters():
            layer_name = name.split('.')[0]
            if layer_name in self.config.freeze_layers:
                if np.random.random() < self.config.freeze_probability:
                    param.requires_grad = False
                    self.frozen_layers.add(name)
                else:
                    param.requires_grad = True
                    self.frozen_layers.discard(name)

class MemoryManager:
    """Memory management for video processing"""
    
    def __init__(self, config: VideoAnalysisConfig):
        self.config = config
        self.max_memory_bytes = config.max_memory_gb * 1024**3
        self.frame_cache = deque(maxlen=config.frame_cache_size)
        self.logger = logging.getLogger(f"{__name__}.MemoryManager")
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in GB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024**3
    
    def check_memory_limit(self) -> bool:
        """Check if memory usage is within limits"""
        current_usage = self.get_memory_usage()
        return current_usage < self.config.max_memory_gb
    
    def optimize_memory(self):
        """Optimize memory usage"""
        if not self.check_memory_limit():
            # Clear frame cache
            self.frame_cache.clear()
            
            # Force garbage collection
            gc.collect()
            
            # Clear PyTorch cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.logger.info("Memory optimization performed")
    
    def cache_frame(self, frame_info: FrameInfo):
        """Cache frame for potential reuse"""
        if self.check_memory_limit():
            self.frame_cache.append(frame_info)
    
    def get_cached_frame(self, frame_idx: int) -> Optional[FrameInfo]:
        """Retrieve cached frame"""
        for frame in self.frame_cache:
            if frame.frame_idx == frame_idx:
                return frame
        return None

class VideoProcessor:
    """Main video processing class"""
    
    def __init__(self, config: VideoAnalysisConfig, ensemble_detector):
        self.config = config
        self.ensemble_detector = ensemble_detector
        self.frame_sampler = FrameSampler(config)
        self.optical_flow_analyzer = OpticalFlowAnalyzer(config)
        self.memory_manager = MemoryManager(config)
        self.temporal_model = TemporalModel(config)
        
        self.logger = logging.getLogger(f"{__name__}.VideoProcessor")
        
        # Processing state
        self.processing_active = False
        self.current_results = []
    
    def analyze_video(self, video_path: str) -> VideoAnalysisResult:
        """Analyze video for deepfake detection"""
        start_time = time.time()
        
        try:
            # Get video metadata
            video_metadata = self._get_video_metadata(video_path)
            
            # Sample frames
            frames = self.frame_sampler.sample_frames(video_path)
            if len(frames) < self.config.min_frames:
                raise ValueError(f"Not enough frames: {len(frames)} < {self.config.min_frames}")
            
            # Process frames
            if self.config.processing_mode == ProcessingMode.BATCH:
                frame_results = self._process_batch(frames)
            elif self.config.processing_mode == ProcessingMode.STREAMING:
                frame_results = self._process_streaming(frames)
            else:
                frame_results = self._process_memory_optimized(frames)
            
            # Analyze optical flow
            flow_analysis = self.optical_flow_analyzer.analyze_flow(frames)
            
            # Temporal consistency analysis
            temporal_analysis = self._analyze_temporal_consistency(frame_results)
            
            # Calculate overall result
            confidences = [r['confidence'] for r in frame_results]
            overall_confidence = self._calculate_weighted_confidence(confidences, frames)
            
            # Apply temporal smoothing
            if self.config.enable_temporal_smoothing:
                confidences = self._apply_temporal_smoothing(confidences)
                overall_confidence = np.mean(confidences)
            
            processing_time = time.time() - start_time
            memory_usage = self.memory_manager.get_memory_usage()
            
            return VideoAnalysisResult(
                video_path=video_path,
                total_frames=video_metadata.get('total_frames', 0),
                analyzed_frames=len(frames),
                overall_confidence=overall_confidence,
                is_deepfake=overall_confidence > 50.0,
                frame_results=frame_results,
                temporal_consistency=temporal_analysis['consistency_score'],
                processing_time=processing_time,
                memory_usage=memory_usage,
                temporal_patterns=temporal_analysis,
                optical_flow_analysis=flow_analysis,
                confidence_timeline=confidences,
                video_metadata=video_metadata,
                analysis_config=self.config
            )
            
        except Exception as e:
            self.logger.error(f"Video analysis failed: {e}")
            return VideoAnalysisResult(
                video_path=video_path,
                total_frames=0,
                analyzed_frames=0,
                overall_confidence=0.0,
                is_deepfake=False,
                frame_results=[],
                temporal_consistency=0.0,
                processing_time=time.time() - start_time,
                memory_usage=self.memory_manager.get_memory_usage(),
                temporal_patterns={},
                optical_flow_analysis={},
                confidence_timeline=[],
                video_metadata={},
                analysis_config=self.config,
                error_info=str(e)
            )
    
    def _process_batch(self, frames: List[FrameInfo]) -> List[Dict[str, Any]]:
        """Process all frames in batch mode"""
        results = []
        
        for frame_info in frames:
            # Convert to PIL Image
            pil_image = Image.fromarray(frame_info.frame_data)
            
            # Analyze with ensemble
            result = self.ensemble_detector.analyze_image_from_pil(pil_image)
            
            results.append({
                'frame_idx': frame_info.frame_idx,
                'timestamp': frame_info.timestamp,
                'confidence': result.get('confidence_score', 0.0),
                'is_deepfake': result.get('is_deepfake', False),
                'motion_score': frame_info.motion_score,
                'is_keyframe': frame_info.is_keyframe,
                'analysis_metadata': result.get('analysis_metadata', {})
            })
            
            # Memory management
            self.memory_manager.optimize_memory()
        
        return results
    
    def _process_streaming(self, frames: List[FrameInfo]) -> List[Dict[str, Any]]:
        """Process frames in streaming chunks"""
        results = []
        chunk_size = self.config.chunk_size
        
        for i in range(0, len(frames), chunk_size):
            chunk = frames[i:i + chunk_size]
            chunk_results = self._process_batch(chunk)
            results.extend(chunk_results)
            
            # Memory optimization after each chunk
            self.memory_manager.optimize_memory()
        
        return results
    
    def _process_memory_optimized(self, frames: List[FrameInfo]) -> List[Dict[str, Any]]:
        """Process frames with memory optimization"""
        results = []
        
        for frame_info in frames:
            # Check memory before processing
            if not self.memory_manager.check_memory_limit():
                self.memory_manager.optimize_memory()
            
            # Process frame
            pil_image = Image.fromarray(frame_info.frame_data)
            result = self.ensemble_detector.analyze_image_from_pil(pil_image)
            
            results.append({
                'frame_idx': frame_info.frame_idx,
                'timestamp': frame_info.timestamp,
                'confidence': result.get('confidence_score', 0.0),
                'is_deepfake': result.get('is_deepfake', False),
                'motion_score': frame_info.motion_score,
                'is_keyframe': frame_info.is_keyframe
            })
            
            # Clear frame data to save memory
            frame_info.frame_data = None
        
        return results
    
    def _analyze_temporal_consistency(self, frame_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze temporal consistency of predictions"""
        if len(frame_results) < 2:
            return {"consistency_score": 1.0, "inconsistencies": []}
        
        confidences = [r['confidence'] for r in frame_results]
        predictions = [r['is_deepfake'] for r in frame_results]
        
        # Calculate confidence variance
        confidence_variance = np.var(confidences)
        
        # Find prediction flips
        prediction_flips = []
        for i in range(len(predictions) - 1):
            if predictions[i] != predictions[i + 1]:
                prediction_flips.append(i)
        
        # Calculate consistency score
        flip_ratio = len(prediction_flips) / max(1, len(predictions) - 1)
        consistency_score = max(0.0, 1.0 - flip_ratio - confidence_variance / 100.0)
        
        return {
            "consistency_score": consistency_score,
            "confidence_variance": confidence_variance,
            "prediction_flips": prediction_flips,
            "flip_ratio": flip_ratio,
            "inconsistencies": prediction_flips
        }
    
    def _calculate_weighted_confidence(self, confidences: List[float], frames: List[FrameInfo]) -> float:
        """Calculate weighted confidence based on frame importance"""
        if not confidences:
            return 0.0
        
        weights = []
        for frame in frames:
            weight = 1.0
            # Give more weight to keyframes
            if frame.is_keyframe:
                weight += 0.5
            # Give more weight to frames with high motion
            weight += frame.motion_score * 0.3
            weights.append(weight)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(confidences)] * len(confidences)
        
        # Calculate weighted average
        weighted_confidence = sum(c * w for c, w in zip(confidences, weights))
        return weighted_confidence
    
    def _apply_temporal_smoothing(self, confidences: List[float]) -> List[float]:
        """Apply temporal smoothing to confidence scores"""
        if len(confidences) <= self.config.smoothing_window:
            return confidences
        
        smoothed = []
        window = self.config.smoothing_window
        
        for i in range(len(confidences)):
            start_idx = max(0, i - window // 2)
            end_idx = min(len(confidences), i + window // 2 + 1)
            window_values = confidences[start_idx:end_idx]
            smoothed.append(np.mean(window_values))
        
        return smoothed
    
    def _get_video_metadata(self, video_path: str) -> Dict[str, Any]:
        """Extract video metadata"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {}
        
        try:
            return {
                'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS),
                'fourcc': int(cap.get(cv2.CAP_PROP_FOURCC))
            }
        finally:
            cap.release()

def create_video_analyzer(ensemble_detector, config: Optional[VideoAnalysisConfig] = None) -> VideoProcessor:
    """Factory function to create video analyzer"""
    if config is None:
        config = VideoAnalysisConfig()
    
    return VideoProcessor(config, ensemble_detector)