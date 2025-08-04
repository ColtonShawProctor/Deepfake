"""
Real-time Video Stream Processing for Deepfake Detection

Advanced real-time processing pipeline that handles live video streams,
webcam feeds, and network streams with optimized latency and throughput.

Key Features:
1. Multi-threaded frame processing
2. Adaptive quality control
3. Buffer management
4. Latency optimization
5. Stream health monitoring
6. Fallback mechanisms
"""

import cv2
import numpy as np
import torch
import threading
import queue
import time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import deque
import psutil
from concurrent.futures import ThreadPoolExecutor, Future
import asyncio
import websockets
import json

from .video_analysis_core import VideoAnalysisConfig, FrameInfo, ProcessingMode
from .optimized_ensemble_detector import OptimizedEnsembleDetector

logger = logging.getLogger(__name__)

class StreamSource(str, Enum):
    """Video stream source types"""
    WEBCAM = "webcam"
    RTMP = "rtmp"
    HTTP = "http"
    FILE = "file"
    RTSP = "rtsp"
    UDP = "udp"

class StreamQuality(str, Enum):
    """Stream quality settings"""
    LOW = "low"          # 480p, 15fps
    MEDIUM = "medium"    # 720p, 25fps  
    HIGH = "high"        # 1080p, 30fps
    AUTO = "auto"        # Adaptive quality

class AlertLevel(str, Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class StreamConfig:
    """Configuration for real-time stream processing"""
    # Stream settings
    source_type: StreamSource = StreamSource.WEBCAM
    source_path: str = "0"  # 0 for default webcam
    target_fps: float = 15.0
    quality: StreamQuality = StreamQuality.MEDIUM
    
    # Processing settings
    processing_threads: int = 3
    max_queue_size: int = 30
    frame_skip_threshold: int = 5  # Skip frames if queue is full
    
    # Buffer management
    input_buffer_size: int = 50
    output_buffer_size: int = 100
    circular_buffer: bool = True
    
    # Latency optimization
    max_latency_ms: float = 1000.0
    adaptive_quality: bool = True
    frame_dropping: bool = True
    
    # Alert system
    deepfake_threshold: float = 70.0
    alert_cooldown: float = 5.0  # Seconds between alerts
    consecutive_alerts: int = 3  # Alerts needed for trigger
    
    # Performance monitoring
    performance_window: int = 100  # Frames for performance averaging
    log_performance: bool = True
    
    # Fallback settings
    fallback_quality: StreamQuality = StreamQuality.LOW
    max_failures: int = 10
    restart_delay: float = 2.0

@dataclass
class StreamFrame:
    """Real-time stream frame with metadata"""
    frame_data: np.ndarray
    timestamp: float
    frame_id: int
    source_fps: float
    processing_start: Optional[float] = None
    processing_end: Optional[float] = None
    confidence: Optional[float] = None
    is_deepfake: Optional[bool] = None
    dropped: bool = False
    quality_level: Optional[StreamQuality] = None

@dataclass
class StreamAlert:
    """Alert generated during stream processing"""
    timestamp: float
    level: AlertLevel
    message: str
    confidence: float
    frame_id: int
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StreamStats:
    """Stream processing statistics"""
    frames_processed: int = 0
    frames_dropped: int = 0
    average_fps: float = 0.0
    average_latency: float = 0.0
    deepfake_detections: int = 0
    alerts_generated: int = 0
    uptime: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0

class FrameBuffer:
    """Thread-safe circular buffer for frames"""
    
    def __init__(self, maxsize: int, circular: bool = True):
        self.maxsize = maxsize
        self.circular = circular
        self.buffer = deque(maxlen=maxsize if circular else None)
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        
    def put(self, item: StreamFrame, block: bool = True, timeout: Optional[float] = None) -> bool:
        """Add frame to buffer"""
        with self.condition:
            if not self.circular and len(self.buffer) >= self.maxsize:
                if not block:
                    return False
                if not self.condition.wait(timeout):
                    return False
            
            self.buffer.append(item)
            self.condition.notify()
            return True
    
    def get(self, block: bool = True, timeout: Optional[float] = None) -> Optional[StreamFrame]:
        """Get frame from buffer"""
        with self.condition:
            while len(self.buffer) == 0:
                if not block:
                    return None
                if not self.condition.wait(timeout):
                    return None
            
            frame = self.buffer.popleft()
            self.condition.notify()
            return frame
    
    def size(self) -> int:
        """Get current buffer size"""
        with self.lock:
            return len(self.buffer)
    
    def clear(self):
        """Clear buffer"""
        with self.lock:
            self.buffer.clear()

class StreamCapture:
    """Video stream capture with error handling"""
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self.cap = None
        self.logger = logging.getLogger(f"{__name__}.StreamCapture")
        self.failure_count = 0
        self.last_restart = 0.0
        
    def initialize(self) -> bool:
        """Initialize video capture"""
        try:
            if self.config.source_type == StreamSource.WEBCAM:
                source = int(self.config.source_path) if self.config.source_path.isdigit() else 0
            else:
                source = self.config.source_path
            
            self.cap = cv2.VideoCapture(source)
            
            if not self.cap.isOpened():
                raise RuntimeError("Failed to open video source")
            
            # Set capture properties
            self._configure_capture()
            
            self.failure_count = 0
            self.logger.info(f"Stream capture initialized: {source}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize capture: {e}")
            self.failure_count += 1
            return False
    
    def _configure_capture(self):
        """Configure capture settings based on quality"""
        if not self.cap:
            return
        
        quality_settings = {
            StreamQuality.LOW: (640, 480, 15),
            StreamQuality.MEDIUM: (1280, 720, 25),
            StreamQuality.HIGH: (1920, 1080, 30)
        }
        
        if self.config.quality != StreamQuality.AUTO:
            width, height, fps = quality_settings[self.config.quality]
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)
        
        # Buffer settings for reduced latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    def read_frame(self) -> Optional[StreamFrame]:
        """Read frame from stream"""
        if not self.cap or not self.cap.isOpened():
            return None
        
        try:
            ret, frame = self.cap.read()
            if not ret:
                self.failure_count += 1
                if self.failure_count > self.config.max_failures:
                    self._restart_capture()
                return None
            
            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Get stream properties
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            timestamp = time.time()
            
            return StreamFrame(
                frame_data=frame_rgb,
                timestamp=timestamp,
                frame_id=int(timestamp * 1000),  # Unique ID based on timestamp
                source_fps=fps,
                quality_level=self.config.quality
            )
            
        except Exception as e:
            self.logger.error(f"Failed to read frame: {e}")
            self.failure_count += 1
            return None
    
    def _restart_capture(self):
        """Restart capture after failure"""
        current_time = time.time()
        if current_time - self.last_restart < self.config.restart_delay:
            return
        
        self.logger.info("Restarting video capture...")
        self.close()
        time.sleep(self.config.restart_delay)
        self.initialize()
        self.last_restart = current_time
    
    def close(self):
        """Close video capture"""
        if self.cap:
            self.cap.release()
            self.cap = None

class AlertManager:
    """Manage alerts and notifications"""
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self.alerts = deque(maxlen=1000)  # Keep last 1000 alerts
        self.last_alert_time = {}  # Per-type cooldown
        self.consecutive_detections = 0
        self.alert_callbacks: List[Callable[[StreamAlert], None]] = []
        self.lock = threading.Lock()
        
    def add_callback(self, callback: Callable[[StreamAlert], None]):
        """Add alert callback function"""
        self.alert_callbacks.append(callback)
    
    def check_deepfake_alert(self, frame: StreamFrame) -> Optional[StreamAlert]:
        """Check if deepfake alert should be generated"""
        if frame.confidence is None or not frame.is_deepfake:
            self.consecutive_detections = 0
            return None
        
        if frame.confidence < self.config.deepfake_threshold:
            self.consecutive_detections = 0
            return None
        
        self.consecutive_detections += 1
        
        # Check if we should trigger alert
        if self.consecutive_detections >= self.config.consecutive_alerts:
            current_time = time.time()
            last_alert = self.last_alert_time.get("deepfake", 0)
            
            if current_time - last_alert >= self.config.alert_cooldown:
                alert = StreamAlert(
                    timestamp=current_time,
                    level=AlertLevel.CRITICAL,
                    message=f"Deepfake detected with {frame.confidence:.1f}% confidence",
                    confidence=frame.confidence,
                    frame_id=frame.frame_id,
                    metadata={
                        "consecutive_detections": self.consecutive_detections,
                        "source_fps": frame.source_fps
                    }
                )
                
                self._trigger_alert(alert)
                self.last_alert_time["deepfake"] = current_time
                self.consecutive_detections = 0
                return alert
        
        return None
    
    def _trigger_alert(self, alert: StreamAlert):
        """Trigger alert to all callbacks"""
        with self.lock:
            self.alerts.append(alert)
            
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")
    
    def get_recent_alerts(self, count: int = 10) -> List[StreamAlert]:
        """Get recent alerts"""
        with self.lock:
            return list(self.alerts)[-count:]

class PerformanceMonitor:
    """Monitor stream processing performance"""
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self.stats = StreamStats()
        self.frame_times = deque(maxlen=config.performance_window)
        self.latencies = deque(maxlen=config.performance_window)
        self.start_time = time.time()
        self.lock = threading.Lock()
        
    def record_frame(self, frame: StreamFrame):
        """Record frame processing metrics"""
        with self.lock:
            self.stats.frames_processed += 1
            
            if frame.dropped:
                self.stats.frames_dropped += 1
            
            if frame.is_deepfake:
                self.stats.deepfake_detections += 1
            
            # Record timing
            current_time = time.time()
            self.frame_times.append(current_time)
            
            if frame.processing_start and frame.processing_end:
                latency = (frame.processing_end - frame.processing_start) * 1000  # ms
                self.latencies.append(latency)
            
            # Update averages
            if len(self.frame_times) > 1:
                time_diff = self.frame_times[-1] - self.frame_times[0]
                self.stats.average_fps = len(self.frame_times) / max(time_diff, 0.001)
            
            if self.latencies:
                self.stats.average_latency = np.mean(self.latencies)
            
            self.stats.uptime = current_time - self.start_time
            
            # System resources
            process = psutil.Process()
            self.stats.memory_usage = process.memory_info().rss / 1024**3  # GB
            self.stats.cpu_usage = process.cpu_percent()
    
    def get_stats(self) -> StreamStats:
        """Get current statistics"""
        with self.lock:
            return self.stats
    
    def reset_stats(self):
        """Reset statistics"""
        with self.lock:
            self.stats = StreamStats()
            self.frame_times.clear()
            self.latencies.clear()
            self.start_time = time.time()

class RealtimeVideoProcessor:
    """Main real-time video processor"""
    
    def __init__(self, ensemble_detector: OptimizedEnsembleDetector, config: StreamConfig):
        self.ensemble_detector = ensemble_detector
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.RealtimeVideoProcessor")
        
        # Components
        self.stream_capture = StreamCapture(config)
        self.alert_manager = AlertManager(config)
        self.performance_monitor = PerformanceMonitor(config)
        
        # Buffers
        self.input_buffer = FrameBuffer(config.input_buffer_size, config.circular_buffer)
        self.output_buffer = FrameBuffer(config.output_buffer_size, config.circular_buffer)
        
        # Processing state
        self.running = False
        self.threads = []
        self.executor = ThreadPoolExecutor(max_workers=config.processing_threads)
        
        # WebSocket server for real-time updates
        self.websocket_clients = set()
        self.websocket_server = None
    
    def start(self) -> bool:
        """Start real-time processing"""
        if self.running:
            self.logger.warning("Processor already running")
            return True
        
        # Initialize stream capture
        if not self.stream_capture.initialize():
            self.logger.error("Failed to initialize stream capture")
            return False
        
        self.running = True
        
        # Start processing threads
        self.threads = [
            threading.Thread(target=self._capture_thread, daemon=True),
            threading.Thread(target=self._processing_thread, daemon=True),
            threading.Thread(target=self._output_thread, daemon=True)
        ]
        
        for thread in self.threads:
            thread.start()
        
        # Start WebSocket server
        self._start_websocket_server()
        
        self.logger.info("Real-time video processor started")
        return True
    
    def stop(self):
        """Stop real-time processing"""
        self.running = False
        
        # Wait for threads to finish
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=5.0)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        # Close stream capture
        self.stream_capture.close()
        
        # Stop WebSocket server
        if self.websocket_server:
            self.websocket_server.close()
        
        self.logger.info("Real-time video processor stopped")
    
    def _capture_thread(self):
        """Thread for capturing frames from stream"""
        while self.running:
            try:
                frame = self.stream_capture.read_frame()
                if frame is None:
                    time.sleep(0.01)  # Short sleep on failure
                    continue
                
                # Check if we should drop frame due to full buffer
                if (self.input_buffer.size() >= self.config.frame_skip_threshold and 
                    self.config.frame_dropping):
                    frame.dropped = True
                    self.performance_monitor.record_frame(frame)
                    continue
                
                # Add to input buffer
                if not self.input_buffer.put(frame, block=False):
                    frame.dropped = True
                    self.performance_monitor.record_frame(frame)
                
            except Exception as e:
                self.logger.error(f"Capture thread error: {e}")
                time.sleep(0.1)
    
    def _processing_thread(self):
        """Thread for processing frames"""
        while self.running:
            try:
                # Get frame from input buffer
                frame = self.input_buffer.get(timeout=1.0)
                if frame is None:
                    continue
                
                # Skip dropped frames
                if frame.dropped:
                    continue
                
                # Process frame
                frame.processing_start = time.time()
                self._process_frame(frame)
                frame.processing_end = time.time()
                
                # Check latency and quality adaptation
                latency = (frame.processing_end - frame.processing_start) * 1000
                if latency > self.config.max_latency_ms and self.config.adaptive_quality:
                    self._adapt_quality()
                
                # Add to output buffer
                self.output_buffer.put(frame, block=False)
                
            except Exception as e:
                self.logger.error(f"Processing thread error: {e}")
    
    def _output_thread(self):
        """Thread for handling processed frames"""
        while self.running:
            try:
                # Get processed frame
                frame = self.output_buffer.get(timeout=1.0)
                if frame is None:
                    continue
                
                # Record performance metrics
                self.performance_monitor.record_frame(frame)
                
                # Check for alerts
                alert = self.alert_manager.check_deepfake_alert(frame)
                if alert:
                    self.performance_monitor.stats.alerts_generated += 1
                
                # Send to WebSocket clients
                self._broadcast_frame_result(frame, alert)
                
            except Exception as e:
                self.logger.error(f"Output thread error: {e}")
    
    def _process_frame(self, frame: StreamFrame):
        """Process individual frame through ensemble"""
        try:
            from PIL import Image
            
            # Convert to PIL Image
            pil_image = Image.fromarray(frame.frame_data)
            
            # Analyze with ensemble detector
            result = self.ensemble_detector.analyze_image_from_pil(pil_image)
            
            # Extract results
            frame.confidence = result.get('confidence_score', 0.0)
            frame.is_deepfake = result.get('is_deepfake', False)
            
        except Exception as e:
            self.logger.error(f"Frame processing failed: {e}")
            frame.confidence = 0.0
            frame.is_deepfake = False
    
    def _adapt_quality(self):
        """Adapt stream quality based on performance"""
        current_quality = self.config.quality
        
        if current_quality == StreamQuality.HIGH:
            self.config.quality = StreamQuality.MEDIUM
        elif current_quality == StreamQuality.MEDIUM:
            self.config.quality = StreamQuality.LOW
        
        # Reconfigure capture
        self.stream_capture._configure_capture()
        self.logger.info(f"Adapted quality to: {self.config.quality}")
    
    def _start_websocket_server(self):
        """Start WebSocket server for real-time updates"""
        async def handle_client(websocket, path):
            self.websocket_clients.add(websocket)
            try:
                await websocket.wait_closed()
            finally:
                self.websocket_clients.remove(websocket)
        
        # Start server in separate thread
        def run_server():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            start_server = websockets.serve(handle_client, "localhost", 8765)
            self.websocket_server = loop.run_until_complete(start_server)
            loop.run_forever()
        
        websocket_thread = threading.Thread(target=run_server, daemon=True)
        websocket_thread.start()
    
    def _broadcast_frame_result(self, frame: StreamFrame, alert: Optional[StreamAlert]):
        """Broadcast frame result to WebSocket clients"""
        if not self.websocket_clients:
            return
        
        message = {
            "type": "frame_result",
            "timestamp": frame.timestamp,
            "frame_id": frame.frame_id,
            "confidence": frame.confidence,
            "is_deepfake": frame.is_deepfake,
            "processing_time": (frame.processing_end - frame.processing_start) * 1000 if frame.processing_start and frame.processing_end else 0,
            "alert": {
                "level": alert.level,
                "message": alert.message,
                "confidence": alert.confidence
            } if alert else None
        }
        
        # Send to all connected clients
        disconnected_clients = set()
        for client in self.websocket_clients:
            try:
                asyncio.run_coroutine_threadsafe(
                    client.send(json.dumps(message)),
                    client.loop
                )
            except Exception:
                disconnected_clients.add(client)
        
        # Remove disconnected clients
        self.websocket_clients -= disconnected_clients
    
    def get_stats(self) -> StreamStats:
        """Get current processing statistics"""
        return self.performance_monitor.get_stats()
    
    def get_recent_alerts(self, count: int = 10) -> List[StreamAlert]:
        """Get recent alerts"""
        return self.alert_manager.get_recent_alerts(count)
    
    def add_alert_callback(self, callback: Callable[[StreamAlert], None]):
        """Add alert callback"""
        self.alert_manager.add_callback(callback)

def create_realtime_processor(ensemble_detector: OptimizedEnsembleDetector, 
                             config: Optional[StreamConfig] = None) -> RealtimeVideoProcessor:
    """Factory function to create real-time processor"""
    if config is None:
        config = StreamConfig()
    
    return RealtimeVideoProcessor(ensemble_detector, config)