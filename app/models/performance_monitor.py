"""
Performance monitoring and logging system for multi-model deepfake detection framework.
This module provides comprehensive performance tracking and monitoring capabilities.
"""

import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil
import torch


@dataclass
class PerformanceMetrics:
    """Performance metrics for a model."""
    model_name: str
    inference_count: int = 0
    total_inference_time: float = 0.0
    average_inference_time: float = 0.0
    min_inference_time: float = float('inf')
    max_inference_time: float = 0.0
    throughput_fps: float = 0.0
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    last_updated: float = field(default_factory=time.time)
    error_count: int = 0
    success_rate: float = 1.0


@dataclass
class SystemMetrics:
    """System resource metrics."""
    timestamp: float = field(default_factory=time.time)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    gpu_memory_used: Optional[float] = None
    gpu_memory_total: Optional[float] = None
    gpu_utilization: Optional[float] = None
    disk_usage_percent: float = 0.0
    network_io: Optional[Tuple[float, float]] = None


@dataclass
class MonitoringConfig:
    """Configuration for performance monitoring."""
    save_performance_data: bool = True
    performance_data_path: str = "performance_data"
    log_interval: float = 60.0  # seconds
    metrics_history_size: int = 1000
    enable_system_monitoring: bool = True
    enable_gpu_monitoring: bool = True
    enable_detailed_logging: bool = True
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "inference_time_ms": 1000.0,
        "error_rate": 0.1,
        "memory_usage_percent": 90.0,
        "cpu_usage_percent": 90.0
    })


class PerformanceMonitor:
    """
    Monitors and tracks model performance metrics.
    
    Provides comprehensive performance tracking including inference times,
    accuracy metrics, system resources, and alerting capabilities.
    """
    
    def __init__(self, config: Optional[MonitoringConfig] = None):
        """
        Initialize the performance monitor.
        
        Args:
            config: Monitoring configuration
        """
        self.config = config or MonitoringConfig()
        self.logger = logging.getLogger(f"{__name__}.PerformanceMonitor")
        
        # Performance tracking
        self.metrics: Dict[str, PerformanceMetrics] = defaultdict(lambda: PerformanceMetrics(""))
        self.timers: Dict[str, float] = {}
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.config.metrics_history_size))
        
        # System monitoring
        self.system_metrics_history: deque = deque(maxlen=self.config.metrics_history_size)
        self.last_system_check = 0.0
        
        # Create performance data directory
        if self.config.save_performance_data:
            self.performance_dir = Path(self.config.performance_data_path)
            self.performance_dir.mkdir(exist_ok=True)
        
        # Initialize GPU monitoring
        self.gpu_available = torch.cuda.is_available()
        
        self.logger.info(f"Performance monitor initialized with config: {self.config}")
    
    def start_timer(self, model_name: str) -> None:
        """
        Start timing inference for a model.
        
        Args:
            model_name: Name of the model being timed
        """
        self.timers[model_name] = time.time()
    
    def end_timer(self, model_name: str) -> float:
        """
        End timing inference for a model.
        
        Args:
            model_name: Name of the model being timed
            
        Returns:
            Inference time in seconds
        """
        if model_name not in self.timers:
            self.logger.warning(f"No timer found for model '{model_name}'")
            return 0.0
        
        inference_time = time.time() - self.timers[model_name]
        del self.timers[model_name]
        
        # Update metrics
        self._update_inference_metrics(model_name, inference_time)
        
        return inference_time
    
    def _update_inference_metrics(self, model_name: str, inference_time: float) -> None:
        """Update inference metrics for a model."""
        metrics = self.metrics[model_name]
        metrics.model_name = model_name
        metrics.inference_count += 1
        metrics.total_inference_time += inference_time
        metrics.average_inference_time = metrics.total_inference_time / metrics.inference_count
        metrics.min_inference_time = min(metrics.min_inference_time, inference_time)
        metrics.max_inference_time = max(metrics.max_inference_time, inference_time)
        metrics.throughput_fps = 1.0 / metrics.average_inference_time if metrics.average_inference_time > 0 else 0
        metrics.last_updated = time.time()
        
        # Add to history
        self.metrics_history[model_name].append({
            "timestamp": time.time(),
            "inference_time": inference_time,
            "throughput_fps": metrics.throughput_fps
        })
    
    def record_accuracy(self, model_name: str, accuracy: float) -> None:
        """
        Record accuracy for a model.
        
        Args:
            model_name: Name of the model
            accuracy: Accuracy value (0.0 to 1.0)
        """
        if model_name in self.metrics:
            self.metrics[model_name].accuracy = accuracy
    
    def record_precision_recall(self, model_name: str, precision: float, recall: float) -> None:
        """
        Record precision and recall for a model.
        
        Args:
            model_name: Name of the model
            precision: Precision value (0.0 to 1.0)
            recall: Recall value (0.0 to 1.0)
        """
        if model_name in self.metrics:
            metrics = self.metrics[model_name]
            metrics.precision = precision
            metrics.recall = recall
            # Calculate F1 score
            if precision + recall > 0:
                metrics.f1_score = 2 * (precision * recall) / (precision + recall)
    
    def record_success(self, model_name: str, success: bool) -> None:
        """
        Record success/failure for a model.
        
        Args:
            model_name: Name of the model
            success: Whether the inference was successful
        """
        if model_name in self.metrics:
            metrics = self.metrics[model_name]
            if not success:
                metrics.error_count += 1
            
            total_attempts = metrics.inference_count + metrics.error_count
            if total_attempts > 0:
                metrics.success_rate = metrics.inference_count / total_attempts
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Get comprehensive performance report.
        
        Returns:
            Dictionary containing performance statistics
        """
        report = {
            "timestamp": time.time(),
            "models": {},
            "system": self._get_system_metrics(),
            "summary": {}
        }
        
        # Model-specific metrics
        for model_name, metrics in self.metrics.items():
            report["models"][model_name] = {
                "inference_count": metrics.inference_count,
                "total_inference_time": metrics.total_inference_time,
                "average_inference_time": metrics.average_inference_time,
                "min_inference_time": metrics.min_inference_time,
                "max_inference_time": metrics.max_inference_time,
                "throughput_fps": metrics.throughput_fps,
                "accuracy": metrics.accuracy,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1_score": metrics.f1_score,
                "error_count": metrics.error_count,
                "success_rate": metrics.success_rate,
                "last_updated": metrics.last_updated
            }
        
        # Summary statistics
        if self.metrics:
            all_inference_times = [m.average_inference_time for m in self.metrics.values()]
            all_throughputs = [m.throughput_fps for m in self.metrics.values()]
            all_accuracies = [m.accuracy for m in self.metrics.values()]
            
            report["summary"] = {
                "total_models": len(self.metrics),
                "avg_inference_time": np.mean(all_inference_times),
                "avg_throughput": np.mean(all_throughputs),
                "avg_accuracy": np.mean(all_accuracies),
                "best_model": max(self.metrics.keys(), key=lambda k: self.metrics[k].accuracy),
                "fastest_model": min(self.metrics.keys(), key=lambda k: self.metrics[k].average_inference_time)
            }
        
        return report
    
    def _get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics."""
        current_time = time.time()
        
        # Check if we need to update system metrics
        if (current_time - self.last_system_check) < self.config.log_interval:
            if self.system_metrics_history:
                return self.system_metrics_history[-1]
        
        self.last_system_check = current_time
        
        metrics = SystemMetrics(timestamp=current_time)
        
        try:
            # CPU and memory
            metrics.cpu_percent = psutil.cpu_percent(interval=0.1)
            metrics.memory_percent = psutil.virtual_memory().percent
            metrics.disk_usage_percent = psutil.disk_usage('/').percent
            
            # Network I/O
            net_io = psutil.net_io_counters()
            metrics.network_io = (net_io.bytes_sent, net_io.bytes_recv)
            
            # GPU metrics
            if self.config.enable_gpu_monitoring and self.gpu_available:
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    metrics.gpu_memory_used = memory_info.used / 1024**3  # GB
                    metrics.gpu_memory_total = memory_info.total / 1024**3  # GB
                    
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    metrics.gpu_utilization = utilization.gpu
                    
                except ImportError:
                    self.logger.warning("pynvml not available for GPU monitoring")
                except Exception as e:
                    self.logger.warning(f"GPU monitoring failed: {str(e)}")
            
            # Add to history
            self.system_metrics_history.append(metrics)
            
        except Exception as e:
            self.logger.error(f"Failed to get system metrics: {str(e)}")
        
        return metrics
    
    def save_performance_data(self, filename: Optional[str] = None) -> bool:
        """
        Save performance data to file.
        
        Args:
            filename: Optional filename, defaults to timestamp-based name
            
        Returns:
            True if saved successfully, False otherwise
        """
        if not self.config.save_performance_data:
            return False
        
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"performance_report_{timestamp}.json"
            
            filepath = self.performance_dir / filename
            
            report = self.get_performance_report()
            
            # Convert to JSON-serializable format
            json_report = self._convert_to_json_serializable(report)
            
            with open(filepath, 'w') as f:
                json.dump(json_report, f, indent=2, default=str)
            
            self.logger.info(f"Performance data saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save performance data: {str(e)}")
            return False
    
    def _convert_to_json_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            return self._convert_to_json_serializable(obj.__dict__)
        else:
            return obj
    
    def check_alerts(self) -> List[Dict[str, Any]]:
        """
        Check for performance alerts based on thresholds.
        
        Returns:
            List of alert dictionaries
        """
        alerts = []
        
        # Check model performance
        for model_name, metrics in self.metrics.items():
            # Inference time alert
            if metrics.average_inference_time * 1000 > self.config.alert_thresholds.get("inference_time_ms", 1000):
                alerts.append({
                    "type": "high_inference_time",
                    "model": model_name,
                    "value": metrics.average_inference_time * 1000,
                    "threshold": self.config.alert_thresholds["inference_time_ms"],
                    "timestamp": time.time()
                })
            
            # Error rate alert
            if metrics.success_rate < (1 - self.config.alert_thresholds.get("error_rate", 0.1)):
                alerts.append({
                    "type": "high_error_rate",
                    "model": model_name,
                    "value": 1 - metrics.success_rate,
                    "threshold": self.config.alert_thresholds["error_rate"],
                    "timestamp": time.time()
                })
        
        # Check system metrics
        system_metrics = self._get_system_metrics()
        
        # Memory usage alert
        if system_metrics.memory_percent > self.config.alert_thresholds.get("memory_usage_percent", 90):
            alerts.append({
                "type": "high_memory_usage",
                "value": system_metrics.memory_percent,
                "threshold": self.config.alert_thresholds["memory_usage_percent"],
                "timestamp": time.time()
            })
        
        # CPU usage alert
        if system_metrics.cpu_percent > self.config.alert_thresholds.get("cpu_usage_percent", 90):
            alerts.append({
                "type": "high_cpu_usage",
                "value": system_metrics.cpu_percent,
                "threshold": self.config.alert_thresholds["cpu_usage_percent"],
                "timestamp": time.time()
            })
        
        return alerts
    
    def get_model_performance(self, model_name: str) -> Optional[PerformanceMetrics]:
        """Get performance metrics for a specific model."""
        return self.metrics.get(model_name)
    
    def reset_metrics(self, model_name: Optional[str] = None) -> None:
        """
        Reset performance metrics.
        
        Args:
            model_name: Specific model to reset, or None to reset all
        """
        if model_name is None:
            self.metrics.clear()
            self.metrics_history.clear()
            self.logger.info("Reset all performance metrics")
        else:
            self.metrics.pop(model_name, None)
            self.metrics_history.pop(model_name, None)
            self.logger.info(f"Reset performance metrics for model '{model_name}'")
    
    def get_metrics_history(self, model_name: str, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get historical metrics for a model.
        
        Args:
            model_name: Name of the model
            hours: Number of hours of history to retrieve
            
        Returns:
            List of historical metrics
        """
        if model_name not in self.metrics_history:
            return []
        
        cutoff_time = time.time() - (hours * 3600)
        history = list(self.metrics_history[model_name])
        
        # Filter by time
        filtered_history = [
            entry for entry in history
            if entry["timestamp"] >= cutoff_time
        ]
        
        return filtered_history
    
    def export_metrics_csv(self, filename: Optional[str] = None) -> bool:
        """
        Export metrics to CSV format.
        
        Args:
            filename: Optional filename
            
        Returns:
            True if exported successfully, False otherwise
        """
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"metrics_export_{timestamp}.csv"
            
            filepath = self.performance_dir / filename
            
            import csv
            
            with open(filepath, 'w', newline='') as csvfile:
                fieldnames = [
                    'model_name', 'timestamp', 'inference_count', 'avg_inference_time',
                    'throughput_fps', 'accuracy', 'precision', 'recall', 'f1_score',
                    'error_count', 'success_rate'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for model_name, metrics in self.metrics.items():
                    writer.writerow({
                        'model_name': model_name,
                        'timestamp': datetime.fromtimestamp(metrics.last_updated).isoformat(),
                        'inference_count': metrics.inference_count,
                        'avg_inference_time': metrics.average_inference_time,
                        'throughput_fps': metrics.throughput_fps,
                        'accuracy': metrics.accuracy,
                        'precision': metrics.precision,
                        'recall': metrics.recall,
                        'f1_score': metrics.f1_score,
                        'error_count': metrics.error_count,
                        'success_rate': metrics.success_rate
                    })
            
            self.logger.info(f"Metrics exported to CSV: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export metrics to CSV: {str(e)}")
            return False 