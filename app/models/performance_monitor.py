"""
Performance Monitoring Integration for Advanced Ensemble Detection

This module implements comprehensive performance monitoring, alerting,
and continuous optimization for the deepfake detection system.
"""

import asyncio
import logging
import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import numpy as np
import psutil
import json
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics
from pathlib import Path

from .base_detector import DetectionResult


class MetricType(str, Enum):
    """Types of performance metrics."""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ACCURACY = "accuracy"
    MEMORY = "memory"
    CPU = "cpu"
    GPU = "gpu"
    ERROR_RATE = "error_rate"
    CACHE_HIT_RATE = "cache_hit_rate"
    MODEL_PERFORMANCE = "model_performance"
    ENSEMBLE_EFFICIENCY = "ensemble_efficiency"


class AlertLevel(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricData:
    """Performance metric data point."""
    metric_type: MetricType
    value: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class Alert:
    """Performance alert."""
    alert_id: str
    level: AlertLevel
    metric_type: MetricType
    message: str
    value: float
    threshold: float
    timestamp: float
    resolved: bool = False
    resolved_at: Optional[float] = None


@dataclass
class PerformanceReport:
    """Comprehensive performance report."""
    report_id: str
    start_time: float
    end_time: float
    duration: float
    metrics: Dict[MetricType, Dict[str, Any]]
    alerts: List[Alert]
    recommendations: List[str]
    overall_health: str


class MetricsCollector:
    """
    Comprehensive metrics collector for performance monitoring.
    Tracks system performance, model performance, and optimization metrics.
    """
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.logger = logging.getLogger(f"{__name__}.MetricsCollector")
        
        # Metric storage
        self.metrics: Dict[MetricType, deque] = {
            metric_type: deque(maxlen=max_history) 
            for metric_type in MetricType
        }
        
        # Performance tracking
        self.total_requests = 0
        self.total_processing_time = 0.0
        self.total_errors = 0
        self.model_performance_history: Dict[str, List[float]] = defaultdict(list)
        self.ensemble_performance_history: List[float] = []
        
        # System monitoring
        self.system_metrics = {
            'cpu_percent': deque(maxlen=1000),
            'memory_percent': deque(maxlen=1000),
            'gpu_memory_percent': deque(maxlen=1000),
            'disk_io': deque(maxlen=1000)
        }
        
        # Start background monitoring
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitor_system, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("MetricsCollector initialized with background monitoring")
    
    def record_metric(self, metric_type: MetricType, value: float, 
                     metadata: Dict[str, Any] = None, tags: Dict[str, str] = None):
        """Record a performance metric."""
        metric_data = MetricData(
            metric_type=metric_type,
            value=value,
            timestamp=time.time(),
            metadata=metadata or {},
            tags=tags or {}
        )
        
        self.metrics[metric_type].append(metric_data)
        
        # Update aggregated metrics
        self._update_aggregated_metrics(metric_type, value, metadata)
    
    def record_request(self, processing_time: float, success: bool, 
                      model_results: Dict[str, DetectionResult] = None,
                      metadata: Dict[str, Any] = None):
        """Record a complete request processing."""
        self.total_requests += 1
        self.total_processing_time += processing_time
        
        if not success:
            self.total_errors += 1
        
        # Record latency
        self.record_metric(MetricType.LATENCY, processing_time, metadata)
        
        # Record throughput (requests per second)
        if self.total_requests > 0:
            avg_processing_time = self.total_processing_time / self.total_requests
            throughput = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
            self.record_metric(MetricType.THROUGHPUT, throughput, metadata)
        
        # Record model performance
        if model_results:
            for model_name, result in model_results.items():
                self.model_performance_history[model_name].append(result.confidence)
                self.record_metric(
                    MetricType.MODEL_PERFORMANCE, 
                    result.confidence,
                    {"model_name": model_name, "inference_time": result.inference_time}
                )
        
        # Record error rate
        error_rate = (self.total_errors / self.total_requests) * 100 if self.total_requests > 0 else 0
        self.record_metric(MetricType.ERROR_RATE, error_rate, metadata)
    
    def record_ensemble_performance(self, accuracy: float, efficiency: float, 
                                  metadata: Dict[str, Any] = None):
        """Record ensemble performance metrics."""
        self.ensemble_performance_history.append(accuracy)
        self.record_metric(MetricType.ACCURACY, accuracy, metadata)
        self.record_metric(MetricType.ENSEMBLE_EFFICIENCY, efficiency, metadata)
    
    def record_cache_performance(self, hit_rate: float, metadata: Dict[str, Any] = None):
        """Record cache performance metrics."""
        self.record_metric(MetricType.CACHE_HIT_RATE, hit_rate, metadata)
    
    def _update_aggregated_metrics(self, metric_type: MetricType, value: float, metadata: Dict[str, Any]):
        """Update aggregated metrics based on new data."""
        # This could include more sophisticated aggregation logic
        pass
    
    def _monitor_system(self):
        """Background system monitoring thread."""
        while self.monitoring_active:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.system_metrics['cpu_percent'].append(cpu_percent)
                self.record_metric(MetricType.CPU, cpu_percent, {"type": "cpu_percent"})
                
                # Memory usage
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                self.system_metrics['memory_percent'].append(memory_percent)
                self.record_metric(MetricType.MEMORY, memory_percent, {
                    "type": "memory_percent",
                    "available_gb": memory.available / (1024**3),
                    "used_gb": memory.used / (1024**3)
                })
                
                # GPU monitoring (if available)
                try:
                    import torch
                    if torch.cuda.is_available():
                        gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
                        self.system_metrics['gpu_memory_percent'].append(gpu_memory)
                        self.record_metric(MetricType.GPU, gpu_memory, {"type": "gpu_memory_percent"})
                except ImportError:
                    pass
                
                time.sleep(5)  # Monitor every 5 seconds
                
            except Exception as e:
                self.logger.error(f"System monitoring error: {str(e)}")
                time.sleep(10)  # Wait longer on error
    
    def get_metric_summary(self, metric_type: MetricType, 
                          duration_minutes: int = 60) -> Dict[str, Any]:
        """Get summary statistics for a metric type."""
        cutoff_time = time.time() - (duration_minutes * 60)
        
        # Filter metrics by time
        recent_metrics = [
            m for m in self.metrics[metric_type] 
            if m.timestamp >= cutoff_time
        ]
        
        if not recent_metrics:
            return {"count": 0, "avg": 0, "min": 0, "max": 0, "std": 0}
        
        values = [m.value for m in recent_metrics]
        
        return {
            "count": len(values),
            "avg": statistics.mean(values),
            "min": min(values),
            "max": max(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0,
            "latest": values[-1] if values else 0,
            "trend": self._calculate_trend(values)
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a series of values."""
        if len(values) < 2:
            return "stable"
        
        # Simple linear trend calculation
        x = list(range(len(values)))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.1:
            return "increasing"
        elif slope < -0.1:
            return "decreasing"
        else:
            return "stable"
    
    def get_performance_report(self, duration_minutes: int = 60) -> PerformanceReport:
        """Generate comprehensive performance report."""
        start_time = time.time() - (duration_minutes * 60)
        end_time = time.time()
        
        # Collect all metrics
        metrics_summary = {}
        for metric_type in MetricType:
            metrics_summary[metric_type.value] = self.get_metric_summary(metric_type, duration_minutes)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(metrics_summary)
        
        # Calculate overall health
        overall_health = self._calculate_overall_health(metrics_summary)
        
        return PerformanceReport(
            report_id=f"report_{int(time.time())}",
            start_time=start_time,
            end_time=end_time,
            duration=duration_minutes * 60,
            metrics=metrics_summary,
            alerts=[],  # Will be populated by AlertingSystem
            recommendations=recommendations,
            overall_health=overall_health
        )
    
    def _generate_recommendations(self, metrics_summary: Dict[str, Any]) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # Check latency
        latency = metrics_summary.get('latency', {})
        if latency.get('avg', 0) > 1.0:  # > 1 second
            recommendations.append("High latency detected. Consider enabling parallel processing or model warmup.")
        
        # Check throughput
        throughput = metrics_summary.get('throughput', {})
        if throughput.get('avg', 0) < 1.0:  # < 1 request per second
            recommendations.append("Low throughput detected. Consider optimizing preprocessing or increasing concurrency.")
        
        # Check error rate
        error_rate = metrics_summary.get('error_rate', {})
        if error_rate.get('avg', 0) > 5.0:  # > 5% error rate
            recommendations.append("High error rate detected. Check model stability and resource allocation.")
        
        # Check memory usage
        memory = metrics_summary.get('memory', {})
        if memory.get('avg', 0) > 80.0:  # > 80% memory usage
            recommendations.append("High memory usage detected. Consider enabling garbage collection or reducing batch sizes.")
        
        # Check CPU usage
        cpu = metrics_summary.get('cpu', {})
        if cpu.get('avg', 0) > 90.0:  # > 90% CPU usage
            recommendations.append("High CPU usage detected. Consider scaling horizontally or optimizing algorithms.")
        
        return recommendations
    
    def _calculate_overall_health(self, metrics_summary: Dict[str, Any]) -> str:
        """Calculate overall system health score."""
        health_score = 100
        
        # Penalize for high latency
        latency = metrics_summary.get('latency', {})
        if latency.get('avg', 0) > 2.0:
            health_score -= 30
        elif latency.get('avg', 0) > 1.0:
            health_score -= 15
        
        # Penalize for high error rate
        error_rate = metrics_summary.get('error_rate', {})
        if error_rate.get('avg', 0) > 10.0:
            health_score -= 40
        elif error_rate.get('avg', 0) > 5.0:
            health_score -= 20
        
        # Penalize for high resource usage
        memory = metrics_summary.get('memory', {})
        if memory.get('avg', 0) > 90.0:
            health_score -= 20
        elif memory.get('avg', 0) > 80.0:
            health_score -= 10
        
        cpu = metrics_summary.get('cpu', {})
        if cpu.get('avg', 0) > 95.0:
            health_score -= 20
        elif cpu.get('avg', 0) > 85.0:
            health_score -= 10
        
        if health_score >= 90:
            return "excellent"
        elif health_score >= 70:
            return "good"
        elif health_score >= 50:
            return "fair"
        else:
            return "poor"
    
    def get_model_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for individual models."""
        stats = {}
        
        for model_name, performances in self.model_performance_history.items():
            if performances:
                stats[model_name] = {
                    "count": len(performances),
                    "avg_confidence": statistics.mean(performances),
                    "min_confidence": min(performances),
                    "max_confidence": max(performances),
                    "std_confidence": statistics.stdev(performances) if len(performances) > 1 else 0
                }
        
        return stats
    
    def export_metrics(self, filepath: str, duration_minutes: int = 60):
        """Export metrics to JSON file."""
        cutoff_time = time.time() - (duration_minutes * 60)
        
        export_data = {
            "export_time": time.time(),
            "duration_minutes": duration_minutes,
            "metrics": {},
            "system_metrics": {k: list(v) for k, v in self.system_metrics.items()},
            "model_performance": self.get_model_performance_stats()
        }
        
        for metric_type, metrics in self.metrics.items():
            recent_metrics = [
                {
                    "value": m.value,
                    "timestamp": m.timestamp,
                    "metadata": m.metadata,
                    "tags": m.tags
                }
                for m in metrics if m.timestamp >= cutoff_time
            ]
            export_data["metrics"][metric_type.value] = recent_metrics
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Metrics exported to {filepath}")
    
    def cleanup(self):
        """Clean up resources and stop monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        self.logger.info("MetricsCollector cleaned up")


class AlertingSystem:
    """
    Performance alerting system that monitors metrics and generates alerts
    when thresholds are exceeded.
    """
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.logger = logging.getLogger(f"{__name__}.AlertingSystem")
        
        # Alert thresholds
        self.thresholds = {
            MetricType.LATENCY: {"warning": 1.0, "error": 2.0, "critical": 5.0},
            MetricType.THROUGHPUT: {"warning": 0.5, "error": 0.2, "critical": 0.1},
            MetricType.ERROR_RATE: {"warning": 5.0, "error": 10.0, "critical": 20.0},
            MetricType.MEMORY: {"warning": 80.0, "error": 90.0, "critical": 95.0},
            MetricType.CPU: {"warning": 85.0, "error": 95.0, "critical": 98.0},
            MetricType.GPU: {"warning": 85.0, "error": 95.0, "critical": 98.0}
        }
        
        # Active alerts
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        
        # Alert callbacks
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        
        # Start monitoring
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitor_alerts, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("AlertingSystem initialized")
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add callback function for alert notifications."""
        self.alert_callbacks.append(callback)
    
    def set_threshold(self, metric_type: MetricType, level: str, value: float):
        """Set alert threshold for a metric type."""
        if metric_type not in self.thresholds:
            self.thresholds[metric_type] = {}
        
        self.thresholds[metric_type][level] = value
        self.logger.info(f"Set {metric_type.value} {level} threshold to {value}")
    
    def _monitor_alerts(self):
        """Background alert monitoring thread."""
        while self.monitoring_active:
            try:
                self._check_alerts()
                time.sleep(10)  # Check every 10 seconds
            except Exception as e:
                self.logger.error(f"Alert monitoring error: {str(e)}")
                time.sleep(30)  # Wait longer on error
    
    def _check_alerts(self):
        """Check all metrics for threshold violations."""
        for metric_type in self.thresholds:
            if metric_type not in self.metrics_collector.metrics:
                continue
            
            # Get latest metric value
            metrics = self.metrics_collector.metrics[metric_type]
            if not metrics:
                continue
            
            latest_metric = metrics[-1]
            value = latest_metric.value
            thresholds = self.thresholds[metric_type]
            
            # Check each threshold level
            for level, threshold in thresholds.items():
                if value >= threshold:
                    alert_id = f"{metric_type.value}_{level}_{int(latest_metric.timestamp)}"
                    
                    # Check if alert already exists
                    if alert_id not in self.active_alerts:
                        alert = Alert(
                            alert_id=alert_id,
                            level=AlertLevel(level),
                            metric_type=metric_type,
                            message=f"{metric_type.value} exceeded {level} threshold: {value:.2f} >= {threshold}",
                            value=value,
                            threshold=threshold,
                            timestamp=latest_metric.timestamp
                        )
                        
                        self.active_alerts[alert_id] = alert
                        self.alert_history.append(alert)
                        
                        # Notify callbacks
                        for callback in self.alert_callbacks:
                            try:
                                callback(alert)
                            except Exception as e:
                                self.logger.error(f"Alert callback error: {str(e)}")
                        
                        self.logger.warning(f"Alert triggered: {alert.message}")
    
    def resolve_alert(self, alert_id: str):
        """Resolve an active alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = time.time()
            del self.active_alerts[alert_id]
            self.logger.info(f"Alert resolved: {alert_id}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get list of active alerts."""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history for specified hours."""
        cutoff_time = time.time() - (hours * 3600)
        return [alert for alert in self.alert_history if alert.timestamp >= cutoff_time]
    
    def cleanup(self):
        """Clean up resources and stop monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        self.logger.info("AlertingSystem cleaned up")


class AutoOptimizationEngine:
    """
    Automatic optimization engine that continuously tunes system parameters
    based on performance metrics and alerts.
    """
    
    def __init__(self, metrics_collector: MetricsCollector, alerting_system: AlertingSystem):
        self.metrics_collector = metrics_collector
        self.alerting_system = alerting_system
        self.logger = logging.getLogger(f"{__name__}.AutoOptimizationEngine")
        
        # Optimization parameters
        self.optimization_enabled = True
        self.optimization_interval = 300  # 5 minutes
        self.last_optimization = 0
        
        # Optimization history
        self.optimization_history: List[Dict[str, Any]] = []
        
        # Start optimization loop
        self.optimization_active = True
        self.optimization_thread = threading.Thread(target=self._optimization_loop, daemon=True)
        self.optimization_thread.start()
        
        self.logger.info("AutoOptimizationEngine initialized")
    
    def _optimization_loop(self):
        """Main optimization loop."""
        while self.optimization_active:
            try:
                if self.optimization_enabled:
                    current_time = time.time()
                    if current_time - self.last_optimization >= self.optimization_interval:
                        self._perform_optimization()
                        self.last_optimization = current_time
                
                time.sleep(60)  # Check every minute
            except Exception as e:
                self.logger.error(f"Optimization loop error: {str(e)}")
                time.sleep(300)  # Wait 5 minutes on error
    
    def _perform_optimization(self):
        """Perform automatic optimization based on current metrics."""
        try:
            # Get recent performance report
            report = self.metrics_collector.get_performance_report(duration_minutes=30)
            
            # Get active alerts
            active_alerts = self.alerting_system.get_active_alerts()
            
            optimizations = []
            
            # Optimize based on latency
            latency_metrics = report.metrics.get('latency', {})
            if latency_metrics.get('avg', 0) > 1.0:
                optimizations.append({
                    "type": "enable_parallel_processing",
                    "reason": "High latency detected",
                    "priority": "high"
                })
            
            # Optimize based on memory usage
            memory_metrics = report.metrics.get('memory', {})
            if memory_metrics.get('avg', 0) > 85.0:
                optimizations.append({
                    "type": "reduce_batch_size",
                    "reason": "High memory usage",
                    "priority": "medium"
                })
            
            # Optimize based on error rate
            error_metrics = report.metrics.get('error_rate', {})
            if error_metrics.get('avg', 0) > 5.0:
                optimizations.append({
                    "type": "enable_model_warmup",
                    "reason": "High error rate",
                    "priority": "high"
                })
            
            # Record optimization attempt
            optimization_record = {
                "timestamp": time.time(),
                "optimizations": optimizations,
                "performance_before": {
                    "latency": latency_metrics.get('avg', 0),
                    "memory": memory_metrics.get('avg', 0),
                    "error_rate": error_metrics.get('avg', 0)
                }
            }
            
            self.optimization_history.append(optimization_record)
            
            if optimizations:
                self.logger.info(f"Auto-optimization suggested {len(optimizations)} optimizations")
                for opt in optimizations:
                    self.logger.info(f"  - {opt['type']}: {opt['reason']} (priority: {opt['priority']})")
            else:
                self.logger.info("No optimizations needed")
                
        except Exception as e:
            self.logger.error(f"Optimization error: {str(e)}")
    
    def get_optimization_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get optimization history for specified hours."""
        cutoff_time = time.time() - (hours * 3600)
        return [opt for opt in self.optimization_history if opt['timestamp'] >= cutoff_time]
    
    def enable_optimization(self):
        """Enable automatic optimization."""
        self.optimization_enabled = True
        self.logger.info("Auto-optimization enabled")
    
    def disable_optimization(self):
        """Disable automatic optimization."""
        self.optimization_enabled = False
        self.logger.info("Auto-optimization disabled")
    
    def cleanup(self):
        """Clean up resources and stop optimization."""
        self.optimization_active = False
        if self.optimization_thread.is_alive():
            self.optimization_thread.join(timeout=5)
        self.logger.info("AutoOptimizationEngine cleaned up")


class PerformanceMonitoringManager:
    """
    High-level manager for performance monitoring, alerting, and optimization.
    Coordinates all monitoring components and provides unified interface.
    """
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.alerting_system = AlertingSystem(self.metrics_collector)
        self.auto_optimization = AutoOptimizationEngine(self.metrics_collector, self.alerting_system)
        self.logger = logging.getLogger(f"{__name__}.PerformanceMonitoringManager")
        
        self.logger.info("PerformanceMonitoringManager initialized")
    
    def record_request(self, processing_time: float, success: bool, 
                      model_results: Dict[str, DetectionResult] = None,
                      metadata: Dict[str, Any] = None):
        """Record a request for monitoring."""
        self.metrics_collector.record_request(processing_time, success, model_results, metadata)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for performance dashboard."""
        report = self.metrics_collector.get_performance_report(duration_minutes=60)
        active_alerts = self.alerting_system.get_active_alerts()
        model_stats = self.metrics_collector.get_model_performance_stats()
        
        return {
            "overall_health": report.overall_health,
            "metrics": report.metrics,
            "active_alerts": [
                {
                    "id": alert.alert_id,
                    "level": alert.level.value,
                    "message": alert.message,
                    "timestamp": alert.timestamp
                }
                for alert in active_alerts
            ],
            "model_performance": model_stats,
            "recommendations": report.recommendations,
            "system_info": {
                "total_requests": self.metrics_collector.total_requests,
                "total_errors": self.metrics_collector.total_errors,
                "uptime": time.time() - self.metrics_collector.metrics[MetricType.LATENCY][0].timestamp if self.metrics_collector.metrics[MetricType.LATENCY] else 0
            }
        }
    
    def export_performance_data(self, filepath: str, duration_minutes: int = 60):
        """Export comprehensive performance data."""
        self.metrics_collector.export_metrics(filepath, duration_minutes)
        
        # Add alert data
        alert_data = {
            "active_alerts": [
                {
                    "id": alert.alert_id,
                    "level": alert.level.value,
                    "metric_type": alert.metric_type.value,
                    "message": alert.message,
                    "value": alert.value,
                    "threshold": alert.threshold,
                    "timestamp": alert.timestamp,
                    "resolved": alert.resolved
                }
                for alert in self.alerting_system.get_active_alerts()
            ],
            "alert_history": [
                {
                    "id": alert.alert_id,
                    "level": alert.level.value,
                    "metric_type": alert.metric_type.value,
                    "message": alert.message,
                    "value": alert.value,
                    "threshold": alert.threshold,
                    "timestamp": alert.timestamp,
                    "resolved": alert.resolved,
                    "resolved_at": alert.resolved_at
                }
                for alert in self.alerting_system.get_alert_history(24)
            ]
        }
        
        # Append alert data to existing file
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            data.update(alert_data)
        except (FileNotFoundError, json.JSONDecodeError):
            data = alert_data
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Performance data exported to {filepath}")
    
    def cleanup(self):
        """Clean up all monitoring components."""
        self.auto_optimization.cleanup()
        self.alerting_system.cleanup()
        self.metrics_collector.cleanup()
        self.logger.info("PerformanceMonitoringManager cleaned up")