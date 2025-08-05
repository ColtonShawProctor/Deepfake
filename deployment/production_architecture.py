"""
Production Deployment Architecture for Multi-Model Deepfake Detection System

Comprehensive production architecture that handles:
- 100+ concurrent users with auto-scaling
- GPU resource management and optimization
- Model caching and memory management
- Security, rate limiting, and monitoring
- Cost optimization strategies

Architecture Components:
1. Container Orchestration (Kubernetes)
2. GPU Resource Management
3. Model Serving Infrastructure
4. Load Balancing and Auto-scaling
5. Database Optimization
6. Monitoring and Observability
7. Security and Rate Limiting
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import yaml
import json
from pathlib import Path

class DeploymentEnvironment(str, Enum):
    """Deployment environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class GPUProvider(str, Enum):
    """GPU cloud providers"""
    AWS = "aws"           # EC2 P3/P4 instances
    GCP = "gcp"           # A100/V100 instances
    AZURE = "azure"       # NC series
    LAMBDA = "lambda"     # Lambda Labs
    RUNPOD = "runpod"    # RunPod.io

class ModelServingFramework(str, Enum):
    """Model serving frameworks"""
    TRITON = "triton"           # NVIDIA Triton
    TORCHSERVE = "torchserve"  # PyTorch Serve
    TFSERVING = "tfserving"    # TensorFlow Serving
    CUSTOM = "custom"          # Custom FastAPI

@dataclass
class GPUResourceConfig:
    """GPU resource configuration"""
    # GPU specifications
    gpu_type: str = "nvidia-tesla-t4"  # T4, V100, A100
    gpu_memory_gb: int = 16
    gpus_per_node: int = 1
    
    # Resource allocation
    models_per_gpu: int = 4  # All 4 models on same GPU
    batch_size: int = 8
    max_concurrent_requests: int = 16
    
    # Memory optimization
    enable_mixed_precision: bool = True
    enable_model_quantization: bool = False
    enable_gradient_checkpointing: bool = True
    
    # Cost optimization
    use_spot_instances: bool = True
    spot_interruption_behavior: str = "hibernate"
    max_spot_price: float = 0.5  # $/hour

@dataclass
class AutoScalingConfig:
    """Auto-scaling configuration"""
    # Horizontal Pod Autoscaling (HPA)
    min_replicas: int = 2
    max_replicas: int = 20
    target_cpu_utilization: int = 70
    target_gpu_utilization: int = 80
    target_memory_utilization: int = 75
    
    # Vertical Pod Autoscaling (VPA)
    enable_vpa: bool = True
    min_cpu_request: str = "2"
    max_cpu_request: str = "16"
    min_memory_request: str = "8Gi"
    max_memory_request: str = "64Gi"
    
    # Cluster autoscaling
    enable_cluster_autoscaling: bool = True
    min_nodes: int = 1
    max_nodes: int = 10
    scale_down_delay: str = "10m"
    
    # Custom metrics
    scale_on_queue_size: bool = True
    queue_size_threshold: int = 100
    scale_on_response_time: bool = True
    response_time_threshold_ms: int = 1000

@dataclass
class DatabaseOptimizationConfig:
    """Database optimization configuration"""
    # Connection pooling
    connection_pool_size: int = 100
    max_overflow: int = 50
    pool_timeout: int = 30
    pool_recycle: int = 3600
    
    # Query optimization
    enable_query_caching: bool = True
    query_cache_size_mb: int = 256
    enable_prepared_statements: bool = True
    
    # Read replica configuration
    enable_read_replicas: bool = True
    read_replica_count: int = 3
    read_replica_regions: List[str] = field(default_factory=lambda: ["us-east-1", "eu-west-1", "ap-southeast-1"])
    
    # Partitioning and sharding
    enable_partitioning: bool = True
    partition_strategy: str = "time_based"  # time_based, hash_based
    retention_days: int = 90
    
    # Caching layer
    enable_redis_cache: bool = True
    redis_cluster_nodes: int = 3
    redis_memory_gb: int = 16
    cache_ttl_seconds: int = 3600

@dataclass
class SecurityConfig:
    """Security configuration"""
    # Rate limiting
    enable_rate_limiting: bool = True
    rate_limit_per_minute: int = 60
    rate_limit_per_hour: int = 1000
    rate_limit_per_day: int = 10000
    
    # Authentication
    auth_provider: str = "auth0"  # auth0, cognito, custom
    enable_api_keys: bool = True
    api_key_rotation_days: int = 90
    
    # Network security
    enable_waf: bool = True
    waf_provider: str = "cloudflare"  # cloudflare, aws_waf
    enable_ddos_protection: bool = True
    
    # Content security
    max_file_size_mb: int = 100
    allowed_file_types: List[str] = field(default_factory=lambda: ["jpg", "jpeg", "png", "mp4", "avi", "mov"])
    enable_virus_scanning: bool = True
    
    # Encryption
    enable_tls: bool = True
    tls_version: str = "1.3"
    enable_data_encryption: bool = True
    encryption_key_provider: str = "aws_kms"  # aws_kms, vault

@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration"""
    # Metrics collection
    metrics_provider: str = "prometheus"
    metrics_retention_days: int = 30
    scrape_interval_seconds: int = 15
    
    # Logging
    logging_provider: str = "elasticsearch"  # elasticsearch, cloudwatch, stackdriver
    log_retention_days: int = 30
    log_level: str = "INFO"
    enable_structured_logging: bool = True
    
    # Tracing
    enable_distributed_tracing: bool = True
    tracing_provider: str = "jaeger"  # jaeger, zipkin, datadog
    trace_sampling_rate: float = 0.1
    
    # Alerting
    alerting_provider: str = "alertmanager"  # alertmanager, pagerduty
    alert_channels: List[str] = field(default_factory=lambda: ["email", "slack", "pagerduty"])
    
    # Custom metrics
    custom_metrics: List[str] = field(default_factory=lambda: [
        "deepfake_detection_rate",
        "model_inference_time",
        "gpu_memory_usage",
        "queue_depth",
        "cache_hit_rate"
    ])

@dataclass
class ProductionConfig:
    """Complete production configuration"""
    environment: DeploymentEnvironment = DeploymentEnvironment.PRODUCTION
    gpu_provider: GPUProvider = GPUProvider.AWS
    model_serving: ModelServingFramework = ModelServingFramework.TRITON
    
    # Component configurations
    gpu_config: GPUResourceConfig = field(default_factory=GPUResourceConfig)
    autoscaling: AutoScalingConfig = field(default_factory=AutoScalingConfig)
    database: DatabaseOptimizationConfig = field(default_factory=DatabaseOptimizationConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # Cost optimization
    enable_cost_allocation_tags: bool = True
    cost_center: str = "ml-inference"
    budget_alert_threshold: float = 0.8
    
    # High availability
    multi_region_deployment: bool = True
    primary_region: str = "us-east-1"
    secondary_regions: List[str] = field(default_factory=lambda: ["us-west-2", "eu-west-1"])
    
    # Backup and disaster recovery
    enable_automated_backups: bool = True
    backup_retention_days: int = 30
    enable_cross_region_replication: bool = True

class KubernetesManifestGenerator:
    """Generate Kubernetes manifests for production deployment"""
    
    def __init__(self, config: ProductionConfig):
        self.config = config
    
    def generate_namespace(self) -> Dict[str, Any]:
        """Generate namespace manifest"""
        return {
            "apiVersion": "v1",
            "kind": "Namespace",
            "metadata": {
                "name": "deepfake-detection",
                "labels": {
                    "app": "deepfake-detection",
                    "environment": self.config.environment
                }
            }
        }
    
    def generate_model_server_deployment(self) -> Dict[str, Any]:
        """Generate model server deployment manifest"""
        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": "model-server",
                "namespace": "deepfake-detection"
            },
            "spec": {
                "replicas": self.config.autoscaling.min_replicas,
                "selector": {
                    "matchLabels": {
                        "app": "model-server"
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": "model-server",
                            "version": "v1"
                        }
                    },
                    "spec": {
                        "containers": [{
                            "name": "triton-server",
                            "image": "nvcr.io/nvidia/tritonserver:23.10-py3",
                            "command": [
                                "tritonserver",
                                "--model-repository=/models",
                                "--allow-gpu-metrics=true",
                                "--gpu-metrics-interval=1000"
                            ],
                            "ports": [{
                                "containerPort": 8000,
                                "name": "http"
                            }, {
                                "containerPort": 8001,
                                "name": "grpc"
                            }, {
                                "containerPort": 8002,
                                "name": "metrics"
                            }],
                            "resources": {
                                "requests": {
                                    "cpu": self.config.autoscaling.min_cpu_request,
                                    "memory": self.config.autoscaling.min_memory_request,
                                    "nvidia.com/gpu": str(self.config.gpu_config.gpus_per_node)
                                },
                                "limits": {
                                    "cpu": self.config.autoscaling.max_cpu_request,
                                    "memory": self.config.autoscaling.max_memory_request,
                                    "nvidia.com/gpu": str(self.config.gpu_config.gpus_per_node)
                                }
                            },
                            "volumeMounts": [{
                                "name": "model-repository",
                                "mountPath": "/models"
                            }, {
                                "name": "shared-memory",
                                "mountPath": "/dev/shm"
                            }],
                            "env": [{
                                "name": "CUDA_VISIBLE_DEVICES",
                                "value": "0"
                            }, {
                                "name": "TF_ENABLE_MIXED_PRECISION",
                                "value": str(self.config.gpu_config.enable_mixed_precision)
                            }],
                            "livenessProbe": {
                                "httpGet": {
                                    "path": "/v2/health/live",
                                    "port": 8000
                                },
                                "initialDelaySeconds": 30,
                                "periodSeconds": 10
                            },
                            "readinessProbe": {
                                "httpGet": {
                                    "path": "/v2/health/ready",
                                    "port": 8000
                                },
                                "initialDelaySeconds": 30,
                                "periodSeconds": 10
                            }
                        }],
                        "volumes": [{
                            "name": "model-repository",
                            "persistentVolumeClaim": {
                                "claimName": "model-repository-pvc"
                            }
                        }, {
                            "name": "shared-memory",
                            "emptyDir": {
                                "medium": "Memory",
                                "sizeLimit": "2Gi"
                            }
                        }],
                        "nodeSelector": {
                            "node.kubernetes.io/instance-type": self.config.gpu_config.gpu_type
                        },
                        "tolerations": [{
                            "key": "nvidia.com/gpu",
                            "operator": "Exists",
                            "effect": "NoSchedule"
                        }]
                    }
                }
            }
        }
    
    def generate_api_deployment(self) -> Dict[str, Any]:
        """Generate API server deployment manifest"""
        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": "api-server",
                "namespace": "deepfake-detection"
            },
            "spec": {
                "replicas": self.config.autoscaling.min_replicas * 2,  # More API servers than GPU nodes
                "selector": {
                    "matchLabels": {
                        "app": "api-server"
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": "api-server",
                            "version": "v1"
                        }
                    },
                    "spec": {
                        "containers": [{
                            "name": "fastapi",
                            "image": "deepfake-detection-api:latest",
                            "ports": [{
                                "containerPort": 8000,
                                "name": "http"
                            }],
                            "env": [
                                {
                                    "name": "DATABASE_URL",
                                    "valueFrom": {
                                        "secretKeyRef": {
                                            "name": "database-credentials",
                                            "key": "url"
                                        }
                                    }
                                },
                                {
                                    "name": "REDIS_URL",
                                    "valueFrom": {
                                        "secretKeyRef": {
                                            "name": "redis-credentials",
                                            "key": "url"
                                        }
                                    }
                                },
                                {
                                    "name": "MODEL_SERVER_URL",
                                    "value": "model-server-service:8000"
                                },
                                {
                                    "name": "ENABLE_RATE_LIMITING",
                                    "value": str(self.config.security.enable_rate_limiting)
                                },
                                {
                                    "name": "MAX_WORKERS",
                                    "value": "4"
                                }
                            ],
                            "resources": {
                                "requests": {
                                    "cpu": "1",
                                    "memory": "2Gi"
                                },
                                "limits": {
                                    "cpu": "4",
                                    "memory": "8Gi"
                                }
                            },
                            "livenessProbe": {
                                "httpGet": {
                                    "path": "/health",
                                    "port": 8000
                                },
                                "initialDelaySeconds": 15,
                                "periodSeconds": 10
                            },
                            "readinessProbe": {
                                "httpGet": {
                                    "path": "/ready",
                                    "port": 8000
                                },
                                "initialDelaySeconds": 15,
                                "periodSeconds": 5
                            }
                        }],
                        "affinity": {
                            "podAntiAffinity": {
                                "preferredDuringSchedulingIgnoredDuringExecution": [{
                                    "weight": 100,
                                    "podAffinityTerm": {
                                        "labelSelector": {
                                            "matchLabels": {
                                                "app": "api-server"
                                            }
                                        },
                                        "topologyKey": "kubernetes.io/hostname"
                                    }
                                }]
                            }
                        }
                    }
                }
            }
        }
    
    def generate_hpa(self, name: str, target: str) -> Dict[str, Any]:
        """Generate Horizontal Pod Autoscaler manifest"""
        return {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {
                "name": f"{name}-hpa",
                "namespace": "deepfake-detection"
            },
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": target
                },
                "minReplicas": self.config.autoscaling.min_replicas,
                "maxReplicas": self.config.autoscaling.max_replicas,
                "metrics": [
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "cpu",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": self.config.autoscaling.target_cpu_utilization
                            }
                        }
                    },
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "memory",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": self.config.autoscaling.target_memory_utilization
                            }
                        }
                    }
                ],
                "behavior": {
                    "scaleDown": {
                        "stabilizationWindowSeconds": 300,
                        "policies": [{
                            "type": "Percent",
                            "value": 10,
                            "periodSeconds": 60
                        }]
                    },
                    "scaleUp": {
                        "stabilizationWindowSeconds": 60,
                        "policies": [{
                            "type": "Percent",
                            "value": 100,
                            "periodSeconds": 60
                        }, {
                            "type": "Pods",
                            "value": 4,
                            "periodSeconds": 60
                        }]
                    }
                }
            }
        }
    
    def generate_service(self, name: str, selector: str) -> Dict[str, Any]:
        """Generate service manifest"""
        return {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": f"{name}-service",
                "namespace": "deepfake-detection",
                "annotations": {
                    "service.beta.kubernetes.io/aws-load-balancer-type": "nlb",
                    "service.beta.kubernetes.io/aws-load-balancer-cross-zone-load-balancing-enabled": "true"
                }
            },
            "spec": {
                "type": "LoadBalancer",
                "selector": {
                    "app": selector
                },
                "ports": [{
                    "port": 80,
                    "targetPort": 8000,
                    "protocol": "TCP",
                    "name": "http"
                }],
                "sessionAffinity": "ClientIP",
                "sessionAffinityConfig": {
                    "clientIP": {
                        "timeoutSeconds": 10800
                    }
                }
            }
        }
    
    def generate_redis_statefulset(self) -> Dict[str, Any]:
        """Generate Redis StatefulSet for caching"""
        return {
            "apiVersion": "apps/v1",
            "kind": "StatefulSet",
            "metadata": {
                "name": "redis-cache",
                "namespace": "deepfake-detection"
            },
            "spec": {
                "serviceName": "redis-service",
                "replicas": self.config.database.redis_cluster_nodes,
                "selector": {
                    "matchLabels": {
                        "app": "redis"
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": "redis"
                        }
                    },
                    "spec": {
                        "containers": [{
                            "name": "redis",
                            "image": "redis:7-alpine",
                            "command": [
                                "redis-server",
                                "--maxmemory", f"{self.config.database.redis_memory_gb}gb",
                                "--maxmemory-policy", "allkeys-lru",
                                "--save", "",
                                "--appendonly", "yes"
                            ],
                            "ports": [{
                                "containerPort": 6379,
                                "name": "redis"
                            }],
                            "resources": {
                                "requests": {
                                    "cpu": "0.5",
                                    "memory": f"{self.config.database.redis_memory_gb}Gi"
                                },
                                "limits": {
                                    "cpu": "2",
                                    "memory": f"{self.config.database.redis_memory_gb * 1.2}Gi"
                                }
                            },
                            "volumeMounts": [{
                                "name": "redis-data",
                                "mountPath": "/data"
                            }]
                        }]
                    }
                },
                "volumeClaimTemplates": [{
                    "metadata": {
                        "name": "redis-data"
                    },
                    "spec": {
                        "accessModes": ["ReadWriteOnce"],
                        "storageClassName": "gp3",
                        "resources": {
                            "requests": {
                                "storage": "50Gi"
                            }
                        }
                    }
                }]
            }
        }
    
    def generate_monitoring_stack(self) -> List[Dict[str, Any]]:
        """Generate monitoring stack manifests"""
        manifests = []
        
        # Prometheus configuration
        prometheus_config = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": "prometheus-config",
                "namespace": "deepfake-detection"
            },
            "data": {
                "prometheus.yml": yaml.dump({
                    "global": {
                        "scrape_interval": f"{self.config.monitoring.scrape_interval_seconds}s",
                        "evaluation_interval": "30s"
                    },
                    "scrape_configs": [
                        {
                            "job_name": "kubernetes-pods",
                            "kubernetes_sd_configs": [{
                                "role": "pod"
                            }],
                            "relabel_configs": [
                                {
                                    "source_labels": ["__meta_kubernetes_pod_annotation_prometheus_io_scrape"],
                                    "action": "keep",
                                    "regex": "true"
                                },
                                {
                                    "source_labels": ["__meta_kubernetes_pod_annotation_prometheus_io_path"],
                                    "action": "replace",
                                    "target_label": "__metrics_path__",
                                    "regex": "(.+)"
                                }
                            ]
                        },
                        {
                            "job_name": "triton-metrics",
                            "static_configs": [{
                                "targets": ["model-server-service:8002"]
                            }]
                        }
                    ]
                })
            }
        }
        manifests.append(prometheus_config)
        
        # Grafana dashboard
        grafana_dashboard = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": "grafana-dashboards",
                "namespace": "deepfake-detection"
            },
            "data": {
                "deepfake-detection.json": json.dumps({
                    "dashboard": {
                        "title": "Deepfake Detection System",
                        "panels": [
                            {
                                "title": "Request Rate",
                                "targets": [{
                                    "expr": "sum(rate(http_requests_total[5m]))"
                                }]
                            },
                            {
                                "title": "GPU Utilization",
                                "targets": [{
                                    "expr": "avg(nv_gpu_utilization)"
                                }]
                            },
                            {
                                "title": "Model Inference Time",
                                "targets": [{
                                    "expr": "histogram_quantile(0.95, model_inference_duration_seconds_bucket)"
                                }]
                            },
                            {
                                "title": "Detection Rate",
                                "targets": [{
                                    "expr": "sum(rate(deepfake_detections_total[5m]))"
                                }]
                            }
                        ]
                    }
                })
            }
        }
        manifests.append(grafana_dashboard)
        
        return manifests
    
    def generate_all_manifests(self) -> Dict[str, List[Dict[str, Any]]]:
        """Generate all Kubernetes manifests"""
        return {
            "namespace": [self.generate_namespace()],
            "deployments": [
                self.generate_model_server_deployment(),
                self.generate_api_deployment()
            ],
            "services": [
                self.generate_service("model-server", "model-server"),
                self.generate_service("api-server", "api-server")
            ],
            "autoscaling": [
                self.generate_hpa("model-server", "model-server"),
                self.generate_hpa("api-server", "api-server")
            ],
            "statefulsets": [
                self.generate_redis_statefulset()
            ],
            "monitoring": self.generate_monitoring_stack()
        }

class ModelOptimizationPipeline:
    """Optimize models for production deployment"""
    
    def __init__(self, config: ProductionConfig):
        self.config = config
    
    def generate_triton_config(self, model_name: str, input_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """Generate Triton model configuration"""
        return {
            "name": model_name,
            "platform": "pytorch_libtorch",
            "max_batch_size": self.config.gpu_config.batch_size,
            "input": [{
                "name": "input",
                "data_type": "TYPE_FP16" if self.config.gpu_config.enable_mixed_precision else "TYPE_FP32",
                "dims": list(input_shape)
            }],
            "output": [{
                "name": "output",
                "data_type": "TYPE_FP32",
                "dims": [2]  # Binary classification
            }],
            "instance_group": [{
                "count": 1,
                "kind": "KIND_GPU",
                "gpus": [0]
            }],
            "dynamic_batching": {
                "max_queue_delay_microseconds": 100000,
                "default_queue_policy": {
                    "timeout_action": "REJECT",
                    "default_timeout_microseconds": 500000,
                    "allow_timeout_override": True,
                    "max_queue_size": 100
                }
            },
            "optimization": {
                "cuda": {
                    "use_cudnn": True,
                    "cudnn_conv_algo_search": "HEURISTIC",
                    "enable_tensor_core": True
                },
                "execution_accelerators": {
                    "gpu_execution_accelerator": [{
                        "name": "tensorrt",
                        "parameters": {
                            "precision_mode": "FP16" if self.config.gpu_config.enable_mixed_precision else "FP32",
                            "max_workspace_size_bytes": str(1 << 30),  # 1GB
                            "max_cached_engines": "5"
                        }
                    }]
                }
            }
        }
    
    def generate_model_optimization_script(self) -> str:
        """Generate model optimization script"""
        return '''#!/bin/bash
# Model Optimization Script for Production Deployment

set -e

echo "Starting model optimization pipeline..."

# Function to optimize PyTorch model
optimize_pytorch_model() {
    local model_path=$1
    local output_path=$2
    
    python3 << EOF
import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic
import torch.jit as jit

# Load model
model = torch.load("${model_path}", map_location='cpu')
model.eval()

# Apply optimizations based on config
if ${ENABLE_QUANTIZATION}; then
    print("Applying dynamic quantization...")
    model = quantize_dynamic(model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8)

# TorchScript compilation
print("Compiling with TorchScript...")
example_input = torch.randn(1, 3, 224, 224)
traced_model = jit.trace(model, example_input)
traced_model = jit.optimize_for_inference(traced_model)

# Save optimized model
traced_model.save("${output_path}")
print(f"Optimized model saved to {output_path}")
EOF
}

# Function to convert to TensorRT
convert_to_tensorrt() {
    local model_path=$1
    local output_path=$2
    
    trtexec \
        --onnx=${model_path} \
        --saveEngine=${output_path} \
        --fp16 \
        --workspace=4096 \
        --maxBatch=${BATCH_SIZE} \
        --buildOnly \
        --verbose
}

# Optimize each model
for model in xception efficientnet_b4 f3net mesonet; do
    echo "Optimizing ${model}..."
    
    # PyTorch optimization
    optimize_pytorch_model "models/${model}.pth" "optimized/${model}_optimized.pt"
    
    # Export to ONNX
    python3 -m torch.onnx.export \
        --input models/${model}.pth \
        --output optimized/${model}.onnx \
        --opset 14
    
    # Convert to TensorRT if GPU available
    if command -v trtexec &> /dev/null; then
        convert_to_tensorrt "optimized/${model}.onnx" "optimized/${model}.trt"
    fi
done

echo "Model optimization complete!"
'''
    
    def generate_benchmark_script(self) -> str:
        """Generate benchmarking script"""
        return '''#!/usr/bin/env python3
"""
Production Model Benchmarking Script

Benchmarks model performance across different optimization strategies
"""

import time
import torch
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import tritonclient.http as httpclient

def benchmark_model(model_name, batch_sizes=[1, 4, 8, 16, 32], num_runs=100):
    """Benchmark model inference performance"""
    client = httpclient.InferenceServerClient(url="localhost:8000")
    
    results = []
    
    for batch_size in batch_sizes:
        # Prepare input
        input_data = np.random.randn(batch_size, 3, 224, 224).astype(np.float32)
        inputs = [httpclient.InferInput("input", input_data.shape, "FP32")]
        inputs[0].set_data_from_numpy(input_data)
        
        # Warmup
        for _ in range(10):
            client.infer(model_name, inputs)
        
        # Benchmark
        latencies = []
        for _ in range(num_runs):
            start_time = time.time()
            response = client.infer(model_name, inputs)
            latency = (time.time() - start_time) * 1000  # ms
            latencies.append(latency)
        
        # Calculate statistics
        results.append({
            'model': model_name,
            'batch_size': batch_size,
            'avg_latency_ms': np.mean(latencies),
            'p50_latency_ms': np.percentile(latencies, 50),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'throughput_fps': (batch_size * 1000) / np.mean(latencies)
        })
    
    return pd.DataFrame(results)

def benchmark_ensemble(batch_size=8, num_concurrent=10):
    """Benchmark ensemble performance with concurrent requests"""
    
    def make_ensemble_request():
        # Simulate full ensemble inference
        models = ['xception', 'efficientnet_b4', 'f3net', 'mesonet']
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(benchmark_single_inference, model) for model in models]
            results = [f.result() for f in futures]
        
        return time.time() - start_time
    
    # Run concurrent benchmark
    with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
        latencies = list(executor.map(lambda _: make_ensemble_request(), range(100)))
    
    print(f"Ensemble Performance (batch_size={batch_size}, concurrent={num_concurrent}):")
    print(f"  Average latency: {np.mean(latencies)*1000:.2f} ms")
    print(f"  P95 latency: {np.percentile(latencies, 95)*1000:.2f} ms")
    print(f"  Throughput: {num_concurrent/np.mean(latencies):.2f} req/s")

if __name__ == "__main__":
    # Benchmark individual models
    for model in ['xception', 'efficientnet_b4', 'f3net', 'mesonet']:
        print(f"\\nBenchmarking {model}...")
        results = benchmark_model(model)
        print(results.to_string())
    
    # Benchmark ensemble
    print("\\nBenchmarking ensemble...")
    benchmark_ensemble()
'''

class CostOptimizationStrategy:
    """Cost optimization strategies for GPU resources"""
    
    def __init__(self, config: ProductionConfig):
        self.config = config
    
    def calculate_monthly_cost(self) -> Dict[str, float]:
        """Calculate estimated monthly costs"""
        # GPU instance costs (approximate)
        gpu_costs = {
            "nvidia-tesla-t4": 0.526,      # $/hour
            "nvidia-tesla-v100": 3.06,     # $/hour
            "nvidia-tesla-a100": 5.12,     # $/hour
        }
        
        # Base cost calculation
        gpu_cost_per_hour = gpu_costs.get(self.config.gpu_config.gpu_type, 1.0)
        
        if self.config.gpu_config.use_spot_instances:
            gpu_cost_per_hour *= 0.3  # ~70% discount for spot
        
        # Calculate based on autoscaling
        avg_nodes = (self.config.autoscaling.min_nodes + self.config.autoscaling.max_nodes) / 2
        avg_gpus = avg_nodes * self.config.gpu_config.gpus_per_node
        
        monthly_gpu_cost = gpu_cost_per_hour * 24 * 30 * avg_gpus
        
        # Additional costs
        storage_cost = 0.10 * 1000  # $0.10/GB * 1000GB
        network_cost = 0.09 * 5000  # $0.09/GB * 5TB egress
        database_cost = 500  # RDS/Cloud SQL estimate
        
        return {
            "gpu_compute": monthly_gpu_cost,
            "storage": storage_cost,
            "network": network_cost,
            "database": database_cost,
            "total": monthly_gpu_cost + storage_cost + network_cost + database_cost
        }
    
    def generate_cost_optimization_recommendations(self) -> List[str]:
        """Generate cost optimization recommendations"""
        recommendations = []
        
        if not self.config.gpu_config.use_spot_instances:
            recommendations.append("Enable spot instances for 70% cost savings")
        
        if self.config.gpu_config.gpu_type == "nvidia-tesla-a100":
            recommendations.append("Consider T4 GPUs for inference (10x cheaper)")
        
        if not self.config.gpu_config.enable_mixed_precision:
            recommendations.append("Enable mixed precision for 2x performance")
        
        if self.config.autoscaling.min_replicas > 2:
            recommendations.append("Reduce minimum replicas during off-peak hours")
        
        if not self.config.database.enable_redis_cache:
            recommendations.append("Enable Redis caching to reduce database load")
        
        return recommendations

def generate_production_architecture(config: Optional[ProductionConfig] = None) -> Dict[str, Any]:
    """Generate complete production architecture"""
    if config is None:
        config = ProductionConfig()
    
    # Generate Kubernetes manifests
    k8s_generator = KubernetesManifestGenerator(config)
    manifests = k8s_generator.generate_all_manifests()
    
    # Generate model optimization pipeline
    model_optimizer = ModelOptimizationPipeline(config)
    optimization_script = model_optimizer.generate_model_optimization_script()
    benchmark_script = model_optimizer.generate_benchmark_script()
    
    # Calculate costs
    cost_optimizer = CostOptimizationStrategy(config)
    monthly_costs = cost_optimizer.calculate_monthly_cost()
    cost_recommendations = cost_optimizer.generate_cost_optimization_recommendations()
    
    return {
        "kubernetes_manifests": manifests,
        "optimization_scripts": {
            "model_optimization": optimization_script,
            "benchmarking": benchmark_script
        },
        "cost_analysis": {
            "monthly_costs": monthly_costs,
            "recommendations": cost_recommendations
        },
        "deployment_config": config
    }

if __name__ == "__main__":
    # Generate production architecture
    architecture = generate_production_architecture()
    
    # Save manifests
    output_dir = Path("deployment/manifests")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for category, manifests in architecture["kubernetes_manifests"].items():
        category_dir = output_dir / category
        category_dir.mkdir(exist_ok=True)
        
        for i, manifest in enumerate(manifests):
            filename = f"{manifest.get('metadata', {}).get('name', f'{category}_{i}')}.yaml"
            with open(category_dir / filename, 'w') as f:
                yaml.dump(manifest, f, default_flow_style=False)
    
    print(f"Production architecture generated in {output_dir}")
    print(f"Estimated monthly cost: ${architecture['cost_analysis']['monthly_costs']['total']:,.2f}")
    print("\nCost optimization recommendations:")
    for rec in architecture['cost_analysis']['recommendations']:
        print(f"  - {rec}")