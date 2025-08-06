"""
Production Training Pipeline for Deepfake Detection

This module provides a production-ready training pipeline including:
- Automated training scripts with proper error handling
- Model versioning and artifact management
- Training data validation and quality checks
- Training pipeline integration with existing detection system
- Model deployment pipeline after training completion
- Monitoring and alerting for training processes
"""

import os
import sys
import logging
import json
import time
import shutil
import subprocess
import signal
import threading
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd
import torch
import yaml
import hashlib
import uuid
from datetime import datetime, timedelta
import schedule
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests

# Add the app directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from training.dataset_management import DatasetManager
from training.model_trainer import ModelTrainer, TrainingConfig
from training.ensemble_coordinator import EnsembleCoordinator, EnsembleConfig
from training.experiment_tracker import ExperimentTracker, ExperimentConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProductionConfig:
    """Configuration for production training pipeline"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load production configuration"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        
        # Default configuration
        return {
            "pipeline": {
                "name": "deepfake_detection_training_pipeline",
                "version": "1.0.0",
                "environment": "production",
                "max_retries": 3,
                "timeout_hours": 24,
                "checkpoint_interval": 3600,  # seconds
                "backup_interval": 7200  # seconds
            },
            "data_validation": {
                "enable_validation": True,
                "min_samples": 1000,
                "max_corruption_rate": 0.05,
                "quality_thresholds": {
                    "min_resolution": (64, 64),
                    "max_file_size_mb": 10,
                    "supported_formats": ["jpg", "jpeg", "png"]
                },
                "label_validation": {
                    "check_balance": True,
                    "min_class_ratio": 0.1,
                    "max_class_ratio": 0.9
                }
            },
            "model_management": {
                "versioning": True,
                "artifact_storage": "local",  # or "s3", "gcs"
                "backup_strategy": "incremental",
                "retention_days": 30,
                "model_registry": {
                    "enabled": True,
                    "registry_path": "model_registry",
                    "metadata_tracking": True
                }
            },
            "deployment": {
                "auto_deploy": True,
                "deployment_target": "local",  # or "kubernetes", "cloud"
                "health_check": True,
                "rollback_enabled": True,
                "deployment_timeout": 300  # seconds
            },
            "monitoring": {
                "enable_monitoring": True,
                "metrics_collection": True,
                "alerting": {
                    "enabled": True,
                    "email_alerts": True,
                    "slack_alerts": False,
                    "alert_thresholds": {
                        "training_failure": True,
                        "data_quality_issues": True,
                        "model_performance_degradation": 0.1
                    }
                },
                "logging": {
                    "level": "INFO",
                    "file_logging": True,
                    "log_retention_days": 7
                }
            },
            "notifications": {
                "email": {
                    "enabled": True,
                    "smtp_server": "localhost",
                    "smtp_port": 587,
                    "from_email": "training@deepfake-detection.com",
                    "to_emails": ["admin@deepfake-detection.com"]
                },
                "slack": {
                    "enabled": False,
                    "webhook_url": "",
                    "channel": "#training-alerts"
                }
            }
        }

class DataValidator:
    """Validates training data quality and integrity"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.validation_results = {}
    
    def validate_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """Comprehensive dataset validation"""
        logger.info(f"Validating dataset: {dataset_path}")
        
        validation_results = {
            "dataset_path": dataset_path,
            "validation_time": datetime.now().isoformat(),
            "passed": True,
            "issues": [],
            "statistics": {}
        }
        
        try:
            # Basic dataset structure validation
            if not self._validate_structure(dataset_path):
                validation_results["passed"] = False
                validation_results["issues"].append("Dataset structure validation failed")
            
            # Data quality validation
            quality_results = self._validate_data_quality(dataset_path)
            validation_results["statistics"].update(quality_results)
            
            if not quality_results.get("passed", True):
                validation_results["passed"] = False
                validation_results["issues"].append("Data quality validation failed")
            
            # Label validation
            label_results = self._validate_labels(dataset_path)
            validation_results["statistics"].update(label_results)
            
            if not label_results.get("passed", True):
                validation_results["passed"] = False
                validation_results["issues"].append("Label validation failed")
            
            # File integrity validation
            integrity_results = self._validate_file_integrity(dataset_path)
            validation_results["statistics"].update(integrity_results)
            
            if not integrity_results.get("passed", True):
                validation_results["passed"] = False
                validation_results["issues"].append("File integrity validation failed")
            
        except Exception as e:
            validation_results["passed"] = False
            validation_results["issues"].append(f"Validation error: {str(e)}")
            logger.error(f"Dataset validation failed: {e}")
        
        self.validation_results = validation_results
        return validation_results
    
    def _validate_structure(self, dataset_path: str) -> bool:
        """Validate dataset directory structure"""
        dataset_dir = Path(dataset_path)
        
        # Check if directory exists
        if not dataset_dir.exists():
            logger.error(f"Dataset directory does not exist: {dataset_path}")
            return False
        
        # Check for required files/directories
        required_items = ["metadata.json", "splits.json"]
        for item in required_items:
            if not (dataset_dir / item).exists():
                logger.warning(f"Required file not found: {item}")
        
        # Check for image directories
        image_dirs = ["real", "fake"]
        for img_dir in image_dirs:
            if not (dataset_dir / img_dir).exists():
                logger.warning(f"Image directory not found: {img_dir}")
        
        return True
    
    def _validate_data_quality(self, dataset_path: str) -> Dict[str, Any]:
        """Validate data quality metrics"""
        from PIL import Image
        import cv2
        
        dataset_dir = Path(dataset_path)
        quality_stats = {
            "total_files": 0,
            "valid_files": 0,
            "corrupted_files": 0,
            "resolution_stats": [],
            "file_size_stats": [],
            "format_stats": {},
            "passed": True
        }
        
        # Scan all image files
        image_extensions = self.config["quality_thresholds"]["supported_formats"]
        min_resolution = self.config["quality_thresholds"]["min_resolution"]
        max_file_size = self.config["quality_thresholds"]["max_file_size_mb"] * 1024 * 1024
        
        for ext in image_extensions:
            for img_file in dataset_dir.rglob(f"*.{ext}"):
                quality_stats["total_files"] += 1
                
                try:
                    # Check file size
                    file_size = img_file.stat().st_size
                    quality_stats["file_size_stats"].append(file_size)
                    
                    if file_size > max_file_size:
                        quality_stats["corrupted_files"] += 1
                        logger.warning(f"File too large: {img_file}")
                        continue
                    
                    # Check image integrity
                    with Image.open(img_file) as img:
                        width, height = img.size
                        quality_stats["resolution_stats"].append((width, height))
                        
                        if width < min_resolution[0] or height < min_resolution[1]:
                            quality_stats["corrupted_files"] += 1
                            logger.warning(f"Image too small: {img_file}")
                            continue
                        
                        # Check format
                        format_name = img.format.lower()
                        quality_stats["format_stats"][format_name] = quality_stats["format_stats"].get(format_name, 0) + 1
                    
                    quality_stats["valid_files"] += 1
                    
                except Exception as e:
                    quality_stats["corrupted_files"] += 1
                    logger.warning(f"Corrupted file: {img_file} - {e}")
        
        # Check corruption rate
        if quality_stats["total_files"] > 0:
            corruption_rate = quality_stats["corrupted_files"] / quality_stats["total_files"]
            if corruption_rate > self.config["max_corruption_rate"]:
                quality_stats["passed"] = False
                logger.error(f"Corruption rate too high: {corruption_rate:.2%}")
        
        # Check minimum samples
        if quality_stats["valid_files"] < self.config["min_samples"]:
            quality_stats["passed"] = False
            logger.error(f"Insufficient valid samples: {quality_stats['valid_files']}")
        
        return quality_stats
    
    def _validate_labels(self, dataset_path: str) -> Dict[str, Any]:
        """Validate label distribution and balance"""
        dataset_dir = Path(dataset_path)
        label_stats = {
            "total_samples": 0,
            "real_samples": 0,
            "fake_samples": 0,
            "real_ratio": 0.0,
            "fake_ratio": 0.0,
            "passed": True
        }
        
        # Load metadata if available
        metadata_file = dataset_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            for sample_info in metadata:
                label_stats["total_samples"] += 1
                if sample_info["label"] == 0:
                    label_stats["real_samples"] += 1
                else:
                    label_stats["fake_samples"] += 1
        
        # Calculate ratios
        if label_stats["total_samples"] > 0:
            label_stats["real_ratio"] = label_stats["real_samples"] / label_stats["total_samples"]
            label_stats["fake_ratio"] = label_stats["fake_samples"] / label_stats["total_samples"]
        
        # Check class balance
        min_ratio = self.config["label_validation"]["min_class_ratio"]
        max_ratio = self.config["label_validation"]["max_class_ratio"]
        
        if label_stats["real_ratio"] < min_ratio or label_stats["real_ratio"] > max_ratio:
            label_stats["passed"] = False
            logger.error(f"Real class ratio out of bounds: {label_stats['real_ratio']:.2%}")
        
        if label_stats["fake_ratio"] < min_ratio or label_stats["fake_ratio"] > max_ratio:
            label_stats["passed"] = False
            logger.error(f"Fake class ratio out of bounds: {label_stats['fake_ratio']:.2%}")
        
        return label_stats
    
    def _validate_file_integrity(self, dataset_path: str) -> Dict[str, Any]:
        """Validate file integrity using checksums"""
        dataset_dir = Path(dataset_path)
        integrity_stats = {
            "files_checked": 0,
            "checksum_errors": 0,
            "passed": True
        }
        
        # Check for checksum file
        checksum_file = dataset_dir / "checksums.json"
        if checksum_file.exists():
            with open(checksum_file, 'r') as f:
                expected_checksums = json.load(f)
            
            for file_path, expected_checksum in expected_checksums.items():
                file_path = dataset_dir / file_path
                if file_path.exists():
                    integrity_stats["files_checked"] += 1
                    
                    # Calculate actual checksum
                    actual_checksum = self._calculate_file_checksum(file_path)
                    
                    if actual_checksum != expected_checksum:
                        integrity_stats["checksum_errors"] += 1
                        logger.warning(f"Checksum mismatch: {file_path}")
            
            if integrity_stats["checksum_errors"] > 0:
                integrity_stats["passed"] = False
        
        return integrity_stats
    
    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

class ModelVersionManager:
    """Manages model versioning and artifact storage"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.registry_path = Path(config["model_registry"]["registry_path"])
        self.registry_path.mkdir(exist_ok=True)
    
    def create_version(self, model_name: str, model_path: str, metadata: Dict[str, Any]) -> str:
        """Create a new model version"""
        version_id = self._generate_version_id()
        version_dir = self.registry_path / model_name / version_id
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy model files
        model_source = Path(model_path)
        if model_source.exists():
            if model_source.is_file():
                shutil.copy2(model_source, version_dir / model_source.name)
            else:
                shutil.copytree(model_source, version_dir / model_source.name, dirs_exist_ok=True)
        
        # Save metadata
        metadata["version_id"] = version_id
        metadata["created_at"] = datetime.now().isoformat()
        metadata["model_name"] = model_name
        
        metadata_file = version_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Update registry index
        self._update_registry_index(model_name, version_id, metadata)
        
        logger.info(f"Created model version: {model_name}/{version_id}")
        return version_id
    
    def _generate_version_id(self) -> str:
        """Generate unique version ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_id = str(uuid.uuid4())[:8]
        return f"v{timestamp}_{random_id}"
    
    def _update_registry_index(self, model_name: str, version_id: str, metadata: Dict[str, Any]):
        """Update registry index file"""
        index_file = self.registry_path / "index.json"
        
        if index_file.exists():
            with open(index_file, 'r') as f:
                index = json.load(f)
        else:
            index = {}
        
        if model_name not in index:
            index[model_name] = {}
        
        index[model_name][version_id] = {
            "created_at": metadata["created_at"],
            "performance": metadata.get("performance", {}),
            "status": "active"
        }
        
        with open(index_file, 'w') as f:
            json.dump(index, f, indent=2)
    
    def get_latest_version(self, model_name: str) -> Optional[str]:
        """Get the latest version of a model"""
        index_file = self.registry_path / "index.json"
        
        if not index_file.exists():
            return None
        
        with open(index_file, 'r') as f:
            index = json.load(f)
        
        if model_name not in index:
            return None
        
        versions = list(index[model_name].keys())
        if not versions:
            return None
        
        # Sort by creation time and return latest
        versions.sort(key=lambda v: index[model_name][v]["created_at"], reverse=True)
        return versions[0]
    
    def deploy_model(self, model_name: str, version_id: str, target_path: str) -> bool:
        """Deploy model to target location"""
        try:
            source_dir = self.registry_path / model_name / version_id
            target_dir = Path(target_path)
            
            if not source_dir.exists():
                logger.error(f"Model version not found: {model_name}/{version_id}")
                return False
            
            # Copy model files to target
            if target_dir.exists():
                shutil.rmtree(target_dir)
            
            shutil.copytree(source_dir, target_dir)
            
            logger.info(f"Deployed model {model_name}/{version_id} to {target_path}")
            return True
            
        except Exception as e:
            logger.error(f"Model deployment failed: {e}")
            return False
    
    def cleanup_old_versions(self, retention_days: int = 30):
        """Clean up old model versions"""
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        index_file = self.registry_path / "index.json"
        if not index_file.exists():
            return
        
        with open(index_file, 'r') as f:
            index = json.load(f)
        
        for model_name, versions in index.items():
            for version_id, version_info in list(versions.items()):
                created_at = datetime.fromisoformat(version_info["created_at"])
                if created_at < cutoff_date:
                    # Remove old version
                    version_dir = self.registry_path / model_name / version_id
                    if version_dir.exists():
                        shutil.rmtree(version_dir)
                    
                    del versions[version_id]
                    logger.info(f"Cleaned up old version: {model_name}/{version_id}")
        
        # Update index
        with open(index_file, 'w') as f:
            json.dump(index, f, indent=2)

class NotificationManager:
    """Manages notifications and alerts"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def send_email_alert(self, subject: str, message: str, priority: str = "normal"):
        """Send email alert"""
        if not self.config["email"]["enabled"]:
            return
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config["email"]["from_email"]
            msg['To'] = ", ".join(self.config["email"]["to_emails"])
            msg['Subject'] = f"[{priority.upper()}] {subject}"
            
            msg.attach(MIMEText(message, 'plain'))
            
            # Send email
            server = smtplib.SMTP(self.config["email"]["smtp_server"], self.config["email"]["smtp_port"])
            server.starttls()
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email alert sent: {subject}")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    def send_slack_alert(self, message: str, channel: str = None):
        """Send Slack alert"""
        if not self.config["slack"]["enabled"]:
            return
        
        try:
            webhook_url = self.config["slack"]["webhook_url"]
            target_channel = channel or self.config["slack"]["channel"]
            
            payload = {
                "channel": target_channel,
                "text": message,
                "username": "Training Pipeline Bot"
            }
            
            response = requests.post(webhook_url, json=payload)
            response.raise_for_status()
            
            logger.info("Slack alert sent")
            
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
    
    def send_training_start_notification(self, experiment_id: str, config: Dict[str, Any]):
        """Send training start notification"""
        subject = "Deepfake Detection Training Started"
        message = f"""
Training pipeline started:
- Experiment ID: {experiment_id}
- Start Time: {datetime.now().isoformat()}
- Configuration: {json.dumps(config, indent=2)}
        """
        
        self.send_email_alert(subject, message, "info")
    
    def send_training_completion_notification(self, experiment_id: str, results: Dict[str, Any]):
        """Send training completion notification"""
        subject = "Deepfake Detection Training Completed"
        message = f"""
Training pipeline completed:
- Experiment ID: {experiment_id}
- Completion Time: {datetime.now().isoformat()}
- Results: {json.dumps(results, indent=2)}
        """
        
        self.send_email_alert(subject, message, "info")
    
    def send_training_failure_notification(self, experiment_id: str, error: str):
        """Send training failure notification"""
        subject = "Deepfake Detection Training Failed"
        message = f"""
Training pipeline failed:
- Experiment ID: {experiment_id}
- Failure Time: {datetime.now().isoformat()}
- Error: {error}
        """
        
        self.send_email_alert(subject, message, "high")

class ProductionTrainingPipeline:
    """Main production training pipeline"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = ProductionConfig(config_path)
        self.data_validator = DataValidator(self.config.config["data_validation"])
        self.model_manager = ModelVersionManager(self.config.config["model_management"])
        self.notification_manager = NotificationManager(self.config.config["notifications"])
        
        self.pipeline_id = self._generate_pipeline_id()
        self.start_time = None
        self.end_time = None
        self.status = "idle"
        self.results = {}
        
        # Setup logging
        self._setup_logging()
    
    def _generate_pipeline_id(self) -> str:
        """Generate unique pipeline ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_id = str(uuid.uuid4())[:8]
        return f"pipeline_{timestamp}_{random_id}"
    
    def _setup_logging(self):
        """Setup pipeline-specific logging"""
        if self.config.config["monitoring"]["logging"]["file_logging"]:
            log_dir = Path("logs") / "pipeline"
            log_dir.mkdir(parents=True, exist_ok=True)
            
            log_file = log_dir / f"{self.pipeline_id}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(getattr(logging, self.config.config["monitoring"]["logging"]["level"]))
            
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            
            logger.addHandler(file_handler)
    
    def run_training_pipeline(self, dataset_path: str, training_config_path: str = None,
                            ensemble_config_path: str = None, experiment_config_path: str = None) -> Dict[str, Any]:
        """Run the complete training pipeline"""
        self.start_time = datetime.now()
        self.status = "running"
        
        logger.info(f"Starting production training pipeline: {self.pipeline_id}")
        
        try:
            # Send start notification
            self.notification_manager.send_training_start_notification(
                self.pipeline_id, self.config.config
            )
            
            # Step 1: Data validation
            logger.info("Step 1: Validating training data")
            validation_results = self.data_validator.validate_dataset(dataset_path)
            
            if not validation_results["passed"]:
                raise ValueError(f"Data validation failed: {validation_results['issues']}")
            
            # Step 2: Setup configurations
            logger.info("Step 2: Setting up training configurations")
            training_config = TrainingConfig(training_config_path)
            ensemble_config = EnsembleConfig(ensemble_config_path)
            experiment_config = ExperimentConfig(experiment_config_path)
            
            # Step 3: Initialize components
            logger.info("Step 3: Initializing training components")
            dataset_manager = DatasetManager()
            ensemble_coordinator = EnsembleCoordinator(
                ensemble_config, training_config, experiment_config, 
                f"outputs/{self.pipeline_id}"
            )
            
            # Step 4: Train ensemble
            logger.info("Step 4: Training ensemble models")
            ensemble_coordinator.train_individual_models(dataset_manager, dataset_path)
            
            # Step 5: Cross-validation
            logger.info("Step 5: Performing cross-validation")
            cv_results = ensemble_coordinator.perform_cross_validation(dataset_manager, dataset_path)
            
            # Step 6: Optimize and evaluate
            logger.info("Step 6: Optimizing ensemble and evaluating performance")
            test_loader = dataset_manager.get_data_loader(
                dataset_path, "test", batch_size=training_config.config["training"]["batch_size"], 
                shuffle=False
            )
            
            ensemble_coordinator.optimize_ensemble_weights(test_loader)
            ensemble_coordinator.analyze_model_agreement(test_loader)
            
            val_loader = dataset_manager.get_data_loader(
                dataset_path, "val", batch_size=training_config.config["training"]["batch_size"], 
                shuffle=False
            )
            ensemble_coordinator.calibrate_ensemble(val_loader)
            
            final_metrics = ensemble_coordinator.evaluate_ensemble(test_loader)
            
            # Step 7: Save ensemble
            logger.info("Step 7: Saving ensemble configuration")
            ensemble_save_path = f"outputs/{self.pipeline_id}/ensemble_config.json"
            ensemble_coordinator.save_ensemble(ensemble_save_path)
            
            # Step 8: Version and deploy models
            logger.info("Step 8: Versioning and deploying models")
            for model_name, trainer in ensemble_coordinator.models.items():
                model_metadata = {
                    "pipeline_id": self.pipeline_id,
                    "performance": {
                        "best_accuracy": trainer.best_val_accuracy,
                        "final_accuracy": trainer.val_accuracies[-1] if trainer.val_accuracies else 0
                    },
                    "training_config": training_config.config,
                    "ensemble_weights": ensemble_coordinator.ensemble_weights.tolist() if ensemble_coordinator.ensemble_weights is not None else None
                }
                
                version_id = self.model_manager.create_version(
                    model_name, trainer.output_dir, model_metadata
                )
                
                # Auto-deploy if enabled
                if self.config.config["deployment"]["auto_deploy"]:
                    deploy_path = f"deployed_models/{model_name}"
                    self.model_manager.deploy_model(model_name, version_id, deploy_path)
            
            # Step 9: Cleanup and finalize
            logger.info("Step 9: Finalizing pipeline")
            ensemble_coordinator.close()
            
            # Compile results
            self.results = {
                "pipeline_id": self.pipeline_id,
                "status": "completed",
                "start_time": self.start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "final_metrics": final_metrics,
                "cross_validation_results": cv_results,
                "validation_results": validation_results
            }
            
            self.status = "completed"
            self.end_time = datetime.now()
            
            # Send completion notification
            self.notification_manager.send_training_completion_notification(
                self.pipeline_id, self.results
            )
            
            logger.info(f"Production training pipeline completed successfully: {self.pipeline_id}")
            return self.results
            
        except Exception as e:
            self.status = "failed"
            self.end_time = datetime.now()
            
            error_msg = f"Pipeline failed: {str(e)}"
            logger.error(error_msg)
            
            # Send failure notification
            self.notification_manager.send_training_failure_notification(
                self.pipeline_id, error_msg
            )
            
            raise
    
    def run_scheduled_training(self, schedule_config: Dict[str, Any]):
        """Run scheduled training based on configuration"""
        def training_job():
            try:
                self.run_training_pipeline(
                    schedule_config["dataset_path"],
                    schedule_config.get("training_config_path"),
                    schedule_config.get("ensemble_config_path"),
                    schedule_config.get("experiment_config_path")
                )
            except Exception as e:
                logger.error(f"Scheduled training failed: {e}")
        
        # Setup schedule
        if schedule_config["schedule_type"] == "daily":
            schedule.every().day.at(schedule_config["time"]).do(training_job)
        elif schedule_config["schedule_type"] == "weekly":
            schedule.every().week.at(schedule_config["time"]).do(training_job)
        elif schedule_config["schedule_type"] == "interval":
            schedule.every(schedule_config["interval_hours"]).hours.do(training_job)
        
        logger.info(f"Scheduled training setup: {schedule_config}")
        
        # Run scheduler
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

def main():
    """Main function for production pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Production Training Pipeline")
    parser.add_argument("--config", help="Path to production configuration file")
    parser.add_argument("--dataset", required=True, help="Path to dataset")
    parser.add_argument("--training-config", help="Path to training configuration file")
    parser.add_argument("--ensemble-config", help="Path to ensemble configuration file")
    parser.add_argument("--experiment-config", help="Path to experiment configuration file")
    parser.add_argument("--schedule", action="store_true", help="Run in scheduled mode")
    parser.add_argument("--schedule-config", help="Path to schedule configuration file")
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = ProductionTrainingPipeline(args.config)
    
    if args.schedule:
        # Run scheduled training
        if args.schedule_config:
            with open(args.schedule_config, 'r') as f:
                schedule_config = yaml.safe_load(f)
        else:
            schedule_config = {
                "dataset_path": args.dataset,
                "training_config_path": args.training_config,
                "ensemble_config_path": args.ensemble_config,
                "experiment_config_path": args.experiment_config,
                "schedule_type": "daily",
                "time": "02:00"
            }
        
        pipeline.run_scheduled_training(schedule_config)
    else:
        # Run single training pipeline
        results = pipeline.run_training_pipeline(
            args.dataset,
            args.training_config,
            args.ensemble_config,
            args.experiment_config
        )
        
        logger.info("Training pipeline completed!")
        logger.info(f"Results: {json.dumps(results, indent=2)}")

if __name__ == "__main__":
    main() 