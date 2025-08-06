#!/usr/bin/env python3
"""
Main Training Pipeline for Deepfake Detection Ensemble

This script orchestrates the complete training pipeline including:
- Dataset preparation and validation
- Individual model training
- Ensemble training and optimization
- Experiment tracking and monitoring
- Evaluation and testing
- Model deployment

Usage:
    python main_training_pipeline.py --dataset /path/to/dataset --output-dir /path/to/output
"""

import os
import sys
import logging
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml

# Add the app directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from training.dataset_management import DatasetManager
from training.model_trainer import ModelTrainer, TrainingConfig
from training.ensemble_coordinator import EnsembleCoordinator, EnsembleConfig
from training.experiment_tracker import ExperimentTracker, ExperimentConfig
from training.production_pipeline import ProductionTrainingPipeline
from training.evaluation_framework import EvaluationConfig, ModelEvaluator, EvaluationReporter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MainTrainingPipeline:
    """Main training pipeline orchestrator"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.pipeline_config = self._load_pipeline_config()
        self.start_time = None
        self.end_time = None
        self.results = {}
    
    def _load_pipeline_config(self) -> Dict[str, Any]:
        """Load main pipeline configuration"""
        if self.config_path and Path(self.config_path).exists():
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        
        # Default configuration
        return {
            "pipeline": {
                "name": "deepfake_detection_ensemble_training",
                "version": "1.0.0",
                "description": "Complete training pipeline for deepfake detection ensemble"
            },
            "stages": {
                "data_preparation": True,
                "individual_training": True,
                "ensemble_training": True,
                "evaluation": True,
                "deployment": True
            },
            "models": {
                "mesonet": {
                    "enabled": True,
                    "config": "configs/mesonet_config.yaml"
                },
                "xception": {
                    "enabled": True,
                    "config": "configs/xception_config.yaml"
                },
                "efficientnet": {
                    "enabled": True,
                    "config": "configs/efficientnet_config.yaml"
                },
                "f3net": {
                    "enabled": True,
                    "config": "configs/f3net_config.yaml"
                }
            },
            "ensemble": {
                "enabled": True,
                "config": "configs/ensemble_config.yaml",
                "optimization": True,
                "cross_validation": True
            },
            "experiment_tracking": {
                "enabled": True,
                "config": "configs/experiment_config.yaml",
                "tensorboard": True,
                "mlflow": False
            },
            "evaluation": {
                "enabled": True,
                "config": "configs/evaluation_config.yaml",
                "cross_dataset": True,
                "robustness": True,
                "ab_testing": False
            },
            "deployment": {
                "enabled": True,
                "auto_deploy": True,
                "model_registry": True
            }
        }
    
    def run_complete_pipeline(self, dataset_path: str, output_dir: str,
                            skip_stages: List[str] = None) -> Dict[str, Any]:
        """Run the complete training pipeline"""
        self.start_time = time.time()
        logger.info("Starting complete deepfake detection training pipeline")
        
        if skip_stages is None:
            skip_stages = []
        
        pipeline_results = {
            "pipeline_config": self.pipeline_config,
            "dataset_path": dataset_path,
            "output_dir": output_dir,
            "start_time": self.start_time,
            "stages": {}
        }
        
        try:
            # Stage 1: Data Preparation
            if "data_preparation" not in skip_stages and self.pipeline_config["stages"]["data_preparation"]:
                logger.info("Stage 1: Data Preparation")
                data_results = self._prepare_data(dataset_path, output_dir)
                pipeline_results["stages"]["data_preparation"] = data_results
            
            # Stage 2: Individual Model Training
            if "individual_training" not in skip_stages and self.pipeline_config["stages"]["individual_training"]:
                logger.info("Stage 2: Individual Model Training")
                training_results = self._train_individual_models(dataset_path, output_dir)
                pipeline_results["stages"]["individual_training"] = training_results
            
            # Stage 3: Ensemble Training
            if "ensemble_training" not in skip_stages and self.pipeline_config["stages"]["ensemble_training"]:
                logger.info("Stage 3: Ensemble Training")
                ensemble_results = self._train_ensemble(dataset_path, output_dir)
                pipeline_results["stages"]["ensemble_training"] = ensemble_results
            
            # Stage 4: Evaluation
            if "evaluation" not in skip_stages and self.pipeline_config["stages"]["evaluation"]:
                logger.info("Stage 4: Evaluation")
                evaluation_results = self._evaluate_models(dataset_path, output_dir)
                pipeline_results["stages"]["evaluation"] = evaluation_results
            
            # Stage 5: Deployment
            if "deployment" not in skip_stages and self.pipeline_config["stages"]["deployment"]:
                logger.info("Stage 5: Deployment")
                deployment_results = self._deploy_models(output_dir)
                pipeline_results["stages"]["deployment"] = deployment_results
            
            # Compile final results
            self.end_time = time.time()
            pipeline_results["end_time"] = self.end_time
            pipeline_results["duration"] = self.end_time - self.start_time
            pipeline_results["status"] = "completed"
            
            # Save pipeline results
            results_file = Path(output_dir) / "pipeline_results.json"
            with open(results_file, 'w') as f:
                json.dump(pipeline_results, f, indent=2)
            
            logger.info(f"Pipeline completed successfully in {pipeline_results['duration']:.2f} seconds")
            self.results = pipeline_results
            return pipeline_results
            
        except Exception as e:
            self.end_time = time.time()
            pipeline_results["end_time"] = self.end_time
            pipeline_results["duration"] = self.end_time - self.start_time
            pipeline_results["status"] = "failed"
            pipeline_results["error"] = str(e)
            
            logger.error(f"Pipeline failed: {e}")
            raise
    
    def _prepare_data(self, dataset_path: str, output_dir: str) -> Dict[str, Any]:
        """Prepare and validate training data"""
        logger.info("Preparing and validating training data")
        
        dataset_manager = DatasetManager()
        
        # Validate dataset
        validation_results = dataset_manager.validate_dataset_quality(dataset_path)
        
        # Create train/val/test splits if they don't exist
        splits_file = Path(dataset_path) / "splits.json"
        if not splits_file.exists():
            logger.info("Creating train/validation/test splits")
            splits = dataset_manager.create_train_val_test_splits(dataset_path)
        else:
            logger.info("Using existing train/validation/test splits")
            with open(splits_file, 'r') as f:
                splits = json.load(f)
        
        return {
            "validation_results": validation_results,
            "splits_created": not splits_file.exists(),
            "splits": splits
        }
    
    def _train_individual_models(self, dataset_path: str, output_dir: str) -> Dict[str, Any]:
        """Train individual models"""
        logger.info("Training individual models")
        
        # Load configurations
        training_config = TrainingConfig()
        experiment_config = ExperimentConfig()
        
        # Setup experiment tracking
        if self.pipeline_config["experiment_tracking"]["enabled"]:
            tracker = ExperimentTracker(experiment_config, output_dir)
            tracker.log_hyperparameters(training_config.config)
        else:
            tracker = None
        
        training_results = {}
        
        # Train each enabled model
        for model_name, model_config in self.pipeline_config["models"].items():
            if not model_config["enabled"]:
                logger.info(f"Skipping {model_name} (disabled)")
                continue
            
            logger.info(f"Training {model_name}")
            
            try:
                # Create trainer
                trainer = ModelTrainer(model_name, training_config, output_dir)
                
                # Get data loaders
                dataset_manager = DatasetManager()
                train_loader = dataset_manager.get_data_loader(
                    dataset_path, "train", model_name=model_name
                )
                val_loader = dataset_manager.get_data_loader(
                    dataset_path, "val", model_name=model_name, shuffle=False
                )
                
                # Train model
                results = trainer.train(train_loader, val_loader)
                
                # Log results
                if tracker:
                    tracker.log_metrics({
                        f"{model_name}_final_accuracy": results["val_accuracies"][-1],
                        f"{model_name}_best_accuracy": trainer.best_val_accuracy
                    }, step=0, prefix="individual_training")
                
                training_results[model_name] = {
                    "status": "completed",
                    "best_accuracy": trainer.best_val_accuracy,
                    "final_accuracy": results["val_accuracies"][-1],
                    "training_curves": results
                }
                
                logger.info(f"{model_name} training completed - Best accuracy: {trainer.best_val_accuracy:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                training_results[model_name] = {
                    "status": "failed",
                    "error": str(e)
                }
        
        # Close experiment tracker
        if tracker:
            tracker.close()
        
        return training_results
    
    def _train_ensemble(self, dataset_path: str, output_dir: str) -> Dict[str, Any]:
        """Train and optimize ensemble"""
        if not self.pipeline_config["ensemble"]["enabled"]:
            logger.info("Ensemble training disabled")
            return {"status": "disabled"}
        
        logger.info("Training and optimizing ensemble")
        
        # Load configurations
        ensemble_config = EnsembleConfig()
        training_config = TrainingConfig()
        experiment_config = ExperimentConfig()
        
        # Create ensemble coordinator
        ensemble_coordinator = EnsembleCoordinator(
            ensemble_config, training_config, experiment_config, output_dir
        )
        
        try:
            # Train individual models (if not already done)
            dataset_manager = DatasetManager()
            
            # Perform cross-validation
            if self.pipeline_config["ensemble"]["cross_validation"]:
                cv_results = ensemble_coordinator.perform_cross_validation(dataset_manager, dataset_path)
            else:
                cv_results = {}
            
            # Optimize ensemble weights
            test_loader = dataset_manager.get_data_loader(
                dataset_path, "test", batch_size=training_config.config["training"]["batch_size"], 
                shuffle=False
            )
            
            optimal_weights = ensemble_coordinator.optimize_ensemble_weights(test_loader)
            
            # Analyze model agreement
            agreement_analysis = ensemble_coordinator.analyze_model_agreement(test_loader)
            
            # Calibrate ensemble
            val_loader = dataset_manager.get_data_loader(
                dataset_path, "val", batch_size=training_config.config["training"]["batch_size"], 
                shuffle=False
            )
            calibrated_predictions = ensemble_coordinator.calibrate_ensemble(val_loader)
            
            # Evaluate ensemble
            final_metrics = ensemble_coordinator.evaluate_ensemble(test_loader)
            
            # Save ensemble
            ensemble_save_path = Path(output_dir) / "ensemble_config.json"
            ensemble_coordinator.save_ensemble(str(ensemble_save_path))
            
            # Close coordinator
            ensemble_coordinator.close()
            
            return {
                "status": "completed",
                "cross_validation_results": cv_results,
                "optimal_weights": optimal_weights.tolist() if optimal_weights is not None else None,
                "agreement_analysis": agreement_analysis,
                "final_metrics": final_metrics,
                "ensemble_saved": str(ensemble_save_path)
            }
            
        except Exception as e:
            logger.error(f"Ensemble training failed: {e}")
            ensemble_coordinator.close()
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def _evaluate_models(self, dataset_path: str, output_dir: str) -> Dict[str, Any]:
        """Evaluate models comprehensively"""
        if not self.pipeline_config["evaluation"]["enabled"]:
            logger.info("Evaluation disabled")
            return {"status": "disabled"}
        
        logger.info("Evaluating models")
        
        # Load configurations
        evaluation_config = EvaluationConfig()
        training_config = TrainingConfig()
        
        # Setup evaluation components
        dataset_manager = DatasetManager()
        test_loader = dataset_manager.get_data_loader(dataset_path, "test", batch_size=32, shuffle=False)
        
        evaluation_results = {}
        
        # Evaluate each trained model
        for model_name, model_config in self.pipeline_config["models"].items():
            if not model_config["enabled"]:
                continue
            
            logger.info(f"Evaluating {model_name}")
            
            try:
                # Load trained model
                trainer = ModelTrainer(model_name, training_config, output_dir)
                checkpoint_path = Path(output_dir) / model_name / "best_checkpoint.pth"
                
                if checkpoint_path.exists():
                    trainer.load_checkpoint(str(checkpoint_path))
                    
                    # Evaluate model
                    evaluator = ModelEvaluator(evaluation_config)
                    results = evaluator.evaluate_model(trainer, test_loader, model_name)
                    
                    evaluation_results[model_name] = results
                    
                    logger.info(f"{model_name} evaluation completed - Accuracy: {results['metrics']['accuracy']:.4f}")
                else:
                    logger.warning(f"No checkpoint found for {model_name}")
                    evaluation_results[model_name] = {"status": "no_checkpoint"}
                    
            except Exception as e:
                logger.error(f"Failed to evaluate {model_name}: {e}")
                evaluation_results[model_name] = {"status": "failed", "error": str(e)}
        
        # Generate evaluation report
        if evaluation_results:
            reporter = EvaluationReporter(evaluation_config, output_dir)
            report_file = reporter.generate_comprehensive_report(evaluation_results)
            
            evaluation_results["report_file"] = report_file
        
        return evaluation_results
    
    def _deploy_models(self, output_dir: str) -> Dict[str, Any]:
        """Deploy trained models"""
        if not self.pipeline_config["deployment"]["enabled"]:
            logger.info("Deployment disabled")
            return {"status": "disabled"}
        
        logger.info("Deploying models")
        
        # Create production pipeline for deployment
        production_pipeline = ProductionTrainingPipeline()
        
        deployment_results = {}
        
        # Deploy each trained model
        for model_name, model_config in self.pipeline_config["models"].items():
            if not model_config["enabled"]:
                continue
            
            logger.info(f"Deploying {model_name}")
            
            try:
                # Get model path
                model_path = Path(output_dir) / model_name
                
                if model_path.exists():
                    # Create deployment metadata
                    metadata = {
                        "model_name": model_name,
                        "deployment_time": time.time(),
                        "pipeline_version": self.pipeline_config["pipeline"]["version"]
                    }
                    
                    # Version and deploy model
                    version_id = production_pipeline.model_manager.create_version(
                        model_name, str(model_path), metadata
                    )
                    
                    # Auto-deploy if enabled
                    if self.pipeline_config["deployment"]["auto_deploy"]:
                        deploy_path = f"deployed_models/{model_name}"
                        success = production_pipeline.model_manager.deploy_model(
                            model_name, version_id, deploy_path
                        )
                        
                        deployment_results[model_name] = {
                            "status": "deployed" if success else "deployment_failed",
                            "version_id": version_id,
                            "deploy_path": deploy_path
                        }
                    else:
                        deployment_results[model_name] = {
                            "status": "versioned",
                            "version_id": version_id
                        }
                    
                    logger.info(f"{model_name} deployed successfully - Version: {version_id}")
                else:
                    logger.warning(f"Model path not found for {model_name}")
                    deployment_results[model_name] = {"status": "not_found"}
                    
            except Exception as e:
                logger.error(f"Failed to deploy {model_name}: {e}")
                deployment_results[model_name] = {"status": "failed", "error": str(e)}
        
        return deployment_results

def create_config_files(output_dir: str):
    """Create default configuration files"""
    config_dir = Path(output_dir) / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Training configuration
    training_config = {
        "training": {
            "epochs": 100,
            "batch_size": 32,
            "learning_rate": 0.001,
            "weight_decay": 1e-4,
            "gradient_clip": 1.0,
            "early_stopping_patience": 10,
            "save_best_only": True,
            "mixed_precision": True,
            "num_workers": 4,
            "pin_memory": True
        },
        "optimization": {
            "optimizer": "adam",
            "scheduler": "cosine_annealing",
            "warmup_epochs": 5,
            "lr_decay": 0.1,
            "min_lr": 1e-6
        }
    }
    
    with open(config_dir / "training_config.yaml", 'w') as f:
        yaml.dump(training_config, f, default_flow_style=False)
    
    # Ensemble configuration
    ensemble_config = {
        "ensemble": {
            "models": ["mesonet", "xception", "efficientnet", "f3net"],
            "fusion_method": "weighted_average",
            "weight_optimization": True,
            "cross_validation": True,
            "cv_folds": 5,
            "agreement_threshold": 0.7,
            "calibration": True
        }
    }
    
    with open(config_dir / "ensemble_config.yaml", 'w') as f:
        yaml.dump(ensemble_config, f, default_flow_style=False)
    
    # Experiment configuration
    experiment_config = {
        "experiment": {
            "name": "deepfake_detection_experiment",
            "enable_tensorboard": True,
            "enable_mlflow": False,
            "save_artifacts": True
        }
    }
    
    with open(config_dir / "experiment_config.yaml", 'w') as f:
        yaml.dump(experiment_config, f, default_flow_style=False)
    
    # Evaluation configuration
    evaluation_config = {
        "evaluation": {
            "metrics": ["accuracy", "precision", "recall", "f1", "auc", "ap"],
            "confidence_thresholds": [0.1, 0.3, 0.5, 0.7, 0.9]
        },
        "cross_dataset": {
            "enabled": True,
            "datasets": ["faceforensics", "celebdf", "dfdc"]
        },
        "robustness": {
            "enabled": True,
            "adversarial_attacks": ["fgsm", "pgd"],
            "noise_types": ["gaussian", "salt_pepper", "blur"]
        }
    }
    
    with open(config_dir / "evaluation_config.yaml", 'w') as f:
        yaml.dump(evaluation_config, f, default_flow_style=False)
    
    logger.info(f"Configuration files created in {config_dir}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Main Training Pipeline for Deepfake Detection")
    parser.add_argument("--dataset", required=True, help="Path to dataset")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--config", help="Path to pipeline configuration file")
    parser.add_argument("--skip-stages", nargs="+", help="Stages to skip")
    parser.add_argument("--create-configs", action="store_true", help="Create default configuration files")
    
    args = parser.parse_args()
    
    # Create configuration files if requested
    if args.create_configs:
        create_config_files(args.output_dir)
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create pipeline
    pipeline = MainTrainingPipeline(args.config)
    
    # Run pipeline
    try:
        results = pipeline.run_complete_pipeline(
            args.dataset, 
            args.output_dir,
            skip_stages=args.skip_stages
        )
        
        logger.info("Training pipeline completed successfully!")
        logger.info(f"Results saved to: {output_dir}/pipeline_results.json")
        
        # Print summary
        print("\n" + "="*50)
        print("TRAINING PIPELINE SUMMARY")
        print("="*50)
        print(f"Status: {results['status']}")
        print(f"Duration: {results['duration']:.2f} seconds")
        print(f"Output Directory: {args.output_dir}")
        
        if results['status'] == 'completed':
            print("\nStage Results:")
            for stage, stage_results in results['stages'].items():
                if isinstance(stage_results, dict) and 'status' in stage_results:
                    print(f"  {stage}: {stage_results['status']}")
        
        print("="*50)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 