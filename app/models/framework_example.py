"""
Example usage of the foundational multi-model deepfake detection framework.
This script demonstrates how to use the core framework components.
"""

import logging
from pathlib import Path
from PIL import Image

from .base_detector import BaseDetector, ModelInfo, ModelStatus
from .model_registry import ModelRegistry, ModelFactory
from .ensemble_manager import EnsembleManager, EnsembleConfig, FusionMethod
from .preprocessing import UnifiedPreprocessor, PreprocessingConfig
from .performance_monitor import PerformanceMonitor, MonitoringConfig


# Example detector implementation
class ExampleDetector(BaseDetector):
    """Example detector that inherits from BaseDetector."""
    
    def __init__(self, model_name: str = "ExampleDetector", device: str = "auto"):
        super().__init__(model_name, device)
        self.model_info = ModelInfo(
            name=model_name,
            version="1.0.0",
            architecture="Example",
            input_size=(224, 224),
            device=self.device,
            status=ModelStatus.UNLOADED
        )
    
    def load_model(self, model_path: str = None) -> bool:
        """Load the model (example implementation)."""
        try:
            # Simulate model loading
            self.logger.info(f"Loading {self.model_name} model...")
            self.is_model_loaded = True
            self.model_info.status = ModelStatus.LOADED
            self.logger.info(f"{self.model_name} model loaded successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load {self.model_name} model: {str(e)}")
            self.model_info.status = ModelStatus.ERROR
            return False
    
    def preprocess(self, image: Image.Image):
        """Preprocess image (example implementation)."""
        # Use the unified preprocessor
        config = PreprocessingConfig(
            input_size=self.model_info.input_size,
            mean=self.model_info.mean,
            std=self.model_info.std
        )
        preprocessor = UnifiedPreprocessor(config)
        return preprocessor.preprocess(image)
    
    def predict(self, image: Image.Image):
        """Perform prediction (example implementation)."""
        from .base_detector import DetectionResult
        
        # Simulate prediction
        import random
        confidence = random.uniform(0.1, 0.9)
        is_deepfake = confidence > 0.5
        
        return DetectionResult(
            is_deepfake=is_deepfake,
            confidence=confidence,
            model_name=self.model_name,
            inference_time=0.1  # Simulated
        )
    
    def get_model_info(self) -> ModelInfo:
        """Get model information."""
        return self.model_info


def main():
    """Main example function demonstrating framework usage."""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting multi-model framework example...")
    
    # 1. Initialize performance monitor
    monitor_config = MonitoringConfig(
        save_performance_data=True,
        performance_data_path="example_performance_data"
    )
    performance_monitor = PerformanceMonitor(monitor_config)
    
    # 2. Initialize model registry
    registry = ModelRegistry(models_dir="example_models")
    
    # 3. Register example models
    detector1 = ExampleDetector("ExampleDetector1")
    detector2 = ExampleDetector("ExampleDetector2")
    
    registry.register_model("detector1", ExampleDetector, {"model_name": "ExampleDetector1"})
    registry.register_model("detector2", ExampleDetector, {"model_name": "ExampleDetector2"})
    
    # 4. Load models
    logger.info("Loading models...")
    load_results = registry.load_all_models()
    logger.info(f"Model loading results: {load_results}")
    
    # 5. Initialize ensemble manager
    ensemble_config = EnsembleConfig(
        fusion_method=FusionMethod.WEIGHTED_AVERAGE,
        default_weights={"detector1": 1.0, "detector2": 1.0}
    )
    ensemble = EnsembleManager(ensemble_config)
    
    # 6. Add models to ensemble
    detector1_instance = registry.get_model("detector1")
    detector2_instance = registry.get_model("detector2")
    
    if detector1_instance and detector2_instance:
        ensemble.add_model("detector1", detector1_instance, weight=1.0)
        ensemble.add_model("detector2", detector2_instance, weight=1.0)
    
    # 7. Create a test image
    test_image = Image.new('RGB', (224, 224), color='red')
    
    # 8. Perform individual predictions with monitoring
    logger.info("Performing individual predictions...")
    
    for model_name in ["detector1", "detector2"]:
        model = registry.get_model(model_name)
        if model:
            # Start timing
            performance_monitor.start_timer(model_name)
            
            try:
                # Perform prediction
                result = model.predict(test_image)
                
                # End timing and record success
                inference_time = performance_monitor.end_timer(model_name)
                performance_monitor.record_success(model_name, True)
                
                logger.info(f"{model_name}: {result.is_deepfake} (confidence: {result.confidence:.3f})")
                
            except Exception as e:
                performance_monitor.end_timer(model_name)
                performance_monitor.record_success(model_name, False)
                logger.error(f"Prediction failed for {model_name}: {str(e)}")
    
    # 9. Perform ensemble prediction
    logger.info("Performing ensemble prediction...")
    try:
        ensemble_result = ensemble.predict_ensemble(test_image)
        logger.info(f"Ensemble result: {ensemble_result.is_deepfake} (confidence: {ensemble_result.ensemble_confidence:.3f})")
        logger.info(f"Fusion method: {ensemble_result.fusion_method}")
        logger.info(f"Uncertainty: {ensemble_result.uncertainty:.3f}")
        
        if ensemble_result.attention_weights:
            logger.info(f"Attention weights: {ensemble_result.attention_weights}")
            
    except Exception as e:
        logger.error(f"Ensemble prediction failed: {str(e)}")
    
    # 10. Get performance report
    logger.info("Generating performance report...")
    performance_report = performance_monitor.get_performance_report()
    
    logger.info("Performance Summary:")
    logger.info(f"Total models: {performance_report['summary'].get('total_models', 0)}")
    logger.info(f"Average inference time: {performance_report['summary'].get('avg_inference_time', 0):.3f}s")
    logger.info(f"Average throughput: {performance_report['summary'].get('avg_throughput', 0):.1f} FPS")
    
    # 11. Check for alerts
    alerts = performance_monitor.check_alerts()
    if alerts:
        logger.warning(f"Found {len(alerts)} performance alerts:")
        for alert in alerts:
            logger.warning(f"  - {alert['type']}: {alert.get('model', 'system')} - {alert['value']:.2f}")
    else:
        logger.info("No performance alerts detected")
    
    # 12. Save performance data
    performance_monitor.save_performance_data("example_performance_report.json")
    
    # 13. Get ensemble information
    ensemble_info = ensemble.get_ensemble_info()
    logger.info(f"Ensemble info: {ensemble_info}")
    
    logger.info("Framework example completed successfully!")


if __name__ == "__main__":
    main() 