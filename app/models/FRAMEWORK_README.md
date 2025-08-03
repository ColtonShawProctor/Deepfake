# Multi-Model Deepfake Detection Framework

This directory contains the foundational framework for multi-model deepfake detection. The framework provides a modular, extensible architecture that allows easy integration of different deepfake detection models with unified preprocessing, ensemble prediction, and performance monitoring.

## 🏗️ Framework Architecture

### Core Components

1. **Base Detector Interface** (`base_detector.py`)
   - Abstract base class that all detectors must inherit from
   - Standardized interface for model loading, preprocessing, and prediction
   - Built-in performance tracking and validation

2. **Model Registry & Factory** (`model_registry.py`)
   - Centralized model management and registration
   - Factory pattern for creating detector instances
   - Automatic model loading and status tracking

3. **Ensemble Manager** (`ensemble_manager.py`)
   - Multi-model prediction fusion using various strategies
   - Support for weighted averaging, voting, and attention-based fusion
   - Uncertainty quantification and confidence calibration

4. **Unified Preprocessing** (`preprocessing.py`)
   - Standardized image preprocessing pipeline
   - Support for face detection, noise reduction, and augmentation
   - Configurable preprocessing for different model requirements

5. **Performance Monitor** (`performance_monitor.py`)
   - Comprehensive performance tracking and metrics
   - System resource monitoring (CPU, GPU, memory)
   - Alerting and reporting capabilities

## 📁 File Structure

```
app/models/
├── __init__.py                 # Framework exports
├── base_detector.py           # Core abstract classes and interfaces
├── model_registry.py          # Model management and factory
├── ensemble_manager.py        # Ensemble prediction infrastructure
├── preprocessing.py           # Unified preprocessing pipeline
├── performance_monitor.py     # Performance monitoring and logging
├── framework_example.py       # Usage examples
├── FRAMEWORK_README.md        # This documentation
├── user.py                    # Database models
├── media_file.py              # Database models
└── detection_result.py        # Database models
```

## 🚀 Quick Start

### 1. Basic Usage

```python
from app.models import (
    BaseDetector, ModelRegistry, EnsembleManager, 
    PerformanceMonitor, UnifiedPreprocessor
)

# Initialize components
registry = ModelRegistry()
ensemble = EnsembleManager()
monitor = PerformanceMonitor()

# Register and load models
registry.register_model("my_model", MyDetector)
registry.load_model("my_model")

# Add to ensemble
model = registry.get_model("my_model")
ensemble.add_model("my_model", model)

# Perform prediction
result = ensemble.predict_ensemble(image)
```

### 2. Creating a Custom Detector

```python
from app.models import BaseDetector, ModelInfo, ModelStatus
from PIL import Image

class MyCustomDetector(BaseDetector):
    def __init__(self, model_name: str = "MyDetector", device: str = "auto"):
        super().__init__(model_name, device)
        self.model_info = ModelInfo(
            name=model_name,
            version="1.0.0",
            architecture="Custom",
            input_size=(224, 224),
            device=self.device
        )
    
    def load_model(self, model_path: str = None) -> bool:
        # Implement model loading logic
        self.is_model_loaded = True
        return True
    
    def preprocess(self, image: Image.Image):
        # Use unified preprocessor
        config = PreprocessingConfig(input_size=(224, 224))
        preprocessor = UnifiedPreprocessor(config)
        return preprocessor.preprocess(image)
    
    def predict(self, image: Image.Image):
        # Implement prediction logic
        return DetectionResult(
            is_deepfake=False,
            confidence=0.5,
            model_name=self.model_name,
            inference_time=0.1
        )
    
    def get_model_info(self) -> ModelInfo:
        return self.model_info
```

### 3. Ensemble Configuration

```python
from app.models import EnsembleConfig, FusionMethod

# Configure ensemble
config = EnsembleConfig(
    fusion_method=FusionMethod.WEIGHTED_AVERAGE,
    default_weights={"model1": 1.0, "model2": 1.5},
    temperature=1.0,
    enable_uncertainty=True
)

ensemble = EnsembleManager(config)
```

### 4. Performance Monitoring

```python
from app.models import MonitoringConfig, PerformanceMonitor

# Configure monitoring
config = MonitoringConfig(
    save_performance_data=True,
    performance_data_path="performance_data",
    alert_thresholds={
        "inference_time_ms": 1000.0,
        "error_rate": 0.1
    }
)

monitor = PerformanceMonitor(config)

# Track performance
monitor.start_timer("model_name")
result = model.predict(image)
inference_time = monitor.end_timer("model_name")
monitor.record_success("model_name", True)

# Get reports
report = monitor.get_performance_report()
alerts = monitor.check_alerts()
```

## 🔧 Configuration Options

### Preprocessing Configuration

```python
from app.models import PreprocessingConfig, AugmentationType

config = PreprocessingConfig(
    input_size=(224, 224),
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
    normalize=True,
    augment=True,
    augmentation_type=AugmentationType.BASIC,
    enable_face_detection=True,
    enable_noise_reduction=True
)
```

### Ensemble Configuration

```python
from app.models import EnsembleConfig, FusionMethod

config = EnsembleConfig(
    fusion_method=FusionMethod.ATTENTION_FUSION,
    temperature=1.0,
    min_models=2,
    max_models=10,
    confidence_threshold=0.5,
    enable_uncertainty=True,
    enable_attention=True
)
```

### Monitoring Configuration

```python
from app.models import MonitoringConfig

config = MonitoringConfig(
    save_performance_data=True,
    performance_data_path="data/performance",
    log_interval=60.0,
    metrics_history_size=1000,
    enable_system_monitoring=True,
    enable_gpu_monitoring=True,
    alert_thresholds={
        "inference_time_ms": 1000.0,
        "error_rate": 0.1,
        "memory_usage_percent": 90.0,
        "cpu_usage_percent": 90.0
    }
)
```

## 📊 Available Fusion Methods

1. **Weighted Average** (`FusionMethod.WEIGHTED_AVERAGE`)
   - Combines predictions using configurable weights
   - Most commonly used method

2. **Majority Voting** (`FusionMethod.MAJORITY_VOTING`)
   - Uses majority vote of binary predictions
   - Good for high-confidence decisions

3. **Soft Voting** (`FusionMethod.SOFT_VOTING`)
   - Averages confidence scores
   - Preserves uncertainty information

4. **Attention Fusion** (`FusionMethod.ATTENTION_FUSION`)
   - Uses attention mechanism to weight predictions
   - Adaptive weighting based on confidence

5. **Max/Min Confidence** (`FusionMethod.MAX_CONFIDENCE`, `FusionMethod.MIN_CONFIDENCE`)
   - Uses highest/lowest confidence prediction
   - Conservative/aggressive strategies

## 🔍 Performance Monitoring Features

### Metrics Tracked

- **Inference Performance**: Time, throughput, success rate
- **Accuracy Metrics**: Precision, recall, F1-score
- **System Resources**: CPU, memory, GPU utilization
- **Error Tracking**: Error rates, failure analysis

### Reporting

- **Real-time Monitoring**: Live performance tracking
- **Historical Analysis**: Performance trends over time
- **Alert System**: Automatic alerts for performance issues
- **Export Capabilities**: JSON and CSV export formats

### Alert Types

- High inference time
- High error rate
- High memory usage
- High CPU usage

## 🛠️ Advanced Features

### Face Detection

```python
config = PreprocessingConfig(
    enable_face_detection=True,
    face_crop_margin=0.1
)
```

### Image Enhancement

```python
config = PreprocessingConfig(
    enable_noise_reduction=True,
    noise_reduction_strength=0.1,
    enable_histogram_equalization=True,
    enable_sharpening=True,
    sharpening_strength=1.5
)
```

### Augmentation

```python
config = PreprocessingConfig(
    augment=True,
    augmentation_type=AugmentationType.ADVANCED
)
```

## 📈 Usage Examples

### Complete Pipeline Example

```python
import logging
from app.models import *

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize framework
registry = ModelRegistry()
ensemble = EnsembleManager()
monitor = PerformanceMonitor()

# Register models
registry.register_model("model1", MyDetector1)
registry.register_model("model2", MyDetector2)

# Load models
registry.load_all_models()

# Add to ensemble
for name in ["model1", "model2"]:
    model = registry.get_model(name)
    if model:
        ensemble.add_model(name, model)

# Process image
image = Image.open("test_image.jpg")

# Monitor performance
monitor.start_timer("ensemble")
result = ensemble.predict_ensemble(image)
inference_time = monitor.end_timer("ensemble")

print(f"Result: {result.is_deepfake}")
print(f"Confidence: {result.ensemble_confidence:.3f}")
print(f"Uncertainty: {result.uncertainty:.3f}")
```

### Performance Analysis

```python
# Get performance report
report = monitor.get_performance_report()

# Check for alerts
alerts = monitor.check_alerts()
for alert in alerts:
    print(f"Alert: {alert['type']} - {alert['value']}")

# Export data
monitor.save_performance_data("report.json")
monitor.export_metrics_csv("metrics.csv")
```

## 🔧 Dependencies

The framework requires the following Python packages:

```
torch>=1.9.0
torchvision>=0.10.0
Pillow>=8.0.0
numpy>=1.21.0
opencv-python>=4.5.0
psutil>=5.8.0
pydantic>=1.8.0
```

Optional dependencies for advanced features:

```
pynvml>=11.0.0  # GPU monitoring
albumentations>=1.3.0  # Advanced augmentation
```

## 🚀 Next Steps

1. **Implement Specific Models**: Create concrete detector implementations
2. **Add Model Weights**: Download or train model weights
3. **Configure Ensemble**: Set up optimal fusion strategies
4. **Deploy to Production**: Integrate with FastAPI backend
5. **Monitor Performance**: Set up continuous monitoring

## 📚 API Reference

For detailed API documentation, see the individual module files:

- `base_detector.py` - Core interfaces and base classes
- `model_registry.py` - Model management and factory
- `ensemble_manager.py` - Ensemble prediction
- `preprocessing.py` - Image preprocessing
- `performance_monitor.py` - Performance tracking

## 🤝 Contributing

To extend the framework:

1. Inherit from `BaseDetector` for new models
2. Implement required abstract methods
3. Register models with `ModelRegistry`
4. Test with ensemble prediction
5. Monitor performance metrics

The framework is designed to be modular and extensible, making it easy to add new models and features while maintaining consistency across the system. 