# Advanced Ensemble System for Deepfake Detection

## Overview

This implementation provides a sophisticated ensemble system for deepfake detection that implements state-of-the-art ensemble techniques based on Claude's strategy. The system combines multiple deep learning models using advanced fusion methods, uncertainty quantification, and adaptive weighting to achieve superior detection performance.

## 🚀 Key Features

### 1. **Attention-Based Model Merging**
- Multi-head attention mechanism for intelligent model fusion
- Learnable attention weights that adapt to input characteristics
- Configurable attention dimensions and heads
- Automatic feature extraction and projection

### 2. **Temperature Scaling for Confidence Calibration**
- Automatic temperature parameter optimization
- Calibration using validation data
- Improved confidence reliability
- Support for different calibration methods

### 3. **Monte Carlo Dropout for Uncertainty Quantification**
- Probabilistic uncertainty estimation
- Configurable dropout rates and sample counts
- Uncertainty-aware decision making
- Correlation analysis between uncertainty and prediction errors

### 4. **Adaptive Ensemble Weighting**
- Input-dependent model weighting
- Feature extraction for weight prediction
- Online learning capabilities
- Performance-based weight updates

### 5. **Model Agreement Analysis and Conflict Resolution**
- Comprehensive agreement scoring
- Conflict detection and resolution strategies
- Consensus strength analysis
- Multiple resolution methods (confidence-weighted, majority voting)

### 6. **Cross-Dataset Evaluation Framework**
- Multi-dataset performance assessment
- Consistency analysis across domains
- Comprehensive evaluation metrics
- Automated report generation

## 📁 File Structure

```
app/models/
├── advanced_ensemble.py              # Core ensemble implementation
├── advanced_ensemble_evaluator.py    # Evaluation framework
├── advanced_ensemble_example.py      # Usage examples
├── ensemble_manager.py               # Base ensemble manager
├── base_detector.py                  # Base detector interface
├── xception_detector.py              # Xception model
├── efficientnet_detector.py          # EfficientNet model
└── f3net_detector.py                 # F3Net model
```

## 🛠️ Installation and Setup

### Prerequisites
```bash
pip install torch torchvision
pip install scikit-learn matplotlib seaborn pandas
pip install fastapi uvicorn pillow numpy
```

### Model Setup
1. Download pre-trained model weights:
   ```bash
   # Create models directory
   mkdir -p models/
   
   # Download model weights (example paths)
   # models/xception_weights.pth
   # models/efficientnet_weights.pth
   # models/f3net_weights.pth
   ```

2. Update model paths in detector classes

## 🎯 Usage Examples

### Basic Ensemble Usage

```python
from app.models.advanced_ensemble import (
    AdvancedEnsembleManager, AdvancedEnsembleConfig, AdvancedFusionMethod
)

# Create ensemble configuration
config = AdvancedEnsembleConfig(
    fusion_method=AdvancedFusionMethod.ATTENTION_MERGE,
    attention_dim=128,
    attention_heads=8,
    mc_dropout_samples=30,
    enable_adaptive_weighting=True
)

# Initialize ensemble manager
ensemble = AdvancedEnsembleManager(config)

# Add models
ensemble.add_model("xception", xception_detector, weight=1.0)
ensemble.add_model("efficientnet", efficientnet_detector, weight=1.0)
ensemble.add_model("f3net", f3net_detector, weight=1.0)

# Perform prediction
result = ensemble.predict_advanced(image)
print(f"Prediction: {'FAKE' if result.is_deepfake else 'REAL'}")
print(f"Confidence: {result.confidence:.4f}")
print(f"Uncertainty: {result.uncertainty:.4f}")
```

### Advanced Configuration

```python
# Attention-based ensemble
attention_config = AdvancedEnsembleConfig(
    fusion_method=AdvancedFusionMethod.ATTENTION_MERGE,
    attention_dim=256,
    attention_heads=16,
    learn_attention_weights=True,
    attention_dropout=0.1
)

# Temperature-scaled ensemble
temp_config = AdvancedEnsembleConfig(
    fusion_method=AdvancedFusionMethod.TEMPERATURE_SCALED,
    temperature=1.5,
    calibrate_temperature=True,
    temperature_epochs=100
)

# Monte Carlo dropout ensemble
mc_config = AdvancedEnsembleConfig(
    fusion_method=AdvancedFusionMethod.MONTE_CARLO_DROPOUT,
    mc_dropout_samples=50,
    mc_dropout_rate=0.2,
    uncertainty_threshold=0.15
)
```

### Evaluation and Benchmarking

```python
from app.models.advanced_ensemble_evaluator import AdvancedEnsembleEvaluator

# Create evaluator
evaluator = AdvancedEnsembleEvaluator(output_dir="evaluation_results")

# Evaluate ensemble
result = evaluator.evaluate_ensemble(
    ensemble, test_data, "test_dataset", "my_ensemble"
)

# Compare multiple ensembles
comparison = evaluator.compare_ensembles(
    [result1, result2, result3], "ensemble_comparison"
)

# Generate comprehensive report
report = evaluator.generate_evaluation_report([result1, result2, result3])
```

## 📊 Evaluation Metrics

The system provides comprehensive evaluation metrics:

### Classification Metrics
- **Accuracy**: Overall classification accuracy
- **Precision**: Precision for deepfake detection
- **Recall**: Recall for deepfake detection
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under ROC curve
- **AUC-PR**: Area under precision-recall curve

### Calibration Metrics
- **Brier Score**: Calibration quality measure
- **Calibration Error**: Mean absolute calibration error

### Uncertainty Metrics
- **Mean Uncertainty**: Average uncertainty across predictions
- **Uncertainty Correlation**: Correlation between uncertainty and prediction errors

### Ensemble-Specific Metrics
- **Agreement Score**: Model agreement level
- **Confidence Variance**: Variance in model confidences
- **Ensemble Diversity**: Disagreement among models

### Performance Metrics
- **Inference Time**: Average inference time per sample
- **Throughput**: Samples processed per second

## 🔧 Configuration Options

### AdvancedEnsembleConfig Parameters

```python
@dataclass
class AdvancedEnsembleConfig:
    # Base configuration
    fusion_method: AdvancedFusionMethod = AdvancedFusionMethod.ATTENTION_MERGE
    temperature: float = 1.0
    min_models: int = 2
    max_models: int = 10
    confidence_threshold: float = 0.5
    
    # Attention-based merging
    attention_dim: int = 128
    attention_heads: int = 8
    attention_dropout: float = 0.1
    learn_attention_weights: bool = True
    
    # Temperature scaling
    calibrate_temperature: bool = True
    temperature_validation_split: float = 0.2
    temperature_learning_rate: float = 0.01
    temperature_epochs: int = 100
    
    # Monte Carlo dropout
    mc_dropout_samples: int = 30
    mc_dropout_rate: float = 0.1
    uncertainty_threshold: float = 0.2
    
    # Adaptive weighting
    enable_adaptive_weighting: bool = True
    feature_extraction_dim: int = 256
    weight_update_rate: float = 0.01
    
    # Agreement analysis
    agreement_threshold: float = 0.7
    conflict_resolution_method: str = "confidence_weighted"
    
    # Cross-dataset evaluation
    enable_cross_dataset: bool = False
    dataset_weights: Optional[Dict[str, float]] = None
    cross_validation_folds: int = 5
```

## 🎨 Fusion Methods

### 1. Attention-Based Merging
- Uses multi-head attention to learn optimal model combinations
- Adapts to input characteristics automatically
- Provides interpretable attention weights

### 2. Temperature Scaling
- Calibrates confidence scores for better reliability
- Optimizes temperature parameter on validation data
- Improves probability estimates

### 3. Monte Carlo Dropout
- Estimates prediction uncertainty through multiple forward passes
- Provides uncertainty-aware decisions
- Helps identify low-confidence predictions

### 4. Adaptive Weighting
- Dynamically adjusts model weights based on input features
- Learns optimal weighting strategies
- Improves performance on diverse inputs

### 5. Agreement Resolution
- Analyzes model disagreements
- Resolves conflicts using multiple strategies
- Provides consensus-based decisions

## 📈 Performance Optimization

### Memory Optimization
- Efficient tensor operations
- Gradient checkpointing for large models
- Memory-efficient attention computation

### Speed Optimization
- Batch processing capabilities
- Parallel model inference
- Optimized data loading

### Accuracy Optimization
- Ensemble diversity maximization
- Uncertainty-aware decision making
- Adaptive model selection

## 🔍 Advanced Features

### State Persistence
```python
# Save ensemble state
ensemble.save_ensemble_state("ensemble_state.json")

# Load ensemble state
new_ensemble = AdvancedEnsembleManager()
new_ensemble.load_ensemble_state("ensemble_state.json")
```

### Temperature Calibration
```python
# Calibrate temperature on validation data
success = ensemble.calibrate_temperature(validation_data)
if success:
    print(f"Optimal temperature: {ensemble.temperature_scaler.temperature.item():.4f}")
```

### Cross-Dataset Evaluation
```python
# Evaluate across multiple datasets
datasets = {
    "dataset1": test_data_1,
    "dataset2": test_data_2,
    "dataset3": test_data_3
}

results = ensemble.evaluate_cross_dataset(datasets)
```

## 🧪 Testing and Validation

### Running the Example
```bash
cd app/models
python advanced_ensemble_example.py
```

### Running Benchmarks
```python
from app.models.advanced_ensemble_example import run_benchmark_comparison

# Run comprehensive benchmark
results = run_benchmark_comparison()
```

### Unit Tests
```bash
# Run tests (when implemented)
python -m pytest tests/test_advanced_ensemble.py
```

## 📋 API Integration

The system can be integrated with FastAPI for web services:

```python
from fastapi import FastAPI, UploadFile, File
from app.models.advanced_ensemble import AdvancedEnsembleManager

app = FastAPI()
ensemble = AdvancedEnsembleManager()

@app.post("/predict")
async def predict_deepfake(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    result = ensemble.predict_advanced(image)
    return {
        "is_deepfake": result.is_deepfake,
        "confidence": result.confidence,
        "uncertainty": result.uncertainty
    }
```

## 🎯 Best Practices

### 1. Model Selection
- Choose diverse models with different architectures
- Ensure models have complementary strengths
- Consider computational constraints

### 2. Configuration Tuning
- Start with default configurations
- Tune temperature scaling on validation data
- Adjust attention parameters based on model count

### 3. Evaluation Strategy
- Use multiple datasets for robust evaluation
- Monitor uncertainty metrics
- Track ensemble diversity

### 4. Performance Monitoring
- Monitor inference times
- Track memory usage
- Evaluate accuracy vs. speed trade-offs

## 🔬 Research Applications

This implementation supports various research applications:

### 1. Ensemble Diversity Analysis
- Study the impact of model diversity on performance
- Analyze agreement patterns across different inputs
- Investigate ensemble robustness

### 2. Uncertainty Quantification
- Research uncertainty-aware decision making
- Study calibration methods
- Investigate uncertainty propagation

### 3. Cross-Domain Generalization
- Evaluate performance across different datasets
- Study domain adaptation strategies
- Analyze generalization capabilities

## 🚨 Troubleshooting

### Common Issues

1. **Model Loading Failures**
   - Check model weight file paths
   - Verify model architecture compatibility
   - Ensure sufficient memory

2. **Performance Issues**
   - Reduce batch sizes
   - Use fewer MC dropout samples
   - Optimize attention dimensions

3. **Memory Issues**
   - Reduce model count
   - Use gradient checkpointing
   - Implement memory-efficient attention

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed logging
ensemble = AdvancedEnsembleManager(config)
ensemble.logger.setLevel(logging.DEBUG)
```

## 📚 References

This implementation is based on research from:

1. **Attention-based Ensemble Methods**
   - "Attention-based Ensemble for Deep Face Forgery Detection"
   - Multi-head attention mechanisms for model fusion

2. **Uncertainty Quantification**
   - "Uncertainty Quantification in Deep Fake Detection"
   - Monte Carlo dropout for uncertainty estimation

3. **Confidence Calibration**
   - "On Calibration of Modern Neural Networks"
   - Temperature scaling for confidence calibration

4. **Ensemble Methods**
   - "DeepfakeBench: A Comprehensive Benchmark"
   - State-of-the-art ensemble techniques

## 🤝 Contributing

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests and documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- DeepfakeBench research team
- PyTorch community
- FastAPI developers
- Open source contributors

---

For more information, see the individual module documentation and example scripts. 