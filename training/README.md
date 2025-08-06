# Deepfake Detection Training Pipeline

A comprehensive, production-ready training pipeline for multi-model deepfake detection systems with advanced ensemble learning, experiment tracking, and automated deployment capabilities.

## 🎯 Overview

This training pipeline implements a complete deepfake detection system based on Claude's training strategy recommendations, featuring:

- **Multi-model ensemble**: MesoNet, Xception, EfficientNet-B4, F3Net
- **Advanced data preprocessing**: Model-specific pipelines with frequency-domain processing
- **Ensemble optimization**: Bayesian weight optimization, cross-validation, calibration
- **Production features**: Automated training, model versioning, deployment pipeline
- **Comprehensive evaluation**: Cross-dataset testing, robustness analysis, A/B testing

## 📁 Project Structure

```
training/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── main_training_pipeline.py          # Main orchestration script
├── dataset_management.py              # Dataset handling and preprocessing
├── model_trainer.py                   # Individual model training
├── ensemble_coordinator.py            # Ensemble training and optimization
├── experiment_tracker.py              # Experiment tracking and monitoring
├── production_pipeline.py             # Production automation
├── evaluation_framework.py            # Model evaluation and testing
└── configs/                           # Configuration files (auto-generated)
    ├── training_config.yaml
    ├── ensemble_config.yaml
    ├── experiment_config.yaml
    └── evaluation_config.yaml
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd deepfake

# Install dependencies
pip install -r training/requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import tensorboard; print('TensorBoard: OK')"
```

### 2. Dataset Preparation

Prepare your dataset in the following structure:
```
dataset/
├── real/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── fake/
│   ├── fake1.jpg
│   ├── fake2.jpg
│   └── ...
└── metadata.json  # Optional: {"image1.jpg": {"label": 0, "source": "real"}}
```

### 3. Create Configuration Files

```bash
python training/main_training_pipeline.py \
    --create-configs \
    --output-dir ./training/configs
```

### 4. Run Complete Training Pipeline

```bash
python training/main_training_pipeline.py \
    --dataset /path/to/your/dataset \
    --output-dir ./training_outputs \
    --config-dir ./training/configs
```

## 📊 Supported Datasets

The pipeline supports multiple deepfake datasets:

- **FaceForensics++**: High-quality deepfake videos
- **CelebDF**: Celebrity deepfake dataset
- **DFDC**: Deepfake Detection Challenge dataset
- **Custom datasets**: Any dataset with real/fake image structure

### Dataset Download

```bash
# Download and prepare FaceForensics++
python training/dataset_management.py \
    --download faceforensics \
    --output-dir ./datasets/faceforensics

# Download CelebDF
python training/dataset_management.py \
    --download celebdf \
    --output-dir ./datasets/celebdf
```

## 🏗️ Model Architecture

### Individual Models

1. **Enhanced MesoNet**
   - Input: 256x256 RGB images
   - Architecture: Custom CNN with mesoscopic features
   - Specialization: Artifact detection

2. **Xception (Fine-tuned)**
   - Input: 299x299 RGB images (ImageNet normalization)
   - Architecture: Pre-trained Xception with custom head
   - Specialization: Transfer learning from ImageNet

3. **EfficientNet-B4**
   - Input: 224x224 RGB images with augmentation
   - Architecture: Pre-trained EfficientNet with deepfake head
   - Specialization: High accuracy with efficient computation

4. **F3Net**
   - Input: 224x224 RGB images
   - Architecture: Frequency-domain analysis with DCT
   - Specialization: Frequency artifact detection

### Ensemble Fusion

- **Weighted Average**: Optimized weights per model
- **Attention Mechanism**: Dynamic weight assignment
- **Confidence Weighting**: Based on prediction confidence
- **Bayesian Optimization**: Automated weight tuning

## ⚙️ Configuration

### Training Configuration

```yaml
# training_config.yaml
training:
  batch_size: 32
  num_epochs: 100
  learning_rate: 0.001
  optimizer: "adam"
  scheduler: "cosine"
  mixed_precision: true
  gradient_accumulation: 4
  early_stopping_patience: 10

models:
  mesonet:
    enabled: true
    input_size: [256, 256]
    learning_rate: 0.001
  xception:
    enabled: true
    input_size: [299, 299]
    learning_rate: 0.0001
  efficientnet:
    enabled: true
    input_size: [224, 224]
    learning_rate: 0.0001
  f3net:
    enabled: true
    input_size: [224, 224]
    learning_rate: 0.001
```

### Ensemble Configuration

```yaml
# ensemble_config.yaml
ensemble:
  fusion_method: "weighted_average"
  weight_optimization: "bayesian"
  cross_validation_folds: 5
  agreement_threshold: 0.8
  calibration: "temperature_scaling"
```

## 🔧 Usage Examples

### Individual Model Training

```bash
# Train EfficientNet only
python training/model_trainer.py \
    --model efficientnet \
    --dataset /path/to/dataset \
    --config training/configs/training_config.yaml \
    --output-dir ./models/efficientnet

# Train with custom parameters
python training/model_trainer.py \
    --model xception \
    --dataset /path/to/dataset \
    --batch-size 16 \
    --learning-rate 0.0001 \
    --epochs 50
```

### Ensemble Training

```bash
# Train complete ensemble
python training/ensemble_coordinator.py \
    --dataset /path/to/dataset \
    --config training/configs/ensemble_config.yaml \
    --output-dir ./ensemble_models

# Optimize ensemble weights only
python training/ensemble_coordinator.py \
    --optimize-weights \
    --model-paths ./models/mesonet/best.pth,./models/xception/best.pth \
    --dataset /path/to/validation_dataset
```

### Model Evaluation

```bash
# Evaluate single model
python training/evaluation_framework.py \
    --model-path ./models/efficientnet/best.pth \
    --dataset /path/to/test_dataset \
    --output-dir ./evaluation_results

# Cross-dataset evaluation
python training/evaluation_framework.py \
    --model-path ./ensemble_models/best_ensemble.pth \
    --datasets /path/to/dataset1,/path/to/dataset2 \
    --cross-dataset-evaluation

# Robustness testing
python training/evaluation_framework.py \
    --model-path ./models/f3net/best.pth \
    --dataset /path/to/test_dataset \
    --robustness-testing \
    --adversarial-attacks fgsm,pgd
```

### Experiment Tracking

```bash
# Start TensorBoard
tensorboard --logdir ./training_outputs/experiments

# View experiment comparisons
python training/experiment_tracker.py \
    --experiment-dir ./training_outputs/experiments \
    --compare-experiments exp1,exp2,exp3
```

## 📈 Monitoring and Visualization

### TensorBoard Integration

The pipeline automatically logs:
- Training/validation loss and metrics
- Learning rate schedules
- Model weights and gradients
- Sample predictions and confusion matrices
- System resource utilization

```bash
# Start TensorBoard
tensorboard --logdir ./training_outputs/experiments --port 6006

# Access at http://localhost:6006
```

### Experiment Reports

Automatically generated reports include:
- Training curves and metrics
- Model comparison charts
- Confusion matrices
- ROC and Precision-Recall curves
- Cross-dataset performance analysis

## 🔄 Production Pipeline

### Automated Training

```bash
# Run production pipeline with notifications
python training/production_pipeline.py \
    --dataset /path/to/dataset \
    --config training/configs/production_config.yaml \
    --notify-email your@email.com \
    --auto-deploy
```

### Scheduled Training

```bash
# Set up cron job for weekly retraining
0 2 * * 0 cd /path/to/deepfake && \
python training/production_pipeline.py \
    --dataset /path/to/dataset \
    --scheduled-training
```

### Model Deployment

```bash
# Deploy trained model
python training/production_pipeline.py \
    --deploy-model ./ensemble_models/best_ensemble.pth \
    --deploy-path /path/to/production/models \
    --health-check
```

## 🧪 Advanced Features

### Data Augmentation

The pipeline includes deepfake-specific augmentations:
- **Spatial**: Random crop, flip, rotation
- **Color**: Brightness, contrast, saturation
- **Frequency**: DCT domain noise injection
- **Compression**: JPEG compression simulation
- **Adversarial**: CutMix, MixUp, GridMask

### Robustness Testing

- **Adversarial Attacks**: FGSM, PGD
- **Noise Robustness**: Gaussian, Salt & Pepper, Blur
- **Compression Robustness**: JPEG compression levels
- **Brightness Changes**: Various lighting conditions

### A/B Testing

```bash
# Compare two models
python training/evaluation_framework.py \
    --ab-test \
    --model-a ./models/model_a.pth \
    --model-b ./models/model_b.pth \
    --dataset /path/to/test_dataset \
    --statistical-significance
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   python training/model_trainer.py --batch-size 8
   
   # Enable gradient accumulation
   python training/model_trainer.py --gradient-accumulation 8
   ```

2. **Dataset Loading Errors**
   ```bash
   # Validate dataset structure
   python training/dataset_management.py \
       --validate-dataset /path/to/dataset
   ```

3. **Training Not Converging**
   ```bash
   # Check learning rate
   python training/model_trainer.py --learning-rate 0.0001
   
   # Enable early stopping
   python training/model_trainer.py --early-stopping-patience 15
   ```

### Performance Optimization

1. **GPU Memory Optimization**
   - Enable mixed precision training
   - Use gradient accumulation
   - Reduce batch size
   - Use model checkpointing

2. **Data Loading Optimization**
   - Use multiple workers for DataLoader
   - Enable prefetching
   - Use memory pinning

3. **Training Speed Optimization**
   - Use mixed precision (FP16)
   - Enable gradient accumulation
   - Use efficient optimizers (AdamW)

## 📋 Requirements

### System Requirements
- **GPU**: NVIDIA GPU with 8GB+ VRAM (recommended)
- **RAM**: 16GB+ system memory
- **Storage**: 100GB+ free space for datasets
- **Python**: 3.8+

### Software Dependencies
See `requirements.txt` for complete list:
- PyTorch 1.12+
- TensorBoard 2.10+
- OpenCV 4.6+
- Albumentations 1.3+
- scikit-learn 1.1+

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- FaceForensics++ dataset creators
- CelebDF dataset contributors
- Deepfake Detection Challenge organizers
- PyTorch and TensorBoard communities

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the configuration examples
3. Open an issue on GitHub
4. Contact the development team

---

**Happy Training! 🚀** 