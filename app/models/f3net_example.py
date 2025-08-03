"""
Example script demonstrating F3Net frequency-domain deepfake detector usage.
Shows frequency visualization, ensemble integration, and performance benchmarking.
"""

import logging
import time
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .f3net_detector import F3NetDetector
from .ensemble_manager import EnsembleManager, EnsembleConfig, FusionMethod
from .performance_monitor import PerformanceMonitor, MonitoringConfig


def demonstrate_f3net_detector():
    """Demonstrate F3Net detector capabilities."""
    print("=== F3Net Frequency-Domain Deepfake Detector Demonstration ===\n")
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize F3Net detector
    detector = F3NetDetector(
        model_name="F3NetDetector",
        device="auto",
        config={
            "enable_frequency_visualization": True,
            "confidence_threshold": 0.5,
            "dropout_rate": 0.3,
            "dct_block_size": 8
        }
    )
    
    # Load model
    print("Loading F3Net model...")
    if detector.load_model():
        print("✅ Model loaded successfully!")
    else:
        print("❌ Failed to load model")
        return
    
    # Display model information
    model_info = detector.get_model_info()
    print(f"\n📊 Model Information:")
    print(f"   Architecture: {model_info.architecture}")
    print(f"   Input Size: {model_info.input_size}")
    print(f"   Parameters: {model_info.parameters_count:,}")
    print(f"   Model Size: {model_info.model_size_mb:.1f} MB")
    print(f"   Device: {model_info.device}")
    
    # Display performance benchmarks
    benchmarks = detector.get_performance_benchmarks()
    print(f"\n🏆 Performance Benchmarks:")
    print(f"   AUROC: {benchmarks['auroc']:.3f}")
    print(f"   FaceForensics++ Accuracy: {benchmarks['faceforensics_accuracy']:.3f}")
    print(f"   Inference Time: {benchmarks['inference_time_ms']:.1f} ms")
    print(f"   Throughput: {benchmarks['throughput_fps']:.1f} FPS")
    
    # Display frequency analysis capabilities
    freq_info = detector.get_frequency_analysis_info()
    print(f"\n🔬 Frequency Analysis Capabilities:")
    print(f"   DCT Block Size: {freq_info['dct_block_size']}")
    print(f"   Frequency Attention: {freq_info['frequency_attention']}")
    print(f"   Frequency Filtering: {freq_info['frequency_filtering']}")
    print(f"   Spatial-Frequency Fusion: {freq_info['spatial_frequency_fusion']}")
    print(f"   Frequency Features: {', '.join(freq_info['frequency_features'])}")
    
    # Create test image
    print(f"\n🖼️  Creating test image...")
    test_image = create_test_image()
    
    # Perform prediction
    print(f"🔍 Performing deepfake detection...")
    start_time = time.time()
    result = detector.predict(test_image)
    inference_time = time.time() - start_time
    
    # Display results
    print(f"\n📈 Detection Results:")
    print(f"   Prediction: {'FAKE' if result.is_deepfake else 'REAL'}")
    print(f"   Confidence: {result.confidence:.3f}")
    print(f"   Inference Time: {result.inference_time:.3f}s")
    print(f"   Model: {result.model_name}")
    
    # Display metadata
    print(f"\n📋 Analysis Metadata:")
    for key, value in result.metadata.items():
        print(f"   {key}: {value}")
    
    # Generate and display frequency visualization
    if result.attention_maps is not None:
        print(f"\n🎨 Generating frequency visualization...")
        visualize_frequency_analysis(test_image, result.attention_maps)
    
    print(f"\n✅ F3Net demonstration completed!")


def create_test_image(size: tuple = (224, 224)) -> Image.Image:
    """Create a test image for demonstration."""
    # Create a simple test image with some patterns
    img_array = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    
    # Add some patterns that might trigger frequency analysis
    for i in range(0, size[0], 20):
        for j in range(0, size[1], 20):
            # Create a checkerboard pattern
            if (i // 20 + j // 20) % 2 == 0:
                img_array[i:i+20, j:j+20] = [100, 150, 200]
            else:
                img_array[i:i+20, j:j+20] = [200, 100, 150]
    
    # Add some noise to simulate artifacts
    noise = np.random.randint(0, 30, img_array.shape, dtype=np.uint8)
    img_array = np.clip(img_array + noise, 0, 255)
    
    return Image.fromarray(img_array)


def visualize_frequency_analysis(image: Image.Image, frequency_heatmap: np.ndarray, save_path: Optional[str] = None):
    """
    Visualize frequency domain analysis results.
    
    Args:
        image: Original image
        frequency_heatmap: Frequency domain heatmap
        save_path: Optional path to save visualization
    """
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Frequency heatmap
    im = axes[1].imshow(frequency_heatmap, cmap='jet')
    axes[1].set_title("Frequency Domain Heatmap")
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Overlay
    img_array = np.array(image)
    # Resize heatmap to match image size
    from scipy.ndimage import zoom
    zoom_factors = (img_array.shape[0] / frequency_heatmap.shape[0], 
                   img_array.shape[1] / frequency_heatmap.shape[1])
    resized_heatmap = zoom(frequency_heatmap, zoom_factors, order=1)
    resized_heatmap = np.clip(resized_heatmap, 0, 1)
    
    # Create overlay
    cmap = plt.cm.jet
    heatmap_colored = cmap(resized_heatmap)[:, :, :3]
    overlay = 0.7 * img_array / 255.0 + 0.3 * heatmap_colored
    overlay = np.clip(overlay, 0, 1)
    
    axes[2].imshow(overlay)
    axes[2].set_title("Frequency Analysis Overlay")
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   Visualization saved to: {save_path}")
    
    plt.show()


def demonstrate_ensemble_integration():
    """Demonstrate F3Net integration with ensemble framework."""
    print("\n=== F3Net Ensemble Integration Demonstration ===\n")
    
    # Initialize performance monitor
    monitor_config = MonitoringConfig(
        save_performance_data=True,
        performance_data_path="f3net_performance_data"
    )
    performance_monitor = PerformanceMonitor(monitor_config)
    
    # Initialize ensemble manager
    ensemble_config = EnsembleConfig(
        fusion_method=FusionMethod.WEIGHTED_AVERAGE,
        default_weights={"f3net": 1.0},
        temperature=1.0,
        min_models=1,
        max_models=5,
        confidence_threshold=0.5,
        enable_uncertainty=True,
        enable_attention=True
    )
    ensemble = EnsembleManager(ensemble_config)
    
    # Initialize F3Net detector
    f3net_detector = F3NetDetector(
        model_name="F3NetDetector",
        device="auto",
        config={"enable_frequency_visualization": True}
    )
    
    if f3net_detector.load_model():
        print("✅ F3Net detector loaded for ensemble")
        
        # Add to ensemble
        ensemble.add_model("f3net", f3net_detector, weight=1.0)
        
        # Create test image
        test_image = create_test_image()
        
        # Perform ensemble prediction
        print("🔍 Performing ensemble prediction...")
        performance_monitor.start_timer("f3net")
        
        try:
            ensemble_result = ensemble.predict_ensemble(test_image)
            inference_time = performance_monitor.end_timer("f3net")
            performance_monitor.record_success("f3net", True)
            
            print(f"\n📊 Ensemble Results:")
            print(f"   Prediction: {'FAKE' if ensemble_result.is_deepfake else 'REAL'}")
            print(f"   Ensemble Confidence: {ensemble_result.ensemble_confidence:.3f}")
            print(f"   Fusion Method: {ensemble_result.fusion_method}")
            print(f"   Uncertainty: {ensemble_result.uncertainty:.3f}")
            print(f"   Individual Predictions:")
            
            for model_name, result in ensemble_result.individual_predictions.items():
                print(f"     {model_name}: {result.is_deepfake} (confidence: {result.confidence:.3f})")
            
            if ensemble_result.attention_weights:
                print(f"   Attention Weights: {ensemble_result.attention_weights}")
            
        except Exception as e:
            performance_monitor.end_timer("f3net")
            performance_monitor.record_success("f3net", False)
            print(f"❌ Ensemble prediction failed: {str(e)}")
    
    # Generate performance report
    print(f"\n📈 Performance Report:")
    performance_report = performance_monitor.get_performance_report()
    print(f"   Models: {performance_report['summary'].get('total_models', 0)}")
    print(f"   Average Inference Time: {performance_report['summary'].get('avg_inference_time', 0):.3f}s")
    print(f"   Average Throughput: {performance_report['summary'].get('avg_throughput', 0):.1f} FPS")
    
    print(f"\n✅ Ensemble integration demonstration completed!")


def benchmark_frequency_performance():
    """Benchmark F3Net frequency domain performance."""
    print("\n=== F3Net Frequency Performance Benchmark ===\n")
    
    # Initialize detector
    detector = F3NetDetector(
        model_name="F3NetDetector",
        device="auto",
        config={"enable_frequency_visualization": False}  # Disable for benchmarking
    )
    
    if not detector.load_model():
        print("❌ Failed to load model for benchmarking")
        return
    
    # Create test images
    test_images = [create_test_image() for _ in range(10)]
    
    # Benchmark inference time
    print("⏱️  Benchmarking inference time...")
    times = []
    for i, image in enumerate(test_images):
        start_time = time.time()
        result = detector.predict(image)
        inference_time = time.time() - start_time
        times.append(inference_time)
        print(f"   Image {i+1}: {inference_time:.3f}s (confidence: {result.confidence:.3f})")
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    throughput = 1.0 / avg_time
    
    print(f"\n📊 Performance Results:")
    print(f"   Average Inference Time: {avg_time:.3f}s ± {std_time:.3f}s")
    print(f"   Throughput: {throughput:.1f} FPS")
    print(f"   Model Size: {detector.model_info.model_size_mb:.1f} MB")
    print(f"   Parameters: {detector.model_info.parameters_count:,}")
    
    # Memory usage (if CUDA available)
    if hasattr(torch.cuda, 'memory_allocated') and torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
        memory_reserved = torch.cuda.memory_reserved() / (1024 * 1024)  # MB
        print(f"   GPU Memory Allocated: {memory_allocated:.1f} MB")
        print(f"   GPU Memory Reserved: {memory_reserved:.1f} MB")
    
    print(f"\n✅ Frequency performance benchmark completed!")


def compare_with_spatial_models():
    """Compare F3Net with spatial models (if available)."""
    print("\n=== F3Net vs Spatial Models Comparison ===\n")
    
    try:
        # Try to import spatial models
        from .xception_detector import XceptionDetector
        from .efficientnet_detector import EfficientNetDetector
        
        # Initialize models
        f3net = F3NetDetector(model_name="F3NetDetector", device="auto")
        xception = XceptionDetector(model_name="XceptionDetector", device="auto")
        efficientnet = EfficientNetDetector(model_name="EfficientNetDetector", device="auto")
        
        models = {
            "F3Net": f3net,
            "Xception": xception,
            "EfficientNet": efficientnet
        }
        
        # Load models
        loaded_models = {}
        for name, model in models.items():
            if model.load_model():
                loaded_models[name] = model
                print(f"✅ {name} loaded successfully")
            else:
                print(f"❌ {name} failed to load")
        
        if len(loaded_models) < 2:
            print("❌ Need at least 2 models for comparison")
            return
        
        # Create test image
        test_image = create_test_image()
        
        # Compare predictions
        print(f"\n🔍 Comparing predictions on test image...")
        results = {}
        
        for name, model in loaded_models.items():
            start_time = time.time()
            result = model.predict(test_image)
            inference_time = time.time() - start_time
            
            results[name] = {
                "prediction": result.is_deepfake,
                "confidence": result.confidence,
                "inference_time": inference_time,
                "model_size_mb": model.model_info.model_size_mb
            }
            
            print(f"   {name}: {'FAKE' if result.is_deepfake else 'REAL'} "
                  f"(confidence: {result.confidence:.3f}, time: {inference_time:.3f}s)")
        
        # Display comparison summary
        print(f"\n📊 Comparison Summary:")
        print(f"   Model Predictions:")
        for name, result in results.items():
            print(f"     {name}: {'FAKE' if result['prediction'] else 'REAL'} "
                  f"(confidence: {result['confidence']:.3f})")
        
        print(f"\n   Performance Comparison:")
        fastest_model = min(results.keys(), key=lambda x: results[x]['inference_time'])
        slowest_model = max(results.keys(), key=lambda x: results[x]['inference_time'])
        
        print(f"     Fastest: {fastest_model} ({results[fastest_model]['inference_time']:.3f}s)")
        print(f"     Slowest: {slowest_model} ({results[slowest_model]['inference_time']:.3f}s)")
        
        print(f"\n   Model Sizes:")
        for name, result in results.items():
            print(f"     {name}: {result['model_size_mb']:.1f} MB")
        
        # Check for agreement
        predictions = [result['prediction'] for result in results.values()]
        agreement = len(set(predictions)) == 1
        
        print(f"\n   Model Agreement: {'✅ All models agree' if agreement else '❌ Models disagree'}")
        
        print(f"\n✅ Model comparison completed!")
        
    except ImportError as e:
        print(f"❌ Spatial models not available for comparison: {str(e)}")
        print("   This is expected if Xception or EfficientNet models are not implemented")


def main():
    """Main demonstration function."""
    print("🚀 Starting F3Net Frequency-Domain Deepfake Detector Demonstrations\n")
    
    # Basic F3Net demonstration
    demonstrate_f3net_detector()
    
    # Ensemble integration
    demonstrate_ensemble_integration()
    
    # Performance benchmarking
    benchmark_frequency_performance()
    
    # Model comparison
    compare_with_spatial_models()
    
    print(f"\n🎉 All F3Net demonstrations completed successfully!")
    print(f"\n💡 Key Features Demonstrated:")
    print(f"   • Frequency-domain analysis with DCT transforms")
    print(f"   • Local Frequency Attention mechanism")
    print(f"   • Spatial-frequency fusion")
    print(f"   • Frequency visualization capabilities")
    print(f"   • Ensemble integration")
    print(f"   • Performance benchmarking")
    print(f"   • Model comparison with spatial detectors")


if __name__ == "__main__":
    main() 