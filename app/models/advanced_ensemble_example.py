"""
Advanced Ensemble Example Usage

This script demonstrates how to use the advanced ensemble system
with different fusion methods and evaluation capabilities.
"""

import logging
import time
from pathlib import Path
from typing import List, Tuple, Any

from PIL import Image
import numpy as np

from .advanced_ensemble import (
    AdvancedEnsembleManager, AdvancedEnsembleConfig, AdvancedFusionMethod
)
from .advanced_ensemble_evaluator import AdvancedEnsembleEvaluator
from .xception_detector import XceptionDetector
from .efficientnet_detector import EfficientNetDetector
from .f3net_detector import F3NetDetector


def create_sample_data(num_samples: int = 100) -> List[Tuple[Any, bool]]:
    """Create sample data for demonstration purposes."""
    # This would normally load real images and labels
    # For demonstration, we'll create placeholder data
    sample_data = []
    
    for i in range(num_samples):
        # Create a dummy image (in practice, this would be a real image)
        # For now, we'll use a placeholder
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        image = Image.fromarray(image)
        
        # Random label (in practice, this would be ground truth)
        label = np.random.choice([True, False])
        
        sample_data.append((image, label))
    
    return sample_data


def demonstrate_advanced_ensemble():
    """Demonstrate the advanced ensemble system capabilities."""
    
    print("=== Advanced Ensemble System Demonstration ===\n")
    
    # 1. Create different ensemble configurations
    print("1. Creating ensemble configurations...")
    
    configs = {
        "attention_ensemble": AdvancedEnsembleConfig(
            fusion_method=AdvancedFusionMethod.ATTENTION_MERGE,
            attention_dim=128,
            attention_heads=8,
            learn_attention_weights=True
        ),
        
        "temperature_ensemble": AdvancedEnsembleConfig(
            fusion_method=AdvancedFusionMethod.TEMPERATURE_SCALED,
            temperature=1.5,
            calibrate_temperature=True
        ),
        
        "mc_dropout_ensemble": AdvancedEnsembleConfig(
            fusion_method=AdvancedFusionMethod.MONTE_CARLO_DROPOUT,
            mc_dropout_samples=30,
            mc_dropout_rate=0.1
        ),
        
        "adaptive_ensemble": AdvancedEnsembleConfig(
            fusion_method=AdvancedFusionMethod.ADAPTIVE_WEIGHTING,
            enable_adaptive_weighting=True,
            feature_extraction_dim=256
        ),
        
        "agreement_ensemble": AdvancedEnsembleConfig(
            fusion_method=AdvancedFusionMethod.AGREEMENT_RESOLUTION,
            agreement_threshold=0.7,
            conflict_resolution_method="confidence_weighted"
        )
    }
    
    # 2. Create ensemble managers
    print("2. Initializing ensemble managers...")
    
    ensembles = {}
    for name, config in configs.items():
        try:
            ensemble = AdvancedEnsembleManager(config)
            
            # Add models to ensemble
            # Note: In practice, you would load actual model weights
            print(f"   Adding models to {name}...")
            
            # Add Xception (placeholder - would load actual model)
            # xception = XceptionDetector()
            # if xception.load_model():
            #     ensemble.add_model("xception", xception, weight=1.0)
            
            # Add EfficientNet (placeholder - would load actual model)
            # efficientnet = EfficientNetDetector()
            # if efficientnet.load_model():
            #     ensemble.add_model("efficientnet", efficientnet, weight=1.0)
            
            # Add F3Net (placeholder - would load actual model)
            # f3net = F3NetDetector()
            # if f3net.load_model():
            #     ensemble.add_model("f3net", f3net, weight=1.0)
            
            ensembles[name] = ensemble
            print(f"   ✓ {name} initialized successfully")
            
        except Exception as e:
            print(f"   ✗ Failed to initialize {name}: {str(e)}")
    
    # 3. Create sample data
    print("\n3. Creating sample data...")
    test_data = create_sample_data(num_samples=50)
    print(f"   Created {len(test_data)} sample images")
    
    # 4. Demonstrate predictions
    print("\n4. Running predictions...")
    
    for name, ensemble in ensembles.items():
        try:
            print(f"   Testing {name}...")
            
            # Test on a few samples
            for i, (image, label) in enumerate(test_data[:5]):
                start_time = time.time()
                result = ensemble.predict_advanced(image)
                inference_time = time.time() - start_time
                
                print(f"     Sample {i+1}:")
                print(f"       Prediction: {'FAKE' if result.is_deepfake else 'REAL'}")
                print(f"       Confidence: {result.confidence:.4f}")
                print(f"       Uncertainty: {result.uncertainty:.4f}")
                print(f"       Agreement Score: {result.agreement_score:.4f}")
                print(f"       Inference Time: {inference_time:.4f}s")
                
                if result.attention_weights:
                    print(f"       Attention Weights: {result.attention_weights}")
                
                if result.adaptive_weights:
                    print(f"       Adaptive Weights: {result.adaptive_weights}")
                
                print()
            
        except Exception as e:
            print(f"     ✗ Failed to run predictions for {name}: {str(e)}")
    
    # 5. Demonstrate evaluation
    print("\n5. Running evaluation...")
    
    try:
        evaluator = AdvancedEnsembleEvaluator(output_dir="evaluation_demo")
        
        # Evaluate each ensemble
        benchmark_results = []
        for name, ensemble in ensembles.items():
            try:
                print(f"   Evaluating {name}...")
                result = evaluator.evaluate_ensemble(
                    ensemble, test_data, "sample_dataset", name
                )
                benchmark_results.append(result)
                print(f"     ✓ Evaluation completed")
                
            except Exception as e:
                print(f"     ✗ Evaluation failed for {name}: {str(e)}")
        
        # Compare ensembles
        if len(benchmark_results) > 1:
            print("\n   Comparing ensembles...")
            comparison = evaluator.compare_ensembles(benchmark_results, "ensemble_comparison")
            
            print("   Comparison Results:")
            for result in benchmark_results:
                print(f"     {result.ensemble_name}:")
                print(f"       Accuracy: {result.metrics.accuracy:.4f}")
                print(f"       F1-Score: {result.metrics.f1_score:.4f}")
                print(f"       AUC-ROC: {result.metrics.auc_roc:.4f}")
                print(f"       Mean Uncertainty: {result.metrics.mean_uncertainty:.4f}")
                print(f"       Agreement Score: {result.metrics.agreement_score:.4f}")
                print(f"       Throughput: {result.metrics.throughput_fps:.2f} FPS")
                print()
        
        # Generate report
        print("   Generating evaluation report...")
        report = evaluator.generate_evaluation_report(benchmark_results, "demo_report")
        print("   ✓ Report generated successfully")
        
    except Exception as e:
        print(f"   ✗ Evaluation failed: {str(e)}")
    
    # 6. Demonstrate temperature calibration
    print("\n6. Demonstrating temperature calibration...")
    
    try:
        # Use a subset of data for calibration
        calibration_data = test_data[:20]
        
        for name, ensemble in ensembles.items():
            if ensemble.config.fusion_method == AdvancedFusionMethod.TEMPERATURE_SCALED:
                print(f"   Calibrating {name}...")
                success = ensemble.calibrate_temperature(calibration_data)
                if success:
                    print(f"     ✓ Temperature calibration completed")
                    print(f"     Optimal temperature: {ensemble.temperature_scaler.temperature.item():.4f}")
                else:
                    print(f"     ✗ Temperature calibration failed")
                break
    
    except Exception as e:
        print(f"   ✗ Temperature calibration failed: {str(e)}")
    
    # 7. Demonstrate state saving/loading
    print("\n7. Demonstrating state persistence...")
    
    try:
        # Save ensemble state
        for name, ensemble in ensembles.items():
            state_file = f"ensemble_state_{name}.json"
            success = ensemble.save_ensemble_state(state_file)
            if success:
                print(f"   ✓ Saved state for {name}")
            else:
                print(f"   ✗ Failed to save state for {name}")
        
        # Load ensemble state
        print("   Loading ensemble state...")
        test_ensemble = AdvancedEnsembleManager()
        success = test_ensemble.load_ensemble_state("ensemble_state_attention_ensemble.json")
        if success:
            print("   ✓ Successfully loaded ensemble state")
        else:
            print("   ✗ Failed to load ensemble state")
    
    except Exception as e:
        print(f"   ✗ State persistence failed: {str(e)}")
    
    print("\n=== Demonstration Completed ===")
    print("\nKey Features Demonstrated:")
    print("✓ Multiple fusion methods (attention, temperature scaling, MC dropout, etc.)")
    print("✓ Comprehensive evaluation metrics")
    print("✓ Ensemble comparison and ranking")
    print("✓ Temperature calibration")
    print("✓ State persistence")
    print("✓ Uncertainty quantification")
    print("✓ Model agreement analysis")
    print("✓ Cross-dataset evaluation framework")
    
    return ensembles


def run_benchmark_comparison():
    """Run a comprehensive benchmark comparison of different ensemble methods."""
    
    print("=== Advanced Ensemble Benchmark Comparison ===\n")
    
    # Create evaluator
    evaluator = AdvancedEnsembleEvaluator(output_dir="benchmark_results")
    
    # Create different ensemble configurations for comparison
    ensemble_configs = {
        "Baseline_Weighted_Average": AdvancedEnsembleConfig(
            fusion_method=AdvancedFusionMethod.ATTENTION_MERGE,
            learn_attention_weights=False
        ),
        
        "Attention_Based": AdvancedEnsembleConfig(
            fusion_method=AdvancedFusionMethod.ATTENTION_MERGE,
            attention_dim=128,
            attention_heads=8,
            learn_attention_weights=True
        ),
        
        "Temperature_Scaled": AdvancedEnsembleConfig(
            fusion_method=AdvancedFusionMethod.TEMPERATURE_SCALED,
            temperature=1.5,
            calibrate_temperature=True
        ),
        
        "MC_Dropout": AdvancedEnsembleConfig(
            fusion_method=AdvancedFusionMethod.MONTE_CARLO_DROPOUT,
            mc_dropout_samples=30,
            mc_dropout_rate=0.1
        ),
        
        "Adaptive_Weighting": AdvancedEnsembleConfig(
            fusion_method=AdvancedFusionMethod.ADAPTIVE_WEIGHTING,
            enable_adaptive_weighting=True
        ),
        
        "Agreement_Resolution": AdvancedEnsembleConfig(
            fusion_method=AdvancedFusionMethod.AGREEMENT_RESOLUTION,
            agreement_threshold=0.7
        )
    }
    
    # Create ensembles
    ensembles = {}
    for name, config in ensemble_configs.items():
        try:
            ensemble = AdvancedEnsembleManager(config)
            # Add models (placeholder)
            ensembles[name] = ensemble
            print(f"✓ Created {name}")
        except Exception as e:
            print(f"✗ Failed to create {name}: {str(e)}")
    
    # Create test datasets
    test_datasets = {
        "synthetic_dataset": create_sample_data(100),
        "validation_dataset": create_sample_data(50)
    }
    
    # Run benchmark suite
    try:
        suite_results = evaluator.create_benchmark_suite(
            ensembles, test_datasets, "advanced_ensemble_benchmark"
        )
        
        print("\n=== Benchmark Results ===")
        print(f"Total evaluations: {suite_results['total_evaluations']}")
        print(f"Number of ensembles: {suite_results['num_ensembles']}")
        print(f"Number of datasets: {suite_results['num_datasets']}")
        
        # Print best ensemble
        best_ensemble = suite_results['comparison_result']['best_ensemble']
        print(f"\nBest Ensemble: {best_ensemble['best_ensemble']}")
        print(f"Weighted Score: {best_ensemble['weighted_score']:.4f}")
        
        print("\n✓ Benchmark completed successfully!")
        print("Check 'benchmark_results' directory for detailed reports and plots.")
        
    except Exception as e:
        print(f"✗ Benchmark failed: {str(e)}")
    
    return suite_results


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run demonstration
    print("Running Advanced Ensemble Demonstration...")
    ensembles = demonstrate_advanced_ensemble()
    
    # Run benchmark comparison
    print("\n" + "="*60)
    print("Running Benchmark Comparison...")
    benchmark_results = run_benchmark_comparison()
    
    print("\nAdvanced Ensemble System demonstration completed!")
    print("Check the generated files for detailed results and analysis.") 