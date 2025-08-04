#!/usr/bin/env python3
"""
Direct Detection Test with Available Deepfake Samples

This script tests the detection system directly (bypassing the web API)
using the available deepfake test samples and provides performance analysis.
"""

import os
import time
import json
import numpy as np
from PIL import Image
from pathlib import Path
import logging
from typing import Dict, List, Any, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DirectDetectionTester:
    """Test the detection system directly with available samples"""
    
    def __init__(self):
        self.test_results = []
        
    def load_test_samples(self) -> Dict[str, Any]:
        """Load test samples and metadata"""
        samples_dir = Path("test_samples")
        metadata_file = samples_dir / "samples_metadata.json"
        
        if not metadata_file.exists():
            raise FileNotFoundError(f"Test samples metadata not found: {metadata_file}")
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Verify all sample files exist
        for sample_name, sample_info in metadata.items():
            sample_path = Path(sample_info["path"])
            if not sample_path.exists():
                logger.warning(f"Sample file not found: {sample_path}")
                continue
            
            # Load and verify image
            try:
                with Image.open(sample_path) as img:
                    sample_info["image_size"] = img.size
                    sample_info["image_mode"] = img.mode
            except Exception as e:
                logger.error(f"Failed to load image {sample_path}: {e}")
                continue
        
        logger.info(f"Loaded {len(metadata)} test samples")
        return metadata
    
    def test_detection_directly(self, samples_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Test detection directly with the detection system"""
        logger.info("Testing detection directly...")
        
        results = []
        
        # Import the detection system
        try:
            from app.utils.deepfake_detector import DeepfakeDetector
            detector = DeepfakeDetector()
            logger.info("✓ DeepfakeDetector loaded successfully")
        except Exception as e:
            logger.error(f"✗ Failed to load DeepfakeDetector: {e}")
            return results
        
        for sample_name, sample_info in samples_metadata.items():
            sample_path = Path(sample_info["path"])
            if not sample_path.exists():
                continue
            
            logger.info(f"Testing sample: {sample_name}")
            
            try:
                # Load image
                image = Image.open(sample_path).convert('RGB')
                
                # Perform detection
                start_time = time.time()
                detection_result = detector.analyze_image(str(sample_path))
                processing_time = time.time() - start_time
                
                # Extract results
                is_deepfake = detection_result.get("is_deepfake", False)
                confidence = detection_result.get("confidence_score", 0.0) / 100.0  # Convert to 0-1
                
                result = {
                    "sample_name": sample_name,
                    "expected": sample_info["expected"],
                    "prediction": is_deepfake,
                    "confidence": confidence,
                    "processing_time": processing_time,
                    "correct": is_deepfake == sample_info["expected"],
                    "sample_description": sample_info["description"],
                    "sample_source": sample_info["source"],
                    "metadata": detection_result.get("analysis_metadata", {})
                }
                
                results.append(result)
                logger.info(f"✓ {sample_name}: "
                          f"Prediction={'FAKE' if result['prediction'] else 'REAL'}, "
                          f"Confidence={result['confidence']:.3f}, "
                          f"Correct={result['correct']}")
                
            except Exception as e:
                logger.error(f"✗ {sample_name}: {e}")
                results.append({
                    "sample_name": sample_name,
                    "expected": sample_info["expected"],
                    "error": str(e),
                    "correct": False
                })
        
        return results
    
    def test_ensemble_detection(self, samples_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Test ensemble detection with multiple models"""
        logger.info("Testing ensemble detection...")
        
        results = []
        
        # Import the ensemble detection system
        try:
            from app.utils.enhanced_deepfake_detector import EnhancedDeepfakeDetector
            detector = EnhancedDeepfakeDetector()
            logger.info("✓ EnhancedDeepfakeDetector loaded successfully")
        except Exception as e:
            logger.error(f"✗ Failed to load EnhancedDeepfakeDetector: {e}")
            return results
        
        for sample_name, sample_info in samples_metadata.items():
            sample_path = Path(sample_info["path"])
            if not sample_path.exists():
                continue
            
            logger.info(f"Testing ensemble sample: {sample_name}")
            
            try:
                # Load image
                image = Image.open(sample_path).convert('RGB')
                
                # Perform ensemble detection
                start_time = time.time()
                detection_result = detector.analyze_image(str(sample_path))
                processing_time = time.time() - start_time
                
                # Extract results
                is_deepfake = detection_result.get("is_deepfake", False)
                confidence = detection_result.get("confidence_score", 0.0) / 100.0  # Convert to 0-1
                
                result = {
                    "sample_name": f"{sample_name}_ensemble",
                    "expected": sample_info["expected"],
                    "prediction": is_deepfake,
                    "confidence": confidence,
                    "processing_time": processing_time,
                    "correct": is_deepfake == sample_info["expected"],
                    "sample_description": sample_info["description"],
                    "sample_source": sample_info["source"],
                    "metadata": detection_result.get("analysis_metadata", {})
                }
                
                results.append(result)
                logger.info(f"✓ {sample_name} (ensemble): "
                          f"Prediction={'FAKE' if result['prediction'] else 'REAL'}, "
                          f"Confidence={result['confidence']:.3f}, "
                          f"Correct={result['correct']}")
                
            except Exception as e:
                logger.error(f"✗ {sample_name} (ensemble): {e}")
                results.append({
                    "sample_name": f"{sample_name}_ensemble",
                    "expected": sample_info["expected"],
                    "error": str(e),
                    "correct": False
                })
        
        return results
    
    def analyze_results(self, detection_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze detection results and generate performance metrics"""
        logger.info("Analyzing results...")
        
        if not detection_results:
            return {"error": "No detection results to analyze"}
        
        # Calculate overall metrics
        correct = sum(1 for r in detection_results if r.get("correct", False))
        total = len(detection_results)
        accuracy = correct / total if total > 0 else 0
        
        confidences = [r.get("confidence", 0) for r in detection_results if "confidence" in r]
        processing_times = [r.get("processing_time", 0) for r in detection_results if "processing_time" in r]
        
        # Group by expected vs actual
        true_positives = sum(1 for r in detection_results 
                           if r.get("expected", False) and r.get("prediction", False))
        false_positives = sum(1 for r in detection_results 
                            if not r.get("expected", False) and r.get("prediction", False))
        true_negatives = sum(1 for r in detection_results 
                           if not r.get("expected", False) and not r.get("prediction", False))
        false_negatives = sum(1 for r in detection_results 
                            if r.get("expected", False) and not r.get("prediction", False))
        
        # Calculate precision, recall, F1
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        analysis = {
            "total_samples": total,
            "correct_predictions": correct,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "true_negatives": true_negatives,
            "false_negatives": false_negatives,
            "avg_confidence": np.mean(confidences) if confidences else 0,
            "avg_processing_time": np.mean(processing_times) if processing_times else 0,
            "min_confidence": np.min(confidences) if confidences else 0,
            "max_confidence": np.max(confidences) if confidences else 0
        }
        
        return analysis
    
    def generate_report(self, test_results: Dict[str, Any]) -> str:
        """Generate comprehensive test report"""
        logger.info("Generating test report...")
        
        report_lines = [
            "=" * 80,
            "DIRECT DETECTION TEST REPORT",
            "=" * 80,
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]
        
        # Detection analysis
        if "detection_analysis" in test_results:
            analysis = test_results["detection_analysis"]
            report_lines.extend([
                "DETECTION PERFORMANCE ANALYSIS",
                "-" * 40,
                f"Total Samples Tested: {analysis.get('total_samples', 0)}",
                f"Overall Accuracy: {analysis.get('accuracy', 0):.3f}",
                f"Precision: {analysis.get('precision', 0):.3f}",
                f"Recall: {analysis.get('recall', 0):.3f}",
                f"F1-Score: {analysis.get('f1_score', 0):.3f}",
                f"Average Processing Time: {analysis.get('avg_processing_time', 0):.3f}s",
                f"Average Confidence: {analysis.get('avg_confidence', 0):.3f}",
                ""
            ])
            
            # Confusion matrix
            report_lines.extend([
                "CONFUSION MATRIX",
                "-" * 40,
                f"True Positives: {analysis.get('true_positives', 0)}",
                f"False Positives: {analysis.get('false_positives', 0)}",
                f"True Negatives: {analysis.get('true_negatives', 0)}",
                f"False Negatives: {analysis.get('false_negatives', 0)}",
                ""
            ])
        
        # Individual sample results
        if "detection_results" in test_results:
            report_lines.extend([
                "INDIVIDUAL SAMPLE RESULTS",
                "-" * 40
            ])
            
            for result in test_results["detection_results"]:
                sample_name = result.get("sample_name", "Unknown")
                expected = "FAKE" if result.get("expected") else "REAL"
                prediction = "FAKE" if result.get("prediction") else "REAL"
                confidence = result.get("confidence", 0)
                correct = "✓" if result.get("correct") else "✗"
                
                report_lines.append(f"{sample_name}: {expected} → {prediction} "
                                  f"(conf: {confidence:.3f}) {correct}")
                
                if "error" in result:
                    report_lines.append(f"  Error: {result['error']}")
            report_lines.append("")
        
        # Recommendations
        report_lines.extend([
            "RECOMMENDATIONS",
            "-" * 40
        ])
        
        if "detection_analysis" in test_results:
            analysis = test_results["detection_analysis"]
            
            accuracy = analysis.get("accuracy", 0)
            if accuracy < 0.8:
                report_lines.append("• Consider improving model accuracy through additional training")
            
            avg_processing_time = analysis.get("avg_processing_time", 0)
            if avg_processing_time > 5.0:
                report_lines.append("• Consider optimizing processing times for better performance")
            
            precision = analysis.get("precision", 0)
            recall = analysis.get("recall", 0)
            if precision < 0.7 or recall < 0.7:
                report_lines.append("• Consider balancing precision and recall for better performance")
        
        report_lines.extend([
            "",
            "=" * 80,
            "END OF REPORT",
            "=" * 80
        ])
        
        return "\n".join(report_lines)
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive test of the detection system"""
        logger.info("Starting comprehensive direct detection test...")
        
        test_results = {}
        
        try:
            # 1. Load test samples
            logger.info("1. Loading test samples...")
            samples_metadata = self.load_test_samples()
            
            # 2. Test basic detection
            logger.info("2. Testing basic detection...")
            basic_results = self.test_detection_directly(samples_metadata)
            test_results["basic_results"] = basic_results
            
            # 3. Test ensemble detection
            logger.info("3. Testing ensemble detection...")
            ensemble_results = self.test_ensemble_detection(samples_metadata)
            test_results["ensemble_results"] = ensemble_results
            
            # 4. Combine results
            all_results = basic_results + ensemble_results
            test_results["detection_results"] = all_results
            
            # 5. Analyze results
            logger.info("4. Analyzing results...")
            test_results["detection_analysis"] = self.analyze_results(all_results)
            
            # 6. Generate report
            logger.info("5. Generating report...")
            report = self.generate_report(test_results)
            test_results["report"] = report
            
            # Save results
            with open("direct_detection_test_results.json", "w") as f:
                json.dump(test_results, f, indent=2, default=str)
            
            with open("direct_detection_test_report.txt", "w") as f:
                f.write(report)
            
            logger.info("✓ Comprehensive test completed successfully!")
            logger.info("Results saved to: direct_detection_test_results.json")
            logger.info("Report saved to: direct_detection_test_report.txt")
            
            return test_results
            
        except Exception as e:
            logger.error(f"✗ Comprehensive test failed: {e}")
            return {"error": str(e)}


def main():
    """Main test function"""
    print("🚀 Direct Detection Tester")
    print("=" * 50)
    
    # Initialize tester
    tester = DirectDetectionTester()
    
    # Run comprehensive test
    results = tester.run_comprehensive_test()
    
    # Print summary
    if "error" not in results:
        print("\n📊 TEST SUMMARY:")
        print(f"✓ Basic detection samples tested: {len(results.get('basic_results', []))}")
        print(f"✓ Ensemble detection samples tested: {len(results.get('ensemble_results', []))}")
        
        if "detection_analysis" in results:
            analysis = results["detection_analysis"]
            overall_accuracy = analysis.get("accuracy", 0)
            f1_score = analysis.get("f1_score", 0)
            print(f"✓ Overall accuracy: {overall_accuracy:.3f}")
            print(f"✓ F1-Score: {f1_score:.3f}")
        
        print("\n📄 Check the generated files for detailed results:")
        print("  - direct_detection_test_results.json (detailed results)")
        print("  - direct_detection_test_report.txt (comprehensive report)")
    else:
        print(f"\n❌ Test failed: {results['error']}")


if __name__ == "__main__":
    main() 