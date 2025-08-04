#!/usr/bin/env python3
"""
Test Advanced Ensemble Website with Real Deepfake Samples

This script tests the website's advanced ensemble system using the available
deepfake test samples and provides comprehensive performance analysis.
"""

import os
import time
import json
import requests
import numpy as np
from PIL import Image
from pathlib import Path
import logging
from typing import Dict, List, Any, Tuple
import base64
from io import BytesIO

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedEnsembleWebsiteTester:
    """Test the advanced ensemble system through the website API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.test_results = []
        self.performance_metrics = {}
        
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
    
    def test_basic_endpoints(self) -> Dict[str, Any]:
        """Test basic website endpoints"""
        logger.info("Testing basic website endpoints...")
        
        endpoints = {
            "health": "/health",
            "info": "/advanced-ensemble/info",
            "models": "/models",
            "upload": "/upload"
        }
        
        results = {}
        for name, endpoint in endpoints.items():
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
                results[name] = {
                    "status_code": response.status_code,
                    "success": response.status_code == 200,
                    "response_time": response.elapsed.total_seconds()
                }
                logger.info(f"✓ {name}: {response.status_code}")
            except Exception as e:
                results[name] = {
                    "status_code": None,
                    "success": False,
                    "error": str(e)
                }
                logger.error(f"✗ {name}: {e}")
        
        return results
    
    def test_ensemble_configuration(self) -> Dict[str, Any]:
        """Test different ensemble configurations"""
        logger.info("Testing ensemble configurations...")
        
        configs = {
            "attention_merge": {
                "fusion_method": "attention_merge",
                "attention_dim": 128,
                "attention_heads": 8,
                "learn_attention_weights": True
            },
            "temperature_scaled": {
                "fusion_method": "temperature_scaled",
                "temperature": 1.5,
                "calibrate_temperature": True
            },
            "mc_dropout": {
                "fusion_method": "monte_carlo_dropout",
                "mc_dropout_samples": 30,
                "mc_dropout_rate": 0.1
            },
            "adaptive_weighting": {
                "fusion_method": "adaptive_weighting",
                "enable_adaptive_weighting": True,
                "feature_extraction_dim": 256
            },
            "agreement_resolution": {
                "fusion_method": "agreement_resolution",
                "agreement_threshold": 0.7,
                "conflict_resolution_method": "confidence_weighted"
            }
        }
        
        results = {}
        for config_name, config in configs.items():
            try:
                response = requests.post(
                    f"{self.base_url}/advanced-ensemble/configure",
                    json=config,
                    timeout=30
                )
                
                results[config_name] = {
                    "status_code": response.status_code,
                    "success": response.status_code == 200,
                    "response_time": response.elapsed.total_seconds()
                }
                
                if response.status_code == 200:
                    response_data = response.json()
                    results[config_name]["config"] = response_data.get("config", {})
                    results[config_name]["models_loaded"] = response_data.get("models_loaded", 0)
                    logger.info(f"✓ {config_name}: {response_data.get('models_loaded', 0)} models loaded")
                else:
                    logger.error(f"✗ {config_name}: {response.status_code}")
                    
            except Exception as e:
                results[config_name] = {
                    "status_code": None,
                    "success": False,
                    "error": str(e)
                }
                logger.error(f"✗ {config_name}: {e}")
        
        return results
    
    def test_prediction_with_samples(self, samples_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Test predictions with all available samples"""
        logger.info("Testing predictions with samples...")
        
        results = []
        
        for sample_name, sample_info in samples_metadata.items():
            sample_path = Path(sample_info["path"])
            if not sample_path.exists():
                continue
            
            logger.info(f"Testing sample: {sample_name}")
            
            # Test with different fusion methods
            fusion_methods = ["attention_merge", "temperature_scaled", "mc_dropout", "adaptive_weighting"]
            
            for fusion_method in fusion_methods:
                try:
                    # Prepare image for upload
                    with open(sample_path, 'rb') as f:
                        files = {'file': (sample_path.name, f, 'image/jpeg')}
                        data = {'fusion_method': fusion_method}
                        
                        # Make prediction request
                        start_time = time.time()
                        response = requests.post(
                            f"{self.base_url}/advanced-ensemble/predict",
                            files=files,
                            data=data,
                            timeout=60
                        )
                        response_time = time.time() - start_time
                        
                        if response.status_code == 200:
                            prediction_data = response.json()
                            
                            result = {
                                "sample_name": sample_name,
                                "fusion_method": fusion_method,
                                "expected": sample_info["expected"],
                                "prediction": prediction_data.get("is_deepfake", False),
                                "confidence": prediction_data.get("confidence", 0.0),
                                "uncertainty": prediction_data.get("uncertainty", 0.0),
                                "agreement_score": prediction_data.get("agreement_score", 0.0),
                                "attention_weights": prediction_data.get("attention_weights", {}),
                                "adaptive_weights": prediction_data.get("adaptive_weights", {}),
                                "processing_time": prediction_data.get("processing_time", 0.0),
                                "response_time": response_time,
                                "correct": prediction_data.get("is_deepfake", False) == sample_info["expected"],
                                "sample_description": sample_info["description"],
                                "sample_source": sample_info["source"]
                            }
                            
                            results.append(result)
                            logger.info(f"✓ {sample_name} ({fusion_method}): "
                                      f"Prediction={'FAKE' if result['prediction'] else 'REAL'}, "
                                      f"Confidence={result['confidence']:.3f}, "
                                      f"Correct={result['correct']}")
                        else:
                            logger.error(f"✗ {sample_name} ({fusion_method}): HTTP {response.status_code}")
                            
                except Exception as e:
                    logger.error(f"✗ {sample_name} ({fusion_method}): {e}")
                    results.append({
                        "sample_name": sample_name,
                        "fusion_method": fusion_method,
                        "expected": sample_info["expected"],
                        "error": str(e),
                        "correct": False
                    })
        
        return results
    
    def test_evaluation_endpoints(self) -> Dict[str, Any]:
        """Test evaluation and benchmarking endpoints"""
        logger.info("Testing evaluation endpoints...")
        
        # Test evaluation request
        evaluation_request = {
            "ensemble_config": {
                "fusion_method": "attention_merge",
                "attention_dim": 128,
                "mc_dropout_samples": 30
            },
            "evaluation_name": "website_test_evaluation"
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/advanced-ensemble/evaluate",
                json=evaluation_request,
                timeout=30
            )
            
            if response.status_code == 200:
                eval_data = response.json()
                logger.info(f"✓ Evaluation started: {eval_data.get('evaluation_id')}")
                
                # Wait a bit and check status
                time.sleep(2)
                eval_id = eval_data.get('evaluation_id')
                
                if eval_id:
                    status_response = requests.get(
                        f"{self.base_url}/advanced-ensemble/evaluation/{eval_id}",
                        timeout=10
                    )
                    
                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        logger.info(f"✓ Evaluation status: {status_data.get('status')}")
                        return {
                            "evaluation_started": True,
                            "evaluation_id": eval_id,
                            "status": status_data.get('status')
                        }
            
            return {"evaluation_started": False, "error": f"HTTP {response.status_code}"}
            
        except Exception as e:
            logger.error(f"✗ Evaluation test failed: {e}")
            return {"evaluation_started": False, "error": str(e)}
    
    def analyze_results(self, prediction_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze prediction results and generate performance metrics"""
        logger.info("Analyzing results...")
        
        if not prediction_results:
            return {"error": "No prediction results to analyze"}
        
        # Group by fusion method
        by_method = {}
        for result in prediction_results:
            method = result.get("fusion_method", "unknown")
            if method not in by_method:
                by_method[method] = []
            by_method[method].append(result)
        
        analysis = {
            "total_samples": len(prediction_results),
            "by_fusion_method": {},
            "overall_metrics": {}
        }
        
        # Calculate metrics for each fusion method
        for method, results in by_method.items():
            correct = sum(1 for r in results if r.get("correct", False))
            total = len(results)
            accuracy = correct / total if total > 0 else 0
            
            confidences = [r.get("confidence", 0) for r in results if "confidence" in r]
            uncertainties = [r.get("uncertainty", 0) for r in results if "uncertainty" in r]
            response_times = [r.get("response_time", 0) for r in results if "response_time" in r]
            
            analysis["by_fusion_method"][method] = {
                "total_samples": total,
                "accuracy": accuracy,
                "correct_predictions": correct,
                "avg_confidence": np.mean(confidences) if confidences else 0,
                "avg_uncertainty": np.mean(uncertainties) if uncertainties else 0,
                "avg_response_time": np.mean(response_times) if response_times else 0,
                "min_confidence": np.min(confidences) if confidences else 0,
                "max_confidence": np.max(confidences) if confidences else 0
            }
        
        # Overall metrics
        all_correct = sum(1 for r in prediction_results if r.get("correct", False))
        all_total = len(prediction_results)
        analysis["overall_metrics"] = {
            "total_accuracy": all_correct / all_total if all_total > 0 else 0,
            "total_correct": all_correct,
            "total_samples": all_total,
            "avg_response_time": np.mean([r.get("response_time", 0) for r in prediction_results])
        }
        
        return analysis
    
    def generate_report(self, test_results: Dict[str, Any]) -> str:
        """Generate comprehensive test report"""
        logger.info("Generating test report...")
        
        report_lines = [
            "=" * 80,
            "ADVANCED ENSEMBLE WEBSITE TEST REPORT",
            "=" * 80,
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Base URL: {self.base_url}",
            ""
        ]
        
        # Basic endpoint test results
        if "basic_endpoints" in test_results:
            report_lines.extend([
                "BASIC ENDPOINT TESTS",
                "-" * 40
            ])
            
            for endpoint, result in test_results["basic_endpoints"].items():
                status = "✓ PASS" if result.get("success") else "✗ FAIL"
                report_lines.append(f"{endpoint}: {status}")
                if not result.get("success"):
                    report_lines.append(f"  Error: {result.get('error', 'Unknown error')}")
            report_lines.append("")
        
        # Ensemble configuration test results
        if "ensemble_configs" in test_results:
            report_lines.extend([
                "ENSEMBLE CONFIGURATION TESTS",
                "-" * 40
            ])
            
            for config, result in test_results["ensemble_configs"].items():
                status = "✓ PASS" if result.get("success") else "✗ FAIL"
                models_loaded = result.get("models_loaded", 0)
                report_lines.append(f"{config}: {status} ({models_loaded} models)")
            report_lines.append("")
        
        # Prediction analysis
        if "prediction_analysis" in test_results:
            analysis = test_results["prediction_analysis"]
            report_lines.extend([
                "PREDICTION PERFORMANCE ANALYSIS",
                "-" * 40,
                f"Total Samples Tested: {analysis.get('total_samples', 0)}",
                f"Overall Accuracy: {analysis.get('overall_metrics', {}).get('total_accuracy', 0):.3f}",
                f"Average Response Time: {analysis.get('overall_metrics', {}).get('avg_response_time', 0):.3f}s",
                ""
            ])
            
            # By fusion method
            for method, metrics in analysis.get("by_fusion_method", {}).items():
                report_lines.extend([
                    f"{method.upper()} FUSION METHOD:",
                    f"  Accuracy: {metrics.get('accuracy', 0):.3f}",
                    f"  Samples: {metrics.get('total_samples', 0)}",
                    f"  Avg Confidence: {metrics.get('avg_confidence', 0):.3f}",
                    f"  Avg Uncertainty: {metrics.get('avg_uncertainty', 0):.3f}",
                    f"  Avg Response Time: {metrics.get('avg_response_time', 0):.3f}s",
                    ""
                ])
        
        # Evaluation test results
        if "evaluation_test" in test_results:
            eval_result = test_results["evaluation_test"]
            report_lines.extend([
                "EVALUATION ENDPOINT TESTS",
                "-" * 40
            ])
            
            if eval_result.get("evaluation_started"):
                report_lines.append(f"✓ Evaluation started successfully")
                report_lines.append(f"  Evaluation ID: {eval_result.get('evaluation_id')}")
                report_lines.append(f"  Status: {eval_result.get('status')}")
            else:
                report_lines.append(f"✗ Evaluation failed")
                report_lines.append(f"  Error: {eval_result.get('error')}")
            report_lines.append("")
        
        # Recommendations
        report_lines.extend([
            "RECOMMENDATIONS",
            "-" * 40
        ])
        
        if "prediction_analysis" in test_results:
            analysis = test_results["prediction_analysis"]
            
            # Find best performing method
            best_method = None
            best_accuracy = 0
            for method, metrics in analysis.get("by_fusion_method", {}).items():
                if metrics.get("accuracy", 0) > best_accuracy:
                    best_accuracy = metrics.get("accuracy", 0)
                    best_method = method
            
            if best_method:
                report_lines.append(f"• Best performing fusion method: {best_method} ({best_accuracy:.3f} accuracy)")
            
            # Performance recommendations
            avg_response_time = analysis.get("overall_metrics", {}).get("avg_response_time", 0)
            if avg_response_time > 5.0:
                report_lines.append("• Consider optimizing response times for better user experience")
            
            # Accuracy recommendations
            overall_accuracy = analysis.get("overall_metrics", {}).get("total_accuracy", 0)
            if overall_accuracy < 0.8:
                report_lines.append("• Consider improving model accuracy through additional training or ensemble diversity")
        
        report_lines.extend([
            "",
            "=" * 80,
            "END OF REPORT",
            "=" * 80
        ])
        
        return "\n".join(report_lines)
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive test of the advanced ensemble website"""
        logger.info("Starting comprehensive website test...")
        
        test_results = {}
        
        try:
            # 1. Test basic endpoints
            logger.info("1. Testing basic endpoints...")
            test_results["basic_endpoints"] = self.test_basic_endpoints()
            
            # 2. Test ensemble configurations
            logger.info("2. Testing ensemble configurations...")
            test_results["ensemble_configs"] = self.test_ensemble_configuration()
            
            # 3. Load test samples
            logger.info("3. Loading test samples...")
            samples_metadata = self.load_test_samples()
            
            # 4. Test predictions with samples
            logger.info("4. Testing predictions with samples...")
            prediction_results = self.test_prediction_with_samples(samples_metadata)
            test_results["prediction_results"] = prediction_results
            
            # 5. Analyze results
            logger.info("5. Analyzing results...")
            test_results["prediction_analysis"] = self.analyze_results(prediction_results)
            
            # 6. Test evaluation endpoints
            logger.info("6. Testing evaluation endpoints...")
            test_results["evaluation_test"] = self.test_evaluation_endpoints()
            
            # 7. Generate report
            logger.info("7. Generating report...")
            report = self.generate_report(test_results)
            test_results["report"] = report
            
            # Save results
            with open("website_test_results.json", "w") as f:
                json.dump(test_results, f, indent=2, default=str)
            
            with open("website_test_report.txt", "w") as f:
                f.write(report)
            
            logger.info("✓ Comprehensive test completed successfully!")
            logger.info("Results saved to: website_test_results.json")
            logger.info("Report saved to: website_test_report.txt")
            
            return test_results
            
        except Exception as e:
            logger.error(f"✗ Comprehensive test failed: {e}")
            return {"error": str(e)}


def main():
    """Main test function"""
    print("🚀 Advanced Ensemble Website Tester")
    print("=" * 50)
    
    # Initialize tester
    tester = AdvancedEnsembleWebsiteTester()
    
    # Run comprehensive test
    results = tester.run_comprehensive_test()
    
    # Print summary
    if "error" not in results:
        print("\n📊 TEST SUMMARY:")
        print(f"✓ Basic endpoints tested: {len(results.get('basic_endpoints', {}))}")
        print(f"✓ Ensemble configs tested: {len(results.get('ensemble_configs', {}))}")
        print(f"✓ Prediction samples tested: {len(results.get('prediction_results', []))}")
        
        if "prediction_analysis" in results:
            analysis = results["prediction_analysis"]
            overall_accuracy = analysis.get("overall_metrics", {}).get("total_accuracy", 0)
            print(f"✓ Overall accuracy: {overall_accuracy:.3f}")
        
        print("\n📄 Check the generated files for detailed results:")
        print("  - website_test_results.json (detailed results)")
        print("  - website_test_report.txt (comprehensive report)")
    else:
        print(f"\n❌ Test failed: {results['error']}")


if __name__ == "__main__":
    main() 