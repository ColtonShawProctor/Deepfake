"""
Advanced Ensemble Evaluation Framework

This module provides comprehensive evaluation capabilities for the advanced ensemble system,
including proper evaluation metrics, benchmarking, and performance analysis.
"""

import logging
import time
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve,
    calibration_curve, brier_score_loss
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
import torch

from .advanced_ensemble import AdvancedEnsembleManager, AdvancedEnsembleConfig, AdvancedEnsembleResult
from .base_detector import BaseDetector, DetectionResult


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics for ensemble performance."""
    # Basic classification metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    auc_pr: float
    
    # Calibration metrics
    brier_score: float
    calibration_error: float
    
    # Uncertainty metrics
    mean_uncertainty: float
    uncertainty_correlation: float
    
    # Ensemble-specific metrics
    agreement_score: float
    confidence_variance: float
    ensemble_diversity: float
    
    # Performance metrics
    inference_time: float
    throughput_fps: float
    
    # Additional metadata
    num_samples: int
    num_models: int
    fusion_method: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Result of ensemble benchmarking."""
    ensemble_name: str
    dataset_name: str
    metrics: EvaluationMetrics
    predictions: List[bool]
    confidences: List[float]
    uncertainties: List[float]
    ground_truths: List[bool]
    individual_predictions: List[Dict[str, DetectionResult]]
    timestamp: float = field(default_factory=time.time)


class AdvancedEnsembleEvaluator:
    """
    Comprehensive evaluator for advanced ensemble systems.
    
    Provides benchmarking, performance analysis, and detailed evaluation metrics
    for ensemble deepfake detection systems.
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize the evaluator.
        
        Args:
            output_dir: Directory to save evaluation results and plots
        """
        self.output_dir = Path(output_dir) if output_dir else Path("evaluation_results")
        self.output_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(f"{__name__}.AdvancedEnsembleEvaluator")
        self.benchmark_results: List[BenchmarkResult] = []
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def evaluate_ensemble(self, 
                         ensemble: AdvancedEnsembleManager,
                         test_data: List[Tuple[Any, bool]],
                         dataset_name: str = "test_dataset",
                         ensemble_name: str = "advanced_ensemble") -> BenchmarkResult:
        """
        Evaluate ensemble performance on test data.
        
        Args:
            ensemble: Advanced ensemble manager to evaluate
            test_data: List of (image, ground_truth) tuples
            dataset_name: Name of the test dataset
            ensemble_name: Name of the ensemble being evaluated
            
        Returns:
            BenchmarkResult containing comprehensive evaluation metrics
        """
        self.logger.info(f"Starting evaluation of {ensemble_name} on {dataset_name}")
        
        predictions = []
        confidences = []
        uncertainties = []
        ground_truths = []
        individual_predictions = []
        inference_times = []
        
        # Run predictions
        for i, (image, ground_truth) in enumerate(test_data):
            try:
                start_time = time.time()
                result = ensemble.predict_advanced(image)
                inference_time = time.time() - start_time
                
                predictions.append(result.is_deepfake)
                confidences.append(result.confidence)
                uncertainties.append(result.uncertainty)
                ground_truths.append(ground_truth)
                individual_predictions.append(result.individual_predictions)
                inference_times.append(inference_time)
                
                if (i + 1) % 100 == 0:
                    self.logger.info(f"Processed {i + 1}/{len(test_data)} samples")
                    
            except Exception as e:
                self.logger.warning(f"Failed to process sample {i}: {str(e)}")
                continue
        
        if not predictions:
            raise ValueError("No valid predictions generated")
        
        # Calculate comprehensive metrics
        metrics = self._calculate_comprehensive_metrics(
            predictions, confidences, uncertainties, ground_truths,
            individual_predictions, inference_times, ensemble
        )
        
        # Create benchmark result
        benchmark_result = BenchmarkResult(
            ensemble_name=ensemble_name,
            dataset_name=dataset_name,
            metrics=metrics,
            predictions=predictions,
            confidences=confidences,
            uncertainties=uncertainties,
            ground_truths=ground_truths,
            individual_predictions=individual_predictions
        )
        
        self.benchmark_results.append(benchmark_result)
        
        self.logger.info(f"Evaluation completed. Accuracy: {metrics.accuracy:.4f}, AUC: {metrics.auc_roc:.4f}")
        
        return benchmark_result
    
    def _calculate_comprehensive_metrics(self,
                                       predictions: List[bool],
                                       confidences: List[float],
                                       uncertainties: List[float],
                                       ground_truths: List[bool],
                                       individual_predictions: List[Dict[str, DetectionResult]],
                                       inference_times: List[float],
                                       ensemble: AdvancedEnsembleManager) -> EvaluationMetrics:
        """Calculate comprehensive evaluation metrics."""
        
        # Basic classification metrics
        accuracy = accuracy_score(ground_truths, predictions)
        precision = precision_score(ground_truths, predictions, zero_division=0)
        recall = recall_score(ground_truths, predictions, zero_division=0)
        f1 = f1_score(ground_truths, predictions, zero_division=0)
        
        # AUC metrics
        try:
            auc_roc = roc_auc_score(ground_truths, confidences)
        except ValueError:
            auc_roc = 0.5  # Fallback for edge cases
        
        try:
            auc_pr = average_precision_score(ground_truths, confidences)
        except ValueError:
            auc_pr = 0.5  # Fallback for edge cases
        
        # Calibration metrics
        brier_score = brier_score_loss(ground_truths, confidences)
        
        # Calculate calibration error
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                ground_truths, confidences, n_bins=10
            )
            calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
        except:
            calibration_error = 0.0
        
        # Uncertainty metrics
        mean_uncertainty = np.mean(uncertainties)
        
        # Calculate correlation between uncertainty and prediction error
        prediction_errors = [abs(pred - gt) for pred, gt in zip(predictions, ground_truths)]
        try:
            uncertainty_correlation = np.corrcoef(uncertainties, prediction_errors)[0, 1]
            if np.isnan(uncertainty_correlation):
                uncertainty_correlation = 0.0
        except:
            uncertainty_correlation = 0.0
        
        # Ensemble-specific metrics
        agreement_scores = []
        confidence_variances = []
        
        for ind_pred in individual_predictions:
            if len(ind_pred) > 1:
                # Calculate agreement score
                binary_preds = [p.is_deepfake for p in ind_pred.values()]
                agreement = sum(binary_preds) / len(binary_preds)
                agreement_scores.append(1 - abs(agreement - 0.5) * 2)  # Convert to [0, 1]
                
                # Calculate confidence variance
                confs = [p.confidence for p in ind_pred.values()]
                confidence_variances.append(np.var(confs))
        
        agreement_score = np.mean(agreement_scores) if agreement_scores else 1.0
        confidence_variance = np.mean(confidence_variances) if confidence_variances else 0.0
        
        # Calculate ensemble diversity (disagreement among models)
        ensemble_diversity = 1.0 - agreement_score
        
        # Performance metrics
        mean_inference_time = np.mean(inference_times)
        throughput_fps = 1.0 / mean_inference_time if mean_inference_time > 0 else 0
        
        # Get ensemble info
        ensemble_info = ensemble.get_ensemble_info()
        
        return EvaluationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc_roc=auc_roc,
            auc_pr=auc_pr,
            brier_score=brier_score,
            calibration_error=calibration_error,
            mean_uncertainty=mean_uncertainty,
            uncertainty_correlation=uncertainty_correlation,
            agreement_score=agreement_score,
            confidence_variance=confidence_variance,
            ensemble_diversity=ensemble_diversity,
            inference_time=mean_inference_time,
            throughput_fps=throughput_fps,
            num_samples=len(predictions),
            num_models=ensemble_info.get("num_models", 0),
            fusion_method=ensemble_info.get("fusion_method", "unknown"),
            metadata={
                "ensemble_info": ensemble_info,
                "inference_times": inference_times
            }
        )
    
    def compare_ensembles(self, 
                         ensemble_results: List[BenchmarkResult],
                         comparison_name: str = "ensemble_comparison") -> Dict[str, Any]:
        """
        Compare multiple ensemble configurations.
        
        Args:
            ensemble_results: List of benchmark results to compare
            comparison_name: Name for the comparison
            
        Returns:
            Dictionary containing comparison metrics and analysis
        """
        self.logger.info(f"Comparing {len(ensemble_results)} ensemble configurations")
        
        # Create comparison dataframe
        comparison_data = []
        for result in ensemble_results:
            comparison_data.append({
                "ensemble_name": result.ensemble_name,
                "dataset_name": result.dataset_name,
                "accuracy": result.metrics.accuracy,
                "precision": result.metrics.precision,
                "recall": result.metrics.recall,
                "f1_score": result.metrics.f1_score,
                "auc_roc": result.metrics.auc_roc,
                "auc_pr": result.metrics.auc_pr,
                "brier_score": result.metrics.brier_score,
                "calibration_error": result.metrics.calibration_error,
                "mean_uncertainty": result.metrics.mean_uncertainty,
                "uncertainty_correlation": result.metrics.uncertainty_correlation,
                "agreement_score": result.metrics.agreement_score,
                "ensemble_diversity": result.metrics.ensemble_diversity,
                "inference_time": result.metrics.inference_time,
                "throughput_fps": result.metrics.throughput_fps,
                "num_models": result.metrics.num_models,
                "fusion_method": result.metrics.fusion_method
            })
        
        df = pd.DataFrame(comparison_data)
        
        # Generate comparison plots
        self._generate_comparison_plots(df, comparison_name)
        
        # Statistical analysis
        statistical_analysis = self._perform_statistical_analysis(df)
        
        # Ranking analysis
        ranking_analysis = self._perform_ranking_analysis(df)
        
        comparison_result = {
            "comparison_name": comparison_name,
            "dataframe": df,
            "statistical_analysis": statistical_analysis,
            "ranking_analysis": ranking_analysis,
            "best_ensemble": self._identify_best_ensemble(df),
            "timestamp": time.time()
        }
        
        # Save comparison results
        self._save_comparison_results(comparison_result, comparison_name)
        
        return comparison_result
    
    def _generate_comparison_plots(self, df: pd.DataFrame, comparison_name: str):
        """Generate comprehensive comparison plots."""
        
        # Set up the plotting grid
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle(f'Ensemble Comparison: {comparison_name}', fontsize=16, fontweight='bold')
        
        # 1. Classification metrics comparison
        metrics_cols = ['accuracy', 'precision', 'recall', 'f1_score']
        df[metrics_cols].plot(kind='bar', ax=axes[0, 0], title='Classification Metrics')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. AUC comparison
        auc_cols = ['auc_roc', 'auc_pr']
        df[auc_cols].plot(kind='bar', ax=axes[0, 1], title='AUC Metrics')
        axes[0, 1].set_ylabel('AUC Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Calibration metrics
        calib_cols = ['brier_score', 'calibration_error']
        df[calib_cols].plot(kind='bar', ax=axes[0, 2], title='Calibration Metrics')
        axes[0, 2].set_ylabel('Score')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. Uncertainty analysis
        uncertainty_cols = ['mean_uncertainty', 'uncertainty_correlation']
        df[uncertainty_cols].plot(kind='bar', ax=axes[1, 0], title='Uncertainty Metrics')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 5. Ensemble characteristics
        ensemble_cols = ['agreement_score', 'ensemble_diversity']
        df[ensemble_cols].plot(kind='bar', ax=axes[1, 1], title='Ensemble Characteristics')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 6. Performance metrics
        perf_cols = ['inference_time', 'throughput_fps']
        df[perf_cols].plot(kind='bar', ax=axes[1, 2], title='Performance Metrics')
        axes[1, 2].set_ylabel('Time/FPS')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        # 7. Model count vs performance
        axes[2, 0].scatter(df['num_models'], df['accuracy'], s=100, alpha=0.7)
        axes[2, 0].set_xlabel('Number of Models')
        axes[2, 0].set_ylabel('Accuracy')
        axes[2, 0].set_title('Model Count vs Accuracy')
        
        # 8. Fusion method analysis
        fusion_methods = df['fusion_method'].value_counts()
        fusion_methods.plot(kind='pie', ax=axes[2, 1], title='Fusion Methods Distribution')
        
        # 9. Correlation heatmap
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   ax=axes[2, 2], title='Metrics Correlation')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{comparison_name}_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _perform_statistical_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform statistical analysis on ensemble comparison."""
        
        # Calculate descriptive statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        descriptive_stats = df[numeric_cols].describe()
        
        # Calculate coefficient of variation (CV) for each metric
        cv_stats = {}
        for col in numeric_cols:
            mean_val = df[col].mean()
            std_val = df[col].std()
            cv_stats[col] = (std_val / mean_val) if mean_val != 0 else 0
        
        # Perform pairwise comparisons (if multiple ensembles)
        pairwise_comparisons = {}
        if len(df) > 1:
            for metric in ['accuracy', 'auc_roc', 'f1_score']:
                if metric in df.columns:
                    # Simple pairwise comparison using t-test
                    from scipy.stats import ttest_ind
                    ensemble_names = df['ensemble_name'].unique()
                    if len(ensemble_names) >= 2:
                        ensemble1_data = df[df['ensemble_name'] == ensemble_names[0]][metric]
                        ensemble2_data = df[df['ensemble_name'] == ensemble_names[1]][metric]
                        if len(ensemble1_data) > 0 and len(ensemble2_data) > 0:
                            t_stat, p_value = ttest_ind(ensemble1_data, ensemble2_data)
                            pairwise_comparisons[metric] = {
                                't_statistic': t_stat,
                                'p_value': p_value,
                                'significant': p_value < 0.05
                            }
        
        return {
            "descriptive_statistics": descriptive_stats.to_dict(),
            "coefficient_of_variation": cv_stats,
            "pairwise_comparisons": pairwise_comparisons
        }
    
    def _perform_ranking_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform ranking analysis of ensembles."""
        
        # Define metrics and their optimal direction (higher is better for most)
        ranking_metrics = {
            'accuracy': 'higher',
            'precision': 'higher',
            'recall': 'higher',
            'f1_score': 'higher',
            'auc_roc': 'higher',
            'auc_pr': 'higher',
            'uncertainty_correlation': 'higher',
            'agreement_score': 'higher',
            'throughput_fps': 'higher',
            'brier_score': 'lower',  # Lower is better
            'calibration_error': 'lower',  # Lower is better
            'mean_uncertainty': 'lower',  # Lower is better
            'inference_time': 'lower'  # Lower is better
        }
        
        # Calculate rankings for each metric
        rankings = {}
        for metric, direction in ranking_metrics.items():
            if metric in df.columns:
                if direction == 'higher':
                    rankings[metric] = df[metric].rank(ascending=False)
                else:
                    rankings[metric] = df[metric].rank(ascending=True)
        
        # Calculate average ranking
        ranking_df = pd.DataFrame(rankings)
        avg_rankings = ranking_df.mean(axis=1)
        
        # Create final ranking
        final_ranking = pd.DataFrame({
            'ensemble_name': df['ensemble_name'],
            'average_rank': avg_rankings,
            'overall_rank': avg_rankings.rank()
        }).sort_values('overall_rank')
        
        return {
            "individual_rankings": rankings,
            "average_rankings": avg_rankings.to_dict(),
            "final_ranking": final_ranking.to_dict('records')
        }
    
    def _identify_best_ensemble(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Identify the best ensemble based on multiple criteria."""
        
        # Weighted scoring system
        weights = {
            'accuracy': 0.25,
            'f1_score': 0.20,
            'auc_roc': 0.20,
            'precision': 0.15,
            'recall': 0.10,
            'throughput_fps': 0.10
        }
        
        # Calculate weighted score
        weighted_scores = {}
        for ensemble_name in df['ensemble_name'].unique():
            ensemble_data = df[df['ensemble_name'] == ensemble_name].iloc[0]
            score = 0
            for metric, weight in weights.items():
                if metric in ensemble_data:
                    score += ensemble_data[metric] * weight
            weighted_scores[ensemble_name] = score
        
        best_ensemble = max(weighted_scores, key=weighted_scores.get)
        best_data = df[df['ensemble_name'] == best_ensemble].iloc[0]
        
        return {
            "best_ensemble": best_ensemble,
            "weighted_score": weighted_scores[best_ensemble],
            "all_scores": weighted_scores,
            "best_metrics": {
                "accuracy": best_data.get('accuracy', 0),
                "f1_score": best_data.get('f1_score', 0),
                "auc_roc": best_data.get('auc_roc', 0),
                "fusion_method": best_data.get('fusion_method', 'unknown')
            }
        }
    
    def _save_comparison_results(self, comparison_result: Dict[str, Any], comparison_name: str):
        """Save comparison results to files."""
        
        # Save detailed results as JSON
        json_path = self.output_dir / f'{comparison_name}_results.json'
        with open(json_path, 'w') as f:
            # Convert DataFrame to dict for JSON serialization
            result_copy = comparison_result.copy()
            result_copy['dataframe'] = result_copy['dataframe'].to_dict('records')
            json.dump(result_copy, f, indent=2, default=str)
        
        # Save summary as CSV
        csv_path = self.output_dir / f'{comparison_name}_summary.csv'
        comparison_result['dataframe'].to_csv(csv_path, index=False)
        
        self.logger.info(f"Comparison results saved to {json_path} and {csv_path}")
    
    def generate_evaluation_report(self, 
                                 benchmark_results: List[BenchmarkResult],
                                 report_name: str = "evaluation_report") -> str:
        """
        Generate comprehensive evaluation report.
        
        Args:
            benchmark_results: List of benchmark results to include in report
            report_name: Name for the report
            
        Returns:
            Generated report as string
        """
        
        report_lines = [
            "=" * 80,
            "ADVANCED ENSEMBLE EVALUATION REPORT",
            "=" * 80,
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Number of evaluations: {len(benchmark_results)}",
            ""
        ]
        
        # Summary statistics
        report_lines.extend([
            "SUMMARY STATISTICS",
            "-" * 40
        ])
        
        for result in benchmark_results:
            report_lines.extend([
                f"Ensemble: {result.ensemble_name}",
                f"Dataset: {result.dataset_name}",
                f"Accuracy: {result.metrics.accuracy:.4f}",
                f"F1-Score: {result.metrics.f1_score:.4f}",
                f"AUC-ROC: {result.metrics.auc_roc:.4f}",
                f"Precision: {result.metrics.precision:.4f}",
                f"Recall: {result.metrics.recall:.4f}",
                f"Mean Uncertainty: {result.metrics.mean_uncertainty:.4f}",
                f"Agreement Score: {result.metrics.agreement_score:.4f}",
                f"Inference Time: {result.metrics.inference_time:.4f}s",
                f"Throughput: {result.metrics.throughput_fps:.2f} FPS",
                ""
            ])
        
        # Detailed analysis
        if len(benchmark_results) > 1:
            report_lines.extend([
                "COMPARATIVE ANALYSIS",
                "-" * 40
            ])
            
            # Find best performing ensemble
            best_result = max(benchmark_results, key=lambda x: x.metrics.accuracy)
            report_lines.extend([
                f"Best Accuracy: {best_result.ensemble_name} ({best_result.metrics.accuracy:.4f})",
                f"Best F1-Score: {max(benchmark_results, key=lambda x: x.metrics.f1_score).ensemble_name}",
                f"Best AUC-ROC: {max(benchmark_results, key=lambda x: x.metrics.auc_roc).ensemble_name}",
                f"Fastest Inference: {min(benchmark_results, key=lambda x: x.metrics.inference_time).ensemble_name}",
                ""
            ])
        
        # Recommendations
        report_lines.extend([
            "RECOMMENDATIONS",
            "-" * 40
        ])
        
        for result in benchmark_results:
            recommendations = []
            
            if result.metrics.calibration_error > 0.1:
                recommendations.append("Consider recalibrating confidence scores")
            
            if result.metrics.uncertainty_correlation < 0.3:
                recommendations.append("Uncertainty quantification may need improvement")
            
            if result.metrics.agreement_score < 0.7:
                recommendations.append("Model agreement is low - consider ensemble diversity")
            
            if result.metrics.throughput_fps < 10:
                recommendations.append("Consider optimization for real-time applications")
            
            if recommendations:
                report_lines.extend([
                    f"{result.ensemble_name}:",
                    *[f"  - {rec}" for rec in recommendations],
                    ""
                ])
        
        report = "\n".join(report_lines)
        
        # Save report
        report_path = self.output_dir / f'{report_name}.txt'
        with open(report_path, 'w') as f:
            f.write(report)
        
        self.logger.info(f"Evaluation report saved to {report_path}")
        
        return report
    
    def create_benchmark_suite(self, 
                              ensembles: Dict[str, AdvancedEnsembleManager],
                              test_datasets: Dict[str, List[Tuple[Any, bool]]],
                              suite_name: str = "benchmark_suite") -> Dict[str, Any]:
        """
        Create and run a comprehensive benchmark suite.
        
        Args:
            ensembles: Dictionary of ensemble managers to benchmark
            test_datasets: Dictionary of test datasets
            suite_name: Name for the benchmark suite
            
        Returns:
            Comprehensive benchmark results
        """
        
        self.logger.info(f"Starting benchmark suite: {suite_name}")
        
        benchmark_results = []
        
        # Run all ensemble-dataset combinations
        for ensemble_name, ensemble in ensembles.items():
            for dataset_name, test_data in test_datasets.items():
                try:
                    result = self.evaluate_ensemble(
                        ensemble, test_data, dataset_name, ensemble_name
                    )
                    benchmark_results.append(result)
                    
                except Exception as e:
                    self.logger.error(f"Failed to evaluate {ensemble_name} on {dataset_name}: {str(e)}")
                    continue
        
        # Generate comprehensive comparison
        comparison_result = self.compare_ensembles(benchmark_results, suite_name)
        
        # Generate evaluation report
        report = self.generate_evaluation_report(benchmark_results, suite_name)
        
        # Create suite summary
        suite_summary = {
            "suite_name": suite_name,
            "num_ensembles": len(ensembles),
            "num_datasets": len(test_datasets),
            "total_evaluations": len(benchmark_results),
            "comparison_result": comparison_result,
            "report": report,
            "timestamp": time.time()
        }
        
        # Save suite results
        suite_path = self.output_dir / f'{suite_name}_suite.json'
        with open(suite_path, 'w') as f:
            json.dump(suite_summary, f, indent=2, default=str)
        
        self.logger.info(f"Benchmark suite completed. Results saved to {suite_path}")
        
        return suite_summary


def create_evaluation_example():
    """Create an example evaluation workflow."""
    
    # This would be used in practice to demonstrate the evaluation framework
    # For now, we'll create a placeholder that shows the structure
    
    evaluator = AdvancedEnsembleEvaluator(output_dir="evaluation_example")
    
    # Example usage structure:
    """
    # 1. Create different ensemble configurations
    configs = {
        "attention_ensemble": AdvancedEnsembleConfig(
            fusion_method=AdvancedFusionMethod.ATTENTION_MERGE
        ),
        "temperature_ensemble": AdvancedEnsembleConfig(
            fusion_method=AdvancedFusionMethod.TEMPERATURE_SCALED
        ),
        "mc_dropout_ensemble": AdvancedEnsembleConfig(
            fusion_method=AdvancedFusionMethod.MONTE_CARLO_DROPOUT
        )
    }
    
    # 2. Create ensembles
    ensembles = {}
    for name, config in configs.items():
        ensemble = AdvancedEnsembleManager(config)
        # Add models to ensemble
        ensembles[name] = ensemble
    
    # 3. Run benchmark suite
    test_datasets = {
        "dataset1": [(image1, label1), (image2, label2), ...],
        "dataset2": [(image3, label3), (image4, label4), ...]
    }
    
    suite_results = evaluator.create_benchmark_suite(ensembles, test_datasets)
    """
    
    return evaluator 