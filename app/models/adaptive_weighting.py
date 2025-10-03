"""
Adaptive Weighting System for Advanced Ensemble Optimization

This module implements dynamic ensemble weighting that adjusts model weights
based on input characteristics, model confidence, and performance correlation
to achieve optimal accuracy and efficiency.
"""

import logging
import time
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
import math
from collections import defaultdict

from .base_detector import DetectionResult


class WeightingStrategy(str, Enum):
    """Strategies for calculating adaptive weights."""
    CONFIDENCE_BASED = "confidence"      # Weight based on model confidence
    CORRELATION_BASED = "correlation"    # Weight based on model agreement
    PERFORMANCE_BASED = "performance"    # Weight based on historical performance
    HYBRID = "hybrid"                    # Combination of all strategies
    UNCERTAINTY_BASED = "uncertainty"    # Weight based on prediction uncertainty


class EnsemblePruningMode(str, Enum):
    """Modes for ensemble pruning."""
    NONE = "none"                        # No pruning
    CONFIDENCE_THRESHOLD = "confidence"  # Remove low-confidence models
    CORRELATION_THRESHOLD = "correlation" # Remove highly correlated models
    PERFORMANCE_THRESHOLD = "performance" # Remove low-performing models
    ADAPTIVE = "adaptive"                # Dynamic pruning based on input


@dataclass
class WeightingContext:
    """Context information for weight calculation."""
    input_complexity: str
    model_confidences: Dict[str, float]
    model_correlations: Dict[str, Dict[str, float]]
    historical_performance: Dict[str, float]
    uncertainty_scores: Dict[str, float]
    processing_time: float
    memory_usage: float


@dataclass
class AdaptiveWeights:
    """Result of adaptive weight calculation."""
    weights: Dict[str, float]
    strategy_used: WeightingStrategy
    pruning_applied: List[str]  # Models that were pruned
    confidence_adjustment: Dict[str, float]
    rationale: Dict[str, Any]
    processing_time: float


class AdaptiveWeighting:
    """
    Dynamic ensemble weighting system that adapts model weights based on
    input characteristics, model performance, and confidence correlation.
    """
    
    def __init__(self, strategy: WeightingStrategy = WeightingStrategy.HYBRID,
                 pruning_mode: EnsemblePruningMode = EnsemblePruningMode.ADAPTIVE):
        self.strategy = strategy
        self.pruning_mode = pruning_mode
        self.logger = logging.getLogger(f"{__name__}.AdaptiveWeighting")
        
        # Historical performance tracking
        self.model_performance_history = defaultdict(list)
        self.model_correlation_history = defaultdict(lambda: defaultdict(list))
        self.complexity_performance = defaultdict(lambda: defaultdict(list))
        
        # Configuration parameters
        self.min_weight_threshold = 0.05  # Minimum weight to keep a model
        self.max_correlation_threshold = 0.95  # Max correlation before pruning
        self.confidence_boost_factor = 1.2  # Factor to boost high-confidence models
        self.uncertainty_penalty_factor = 0.8  # Factor to penalize uncertain models
        
        # Performance tracking
        self.total_weight_calculations = 0
        self.total_pruning_operations = 0
        self.weight_calculation_time = 0.0
        
    def calculate_adaptive_weights(self, model_results: Dict[str, DetectionResult],
                                 context: WeightingContext) -> AdaptiveWeights:
        """
        Calculate adaptive weights for ensemble models.
        
        Args:
            model_results: Dictionary of model results
            context: Context information for weight calculation
            
        Returns:
            AdaptiveWeights with calculated weights and rationale
        """
        start_time = time.time()
        
        try:
            # Extract model information
            model_names = list(model_results.keys())
            confidences = {name: result.confidence for name, result in model_results.items()}
            
            # Calculate base weights using selected strategy
            if self.strategy == WeightingStrategy.CONFIDENCE_BASED:
                weights = self._calculate_confidence_weights(confidences, context)
            elif self.strategy == WeightingStrategy.CORRELATION_BASED:
                weights = self._calculate_correlation_weights(model_results, context)
            elif self.strategy == WeightingStrategy.PERFORMANCE_BASED:
                weights = self._calculate_performance_weights(model_names, context)
            elif self.strategy == WeightingStrategy.UNCERTAINTY_BASED:
                weights = self._calculate_uncertainty_weights(model_results, context)
            else:  # HYBRID
                weights = self._calculate_hybrid_weights(model_results, context)
            
            # Apply ensemble pruning if enabled
            pruned_models = []
            if self.pruning_mode != EnsemblePruningMode.NONE:
                weights, pruned_models = self._apply_ensemble_pruning(weights, model_results, context)
            
            # Normalize weights
            weights = self._normalize_weights(weights)
            
            # Calculate confidence adjustments
            confidence_adjustment = self._calculate_confidence_adjustment(weights, confidences)
            
            # Generate rationale
            rationale = self._generate_weighting_rationale(weights, model_results, context, pruned_models)
            
            # Update performance tracking
            processing_time = time.time() - start_time
            self._update_performance_tracking(weights, context, processing_time)
            
            return AdaptiveWeights(
                weights=weights,
                strategy_used=self.strategy,
                pruning_applied=pruned_models,
                confidence_adjustment=confidence_adjustment,
                rationale=rationale,
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Adaptive weight calculation failed: {str(e)}")
            # Return equal weights as fallback
            equal_weight = 1.0 / len(model_results)
            return AdaptiveWeights(
                weights={name: equal_weight for name in model_results.keys()},
                strategy_used=self.strategy,
                pruning_applied=[],
                confidence_adjustment={},
                rationale={"error": str(e), "fallback": True},
                processing_time=time.time() - start_time
            )
    
    def _calculate_confidence_weights(self, confidences: Dict[str, float], 
                                    context: WeightingContext) -> Dict[str, float]:
        """Calculate weights based on model confidence scores."""
        weights = {}
        
        # Apply confidence-based weighting with boosting
        for model_name, confidence in confidences.items():
            # Normalize confidence to 0-1 range
            normalized_confidence = confidence / 100.0
            
            # Apply confidence boost for high-confidence models
            if normalized_confidence > 0.8:
                weight = normalized_confidence * self.confidence_boost_factor
            else:
                weight = normalized_confidence
            
            # Apply uncertainty penalty
            if model_name in context.uncertainty_scores:
                uncertainty = context.uncertainty_scores[model_name]
                weight *= (1.0 - uncertainty * self.uncertainty_penalty_factor)
            
            weights[model_name] = max(0.0, weight)
        
        return weights
    
    def _calculate_correlation_weights(self, model_results: Dict[str, DetectionResult],
                                     context: WeightingContext) -> Dict[str, float]:
        """Calculate weights based on model correlation and agreement."""
        weights = {}
        model_names = list(model_results.keys())
        
        # Calculate pairwise correlations
        correlations = self._calculate_model_correlations(model_results)
        
        # Weight models based on their agreement with others
        for model_name in model_names:
            # Calculate average agreement with other models
            agreements = []
            for other_model in model_names:
                if other_model != model_name:
                    # Check if predictions agree (same verdict)
                    verdict_match = (model_results[model_name].is_deepfake == 
                                   model_results[other_model].is_deepfake)
                    agreements.append(1.0 if verdict_match else 0.0)
            
            # Weight based on agreement and confidence
            agreement_score = np.mean(agreements) if agreements else 0.5
            confidence = model_results[model_name].confidence / 100.0
            
            # Combine agreement and confidence
            weight = (agreement_score * 0.6 + confidence * 0.4)
            weights[model_name] = max(0.0, weight)
        
        return weights
    
    def _calculate_performance_weights(self, model_names: List[str], 
                                     context: WeightingContext) -> Dict[str, float]:
        """Calculate weights based on historical performance."""
        weights = {}
        
        for model_name in model_names:
            # Get historical performance for this complexity level
            complexity = context.input_complexity
            if model_name in self.complexity_performance[complexity]:
                perf_scores = self.complexity_performance[complexity][model_name]
                avg_performance = np.mean(perf_scores) if perf_scores else 0.5
            else:
                # Use general historical performance
                if model_name in self.model_performance_history:
                    avg_performance = np.mean(self.model_performance_history[model_name])
                else:
                    avg_performance = 0.5  # Default performance
            
            weights[model_name] = max(0.0, avg_performance)
        
        return weights
    
    def _calculate_uncertainty_weights(self, model_results: Dict[str, DetectionResult],
                                     context: WeightingContext) -> Dict[str, float]:
        """Calculate weights based on prediction uncertainty."""
        weights = {}
        
        for model_name, result in model_results.items():
            # Base weight from confidence
            base_weight = result.confidence / 100.0
            
            # Apply uncertainty penalty
            if model_name in context.uncertainty_scores:
                uncertainty = context.uncertainty_scores[model_name]
                uncertainty_penalty = uncertainty * self.uncertainty_penalty_factor
                weight = base_weight * (1.0 - uncertainty_penalty)
            else:
                weight = base_weight
            
            # Apply processing time penalty (faster models get slight boost)
            if hasattr(result, 'processing_time'):
                time_penalty = min(0.1, result.processing_time / 1000.0)  # Max 10% penalty
                weight *= (1.0 - time_penalty)
            
            weights[model_name] = max(0.0, weight)
        
        return weights
    
    def _calculate_hybrid_weights(self, model_results: Dict[str, DetectionResult],
                                context: WeightingContext) -> Dict[str, float]:
        """Calculate weights using hybrid approach combining all strategies."""
        # Get weights from each strategy
        conf_weights = self._calculate_confidence_weights(
            {name: result.confidence for name, result in model_results.items()}, context
        )
        corr_weights = self._calculate_correlation_weights(model_results, context)
        perf_weights = self._calculate_performance_weights(list(model_results.keys()), context)
        unc_weights = self._calculate_uncertainty_weights(model_results, context)
        
        # Combine weights with different importance based on context
        weights = {}
        for model_name in model_results.keys():
            # Weight the strategies based on input complexity
            if context.input_complexity == "simple":
                # For simple inputs, prioritize confidence and performance
                weight = (conf_weights[model_name] * 0.4 + 
                         perf_weights[model_name] * 0.3 + 
                         corr_weights[model_name] * 0.2 + 
                         unc_weights[model_name] * 0.1)
            elif context.input_complexity == "complex":
                # For complex inputs, prioritize correlation and uncertainty
                weight = (corr_weights[model_name] * 0.4 + 
                         unc_weights[model_name] * 0.3 + 
                         conf_weights[model_name] * 0.2 + 
                         perf_weights[model_name] * 0.1)
            else:  # medium
                # For medium inputs, balanced approach
                weight = (conf_weights[model_name] * 0.3 + 
                         corr_weights[model_name] * 0.3 + 
                         perf_weights[model_name] * 0.2 + 
                         unc_weights[model_name] * 0.2)
            
            weights[model_name] = max(0.0, weight)
        
        return weights
    
    def _calculate_model_correlations(self, model_results: Dict[str, DetectionResult]) -> Dict[str, Dict[str, float]]:
        """Calculate correlations between model predictions."""
        correlations = {}
        model_names = list(model_results.keys())
        
        for i, model1 in enumerate(model_names):
            correlations[model1] = {}
            for j, model2 in enumerate(model_names):
                if i == j:
                    correlations[model1][model2] = 1.0
                else:
                    # Calculate correlation based on verdict agreement and confidence similarity
                    verdict_match = (model_results[model1].is_deepfake == model_results[model2].is_deepfake)
                    conf_diff = abs(model_results[model1].confidence - model_results[model2].confidence)
                    conf_similarity = 1.0 - (conf_diff / 100.0)
                    
                    # Combine verdict agreement and confidence similarity
                    correlation = (verdict_match * 0.7 + conf_similarity * 0.3)
                    correlations[model1][model2] = max(0.0, correlation)
        
        return correlations
    
    def _apply_ensemble_pruning(self, weights: Dict[str, float], 
                               model_results: Dict[str, DetectionResult],
                               context: WeightingContext) -> Tuple[Dict[str, float], List[str]]:
        """Apply ensemble pruning to remove low-contribution models."""
        pruned_models = []
        
        if self.pruning_mode == EnsemblePruningMode.CONFIDENCE_THRESHOLD:
            # Remove models with low confidence
            for model_name, weight in list(weights.items()):
                if weight < self.min_weight_threshold:
                    del weights[model_name]
                    pruned_models.append(model_name)
        
        elif self.pruning_mode == EnsemblePruningMode.CORRELATION_THRESHOLD:
            # Remove highly correlated models (keep the one with higher weight)
            correlations = self._calculate_model_correlations(model_results)
            models_to_remove = set()
            
            for model1 in list(weights.keys()):
                if model1 in models_to_remove:
                    continue
                for model2 in list(weights.keys()):
                    if model2 in models_to_remove or model1 == model2:
                        continue
                    
                    if correlations[model1][model2] > self.max_correlation_threshold:
                        # Remove the model with lower weight
                        if weights[model1] < weights[model2]:
                            models_to_remove.add(model1)
                        else:
                            models_to_remove.add(model2)
            
            for model_name in models_to_remove:
                del weights[model_name]
                pruned_models.append(model_name)
        
        elif self.pruning_mode == EnsemblePruningMode.ADAPTIVE:
            # Adaptive pruning based on multiple factors
            for model_name, weight in list(weights.items()):
                should_prune = False
                
                # Prune if weight is too low
                if weight < self.min_weight_threshold:
                    should_prune = True
                
                # Prune if confidence is too low
                if model_name in context.model_confidences:
                    if context.model_confidences[model_name] < 30.0:  # 30% confidence threshold
                        should_prune = True
                
                # Prune if uncertainty is too high
                if model_name in context.uncertainty_scores:
                    if context.uncertainty_scores[model_name] > 0.7:  # 70% uncertainty threshold
                        should_prune = True
                
                if should_prune:
                    del weights[model_name]
                    pruned_models.append(model_name)
        
        self.total_pruning_operations += len(pruned_models)
        return weights, pruned_models
    
    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Normalize weights to sum to 1.0."""
        total_weight = sum(weights.values())
        if total_weight > 0:
            return {name: weight / total_weight for name, weight in weights.items()}
        else:
            # Fallback to equal weights
            equal_weight = 1.0 / len(weights)
            return {name: equal_weight for name in weights.keys()}
    
    def _calculate_confidence_adjustment(self, weights: Dict[str, float], 
                                       confidences: Dict[str, float]) -> Dict[str, float]:
        """Calculate confidence adjustments for each model."""
        adjustments = {}
        
        for model_name, weight in weights.items():
            if model_name in confidences:
                confidence = confidences[model_name]
                # Calculate adjustment factor based on confidence
                if confidence > 80:
                    adjustment = 1.1  # 10% boost for high confidence
                elif confidence > 60:
                    adjustment = 1.0  # No adjustment
                else:
                    adjustment = 0.9  # 10% penalty for low confidence
                
                adjustments[model_name] = adjustment
            else:
                adjustments[model_name] = 1.0
        
        return adjustments
    
    def _generate_weighting_rationale(self, weights: Dict[str, float], 
                                    model_results: Dict[str, DetectionResult],
                                    context: WeightingContext, 
                                    pruned_models: List[str]) -> Dict[str, Any]:
        """Generate detailed rationale for weight calculation."""
        rationale = {
            "strategy_used": self.strategy.value,
            "pruning_mode": self.pruning_mode.value,
            "pruned_models": pruned_models,
            "weight_distribution": weights,
            "confidence_scores": {name: result.confidence for name, result in model_results.items()},
            "input_complexity": context.input_complexity,
            "total_models": len(model_results),
            "active_models": len(weights),
            "pruning_ratio": len(pruned_models) / len(model_results) if model_results else 0.0
        }
        
        # Add strategy-specific rationale
        if self.strategy == WeightingStrategy.HYBRID:
            rationale["hybrid_weights"] = {
                "confidence_weight": 0.3 if context.input_complexity == "medium" else 0.4,
                "correlation_weight": 0.3 if context.input_complexity == "medium" else 0.2,
                "performance_weight": 0.2 if context.input_complexity == "medium" else 0.3,
                "uncertainty_weight": 0.2 if context.input_complexity == "medium" else 0.1
            }
        
        return rationale
    
    def _update_performance_tracking(self, weights: Dict[str, float], 
                                   context: WeightingContext, processing_time: float):
        """Update performance tracking metrics."""
        self.total_weight_calculations += 1
        self.weight_calculation_time += processing_time
        
        # Track model performance by complexity
        for model_name, weight in weights.items():
            self.complexity_performance[context.input_complexity][model_name].append(weight)
            
            # Keep only recent history (last 100 entries)
            if len(self.complexity_performance[context.input_complexity][model_name]) > 100:
                self.complexity_performance[context.input_complexity][model_name] = \
                    self.complexity_performance[context.input_complexity][model_name][-100:]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the adaptive weighting system."""
        avg_calculation_time = (self.weight_calculation_time / self.total_weight_calculations 
                              if self.total_weight_calculations > 0 else 0.0)
        
        return {
            "total_calculations": self.total_weight_calculations,
            "total_pruning_operations": self.total_pruning_operations,
            "average_calculation_time": avg_calculation_time,
            "strategy": self.strategy.value,
            "pruning_mode": self.pruning_mode.value,
            "complexity_performance": {
                complexity: {
                    model: {
                        "count": len(scores),
                        "avg_weight": np.mean(scores) if scores else 0.0,
                        "std_weight": np.std(scores) if scores else 0.0
                    } for model, scores in models.items()
                } for complexity, models in self.complexity_performance.items()
            }
        }
    
    def update_strategy(self, strategy: WeightingStrategy):
        """Update the weighting strategy."""
        self.strategy = strategy
        self.logger.info(f"Updated weighting strategy to: {strategy.value}")
    
    def update_pruning_mode(self, pruning_mode: EnsemblePruningMode):
        """Update the ensemble pruning mode."""
        self.pruning_mode = pruning_mode
        self.logger.info(f"Updated pruning mode to: {pruning_mode.value}")
    
    def reset_performance_tracking(self):
        """Reset performance tracking data."""
        self.model_performance_history.clear()
        self.model_correlation_history.clear()
        self.complexity_performance.clear()
        self.total_weight_calculations = 0
        self.total_pruning_operations = 0
        self.weight_calculation_time = 0.0
        self.logger.info("Performance tracking reset")
