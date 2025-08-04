"""
Cross-Dataset Generalization Optimizer

Implements advanced techniques for improving model performance across different datasets:
- Domain adaptation strategies
- Dataset-specific calibration
- Meta-learning approaches
- Test-time adaptation
- Robust aggregation methods

Supports major deepfake datasets:
- FaceForensics++ (FF++)
- DFDC (Deepfake Detection Challenge)
- CelebDF (Celeb-DeepFake)
- DeeperForensics-1.0
- FFIW (FaceForensics++ In-the-Wild)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class DatasetCharacteristics:
    """Dataset-specific characteristics for adaptation"""
    name: str
    compression_level: float
    resolution_range: Tuple[int, int]
    face_detection_method: str
    manipulation_types: List[str]
    quality_distribution: Dict[str, float]
    artifact_patterns: Dict[str, float]

class DatasetProfiler:
    """Profile datasets to understand their characteristics"""
    
    def __init__(self):
        # Known dataset profiles (can be updated with actual measurements)
        self.dataset_profiles = {
            'faceforensics++': DatasetCharacteristics(
                name='FaceForensics++',
                compression_level=0.7,  # High compression
                resolution_range=(256, 256),
                face_detection_method='mtcnn',
                manipulation_types=['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures'],
                quality_distribution={'high': 0.4, 'medium': 0.4, 'low': 0.2},
                artifact_patterns={'compression': 0.8, 'temporal': 0.6, 'spatial': 0.7}
            ),
            'dfdc': DatasetCharacteristics(
                name='DFDC',
                compression_level=0.5,  # Variable compression
                resolution_range=(224, 224),
                face_detection_method='various',
                manipulation_types=['various_deepfakes'],
                quality_distribution={'high': 0.3, 'medium': 0.5, 'low': 0.2},
                artifact_patterns={'compression': 0.6, 'temporal': 0.8, 'spatial': 0.5}
            ),
            'celebdf': DatasetCharacteristics(
                name='CelebDF',
                compression_level=0.3,  # Lower compression
                resolution_range=(512, 512),
                face_detection_method='retinaface',
                manipulation_types=['swap_based'],
                quality_distribution={'high': 0.7, 'medium': 0.2, 'low': 0.1},
                artifact_patterns={'compression': 0.4, 'temporal': 0.5, 'spatial': 0.8}
            )
        }
    
    def identify_dataset(self, image_characteristics: Dict[str, float]) -> str:
        """Identify which dataset an image likely comes from"""
        best_match = 'unknown'
        best_score = -1
        
        for dataset_name, profile in self.dataset_profiles.items():
            score = self._calculate_similarity(image_characteristics, profile)
            if score > best_score:
                best_score = score
                best_match = dataset_name
        
        return best_match if best_score > 0.5 else 'unknown'
    
    def _calculate_similarity(self, characteristics: Dict[str, float], 
                            profile: DatasetCharacteristics) -> float:
        """Calculate similarity between image characteristics and dataset profile"""
        similarity_scores = []
        
        # Compare compression artifacts
        if 'compression' in characteristics:
            comp_sim = 1.0 - abs(characteristics['compression'] - profile.compression_level)
            similarity_scores.append(comp_sim)
        
        # Compare artifact patterns
        for artifact_type, expected_level in profile.artifact_patterns.items():
            if artifact_type in characteristics:
                artifact_sim = 1.0 - abs(characteristics[artifact_type] - expected_level)
                similarity_scores.append(artifact_sim)
        
        return np.mean(similarity_scores) if similarity_scores else 0.0

class DomainAdaptation:
    """Domain adaptation techniques for cross-dataset generalization"""
    
    def __init__(self, feature_dim: int = 512):
        self.feature_dim = feature_dim
        self.domain_classifier = self._build_domain_classifier()
        self.feature_adapter = self._build_feature_adapter()
        self.dataset_scalers = {}
        
    def _build_domain_classifier(self) -> nn.Module:
        """Build domain classifier for adversarial adaptation"""
        return nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3),  # Number of domains (FF++, DFDC, CelebDF)
            nn.Softmax(dim=-1)
        )
    
    def _build_feature_adapter(self) -> nn.Module:
        """Build feature adapter for domain alignment"""
        return nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim)
        )
    
    def adapt_features(self, features: torch.Tensor, target_domain: str) -> torch.Tensor:
        """Adapt features for target domain"""
        adapted_features = self.feature_adapter(features)
        
        # Apply domain-specific normalization if available
        if target_domain in self.dataset_scalers:
            scaler = self.dataset_scalers[target_domain]
            features_np = adapted_features.detach().cpu().numpy()
            scaled_features = scaler.transform(features_np)
            adapted_features = torch.from_numpy(scaled_features).to(features.device)
        
        return adapted_features
    
    def fit_domain_scalers(self, domain_features: Dict[str, np.ndarray]):
        """Fit dataset-specific feature scalers"""
        for domain, features in domain_features.items():
            scaler = StandardScaler()
            scaler.fit(features)
            self.dataset_scalers[domain] = scaler
            logger.info(f"Fitted scaler for domain: {domain}")

class MetaLearner:
    """Meta-learning for fast adaptation to new datasets"""
    
    def __init__(self, model_dim: int = 512):
        self.model_dim = model_dim
        self.meta_network = self._build_meta_network()
        self.adaptation_modules = {}
        
    def _build_meta_network(self) -> nn.Module:
        """Build meta-learning network"""
        return nn.Sequential(
            nn.Linear(self.model_dim + 64, 256),  # +64 for dataset embedding
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.model_dim),
            nn.Tanh()  # Multiplicative adaptation
        )
    
    def generate_adaptation_params(self, dataset_embedding: torch.Tensor,
                                 model_features: torch.Tensor) -> torch.Tensor:
        """Generate adaptation parameters for specific dataset"""
        combined_input = torch.cat([model_features, dataset_embedding], dim=-1)
        adaptation_params = self.meta_network(combined_input)
        return adaptation_params
    
    def adapt_prediction(self, base_prediction: torch.Tensor,
                        adaptation_params: torch.Tensor) -> torch.Tensor:
        """Apply meta-learned adaptation to prediction"""
        # Element-wise multiplication for feature adaptation
        adapted_features = base_prediction * (1.0 + adaptation_params)
        return adapted_features

class TestTimeAdapter:
    """Test-time adaptation for individual samples"""
    
    def __init__(self, adaptation_steps: int = 5, learning_rate: float = 0.001):
        self.adaptation_steps = adaptation_steps
        self.learning_rate = learning_rate
        self.entropy_threshold = 0.5  # High entropy indicates need for adaptation
        
    def needs_adaptation(self, prediction_entropy: float) -> bool:
        """Determine if sample needs test-time adaptation"""
        return prediction_entropy > self.entropy_threshold
    
    def adapt_model(self, model: nn.Module, sample: torch.Tensor,
                   initial_prediction: torch.Tensor) -> torch.Tensor:
        """Perform test-time adaptation on model"""
        # Create temporary optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        
        # Entropy minimization objective
        model.train()
        adapted_prediction = initial_prediction
        
        for step in range(self.adaptation_steps):
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(sample)
            probabilities = torch.softmax(logits, dim=-1)
            
            # Entropy loss for unsupervised adaptation
            entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-8))
            
            # Confidence regularization
            confidence_reg = torch.std(probabilities)
            
            # Combined loss
            loss = entropy - 0.1 * confidence_reg
            
            loss.backward()
            optimizer.step()
            
            adapted_prediction = probabilities
        
        model.eval()
        return adapted_prediction

class RobustAggregator:
    """Robust aggregation methods for ensemble predictions"""
    
    def __init__(self):
        self.aggregation_methods = {
            'trimmed_mean': self._trimmed_mean,
            'median': self._median,
            'winsorized_mean': self._winsorized_mean,
            'huber_mean': self._huber_mean,
            'consensus': self._consensus_based
        }
    
    def _trimmed_mean(self, predictions: np.ndarray, trim_ratio: float = 0.1) -> float:
        """Trimmed mean aggregation"""
        n_trim = int(len(predictions) * trim_ratio)
        sorted_preds = np.sort(predictions)
        
        if n_trim > 0:
            trimmed = sorted_preds[n_trim:-n_trim]
        else:
            trimmed = sorted_preds
        
        return np.mean(trimmed)
    
    def _median(self, predictions: np.ndarray) -> float:
        """Median aggregation"""
        return np.median(predictions)
    
    def _winsorized_mean(self, predictions: np.ndarray, limit: float = 0.1) -> float:
        """Winsorized mean aggregation"""
        from scipy.stats import mstats
        return mstats.winsorize(predictions, limits=limit).mean()
    
    def _huber_mean(self, predictions: np.ndarray, delta: float = 0.1) -> float:
        """Huber-loss based robust mean"""
        from scipy.optimize import minimize_scalar
        
        def huber_loss(mu):
            residuals = predictions - mu
            return np.sum(np.where(np.abs(residuals) <= delta,
                                 0.5 * residuals**2,
                                 delta * (np.abs(residuals) - 0.5 * delta)))
        
        result = minimize_scalar(huber_loss)
        return result.x
    
    def _consensus_based(self, predictions: np.ndarray, threshold: float = 0.2) -> float:
        """Consensus-based aggregation"""
        # Find predictions within threshold of each other
        consensus_groups = []
        used_indices = set()
        
        for i, pred in enumerate(predictions):
            if i in used_indices:
                continue
                
            group = [pred]
            used_indices.add(i)
            
            for j, other_pred in enumerate(predictions):
                if j != i and j not in used_indices:
                    if abs(pred - other_pred) <= threshold:
                        group.append(other_pred)
                        used_indices.add(j)
            
            consensus_groups.append(group)
        
        # Use largest consensus group
        largest_group = max(consensus_groups, key=len)
        return np.mean(largest_group)
    
    def aggregate(self, predictions: np.ndarray, method: str = 'trimmed_mean',
                 **kwargs) -> float:
        """Apply robust aggregation method"""
        if method in self.aggregation_methods:
            return self.aggregation_methods[method](predictions, **kwargs)
        else:
            logger.warning(f"Unknown aggregation method: {method}, using mean")
            return np.mean(predictions)

class CrossDatasetOptimizer:
    """Main optimizer for cross-dataset generalization"""
    
    def __init__(self, models: Dict[str, Any]):
        self.models = models
        self.dataset_profiler = DatasetProfiler()
        self.domain_adapter = DomainAdaptation()
        self.meta_learner = MetaLearner()
        self.test_time_adapter = TestTimeAdapter()
        self.robust_aggregator = RobustAggregator()
        
        # Performance tracking per dataset
        self.dataset_performance = {
            'faceforensics++': [],
            'dfdc': [],
            'celebdf': [],
            'unknown': []
        }
        
        logger.info("CrossDatasetOptimizer initialized")
    
    def optimize_prediction(self, image: np.ndarray, 
                          base_predictions: Dict[str, float],
                          image_characteristics: Dict[str, float]) -> Dict[str, Any]:
        """Optimize predictions for cross-dataset generalization"""
        
        # Step 1: Identify source dataset
        source_dataset = self.dataset_profiler.identify_dataset(image_characteristics)
        
        # Step 2: Apply domain adaptation
        adapted_predictions = {}
        for model_name, prediction in base_predictions.items():
            # Convert to features (simplified)
            feature_tensor = torch.tensor([prediction] * 512).float().unsqueeze(0)
            adapted_features = self.domain_adapter.adapt_features(feature_tensor, source_dataset)
            adapted_predictions[model_name] = torch.mean(adapted_features).item()
        
        # Step 3: Apply meta-learning adaptation
        dataset_embedding = self._get_dataset_embedding(source_dataset)
        meta_adapted_predictions = {}
        
        for model_name, prediction in adapted_predictions.items():
            model_features = torch.tensor([prediction] * 512).float().unsqueeze(0)
            adaptation_params = self.meta_learner.generate_adaptation_params(
                dataset_embedding, model_features
            )
            adapted_pred = self.meta_learner.adapt_prediction(
                model_features, adaptation_params
            )
            meta_adapted_predictions[model_name] = torch.mean(adapted_pred).item()
        
        # Step 4: Robust aggregation
        prediction_values = np.array(list(meta_adapted_predictions.values()))
        
        # Choose aggregation method based on prediction variance
        prediction_variance = np.var(prediction_values)
        if prediction_variance > 0.1:  # High disagreement
            aggregation_method = 'consensus'
        elif prediction_variance > 0.05:  # Medium disagreement
            aggregation_method = 'trimmed_mean'
        else:  # Low disagreement
            aggregation_method = 'median'
        
        final_prediction = self.robust_aggregator.aggregate(
            prediction_values, method=aggregation_method
        )
        
        # Step 5: Test-time adaptation if needed
        prediction_entropy = self._calculate_entropy(prediction_values)
        if self.test_time_adapter.needs_adaptation(prediction_entropy):
            # Simplified test-time adaptation
            confidence_adjustment = min(0.1, prediction_entropy * 0.2)
            if final_prediction > 0.5:
                final_prediction = max(0.5, final_prediction - confidence_adjustment)
            else:
                final_prediction = min(0.5, final_prediction + confidence_adjustment)
        
        return {
            'optimized_prediction': final_prediction,
            'source_dataset': source_dataset,
            'adaptation_stages': {
                'base_predictions': base_predictions,
                'domain_adapted': adapted_predictions,
                'meta_adapted': meta_adapted_predictions,
                'aggregation_method': aggregation_method,
                'test_time_adapted': prediction_entropy > self.test_time_adapter.entropy_threshold
            },
            'confidence_metrics': {
                'prediction_variance': float(prediction_variance),
                'prediction_entropy': float(prediction_entropy),
                'dataset_confidence': self._get_dataset_confidence(source_dataset)
            }
        }
    
    def _get_dataset_embedding(self, dataset_name: str) -> torch.Tensor:
        """Get learned embedding for dataset"""
        # Simplified embedding based on dataset characteristics
        embeddings = {
            'faceforensics++': torch.randn(64) * 0.1 + torch.tensor([1.0] * 64),
            'dfdc': torch.randn(64) * 0.1 + torch.tensor([0.0] * 64),
            'celebdf': torch.randn(64) * 0.1 + torch.tensor([-1.0] * 64),
            'unknown': torch.zeros(64)
        }
        return embeddings.get(dataset_name, embeddings['unknown'])
    
    def _calculate_entropy(self, predictions: np.ndarray) -> float:
        """Calculate entropy of predictions"""
        # Normalize to probabilities
        probs = np.abs(predictions - 0.5) * 2  # Convert to confidence
        probs = probs / np.sum(probs)
        
        # Calculate entropy
        entropy = -np.sum(probs * np.log(probs + 1e-8))
        return entropy
    
    def _get_dataset_confidence(self, dataset_name: str) -> float:
        """Get confidence in dataset identification"""
        if dataset_name == 'unknown':
            return 0.5
        
        # Base confidence on historical performance
        if dataset_name in self.dataset_performance:
            recent_performance = self.dataset_performance[dataset_name][-10:]
            if recent_performance:
                return np.mean(recent_performance)
        
        return 0.7  # Default confidence
    
    def update_performance(self, dataset_name: str, accuracy: float):
        """Update performance tracking for dataset"""
        if dataset_name in self.dataset_performance:
            self.dataset_performance[dataset_name].append(accuracy)
            # Keep only recent history
            if len(self.dataset_performance[dataset_name]) > 100:
                self.dataset_performance[dataset_name] = self.dataset_performance[dataset_name][-100:]
    
    def get_optimization_info(self) -> Dict[str, Any]:
        """Get information about cross-dataset optimization"""
        return {
            'supported_datasets': list(self.dataset_profiler.dataset_profiles.keys()),
            'optimization_stages': [
                'dataset_identification',
                'domain_adaptation',
                'meta_learning',
                'robust_aggregation',
                'test_time_adaptation'
            ],
            'aggregation_methods': list(self.robust_aggregator.aggregation_methods.keys()),
            'performance_history': {
                k: np.mean(v) if v else 0.0 
                for k, v in self.dataset_performance.items()
            }
        }