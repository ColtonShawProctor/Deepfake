"""
Model registry and factory pattern for multi-model deepfake detection framework.
This module provides centralized model management and instantiation capabilities.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from .base_detector import BaseDetector, ModelInfo, ModelStatus


class ModelRegistry:
    """
    Registry for managing multiple detector models.
    
    Provides centralized registration, loading, and management of different
    deepfake detection models.
    """
    
    def __init__(self, models_dir: Optional[str] = None):
        """
        Initialize the model registry.
        
        Args:
            models_dir: Directory containing model weights and configurations
        """
        self.models: Dict[str, BaseDetector] = {}
        self.model_configs: Dict[str, Dict[str, Any]] = {}
        self.model_classes: Dict[str, Type[BaseDetector]] = {}
        self.models_dir = Path(models_dir) if models_dir else Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(f"{__name__}.ModelRegistry")
        self.logger.info(f"Model registry initialized with models directory: {self.models_dir}")
    
    def register_model(
        self,
        name: str,
        detector_class: Type[BaseDetector],
        config: Optional[Dict[str, Any]] = None,
        auto_load: bool = False
    ) -> bool:
        """
        Register a new detector model.
        
        Args:
            name: Unique name for the model
            detector_class: Class that inherits from BaseDetector
            config: Configuration dictionary for the model
            auto_load: Whether to automatically load the model after registration
            
        Returns:
            True if registration successful, False otherwise
        """
        try:
            if name in self.model_classes:
                self.logger.warning(f"Model '{name}' already registered, overwriting")
            
            self.model_classes[name] = detector_class
            self.model_configs[name] = config or {}
            
            self.logger.info(f"Model '{name}' registered successfully")
            
            if auto_load:
                return self.load_model(name)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register model '{name}': {str(e)}")
            return False
    
    def create_model(self, name: str, **kwargs) -> Optional[BaseDetector]:
        """
        Create a new instance of a registered model.
        
        Args:
            name: Name of the registered model
            **kwargs: Additional arguments to pass to the model constructor
            
        Returns:
            Model instance if successful, None otherwise
        """
        try:
            if name not in self.model_classes:
                self.logger.error(f"Model '{name}' not registered")
                return None
            
            detector_class = self.model_classes[name]
            config = self.model_configs.get(name, {})
            
            # Merge config with kwargs
            model_config = {**config, **kwargs}
            
            detector = detector_class(**model_config)
            self.logger.info(f"Created model instance for '{name}'")
            
            return detector
            
        except Exception as e:
            self.logger.error(f"Failed to create model '{name}': {str(e)}")
            return None
    
    def load_model(self, name: str, model_path: Optional[str] = None) -> bool:
        """
        Load a registered model.
        
        Args:
            name: Name of the registered model
            model_path: Optional path to model weights
            
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            if name not in self.model_classes:
                self.logger.error(f"Model '{name}' not registered")
                return False
            
            # Create model instance if not already created
            if name not in self.models:
                detector = self.create_model(name)
                if detector is None:
                    return False
                self.models[name] = detector
            
            detector = self.models[name]
            
            # Use default model path if not provided
            if model_path is None:
                model_path = str(self.models_dir / f"{name}_weights.pth")
            
            # Load the model
            success = detector.load_model(model_path)
            
            if success:
                self.logger.info(f"Model '{name}' loaded successfully")
                detector.model_info.status = ModelStatus.LOADED
            else:
                self.logger.error(f"Failed to load model '{name}'")
                detector.model_info.status = ModelStatus.ERROR
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error loading model '{name}': {str(e)}")
            return False
    
    def load_all_models(self) -> Dict[str, bool]:
        """
        Load all registered models.
        
        Returns:
            Dictionary mapping model names to load success status
        """
        results = {}
        
        for name in self.model_classes.keys():
            results[name] = self.load_model(name)
        
        self.logger.info(f"Loaded {sum(results.values())}/{len(results)} models")
        return results
    
    def get_model(self, name: str) -> Optional[BaseDetector]:
        """
        Get a loaded model instance.
        
        Args:
            name: Name of the model
            
        Returns:
            Model instance if loaded, None otherwise
        """
        if name not in self.models:
            self.logger.warning(f"Model '{name}' not loaded, attempting to load")
            if not self.load_model(name):
                return None
        
        return self.models.get(name)
    
    def list_models(self) -> List[str]:
        """Get list of all registered model names."""
        return list(self.model_classes.keys())
    
    def list_loaded_models(self) -> List[str]:
        """Get list of currently loaded model names."""
        return [name for name, model in self.models.items() if model.is_loaded()]
    
    def unregister_model(self, name: str) -> bool:
        """
        Unregister a model from the registry.
        
        Args:
            name: Name of the model to unregister
            
        Returns:
            True if unregistered successfully, False otherwise
        """
        try:
            # Remove from all collections
            self.model_classes.pop(name, None)
            self.model_configs.pop(name, None)
            self.models.pop(name, None)
            
            self.logger.info(f"Model '{name}' unregistered successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to unregister model '{name}': {str(e)}")
            return False
    
    def get_model_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status information for all models.
        
        Returns:
            Dictionary containing status information for each model
        """
        status = {}
        
        for name in self.model_classes.keys():
            model = self.models.get(name)
            if model:
                status[name] = {
                    "loaded": model.is_loaded(),
                    "status": model.model_info.status.value,
                    "inference_count": model.inference_count,
                    "average_inference_time": model.average_inference_time,
                    "device": model.device
                }
            else:
                status[name] = {
                    "loaded": False,
                    "status": ModelStatus.UNLOADED.value,
                    "inference_count": 0,
                    "average_inference_time": 0.0,
                    "device": "unknown"
                }
        
        return status
    
    def get_model_info(self, name: str) -> Optional[ModelInfo]:
        """
        Get detailed information about a specific model.
        
        Args:
            name: Name of the model
            
        Returns:
            ModelInfo object if model exists, None otherwise
        """
        model = self.get_model(name)
        if model:
            return model.get_model_info()
        return None


class ModelFactory:
    """
    Factory for creating detector instances.
    
    Provides convenient methods for creating different types of detector models
    with appropriate configurations.
    """
    
    def __init__(self, registry: Optional[ModelRegistry] = None):
        """
        Initialize the model factory.
        
        Args:
            registry: Model registry instance to use
        """
        self.registry = registry or ModelRegistry()
        self.logger = logging.getLogger(f"{__name__}.ModelFactory")
    
    @staticmethod
    def create_mesonet(config: Optional[Dict[str, Any]] = None) -> 'MesoNetDetector':
        """
        Create a MesoNet detector instance.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            MesoNet detector instance
        """
        from .mesonet_detector import MesoNetDetector
        return MesoNetDetector(config or {})
    
    @staticmethod
    def create_resnet(config: Optional[Dict[str, Any]] = None) -> 'ResNetDetector':
        """
        Create a ResNet detector instance.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            ResNet detector instance
        """
        from .resnet_detector import ResNetDetector
        return ResNetDetector(config or {})
    
    @staticmethod
    def create_efficientnet(config: Optional[Dict[str, Any]] = None) -> 'EfficientNetDetector':
        """
        Create an EfficientNet detector instance.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            EfficientNet detector instance
        """
        from .efficientnet_detector import EfficientNetDetector
        return EfficientNetDetector(config or {})
    
    @staticmethod
    def create_f3net(config: Optional[Dict[str, Any]] = None) -> 'F3NetDetector':
        """
        Create an F3Net detector instance.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            F3Net detector instance
        """
        from .f3net_detector import F3NetDetector
        return F3NetDetector(config or {})
    
    def register_default_models(self) -> Dict[str, bool]:
        """
        Register all default models with the registry.
        
        Returns:
            Dictionary mapping model names to registration success status
        """
        results = {}
        
        # Register MesoNet
        try:
            from .mesonet_detector import MesoNetDetector
            results["mesonet"] = self.registry.register_model(
                "mesonet", MesoNetDetector, {"device": "auto"}
            )
        except ImportError:
            self.logger.warning("MesoNet detector not available")
            results["mesonet"] = False
        
        # Register ResNet
        try:
            from .resnet_detector import ResNetDetector
            results["resnet"] = self.registry.register_model(
                "resnet", ResNetDetector, {"device": "auto"}
            )
        except ImportError:
            self.logger.warning("ResNet detector not available")
            results["resnet"] = False
        
        # Register EfficientNet
        try:
            from .efficientnet_detector import EfficientNetDetector
            results["efficientnet"] = self.registry.register_model(
                "efficientnet", EfficientNetDetector, {"device": "auto"}
            )
        except ImportError:
            self.logger.warning("EfficientNet detector not available")
            results["efficientnet"] = False
        
        # Register F3Net
        try:
            from .f3net_detector import F3NetDetector
            results["f3net"] = self.registry.register_model(
                "f3net", F3NetDetector, {"device": "auto"}
            )
        except ImportError:
            self.logger.warning("F3Net detector not available")
            results["f3net"] = False
        
        self.logger.info(f"Registered {sum(results.values())}/{len(results)} default models")
        return results 