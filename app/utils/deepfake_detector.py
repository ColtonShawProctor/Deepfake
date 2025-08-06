import os
import logging
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image, UnidentifiedImageError
import random
import time
from datetime import datetime

# Import enhanced detector
from .enhanced_deepfake_detector import EnhancedDeepfakeDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeepfakeDetector:
    """
    Mock deepfake detection service for testing purposes.
    
    This is a placeholder implementation that generates realistic-looking
    fake results based on basic image analysis until a real ML model is integrated.
    """
    
    def __init__(self, use_enhanced: bool = True):
        self.logger = logging.getLogger(__name__)
        
        # Initialize enhanced detector if requested
        self.enhanced_detector = None
        if use_enhanced:
            try:
                self.enhanced_detector = EnhancedDeepfakeDetector(use_ensemble=True)
                self.logger.info("Enhanced DeepfakeDetector initialized with multi-model ensemble")
            except Exception as e:
                self.logger.warning(f"Failed to initialize enhanced detector: {str(e)}")
                self.logger.info("Falling back to mock detector")
                self.enhanced_detector = None
        
        self.logger.info("DeepfakeDetector initialized")
    
    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze an image for deepfake detection.
        
        Args:
            image_path (str): Path to the image file to analyze
            
        Returns:
            Dict containing:
                - confidence_score (float): 0-100 confidence score
                - is_deepfake (bool): True if confidence > 50%
                - analysis_metadata (Dict): Additional analysis information
                - analysis_time (str): ISO timestamp of analysis
                - error (str): Error message if analysis failed
                
        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image is corrupted or invalid
        """
        start_time = time.time()
        
        try:
            # Use enhanced detector if available
            if self.enhanced_detector:
                self.logger.info("Using enhanced multi-model detector")
                return self.enhanced_detector.analyze_image(image_path)
            
            # Fallback to mock analysis
            self.logger.info("Using mock detector (enhanced detector not available)")
            
            # Validate file exists
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            self.logger.info(f"Starting deepfake analysis for: {image_path}")
            
            # Load and validate image
            with Image.open(image_path) as img:
                # Get basic image properties
                width, height = img.size
                file_size = os.path.getsize(image_path)
                format_name = img.format
                mode = img.mode
                
                # Generate mock analysis based on image properties
                confidence_score = self._generate_mock_confidence(
                    width, height, file_size, format_name, mode, image_path
                )
                
                # Determine if it's a deepfake (confidence > 50%)
                is_deepfake = confidence_score > 50.0
                
                # Generate analysis metadata
                analysis_metadata = self._generate_analysis_metadata(
                    width, height, file_size, format_name, mode, 
                    confidence_score, is_deepfake, image_path
                )
                
                analysis_time = time.time() - start_time
                
                result = {
                    "confidence_score": confidence_score,
                    "is_deepfake": is_deepfake,
                    "analysis_metadata": analysis_metadata,
                    "analysis_time": datetime.utcnow().isoformat(),
                    "processing_time_seconds": round(analysis_time, 3),
                    "error": None
                }
                
                self.logger.info(
                    f"Analysis complete for {image_path}: "
                    f"confidence={confidence_score:.1f}%, "
                    f"is_deepfake={is_deepfake}, "
                    f"processing_time={analysis_time:.3f}s"
                )
                
                return result
                
        except FileNotFoundError as e:
            error_msg = f"File not found: {str(e)}"
            self.logger.error(f"Analysis failed for {image_path}: {error_msg}")
            return {
                "confidence_score": 0.0,
                "is_deepfake": False,
                "analysis_metadata": {},
                "analysis_time": datetime.utcnow().isoformat(),
                "processing_time_seconds": 0.0,
                "error": error_msg
            }
            
        except UnidentifiedImageError as e:
            error_msg = f"Invalid or corrupted image file: {str(e)}"
            self.logger.error(f"Analysis failed for {image_path}: {error_msg}")
            return {
                "confidence_score": 0.0,
                "is_deepfake": False,
                "analysis_metadata": {},
                "analysis_time": datetime.utcnow().isoformat(),
                "processing_time_seconds": 0.0,
                "error": error_msg
            }
            
        except Exception as e:
            error_msg = f"Unexpected error during analysis: {str(e)}"
            self.logger.error(f"Analysis failed for {image_path}: {error_msg}")
            return {
                "confidence_score": 0.0,
                "is_deepfake": False,
                "analysis_metadata": {},
                "analysis_time": datetime.utcnow().isoformat(),
                "processing_time_seconds": 0.0,
                "error": error_msg
            }
    
    def _generate_mock_confidence(
        self, 
        width: int, 
        height: int, 
        file_size: int, 
        format_name: str, 
        mode: str,
        image_path: str
    ) -> float:
        """
        Generate a realistic mock confidence score based on image properties.
        
        This creates varied but realistic-looking results for testing purposes.
        """
        # Use file hash as seed for consistent results per image
        file_hash = hashlib.md5(image_path.encode()).hexdigest()
        random.seed(int(file_hash[:8], 16))
        
        # Base confidence factors
        base_confidence = 30.0  # Base 30% confidence
        
        # Factor 1: Image resolution (higher res = more suspicious)
        resolution_factor = min((width * height) / (1920 * 1080), 2.0) * 15
        
        # Factor 2: File size (larger files = more suspicious)
        size_factor = min(file_size / (1024 * 1024), 5.0) * 8  # Max 5MB factor
        
        # Factor 3: Format preference (JPEG more suspicious than PNG)
        format_factor = 10 if format_name == 'JPEG' else 5
        
        # Factor 4: Color mode (RGB more suspicious than grayscale)
        color_factor = 8 if mode == 'RGB' else 2
        
        # Factor 5: Random variation (±10%)
        random_factor = random.uniform(-10, 10)
        
        # Calculate final confidence
        confidence = base_confidence + resolution_factor + size_factor + format_factor + color_factor + random_factor
        
        # Ensure confidence is between 0-100
        confidence = max(0.0, min(100.0, confidence))
        
        return round(confidence, 1)
    
    def analyze_batch(self, image_paths: list) -> list:
        """
        Analyze multiple images for deepfake detection.
        
        Args:
            image_paths (list): List of paths to image files to analyze
            
        Returns:
            List of analysis results, one for each image
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.analyze_image(image_path)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to analyze {image_path}: {str(e)}")
                results.append({
                    "confidence_score": 0.0,
                    "is_deepfake": False,
                    "analysis_metadata": {},
                    "analysis_time": datetime.utcnow().isoformat(),
                    "processing_time_seconds": 0.0,
                    "error": f"Analysis failed: {str(e)}"
                })
        return results
    
    def _generate_analysis_metadata(
        self,
        width: int,
        height: int,
        file_size: int,
        format_name: str,
        mode: str,
        confidence_score: float,
        is_deepfake: bool,
        image_path: str
    ) -> Dict[str, Any]:
        """
        Generate detailed analysis metadata for the detection result.
        """
        # Calculate image statistics
        aspect_ratio = round(width / height, 2) if height > 0 else 0
        file_size_mb = round(file_size / (1024 * 1024), 2)
        
        # Generate mock detection features
        features = {
            "face_detection": {
                "faces_found": random.randint(0, 3),
                "face_confidence": random.uniform(0.1, 0.9)
            },
            "texture_analysis": {
                "noise_level": random.uniform(0.1, 0.8),
                "compression_artifacts": random.uniform(0.0, 0.6)
            },
            "color_analysis": {
                "color_consistency": random.uniform(0.3, 0.9),
                "saturation_variance": random.uniform(0.1, 0.7)
            },
            "edge_detection": {
                "edge_sharpness": random.uniform(0.2, 0.8),
                "artificial_edges": random.uniform(0.0, 0.5)
            }
        }
        
        # Generate mock model predictions
        model_predictions = {
            "face_swap_probability": random.uniform(0.0, 0.8),
            "style_transfer_probability": random.uniform(0.0, 0.6),
            "gan_generated_probability": random.uniform(0.0, 0.7),
            "deepfake_indicators": random.randint(0, 5)
        }
        
        metadata = {
            "image_properties": {
                "width": width,
                "height": height,
                "aspect_ratio": aspect_ratio,
                "file_size_bytes": file_size,
                "file_size_mb": file_size_mb,
                "format": format_name,
                "color_mode": mode,
                "filename": Path(image_path).name
            },
            "detection_features": features,
            "model_predictions": model_predictions,
            "analysis_parameters": {
                "model_version": "mock-v1.0.0",
                "analysis_method": "mock_detection",
                "confidence_threshold": 50.0,
                "processing_notes": "Mock analysis for testing purposes"
            },
            "result_summary": {
                "primary_indicator": random.choice([
                    "face_inconsistency", "texture_anomaly", 
                    "color_artifacts", "edge_irregularity", "compression_patterns"
                ]),
                "secondary_indicators": random.sample([
                    "lighting_inconsistency", "perspective_error",
                    "blending_artifacts", "resolution_mismatch"
                ], random.randint(0, 2)),
                "recommendation": "Further analysis recommended" if confidence_score > 30 else "Image appears authentic"
            }
        }
        
        return metadata
    
    def get_detector_info(self) -> Dict[str, Any]:
        """
        Get information about the detector service.
        """
        if self.enhanced_detector:
            return self.enhanced_detector.get_detector_info()
        
        return {
            "name": "Mock Deepfake Detector",
            "version": "1.0.0",
            "description": "Placeholder detector for testing purposes",
            "capabilities": [
                "Image format validation",
                "Basic image analysis",
                "Mock confidence scoring",
                "Metadata generation"
            ],
            "supported_formats": ["JPEG", "PNG", "JPG"],
            "max_file_size_mb": 10,
            "confidence_threshold": 50.0
        } 