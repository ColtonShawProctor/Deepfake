# Models initialization - prevent multiple imports
# Import models only once to avoid table conflicts

# Check if models are already imported
_models_imported = False

if not _models_imported:
    from .user import User
    from .media_file import MediaFile
    from .detection_result import DetectionResult
    _models_imported = True

# Export models
__all__ = ['User', 'MediaFile', 'DetectionResult']
