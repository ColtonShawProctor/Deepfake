import React, { useState, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { fileAPI, analysisAPI } from '../services/api';
import { useApiError } from '../hooks/useApiError';

const Upload = () => {
  const navigate = useNavigate();
  const { error, handleError, clearError } = useApiError();
  const fileInputRef = useRef(null);
  const dropZoneRef = useRef(null);
  
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadResult, setUploadResult] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [validationErrors, setValidationErrors] = useState({});

  // File validation constants
  const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB
  const ALLOWED_TYPES = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/bmp', 'image/webp'];
  const ALLOWED_EXTENSIONS = ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'];

  const validateFile = useCallback((file) => {
    const errors = {};

    // Check file type
    if (!ALLOWED_TYPES.includes(file.type)) {
      errors.type = `File type not supported. Allowed types: ${ALLOWED_EXTENSIONS.join(', ').toUpperCase()}`;
    }

    // Check file size
    if (file.size > MAX_FILE_SIZE) {
      errors.size = `File size must be less than ${(MAX_FILE_SIZE / 1024 / 1024).toFixed(0)}MB`;
    }

    // Check if file is actually an image
    if (!file.type.startsWith('image/')) {
      errors.type = ' Please select a valid image file';
    }

    setValidationErrors(errors);
    return Object.keys(errors).length === 0;
  }, [ALLOWED_EXTENSIONS, ALLOWED_TYPES, MAX_FILE_SIZE]);

  const handleFileSelect = useCallback((file) => {
    if (!file) return;

    clearError();
    setValidationErrors({});
    setUploadResult(null);

    if (validateFile(file)) {
      setSelectedFile(file);
      
      // Create preview URL
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
    } else {
      setSelectedFile(null);
      setPreviewUrl(null);
    }
  }, [validateFile, clearError]);

  const handleInputChange = (e) => {
    const file = e.target.files[0];
    handleFileSelect(file);
  };

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDragIn = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.dataTransfer.items && e.dataTransfer.items.length > 0) {
      setDragActive(true);
    }
  }, []);

  const handleDragOut = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      handleFileSelect(file);
      e.dataTransfer.clearData();
    }
  }, [handleFileSelect]);

  const handleUpload = async (e) => {
    e.preventDefault();
    
    if (!selectedFile) {
      handleError(new Error('Please select a file to upload'));
      return;
    }

    setUploading(true);
    setAnalyzing(false);
    setUploadProgress(0);
    setUploadResult(null);
    clearError();

    try {
      // Upload file with progress tracking
      const uploadResponse = await fileAPI.uploadFile(selectedFile, (progress) => {
        setUploadProgress(progress);
      });

      setUploadResult({
        fileId: uploadResponse.file_id,
        fileName: selectedFile.name,
        fileSize: selectedFile.size,
        uploadTime: new Date().toISOString()
      });

      setUploading(false);
      setAnalyzing(true);

      // Start analysis
      await analysisAPI.analyzeFile(uploadResponse.file_id);
      
      // Redirect to results page after successful analysis
      setTimeout(() => {
        navigate(`/results/${uploadResponse.file_id}`);
      }, 2000);

    } catch (err) {
      handleError(err);
      setUploading(false);
      setAnalyzing(false);
      setUploadProgress(0);
    }
  };

  const handleFileRemove = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setValidationErrors({});
    setUploadResult(null);
    clearError();
    
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const isProcessing = uploading || analyzing;
  const hasErrors = Object.keys(validationErrors).length > 0;

  return (
    <div className="container mt-4">
      <div className="row justify-content-center">
        <div className="col-lg-10">
          <div className="card border-0 shadow-lg">
            <div className="card-header bg-primary text-white py-3">
              <h3 className="mb-0">
                <i className="fas fa-cloud-upload-alt me-2"></i>
                Upload Media for Deepfake Detection
              </h3>
            </div>
            
            <div className="card-body p-4">
              {/* Error Alert */}
              {error && (
                <div className="alert alert-danger alert-dismissible fade show mb-4" role="alert">
                  <i className="fas fa-exclamation-triangle me-2"></i>
                  {error}
                  <button 
                    type="button" 
                    className="btn-close" 
                    onClick={clearError}
                    aria-label="Close"
                  ></button>
                </div>
              )}

              {/* Success Message */}
              {uploadResult && !isProcessing && (
                <div className="alert alert-success alert-dismissible fade show mb-4" role="alert">
                  <i className="fas fa-check-circle me-2"></i>
                  <strong>Upload Successful!</strong> File uploaded and analysis started.
                  <button 
                    type="button" 
                    className="btn-close" 
                    onClick={() => setUploadResult(null)}
                    aria-label="Close"
                  ></button>
                </div>
              )}

              {/* Drag & Drop Zone */}
              <div 
                ref={dropZoneRef}
                className={`border-2 border-dashed rounded-3 p-5 text-center mb-4 ${
                  dragActive 
                    ? 'border-primary bg-primary bg-opacity-10' 
                    : selectedFile 
                    ? 'border-success bg-success bg-opacity-10' 
                    : 'border-secondary bg-light'
                }`}
                onDragEnter={handleDragIn}
                onDragLeave={handleDragOut}
                onDragOver={handleDrag}
                onDrop={handleDrop}
                style={{ 
                  borderStyle: 'dashed',
                  transition: 'all 0.3s ease',
                  cursor: 'pointer'
                }}
                onClick={() => fileInputRef.current?.click()}
              >
                {!selectedFile ? (
                  <div>
                    <i className="fas fa-cloud-upload-alt fa-3x text-muted mb-3"></i>
                    <h5 className="text-muted mb-2">Drag & Drop your image here</h5>
                    <p className="text-muted mb-3">or click to browse files</p>
                    <button 
                      type="button" 
                      className="btn btn-outline-primary"
                      onClick={(e) => {
                        e.stopPropagation();
                        fileInputRef.current?.click();
                      }}
                    >
                      <i className="fas fa-folder-open me-2"></i>
                      Choose File
                    </button>
                  </div>
                ) : (
                  <div>
                    <i className="fas fa-check-circle fa-3x text-success mb-3"></i>
                    <h5 className="text-success mb-2">File Selected!</h5>
                    <p className="text-muted mb-0">{selectedFile.name}</p>
                  </div>
                )}
              </div>

              {/* Hidden File Input */}
              <input
                ref={fileInputRef}
                type="file"
                className="d-none"
                accept={ALLOWED_TYPES.join(',')}
                onChange={handleInputChange}
                disabled={isProcessing}
              />

              {/* Validation Errors */}
              {hasErrors && (
                <div className="alert alert-warning mb-4">
                  <i className="fas fa-exclamation-triangle me-2"></i>
                  <strong>File Validation Errors:</strong>
                  <ul className="mb-0 mt-2">
                    {Object.values(validationErrors).map((error, index) => (
                      <li key={index}>{error}</li>
                    ))}
                  </ul>
                </div>
              )}

              {/* File Preview */}
              {selectedFile && previewUrl && !hasErrors && (
                <div className="card mb-4">
                  <div className="card-header bg-light">
                    <div className="d-flex justify-content-between align-items-center">
                      <h6 className="mb-0">
                        <i className="fas fa-image me-2"></i>
                        File Preview
                      </h6>
                      <button
                        type="button"
                        className="btn btn-sm btn-outline-danger"
                        onClick={handleFileRemove}
                        disabled={isProcessing}
                      >
                        <i className="fas fa-times me-1"></i>
                        Remove
                      </button>
                    </div>
                  </div>
                  <div className="card-body text-center">
                    <img
                      src={previewUrl}
                      alt="Preview"
                      className="img-fluid rounded"
                      style={{ maxHeight: '300px', maxWidth: '100%' }}
                    />
                    <div className="mt-3">
                      <p className="mb-1">
                        <strong>Name:</strong> {selectedFile.name}
                      </p>
                      <p className="mb-1">
                        <strong>Size:</strong> {formatFileSize(selectedFile.size)}
                      </p>
                      <p className="mb-0">
                        <strong>Type:</strong> {selectedFile.type}
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {/* Upload Progress */}
              {uploading && uploadProgress > 0 && (
                <div className="card mb-4">
                  <div className="card-header bg-info text-white">
                    <h6 className="mb-0">
                      <i className="fas fa-upload me-2"></i>
                      Uploading File...
                    </h6>
                  </div>
                  <div className="card-body">
                    <div className="progress mb-2">
                      <div 
                        className="progress-bar progress-bar-striped progress-bar-animated bg-info" 
                        style={{ width: `${uploadProgress}%` }}
                        role="progressbar"
                        aria-valuenow={uploadProgress}
                        aria-valuemin="0"
                        aria-valuemax="100"
                      >
                        {uploadProgress}%
                      </div>
                    </div>
                    <small className="text-muted">
                      Uploading {selectedFile?.name}...
                    </small>
                  </div>
                </div>
              )}

              {/* Analysis Progress */}
              {analyzing && (
                <div className="card mb-4">
                  <div className="card-header bg-warning text-dark">
                    <h6 className="mb-0">
                      <i className="fas fa-search me-2"></i>
                      Analyzing for Deepfake Detection...
                    </h6>
                  </div>
                  <div className="card-body text-center">
                    <div className="spinner-border text-warning mb-3" role="status">
                      <span className="visually-hidden">Loading...</span>
                    </div>
                    <p className="mb-0">Processing your image with advanced AI algorithms...</p>
                  </div>
                </div>
              )}

              {/* Upload Button */}
              <div className="d-grid gap-2">
                <button
                  type="button"
                  className="btn btn-primary btn-lg"
                  onClick={handleUpload}
                  disabled={!selectedFile || hasErrors || isProcessing}
                >
                  {isProcessing ? (
                    <>
                      <span className="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
                      {uploading ? 'Uploading...' : 'Analyzing...'}
                    </>
                  ) : (
                    <>
                      <i className="fas fa-search me-2"></i>
                      Upload & Analyze
                    </>
                  )}
                </button>
              </div>

              {/* File Requirements */}
              <div className="mt-4">
                <div className="card border-0 bg-light">
                  <div className="card-body">
                    <h6 className="card-title">
                      <i className="fas fa-info-circle me-2 text-info"></i>
                      File Requirements
                    </h6>
                    <div className="row">
                      <div className="col-md-6">
                        <ul className="list-unstyled mb-0">
                          <li><i className="fas fa-check text-success me-2"></i>Supported formats: JPG, PNG, GIF, BMP, WebP</li>
                          <li><i className="fas fa-check text-success me-2"></i>Maximum file size: 10MB</li>
                        </ul>
                      </div>
                      <div className="col-md-6">
                        <ul className="list-unstyled mb-0">
                          <li><i className="fas fa-check text-success me-2"></i>High-quality images recommended</li>
                          <li><i className="fas fa-check text-success me-2"></i>Secure and private processing</li>
                        </ul>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Upload; 