import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { fileAPI, analysisAPI, apiUtils } from '../services/api';

const Upload = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      // Clear previous messages
      setMessage('');
      setError('');
      
      // Check file type
      const allowedTypes = ['image/jpeg', 'image/png', 'image/jpg'];
      if (!allowedTypes.includes(file.type)) {
        setError('Please select a valid image file (JPEG, PNG). Video analysis is not yet supported.');
        setSelectedFile(null);
        return;
      }
      
      // Check file size (max 10MB for images)
      if (file.size > 10 * 1024 * 1024) {
        setError('File size must be less than 10MB.');
        setSelectedFile(null);
        return;
      }
      
      setSelectedFile(file);
      setMessage(`Selected: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`);
    }
  };

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!selectedFile) {
      setError('Please select a file to upload.');
      return;
    }

    setUploading(true);
    setAnalyzing(false);
    setMessage('');
    setError('');
    setUploadProgress(0);

    try {
      // Upload file
      setMessage('Uploading file...');
      const uploadResponse = await fileAPI.uploadFile(selectedFile, (progress) => {
        setUploadProgress(progress);
      });

      setMessage('File uploaded successfully! Starting analysis...');
      setUploading(false);
      setAnalyzing(true);

      // Start analysis
      await analysisAPI.analyzeFile(uploadResponse.file_id);
      
      setMessage('Analysis completed successfully! Redirecting to results...');
      setTimeout(() => {
        navigate('/results');
      }, 1500);
    } catch (error) {
      console.error('Upload/Analysis error:', error);
      setError(apiUtils.handleError(error));
      setUploading(false);
      setAnalyzing(false);
      setUploadProgress(0);
    }
  };

  const isProcessing = uploading || analyzing;

  return (
    <div className="container mt-5">
      <div className="row justify-content-center">
        <div className="col-md-8">
          <div className="card shadow">
            <div className="card-header bg-primary text-white">
              <h3 className="mb-0">
                <i className="fas fa-cloud-upload-alt me-2"></i>
                Upload Media for Analysis
              </h3>
            </div>
            <div className="card-body p-5">
              {message && (
                <div className={`alert ${message.includes('successfully') ? 'alert-success' : 'alert-info'}`}>
                  <i className={`fas ${message.includes('successfully') ? 'fa-check-circle' : 'fa-info-circle'} me-2`}></i>
                  {message}
                </div>
              )}

              {error && (
                <div className="alert alert-danger">
                  <i className="fas fa-exclamation-triangle me-2"></i>
                  {error}
                </div>
              )}

              <form onSubmit={handleUpload}>
                <div className="mb-4">
                  <label htmlFor="fileInput" className="form-label">
                    <i className="fas fa-image me-2"></i>
                    Select Image File
                  </label>
                  <input
                    type="file"
                    className="form-control"
                    id="fileInput"
                    accept="image/jpeg,image/png,image/jpg"
                    onChange={handleFileSelect}
                    disabled={isProcessing}
                  />
                  <div className="form-text">
                    <i className="fas fa-info-circle me-1"></i>
                    Supported formats: JPEG, PNG (Max size: 10MB)
                  </div>
                </div>

                {uploadProgress > 0 && uploading && (
                  <div className="mb-3">
                    <label className="form-label">Upload Progress</label>
                    <div className="progress">
                      <div 
                        className="progress-bar progress-bar-striped progress-bar-animated" 
                        style={{ width: `${uploadProgress}%` }}
                      >
                        {uploadProgress}%
                      </div>
                    </div>
                  </div>
                )}

                {analyzing && (
                  <div className="mb-3">
                    <div className="alert alert-info">
                      <div className="d-flex align-items-center">
                        <div className="spinner-border spinner-border-sm me-2" role="status">
                          <span className="visually-hidden">Loading...</span>
                        </div>
                        <span>Analyzing image for deepfake detection...</span>
                      </div>
                    </div>
                  </div>
                )}

                <div className="d-grid gap-2">
                  <button
                    type="submit"
                    className="btn btn-primary btn-lg"
                    disabled={!selectedFile || isProcessing}
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
              </form>

              <hr className="my-4" />

              <div className="row text-center">
                <div className="col-md-4">
                  <div className="card border-0 bg-light">
                    <div className="card-body">
                      <i className="fas fa-shield-alt fa-2x text-primary mb-2"></i>
                      <h6>Secure Upload</h6>
                      <small className="text-muted">Your files are processed securely</small>
                    </div>
                  </div>
                </div>
                <div className="col-md-4">
                  <div className="card border-0 bg-light">
                    <div className="card-body">
                      <i className="fas fa-bolt fa-2x text-warning mb-2"></i>
                      <h6>Fast Analysis</h6>
                      <small className="text-muted">Get results in seconds</small>
                    </div>
                  </div>
                </div>
                <div className="col-md-4">
                  <div className="card border-0 bg-light">
                    <div className="card-body">
                      <i className="fas fa-chart-line fa-2x text-success mb-2"></i>
                      <h6>Detailed Results</h6>
                      <small className="text-muted">Comprehensive analysis reports</small>
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