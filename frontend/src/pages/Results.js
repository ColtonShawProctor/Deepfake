import React, { useState, useEffect, useCallback, useRef } from 'react';
import { useParams, Link } from 'react-router-dom';
import { analysisAPI } from '../services/api';
import { useApiError } from '../hooks/useApiError';

const Results = () => {
  const { fileId } = useParams();
  const { error, handleError, clearError } = useApiError();
  
  const [result, setResult] = useState(null);
  const [allResults, setAllResults] = useState([]);
  const [loading, setLoading] = useState(true);
  const [retryLoading, setRetryLoading] = useState(false);
  const [downloadLoading, setDownloadLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [loadingMessage] = useState('Loading Analysis Results...');
  
  // Use ref to track if we've already fetched for this fileId
  const hasFetchedRef = useRef(false);
  const currentFileIdRef = useRef(null);

  const fetchResult = useCallback(async () => {
    // Prevent duplicate fetches for same fileId (but allow initial fetch even when loading)
    if (hasFetchedRef.current && currentFileIdRef.current === fileId) {
      return;
    }
    
    setLoading(true);
    clearError();
    
    // Mark that we're fetching for this fileId
    hasFetchedRef.current = true;
    currentFileIdRef.current = fileId;
    
    try {
      // Simple single request with timeout
      const timeoutPromise = new Promise((_, reject) => 
        setTimeout(() => reject(new Error('Request timeout')), 10000)
      );
      
      const responsePromise = analysisAPI.getResults(parseInt(fileId));
      const response = await Promise.race([responsePromise, timeoutPromise]);
      
      if (!response || !response.detection_result) {
        throw new Error('Invalid response format');
      }
      
      const transformedResult = {
        id: response.file_id,
        filename: response.filename,
        uploadDate: new Date(response.created_at).toLocaleDateString(),
        uploadTime: new Date(response.created_at).toLocaleTimeString(),
        status: 'completed',
        confidence: response.detection_result.confidence_score,
        isDeepfake: response.detection_result.is_deepfake,
        analysisTime: `${response.detection_result.processing_time_seconds.toFixed(1)}s`,
        createdAt: response.created_at,
        detectionResult: response.detection_result,
        // Try different possible field names for file URL
        fileUrl: response.file_url || response.file_path || response.url || response.media_url || constructFileUrl(response.file_id, response.filename),
        // Try different possible field names for file size
        fileSize: response.file_size || response.size || response.fileSize || 'Unknown',
        // Try different possible field names for file type
        fileType: response.file_type || response.mime_type || response.content_type || response.type || 'Unknown'
      };
      
      setResult(transformedResult);
    } catch (err) {
      handleError(err);
      console.error('Error fetching result:', err);
    } finally {
      setLoading(false);
    }
  }, [fileId, clearError, handleError]);

  const fetchAllResults = useCallback(async () => {
    setLoading(true);
    clearError();
    
    try {
      const response = await analysisAPI.getAllResults(1, 50); // Get first 50 results
      setAllResults(response || []);
    } catch (err) {
      handleError(err);
      console.error('Error fetching all results:', err);
    } finally {
      setLoading(false);
    }
  }, [clearError, handleError]);

  useEffect(() => {
    if (fileId) {
      // Individual result page
      // Reset fetch flag when fileId changes
      if (currentFileIdRef.current !== fileId) {
        hasFetchedRef.current = false;
        currentFileIdRef.current = fileId;
      }
      fetchResult();
    } else {
      // General results page - fetch all results
      fetchAllResults();
    }
  }, [fileId, fetchResult, fetchAllResults]);

  const handleRetryAnalysis = async () => {
    setRetryLoading(true);
    clearError();
    setMessage('');
    
    // Reset fetch flag to allow new fetch after retry
    hasFetchedRef.current = false;
    
    try {
      const response = await analysisAPI.analyzeFile(fileId);
      
      // Show success message
      setMessage('Analysis restarted successfully! Checking status...');
      
      // Poll for completion (check every 3 seconds for up to 2 minutes)
      let attempts = 0;
      const maxAttempts = 40; // 2 minutes total
      
      const pollForCompletion = async () => {
        try {
          const resultResponse = await analysisAPI.getResults(parseInt(fileId));
          
          if (resultResponse.detection_result && resultResponse.detection_result.confidence_score !== undefined) {
            setMessage('Analysis completed! Refreshing results...');
            setTimeout(() => {
              fetchResult();
              setMessage('');
            }, 1000);
            return;
          }
        } catch (err) {
          console.log('Poll attempt failed:', err);
        }
        
        attempts++;
        if (attempts < maxAttempts) {
          setTimeout(pollForCompletion, 3000);
        } else {
          setMessage('Analysis is taking longer than expected. Please refresh the page in a few minutes.');
        }
      };
      
      // Start polling after 3 seconds
      setTimeout(pollForCompletion, 3000);
      
    } catch (err) {
      console.error('Error retrying analysis:', err);
      handleError(err);
      setMessage('Failed to retry analysis. Please try again.');
    } finally {
      setRetryLoading(false);
    }
  };

  const handleDownloadResults = async () => {
    setDownloadLoading(true);
    
    try {
      // Create JSON data for download
      const jsonData = {
        analysis_id: result.id,
        filename: result.filename,
        analysis_date: result.createdAt,
        upload_date: result.uploadDate,
        upload_time: result.uploadTime,
        file_size: result.fileSize,
        file_type: result.fileType,
        processing_time: result.analysisTime,
        detection_result: {
          confidence_score: result.confidence,
          is_deepfake: result.isDeepfake,
          verdict: getVerdict(result.confidence),
          processing_time_seconds: result.detectionResult.processing_time_seconds
        },
        metadata: result.detectionResult
      };

      // Create and download file
      const blob = new Blob([JSON.stringify(jsonData, null, 2)], { type: 'application/json' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `deepfake-analysis-${result.filename}-${new Date().toISOString().split('T')[0]}.json`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err) {
      handleError(err);
      console.error('Error downloading results:', err);
    } finally {
      setDownloadLoading(false);
    }
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 70) return 'danger';
    if (confidence >= 30) return 'warning';
    return 'success';
  };

  const getConfidenceText = (confidence) => {
    if (confidence >= 70) return 'High Risk';
    if (confidence >= 30) return 'Medium Risk';
    return 'Low Risk';
  };

  const getVerdict = (confidence) => {
    if (confidence >= 70) return 'FAKE';
    if (confidence >= 30) return 'UNCERTAIN';
    return 'REAL';
  };

  const getVerdictColor = (confidence) => {
    if (confidence === 0) return 'secondary'; // Pending/Processing
    if (confidence >= 70) return 'danger';
    if (confidence >= 30) return 'warning';
    return 'success';
  };

  const getFileTypeIcon = (filename) => {
    const extension = filename.split('.').pop()?.toLowerCase();
    if (['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'].includes(extension)) {
      return 'fas fa-image text-primary';
    } else if (['mp4', 'avi', 'mov', 'wmv', 'flv', 'webm'].includes(extension)) {
      return 'fas fa-video text-danger';
    } else {
      return 'fas fa-file text-secondary';
    }
  };

  const formatFileSize = (bytes) => {
    if (!bytes || bytes === 'Unknown') return 'Unknown';
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
  };

  const constructFileUrl = (fileId, filename) => {
    // If we have a file ID, try to construct a URL to the file
    if (fileId) {
      // Use the public endpoint for image display (no authentication required)
      const baseUrl = process.env.REACT_APP_API_URL || 'http://localhost:8000';
      return `${baseUrl}/api/files/public/${fileId}`;
    }
    return null;
  };

  const getFullFileUrl = (fileUrl) => {
    // If the URL is already absolute, return it
    if (fileUrl && (fileUrl.startsWith('http://') || fileUrl.startsWith('https://'))) {
      return fileUrl;
    }
    // If it's a relative URL, make it absolute
    if (fileUrl && fileUrl.startsWith('/')) {
      const baseUrl = process.env.REACT_APP_API_URL || 'http://localhost:8000';
      return `${baseUrl}${fileUrl}`;
    }
    return fileUrl;
  };

  // Function to capture video frame at specific time
  const captureVideoFrame = useCallback((videoUrl, timeInSeconds = 1) => {
    return new Promise((resolve, reject) => {
      const video = document.createElement('video');
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      
      video.crossOrigin = 'anonymous';
      video.muted = true;
      video.playsInline = true;
      
      video.onloadedmetadata = () => {
        // Set canvas size to match video dimensions
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        // Seek to specific time
        video.currentTime = timeInSeconds;
      };
      
      video.onseeked = () => {
        try {
          // Draw the current frame to canvas
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          
          // Convert to data URL
          const thumbnailUrl = canvas.toDataURL('image/jpeg', 0.8);
          resolve(thumbnailUrl);
          
          // Clean up
          video.remove();
          canvas.remove();
        } catch (error) {
          reject(error);
        }
      };
      
      video.onerror = () => {
        reject(new Error('Failed to load video for thumbnail generation'));
      };
      
      // Start loading the video
      video.src = videoUrl;
      video.load();
    });
  }, []);

  // State for video thumbnail
  const [videoThumbnail, setVideoThumbnail] = useState(null);
  const [thumbnailLoading, setThumbnailLoading] = useState(false);

  // Generate video thumbnail when video result is loaded
  useEffect(() => {
    if (result && result.fileUrl && (result.fileType?.startsWith('video') || result.filename?.match(/\.(mp4|avi|mov|wmv|flv|webm)$/i))) {
      setThumbnailLoading(true);
      setVideoThumbnail(null);
      
      captureVideoFrame(getFullFileUrl(result.fileUrl), 1)
        .then(thumbnail => {
          setVideoThumbnail(thumbnail);
        })
        .catch(error => {
          console.log('Thumbnail generation failed:', error);
          // Fallback to info card
        })
        .finally(() => {
          setThumbnailLoading(false);
        });
    }
  }, [result, captureVideoFrame]);

  if (loading) {
    return (
      <div className="container mt-5">
        <div className="row justify-content-center">
          <div className="col-md-8 text-center">
            <div className="card shadow">
              <div className="card-body p-5">
                <div className="spinner-border text-primary mb-3" role="status">
                  <span className="visually-hidden">Loading...</span>
                </div>
                <h5>{loadingMessage}</h5>
                <p className="text-muted">Please wait while we process your file</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Handle general results page (no fileId)
  if (!fileId) {
    return (
      <div className="container-fluid mt-4">
        {/* Error Alert */}
        {error && (
          <div className="alert alert-danger alert-dismissible fade show" role="alert">
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

        {/* Header */}
        <div className="row mb-4">
          <div className="col-12">
            <div className="d-flex justify-content-between align-items-center">
              <div>
                <h1 className="h2 mb-1">
                  <i className="fas fa-chart-bar me-2 text-primary"></i>
                  Analysis Results
                </h1>
                <p className="text-muted mb-0">
                  View all your deepfake detection analysis results
                </p>
              </div>
              <Link to="/upload" className="btn btn-primary">
                <i className="fas fa-plus me-2"></i>
                New Analysis
              </Link>
            </div>
          </div>
        </div>

        {/* Results List */}
        {allResults.length === 0 ? (
          <div className="row justify-content-center">
            <div className="col-md-8 text-center">
              <div className="card shadow">
                <div className="card-body p-5">
                  <i className="fas fa-chart-bar fa-3x text-muted mb-3"></i>
                  <h5>No Analysis Results</h5>
                  <p className="text-muted mb-4">You haven't performed any analysis yet. Upload a file to get started!</p>
                  <Link to="/upload" className="btn btn-primary">
                    <i className="fas fa-upload me-2"></i>
                    Upload File
                  </Link>
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="row">
            {allResults.map((result, index) => {
              // Debug: Log the result structure
              console.log(`Result ${index}:`, result);
              
              // Safely access the confidence score
              const confidenceScore = result.detection_result?.confidence_score || 0;
              const isDeepfake = result.detection_result?.is_deepfake || false;
              
              return (
                <div key={index} className="col-md-6 col-lg-4 mb-4">
                  <div className="card border-0 shadow-sm h-100">
                    <div className="card-body">
                      <div className="d-flex align-items-center mb-3">
                        <i className={`${getFileTypeIcon(result.filename)} fa-2x me-3`}></i>
                        <div>
                          <h6 className="card-title mb-1">{result.filename}</h6>
                          <small className="text-muted">
                            {new Date(result.created_at).toLocaleDateString()}
                          </small>
                        </div>
                      </div>
                      
                      <div className="mb-3">
                        <div className="d-flex justify-content-between align-items-center mb-2">
                          <span className="text-muted">Confidence:</span>
                          <span className={`badge bg-${getConfidenceColor(confidenceScore)}`}>
                            {confidenceScore === 0 ? 'N/A' : `${confidenceScore.toFixed(1)}%`}
                          </span>
                        </div>
                        <div className="d-flex justify-content-between align-items-center">
                          <span className="text-muted">Verdict:</span>
                          <span className={`badge bg-${getVerdictColor(confidenceScore)}`}>
                            {confidenceScore === 0 ? 'PENDING' : getVerdict(confidenceScore)}
                          </span>
                        </div>
                        <div className="d-flex justify-content-between align-items-center">
                          <span className="text-muted">Status:</span>
                          <span className={`badge ${confidenceScore === 0 ? 'bg-secondary' : (isDeepfake ? 'bg-danger' : 'bg-success')}`}>
                            {confidenceScore === 0 ? 'Processing' : (isDeepfake ? 'Deepfake' : 'Authentic')}
                          </span>
                        </div>
                      </div>
                      
                      <Link 
                        to={`/results/${result.file_id}`} 
                        className="btn btn-outline-primary btn-sm w-100"
                      >
                        <i className="fas fa-eye me-2"></i>
                        View Details
                      </Link>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    );
  }

  // Handle individual result page (with fileId)
  if (!result) {
    return (
      <div className="container mt-5">
        <div className="row justify-content-center">
          <div className="col-md-8 text-center">
            <div className="card shadow">
              <div className="card-body p-5">
                <i className="fas fa-exclamation-triangle fa-3x text-warning mb-3"></i>
                <h5>No Results Found</h5>
                <p className="text-muted mb-4">The requested analysis result could not be found.</p>
                <Link to="/dashboard" className="btn btn-primary">
                  <i className="fas fa-arrow-left me-2"></i>
                  Back to Dashboard
                </Link>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="container-fluid mt-4">
      {/* Error Alert */}
      {error && (
        <div className="alert alert-danger alert-dismissible fade show" role="alert">
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

      {/* Success/Info Alert */}
      {message && (
        <div className="alert alert-info alert-dismissible fade show" role="alert">
          <i className="fas fa-info-circle me-2"></i>
          {message}
          <button 
            type="button" 
            className="btn-close" 
            onClick={() => setMessage('')}
            aria-label="Close"
          ></button>
        </div>
      )}

      {/* Header */}
      <div className="row mb-4">
        <div className="col-12">
          <div className="d-flex justify-content-between align-items-center">
            <div>
              <h1 className="h2 mb-1">
                <i className="fas fa-chart-bar me-2 text-primary"></i>
                Analysis Results
              </h1>
              <p className="text-muted mb-0">
                Deepfake detection results for <strong>{result.filename}</strong>
              </p>
            </div>
            <div className="d-flex gap-2">
              <button
                onClick={handleRetryAnalysis}
                className="btn btn-outline-primary"
                disabled={retryLoading}
              >
                {retryLoading ? (
                  <>
                    <span className="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
                    Retrying...
                  </>
                ) : (
                  <>
                    <i className="fas fa-redo me-2"></i>
                    Retry Analysis
                  </>
                )}
              </button>
              <button
                onClick={handleDownloadResults}
                className="btn btn-success"
                disabled={downloadLoading}
              >
                {downloadLoading ? (
                  <>
                    <span className="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
                    Downloading...
                  </>
                ) : (
                  <>
                    <i className="fas fa-download me-2"></i>
                    Download Results
                  </>
                )}
              </button>
              <Link to="/dashboard" className="btn btn-outline-secondary">
                <i className="fas fa-arrow-left me-2"></i>
                Back to Dashboard
              </Link>
            </div>
          </div>
        </div>
      </div>

      <div className="row">
        {/* Image/Video Preview */}
        <div className="col-lg-6 mb-4">
          <div className="card border-0 shadow-sm h-100">
            <div className="card-header bg-white border-0 py-3">
              <h5 className="mb-0">
                <i className="fas fa-eye me-2 text-primary"></i>
                Media Preview
              </h5>
            </div>
            <div className="card-body text-center">
              <style>
                {`
                  .video-preview-container {
                    position: relative;
                    overflow: hidden;
                  }
                  
                  .video-thumbnail-container {
                    position: relative;
                  }
                  
                  .thumbnail-overlay {
                    position: absolute;
                    top: 0;
                    left: 0;
                    right: 0;
                    bottom: 0;
                    background: rgba(0,0,0,0.3);
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    align-items: center;
                    opacity: 0;
                    transition: opacity 0.3s ease;
                  }
                  
                  .video-thumbnail-container:hover .thumbnail-overlay {
                    opacity: 1;
                  }
                  
                  .play-button {
                    background: rgba(0,0,0,0.7);
                    border-radius: 50%;
                    width: 80px;
                    height: 80px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    border: 3px solid rgba(255,255,255,0.8);
                    margin-bottom: 20px;
                  }
                  
                  .thumbnail-info {
                    position: absolute;
                    top: 10px;
                    right: 10px;
                  }
                  
                  .video-info-card {
                    border: 1px solid #dee2e6;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                  }
                  
                  .video-info-summary {
                    border: 1px solid #dee2e6;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                  }
                  
                  .video-details .detail-row {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 4px 0;
                  }
                  
                  .analysis-result {
                    border: 1px solid #dee2e6;
                    background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
                  }
                  
                  .analysis-result .badge {
                    font-size: 14px;
                    padding: 8px 16px;
                  }
                  
                  .badge-sm {
                    font-size: 12px;
                    padding: 4px 8px;
                  }
                `}
              </style>
              {result.fileUrl ? (
                result.fileType?.startsWith('video') || result.filename?.match(/\.(mp4|avi|mov|wmv|flv|webm)$/i) ? (
                  <div className="video-preview-container">
                    {thumbnailLoading ? (
                      <div className="video-info-card p-4 bg-light rounded text-center">
                        <div className="spinner-border text-primary mb-3" role="status">
                          <span className="visually-hidden">Loading...</span>
                        </div>
                        <p className="text-muted">Generating video thumbnail...</p>
                      </div>
                    ) : videoThumbnail ? (
                      <div className="video-thumbnail-container">
                        <img 
                          src={videoThumbnail} 
                          alt={`Video thumbnail - ${result.filename}`}
                          className="img-fluid rounded"
                          style={{ maxHeight: '400px', width: '100%' }}
                        />
                        <div className="thumbnail-overlay">
                          <div className="play-button">
                            <i className="fas fa-play fa-2x text-white"></i>
                          </div>
                          <div className="thumbnail-info">
                            <span className="badge bg-dark">Frame at 1s</span>
                          </div>
                        </div>
                        
                        {/* Video info below thumbnail */}
                        <div className="video-info-summary mt-3 p-3 bg-light rounded">
                          <h6 className="text-dark mb-2">{result.filename}</h6>
                          <div className="row text-center">
                            <div className="col-4">
                              <small className="text-muted">Size</small><br/>
                              <span className="fw-bold">{formatFileSize(result.fileSize)}</span>
                            </div>
                            <div className="col-4">
                              <small className="text-muted">Result</small><br/>
                              <span className={`badge bg-${result.isDeepfake ? 'danger' : 'success'} badge-sm`}>
                                {result.isDeepfake ? 'Deepfake' : 'Real'}
                              </span>
                            </div>
                            <div className="col-4">
                              <small className="text-muted">Confidence</small><br/>
                              <span className="fw-bold">{result.confidence?.toFixed(1)}%</span>
                            </div>
                          </div>
                          
                          <div className="text-center mt-3">
                            <a 
                              href={getFullFileUrl(result.fileUrl)} 
                              target="_blank" 
                              rel="noopener noreferrer"
                              className="btn btn-primary btn-sm me-2"
                            >
                              <i className="fas fa-download me-1"></i>
                              Download
                            </a>
                            <button 
                              className="btn btn-outline-secondary btn-sm"
                              onClick={() => {
                                window.open(getFullFileUrl(result.fileUrl), '_blank');
                              }}
                            >
                              <i className="fas fa-external-link-alt me-1"></i>
                              Open
                            </button>
                          </div>
                        </div>
                      </div>
                    ) : (
                      <div className="video-info-card p-4 bg-light rounded">
                        <div className="text-center mb-3">
                          <i className="fas fa-video fa-4x text-primary"></i>
                        </div>
                        
                        <h6 className="text-dark mb-3">{result.filename}</h6>
                        
                        <div className="video-details">
                          <div className="detail-row mb-2">
                            <span className="text-muted">File Size:</span>
                            <span className="fw-bold">{formatFileSize(result.fileSize)}</span>
                          </div>
                          <div className="detail-row mb-2">
                            <span className="text-muted">File Type:</span>
                            <span className="fw-bold">{result.fileType || 'Video'}</span>
                          </div>
                          <div className="detail-row mb-2">
                            <span className="text-muted">Analysis Time:</span>
                            <span className="fw-bold">{result.analysisTime}</span>
                          </div>
                          <div className="detail-row mb-2">
                            <span className="text-muted">Upload Date:</span>
                            <span className="fw-bold">{result.uploadDate}</span>
                          </div>
                        </div>
                        
                        <div className="analysis-result mt-3 p-3 bg-white rounded">
                          <div className="text-center">
                            <span className="text-muted">Result: </span>
                            <span className={`badge bg-${result.isDeepfake ? 'danger' : 'success'} fs-6`}>
                              {result.isDeepfake ? 'Deepfake Detected' : 'Real Video'}
                            </span>
                          </div>
                          <div className="text-center mt-2">
                            <span className="text-muted">Confidence: </span>
                            <span className="fw-bold">{result.confidence?.toFixed(1)}%</span>
                          </div>
                        </div>
                        
                        <div className="text-center mt-3">
                          <a 
                            href={getFullFileUrl(result.fileUrl)} 
                            target="_blank" 
                            rel="noopener noreferrer"
                            className="btn btn-primary me-2"
                          >
                            <i className="fas fa-download me-2"></i>
                            Download Video
                          </a>
                          <button 
                            className="btn btn-outline-secondary"
                            onClick={() => {
                              // Try to open video in new tab
                              window.open(getFullFileUrl(result.fileUrl), '_blank');
                            }}
                          >
                            <i className="fas fa-external-link-alt me-2"></i>
                            Open Video
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <img 
                    src={getFullFileUrl(result.fileUrl)} 
                    alt={result.filename}
                    className="img-fluid rounded"
                    style={{ maxHeight: '400px' }}
                    onError={(e) => {
                      e.target.style.display = 'none';
                      // Show the fallback display
                      const fallback = document.getElementById('fallback-display');
                      if (fallback) {
                        fallback.classList.remove('d-none');
                      }
                    }}
                  />
                )
              ) : null}
              
              {/* Fallback display when no media URL or media fails to load */}
              <div className={`py-5 ${result.fileUrl ? 'd-none' : ''}`} id="fallback-display">
                <i className={`${getFileTypeIcon(result.filename)} fa-4x text-muted mb-3`}></i>
                <h6 className="text-muted">{result.filename}</h6>
                <p className="text-muted small">
                  File size: {formatFileSize(result.fileSize)}
                </p>
                {result.fileUrl ? (
                  <div className="mt-3">
                    <div className="video-placeholder p-4 bg-light rounded">
                      <i className="fas fa-video fa-3x text-primary mb-3"></i>
                      <h6 className="text-primary">Video File Ready</h6>
                      <p className="text-muted small mb-3">
                        This video has been analyzed for deepfake detection.
                      </p>
                      <a 
                        href={getFullFileUrl(result.fileUrl)} 
                        target="_blank" 
                        rel="noopener noreferrer"
                        className="btn btn-sm btn-outline-primary"
                      >
                        <i className="fas fa-download me-1"></i>
                        Download Video
                      </a>
                    </div>
                  </div>
                ) : (
                  <div className="mt-3">
                    <div className="alert alert-info">
                      <i className="fas fa-info-circle me-2"></i>
                      <strong>File Information</strong><br />
                      <small>This file has been processed for deepfake detection.</small>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Analysis Results */}
        <div className="col-lg-6 mb-4">
          <div className="card border-0 shadow-sm h-100">
            <div className="card-header bg-white border-0 py-3">
              <h5 className="mb-0">
                <i className="fas fa-microscope me-2 text-success"></i>
                Detection Results
              </h5>
            </div>
            <div className="card-body">
              {/* Verdict */}
              <div className="text-center mb-4">
                <h3 className={`text-${getVerdictColor(result.confidence)} mb-2`}>
                  {getVerdict(result.confidence)}
                </h3>
                <span className={`badge bg-${getVerdictColor(result.confidence)} fs-6`}>
                  {getConfidenceText(result.confidence)}
                </span>
              </div>

              {/* Confidence Score */}
              <div className="mb-4">
                <div className="d-flex justify-content-between align-items-center mb-2">
                  <span className="fw-bold">Confidence Score</span>
                  <span className={`badge bg-${getConfidenceColor(result.confidence)}`}>
                    {result.confidence.toFixed(1)}%
                  </span>
                </div>
                <div className="progress" style={{ height: '25px' }}>
                  <div 
                    className={`progress-bar bg-${getConfidenceColor(result.confidence)}`}
                    role="progressbar"
                    style={{ width: `${result.confidence}%` }}
                    aria-valuenow={result.confidence}
                    aria-valuemin="0"
                    aria-valuemax="100"
                  >
                    {result.confidence.toFixed(1)}%
                  </div>
                </div>
                <div className="d-flex justify-content-between mt-1">
                  <small className="text-muted">Real</small>
                  <small className="text-muted">Fake</small>
                </div>
              </div>

              {/* Metadata */}
              <div className="row">
                <div className="col-6 mb-3">
                  <div className="d-flex align-items-center">
                    <i className="fas fa-calendar text-primary me-2"></i>
                    <div>
                      <small className="text-muted d-block">Upload Date</small>
                      <span className="fw-bold">{result.uploadDate}</span>
                    </div>
                  </div>
                </div>
                <div className="col-6 mb-3">
                  <div className="d-flex align-items-center">
                    <i className="fas fa-clock text-info me-2"></i>
                    <div>
                      <small className="text-muted d-block">Upload Time</small>
                      <span className="fw-bold">{result.uploadTime}</span>
                    </div>
                  </div>
                </div>
                <div className="col-6 mb-3">
                  <div className="d-flex align-items-center">
                    <i className="fas fa-stopwatch text-warning me-2"></i>
                    <div>
                      <small className="text-muted d-block">Processing Time</small>
                      <span className="fw-bold">{result.analysisTime}</span>
                    </div>
                  </div>
                </div>
                <div className="col-6 mb-3">
                  <div className="d-flex align-items-center">
                    <i className="fas fa-file text-secondary me-2"></i>
                    <div>
                      <small className="text-muted d-block">File Size</small>
                      <span className="fw-bold">{formatFileSize(result.fileSize)}</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Additional Details */}
      <div className="row">
        <div className="col-12">
          <div className="card border-0 shadow-sm">
            <div className="card-header bg-white border-0 py-3">
              <h5 className="mb-0">
                <i className="fas fa-info-circle me-2 text-info"></i>
                Analysis Details
              </h5>
            </div>
            <div className="card-body">
              <div className="row">
                <div className="col-md-6">
                  <h6>File Information</h6>
                  <ul className="list-unstyled">
                    <li><strong>Filename:</strong> {result.filename}</li>
                    <li><strong>File Type:</strong> {result.fileType}</li>
                    <li><strong>File Size:</strong> {formatFileSize(result.fileSize)}</li>
                    <li><strong>Analysis ID:</strong> {result.id}</li>
                  </ul>
                </div>
                <div className="col-md-6">
                  <h6>Detection Information</h6>
                  <ul className="list-unstyled">
                    <li><strong>Deepfake Probability:</strong> {result.confidence.toFixed(1)}%</li>
                    <li><strong>Verdict:</strong> 
                      <span className={`badge bg-${getVerdictColor(result.confidence)} ms-2`}>
                        {getVerdict(result.confidence)}
                      </span>
                    </li>
                    <li><strong>Processing Time:</strong> {result.analysisTime}</li>
                    <li><strong>Analysis Date:</strong> {new Date(result.createdAt).toLocaleString()}</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Results; 