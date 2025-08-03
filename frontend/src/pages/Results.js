import React, { useState, useEffect, useCallback, useRef } from 'react';
import { useParams, Link } from 'react-router-dom';
import { analysisAPI } from '../services/api';
import { useApiError } from '../hooks/useApiError';

const Results = () => {
  const { fileId } = useParams();
  const { error, handleError, clearError } = useApiError();
  
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(true);
  const [retryLoading, setRetryLoading] = useState(false);
  const [downloadLoading, setDownloadLoading] = useState(false);
  const [message, setMessage] = useState('');
  
  // Use ref to track if we've already fetched for this fileId
  const hasFetchedRef = useRef(false);
  const currentFileIdRef = useRef(null);

  const fetchResult = useCallback(async () => {
    // Prevent multiple simultaneous fetches and duplicate fetches for same fileId
    if (loading || (hasFetchedRef.current && currentFileIdRef.current === fileId)) {
      return;
    }
    
    setLoading(true);
    clearError();
    
    // Mark that we're fetching for this fileId
    hasFetchedRef.current = true;
    currentFileIdRef.current = fileId;
    
    try {
      // Add timeout to prevent hanging
      const timeoutPromise = new Promise((_, reject) => 
        setTimeout(() => reject(new Error('Request timeout')), 10000)
      );
      
      const responsePromise = analysisAPI.getResults(parseInt(fileId));
      const response = await Promise.race([responsePromise, timeoutPromise]);
      console.log('Result response:', response);
      
      // Debug: Log the full response to see what fields are available
      console.log('Full API response:', response);
      
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
      
      console.log('Transformed result:', transformedResult);
      
      // Note: File URL testing removed to prevent infinite loops
      // File URLs are constructed but not tested for accessibility
      
      setResult(transformedResult);
    } catch (err) {
      handleError(err);
      console.error('Error fetching result:', err);
    } finally {
      setLoading(false);
    }
  }, [fileId, clearError, handleError, loading]);

  useEffect(() => {
    if (fileId) {
      // Reset fetch flag when fileId changes
      if (currentFileIdRef.current !== fileId) {
        hasFetchedRef.current = false;
        currentFileIdRef.current = fileId;
      }
      fetchResult();
    }
  }, [fileId, fetchResult]);

  const handleRetryAnalysis = async () => {
    setRetryLoading(true);
    clearError();
    setMessage('');
    
    // Reset fetch flag to allow new fetch after retry
    hasFetchedRef.current = false;
    
    try {
      console.log('Retrying analysis for file ID:', fileId);
      const response = await analysisAPI.analyzeFile(fileId);
      console.log('Retry analysis response:', response);
      
      // Show success message
      setMessage('Analysis restarted successfully! Checking status...');
      
      // Poll for completion (check every 3 seconds for up to 2 minutes)
      let attempts = 0;
      const maxAttempts = 40; // 2 minutes total
      
      const pollForCompletion = async () => {
        try {
          const resultResponse = await analysisAPI.getResults(parseInt(fileId));
          console.log('Poll result:', resultResponse);
          
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
      // Try different possible URL patterns
      const baseUrl = process.env.REACT_APP_API_URL || 'http://localhost:8000';
      return `${baseUrl}/api/files/${fileId}`;
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
                <h5>Loading Analysis Results...</h5>
                <p className="text-muted">Please wait while we fetch the analysis data</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

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
              {result.fileUrl ? (
                result.fileType?.startsWith('video') || result.filename?.match(/\.(mp4|avi|mov|wmv|flv|webm)$/i) ? (
                  <video 
                    controls 
                    className="img-fluid rounded"
                    style={{ maxHeight: '400px' }}
                    onError={(e) => {
                      console.error('Video loading error:', e);
                      console.error('Video URL:', getFullFileUrl(result.fileUrl));
                      console.error('File info:', {
                        fileUrl: result.fileUrl,
                        fileType: result.fileType,
                        fileSize: result.fileSize,
                        filename: result.filename
                      });
                      e.target.style.display = 'none';
                      // Show the fallback display
                      const fallback = document.getElementById('fallback-display');
                      if (fallback) {
                        fallback.classList.remove('d-none');
                      }
                    }}
                  >
                    <source src={getFullFileUrl(result.fileUrl)} type={result.fileType} />
                    Your browser does not support the video tag.
                  </video>
                ) : (
                  <img 
                    src={getFullFileUrl(result.fileUrl)} 
                    alt={result.filename}
                    className="img-fluid rounded"
                    style={{ maxHeight: '400px' }}
                    onError={(e) => {
                      console.error('Image loading error:', e);
                      console.error('Image URL:', getFullFileUrl(result.fileUrl));
                      console.error('File info:', {
                        fileUrl: result.fileUrl,
                        fileType: result.fileType,
                        fileSize: result.fileSize,
                        filename: result.filename
                      });
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
                {process.env.NODE_ENV === 'development' && (
                  <div className="alert alert-info small mb-3">
                    <strong>Debug Info:</strong><br />
                    File URL: {result.fileUrl || 'Not available'}<br />
                    File Type: {result.fileType || 'Not available'}<br />
                    File Size: {result.fileSize || 'Not available'}
                  </div>
                )}
                <i className={`${getFileTypeIcon(result.filename)} fa-4x text-muted mb-3`}></i>
                <h6 className="text-muted">{result.filename}</h6>
                <p className="text-muted small">
                  File size: {formatFileSize(result.fileSize)}
                </p>
                {result.fileUrl && (
                  <div className="mt-3">
                    <small className="text-muted">
                      <i className="fas fa-info-circle me-1"></i>
                      Media preview not available
                    </small>
                    <br />
                    <a 
                      href={getFullFileUrl(result.fileUrl)} 
                      target="_blank" 
                      rel="noopener noreferrer"
                      className="btn btn-sm btn-outline-primary mt-2"
                    >
                      <i className="fas fa-external-link-alt me-1"></i>
                      Open Media
                    </a>
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