import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { analysisAPI } from '../services/api';

const Results = () => {
  const { fileId } = useParams();
  const navigate = useNavigate();
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedResult, setSelectedResult] = useState(null);
  const [showDetails, setShowDetails] = useState(false);
  const [individualResult, setIndividualResult] = useState(null);

  useEffect(() => {
    const fetchResults = async () => {
      try {
        if (fileId) {
          // Fetch individual result
          const response = await analysisAPI.getResults(parseInt(fileId));
          console.log('Individual result response:', response);
          
          const transformedResult = {
            id: response.file_id,
            filename: response.filename,
            uploadDate: new Date(response.created_at).toLocaleDateString(),
            status: 'completed',
            confidence: response.detection_result.confidence_score,
            isDeepfake: response.detection_result.is_deepfake,
            analysisTime: `${response.detection_result.processing_time_seconds.toFixed(1)}s`,
            createdAt: response.created_at,
            detectionResult: response.detection_result
          };
          
          setIndividualResult(transformedResult);
          setResults([transformedResult]);
        } else {
          // Fetch all results
          const response = await analysisAPI.getAllResults();
          console.log('Analysis results response:', response);
          
          // Transform the backend data to match frontend expectations
          const transformedResults = response.map(item => ({
            id: item.file_id,
            filename: item.filename,
            uploadDate: new Date(item.created_at).toLocaleDateString(),
            status: 'completed',
            confidence: item.detection_result.confidence_score,
            isDeepfake: item.detection_result.is_deepfake,
            analysisTime: `${item.detection_result.processing_time_seconds.toFixed(1)}s`,
            createdAt: item.created_at,
            detectionResult: item.detection_result
          }));
          
          setResults(transformedResults);
        }
      } catch (error) {
        console.error('Error fetching results:', error);
        setResults([]);
        setIndividualResult(null);
      } finally {
        setLoading(false);
      }
    };

    fetchResults();
  }, [fileId]);

  const getStatusBadge = (status) => {
    const badges = {
      completed: 'success',
      processing: 'warning',
      failed: 'danger'
    };
    return `badge bg-${badges[status] || 'secondary'}`;
  };

  const getResultBadge = (isDeepfake) => {
    return isDeepfake 
      ? 'badge bg-danger' 
      : 'badge bg-success';
  };

  if (loading) {
    return (
      <div className="container mt-5">
        <div className="row justify-content-center">
          <div className="col-md-8 text-center">
            <div className="spinner-border text-primary" role="status">
              <span className="visually-hidden">Loading...</span>
            </div>
            <p className="mt-3">Loading analysis results...</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="container mt-5">
      <div className="row">
        <div className="col-12">
          <div className="d-flex justify-content-between align-items-center mb-4">
            <h1>{fileId ? 'Analysis Result' : 'Analysis Results'}</h1>
            {fileId && (
              <button 
                className="btn btn-outline-primary"
                onClick={() => navigate('/results')}
              >
                <i className="fas fa-arrow-left me-2"></i>
                Back to All Results
              </button>
            )}
          </div>
          
          {results.length === 0 ? (
            <div className="alert alert-info">
              <h5>No results found</h5>
              <p>Upload your first media file to see analysis results here.</p>
            </div>
          ) : (
            <div className="card">
              <div className="card-header">
                <h5 className="mb-0">Recent Analyses</h5>
              </div>
              <div className="card-body p-0">
                <div className="table-responsive">
                  <table className="table table-hover mb-0">
                    <thead className="table-light">
                      <tr>
                        <th>File</th>
                        <th>Upload Date</th>
                        <th>Status</th>
                        <th>Result</th>
                        <th>Confidence</th>
                        <th>Analysis Time</th>
                        <th>Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {results.map((result) => (
                        <tr key={result.id}>
                          <td>
                            <strong>{result.filename}</strong>
                          </td>
                          <td>{result.uploadDate}</td>
                          <td>
                            <span className={getStatusBadge(result.status)}>
                              {result.status}
                            </span>
                          </td>
                          <td>
                            <span className={getResultBadge(result.isDeepfake)}>
                              {result.isDeepfake ? 'Deepfake Detected' : 'Authentic'}
                            </span>
                          </td>
                          <td>
                            <div className="progress" style={{ width: '80px', height: '20px' }}>
                              <div 
                                className={`progress-bar ${result.isDeepfake ? 'bg-danger' : 'bg-success'}`}
                                style={{ width: `${result.confidence}%` }}
                              >
                                {result.confidence}%
                              </div>
                            </div>
                          </td>
                          <td>{result.analysisTime}</td>
                          <td>
                            <button 
                              className="btn btn-sm btn-outline-primary me-2"
                              onClick={() => {
                                setSelectedResult(result);
                                setShowDetails(true);
                              }}
                            >
                              View Details
                            </button>
                            <button className="btn btn-sm btn-outline-secondary">
                              Download
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="row mt-4">
        <div className="col-md-6">
          <div className="card">
            <div className="card-header">
              <h5 className="mb-0">Statistics</h5>
            </div>
            <div className="card-body">
              <div className="row text-center">
                <div className="col-6">
                  <h3 className="text-primary">{results.length}</h3>
                  <p className="text-muted">Total Analyses</p>
                </div>
                <div className="col-6">
                  <h3 className="text-success">
                    {results.filter(r => !r.isDeepfake).length}
                  </h3>
                  <p className="text-muted">Authentic Files</p>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        <div className="col-md-6">
          <div className="card">
            <div className="card-header">
              <h5 className="mb-0">Quick Actions</h5>
            </div>
            <div className="card-body">
              <button className="btn btn-primary me-2">
                Upload New File
              </button>
              <button className="btn btn-outline-secondary">
                Export Results
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Analysis Details Modal */}
      {showDetails && selectedResult && (
        <div className="modal fade show" style={{ display: 'block' }} tabIndex="-1">
          <div className="modal-dialog modal-lg">
            <div className="modal-content">
              <div className="modal-header">
                <h5 className="modal-title">Analysis Details - {selectedResult.filename}</h5>
                <button 
                  type="button" 
                  className="btn-close" 
                  onClick={() => setShowDetails(false)}
                ></button>
              </div>
              <div className="modal-body">
                <div className="row">
                  <div className="col-md-6">
                    <h6>Basic Information</h6>
                    <ul className="list-unstyled">
                      <li><strong>File:</strong> {selectedResult.filename}</li>
                      <li><strong>Upload Date:</strong> {selectedResult.uploadDate}</li>
                      <li><strong>Analysis Time:</strong> {selectedResult.analysisTime}</li>
                      <li><strong>Confidence Score:</strong> {selectedResult.confidence}%</li>
                      <li><strong>Result:</strong> 
                        <span className={`badge ${selectedResult.isDeepfake ? 'bg-danger' : 'bg-success'} ms-2`}>
                          {selectedResult.isDeepfake ? 'Deepfake Detected' : 'Authentic'}
                        </span>
                      </li>
                    </ul>
                  </div>
                  <div className="col-md-6">
                    <h6>Detection Features</h6>
                    {selectedResult.detectionResult?.analysis_metadata?.detection_features && (
                      <ul className="list-unstyled">
                        <li><strong>Faces Found:</strong> {selectedResult.detectionResult.analysis_metadata.detection_features.face_detection.faces_found}</li>
                        <li><strong>Noise Level:</strong> {(selectedResult.detectionResult.analysis_metadata.detection_features.texture_analysis.noise_level * 100).toFixed(1)}%</li>
                        <li><strong>Edge Sharpness:</strong> {(selectedResult.detectionResult.analysis_metadata.detection_features.edge_detection.edge_sharpness * 100).toFixed(1)}%</li>
                      </ul>
                    )}
                  </div>
                </div>
                
                {selectedResult.detectionResult?.analysis_metadata?.result_summary && (
                  <div className="mt-3">
                    <h6>Analysis Summary</h6>
                    <div className="alert alert-info">
                      <strong>Primary Indicator:</strong> {selectedResult.detectionResult.analysis_metadata.result_summary.primary_indicator}<br/>
                      <strong>Recommendation:</strong> {selectedResult.detectionResult.analysis_metadata.result_summary.recommendation}
                    </div>
                  </div>
                )}
              </div>
              <div className="modal-footer">
                <button 
                  type="button" 
                  className="btn btn-secondary" 
                  onClick={() => setShowDetails(false)}
                >
                  Close
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Modal Backdrop */}
      {showDetails && (
        <div className="modal-backdrop fade show"></div>
      )}
    </div>
  );
};

export default Results; 