import React, { useState, useEffect, useCallback } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../components/AuthContext';
import { analysisAPI } from '../services/api';
import { useApiError } from '../hooks/useApiError';

const Dashboard = () => {
  const { user, logout, isAuthenticated, authChecked } = useAuth();
  const navigate = useNavigate();
  const { error, handleError, clearError } = useApiError();
  
  const [loading, setLoading] = useState(true);
  const [stats, setStats] = useState({
    totalUploads: 0,
    recentAnalyses: 0,
    completedAnalyses: 0,
    pendingAnalyses: 0
  });
  const [recentUploads, setRecentUploads] = useState([]);
  const [refreshLoading, setRefreshLoading] = useState(false);

  const fetchDashboardData = useCallback(async () => {
    setLoading(true);
    clearError();
    
    try {
      // Fetch analysis statistics (fallback to empty stats if endpoint doesn't exist)
      try {
        const statsData = await analysisAPI.getStats();
        setStats(statsData);
      } catch (statsErr) {
        console.warn('Stats endpoint not available, using default values:', statsErr);
        // Keep default stats values
      }
      
      // Fetch recent analysis history
      console.log('Fetching history...');
      const historyData = await analysisAPI.getAllResults(1, 5);
      console.log('History data received:', historyData);
      setRecentUploads(historyData || []);
    } catch (err) {
      handleError(err);
      console.error('Failed to fetch dashboard data:', err);
    } finally {
      setLoading(false);
    }
  }, [clearError, handleError]);

  const handleRefresh = async () => {
    setRefreshLoading(true);
    await fetchDashboardData();
    setRefreshLoading(false);
  };

  useEffect(() => {
    // Only fetch data if authenticated AND auth check is complete
    if (isAuthenticated && authChecked) {
      fetchDashboardData();
    }
  }, [isAuthenticated, authChecked, fetchDashboardData]);

  const handleLogout = async () => {
    try {
      await logout();
      navigate('/login');
    } catch (err) {
      handleError(err);
      console.error('Logout error:', err);
    }
  };

  const handleQuickUpload = () => {
    navigate('/upload');
  };

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const getFileTypeIcon = (fileName) => {
    const extension = fileName.split('.').pop()?.toLowerCase();
    if (['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'].includes(extension)) {
      return 'fas fa-image text-primary';
    } else if (['mp4', 'avi', 'mov', 'wmv', 'flv', 'webm'].includes(extension)) {
      return 'fas fa-video text-danger';
    } else {
      return 'fas fa-file text-secondary';
    }
  };

  // Redirect to login if not authenticated
  if (!isAuthenticated && authChecked) {
    navigate('/login');
    return null;
  }

  // Show loading while auth is being checked
  if (!authChecked || loading) {
    return (
      <div className="container mt-5">
        <div className="row justify-content-center">
          <div className="col-md-8 text-center">
            <div className="card shadow">
              <div className="card-body p-5">
                <div className="spinner-border text-primary mb-3" role="status">
                  <span className="visually-hidden">Loading...</span>
                </div>
                <h5>Loading Dashboard...</h5>
                <p className="text-muted">Please wait while we fetch your data</p>
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

      {/* Header Section */}
      <div className="row mb-4">
        <div className="col-12">
          <div className="d-flex justify-content-between align-items-center">
            <div>
              <h1 className="h2 mb-1">
                <i className="fas fa-tachometer-alt me-2 text-primary"></i>
                Dashboard
              </h1>
              <p className="text-muted mb-0">
                Welcome back, <strong>{user?.email}</strong>! Here's your deepfake detection overview.
              </p>
            </div>
            <div className="d-flex gap-2">
              <button
                onClick={handleQuickUpload}
                className="btn btn-primary"
              >
                <i className="fas fa-plus me-2"></i>
                Quick Upload
              </button>
              <button
                onClick={handleRefresh}
                className="btn btn-outline-primary"
                disabled={refreshLoading}
              >
                {refreshLoading ? (
                  <>
                    <span className="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
                    Refreshing...
                  </>
                ) : (
                  <>
                    <i className="fas fa-sync-alt me-2"></i>
                    Refresh
                  </>
                )}
              </button>
              <button
                onClick={handleLogout}
                className="btn btn-outline-secondary"
              >
                <i className="fas fa-sign-out-alt me-2"></i>
                Logout
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Quick Stats Cards */}
      <div className="row mb-4">
        <div className="col-md-3 mb-3">
          <div className="card border-0 shadow-sm h-100">
            <div className="card-body">
              <div className="d-flex align-items-center">
                <div className="flex-shrink-0">
                  <div className="bg-primary bg-opacity-10 rounded-circle p-3">
                    <i className="fas fa-upload fa-2x text-primary"></i>
                  </div>
                </div>
                <div className="flex-grow-1 ms-3">
                  <h4 className="mb-1">{stats.totalUploads}</h4>
                  <p className="text-muted mb-0">Total Uploads</p>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="col-md-3 mb-3">
          <div className="card border-0 shadow-sm h-100">
            <div className="card-body">
              <div className="d-flex align-items-center">
                <div className="flex-shrink-0">
                  <div className="bg-success bg-opacity-10 rounded-circle p-3">
                    <i className="fas fa-check-circle fa-2x text-success"></i>
                  </div>
                </div>
                <div className="flex-grow-1 ms-3">
                  <h4 className="mb-1">{stats.completedAnalyses}</h4>
                  <p className="text-muted mb-0">Completed</p>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="col-md-3 mb-3">
          <div className="card border-0 shadow-sm h-100">
            <div className="card-body">
              <div className="d-flex align-items-center">
                <div className="flex-shrink-0">
                  <div className="bg-warning bg-opacity-10 rounded-circle p-3">
                    <i className="fas fa-clock fa-2x text-warning"></i>
                  </div>
                </div>
                <div className="flex-grow-1 ms-3">
                  <h4 className="mb-1">{stats.pendingAnalyses}</h4>
                  <p className="text-muted mb-0">Pending</p>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="col-md-3 mb-3">
          <div className="card border-0 shadow-sm h-100">
            <div className="card-body">
              <div className="d-flex align-items-center">
                <div className="flex-shrink-0">
                  <div className="bg-info bg-opacity-10 rounded-circle p-3">
                    <i className="fas fa-chart-line fa-2x text-info"></i>
                  </div>
                </div>
                <div className="flex-grow-1 ms-3">
                  <h4 className="mb-1">{stats.recentAnalyses}</h4>
                  <p className="text-muted mb-0">Recent Analyses</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>



      

      {/* Main Content Row */}
      <div className="row">
        {/* Recent Uploads */}
        <div className="col-lg-8 mb-4">
          <div className="card border-0 shadow-sm">
            <div className="card-header bg-white border-0 py-3">
              <div className="d-flex justify-content-between align-items-center">
                <h5 className="mb-0">
                  <i className="fas fa-history me-2 text-primary"></i>
                  Recent Uploads
                </h5>
                <Link to="/results" className="btn btn-sm btn-outline-primary">
                  View All
                </Link>
              </div>
            </div>
            <div className="card-body p-0">
              {recentUploads.length === 0 ? (
                <div className="text-center py-5">
                  <i className="fas fa-inbox fa-3x text-muted mb-3"></i>
                  <h6 className="text-muted">No uploads yet</h6>
                  <p className="text-muted mb-3">Start by uploading your first media file for analysis</p>
                  <button
                    onClick={handleQuickUpload}
                    className="btn btn-primary"
                  >
                    <i className="fas fa-plus me-2"></i>
                    Upload First File
                  </button>
                </div>
              ) : (
                <div className="list-group list-group-flush">
                  {recentUploads.map((upload, index) => (
                    <div key={upload.file_id || index} className="list-group-item border-0 py-3">
                      <div className="d-flex align-items-center">
                        <div className="flex-shrink-0 me-3">
                          <i className={`${getFileTypeIcon(upload.filename)} fa-2x`}></i>
                        </div>
                        <div className="flex-grow-1">
                          <h6 className="mb-1 text-truncate">{upload.filename}</h6>
                          <div className="d-flex align-items-center gap-3">
                            <small className="text-muted">
                              <i className="fas fa-clock me-1"></i>
                              {formatDate(upload.created_at)}
                            </small>
                            <div>
                              <span className="badge bg-success">Completed</span>
                            </div>
                          </div>
                        </div>
                        <div className="flex-shrink-0">
                          <Link 
                            to={`/results/${upload.file_id}`} 
                            className="btn btn-sm btn-outline-primary"
                          >
                            <i className="fas fa-eye me-1"></i>
                            View
                          </Link>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Quick Actions Sidebar */}
        <div className="col-lg-4 mb-4">
          <div className="card border-0 shadow-sm">
            <div className="card-header bg-white border-0 py-3">
              <h5 className="mb-0">
                <i className="fas fa-bolt me-2 text-warning"></i>
                Quick Actions
              </h5>
            </div>
            <div className="card-body">
              <div className="d-grid gap-3">
                <button
                  onClick={handleQuickUpload}
                  className="btn btn-primary btn-lg"
                >
                  <i className="fas fa-upload me-2"></i>
                  Upload New File
                </button>
                

                
                <Link to="/results" className="btn btn-outline-success">
                  <i className="fas fa-chart-bar me-2"></i>
                  View All Results
                </Link>
              </div>
            </div>
          </div>

          {/* System Status */}
          <div className="card border-0 shadow-sm mt-4">
            <div className="card-header bg-white border-0 py-3">
              <h5 className="mb-0">
                <i className="fas fa-server me-2 text-info"></i>
                System Status
              </h5>
            </div>
            <div className="card-body">
              <div className="d-flex justify-content-between align-items-center mb-2">
                <span>Detection Engine</span>
                <span className="badge bg-success">Online</span>
              </div>
              <div className="d-flex justify-content-between align-items-center mb-2">
                <span>API Service</span>
                <span className="badge bg-success">Online</span>
              </div>
              <div className="d-flex justify-content-between align-items-center">
                <span>Database</span>
                <span className="badge bg-success">Online</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Navigation Cards */}
      <div className="row mt-4">
        <div className="col-12">
          <div className="card border-0 shadow-sm">
            <div className="card-header bg-white border-0 py-3">
              <h5 className="mb-0">
                <i className="fas fa-compass me-2 text-primary"></i>
                Navigation
              </h5>
            </div>
            <div className="card-body">
              <div className="row">
                <div className="col-md-3 mb-3">
                  <Link to="/upload" className="text-decoration-none">
                    <div className="card border-0 bg-light h-100">
                      <div className="card-body text-center">
                        <i className="fas fa-upload fa-2x text-primary mb-2"></i>
                        <h6 className="card-title">Upload</h6>
                        <p className="card-text small text-muted">Upload media files for analysis</p>
                      </div>
                    </div>
                  </Link>
                </div>
                
                <div className="col-md-3 mb-3">
                  <Link to="/results" className="text-decoration-none">
                    <div className="card border-0 bg-light h-100">
                      <div className="card-body text-center">
                        <i className="fas fa-chart-bar fa-2x text-success mb-2"></i>
                        <h6 className="card-title">Results</h6>
                        <p className="card-text small text-muted">View analysis results</p>
                      </div>
                    </div>
                  </Link>
                </div>
                
                <div className="col-md-3 mb-3">
                  <Link to="/results" className="text-decoration-none">
                    <div className="card border-0 bg-light h-100">
                      <div className="card-body text-center">
                        <i className="fas fa-history fa-2x text-info mb-2"></i>
                        <h6 className="card-title">History</h6>
                        <p className="card-text small text-muted">Analysis history</p>
                      </div>
                    </div>
                  </Link>
                </div>
                
                <div className="col-md-3 mb-3">
                  <div className="card border-0 bg-light h-100">
                    <div className="card-body text-center">
                      <i className="fas fa-cog fa-2x text-warning mb-2"></i>
                      <h6 className="card-title">Settings</h6>
                      <p className="card-text small text-muted">Account settings</p>
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

export default Dashboard; 