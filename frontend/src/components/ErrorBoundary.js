import React from 'react';

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error) {
    // Update state so the next render will show the fallback UI
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    // Log error details for debugging
    console.error('ErrorBoundary caught an error:', error, errorInfo);
    this.setState({
      error: error,
      errorInfo: errorInfo
    });
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null, errorInfo: null });
    // Clear any throttled requests
    if (window.requestThrottle) {
      window.requestThrottle.reset();
    }
    // Optionally reload the page
    if (this.props.autoReload) {
      window.location.reload();
    }
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="container mt-5">
          <div className="row justify-content-center">
            <div className="col-md-8">
              <div className="card border-danger">
                <div className="card-header bg-danger text-white">
                  <h5 className="mb-0">
                    <i className="fas fa-exclamation-triangle me-2"></i>
                    Something went wrong
                  </h5>
                </div>
                <div className="card-body">
                  <p className="text-muted">
                    An unexpected error occurred. This might be due to network issues or too many requests.
                  </p>
                  
                  {process.env.NODE_ENV === 'development' && this.state.error && (
                    <div className="mt-3">
                      <h6>Error Details (Development Only):</h6>
                      <pre className="bg-light p-2 rounded">
                        {this.state.error.toString()}
                      </pre>
                      {this.state.errorInfo && (
                        <details className="mt-2">
                          <summary>Component Stack</summary>
                          <pre className="bg-light p-2 rounded mt-2">
                            {this.state.errorInfo.componentStack}
                          </pre>
                        </details>
                      )}
                    </div>
                  )}
                  
                  <div className="mt-4 d-flex gap-2">
                    <button 
                      onClick={this.handleReset}
                      className="btn btn-primary"
                    >
                      <i className="fas fa-redo me-2"></i>
                      Try Again
                    </button>
                    <button 
                      onClick={() => window.location.href = '/dashboard'}
                      className="btn btn-outline-secondary"
                    >
                      <i className="fas fa-home me-2"></i>
                      Go to Dashboard
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;