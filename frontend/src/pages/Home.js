import React from 'react';
import { Link } from 'react-router-dom';

const Home = () => {
  return (
    <div className="container mt-5">
      <div className="row justify-content-center">
        <div className="col-md-8 text-center">
          <h1 className="display-4 mb-4">Welcome to Deepfake Detection</h1>
          <p className="lead mb-4">
            Advanced AI-powered platform for detecting deepfake images and videos. 
            Upload your media files and get instant analysis results.
          </p>
          <div className="row mt-5">
            <div className="col-md-6 mb-3">
              <div className="card h-100">
                <div className="card-body">
                  <h5 className="card-title">🔍 Analyze Media</h5>
                  <p className="card-text">
                    Upload images or videos to detect potential deepfakes using our advanced AI models.
                  </p>
                  <Link to="/upload" className="btn btn-primary">
                    Start Analysis
                  </Link>
                </div>
              </div>
            </div>
            <div className="col-md-6 mb-3">
              <div className="card h-100">
                <div className="card-body">
                  <h5 className="card-title">📊 View Results</h5>
                  <p className="card-text">
                    Check your previous analysis results and track detection history.
                  </p>
                  <Link to="/results" className="btn btn-outline-primary">
                    View Results
                  </Link>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Home; 