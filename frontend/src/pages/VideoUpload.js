import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import VideoUpload from '../components/video/VideoUpload';
import { ArrowLeft, Video, CheckCircle } from 'lucide-react';
import './VideoUpload.css';

const VideoUploadPage = () => {
  const [uploadComplete, setUploadComplete] = useState(false);
  const [uploadData, setUploadData] = useState(null);
  const navigate = useNavigate();

  const handleUploadComplete = (data) => {
    setUploadComplete(true);
    setUploadData(data);
    console.log('Video upload completed:', data);
  };

  const handleProgressUpdate = (progress) => {
    console.log('Upload progress:', progress);
  };

  const handleViewResults = () => {
    if (uploadData) {
      navigate(`/results/${uploadData.file_id}`);
    }
  };

  const handleBackToDashboard = () => {
    navigate('/dashboard');
  };

  return (
    <div className="video-upload-page">
      <div className="page-header">
        <button className="back-button" onClick={handleBackToDashboard}>
          <ArrowLeft size={20} />
          Back to Dashboard
        </button>
        <h1>Video Deepfake Analysis</h1>
        <p>Upload a video file to analyze for deepfake detection</p>
      </div>

      <div className="upload-container">
        {!uploadComplete ? (
          <VideoUpload 
            onUploadComplete={handleUploadComplete}
            onProgressUpdate={handleProgressUpdate}
          />
        ) : (
          <div className="upload-success">
            <CheckCircle size={64} className="success-icon" />
            <h2>Video Upload Successful!</h2>
            <p>Your video has been uploaded and is being processed.</p>
            
            {uploadData && (
              <div className="upload-details">
                <h3>Upload Details:</h3>
                <div className="details-grid">
                  <div className="detail-item">
                    <span className="label">Filename:</span>
                    <span className="value">{uploadData.filename}</span>
                  </div>
                  <div className="detail-item">
                    <span className="label">Duration:</span>
                    <span className="value">
                      {Math.floor(uploadData.video_metadata.duration / 60)}:
                      {Math.floor(uploadData.video_metadata.duration % 60).toString().padStart(2, '0')}
                    </span>
                  </div>
                  <div className="detail-item">
                    <span className="label">Resolution:</span>
                    <span className="value">
                      {uploadData.video_metadata.width} × {uploadData.video_metadata.height}
                    </span>
                  </div>
                  <div className="detail-item">
                    <span className="label">FPS:</span>
                    <span className="value">{uploadData.video_metadata.fps.toFixed(2)}</span>
                  </div>
                  <div className="detail-item">
                    <span className="label">Total Frames:</span>
                    <span className="value">{uploadData.video_metadata.frame_count.toLocaleString()}</span>
                  </div>
                </div>
              </div>
            )}
            
            <div className="action-buttons">
              <button className="primary-button" onClick={handleViewResults}>
                <Video size={20} />
                View Analysis Results
              </button>
              <button className="secondary-button" onClick={() => setUploadComplete(false)}>
                Upload Another Video
              </button>
            </div>
          </div>
        )}
      </div>

      <div className="info-section">
        <h3>How Video Analysis Works</h3>
        <div className="info-grid">
          <div className="info-item">
            <h4>Frame Extraction</h4>
            <p>Our system intelligently samples frames from your video to analyze for deepfake detection.</p>
          </div>
          <div className="info-item">
            <h4>Multi-Model Analysis</h4>
            <p>Each frame is analyzed using our ensemble of deepfake detection models for maximum accuracy.</p>
          </div>
          <div className="info-item">
            <h4>Temporal Consistency</h4>
            <p>We analyze patterns across frames to detect inconsistencies that indicate deepfake manipulation.</p>
          </div>
          <div className="info-item">
            <h4>Detailed Results</h4>
            <p>Get frame-by-frame analysis with confidence scores and temporal pattern detection.</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default VideoUploadPage; 