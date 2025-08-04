import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, Video, AlertCircle, CheckCircle, Clock } from 'lucide-react';
import { getApiConfig } from '../../config/api';
import './VideoUpload.css';

const VideoUpload = ({ onUploadComplete, onProgressUpdate }) => {
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [currentTask, setCurrentTask] = useState(null);
  const [error, setError] = useState(null);

  const onDrop = useCallback(async (acceptedFiles) => {
    const file = acceptedFiles[0];
    if (!file) return;

    // Debug: Log file information
    console.log('File upload attempt:', {
      name: file.name,
      type: file.type,
      size: file.size,
      lastModified: file.lastModified
    });

    // Validate file type - more flexible validation
    const allowedTypes = [
      'video/mp4', 'video/avi', 'video/mov', 'video/mkv', 'video/wmv', 'video/flv',
      'video/x-msvideo', 'video/quicktime', 'video/x-matroska', 'video/x-ms-wmv'
    ];
    const allowedExtensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'];
    
    // Check MIME type first
    const isValidMimeType = allowedTypes.includes(file.type);
    
    // If MIME type check fails, check file extension
    const fileExtension = file.name.toLowerCase().substring(file.name.lastIndexOf('.'));
    const isValidExtension = allowedExtensions.includes(fileExtension);
    
    if (!isValidMimeType && !isValidExtension) {
      setError(`Unsupported file type: ${file.type || fileExtension}. Please upload MP4, AVI, MOV, MKV, WMV, or FLV files.`);
      return;
    }

    // Validate file size (500MB max)
    const maxSize = 500 * 1024 * 1024; // 500MB
    if (file.size > maxSize) {
      setError('File too large. Maximum size is 500MB.');
      return;
    }

    setError(null);
    setUploading(true);
    setUploadProgress(0);

    try {
      // Create form data
      const formData = new FormData();
      formData.append('video_file', file);

      // Get auth token
      const token = localStorage.getItem('token');
      if (!token) {
        throw new Error('Authentication required');
      }

      // Get API configuration
      const apiConfig = getApiConfig();
      
      // Upload video
      const uploadResponse = await fetch(`${apiConfig.BASE_URL}/api/video/upload`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
        body: formData,
      });

      if (!uploadResponse.ok) {
        const errorData = await uploadResponse.json();
        throw new Error(errorData.detail || 'Upload failed');
      }

      const uploadData = await uploadResponse.json();
      setCurrentTask(uploadData.task_id);

      // Start progress polling
      pollProgress(uploadData.task_id);

      if (onUploadComplete) {
        onUploadComplete(uploadData);
      }

    } catch (err) {
      setError(err.message);
      setUploading(false);
    }
  }, [onUploadComplete]);

  const pollProgress = async (taskId) => {
    const token = localStorage.getItem('token');
    const apiConfig = getApiConfig();
    
    const pollInterval = setInterval(async () => {
      try {
        const response = await fetch(`${apiConfig.BASE_URL}/api/video/progress/${taskId}`, {
          headers: {
            'Authorization': `Bearer ${token}`,
          },
        });

        if (response.ok) {
          const progressData = await response.json();
          setUploadProgress(progressData.progress_percent);

          if (onProgressUpdate) {
            onProgressUpdate(progressData);
          }

          if (progressData.status === 'completed') {
            setUploading(false);
            setCurrentTask(null);
            clearInterval(pollInterval);
          } else if (progressData.status === 'failed') {
            setError('Video processing failed');
            setUploading(false);
            setCurrentTask(null);
            clearInterval(pollInterval);
          }
        }
      } catch (err) {
        console.error('Error polling progress:', err);
      }
    }, 2000); // Poll every 2 seconds
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'video/*': ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'],
      'video/mp4': ['.mp4'],
      'video/avi': ['.avi'],
      'video/quicktime': ['.mov'],
      'video/x-matroska': ['.mkv'],
      'video/x-ms-wmv': ['.wmv'],
      'video/x-flv': ['.flv']
    },
    multiple: false,
    disabled: uploading
  });

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="video-upload-container">
      <div className="video-upload-header">
        <h2>Video Deepfake Analysis</h2>
        <p>Upload a video file to analyze for deepfake detection</p>
      </div>

      {error && (
        <div className="error-message">
          <AlertCircle size={20} />
          <span>{error}</span>
        </div>
      )}

      <div
        {...getRootProps()}
        className={`video-dropzone ${isDragActive ? 'drag-active' : ''} ${uploading ? 'uploading' : ''}`}
      >
        <input {...getInputProps()} />
        
        {uploading ? (
          <div className="upload-progress">
            <Clock size={48} className="upload-icon" />
            <h3>Processing Video...</h3>
            <div className="progress-bar">
              <div 
                className="progress-fill" 
                style={{ width: `${uploadProgress}%` }}
              ></div>
            </div>
            <p>{Math.round(uploadProgress)}% Complete</p>
            <p className="progress-details">
              Analyzing frames for deepfake detection...
            </p>
          </div>
        ) : (
          <div className="upload-prompt">
            <Video size={48} className="upload-icon" />
            <h3>Drop your video here</h3>
            <p>or click to browse files</p>
            <div className="upload-info">
              <p>Supported formats: MP4, AVI, MOV, MKV, WMV, FLV</p>
              <p>Maximum size: 500MB</p>
            </div>
          </div>
        )}
      </div>

      {currentTask && (
        <div className="task-info">
          <p>Task ID: {currentTask}</p>
          <p>Processing in background...</p>
        </div>
      )}
    </div>
  );
};

export default VideoUpload; 