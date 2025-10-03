import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, Image as ImageIcon, AlertCircle, CheckCircle, X } from 'lucide-react';
import { getApiConfig } from '../../config/api';
import './ImageUploader.css';

const ImageUploader = ({ onImageSelected, onUploadComplete, onError }) => {
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [selectedFile, setSelectedFile] = useState(null);
  const [error, setError] = useState(null);
  const [uploadSuccess, setUploadSuccess] = useState(false);

  const onDrop = useCallback(async (acceptedFiles) => {
    const file = acceptedFiles[0];
    if (!file) return;

    // Debug: Log file information
    console.log('Image upload attempt:', {
      name: file.name,
      type: file.type,
      size: file.size,
      lastModified: file.lastModified
    });

    // Validate file type
    const allowedTypes = [
      'image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/bmp', 'image/webp'
    ];
    const allowedExtensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'];
    
    // Check MIME type first
    const isValidMimeType = allowedTypes.includes(file.type);
    
    // If MIME type check fails, check file extension
    const fileExtension = file.name.toLowerCase().substring(file.name.lastIndexOf('.'));
    const isValidExtension = allowedExtensions.includes(fileExtension);
    
    if (!isValidMimeType && !isValidExtension) {
      const errorMsg = `Unsupported file type: ${file.type || fileExtension}. Please upload JPEG, PNG, GIF, BMP, or WebP images.`;
      setError(errorMsg);
      if (onError) onError(errorMsg);
      return;
    }

    // Validate file size (10MB max)
    const maxSize = 10 * 1024 * 1024; // 10MB
    if (file.size > maxSize) {
      const errorMsg = 'File too large. Maximum size is 10MB.';
      setError(errorMsg);
      if (onError) onError(errorMsg);
      return;
    }

    setError(null);
    setSelectedFile(file);
    setUploadSuccess(false);

    // Call the callback if provided
    if (onImageSelected) {
      onImageSelected(file);
    }

    // Auto-upload if onUploadComplete is provided
    if (onUploadComplete) {
      await uploadFile(file);
    }
  }, [onImageSelected, onUploadComplete, onError]);

  const uploadFile = async (file) => {
    if (!file) return;

    setUploading(true);
    setUploadProgress(0);
    setError(null);

    try {
      // Create form data
      const formData = new FormData();
      formData.append('file', file);

      // Get auth token
      const token = localStorage.getItem('token');
      if (!token) {
        throw new Error('Authentication required. Please log in again.');
      }

      // Get API configuration
      const apiConfig = getApiConfig();
      
      // Upload image
      const uploadResponse = await fetch(`${apiConfig.BASE_URL}/api/upload`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
        body: formData,
      });

      if (!uploadResponse.ok) {
        const errorData = await uploadResponse.json();
        throw new Error(errorData.detail || `Upload failed with status ${uploadResponse.status}`);
      }

      const uploadData = await uploadResponse.json();
      setUploadSuccess(true);
      setUploadProgress(100);

      // Call the callback if provided
      if (onUploadComplete) {
        onUploadComplete(uploadData);
      }

      // Reset after a delay
      setTimeout(() => {
        setUploadSuccess(false);
        setUploadProgress(0);
      }, 3000);

    } catch (err) {
      console.error('Upload error:', err);
      const errorMsg = err.message || 'Upload failed. Please try again.';
      setError(errorMsg);
      if (onError) onError(errorMsg);
    } finally {
      setUploading(false);
    }
  };

  const handleManualUpload = () => {
    if (selectedFile) {
      uploadFile(selectedFile);
    }
  };

  const clearSelection = () => {
    setSelectedFile(null);
    setError(null);
    setUploadSuccess(false);
    setUploadProgress(0);
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.gif', '.bmp', '.webp']
    },
    maxFiles: 1,
    multiple: false
  });

  return (
    <div className="image-uploader">
      <div className="upload-container">
        <div
          {...getRootProps()}
          className={`dropzone ${isDragActive ? 'drag-active' : ''} ${selectedFile ? 'has-file' : ''}`}
        >
          <input {...getInputProps()} />
          
          {!selectedFile ? (
            <div className="upload-prompt">
              <ImageIcon className="upload-icon" size={48} />
              <h3>Upload Image</h3>
              <p>Drag & drop an image here, or click to select</p>
              <p className="file-types">Supports: JPEG, PNG, GIF, BMP, WebP (max 10MB)</p>
            </div>
          ) : (
            <div className="file-info">
              <ImageIcon className="file-icon" size={32} />
              <div className="file-details">
                <h4>{selectedFile.name}</h4>
                <p>{(selectedFile.size / 1024 / 1024).toFixed(2)} MB</p>
                <p>{selectedFile.type}</p>
              </div>
              <button 
                className="clear-btn"
                onClick={(e) => {
                  e.stopPropagation();
                  clearSelection();
                }}
              >
                <X size={20} />
              </button>
            </div>
          )}
        </div>

        {selectedFile && !onUploadComplete && (
          <button 
            className="upload-btn"
            onClick={handleManualUpload}
            disabled={uploading}
          >
            {uploading ? 'Uploading...' : 'Upload Image'}
          </button>
        )}

        {uploading && (
          <div className="upload-progress">
            <div className="progress-bar">
              <div 
                className="progress-fill" 
                style={{ width: `${uploadProgress}%` }}
              ></div>
            </div>
            <span>{uploadProgress}%</span>
          </div>
        )}

        {error && (
          <div className="error-message">
            <AlertCircle size={20} />
            <span>{error}</span>
          </div>
        )}

        {uploadSuccess && (
          <div className="success-message">
            <CheckCircle size={20} />
            <span>Image uploaded successfully!</span>
          </div>
        )}
      </div>
    </div>
  );
};

export default ImageUploader;





