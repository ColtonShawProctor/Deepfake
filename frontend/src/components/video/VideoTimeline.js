import React, { useState, useRef, useEffect } from 'react';
import { Play, Pause, SkipBack, SkipForward, Maximize2 } from 'lucide-react';
import './VideoTimeline.css';

const VideoTimeline = ({ 
  videoMetadata, 
  frameAnalyses, 
  onFrameSelect, 
  selectedFrame = 0,
  videoUrl = null 
}) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const videoRef = useRef(null);
  const timelineRef = useRef(null);

  useEffect(() => {
    if (videoRef.current && videoMetadata) {
      setDuration(videoMetadata.duration);
    }
  }, [videoMetadata]);

  useEffect(() => {
    if (videoRef.current) {
      const video = videoRef.current;
      
      const handleTimeUpdate = () => {
        setCurrentTime(video.currentTime);
        const frameIndex = Math.floor(video.currentTime * videoMetadata.fps);
        if (onFrameSelect && frameIndex !== selectedFrame) {
          onFrameSelect(frameIndex);
        }
      };

      const handleLoadedMetadata = () => {
        setDuration(video.duration);
      };

      video.addEventListener('timeupdate', handleTimeUpdate);
      video.addEventListener('loadedmetadata', handleLoadedMetadata);

      return () => {
        video.removeEventListener('timeupdate', handleTimeUpdate);
        video.removeEventListener('loadedmetadata', handleLoadedMetadata);
      };
    }
  }, [videoMetadata, onFrameSelect, selectedFrame]);

  const togglePlay = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause();
      } else {
        videoRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const skipBackward = () => {
    if (videoRef.current) {
      videoRef.current.currentTime = Math.max(0, videoRef.current.currentTime - 5);
    }
  };

  const skipForward = () => {
    if (videoRef.current) {
      videoRef.current.currentTime = Math.min(duration, videoRef.current.currentTime + 5);
    }
  };

  const handleTimelineClick = (e) => {
    if (timelineRef.current && videoRef.current) {
      const rect = timelineRef.current.getBoundingClientRect();
      const clickX = e.clientX - rect.left;
      const percentage = clickX / rect.width;
      const newTime = percentage * duration;
      videoRef.current.currentTime = newTime;
    }
  };

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getFrameAnalysis = (frameIndex) => {
    return frameAnalyses?.find(fa => fa.frame_number === frameIndex) || null;
  };

  const getConfidenceColor = (confidence) => {
    if (confidence > 0.7) return '#38a169'; // Green for high confidence real
    if (confidence > 0.5) return '#d69e2e'; // Yellow for medium confidence
    return '#e53e3e'; // Red for low confidence (likely deepfake)
  };

  return (
    <div className="video-timeline-container">
      {/* Video Player */}
      {videoUrl && (
        <div className="video-player">
          <video
            ref={videoRef}
            src={videoUrl}
            className="video-element"
            controls={false}
            muted
          />
          <div className="video-overlay">
            <button className="play-button" onClick={togglePlay}>
              {isPlaying ? <Pause size={24} /> : <Play size={24} />}
            </button>
          </div>
        </div>
      )}

      {/* Timeline Controls */}
      <div className="timeline-controls">
        <button className="control-button" onClick={skipBackward}>
          <SkipBack size={20} />
        </button>
        
        <button className="control-button play-pause" onClick={togglePlay}>
          {isPlaying ? <Pause size={20} /> : <Play size={20} />}
        </button>
        
        <button className="control-button" onClick={skipForward}>
          <SkipForward size={20} />
        </button>
        
        <div className="time-display">
          <span>{formatTime(currentTime)}</span>
          <span>/</span>
          <span>{formatTime(duration)}</span>
        </div>
      </div>

      {/* Timeline */}
      <div className="timeline-wrapper">
        <div 
          ref={timelineRef}
          className="timeline-track"
          onClick={handleTimelineClick}
        >
          {/* Progress bar */}
          <div 
            className="timeline-progress"
            style={{ width: `${(currentTime / duration) * 100}%` }}
          ></div>
          
          {/* Frame analysis indicators */}
          {frameAnalyses?.map((frameAnalysis, index) => {
            const frameTime = frameAnalysis.timestamp;
            const position = (frameTime / duration) * 100;
            const confidence = frameAnalysis.confidence_score;
            const isSelected = frameAnalysis.frame_number === selectedFrame;
            
            return (
              <div
                key={frameAnalysis.frame_number}
                className={`frame-indicator ${isSelected ? 'selected' : ''}`}
                style={{
                  left: `${position}%`,
                  backgroundColor: getConfidenceColor(confidence)
                }}
                title={`Frame ${frameAnalysis.frame_number}: ${(confidence * 100).toFixed(1)}% confidence (${frameAnalysis.is_deepfake ? 'FAKE' : 'REAL'})`}
                onClick={(e) => {
                  e.stopPropagation();
                  if (onFrameSelect) {
                    onFrameSelect(frameAnalysis.frame_number);
                  }
                  if (videoRef.current) {
                    videoRef.current.currentTime = frameTime;
                  }
                }}
              >
                <div className="frame-tooltip">
                  <div>Frame {frameAnalysis.frame_number}</div>
                  <div>Time: {formatTime(frameTime)}</div>
                  <div>Confidence: {(confidence * 100).toFixed(1)}% ({frameAnalysis.is_deepfake ? 'FAKE' : 'REAL'})</div>
                  <div>Result: {frameAnalysis.is_deepfake ? 'Deepfake' : 'Real'}</div>
                </div>
              </div>
            );
          })}
          
          {/* Playhead */}
          <div 
            className="timeline-playhead"
            style={{ left: `${(currentTime / duration) * 100}%` }}
          ></div>
        </div>
      </div>

      {/* Frame Analysis Display */}
      <div className="frame-analysis-display">
        <h4>Frame Analysis</h4>
        <div className="frame-info">
          <div className="frame-details">
            <span>Frame: {selectedFrame}</span>
            <span>Time: {formatTime(currentTime)}</span>
          </div>
          
          {(() => {
            const analysis = getFrameAnalysis(selectedFrame);
            if (analysis) {
              return (
                <div className="analysis-results">
                  <div className="confidence-bar">
                    <div 
                      className="confidence-fill"
                      style={{ 
                        width: `${analysis.confidence_score * 100}%`,
                        backgroundColor: getConfidenceColor(analysis.confidence_score)
                      }}
                    ></div>
                  </div>
                  <div className="confidence-text">
                    Confidence: {(analysis.confidence_score * 100).toFixed(1)}% ({analysis.is_deepfake ? 'FAKE' : 'REAL'})
                  </div>
                  <div className={`result-badge ${analysis.is_deepfake ? 'deepfake' : 'real'}`}>
                    {analysis.is_deepfake ? 'Deepfake Detected' : 'Real'}
                  </div>
                </div>
              );
            }
            return <div className="no-analysis">No analysis available for this frame</div>;
          })()}
        </div>
      </div>

      {/* Video Metadata */}
      {videoMetadata && (
        <div className="video-metadata">
          <h4>Video Information</h4>
          <div className="metadata-grid">
            <div className="metadata-item">
              <span className="label">Duration:</span>
              <span className="value">{formatTime(videoMetadata.duration)}</span>
            </div>
            <div className="metadata-item">
              <span className="label">FPS:</span>
              <span className="value">{videoMetadata.fps.toFixed(2)}</span>
            </div>
            <div className="metadata-item">
              <span className="label">Frames:</span>
              <span className="value">{videoMetadata.frame_count.toLocaleString()}</span>
            </div>
            <div className="metadata-item">
              <span className="label">Resolution:</span>
              <span className="value">{videoMetadata.width} × {videoMetadata.height}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default VideoTimeline; 