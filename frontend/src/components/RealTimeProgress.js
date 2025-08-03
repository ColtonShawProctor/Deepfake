import React, { useState, useEffect, useRef } from 'react';
import { Card, Progress, Badge, Alert, Spinner, Button } from 'react-bootstrap';
import { FaPlay, FaPause, FaStop, FaCheckCircle, FaTimesCircle, FaExclamationTriangle, FaClock, FaCog } from 'react-icons/fa';
import './RealTimeProgress.css';

const RealTimeProgress = ({ 
  analysisId, 
  models, 
  onComplete, 
  onError, 
  onCancel,
  autoStart = true 
}) => {
  const [analysisState, setAnalysisState] = useState('idle'); // idle, running, paused, completed, error
  const [overallProgress, setOverallProgress] = useState(0);
  const [modelProgress, setModelProgress] = useState({});
  const [modelStatus, setModelStatus] = useState({});
  const [currentModel, setCurrentModel] = useState(null);
  const [startTime, setStartTime] = useState(null);
  const [elapsedTime, setElapsedTime] = useState(0);
  const [estimatedTime, setEstimatedTime] = useState(0);
  const [logs, setLogs] = useState([]);
  const [ensembleStatus, setEnsembleStatus] = useState('waiting');
  
  const intervalRef = useRef(null);
  const timeIntervalRef = useRef(null);

  // Initialize model progress tracking
  useEffect(() => {
    const initialProgress = {};
    const initialStatus = {};
    
    models.forEach(model => {
      initialProgress[model.id] = 0;
      initialStatus[model.id] = 'pending';
    });
    
    setModelProgress(initialProgress);
    setModelStatus(initialStatus);
    
    if (autoStart) {
      startAnalysis();
    }
  }, [models, autoStart]);

  // Cleanup intervals on unmount
  useEffect(() => {
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
      if (timeIntervalRef.current) clearInterval(timeIntervalRef.current);
    };
  }, []);

  const startAnalysis = () => {
    setAnalysisState('running');
    setStartTime(Date.now());
    setLogs([]);
    
    // Start time tracking
    timeIntervalRef.current = setInterval(() => {
      setElapsedTime(prev => prev + 1);
    }, 1000);

    // Simulate model processing
    simulateModelProcessing();
  };

  const pauseAnalysis = () => {
    setAnalysisState('paused');
    if (timeIntervalRef.current) clearInterval(timeIntervalRef.current);
    if (intervalRef.current) clearInterval(intervalRef.current);
  };

  const resumeAnalysis = () => {
    setAnalysisState('running');
    timeIntervalRef.current = setInterval(() => {
      setElapsedTime(prev => prev + 1);
    }, 1000);
    simulateModelProcessing();
  };

  const cancelAnalysis = () => {
    setAnalysisState('cancelled');
    if (timeIntervalRef.current) clearInterval(timeIntervalRef.current);
    if (intervalRef.current) clearInterval(intervalRef.current);
    
    if (onCancel) {
      onCancel(analysisId);
    }
  };

  const simulateModelProcessing = () => {
    let currentModelIndex = 0;
    const modelIds = models.map(m => m.id);
    
    const processNextModel = () => {
      if (currentModelIndex >= modelIds.length) {
        // All models completed, start ensemble processing
        processEnsemble();
        return;
      }

      const modelId = modelIds[currentModelIndex];
      setCurrentModel(modelId);
      setModelStatus(prev => ({ ...prev, [modelId]: 'running' }));
      
      addLog(`Starting ${models.find(m => m.id === modelId).name} analysis...`, 'info');

      // Simulate model processing with realistic timing
      const model = models.find(m => m.id === modelId);
      const processingTime = model.estimatedTime || 5000; // Default 5 seconds
      const updateInterval = 100; // Update every 100ms
      const totalUpdates = processingTime / updateInterval;
      let currentUpdate = 0;

      intervalRef.current = setInterval(() => {
        currentUpdate++;
        const progress = Math.min((currentUpdate / totalUpdates) * 100, 100);
        
        setModelProgress(prev => ({ ...prev, [modelId]: progress }));
        
        if (progress >= 100) {
          clearInterval(intervalRef.current);
          setModelStatus(prev => ({ ...prev, [modelId]: 'completed' }));
          addLog(`${models.find(m => m.id === modelId).name} analysis completed`, 'success');
          
          // Update overall progress
          const completedModels = Object.values(modelStatus).filter(status => status === 'completed').length + 1;
          const overallProgress = (completedModels / models.length) * 100;
          setOverallProgress(overallProgress);
          
          currentModelIndex++;
          setTimeout(processNextModel, 500); // Small delay between models
        }
      }, updateInterval);
    };

    processNextModel();
  };

  const processEnsemble = () => {
    setEnsembleStatus('processing');
    addLog('Starting ensemble analysis...', 'info');
    
    // Simulate ensemble processing
    setTimeout(() => {
      setEnsembleStatus('completed');
      setAnalysisState('completed');
      addLog('Ensemble analysis completed successfully', 'success');
      
      if (timeIntervalRef.current) clearInterval(timeIntervalRef.current);
      
      if (onComplete) {
        onComplete({
          analysisId,
          results: generateMockResults(),
          processingTime: elapsedTime
        });
      }
    }, 2000);
  };

  const generateMockResults = () => {
    return {
      ensemble: {
        prediction: 'FAKE',
        confidence: 0.76,
        fusionMethod: 'weighted_average',
        uncertainty: 0.18
      },
      models: models.map(model => ({
        id: model.id,
        name: model.name,
        prediction: Math.random() > 0.5 ? 'FAKE' : 'REAL',
        confidence: 0.6 + Math.random() * 0.3,
        inferenceTime: (Math.random() * 2 + 0.5).toFixed(3)
      }))
    };
  };

  const addLog = (message, type = 'info') => {
    const timestamp = new Date().toLocaleTimeString();
    setLogs(prev => [...prev, { message, type, timestamp }]);
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed':
        return <FaCheckCircle className="text-success" />;
      case 'running':
        return <Spinner animation="border" size="sm" />;
      case 'error':
        return <FaTimesCircle className="text-danger" />;
      case 'pending':
        return <FaClock className="text-muted" />;
      default:
        return <FaExclamationTriangle className="text-warning" />;
    }
  };

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getEstimatedTimeRemaining = () => {
    if (overallProgress === 0) return 'Calculating...';
    
    const completedModels = Object.values(modelStatus).filter(status => status === 'completed').length;
    const remainingModels = models.length - completedModels;
    const avgTimePerModel = elapsedTime / Math.max(completedModels, 1);
    const estimatedRemaining = remainingModels * avgTimePerModel + (ensembleStatus === 'waiting' ? 2 : 0);
    
    return formatTime(Math.round(estimatedRemaining));
  };

  return (
    <div className="real-time-progress">
      <Card className="progress-card">
        <Card.Header className="progress-header">
          <div className="header-content">
            <h5>Real-Time Analysis Progress</h5>
            <div className="header-badges">
              <Badge bg={analysisState === 'running' ? 'success' : analysisState === 'completed' ? 'primary' : 'secondary'}>
                {analysisState.toUpperCase()}
              </Badge>
              <Badge bg="info">ID: {analysisId}</Badge>
            </div>
          </div>
          <div className="header-actions">
            {analysisState === 'idle' && (
              <Button variant="success" size="sm" onClick={startAnalysis}>
                <FaPlay /> Start
              </Button>
            )}
            {analysisState === 'running' && (
              <Button variant="warning" size="sm" onClick={pauseAnalysis}>
                <FaPause /> Pause
              </Button>
            )}
            {analysisState === 'paused' && (
              <Button variant="success" size="sm" onClick={resumeAnalysis}>
                <FaPlay /> Resume
              </Button>
            )}
            {(analysisState === 'running' || analysisState === 'paused') && (
              <Button variant="danger" size="sm" onClick={cancelAnalysis}>
                <FaStop /> Cancel
              </Button>
            )}
          </div>
        </Card.Header>

        <Card.Body className="progress-body">
          {/* Overall Progress */}
          <div className="overall-progress-section">
            <div className="progress-header-row">
              <h6>Overall Progress</h6>
              <div className="progress-stats">
                <span className="stat">
                  <FaClock /> {formatTime(elapsedTime)}
                </span>
                <span className="stat">
                  ETA: {getEstimatedTimeRemaining()}
                </span>
              </div>
            </div>
            <Progress 
              now={overallProgress} 
              className="overall-progress-bar"
              variant={analysisState === 'completed' ? 'success' : 'primary'}
              animated={analysisState === 'running'}
            />
            <div className="progress-percentage">
              {Math.round(overallProgress)}% Complete
            </div>
          </div>

          {/* Model Progress */}
          <div className="model-progress-section">
            <h6>Model Analysis Progress</h6>
            <div className="model-progress-list">
              {models.map(model => (
                <div key={model.id} className={`model-progress-item ${currentModel === model.id ? 'active' : ''}`}>
                  <div className="model-info">
                    <div className="model-header">
                      <span className="model-name">{model.name}</span>
                      {getStatusIcon(modelStatus[model.id])}
                    </div>
                    <div className="model-details">
                      <small className="text-muted">
                        {modelStatus[model.id] === 'running' && currentModel === model.id && (
                          <span className="processing-indicator">
                            <FaCog className="spinning" /> Processing...
                          </span>
                        )}
                        {modelStatus[model.id] === 'completed' && (
                          <span className="text-success">Completed</span>
                        )}
                        {model.estimatedTime && `Est. ${model.estimatedTime / 1000}s`}
                      </small>
                    </div>
                  </div>
                  <div className="model-progress">
                    <Progress 
                      now={modelProgress[model.id] || 0} 
                      size="sm"
                      variant={modelStatus[model.id] === 'completed' ? 'success' : 'primary'}
                      animated={modelStatus[model.id] === 'running'}
                    />
                    <span className="progress-text">
                      {Math.round(modelProgress[model.id] || 0)}%
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Ensemble Status */}
          <div className="ensemble-status-section">
            <h6>Ensemble Analysis</h6>
            <div className={`ensemble-status ${ensembleStatus}`}>
              <div className="ensemble-info">
                <span className="ensemble-label">Fusion Method:</span>
                <span className="ensemble-value">Weighted Average</span>
              </div>
              <div className="ensemble-progress">
                {ensembleStatus === 'waiting' && (
                  <div className="waiting-status">
                    <FaClock /> Waiting for all models to complete
                  </div>
                )}
                {ensembleStatus === 'processing' && (
                  <div className="processing-status">
                    <Spinner animation="border" size="sm" /> Combining model predictions...
                  </div>
                )}
                {ensembleStatus === 'completed' && (
                  <div className="completed-status">
                    <FaCheckCircle className="text-success" /> Ensemble analysis completed
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Analysis Logs */}
          <div className="analysis-logs">
            <h6>Analysis Logs</h6>
            <div className="logs-container">
              {logs.slice(-5).map((log, index) => (
                <div key={index} className={`log-entry log-${log.type}`}>
                  <span className="log-timestamp">{log.timestamp}</span>
                  <span className="log-message">{log.message}</span>
                </div>
              ))}
              {logs.length === 0 && (
                <div className="no-logs">
                  <small className="text-muted">No logs yet...</small>
                </div>
              )}
            </div>
          </div>
        </Card.Body>
      </Card>
    </div>
  );
};

export default RealTimeProgress; 