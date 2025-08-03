import React, { useState, useEffect, useCallback } from 'react';
import { Card, Row, Col, Progress, Badge, Alert, Spinner, Button, Modal } from 'react-bootstrap';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell } from 'recharts';
import { FaCheckCircle, FaTimesCircle, FaExclamationTriangle, FaChartLine, FaEye, FaDownload } from 'react-icons/fa';
import './MultiModelDashboard.css';

const MultiModelDashboard = ({ analysisId, onModelSelect }) => {
  const [analysisData, setAnalysisData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedModel, setSelectedModel] = useState(null);
  const [showDetailsModal, setShowDetailsModal] = useState(false);
  const [realTimeProgress, setRealTimeProgress] = useState(0);
  const [modelAgreement, setModelAgreement] = useState(null);

  // Mock data for demonstration - replace with actual API calls
  const mockAnalysisData = {
    id: analysisId,
    timestamp: new Date().toISOString(),
    imageUrl: '/api/images/sample.jpg',
    status: 'completed',
    models: {
      xception: {
        name: 'Xception',
        prediction: 'FAKE',
        confidence: 0.87,
        inferenceTime: 0.145,
        accuracy: 0.966,
        status: 'completed',
        attentionMaps: true,
        gradCam: true
      },
      efficientnet: {
        name: 'EfficientNet-B4',
        prediction: 'FAKE',
        confidence: 0.79,
        inferenceTime: 0.098,
        accuracy: 0.894,
        status: 'completed',
        attentionMaps: true,
        mobileOptimized: true
      },
      f3net: {
        name: 'F3Net',
        prediction: 'REAL',
        confidence: 0.62,
        inferenceTime: 0.156,
        accuracy: 0.945,
        status: 'completed',
        frequencyAnalysis: true,
        dctCoefficients: true
      }
    },
    ensemble: {
      prediction: 'FAKE',
      confidence: 0.76,
      fusionMethod: 'weighted_average',
      uncertainty: 0.18,
      agreement: 'majority',
      individualPredictions: {
        xception: { prediction: 'FAKE', confidence: 0.87 },
        efficientnet: { prediction: 'FAKE', confidence: 0.79 },
        f3net: { prediction: 'REAL', confidence: 0.62 }
      }
    },
    performance: {
      totalInferenceTime: 0.399,
      averageConfidence: 0.76,
      modelAgreement: 0.67,
      consensusStrength: 'strong'
    }
  };

  useEffect(() => {
    // Simulate API call
    const fetchAnalysisData = async () => {
      try {
        setLoading(true);
        // Simulate network delay
        await new Promise(resolve => setTimeout(resolve, 1500));
        
        // Simulate real-time progress updates
        const progressInterval = setInterval(() => {
          setRealTimeProgress(prev => {
            if (prev >= 100) {
              clearInterval(progressInterval);
              return 100;
            }
            return prev + Math.random() * 15;
          });
        }, 200);

        setAnalysisData(mockAnalysisData);
        calculateModelAgreement(mockAnalysisData);
        setLoading(false);
      } catch (err) {
        setError('Failed to load analysis data');
        setLoading(false);
      }
    };

    fetchAnalysisData();
  }, [analysisId]);

  const calculateModelAgreement = useCallback((data) => {
    if (!data?.ensemble?.individualPredictions) return;

    const predictions = Object.values(data.ensemble.individualPredictions);
    const fakeCount = predictions.filter(p => p.prediction === 'FAKE').length;
    const realCount = predictions.filter(p => p.prediction === 'REAL').length;
    
    const agreement = {
      consensus: fakeCount > realCount ? 'FAKE' : 'REAL',
      agreementLevel: Math.max(fakeCount, realCount) / predictions.length,
      disagreement: fakeCount !== realCount ? Math.min(fakeCount, realCount) : 0,
      confidenceSpread: Math.max(...predictions.map(p => p.confidence)) - Math.min(...predictions.map(p => p.confidence))
    };

    setModelAgreement(agreement);
  }, []);

  const getModelStatusIcon = (status) => {
    switch (status) {
      case 'completed':
        return <FaCheckCircle className="text-success" />;
      case 'processing':
        return <Spinner animation="border" size="sm" />;
      case 'error':
        return <FaTimesCircle className="text-danger" />;
      default:
        return <FaExclamationTriangle className="text-warning" />;
    }
  };

  const getPredictionBadge = (prediction, confidence) => {
    const variant = prediction === 'FAKE' ? 'danger' : 'success';
    return (
      <Badge bg={variant} className="prediction-badge">
        {prediction} ({Math.round(confidence * 100)}%)
      </Badge>
    );
  };

  const getAgreementStrength = (agreementLevel) => {
    if (agreementLevel >= 0.8) return { text: 'Strong Consensus', variant: 'success' };
    if (agreementLevel >= 0.6) return { text: 'Moderate Agreement', variant: 'warning' };
    return { text: 'Low Agreement', variant: 'danger' };
  };

  const performanceChartData = analysisData ? [
    { name: 'Xception', accuracy: analysisData.models.xception.accuracy * 100, inferenceTime: analysisData.models.xception.inferenceTime * 1000 },
    { name: 'EfficientNet', accuracy: analysisData.models.efficientnet.accuracy * 100, inferenceTime: analysisData.models.efficientnet.inferenceTime * 1000 },
    { name: 'F3Net', accuracy: analysisData.models.f3net.accuracy * 100, inferenceTime: analysisData.models.f3net.inferenceTime * 1000 }
  ] : [];

  const confidenceChartData = analysisData ? Object.entries(analysisData.ensemble.individualPredictions).map(([model, data]) => ({
    name: model.charAt(0).toUpperCase() + model.slice(1),
    confidence: data.confidence * 100,
    prediction: data.prediction
  })) : [];

  const agreementChartData = modelAgreement ? [
    { name: 'Agreement', value: modelAgreement.agreementLevel * 100, fill: '#28a745' },
    { name: 'Disagreement', value: (1 - modelAgreement.agreementLevel) * 100, fill: '#dc3545' }
  ] : [];

  if (loading) {
    return (
      <div className="multi-model-dashboard loading">
        <div className="text-center">
          <Spinner animation="border" size="lg" />
          <h4 className="mt-3">Analyzing with Multiple Models</h4>
          <Progress 
            now={realTimeProgress} 
            className="mt-3"
            variant="primary"
            animated
          />
          <p className="text-muted mt-2">
            Processing with Xception, EfficientNet, and F3Net models...
          </p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <Alert variant="danger">
        <Alert.Heading>Analysis Error</Alert.Heading>
        <p>{error}</p>
      </Alert>
    );
  }

  return (
    <div className="multi-model-dashboard">
      {/* Header */}
      <div className="dashboard-header">
        <h2>Multi-Model Analysis Results</h2>
        <div className="header-actions">
          <Button variant="outline-primary" size="sm">
            <FaDownload /> Export Report
          </Button>
          <Button variant="outline-secondary" size="sm">
            <FaChartLine /> Performance Metrics
          </Button>
        </div>
      </div>

      {/* Ensemble Result Summary */}
      <Card className="ensemble-summary mb-4">
        <Card.Body>
          <Row>
            <Col md={6}>
              <h4>Ensemble Prediction</h4>
              <div className="ensemble-result">
                {getPredictionBadge(analysisData.ensemble.prediction, analysisData.ensemble.confidence)}
                <div className="ensemble-details mt-2">
                  <p><strong>Fusion Method:</strong> {analysisData.ensemble.fusionMethod.replace('_', ' ').toUpperCase()}</p>
                  <p><strong>Uncertainty:</strong> {Math.round(analysisData.ensemble.uncertainty * 100)}%</p>
                  <p><strong>Agreement:</strong> {analysisData.ensemble.agreement.toUpperCase()}</p>
                </div>
              </div>
            </Col>
            <Col md={6}>
              <h4>Model Agreement Analysis</h4>
              {modelAgreement && (
                <div className="agreement-analysis">
                  <div className="agreement-strength">
                    <Badge bg={getAgreementStrength(modelAgreement.agreementLevel).variant}>
                      {getAgreementStrength(modelAgreement.agreementLevel).text}
                    </Badge>
                  </div>
                  <p><strong>Consensus:</strong> {modelAgreement.consensus}</p>
                  <p><strong>Agreement Level:</strong> {Math.round(modelAgreement.agreementLevel * 100)}%</p>
                  <p><strong>Confidence Spread:</strong> {Math.round(modelAgreement.confidenceSpread * 100)}%</p>
                </div>
              )}
            </Col>
          </Row>
        </Card.Body>
      </Card>

      {/* Individual Model Results */}
      <Row className="mb-4">
        <Col>
          <h4>Individual Model Results</h4>
        </Col>
      </Row>
      
      <Row className="model-results">
        {Object.entries(analysisData.models).map(([modelKey, modelData]) => (
          <Col key={modelKey} lg={4} md={6} className="mb-3">
            <Card 
              className={`model-card ${selectedModel === modelKey ? 'selected' : ''}`}
              onClick={() => setSelectedModel(modelKey)}
            >
              <Card.Body>
                <div className="model-header">
                  <h5>{modelData.name}</h5>
                  {getModelStatusIcon(modelData.status)}
                </div>
                
                <div className="model-prediction">
                  {getPredictionBadge(modelData.prediction, modelData.confidence)}
                </div>
                
                <div className="model-metrics">
                  <div className="metric">
                    <span className="label">Accuracy:</span>
                    <span className="value">{Math.round(modelData.accuracy * 100)}%</span>
                  </div>
                  <div className="metric">
                    <span className="label">Inference Time:</span>
                    <span className="value">{modelData.inferenceTime.toFixed(3)}s</span>
                  </div>
                  <div className="metric">
                    <span className="label">Features:</span>
                    <div className="feature-badges">
                      {modelData.attentionMaps && <Badge bg="info" size="sm">Attention Maps</Badge>}
                      {modelData.gradCam && <Badge bg="info" size="sm">Grad-CAM</Badge>}
                      {modelData.frequencyAnalysis && <Badge bg="info" size="sm">Frequency Analysis</Badge>}
                      {modelData.mobileOptimized && <Badge bg="info" size="sm">Mobile Optimized</Badge>}
                    </div>
                  </div>
                </div>
                
                <div className="model-actions">
                  <Button 
                    variant="outline-primary" 
                    size="sm"
                    onClick={(e) => {
                      e.stopPropagation();
                      onModelSelect(modelKey, modelData);
                    }}
                  >
                    <FaEye /> View Details
                  </Button>
                </div>
              </Card.Body>
            </Card>
          </Col>
        ))}
      </Row>

      {/* Performance Charts */}
      <Row className="mb-4">
        <Col lg={6}>
          <Card>
            <Card.Body>
              <h5>Model Accuracy Comparison</h5>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={performanceChartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip formatter={(value) => [`${value}%`, 'Accuracy']} />
                  <Bar dataKey="accuracy" fill="#007bff" />
                </BarChart>
              </ResponsiveContainer>
            </Card.Body>
          </Card>
        </Col>
        
        <Col lg={6}>
          <Card>
            <Card.Body>
              <h5>Confidence Distribution</h5>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={confidenceChartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip formatter={(value) => [`${value}%`, 'Confidence']} />
                  <Bar dataKey="confidence" fill="#28a745" />
                </BarChart>
              </ResponsiveContainer>
            </Card.Body>
          </Card>
        </Col>
      </Row>

      {/* Agreement Visualization */}
      <Row className="mb-4">
        <Col lg={6}>
          <Card>
            <Card.Body>
              <h5>Model Agreement Analysis</h5>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={agreementChartData}
                    cx="50%"
                    cy="50%"
                    outerRadius={80}
                    dataKey="value"
                    label={({ name, value }) => `${name}: ${Math.round(value)}%`}
                  >
                    {agreementChartData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.fill} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(value) => [`${Math.round(value)}%`, 'Percentage']} />
                </PieChart>
              </ResponsiveContainer>
            </Card.Body>
          </Card>
        </Col>
        
        <Col lg={6}>
          <Card>
            <Card.Body>
              <h5>Performance Metrics</h5>
              <div className="performance-metrics">
                <div className="metric-item">
                  <span className="metric-label">Total Inference Time:</span>
                  <span className="metric-value">{analysisData.performance.totalInferenceTime.toFixed(3)}s</span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">Average Confidence:</span>
                  <span className="metric-value">{Math.round(analysisData.performance.averageConfidence * 100)}%</span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">Model Agreement:</span>
                  <span className="metric-value">{Math.round(analysisData.performance.modelAgreement * 100)}%</span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">Consensus Strength:</span>
                  <Badge bg={analysisData.performance.consensusStrength === 'strong' ? 'success' : 'warning'}>
                    {analysisData.performance.consensusStrength.toUpperCase()}
                  </Badge>
                </div>
              </div>
            </Card.Body>
          </Card>
        </Col>
      </Row>

      {/* Model Details Modal */}
      <Modal 
        show={showDetailsModal} 
        onHide={() => setShowDetailsModal(false)}
        size="lg"
      >
        <Modal.Header closeButton>
          <Modal.Title>Model Analysis Details</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          {selectedModel && analysisData.models[selectedModel] && (
            <div className="model-details">
              <h5>{analysisData.models[selectedModel].name} Analysis</h5>
              <p>Detailed analysis information and visualizations would be displayed here.</p>
            </div>
          )}
        </Modal.Body>
      </Modal>
    </div>
  );
};

export default MultiModelDashboard; 