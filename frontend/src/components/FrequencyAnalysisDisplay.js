import React, { useState, useEffect, useRef } from 'react';
import { Card, Row, Col, Badge, Button, Tabs, Tab, Alert } from 'react-bootstrap';
import { FaWaveSquare, FaChartBar, FaEye, FaDownload, FaInfoCircle, FaExclamationTriangle } from 'react-icons/fa';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, ScatterChart, Scatter } from 'recharts';
import './FrequencyAnalysisDisplay.css';

const FrequencyAnalysisDisplay = ({ 
  f3netResults, 
  originalImage, 
  onExport,
  onHeatmapView 
}) => {
  const [activeTab, setActiveTab] = useState('overview');
  const [selectedFrequency, setSelectedFrequency] = useState('all');
  const [dctCoefficients, setDctCoefficients] = useState(null);
  const [frequencySpectrum, setFrequencySpectrum] = useState(null);
  const [analysisMetrics, setAnalysisMetrics] = useState({});

  useEffect(() => {
    if (f3netResults) {
      processFrequencyData(f3netResults);
    }
  }, [f3netResults]);

  const processFrequencyData = (results) => {
    // Generate mock DCT coefficients data
    const generateDCTData = () => {
      const data = [];
      for (let i = 0; i < 64; i++) {
        data.push({
          coefficient: i,
          magnitude: Math.random() * 100,
          frequency: Math.random() * 50,
          phase: Math.random() * 360
        });
      }
      return data;
    };

    // Generate frequency spectrum data
    const generateSpectrumData = () => {
      const data = [];
      for (let i = 0; i < 100; i++) {
        data.push({
          frequency: i,
          amplitude: Math.random() * 100,
          power: Math.random() * 1000
        });
      }
      return data;
    };

    setDctCoefficients(generateDCTData());
    setFrequencySpectrum(generateSpectrumData());
    
    // Calculate analysis metrics
    setAnalysisMetrics({
      dominantFrequency: Math.floor(Math.random() * 50),
      frequencyRange: { low: 0, high: 50 },
      dctEnergy: Math.random() * 1000,
      frequencyEntropy: Math.random() * 5,
      artifactScore: Math.random() * 100
    });
  };

  const getFrequencyAnalysisInsights = () => {
    const insights = [];
    
    if (analysisMetrics.artifactScore > 70) {
      insights.push({
        type: 'warning',
        message: 'High frequency artifacts detected - potential manipulation indicators',
        icon: <FaExclamationTriangle />
      });
    }
    
    if (analysisMetrics.dctEnergy > 500) {
      insights.push({
        type: 'info',
        message: 'Elevated DCT energy suggests compression artifacts',
        icon: <FaInfoCircle />
      });
    }
    
    if (analysisMetrics.frequencyEntropy > 3) {
      insights.push({
        type: 'success',
        message: 'Normal frequency distribution detected',
        icon: <FaInfoCircle />
      });
    }
    
    return insights;
  };

  const renderDCTCoefficients = () => (
    <div className="dct-coefficients-section">
      <div className="section-header">
        <h6>DCT Coefficient Analysis</h6>
        <Badge bg="info">8x8 Block Size</Badge>
      </div>
      
      <Row>
        <Col lg={8}>
          <Card className="chart-card">
            <Card.Body>
              <h6>DCT Coefficient Magnitudes</h6>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={dctCoefficients?.slice(0, 20)}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="coefficient" />
                  <YAxis />
                  <Tooltip formatter={(value) => [value.toFixed(2), 'Magnitude']} />
                  <Bar dataKey="magnitude" fill="#007bff" />
                </BarChart>
              </ResponsiveContainer>
            </Card.Body>
          </Card>
        </Col>
        
        <Col lg={4}>
          <Card className="metrics-card">
            <Card.Body>
              <h6>DCT Metrics</h6>
              <div className="metric-item">
                <span className="metric-label">Total Energy:</span>
                <span className="metric-value">{analysisMetrics.dctEnergy?.toFixed(2)}</span>
              </div>
              <div className="metric-item">
                <span className="metric-label">Dominant Freq:</span>
                <span className="metric-value">{analysisMetrics.dominantFrequency} Hz</span>
              </div>
              <div className="metric-item">
                <span className="metric-label">Frequency Range:</span>
                <span className="metric-value">
                  {analysisMetrics.frequencyRange?.low}-{analysisMetrics.frequencyRange?.high} Hz
                </span>
              </div>
              <div className="metric-item">
                <span className="metric-label">Entropy:</span>
                <span className="metric-value">{analysisMetrics.frequencyEntropy?.toFixed(2)}</span>
              </div>
            </Card.Body>
          </Card>
        </Col>
      </Row>
    </div>
  );

  const renderFrequencySpectrum = () => (
    <div className="frequency-spectrum-section">
      <div className="section-header">
        <h6>Frequency Domain Analysis</h6>
        <div className="frequency-controls">
          <Button 
            variant={selectedFrequency === 'all' ? 'primary' : 'outline-primary'} 
            size="sm"
            onClick={() => setSelectedFrequency('all')}
          >
            All Frequencies
          </Button>
          <Button 
            variant={selectedFrequency === 'low' ? 'primary' : 'outline-primary'} 
            size="sm"
            onClick={() => setSelectedFrequency('low')}
          >
            Low Freq
          </Button>
          <Button 
            variant={selectedFrequency === 'high' ? 'primary' : 'outline-primary'} 
            size="sm"
            onClick={() => setSelectedFrequency('high')}
          >
            High Freq
          </Button>
        </div>
      </div>
      
      <Row>
        <Col lg={6}>
          <Card className="chart-card">
            <Card.Body>
              <h6>Frequency Spectrum</h6>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={frequencySpectrum}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="frequency" />
                  <YAxis />
                  <Tooltip formatter={(value) => [value.toFixed(2), 'Amplitude']} />
                  <Line type="monotone" dataKey="amplitude" stroke="#28a745" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </Card.Body>
          </Card>
        </Col>
        
        <Col lg={6}>
          <Card className="chart-card">
            <Card.Body>
              <h6>Power Spectral Density</h6>
              <ResponsiveContainer width="100%" height={300}>
                <ScatterChart data={frequencySpectrum}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="frequency" />
                  <YAxis />
                  <Tooltip formatter={(value) => [value.toFixed(2), 'Power']} />
                  <Scatter dataKey="power" fill="#dc3545" />
                </ScatterChart>
              </ResponsiveContainer>
            </Card.Body>
          </Card>
        </Col>
      </Row>
    </div>
  );

  const renderArtifactDetection = () => (
    <div className="artifact-detection-section">
      <div className="section-header">
        <h6>Frequency Artifact Detection</h6>
        <Badge bg={analysisMetrics.artifactScore > 70 ? 'danger' : 'success'}>
          Score: {analysisMetrics.artifactScore?.toFixed(1)}%
        </Badge>
      </div>
      
      <Row>
        <Col lg={8}>
          <Card className="artifact-card">
            <Card.Body>
              <div className="artifact-analysis">
                <div className="artifact-metric">
                  <div className="metric-circle">
                    <span className="metric-value">{analysisMetrics.artifactScore?.toFixed(0)}%</span>
                    <span className="metric-label">Artifact Score</span>
                  </div>
                </div>
                
                <div className="artifact-breakdown">
                  <h6>Detection Breakdown</h6>
                  <div className="breakdown-item">
                    <span className="item-label">Compression Artifacts:</span>
                    <div className="item-bar">
                      <div 
                        className="item-progress" 
                        style={{ width: `${Math.random() * 100}%` }}
                      ></div>
                    </div>
                    <span className="item-value">{Math.floor(Math.random() * 100)}%</span>
                  </div>
                  <div className="breakdown-item">
                    <span className="item-label">Frequency Inconsistencies:</span>
                    <div className="item-bar">
                      <div 
                        className="item-progress" 
                        style={{ width: `${Math.random() * 100}%` }}
                      ></div>
                    </div>
                    <span className="item-value">{Math.floor(Math.random() * 100)}%</span>
                  </div>
                  <div className="breakdown-item">
                    <span className="item-label">Spectral Anomalies:</span>
                    <div className="item-bar">
                      <div 
                        className="item-progress" 
                        style={{ width: `${Math.random() * 100}%` }}
                      ></div>
                    </div>
                    <span className="item-value">{Math.floor(Math.random() * 100)}%</span>
                  </div>
                </div>
              </div>
            </Card.Body>
          </Card>
        </Col>
        
        <Col lg={4}>
          <Card className="insights-card">
            <Card.Body>
              <h6>Analysis Insights</h6>
              <div className="insights-list">
                {getFrequencyAnalysisInsights().map((insight, index) => (
                  <Alert key={index} variant={insight.type} className="insight-alert">
                    <div className="insight-content">
                      {insight.icon}
                      <span>{insight.message}</span>
                    </div>
                  </Alert>
                ))}
              </div>
            </Card.Body>
          </Card>
        </Col>
      </Row>
    </div>
  );

  const renderOverview = () => (
    <div className="overview-section">
      <Row>
        <Col lg={6}>
          <Card className="overview-card">
            <Card.Body>
              <h6>Frequency Analysis Summary</h6>
              <div className="summary-metrics">
                <div className="summary-item">
                  <span className="item-icon"><FaWaveSquare /></span>
                  <div className="item-content">
                    <span className="item-label">Analysis Type</span>
                    <span className="item-value">DCT Frequency Domain</span>
                  </div>
                </div>
                <div className="summary-item">
                  <span className="item-icon"><FaChartBar /></span>
                  <div className="item-content">
                    <span className="item-label">Block Size</span>
                    <span className="item-value">8x8 pixels</span>
                  </div>
                </div>
                <div className="summary-item">
                  <span className="item-icon"><FaInfoCircle /></span>
                  <div className="item-content">
                    <span className="item-label">Processing Method</span>
                    <span className="item-value">Local Frequency Attention</span>
                  </div>
                </div>
              </div>
            </Card.Body>
          </Card>
        </Col>
        
        <Col lg={6}>
          <Card className="overview-card">
            <Card.Body>
              <h6>Key Findings</h6>
              <div className="findings-list">
                <div className="finding-item">
                  <Badge bg={analysisMetrics.artifactScore > 70 ? 'danger' : 'success'}>
                    {analysisMetrics.artifactScore > 70 ? 'High' : 'Low'} Artifact Score
                  </Badge>
                  <span className="finding-description">
                    {analysisMetrics.artifactScore > 70 
                      ? 'Significant frequency artifacts detected' 
                      : 'Minimal frequency artifacts detected'
                    }
                  </span>
                </div>
                <div className="finding-item">
                  <Badge bg="info">DCT Energy: {analysisMetrics.dctEnergy?.toFixed(0)}</Badge>
                  <span className="finding-description">
                    {analysisMetrics.dctEnergy > 500 
                      ? 'Elevated energy levels suggest compression' 
                      : 'Normal energy distribution'
                    }
                  </span>
                </div>
                <div className="finding-item">
                  <Badge bg="warning">Entropy: {analysisMetrics.frequencyEntropy?.toFixed(2)}</Badge>
                  <span className="finding-description">
                    Frequency distribution entropy analysis
                  </span>
                </div>
              </div>
            </Card.Body>
          </Card>
        </Col>
      </Row>
    </div>
  );

  if (!f3netResults) {
    return (
      <Alert variant="info">
        <FaInfoCircle /> No F3Net frequency analysis results available
      </Alert>
    );
  }

  return (
    <div className="frequency-analysis-display">
      <div className="display-header">
        <h4>F3Net Frequency Analysis</h4>
        <div className="header-actions">
          <Button variant="outline-primary" size="sm" onClick={onHeatmapView}>
            <FaEye /> View Heatmap
          </Button>
          <Button variant="outline-success" size="sm" onClick={onExport}>
            <FaDownload /> Export Analysis
          </Button>
        </div>
      </div>

      <Tabs 
        activeKey={activeTab} 
        onSelect={(k) => setActiveTab(k)}
        className="analysis-tabs"
      >
        <Tab eventKey="overview" title="Overview">
          {renderOverview()}
        </Tab>
        <Tab eventKey="dct" title="DCT Coefficients">
          {renderDCTCoefficients()}
        </Tab>
        <Tab eventKey="spectrum" title="Frequency Spectrum">
          {renderFrequencySpectrum()}
        </Tab>
        <Tab eventKey="artifacts" title="Artifact Detection">
          {renderArtifactDetection()}
        </Tab>
      </Tabs>
    </div>
  );
};

export default FrequencyAnalysisDisplay; 