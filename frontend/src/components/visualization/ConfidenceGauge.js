import React from 'react';

const ConfidenceGauge = ({ 
  confidence = 0, 
  prediction = false, 
  uncertainty = 0, 
  title = "Confidence", 
  detailed = false 
}) => {
  const getConfidenceColor = (conf) => {
    if (conf >= 80) return 'success';
    if (conf >= 60) return 'warning';
    return 'danger';
  };

  return (
    <div className={`confidence-gauge ${detailed ? 'detailed' : ''}`}>
      <h6 className="mb-2">{title}</h6>
      <div className="progress mb-2" style={{ height: '20px' }}>
        <div 
          className={`progress-bar bg-${getConfidenceColor(confidence)}`}
          role="progressbar"
          style={{ width: `${confidence}%` }}
          aria-valuenow={confidence}
          aria-valuemin="0"
          aria-valuemax="100"
        >
          {confidence.toFixed(1)}%
        </div>
      </div>
      <div className="d-flex justify-content-between">
        <small className="text-muted">
          {prediction ? 'FAKE' : 'REAL'}
        </small>
        {uncertainty > 0 && (
          <small className="text-muted">
            Uncertainty: {uncertainty.toFixed(1)}%
          </small>
        )}
      </div>
    </div>
  );
};

export default ConfidenceGauge;
