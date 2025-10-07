import React from 'react';

const ModelVoting = ({ 
  results = {}, 
  ensembleResult = {}, 
  onModelSelect = () => {}, 
  activeModel = 'ensemble',
  detailed = false 
}) => {
  const getConfidenceColor = (conf) => {
    if (conf >= 80) return 'success';
    if (conf >= 60) return 'warning';
    return 'danger';
  };

  return (
    <div className={`model-voting ${detailed ? 'detailed' : ''}`}>
      <h6 className="mb-3">Model Predictions</h6>
      <div className="row">
        {Object.entries(results).map(([modelName, result]) => (
          <div key={modelName} className="col-12 mb-2">
            <div 
              className={`card ${activeModel === modelName ? 'border-primary' : ''}`}
              style={{ cursor: 'pointer' }}
              onClick={() => onModelSelect(modelName)}
            >
              <div className="card-body p-2">
                <div className="d-flex justify-content-between align-items-center">
                  <span className="fw-bold">{modelName}</span>
                  <div className="d-flex align-items-center gap-2">
                    <span className={`badge bg-${getConfidenceColor(result.confidence)}`}>
                      {result.confidence?.toFixed(1)}%
                    </span>
                    <span className={`badge ${result.is_deepfake ? 'bg-danger' : 'bg-success'}`}>
                      {result.is_deepfake ? 'FAKE' : 'REAL'}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ModelVoting;
