import React from 'react';

const FrequencyVisualization = ({ data, compact = false }) => {
  if (!data) {
    return null;
  }

  return (
    <div className={`frequency-visualization ${compact ? 'compact' : ''}`}>
      <h6 className="mb-2">Frequency Analysis</h6>
      <div className="text-muted small">
        Frequency analysis data not available
      </div>
    </div>
  );
};

export default FrequencyVisualization;
