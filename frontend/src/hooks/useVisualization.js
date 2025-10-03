import { useState, useCallback } from 'react';

export const useVisualization = () => {
  const [visualizationMode, setVisualizationMode] = useState('standard');
  const [overlaySettings, setOverlaySettings] = useState({});
  const [heatmapSettings, setHeatmapSettings] = useState({});

  const updateSettings = useCallback((type, settings) => {
    if (type === 'overlay') {
      setOverlaySettings(prev => ({ ...prev, ...settings }));
    } else if (type === 'heatmap') {
      setHeatmapSettings(prev => ({ ...prev, ...settings }));
    }
  }, []);

  return {
    visualizationMode,
    setVisualizationMode,
    overlaySettings,
    heatmapSettings,
    updateSettings
  };
};





