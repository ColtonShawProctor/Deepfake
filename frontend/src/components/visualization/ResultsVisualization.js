/**
 * ResultsVisualization - Advanced visualization component for multi-model analysis results
 * Features interactive heatmaps, overlays, and sophisticated visual comparisons
 */

import React, { useState, useEffect, useRef, forwardRef, useImperativeHandle } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  ZoomIn, 
  ZoomOut, 
  RotateCw, 
  Layers, 
  Eye, 
  EyeOff,
  Download,
  Maximize2,
  Grid3x3,
  BarChart,
  Activity
} from 'lucide-react';

import HeatmapOverlay from './HeatmapOverlay';
import FrequencyVisualization from './FrequencyVisualization';
import ConfidenceGauge from './ConfidenceGauge';
import ModelVoting from './ModelVoting';
import InteractiveImageViewer from './InteractiveImageViewer';

const ResultsVisualization = forwardRef(({ 
  results, 
  image, 
  mode = 'overview',
  settings = {},
  userType = 'general',
  onModeChange,
  onSettingsChange 
}, ref) => {
  // State for visualization controls
  const [zoomLevel, setZoomLevel] = useState(1);
  const [imagePosition, setImagePosition] = useState({ x: 0, y: 0 });
  const [activeModel, setActiveModel] = useState('ensemble');
  const [showOverlays, setShowOverlays] = useState(true);
  const [overlayOpacity, setOverlayOpacity] = useState(0.7);
  const [viewMode, setViewMode] = useState(mode);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [hoveredRegion, setHoveredRegion] = useState(null);

  // Refs
  const containerRef = useRef(null);
  const imageRef = useRef(null);
  const canvasRef = useRef(null);

  // Expose methods to parent components
  useImperativeHandle(ref, () => ({
    exportVisualization: () => exportCurrentView(),
    resetView: () => resetImageView(),
    focusOnRegion: (region) => focusOnImageRegion(region),
    toggleFullscreen: () => setIsFullscreen(!isFullscreen)
  }));

  // Extract results data
  const ensembleResult = results?.ensemble || {};
  const individualResults = results?.individual || {};
  const spatialResults = results?.spatial || {};
  const frequencyResults = results?.frequency || {};

  // Visualization modes configuration
  const visualizationModes = {
    overview: {
      title: 'Analysis Overview',
      description: 'Complete ensemble analysis with all model insights',
      layout: 'grid',
      showControls: true
    },
    detailed: {
      title: 'Detailed Analysis',
      description: 'In-depth view with interactive exploration',
      layout: 'detailed',
      showControls: true
    },
    comparison: {
      title: 'Model Comparison',
      description: 'Side-by-side comparison of all models',
      layout: 'split',
      showControls: true
    },
    frequency: {
      title: 'Frequency Analysis',
      description: 'Focus on frequency-domain insights',
      layout: 'frequency',
      showControls: true
    }
  };

  const currentMode = visualizationModes[viewMode] || visualizationModes.overview;

  // Handle mode changes
  useEffect(() => {
    if (mode !== viewMode) {
      setViewMode(mode);
      if (onModeChange) {
        onModeChange(mode);
      }
    }
  }, [mode, viewMode, onModeChange]);

  // Image interaction handlers
  const handleZoomIn = () => {
    setZoomLevel(prev => Math.min(prev * 1.2, 5));
  };

  const handleZoomOut = () => {
    setZoomLevel(prev => Math.max(prev / 1.2, 0.5));
  };

  const resetImageView = () => {
    setZoomLevel(1);
    setImagePosition({ x: 0, y: 0 });
  };

  const focusOnImageRegion = (region) => {
    if (region && imageRef.current) {
      const imageRect = imageRef.current.getBoundingClientRect();
      const { x, y, width, height } = region;
      
      // Calculate zoom and position to focus on region
      const targetZoom = Math.min(imageRect.width / width, imageRect.height / height) * 0.8;
      const centerX = x + width / 2;
      const centerY = y + height / 2;
      
      setZoomLevel(Math.min(targetZoom, 3));
      setImagePosition({
        x: imageRect.width / 2 - centerX * targetZoom,
        y: imageRect.height / 2 - centerY * targetZoom
      });
    }
  };

  const exportCurrentView = () => {
    if (canvasRef.current && imageRef.current) {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      
      // Set canvas size to match image
      canvas.width = imageRef.current.naturalWidth;
      canvas.height = imageRef.current.naturalHeight;
      
      // Draw image
      ctx.drawImage(imageRef.current, 0, 0);
      
      // Add overlays if enabled
      if (showOverlays && activeModel !== 'original') {
        // Add heatmap overlay logic here
        drawHeatmapOverlay(ctx, canvas.width, canvas.height);
      }
      
      // Trigger download
      const link = document.createElement('a');
      link.download = `deepfake-analysis-${activeModel}-${Date.now()}.png`;
      link.href = canvas.toDataURL();
      link.click();
    }
  };

  const drawHeatmapOverlay = (ctx, width, height) => {
    // Implementation for drawing heatmap overlay on canvas
    const heatmapData = getHeatmapData(activeModel);
    if (!heatmapData) return;

    const imageData = ctx.createImageData(width, height);
    // Apply heatmap colors based on confidence scores
    // This is a simplified implementation
    for (let i = 0; i < imageData.data.length; i += 4) {
      const x = (i / 4) % width;
      const y = Math.floor((i / 4) / width);
      const intensity = getHeatmapIntensity(x, y, heatmapData);
      
      imageData.data[i] = 255 * intensity;     // Red
      imageData.data[i + 1] = 0;               // Green
      imageData.data[i + 2] = 0;               // Blue
      imageData.data[i + 3] = 100 * intensity; // Alpha
    }
    
    ctx.putImageData(imageData, 0, 0);
  };

  const getHeatmapData = (modelName) => {
    if (modelName === 'ensemble') {
      return ensembleResult.attentionMaps;
    }
    return individualResults[modelName]?.attentionMaps;
  };

  const getHeatmapIntensity = (x, y, heatmapData) => {
    // Simplified heatmap intensity calculation
    if (!heatmapData || !heatmapData.length) return 0;
    
    const normalizedX = Math.floor((x / image.width) * heatmapData[0].length);
    const normalizedY = Math.floor((y / image.height) * heatmapData.length);
    
    if (normalizedY < heatmapData.length && normalizedX < heatmapData[0].length) {
      return heatmapData[normalizedY][normalizedX] || 0;
    }
    return 0;
  };

  // Get available models for visualization
  const availableModels = [
    { id: 'ensemble', name: 'Ensemble', confidence: ensembleResult.confidence },
    ...Object.entries(individualResults).map(([key, result]) => ({
      id: key,
      name: key.charAt(0).toUpperCase() + key.slice(1),
      confidence: result.confidence
    })),
    { id: 'original', name: 'Original', confidence: null }
  ];

  // Render different layouts based on mode
  const renderVisualizationContent = () => {
    switch (currentMode.layout) {
      case 'grid':
        return renderGridLayout();
      case 'detailed':
        return renderDetailedLayout();
      case 'split':
        return renderSplitLayout();
      case 'frequency':
        return renderFrequencyLayout();
      default:
        return renderGridLayout();
    }
  };

  const renderGridLayout = () => (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Main Image Visualization */}
      <div className="space-y-4">
        <div className="bg-slate-50 rounded-lg p-4">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-semibold text-slate-900">
              {activeModel === 'ensemble' ? 'Ensemble Analysis' : 
               activeModel === 'original' ? 'Original Image' :
               `${availableModels.find(m => m.id === activeModel)?.name} Analysis`}
            </h3>
            <div className="flex items-center space-x-2">
              <button
                onClick={handleZoomOut}
                className="p-1 text-slate-500 hover:text-slate-700"
                title="Zoom out"
              >
                <ZoomOut className="w-4 h-4" />
              </button>
              <span className="text-sm text-slate-600">{Math.round(zoomLevel * 100)}%</span>
              <button
                onClick={handleZoomIn}
                className="p-1 text-slate-500 hover:text-slate-700"
                title="Zoom in"
              >
                <ZoomIn className="w-4 h-4" />
              </button>
              <button
                onClick={() => setShowOverlays(!showOverlays)}
                className={`p-1 transition-colors ${showOverlays ? 'text-blue-600' : 'text-slate-500'}`}
                title="Toggle overlays"
              >
                {showOverlays ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />}
              </button>
            </div>
          </div>
          
          <InteractiveImageViewer
            ref={imageRef}
            image={image}
            zoomLevel={zoomLevel}
            position={imagePosition}
            onPositionChange={setImagePosition}
            onHover={setHoveredRegion}
            className="w-full h-96 border border-slate-200 rounded-lg overflow-hidden"
          >
            {showOverlays && activeModel !== 'original' && (
              <HeatmapOverlay
                data={getHeatmapData(activeModel)}
                opacity={overlayOpacity}
                colorMap="plasma"
                onRegionClick={focusOnImageRegion}
              />
            )}
          </InteractiveImageViewer>
          
          {/* Model Selection */}
          <div className="mt-4">
            <div className="flex flex-wrap gap-2">
              {availableModels.map((model) => (
                <button
                  key={model.id}
                  onClick={() => setActiveModel(model.id)}
                  className={`px-3 py-1 rounded-full text-sm font-medium transition-colors ${
                    activeModel === model.id
                      ? 'bg-blue-600 text-white'
                      : 'bg-slate-200 text-slate-700 hover:bg-slate-300'
                  }`}
                >
                  {model.name}
                  {model.confidence !== null && (
                    <span className="ml-1 text-xs opacity-75">
                      {(model.confidence * 100).toFixed(1)}%
                    </span>
                  )}
                </button>
              ))}
            </div>
          </div>
        </div>
        
        {/* Overlay Controls */}
        {showOverlays && activeModel !== 'original' && (
          <div className="bg-white rounded-lg p-4 border border-slate-200">
            <h4 className="font-medium text-slate-900 mb-3">Visualization Controls</h4>
            <div className="space-y-3">
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1">
                  Overlay Opacity: {Math.round(overlayOpacity * 100)}%
                </label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={overlayOpacity}
                  onChange={(e) => setOverlayOpacity(parseFloat(e.target.value))}
                  className="w-full"
                />
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Analysis Metrics */}
      <div className="space-y-4">
        {/* Ensemble Confidence */}
        <div className="bg-white rounded-lg p-4 border border-slate-200">
          <ConfidenceGauge
            confidence={ensembleResult.confidence || 0}
            prediction={ensembleResult.isDeepfake}
            uncertainty={ensembleResult.uncertainty}
            title="Ensemble Prediction"
          />
        </div>

        {/* Model Voting */}
        <div className="bg-white rounded-lg p-4 border border-slate-200">
          <ModelVoting
            results={individualResults}
            ensembleResult={ensembleResult}
            onModelSelect={setActiveModel}
            activeModel={activeModel}
          />
        </div>

        {/* Frequency Analysis Preview */}
        {frequencyResults && Object.keys(frequencyResults).length > 0 && (
          <div className="bg-white rounded-lg p-4 border border-slate-200">
            <h4 className="font-medium text-slate-900 mb-3">Frequency Domain</h4>
            <FrequencyVisualization
              data={frequencyResults}
              compact={true}
              onExpand={() => setViewMode('frequency')}
            />
          </div>
        )}

        {/* Detailed Metrics */}
        {userType !== 'general' && (
          <div className="bg-white rounded-lg p-4 border border-slate-200">
            <h4 className="font-medium text-slate-900 mb-3">Advanced Metrics</h4>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-slate-600">Domain Agreement:</span>
                <span className="font-medium">
                  {(ensembleResult.domainAgreement * 100).toFixed(1)}%
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-600">Processing Time:</span>
                <span className="font-medium">{results.metadata?.processingTime}ms</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-600">Models Used:</span>
                <span className="font-medium">{Object.keys(individualResults).length}</span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );

  const renderDetailedLayout = () => (
    <div className="space-y-6">
      {/* Full-width image viewer */}
      <div className="bg-slate-50 rounded-lg p-6">
        <InteractiveImageViewer
          ref={imageRef}
          image={image}
          zoomLevel={zoomLevel}
          position={imagePosition}
          onPositionChange={setImagePosition}
          onHover={setHoveredRegion}
          className="w-full h-[600px] border border-slate-200 rounded-lg overflow-hidden"
        >
          {showOverlays && activeModel !== 'original' && (
            <HeatmapOverlay
              data={getHeatmapData(activeModel)}
              opacity={overlayOpacity}
              colorMap="plasma"
              onRegionClick={focusOnImageRegion}
              interactive={true}
            />
          )}
        </InteractiveImageViewer>
      </div>

      {/* Detailed controls and information */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <ConfidenceGauge
          confidence={ensembleResult.confidence || 0}
          prediction={ensembleResult.isDeepfake}
          uncertainty={ensembleResult.uncertainty}
          title="Ensemble Prediction"
          detailed={true}
        />
        
        <ModelVoting
          results={individualResults}
          ensembleResult={ensembleResult}
          onModelSelect={setActiveModel}
          activeModel={activeModel}
          detailed={true}
        />

        {frequencyResults && (
          <FrequencyVisualization
            data={frequencyResults}
            compact={false}
          />
        )}
      </div>
    </div>
  );

  const renderSplitLayout = () => (
    <div className="grid grid-cols-2 lg:grid-cols-3 gap-4">
      {availableModels.slice(0, -1).map((model) => (
        <div key={model.id} className="bg-white rounded-lg p-4 border border-slate-200">
          <h4 className="font-medium text-slate-900 mb-2">{model.name}</h4>
          <div className="aspect-square bg-slate-50 rounded-lg mb-3 overflow-hidden">
            <InteractiveImageViewer
              image={image}
              zoomLevel={1}
              position={{ x: 0, y: 0 }}
              className="w-full h-full"
            >
              <HeatmapOverlay
                data={getHeatmapData(model.id)}
                opacity={0.6}
                colorMap="plasma"
              />
            </InteractiveImageViewer>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-slate-900">
              {(model.confidence * 100).toFixed(1)}%
            </div>
            <div className="text-sm text-slate-600">Confidence ({ensembleResult.isDeepfake ? 'FAKE' : 'REAL'})</div>
          </div>
        </div>
      ))}
    </div>
  );

  const renderFrequencyLayout = () => (
    <div className="space-y-6">
      <FrequencyVisualization
        data={frequencyResults}
        image={image}
        fullscreen={true}
        interactive={true}
      />
    </div>
  );

  return (
    <div className={`relative ${isFullscreen ? 'fixed inset-0 z-50 bg-white' : ''}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-semibold text-slate-900">{currentMode.title}</h2>
          <p className="text-slate-600 text-sm">{currentMode.description}</p>
        </div>
        
        <div className="flex items-center space-x-2">
          {/* Mode Selector */}
          <select
            value={viewMode}
            onChange={(e) => setViewMode(e.target.value)}
            className="px-3 py-1 border border-slate-300 rounded text-sm"
          >
            {Object.entries(visualizationModes).map(([key, mode]) => (
              <option key={key} value={key}>{mode.title}</option>
            ))}
          </select>
          
          <button
            onClick={exportCurrentView}
            className="p-2 text-slate-600 hover:text-slate-900 hover:bg-slate-100 
                     rounded-lg transition-colors duration-200"
            title="Export visualization"
          >
            <Download className="w-4 h-4" />
          </button>
          
          <button
            onClick={() => setIsFullscreen(!isFullscreen)}
            className="p-2 text-slate-600 hover:text-slate-900 hover:bg-slate-100 
                     rounded-lg transition-colors duration-200"
            title="Toggle fullscreen"
          >
            <Maximize2 className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Main Content */}
      <div ref={containerRef} className="relative">
        {renderVisualizationContent()}
      </div>

      {/* Hidden canvas for export */}
      <canvas ref={canvasRef} className="hidden" />

      {/* Hover Information */}
      <AnimatePresence>
        {hoveredRegion && (
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            className="fixed bottom-4 left-4 bg-black text-white p-3 rounded-lg z-50"
          >
            <div className="text-sm">
              <div>Position: ({hoveredRegion.x}, {hoveredRegion.y})</div>
              <div>Confidence: {(hoveredRegion.confidence * 100).toFixed(1)}% ({ensembleResult.isDeepfake ? 'FAKE' : 'REAL'})</div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
});

ResultsVisualization.displayName = 'ResultsVisualization';

export default ResultsVisualization;