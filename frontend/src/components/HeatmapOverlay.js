import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Card, Button, ButtonGroup, Badge, Slider, Form } from 'react-bootstrap';
import { FaSearch, FaSearchMinus, FaExpand, FaCompress, FaDownload, FaEye, FaEyeSlash } from 'react-icons/fa';
import './HeatmapOverlay.css';

const HeatmapOverlay = ({ 
  originalImage, 
  heatmapData, 
  modelName, 
  analysisType = 'attention',
  onClose,
  onExport 
}) => {
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [overlayOpacity, setOverlayOpacity] = useState(0.7);
  const [showOriginal, setShowOriginal] = useState(true);
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [colorScheme, setColorScheme] = useState('jet');
  const [isFullscreen, setIsFullscreen] = useState(false);
  
  const canvasRef = useRef(null);
  const containerRef = useRef(null);

  // Color schemes for different analysis types
  const colorSchemes = {
    jet: ['#000080', '#0000ff', '#0080ff', '#00ffff', '#00ff80', '#80ff00', '#ffff00', '#ff8000', '#ff0000', '#800000'],
    hot: ['#000000', '#330000', '#660000', '#990000', '#cc0000', '#ff0000', '#ff3300', '#ff6600', '#ff9900', '#ffcc00', '#ffff00'],
    cool: ['#00ffff', '#00ccff', '#0099ff', '#0066ff', '#0033ff', '#0000ff', '#3300ff', '#6600ff', '#9900ff', '#cc00ff', '#ff00ff'],
    viridis: ['#440154', '#482878', '#3e4989', '#31688e', '#26828e', '#1f9e89', '#35b779', '#6ece58', '#b5de2b', '#fde725'],
    plasma: ['#0d0887', '#46039f', '#7201a8', '#9c179e', '#bd3786', '#d8576b', '#ed7953', '#fb9b3a', '#fdca26', '#f0f921']
  };

  // Generate heatmap canvas
  const generateHeatmap = useCallback(() => {
    if (!canvasRef.current || !heatmapData) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const { width, height } = canvas;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    if (!showHeatmap) return;

    // Create gradient for heatmap
    const gradient = ctx.createLinearGradient(0, 0, width, 0);
    const colors = colorSchemes[colorScheme];
    colors.forEach((color, index) => {
      gradient.addColorStop(index / (colors.length - 1), color);
    });

    // Draw heatmap
    const imageData = ctx.createImageData(width, height);
    const data = imageData.data;

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const heatmapX = Math.floor((x - pan.x) / zoom);
        const heatmapY = Math.floor((y - pan.y) / zoom);
        
        if (heatmapX >= 0 && heatmapX < heatmapData.width && 
            heatmapY >= 0 && heatmapY < heatmapData.height) {
          const intensity = heatmapData.data[heatmapY * heatmapData.width + heatmapX];
          const colorIndex = Math.floor(intensity * (colors.length - 1));
          const color = colors[colorIndex];
          
          const rgb = hexToRgb(color);
          const alpha = intensity * overlayOpacity * 255;
          
          const pixelIndex = (y * width + x) * 4;
          data[pixelIndex] = rgb.r;     // R
          data[pixelIndex + 1] = rgb.g; // G
          data[pixelIndex + 2] = rgb.b; // B
          data[pixelIndex + 3] = alpha; // A
        }
      }
    }

    ctx.putImageData(imageData, 0, 0);
  }, [heatmapData, zoom, pan, overlayOpacity, colorScheme, showHeatmap]);

  // Convert hex color to RGB
  const hexToRgb = (hex) => {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? {
      r: parseInt(result[1], 16),
      g: parseInt(result[2], 16),
      b: parseInt(result[3], 16)
    } : { r: 0, g: 0, b: 0 };
  };

  // Mouse event handlers
  const handleMouseDown = (e) => {
    setIsDragging(true);
    setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
  };

  const handleMouseMove = (e) => {
    if (!isDragging) return;
    
    setPan({
      x: e.clientX - dragStart.x,
      y: e.clientY - dragStart.y
    });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  // Wheel event for zoom
  const handleWheel = (e) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    const newZoom = Math.max(0.1, Math.min(5, zoom * delta));
    setZoom(newZoom);
  };

  // Zoom controls
  const zoomIn = () => setZoom(prev => Math.min(5, prev * 1.2));
  const zoomOut = () => setZoom(prev => Math.max(0.1, prev / 1.2));
  const resetView = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  // Fullscreen toggle
  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      containerRef.current?.requestFullscreen();
      setIsFullscreen(true);
    } else {
      document.exitFullscreen();
      setIsFullscreen(false);
    }
  };

  // Export functionality
  const handleExport = () => {
    if (!canvasRef.current) return;
    
    const canvas = canvasRef.current;
    const link = document.createElement('a');
    link.download = `${modelName}_${analysisType}_heatmap.png`;
    link.href = canvas.toDataURL();
    link.click();
    
    if (onExport) {
      onExport(canvas.toDataURL());
    }
  };

  // Update canvas size and redraw
  useEffect(() => {
    if (!canvasRef.current || !originalImage) return;

    const canvas = canvasRef.current;
    const container = containerRef.current;
    
    if (container) {
      const rect = container.getBoundingClientRect();
      canvas.width = rect.width;
      canvas.height = rect.height;
    }

    generateHeatmap();
  }, [originalImage, generateHeatmap]);

  // Redraw on changes
  useEffect(() => {
    generateHeatmap();
  }, [generateHeatmap]);

  // Handle fullscreen changes
  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };

    document.addEventListener('fullscreenchange', handleFullscreenChange);
    return () => document.removeEventListener('fullscreenchange', handleFullscreenChange);
  }, []);

  if (!originalImage || !heatmapData) {
    return (
      <div className="heatmap-overlay">
        <Card>
          <Card.Body className="text-center">
            <p>No heatmap data available</p>
            <Button onClick={onClose}>Close</Button>
          </Card.Body>
        </Card>
      </div>
    );
  }

  return (
    <div className="heatmap-overlay">
      <Card className="heatmap-card">
        <Card.Header className="heatmap-header">
          <div className="header-content">
            <h5>{modelName} - {analysisType.charAt(0).toUpperCase() + analysisType.slice(1)} Analysis</h5>
            <div className="header-badges">
              <Badge bg="info">{analysisType}</Badge>
              <Badge bg="secondary">Zoom: {Math.round(zoom * 100)}%</Badge>
            </div>
          </div>
          <ButtonGroup size="sm">
            <Button variant="outline-secondary" onClick={onClose}>
              ×
            </Button>
          </ButtonGroup>
        </Card.Header>

        <Card.Body className="heatmap-body">
          {/* Controls */}
          <div className="heatmap-controls">
            <div className="control-group">
              <ButtonGroup size="sm">
                <Button variant="outline-primary" onClick={zoomIn} title="Zoom In">
                  <FaSearch />
                </Button>
                <Button variant="outline-primary" onClick={zoomOut} title="Zoom Out">
                  <FaSearchMinus />
                </Button>
                <Button variant="outline-secondary" onClick={resetView} title="Reset View">
                  Reset
                </Button>
              </ButtonGroup>
            </div>

            <div className="control-group">
              <ButtonGroup size="sm">
                <Button 
                  variant={showOriginal ? "primary" : "outline-primary"} 
                  onClick={() => setShowOriginal(!showOriginal)}
                  title="Toggle Original Image"
                >
                  <FaEye />
                </Button>
                <Button 
                  variant={showHeatmap ? "primary" : "outline-primary"} 
                  onClick={() => setShowHeatmap(!showHeatmap)}
                  title="Toggle Heatmap"
                >
                  <FaEyeSlash />
                </Button>
                <Button variant="outline-secondary" onClick={toggleFullscreen} title="Toggle Fullscreen">
                  {isFullscreen ? <FaCompress /> : <FaExpand />}
                </Button>
                <Button variant="outline-success" onClick={handleExport} title="Export Heatmap">
                  <FaDownload />
                </Button>
              </ButtonGroup>
            </div>
          </div>

          {/* Settings Panel */}
          <div className="heatmap-settings">
            <div className="setting-group">
              <label>Overlay Opacity</label>
              <Slider
                min={0}
                max={1}
                step={0.1}
                value={overlayOpacity}
                onChange={setOverlayOpacity}
                className="opacity-slider"
              />
              <span className="setting-value">{Math.round(overlayOpacity * 100)}%</span>
            </div>

            <div className="setting-group">
              <label>Color Scheme</label>
              <Form.Select 
                size="sm" 
                value={colorScheme} 
                onChange={(e) => setColorScheme(e.target.value)}
                className="color-scheme-select"
              >
                {Object.keys(colorSchemes).map(scheme => (
                  <option key={scheme} value={scheme}>
                    {scheme.charAt(0).toUpperCase() + scheme.slice(1)}
                  </option>
                ))}
              </Form.Select>
            </div>
          </div>

          {/* Canvas Container */}
          <div 
            ref={containerRef}
            className="canvas-container"
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
            onWheel={handleWheel}
          >
            <canvas
              ref={canvasRef}
              className="heatmap-canvas"
              style={{
                cursor: isDragging ? 'grabbing' : 'grab',
                backgroundImage: showOriginal ? `url(${originalImage})` : 'none',
                backgroundSize: 'contain',
                backgroundRepeat: 'no-repeat',
                backgroundPosition: 'center'
              }}
            />
            
            {/* Zoom indicator */}
            <div className="zoom-indicator">
              <Badge bg="dark">
                {Math.round(zoom * 100)}% | Pan: ({Math.round(pan.x)}, {Math.round(pan.y)})
              </Badge>
            </div>
          </div>

          {/* Instructions */}
          <div className="heatmap-instructions">
            <small className="text-muted">
              <strong>Controls:</strong> Mouse wheel to zoom • Drag to pan • Use controls above to adjust visualization
            </small>
          </div>
        </Card.Body>
      </Card>
    </div>
  );
};

export default HeatmapOverlay; 