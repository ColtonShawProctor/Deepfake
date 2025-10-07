/**
 * HeatmapOverlay - Interactive heatmap overlay component for deepfake detection visualization
 * Displays attention maps and manipulation regions with interactive features
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { motion } from 'framer-motion';

const HeatmapOverlay = ({
  data,
  opacity = 0.9,
  colorMap = 'fire',
  interactive = true,
  onRegionClick,
  onRegionHover,
  className = '',
  style = {}
}) => {
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  const [isHovering, setIsHovering] = useState(false);
  const [hoveredRegion, setHoveredRegion] = useState(null);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });

  // Color maps for different visualization styles
  const colorMaps = {
    plasma: [
      [0.0, '#0d0887'],
      [0.1, '#46039f'],
      [0.2, '#7201a8'],
      [0.3, '#9c179e'],
      [0.4, '#bd3786'],
      [0.5, '#d8576b'],
      [0.6, '#ed7953'],
      [0.7, '#fb9f3a'],
      [0.8, '#fdca26'],
      [0.9, '#f0f921'],
      [1.0, '#f0f921']
    ],
    viridis: [
      [0.0, '#440154'],
      [0.1, '#482777'],
      [0.2, '#3f4a8a'],
      [0.3, '#31678e'],
      [0.4, '#26838e'],
      [0.5, '#1f9d8a'],
      [0.6, '#6cce5a'],
      [0.7, '#b6de2b'],
      [0.8, '#fee825'],
      [0.9, '#f0f921'],
      [1.0, '#f0f921']
    ],
    hot: [
      [0.0, '#000000'],
      [0.1, '#220000'],
      [0.2, '#440000'],
      [0.3, '#660000'],
      [0.4, '#880000'],
      [0.5, '#aa0000'],
      [0.6, '#cc0000'],
      [0.7, '#ee0000'],
      [0.8, '#ff2200'],
      [0.9, '#ff4400'],
      [1.0, '#ff6600']
    ],
    cool: [
      [0.0, '#00ffff'],
      [0.5, '#0080ff'],
      [1.0, '#0000ff']
    ],
    deepfake: [
      [0.0, '#000000'],
      [0.1, '#1a0033'],
      [0.2, '#330066'],
      [0.3, '#4d0099'],
      [0.4, '#6600cc'],
      [0.5, '#8000ff'],
      [0.6, '#9933ff'],
      [0.7, '#b366ff'],
      [0.8, '#cc99ff'],
      [0.9, '#e6ccff'],
      [1.0, '#ffffff']
    ],
    fire: [
      [0.0, '#000000'],
      [0.1, '#330000'],
      [0.2, '#660000'],
      [0.3, '#990000'],
      [0.4, '#cc0000'],
      [0.5, '#ff0000'],
      [0.6, '#ff3300'],
      [0.7, '#ff6600'],
      [0.8, '#ff9900'],
      [0.9, '#ffcc00'],
      [1.0, '#ffff00']
    ]
  };

  // Get color for a given intensity value
  const getColor = useCallback((intensity) => {
    const map = colorMaps[colorMap] || colorMaps.plasma;
    
    // Clamp intensity to [0, 1]
    intensity = Math.max(0, Math.min(1, intensity));
    
    // Find the two colors to interpolate between
    for (let i = 0; i < map.length - 1; i++) {
      const [pos1, color1] = map[i];
      const [pos2, color2] = map[i + 1];
      
      if (intensity >= pos1 && intensity <= pos2) {
        // Linear interpolation
        const t = (intensity - pos1) / (pos2 - pos1);
        return interpolateColor(color1, color2, t);
      }
    }
    
    return map[map.length - 1][1]; // Fallback to last color
  }, [colorMap]);

  // Interpolate between two hex colors
  const interpolateColor = (color1, color2, t) => {
    const hex1 = color1.replace('#', '');
    const hex2 = color2.replace('#', '');
    
    const r1 = parseInt(hex1.substr(0, 2), 16);
    const g1 = parseInt(hex1.substr(2, 2), 16);
    const b1 = parseInt(hex1.substr(4, 2), 16);
    
    const r2 = parseInt(hex2.substr(0, 2), 16);
    const g2 = parseInt(hex2.substr(2, 2), 16);
    const b2 = parseInt(hex2.substr(4, 2), 16);
    
    const r = Math.round(r1 + (r2 - r1) * t);
    const g = Math.round(g1 + (g2 - g1) * t);
    const b = Math.round(b1 + (b2 - b1) * t);
    
    return `rgb(${r}, ${g}, ${b})`;
  };

  // Draw heatmap on canvas
  const drawHeatmap = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !data) return;

    const ctx = canvas.getContext('2d');
    const { width, height } = dimensions;
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    // Create image data for heatmap
    const imageData = ctx.createImageData(width, height);
    const pixels = imageData.data;
    
    // Normalize data if needed
    const normalizedData = normalizeData(data);
    
    // Draw heatmap
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const dataX = Math.floor((x / width) * normalizedData[0].length);
        const dataY = Math.floor((y / height) * normalizedData.length);
        
        const intensity = normalizedData[dataY]?.[dataX] || 0;
        const color = getColor(intensity);
        
        const pixelIndex = (y * width + x) * 4;
        const rgb = color.match(/\d+/g);
        
        pixels[pixelIndex] = parseInt(rgb[0]);     // R
        pixels[pixelIndex + 1] = parseInt(rgb[1]); // G
        pixels[pixelIndex + 2] = parseInt(rgb[2]); // B
        pixels[pixelIndex + 3] = Math.round(255 * opacity); // A
      }
    }
    
    ctx.putImageData(imageData, 0, 0);
  }, [data, dimensions, opacity, getColor]);

  // Normalize data to [0, 1] range with enhanced contrast
  const normalizeData = (data) => {
    if (!data || !Array.isArray(data)) return [];
    
    const flatData = data.flat();
    const min = Math.min(...flatData);
    const max = Math.max(...flatData);
    const range = max - min;
    
    if (range === 0) return data;
    
    // Apply power curve to enhance contrast for high-intensity areas
    const power = 2.5; // Higher values make high-intensity areas more prominent
    
    return data.map(row => 
      row.map(value => {
        const normalized = (value - min) / range;
        // Apply power curve to enhance contrast
        return Math.pow(normalized, 1/power);
      })
    );
  };

  // Handle mouse events
  const handleMouseMove = useCallback((event) => {
    if (!interactive || !data) return;
    
    const rect = canvasRef.current.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    
    const { width, height } = dimensions;
    const dataX = Math.floor((x / width) * data[0].length);
    const dataY = Math.floor((y / height) * data.length);
    
    if (dataY >= 0 && dataY < data.length && dataX >= 0 && dataX < data[0].length) {
      const intensity = data[dataY][dataX];
      const normalizedIntensity = normalizeData(data)[dataY][dataX];
      
      const region = {
        x: dataX,
        y: dataY,
        intensity,
        normalizedIntensity,
        color: getColor(normalizedIntensity)
      };
      
      setHoveredRegion(region);
      
      if (onRegionHover) {
        onRegionHover(region);
      }
    }
  }, [interactive, data, dimensions, onRegionHover, getColor]);

  const handleMouseLeave = useCallback(() => {
    setIsHovering(false);
    setHoveredRegion(null);
  }, []);

  const handleClick = useCallback((event) => {
    if (!interactive || !data || !onRegionClick) return;
    
    const rect = canvasRef.current.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    
    const { width, height } = dimensions;
    const dataX = Math.floor((x / width) * data[0].length);
    const dataY = Math.floor((y / height) * data.length);
    
    if (dataY >= 0 && dataY < data.length && dataX >= 0 && dataX < data[0].length) {
      const intensity = data[dataY][dataX];
      const normalizedIntensity = normalizeData(data)[dataY][dataX];
      
      const region = {
        x: dataX,
        y: dataY,
        intensity,
        normalizedIntensity,
        color: getColor(normalizedIntensity)
      };
      
      onRegionClick(region);
    }
  }, [interactive, data, dimensions, onRegionClick, getColor]);

  // Update dimensions when container size changes
  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect();
        setDimensions({ width: rect.width, height: rect.height });
      }
    };

    updateDimensions();
    window.addEventListener('resize', updateDimensions);
    return () => window.removeEventListener('resize', updateDimensions);
  }, []);

  // Redraw heatmap when data or dimensions change
  useEffect(() => {
    drawHeatmap();
  }, [drawHeatmap]);

  // Set canvas size
  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas && dimensions.width > 0 && dimensions.height > 0) {
      canvas.width = dimensions.width;
      canvas.height = dimensions.height;
    }
  }, [dimensions]);

  return (
    <div 
      ref={containerRef}
      className={`heatmap-overlay ${className}`}
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        pointerEvents: interactive ? 'auto' : 'none',
        ...style
      }}
    >
      <canvas
        ref={canvasRef}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        onClick={handleClick}
        style={{
          width: '100%',
          height: '100%',
          cursor: interactive ? 'crosshair' : 'default'
        }}
      />
      
      {/* Hover tooltip */}
      {isHovering && hoveredRegion && interactive && (
        <motion.div
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.8 }}
          className="heatmap-tooltip"
          style={{
            position: 'absolute',
            left: `${(hoveredRegion.x / data[0].length) * 100}%`,
            top: `${(hoveredRegion.y / data.length) * 100}%`,
            transform: 'translate(-50%, -100%)',
            background: 'rgba(0, 0, 0, 0.8)',
            color: 'white',
            padding: '8px 12px',
            borderRadius: '6px',
            fontSize: '12px',
            pointerEvents: 'none',
            zIndex: 1000,
            whiteSpace: 'nowrap'
          }}
        >
          <div>Intensity: {(hoveredRegion.normalizedIntensity * 100).toFixed(1)}%</div>
          <div>Position: ({hoveredRegion.x}, {hoveredRegion.y})</div>
        </motion.div>
      )}
    </div>
  );
};

export default HeatmapOverlay;
