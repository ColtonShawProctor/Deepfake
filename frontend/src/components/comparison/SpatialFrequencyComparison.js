/**
 * SpatialFrequencyComparison - Interactive comparison between spatial and frequency domain analysis
 * Showcases the unique perspectives of different model types
 */

import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Layers,
  Zap,
  Eye,
  BarChart3,
  TrendingUp,
  Split,
  Maximize2,
  Download,
  Play,
  Pause,
  RotateCcw,
  Settings,
  Info
} from 'lucide-react';

import FrequencyHeatmap from '../visualization/FrequencyHeatmap';
import SpatialHeatmap from '../visualization/SpatialHeatmap';
import DomainMetrics from '../metrics/DomainMetrics';
import InteractiveSlider from '../controls/InteractiveSlider';

const SpatialFrequencyComparison = ({ 
  results, 
  image, 
  spatialModels = ['xception', 'efficientnet'],
  frequencyModels = ['f3net'],
  onInsightSelect,
  className = ""
}) => {
  // View modes
  const [viewMode, setViewMode] = useState('split'); // split, overlay, animated, detailed
  const [activeRegion, setActiveRegion] = useState(null);
  const [comparisonMetric, setComparisonMetric] = useState('confidence');
  const [animationSpeed, setAnimationSpeed] = useState(1);
  const [isAnimating, setIsAnimating] = useState(false);
  const [overlayOpacity, setOverlayOpacity] = useState(0.7);
  const [showInsights, setShowInsights] = useState(true);
  
  // Analysis state
  const [spatialAnalysis, setSpatialAnalysis] = useState(null);
  const [frequencyAnalysis, setFrequencyAnalysis] = useState(null);
  const [domainAgreement, setDomainAgreement] = useState(null);
  const [keyDifferences, setKeyDifferences] = useState([]);
  
  const animationRef = useRef(null);
  const canvasRef = useRef(null);

  // Extract domain-specific results
  useEffect(() => {
    if (results) {
      // Aggregate spatial model results
      const spatialResults = spatialModels.reduce((acc, modelName) => {
        const modelResult = results.individual?.[modelName];
        if (modelResult) {
          acc.predictions.push(modelResult);
          if (modelResult.attentionMaps) {
            acc.attentionMaps.push(modelResult.attentionMaps);
          }
        }
        return acc;
      }, { predictions: [], attentionMaps: [] });

      // Aggregate frequency model results
      const frequencyResults = frequencyModels.reduce((acc, modelName) => {
        const modelResult = results.individual?.[modelName];
        if (modelResult) {
          acc.predictions.push(modelResult);
          if (modelResult.frequencyMaps) {
            acc.frequencyMaps.push(modelResult.frequencyMaps);
          }
        }
        return acc;
      }, { predictions: [], frequencyMaps: [] });

      setSpatialAnalysis(aggregateSpatialAnalysis(spatialResults));
      setFrequencyAnalysis(aggregateFrequencyAnalysis(frequencyResults));
      
      // Calculate domain agreement
      if (spatialResults.predictions.length > 0 && frequencyResults.predictions.length > 0) {
        const agreement = calculateDomainAgreement(spatialResults, frequencyResults);
        setDomainAgreement(agreement);
        setKeyDifferences(identifyKeyDifferences(spatialResults, frequencyResults));
      }
    }
  }, [results, spatialModels, frequencyModels]);

  const aggregateSpatialAnalysis = (spatialResults) => {
    if (!spatialResults.predictions.length) return null;

    const avgConfidence = spatialResults.predictions.reduce((sum, p) => sum + p.confidence, 0) / spatialResults.predictions.length;
    const consensus = spatialResults.predictions.filter(p => p.isDeepfake).length / spatialResults.predictions.length;
    
    return {
      confidence: avgConfidence,
      consensus,
      predictions: spatialResults.predictions,
      attentionMaps: spatialResults.attentionMaps,
      primaryIndicators: extractSpatialIndicators(spatialResults),
      strengths: ['Facial inconsistencies', 'Texture artifacts', 'Lighting anomalies'],
      weaknesses: ['Compression artifacts', 'Frequency-domain manipulation']
    };
  };

  const aggregateFrequencyAnalysis = (frequencyResults) => {
    if (!frequencyResults.predictions.length) return null;

    const avgConfidence = frequencyResults.predictions.reduce((sum, p) => sum + p.confidence, 0) / frequencyResults.predictions.length;
    const consensus = frequencyResults.predictions.filter(p => p.isDeepfake).length / frequencyResults.predictions.length;
    
    return {
      confidence: avgConfidence,
      consensus,
      predictions: frequencyResults.predictions,
      frequencyMaps: frequencyResults.frequencyMaps,
      primaryIndicators: extractFrequencyIndicators(frequencyResults),
      strengths: ['Compression artifacts', 'DCT anomalies', 'Frequency inconsistencies'],
      weaknesses: ['High-quality deepfakes', 'Uncompressed images']
    };
  };

  const calculateDomainAgreement = (spatial, frequency) => {
    const spatialAvg = spatial.predictions.reduce((sum, p) => sum + p.confidence, 0) / spatial.predictions.length;
    const frequencyAvg = frequency.predictions.reduce((sum, p) => sum + p.confidence, 0) / frequency.predictions.length;
    
    return {
      confidenceAlignment: 1 - Math.abs(spatialAvg - frequencyAvg) / 100,
      consensusAlignment: 1 - Math.abs(spatial.predictions.filter(p => p.isDeepfake).length / spatial.predictions.length - 
                                      frequency.predictions.filter(p => p.isDeepfake).length / frequency.predictions.length),
      overallAgreement: 1 - Math.abs(spatialAvg - frequencyAvg) / 100
    };
  };

  const extractSpatialIndicators = (spatialResults) => [
    'Facial boundary inconsistencies',
    'Texture blending artifacts',
    'Lighting direction mismatches',
    'Micro-expression anomalies'
  ];

  const extractFrequencyIndicators = (frequencyResults) => [
    'JPEG compression patterns',
    'DCT coefficient anomalies',
    'High-frequency noise',
    'Quantization artifacts'
  ];

  const identifyKeyDifferences = (spatial, frequency) => {
    const differences = [];
    
    const spatialConfidence = spatial.predictions.reduce((sum, p) => sum + p.confidence, 0) / spatial.predictions.length;
    const frequencyConfidence = frequency.predictions.reduce((sum, p) => sum + p.confidence, 0) / frequency.predictions.length;
    
    if (Math.abs(spatialConfidence - frequencyConfidence) > 20) {
      differences.push({
        type: 'confidence_divergence',
        severity: 'high',
        description: `Spatial models show ${spatialConfidence > frequencyConfidence ? 'higher' : 'lower'} confidence than frequency models`,
        impact: 'May indicate domain-specific artifacts'
      });
    }
    
    return differences;
  };

  // Animation control
  const toggleAnimation = () => {
    if (isAnimating) {
      setIsAnimating(false);
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    } else {
      setIsAnimating(true);
      animateComparison();
    }
  };

  const animateComparison = () => {
    let progress = 0;
    const animate = () => {
      progress += 0.01 * animationSpeed;
      if (progress >= 1) {
        progress = 0;
      }
      
      // Update opacity based on animation progress
      const spatialOpacity = Math.sin(progress * Math.PI * 2) * 0.3 + 0.7;
      const frequencyOpacity = Math.cos(progress * Math.PI * 2) * 0.3 + 0.7;
      
      // Apply animation updates here
      
      if (isAnimating) {
        animationRef.current = requestAnimationFrame(animate);
      }
    };
    animate();
  };

  const handleRegionSelect = (region, domain) => {
    setActiveRegion({ ...region, domain });
    if (onInsightSelect) {
      onInsightSelect({
        type: 'region_analysis',
        domain,
        region,
        spatialData: spatialAnalysis,
        frequencyData: frequencyAnalysis
      });
    }
  };

  const exportComparison = () => {
    if (canvasRef.current) {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      
      // Set canvas size
      canvas.width = 1920;
      canvas.height = 1080;
      
      // Draw comparison visualization
      drawComparisonToCanvas(ctx);
      
      // Download
      const link = document.createElement('a');
      link.download = `spatial-frequency-comparison-${Date.now()}.png`;
      link.href = canvas.toDataURL();
      link.click();
    }
  };

  const drawComparisonToCanvas = (ctx) => {
    // Implementation for drawing comprehensive comparison to canvas
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    
    // Add title
    ctx.fillStyle = '#1f2937';
    ctx.font = 'bold 32px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Spatial vs Frequency Domain Analysis', ctx.canvas.width / 2, 50);
    
    // Draw split view with both domains
    // This would include the actual image analysis visualization
  };

  const renderSplitView = () => (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Spatial Domain */}
      <motion.div
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        className="bg-white rounded-xl p-6 border border-slate-200 shadow-lg"
      >
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-blue-100 rounded-lg">
              <Layers className="w-6 h-6 text-blue-600" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-slate-900">Spatial Domain</h3>
              <p className="text-sm text-slate-600">Facial artifacts & texture analysis</p>
            </div>
          </div>
          <div className="text-right">
            <div className="text-2xl font-bold text-blue-600">
              {spatialAnalysis ? `${spatialAnalysis.confidence.toFixed(1)}%` : '--'}
            </div>
            <div className="text-sm text-slate-600">Confidence</div>
          </div>
        </div>

        <div className="relative mb-4">
          <div className="aspect-square bg-slate-50 rounded-lg overflow-hidden border border-slate-200">
            {image && spatialAnalysis && (
              <SpatialHeatmap
                image={image}
                attentionMaps={spatialAnalysis.attentionMaps}
                opacity={overlayOpacity}
                onRegionSelect={(region) => handleRegionSelect(region, 'spatial')}
                interactive={true}
              />
            )}
          </div>
          
          {activeRegion && activeRegion.domain === 'spatial' && (
            <div className="absolute top-2 right-2 bg-white/90 backdrop-blur-sm rounded-lg p-2 text-sm">
              <div className="font-medium">Region Analysis</div>
              <div>Confidence: {(activeRegion.confidence * 100).toFixed(1)}%</div>
            </div>
          )}
        </div>

        <div className="space-y-3">
          <div>
            <h4 className="font-medium text-slate-900 mb-2">Primary Indicators</h4>
            <div className="space-y-1">
              {spatialAnalysis?.primaryIndicators.map((indicator, index) => (
                <div key={index} className="flex items-center space-x-2 text-sm">
                  <div className="w-2 h-2 bg-blue-600 rounded-full" />
                  <span className="text-slate-700">{indicator}</span>
                </div>
              ))}
            </div>
          </div>
          
          <div className="grid grid-cols-2 gap-3 text-sm">
            <div>
              <div className="font-medium text-green-700 mb-1">Strengths</div>
              <ul className="space-y-1">
                {spatialAnalysis?.strengths.map((strength, index) => (
                  <li key={index} className="text-slate-600">• {strength}</li>
                ))}
              </ul>
            </div>
            <div>
              <div className="font-medium text-orange-700 mb-1">Limitations</div>
              <ul className="space-y-1">
                {spatialAnalysis?.weaknesses.map((weakness, index) => (
                  <li key={index} className="text-slate-600">• {weakness}</li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Frequency Domain */}
      <motion.div
        initial={{ opacity: 0, x: 20 }}
        animate={{ opacity: 1, x: 0 }}
        className="bg-white rounded-xl p-6 border border-slate-200 shadow-lg"
      >
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-purple-100 rounded-lg">
              <Zap className="w-6 h-6 text-purple-600" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-slate-900">Frequency Domain</h3>
              <p className="text-sm text-slate-600">DCT analysis & compression artifacts</p>
            </div>
          </div>
          <div className="text-right">
            <div className="text-2xl font-bold text-purple-600">
              {frequencyAnalysis ? `${frequencyAnalysis.confidence.toFixed(1)}%` : '--'}
            </div>
            <div className="text-sm text-slate-600">Confidence</div>
          </div>
        </div>

        <div className="relative mb-4">
          <div className="aspect-square bg-slate-50 rounded-lg overflow-hidden border border-slate-200">
            {image && frequencyAnalysis && (
              <FrequencyHeatmap
                image={image}
                frequencyMaps={frequencyAnalysis.frequencyMaps}
                opacity={overlayOpacity}
                onRegionSelect={(region) => handleRegionSelect(region, 'frequency')}
                interactive={true}
              />
            )}
          </div>
          
          {activeRegion && activeRegion.domain === 'frequency' && (
            <div className="absolute top-2 right-2 bg-white/90 backdrop-blur-sm rounded-lg p-2 text-sm">
              <div className="font-medium">Frequency Analysis</div>
              <div>DCT Anomaly: {(activeRegion.anomalyScore * 100).toFixed(1)}%</div>
            </div>
          )}
        </div>

        <div className="space-y-3">
          <div>
            <h4 className="font-medium text-slate-900 mb-2">Primary Indicators</h4>
            <div className="space-y-1">
              {frequencyAnalysis?.primaryIndicators.map((indicator, index) => (
                <div key={index} className="flex items-center space-x-2 text-sm">
                  <div className="w-2 h-2 bg-purple-600 rounded-full" />
                  <span className="text-slate-700">{indicator}</span>
                </div>
              ))}
            </div>
          </div>
          
          <div className="grid grid-cols-2 gap-3 text-sm">
            <div>
              <div className="font-medium text-green-700 mb-1">Strengths</div>
              <ul className="space-y-1">
                {frequencyAnalysis?.strengths.map((strength, index) => (
                  <li key={index} className="text-slate-600">• {strength}</li>
                ))}
              </ul>
            </div>
            <div>
              <div className="font-medium text-orange-700 mb-1">Limitations</div>
              <ul className="space-y-1">
                {frequencyAnalysis?.weaknesses.map((weakness, index) => (
                  <li key={index} className="text-slate-600">• {weakness}</li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  );

  const renderOverlayView = () => (
    <div className="relative">
      <div className="bg-white rounded-xl p-6 border border-slate-200 shadow-lg">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-semibold text-slate-900">Unified Analysis View</h3>
          
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <span className="text-sm text-slate-600">Overlay Opacity:</span>
              <InteractiveSlider
                value={overlayOpacity}
                onChange={setOverlayOpacity}
                min={0}
                max={1}
                step={0.1}
                className="w-24"
              />
            </div>
            
            <button
              onClick={toggleAnimation}
              className={`flex items-center space-x-2 px-3 py-2 rounded-lg transition-colors ${
                isAnimating 
                  ? 'bg-red-100 text-red-600 hover:bg-red-200' 
                  : 'bg-blue-100 text-blue-600 hover:bg-blue-200'
              }`}
            >
              {isAnimating ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
              <span>{isAnimating ? 'Stop' : 'Animate'}</span>
            </button>
          </div>
        </div>

        <div className="relative">
          <div className="aspect-video bg-slate-50 rounded-lg overflow-hidden border border-slate-200">
            {image && (
              <div className="relative w-full h-full">
                <img src={image} alt="Analysis" className="w-full h-full object-contain" />
                
                {/* Spatial overlay */}
                {spatialAnalysis && (
                  <div className="absolute inset-0" style={{ opacity: overlayOpacity }}>
                    <SpatialHeatmap
                      image={image}
                      attentionMaps={spatialAnalysis.attentionMaps}
                      opacity={0.6}
                      colorMap="blues"
                    />
                  </div>
                )}
                
                {/* Frequency overlay */}
                {frequencyAnalysis && (
                  <div className="absolute inset-0" style={{ opacity: overlayOpacity }}>
                    <FrequencyHeatmap
                      image={image}
                      frequencyMaps={frequencyAnalysis.frequencyMaps}
                      opacity={0.4}
                      colorMap="purples"
                      blendMode="multiply"
                    />
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );

  const renderDomainAgreement = () => {
    if (!domainAgreement) return null;

    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-gradient-to-r from-slate-50 to-blue-50 rounded-xl p-6 border border-slate-200"
      >
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-indigo-100 rounded-lg">
              <TrendingUp className="w-6 h-6 text-indigo-600" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-slate-900">Domain Agreement Analysis</h3>
              <p className="text-sm text-slate-600">How well spatial and frequency domains agree</p>
            </div>
          </div>
          
          <div className="text-right">
            <div className="text-2xl font-bold text-indigo-600">
              {(domainAgreement.overallAgreement * 100).toFixed(1)}%
            </div>
            <div className="text-sm text-slate-600">Agreement</div>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <DomainMetrics
            title="Confidence Alignment"
            value={domainAgreement.confidenceAlignment}
            description="How similar the confidence scores are"
            color="blue"
          />
          
          <DomainMetrics
            title="Consensus Alignment"
            value={domainAgreement.consensusAlignment}
            description="Agreement on deepfake classification"
            color="green"
          />
          
          <DomainMetrics
            title="Overall Agreement"
            value={domainAgreement.overallAgreement}
            description="Combined domain agreement score"
            color="indigo"
          />
        </div>

        {keyDifferences.length > 0 && (
          <div className="mt-6 pt-4 border-t border-slate-200">
            <h4 className="font-medium text-slate-900 mb-3">Key Differences</h4>
            <div className="space-y-2">
              {keyDifferences.map((diff, index) => (
                <div key={index} className="flex items-start space-x-3 p-3 bg-white rounded-lg">
                  <div className={`w-3 h-3 rounded-full mt-1 ${
                    diff.severity === 'high' ? 'bg-red-500' : 
                    diff.severity === 'medium' ? 'bg-yellow-500' : 'bg-green-500'
                  }`} />
                  <div className="flex-1">
                    <div className="font-medium text-slate-900">{diff.description}</div>
                    <div className="text-sm text-slate-600">{diff.impact}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </motion.div>
    );
  };

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Controls */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setViewMode('split')}
              className={`flex items-center space-x-2 px-3 py-2 rounded-lg transition-colors ${
                viewMode === 'split' ? 'bg-blue-100 text-blue-600' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
              }`}
            >
              <Split className="w-4 h-4" />
              <span>Split View</span>
            </button>
            
            <button
              onClick={() => setViewMode('overlay')}
              className={`flex items-center space-x-2 px-3 py-2 rounded-lg transition-colors ${
                viewMode === 'overlay' ? 'bg-blue-100 text-blue-600' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
              }`}
            >
              <Layers className="w-4 h-4" />
              <span>Overlay</span>
            </button>
          </div>
        </div>

        <div className="flex items-center space-x-2">
          <button
            onClick={() => setShowInsights(!showInsights)}
            className={`p-2 rounded-lg transition-colors ${
              showInsights ? 'bg-blue-100 text-blue-600' : 'bg-slate-100 text-slate-600'
            }`}
            title="Toggle insights"
          >
            <Info className="w-4 h-4" />
          </button>
          
          <button
            onClick={exportComparison}
            className="p-2 bg-slate-100 text-slate-600 hover:bg-slate-200 rounded-lg transition-colors"
            title="Export comparison"
          >
            <Download className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Main Comparison View */}
      <AnimatePresence mode="wait">
        <motion.div
          key={viewMode}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.3 }}
        >
          {viewMode === 'split' && renderSplitView()}
          {viewMode === 'overlay' && renderOverlayView()}
        </motion.div>
      </AnimatePresence>

      {/* Domain Agreement */}
      {showInsights && renderDomainAgreement()}

      {/* Hidden canvas for export */}
      <canvas ref={canvasRef} className="hidden" />
    </div>
  );
};

export default SpatialFrequencyComparison;