/**
 * AnalysisProgress - Real-time progress tracking for multi-model analysis
 * Features live updates, model-specific progress, and interactive controls
 */

import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Play, 
  Pause, 
  Square, 
  RotateCcw,
  Activity,
  Clock,
  Cpu,
  Zap,
  Brain,
  CheckCircle,
  AlertCircle,
  Loader,
  TrendingUp
} from 'lucide-react';

const AnalysisProgress = ({
  isAnalyzing,
  progress = {},
  selectedModels = [],
  realTimeUpdates = {},
  onPause,
  onResume,
  onReset,
  estimatedTime,
  showAdvanced = false
}) => {
  const [elapsedTime, setElapsedTime] = useState(0);
  const [currentStage, setCurrentStage] = useState('initializing');
  const [modelProgress, setModelProgress] = useState({});
  const [systemMetrics, setSystemMetrics] = useState({});
  const [throughputData, setThroughputData] = useState([]);
  
  const startTimeRef = useRef(null);
  const intervalRef = useRef(null);

  // Model configurations with expected processing times
  const modelConfigs = {
    xception: {
      name: 'Xception',
      icon: Brain,
      color: 'blue',
      estimatedTime: 2000, // ms
      stages: ['preprocessing', 'feature_extraction', 'classification', 'gradcam']
    },
    efficientnet: {
      name: 'EfficientNet-B4',
      icon: Zap,
      color: 'green',
      estimatedTime: 1500,
      stages: ['preprocessing', 'mobile_inference', 'classification', 'attention']
    },
    f3net: {
      name: 'F3Net',
      icon: Activity,
      color: 'purple',
      estimatedTime: 3000,
      stages: ['preprocessing', 'dct_transform', 'frequency_analysis', 'spatial_fusion']
    }
  };

  // Analysis stages with descriptions
  const analysisStages = {
    initializing: { label: 'Initializing Models', description: 'Loading and preparing models' },
    preprocessing: { label: 'Preprocessing Image', description: 'Preparing image for analysis' },
    individual_analysis: { label: 'Individual Model Analysis', description: 'Running each model' },
    ensemble_fusion: { label: 'Ensemble Fusion', description: 'Combining model predictions' },
    postprocessing: { label: 'Generating Insights', description: 'Creating visualizations and reports' },
    completed: { label: 'Analysis Complete', description: 'Ready to view results' }
  };

  // Start/stop timer
  useEffect(() => {
    if (isAnalyzing) {
      startTimeRef.current = Date.now();
      intervalRef.current = setInterval(() => {
        setElapsedTime(Date.now() - startTimeRef.current);
      }, 100);
    } else {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [isAnalyzing]);

  // Update progress based on real-time updates
  useEffect(() => {
    if (realTimeUpdates.stage) {
      setCurrentStage(realTimeUpdates.stage);
    }
    if (realTimeUpdates.modelProgress) {
      setModelProgress(realTimeUpdates.modelProgress);
    }
    if (realTimeUpdates.systemMetrics) {
      setSystemMetrics(realTimeUpdates.systemMetrics);
    }
    if (realTimeUpdates.throughput) {
      setThroughputData(prev => [
        ...prev.slice(-20), // Keep last 20 data points
        { time: Date.now(), value: realTimeUpdates.throughput }
      ]);
    }
  }, [realTimeUpdates]);

  // Calculate overall progress
  const calculateOverallProgress = () => {
    if (!isAnalyzing) return progress.overall || 0;
    
    const stageProgress = {
      initializing: 10,
      preprocessing: 20,
      individual_analysis: 70,
      ensemble_fusion: 90,
      postprocessing: 95,
      completed: 100
    };
    
    let baseProgress = stageProgress[currentStage] || 0;
    
    // Add model-specific progress
    if (currentStage === 'individual_analysis') {
      const modelProgressValues = Object.values(modelProgress);
      const avgModelProgress = modelProgressValues.length > 0 
        ? modelProgressValues.reduce((sum, p) => sum + p, 0) / modelProgressValues.length 
        : 0;
      baseProgress = 20 + (avgModelProgress * 0.5); // 20% to 70%
    }
    
    return Math.min(baseProgress, 100);
  };

  const formatTime = (ms) => {
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    
    if (minutes > 0) {
      return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
    }
    return `${remainingSeconds}s`;
  };

  const getEstimatedRemaining = () => {
    const overall = calculateOverallProgress();
    if (overall === 0) return estimatedTime || 5000;
    
    const estimated = (elapsedTime / overall) * (100 - overall);
    return Math.max(0, estimated);
  };

  const getStageIcon = (stage) => {
    switch (stage) {
      case 'initializing': return Loader;
      case 'preprocessing': return Activity;
      case 'individual_analysis': return Brain;
      case 'ensemble_fusion': return TrendingUp;
      case 'postprocessing': return CheckCircle;
      default: return Activity;
    }
  };

  const renderModelProgress = (modelId) => {
    const config = modelConfigs[modelId];
    if (!config) return null;

    const progress = modelProgress[modelId] || 0;
    const Icon = config.icon;
    const isCompleted = progress >= 100;
    const isActive = isAnalyzing && progress > 0 && progress < 100;

    return (
      <motion.div
        key={modelId}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white rounded-lg p-4 border border-slate-200"
      >
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center space-x-3">
            <div className={`p-2 rounded-lg bg-${config.color}-100`}>
              <Icon className={`w-5 h-5 text-${config.color}-600`} />
            </div>
            <div>
              <h4 className="font-medium text-slate-900">{config.name}</h4>
              <p className="text-sm text-slate-600">
                Est. {config.estimatedTime}ms
              </p>
            </div>
          </div>
          
          <div className="flex items-center space-x-2">
            {isCompleted && (
              <CheckCircle className="w-5 h-5 text-green-600" />
            )}
            {isActive && (
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
              >
                <Loader className="w-5 h-5 text-blue-600" />
              </motion.div>
            )}
            <span className="text-sm font-medium text-slate-900">
              {progress.toFixed(0)}%
            </span>
          </div>
        </div>

        {/* Progress Bar */}
        <div className="relative">
          <div className="h-2 bg-slate-200 rounded-full overflow-hidden">
            <motion.div
              className={`h-full bg-${config.color}-600 rounded-full`}
              initial={{ width: 0 }}
              animate={{ width: `${progress}%` }}
              transition={{ duration: 0.3 }}
            />
          </div>
          
          {/* Stage Indicators */}
          <div className="flex justify-between mt-2">
            {config.stages.map((stage, index) => {
              const stageProgress = Math.max(0, Math.min(100, (progress - (index * 25))));
              const isStageActive = stageProgress > 0 && stageProgress < 100;
              const isStageCompleted = stageProgress >= 100;
              
              return (
                <div
                  key={stage}
                  className={`text-xs px-2 py-1 rounded ${
                    isStageCompleted 
                      ? `bg-${config.color}-100 text-${config.color}-700`
                      : isStageActive
                      ? `bg-${config.color}-50 text-${config.color}-600 animate-pulse`
                      : 'bg-slate-100 text-slate-500'
                  }`}
                >
                  {stage.replace('_', ' ')}
                </div>
              );
            })}
          </div>
        </div>

        {/* Model-specific metrics */}
        {showAdvanced && realTimeUpdates.modelMetrics?.[modelId] && (
          <div className="mt-3 pt-3 border-t border-slate-200">
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div>
                <span className="text-slate-600">Inference:</span>
                <span className="ml-1 font-medium">
                  {realTimeUpdates.modelMetrics[modelId].inferenceTime}ms
                </span>
              </div>
              <div>
                <span className="text-slate-600">Memory:</span>
                <span className="ml-1 font-medium">
                  {realTimeUpdates.modelMetrics[modelId].memoryUsage}MB
                </span>
              </div>
            </div>
          </div>
        )}
      </motion.div>
    );
  };

  const renderSystemMetrics = () => {
    if (!showAdvanced || Object.keys(systemMetrics).length === 0) return null;

    return (
      <div className="bg-white rounded-lg p-4 border border-slate-200">
        <h4 className="font-medium text-slate-900 mb-3 flex items-center">
          <Cpu className="w-4 h-4 mr-2" />
          System Metrics
        </h4>
        
        <div className="grid grid-cols-2 gap-4">
          <div>
            <div className="flex justify-between items-center mb-1">
              <span className="text-sm text-slate-600">CPU Usage</span>
              <span className="text-sm font-medium">{systemMetrics.cpu}%</span>
            </div>
            <div className="h-2 bg-slate-200 rounded-full">
              <div 
                className="h-full bg-blue-600 rounded-full transition-all duration-300"
                style={{ width: `${systemMetrics.cpu}%` }}
              />
            </div>
          </div>
          
          <div>
            <div className="flex justify-between items-center mb-1">
              <span className="text-sm text-slate-600">GPU Memory</span>
              <span className="text-sm font-medium">{systemMetrics.gpuMemory}%</span>
            </div>
            <div className="h-2 bg-slate-200 rounded-full">
              <div 
                className="h-full bg-green-600 rounded-full transition-all duration-300"
                style={{ width: `${systemMetrics.gpuMemory}%` }}
              />
            </div>
          </div>
        </div>
        
        {throughputData.length > 0 && (
          <div className="mt-4">
            <div className="text-sm text-slate-600 mb-2">Processing Throughput</div>
            <div className="h-16 bg-slate-50 rounded border relative overflow-hidden">
              <svg className="w-full h-full">
                <polyline
                  fill="none"
                  stroke="#3b82f6"
                  strokeWidth="2"
                  points={throughputData.map((point, index) => 
                    `${(index / (throughputData.length - 1)) * 100},${100 - (point.value * 100)}`
                  ).join(' ')}
                />
              </svg>
            </div>
          </div>
        )}
      </div>
    );
  };

  const overallProgress = calculateOverallProgress();
  const StageIcon = getStageIcon(currentStage);
  const currentStageInfo = analysisStages[currentStage] || analysisStages.initializing;

  return (
    <div className="space-y-6">
      {/* Main Progress Header */}
      <div className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-xl p-6 text-white">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-3">
            <motion.div
              animate={isAnalyzing ? { rotate: 360 } : { rotate: 0 }}
              transition={{ duration: 2, repeat: isAnalyzing ? Infinity : 0, ease: "linear" }}
            >
              <StageIcon className="w-8 h-8" />
            </motion.div>
            <div>
              <h3 className="text-xl font-semibold">{currentStageInfo.label}</h3>
              <p className="text-blue-100">{currentStageInfo.description}</p>
            </div>
          </div>
          
          <div className="flex items-center space-x-4">
            <div className="text-right">
              <div className="text-2xl font-bold">{overallProgress.toFixed(0)}%</div>
              <div className="text-sm text-blue-100">Complete</div>
            </div>
            
            <div className="flex space-x-2">
              {isAnalyzing ? (
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={onPause}
                  className="bg-white/20 hover:bg-white/30 p-3 rounded-lg transition-colors"
                  title="Pause analysis"
                >
                  <Pause className="w-5 h-5" />
                </motion.button>
              ) : (
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={onResume}
                  className="bg-white/20 hover:bg-white/30 p-3 rounded-lg transition-colors"
                  title="Resume analysis"
                >
                  <Play className="w-5 h-5" />
                </motion.button>
              )}
              
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={onReset}
                className="bg-white/20 hover:bg-white/30 p-3 rounded-lg transition-colors"
                title="Reset analysis"
              >
                <RotateCcw className="w-5 h-5" />
              </motion.button>
            </div>
          </div>
        </div>
        
        {/* Overall Progress Bar */}
        <div className="relative">
          <div className="h-3 bg-white/20 rounded-full overflow-hidden">
            <motion.div
              className="h-full bg-white rounded-full"
              initial={{ width: 0 }}
              animate={{ width: `${overallProgress}%` }}
              transition={{ duration: 0.5 }}
            />
          </div>
          
          {/* Time Information */}
          <div className="flex justify-between items-center mt-3 text-sm text-blue-100">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-1">
                <Clock className="w-4 h-4" />
                <span>Elapsed: {formatTime(elapsedTime)}</span>
              </div>
              {isAnalyzing && (
                <div>
                  Remaining: ~{formatTime(getEstimatedRemaining())}
                </div>
              )}
            </div>
            
            <div className="text-right">
              Models: {selectedModels.length} active
            </div>
          </div>
        </div>
      </div>

      {/* Individual Model Progress */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {selectedModels.map(modelId => renderModelProgress(modelId))}
      </div>

      {/* System Metrics */}
      {renderSystemMetrics()}

      {/* Real-time Updates Log */}
      {showAdvanced && realTimeUpdates.log && (
        <div className="bg-black rounded-lg p-4 font-mono text-sm">
          <h4 className="text-green-400 mb-2">Analysis Log</h4>
          <div className="text-green-300 space-y-1 max-h-32 overflow-y-auto">
            {realTimeUpdates.log.slice(-10).map((entry, index) => (
              <div key={index} className="flex items-center space-x-2">
                <span className="text-gray-500">[{entry.timestamp}]</span>
                <span>{entry.message}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default AnalysisProgress;