/**
 * AnalysisWorkspace - Main component for multi-model deepfake analysis
 * Orchestrates the entire analysis workflow with sophisticated visualizations
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Upload, 
  Brain, 
  Zap, 
  BarChart3, 
  Eye,
  Settings,
  Download,
  Share2,
  Play,
  Pause,
  RotateCcw
} from 'lucide-react';

import ImageUploader from './ImageUploader';
import ModelSelectionPanel from './ModelSelectionPanel';
import AnalysisProgress from './AnalysisProgress';
import ResultsVisualization from './visualization/ResultsVisualization';
import ModelComparison from './comparison/ModelComparison';
import FrequencyAnalysis from './frequency/FrequencyAnalysis';
import SpatialAnalysis from './spatial/SpatialAnalysis';
import EnsembleInsights from './ensemble/EnsembleInsights';
import AdvancedControls from './controls/AdvancedControls';
import ExportPanel from './export/ExportPanel';

import { useAnalysis } from '../../hooks/useAnalysis';
import { useVisualization } from '../../hooks/useVisualization';
import { useRealTimeUpdates } from '../../hooks/useRealTimeUpdates';

const AnalysisWorkspace = ({ userType = 'general' }) => {
  // State management
  const [activeTab, setActiveTab] = useState('upload');
  const [selectedImage, setSelectedImage] = useState(null);
  const [analysisConfig, setAnalysisConfig] = useState({
    models: ['xception', 'efficientnet', 'f3net'],
    ensemble: true,
    visualizations: true,
    exportResults: false
  });
  const [layoutMode, setLayoutMode] = useState('standard'); // standard, compact, expert
  
  // Custom hooks
  const {
    analysisState,
    startAnalysis,
    pauseAnalysis,
    resetAnalysis,
    analysisResults,
    isAnalyzing,
    progress,
    error
  } = useAnalysis();
  
  const {
    visualizationMode,
    setVisualizationMode,
    overlaySettings,
    heatmapSettings,
    updateSettings
  } = useVisualization();
  
  const { 
    realTimeUpdates,
    subscribeToUpdates,
    unsubscribeFromUpdates
  } = useRealTimeUpdates();

  // Refs for advanced interactions
  const workspaceRef = useRef(null);
  const visualizationRef = useRef(null);

  // User type configurations
  const userConfigs = {
    general: {
      showAdvancedControls: false,
      defaultModels: ['xception', 'efficientnet'],
      simplifiedUI: true
    },
    researcher: {
      showAdvancedControls: true,
      defaultModels: ['xception', 'efficientnet', 'f3net'],
      simplifiedUI: false,
      enableExperimental: true
    },
    professional: {
      showAdvancedControls: true,
      defaultModels: ['xception', 'efficientnet', 'f3net'],
      simplifiedUI: false,
      enableBatchProcessing: true
    }
  };

  const currentConfig = userConfigs[userType] || userConfigs.general;

  // Main tabs configuration
  const mainTabs = [
    {
      id: 'upload',
      label: 'Upload & Configure',
      icon: Upload,
      component: (
        <div className="space-y-6">
          <ImageUploader
            onImageSelect={setSelectedImage}
            selectedImage={selectedImage}
            acceptedFormats={['image/jpeg', 'image/png', 'image/webp']}
            maxFileSize={10 * 1024 * 1024} // 10MB
          />
          <ModelSelectionPanel
            selectedModels={analysisConfig.models}
            onModelsChange={(models) => setAnalysisConfig({...analysisConfig, models})}
            userType={userType}
            modelInfo={{
              xception: { accuracy: '96.6%', speed: 'Fast', specialty: 'Facial artifacts' },
              efficientnet: { accuracy: '89.35%', speed: 'Very Fast', specialty: 'Mobile optimized' },
              f3net: { accuracy: '94.5%', speed: 'Medium', specialty: 'Frequency analysis' }
            }}
          />
          {currentConfig.showAdvancedControls && (
            <AdvancedControls
              config={analysisConfig}
              onConfigChange={setAnalysisConfig}
            />
          )}
        </div>
      )
    },
    {
      id: 'analysis',
      label: 'Real-time Analysis',
      icon: Brain,
      component: (
        <div className="space-y-6">
          <AnalysisProgress
            isAnalyzing={isAnalyzing}
            progress={progress}
            selectedModels={analysisConfig.models}
            realTimeUpdates={realTimeUpdates}
            onPause={pauseAnalysis}
            onResume={startAnalysis}
            onReset={resetAnalysis}
          />
          {analysisResults && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
            >
              <ResultsVisualization
                ref={visualizationRef}
                results={analysisResults}
                image={selectedImage}
                mode={visualizationMode}
                settings={{ overlay: overlaySettings, heatmap: heatmapSettings }}
                userType={userType}
              />
            </motion.div>
          )}
        </div>
      )
    },
    {
      id: 'comparison',
      label: 'Model Comparison',
      icon: BarChart3,
      component: (
        <ModelComparison
          results={analysisResults}
          selectedModels={analysisConfig.models}
          image={selectedImage}
          showFrequencyAnalysis={analysisConfig.models.includes('f3net')}
        />
      )
    },
    {
      id: 'insights',
      label: 'Ensemble Insights',
      icon: Eye,
      component: (
        <EnsembleInsights
          results={analysisResults}
          spatialModels={['xception', 'efficientnet']}
          frequencyModels={['f3net']}
          showAdvanced={currentConfig.showAdvancedControls}
        />
      )
    }
  ];

  // Add advanced tabs for expert users
  if (userType === 'researcher' || userType === 'professional') {
    mainTabs.push(
      {
        id: 'frequency',
        label: 'Frequency Domain',
        icon: Zap,
        component: (
          <FrequencyAnalysis
            results={analysisResults}
            image={selectedImage}
            enableInteraction={true}
            showDCTVisualization={true}
          />
        )
      },
      {
        id: 'spatial',
        label: 'Spatial Analysis',
        icon: Settings,
        component: (
          <SpatialAnalysis
            results={analysisResults}
            image={selectedImage}
            models={['xception', 'efficientnet']}
            enableGradCAM={true}
          />
        )
      }
    );
  }

  // Event handlers
  const handleStartAnalysis = useCallback(async () => {
    if (!selectedImage) {
      alert('Please select an image first');
      return;
    }

    try {
      await startAnalysis({
        image: selectedImage,
        config: analysisConfig
      });
      setActiveTab('analysis');
    } catch (error) {
      console.error('Analysis failed:', error);
    }
  }, [selectedImage, analysisConfig, startAnalysis]);

  const handleExportResults = useCallback(() => {
    if (analysisResults) {
      // Trigger export modal or direct download
      setActiveTab('export');
    }
  }, [analysisResults]);

  // Effect for real-time updates
  useEffect(() => {
    if (isAnalyzing) {
      subscribeToUpdates();
    } else {
      unsubscribeFromUpdates();
    }

    return () => unsubscribeFromUpdates();
  }, [isAnalyzing, subscribeToUpdates, unsubscribeFromUpdates]);

  // Adaptive layout based on screen size and user type
  const getLayoutClass = () => {
    const base = 'min-h-screen bg-gradient-to-br from-slate-50 to-blue-50';
    
    switch (layoutMode) {
      case 'compact':
        return `${base} p-2`;
      case 'expert':
        return `${base} p-1`;
      default:
        return `${base} p-4 lg:p-6`;
    }
  };

  return (
    <div className={getLayoutClass()} ref={workspaceRef}>
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white rounded-xl shadow-lg border border-slate-200 mb-6"
      >
        <div className="px-6 py-4 border-b border-slate-200">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-slate-900">
                Multi-Model Deepfake Analysis
              </h1>
              <p className="text-slate-600 mt-1">
                {userType === 'researcher' ? 'Advanced Research Interface' :
                 userType === 'professional' ? 'Professional Analysis Suite' :
                 'Intelligent Detection Platform'}
              </p>
            </div>
            
            <div className="flex items-center space-x-3">
              {/* Quick Actions */}
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={handleStartAnalysis}
                disabled={!selectedImage || isAnalyzing}
                className="bg-blue-600 text-white px-4 py-2 rounded-lg font-medium 
                         disabled:opacity-50 disabled:cursor-not-allowed
                         hover:bg-blue-700 transition-colors duration-200
                         flex items-center space-x-2"
              >
                <Play className="w-4 h-4" />
                <span>Analyze</span>
              </motion.button>
              
              {analysisResults && (
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={handleExportResults}
                  className="bg-green-600 text-white px-4 py-2 rounded-lg font-medium
                           hover:bg-green-700 transition-colors duration-200
                           flex items-center space-x-2"
                >
                  <Download className="w-4 h-4" />
                  <span>Export</span>
                </motion.button>
              )}
              
              <button
                onClick={() => setLayoutMode(layoutMode === 'standard' ? 'compact' : 'standard')}
                className="p-2 text-slate-600 hover:text-slate-900 hover:bg-slate-100 
                         rounded-lg transition-colors duration-200"
                title="Toggle layout"
              >
                <Settings className="w-5 h-5" />
              </button>
            </div>
          </div>
          
          {/* Tab Navigation */}
          <div className="flex space-x-1 mt-4 bg-slate-100 rounded-lg p-1">
            {mainTabs.map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex-1 flex items-center justify-center space-x-2 py-2 px-3 
                            rounded-md font-medium transition-all duration-200 ${
                    activeTab === tab.id
                      ? 'bg-white text-blue-600 shadow-sm'
                      : 'text-slate-600 hover:text-slate-900 hover:bg-slate-200'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span className="hidden sm:inline">{tab.label}</span>
                </button>
              );
            })}
          </div>
        </div>
      </motion.div>

      {/* Main Content Area */}
      <div className="space-y-6">
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            transition={{ duration: 0.3 }}
            className="bg-white rounded-xl shadow-lg border border-slate-200 p-6"
          >
            {mainTabs.find(tab => tab.id === activeTab)?.component}
          </motion.div>
        </AnimatePresence>

        {/* Side Panel for Expert Users */}
        {(userType === 'researcher' || userType === 'professional') && analysisResults && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="fixed right-4 top-1/2 transform -translate-y-1/2 w-80 z-50"
          >
            <ExpertSidePanel
              results={analysisResults}
              onVisualizationChange={setVisualizationMode}
              onSettingsChange={updateSettings}
            />
          </motion.div>
        )}
      </div>

      {/* Error Display */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 50 }}
            className="fixed bottom-4 right-4 bg-red-600 text-white p-4 rounded-lg shadow-lg z-50"
          >
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-white rounded-full animate-pulse" />
              <span>{error}</span>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Export Modal */}
      <AnimatePresence>
        {activeTab === 'export' && (
          <ExportPanel
            results={analysisResults}
            image={selectedImage}
            onClose={() => setActiveTab('insights')}
          />
        )}
      </AnimatePresence>
    </div>
  );
};

// Expert Side Panel Component
const ExpertSidePanel = ({ results, onVisualizationChange, onSettingsChange }) => {
  const [isPinned, setIsPinned] = useState(false);
  
  return (
    <motion.div
      className={`bg-white rounded-lg shadow-xl border border-slate-200 p-4 ${
        isPinned ? 'opacity-100' : 'opacity-90 hover:opacity-100'
      } transition-opacity duration-200`}
    >
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-semibold text-slate-900">Quick Controls</h3>
        <button
          onClick={() => setIsPinned(!isPinned)}
          className="p-1 text-slate-500 hover:text-slate-700"
        >
          📌
        </button>
      </div>
      
      <div className="space-y-3">
        <div>
          <label className="block text-sm font-medium text-slate-700 mb-1">
            Visualization Mode
          </label>
          <select
            onChange={(e) => onVisualizationChange(e.target.value)}
            className="w-full p-2 border border-slate-300 rounded text-sm"
          >
            <option value="overview">Overview</option>
            <option value="detailed">Detailed Analysis</option>
            <option value="comparison">Side-by-side</option>
            <option value="frequency">Frequency Focus</option>
          </select>
        </div>
        
        <div>
          <label className="block text-sm font-medium text-slate-700 mb-1">
            Heatmap Opacity
          </label>
          <input
            type="range"
            min="0"
            max="1"
            step="0.1"
            defaultValue="0.7"
            onChange={(e) => onSettingsChange('heatmapOpacity', e.target.value)}
            className="w-full"
          />
        </div>
        
        <div className="pt-2 border-t border-slate-200">
          <div className="text-xs text-slate-500 space-y-1">
            <div>Ensemble Score: {results?.ensemble?.confidence?.toFixed(1)}% ({results?.ensemble?.isDeepfake ? 'FAKE' : 'REAL'})</div>
            <div>Domain Agreement: {results?.ensemble?.agreement?.toFixed(2)}</div>
            <div>Processing Time: {results?.metadata?.processingTime}ms</div>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default AnalysisWorkspace;