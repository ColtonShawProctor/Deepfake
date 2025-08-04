/**
 * UserTypeAdapter - Adaptive user experience based on user expertise level
 * Customizes interface complexity, terminology, and available features
 */

import React, { useState, useEffect, useContext, createContext } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  User, 
  GraduationCap, 
  Briefcase,
  Settings,
  HelpCircle,
  ChevronRight,
  Star,
  BookOpen,
  Zap,
  Shield
} from 'lucide-react';

// User Type Context
const UserTypeContext = createContext();

export const useUserType = () => {
  const context = useContext(UserTypeContext);
  if (!context) {
    throw new Error('useUserType must be used within a UserTypeProvider');
  }
  return context;
};

// User type configurations
const userTypeConfigs = {
  general: {
    label: 'General User',
    description: 'Simple, intuitive interface for everyday deepfake detection',
    icon: User,
    color: 'blue',
    complexity: 'simple',
    features: {
      basicAnalysis: true,
      advancedMetrics: false,
      modelComparison: false,
      frequencyAnalysis: false,
      expertControls: false,
      batchProcessing: false,
      apiAccess: false,
      customModels: false
    },
    terminology: {
      confidence: 'Certainty',
      ensemble: 'Combined Analysis',
      heatmap: 'Focus Areas',
      frequency: 'Technical Analysis',
      spatial: 'Visual Analysis'
    },
    defaultSettings: {
      showTooltips: true,
      autoAnalysis: true,
      simplifiedResults: true,
      guidedWorkflow: true,
      showExplanations: true
    },
    onboarding: {
      required: true,
      steps: 5,
      interactive: true
    }
  },
  
  researcher: {
    label: 'Researcher',
    description: 'Comprehensive tools for academic research and analysis',
    icon: GraduationCap,
    color: 'purple',
    complexity: 'advanced',
    features: {
      basicAnalysis: true,
      advancedMetrics: true,
      modelComparison: true,
      frequencyAnalysis: true,
      expertControls: true,
      batchProcessing: true,
      apiAccess: true,
      customModels: true,
      experimentalFeatures: true,
      dataExport: true,
      statisticalAnalysis: true
    },
    terminology: {
      confidence: 'Confidence Score',
      ensemble: 'Ensemble Prediction',
      heatmap: 'Attention Heatmap',
      frequency: 'Frequency Domain Analysis',
      spatial: 'Spatial Domain Analysis'
    },
    defaultSettings: {
      showTooltips: false,
      autoAnalysis: false,
      simplifiedResults: false,
      guidedWorkflow: false,
      showExplanations: true,
      enableLogging: true,
      showPerformanceMetrics: true
    },
    onboarding: {
      required: false,
      steps: 3,
      interactive: false
    }
  },
  
  professional: {
    label: 'Professional',
    description: 'Enterprise-grade tools for business and security applications',
    icon: Briefcase,
    color: 'green',
    complexity: 'professional',
    features: {
      basicAnalysis: true,
      advancedMetrics: true,
      modelComparison: true,
      frequencyAnalysis: true,
      expertControls: true,
      batchProcessing: true,
      apiAccess: true,
      customModels: false,
      auditTrail: true,
      reportGeneration: true,
      workflowAutomation: true,
      teamCollaboration: true
    },
    terminology: {
      confidence: 'Detection Confidence',
      ensemble: 'Multi-Model Analysis',
      heatmap: 'Detection Heatmap',
      frequency: 'Frequency Analysis',
      spatial: 'Spatial Analysis'
    },
    defaultSettings: {
      showTooltips: false,
      autoAnalysis: false,
      simplifiedResults: false,
      guidedWorkflow: false,
      showExplanations: true,
      enableAuditLog: true,
      requireConfirmation: true
    },
    onboarding: {
      required: true,
      steps: 4,
      interactive: true
    }
  }
};

// User Type Provider
export const UserTypeProvider = ({ children, initialUserType = 'general' }) => {
  const [userType, setUserType] = useState(initialUserType);
  const [userPreferences, setUserPreferences] = useState({});
  const [onboardingComplete, setOnboardingComplete] = useState(false);
  const [adaptiveSettings, setAdaptiveSettings] = useState({});

  const currentConfig = userTypeConfigs[userType];

  // Load user preferences from localStorage
  useEffect(() => {
    const savedPreferences = localStorage.getItem(`deepfake-preferences-${userType}`);
    if (savedPreferences) {
      setUserPreferences(JSON.parse(savedPreferences));
    } else {
      setUserPreferences(currentConfig.defaultSettings);
    }

    const onboardingStatus = localStorage.getItem(`deepfake-onboarding-${userType}`);
    setOnboardingComplete(onboardingStatus === 'complete');
  }, [userType, currentConfig.defaultSettings]);

  // Save preferences to localStorage
  const updatePreferences = (newPreferences) => {
    const updatedPreferences = { ...userPreferences, ...newPreferences };
    setUserPreferences(updatedPreferences);
    localStorage.setItem(`deepfake-preferences-${userType}`, JSON.stringify(updatedPreferences));
  };

  // Switch user type
  const switchUserType = (newUserType) => {
    setUserType(newUserType);
    setOnboardingComplete(false);
  };

  // Complete onboarding
  const completeOnboarding = () => {
    setOnboardingComplete(true);
    localStorage.setItem(`deepfake-onboarding-${userType}`, 'complete');
  };

  // Adaptive feature availability
  const hasFeature = (featureName) => {
    return currentConfig.features[featureName] === true;
  };

  // Get appropriate terminology
  const getTerm = (key, fallback = key) => {
    return currentConfig.terminology[key] || fallback;
  };

  // Context value
  const value = {
    userType,
    config: currentConfig,
    preferences: userPreferences,
    onboardingComplete,
    hasFeature,
    getTerm,
    updatePreferences,
    switchUserType,
    completeOnboarding,
    adaptiveSettings,
    setAdaptiveSettings
  };

  return (
    <UserTypeContext.Provider value={value}>
      {children}
    </UserTypeContext.Provider>
  );
};

// User Type Selector Component
export const UserTypeSelector = ({ onSelect, className = "" }) => {
  const [selectedType, setSelectedType] = useState(null);
  const [showDetails, setShowDetails] = useState(false);

  const handleSelect = (userType) => {
    setSelectedType(userType);
    if (onSelect) {
      onSelect(userType);
    }
  };

  return (
    <div className={`max-w-4xl mx-auto ${className}`}>
      <div className="text-center mb-8">
        <h2 className="text-3xl font-bold text-slate-900 mb-4">
          Choose Your Experience
        </h2>
        <p className="text-lg text-slate-600">
          We'll customize the interface based on your expertise level
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {Object.entries(userTypeConfigs).map(([key, config]) => {
          const Icon = config.icon;
          const isSelected = selectedType === key;
          
          return (
            <motion.div
              key={key}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => handleSelect(key)}
              className={`relative bg-white rounded-xl p-6 border-2 cursor-pointer transition-all duration-200 ${
                isSelected 
                  ? `border-${config.color}-500 shadow-lg` 
                  : 'border-slate-200 hover:border-slate-300 shadow-sm hover:shadow-md'
              }`}
            >
              {/* Selection indicator */}
              {isSelected && (
                <motion.div
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  className={`absolute -top-2 -right-2 w-6 h-6 bg-${config.color}-500 rounded-full flex items-center justify-center`}
                >
                  <Star className="w-4 h-4 text-white" />
                </motion.div>
              )}

              {/* Icon */}
              <div className={`inline-flex p-3 rounded-lg bg-${config.color}-100 mb-4`}>
                <Icon className={`w-8 h-8 text-${config.color}-600`} />
              </div>

              {/* Content */}
              <h3 className="text-xl font-semibold text-slate-900 mb-2">
                {config.label}
              </h3>
              
              <p className="text-slate-600 mb-4">
                {config.description}
              </p>

              {/* Features preview */}
              <div className="space-y-2 mb-4">
                <div className="text-sm font-medium text-slate-700">
                  Key Features:
                </div>
                <div className="space-y-1">
                  {Object.entries(config.features)
                    .filter(([_, enabled]) => enabled)
                    .slice(0, 3)
                    .map(([feature, _]) => (
                      <div key={feature} className="flex items-center text-sm text-slate-600">
                        <div className={`w-2 h-2 bg-${config.color}-500 rounded-full mr-2`} />
                        {feature.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}
                      </div>
                    ))}
                </div>
              </div>

              {/* Complexity indicator */}
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-1">
                  <span className="text-sm text-slate-600">Complexity:</span>
                  {[...Array(3)].map((_, i) => (
                    <div
                      key={i}
                      className={`w-2 h-2 rounded-full ${
                        i < (config.complexity === 'simple' ? 1 : config.complexity === 'advanced' ? 3 : 2)
                          ? `bg-${config.color}-500`
                          : 'bg-slate-200'
                      }`}
                    />
                  ))}
                </div>
                
                <ChevronRight className={`w-4 h-4 text-${config.color}-500`} />
              </div>
            </motion.div>
          );
        })}
      </div>

      {/* Details panel */}
      <AnimatePresence>
        {selectedType && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="mt-8 bg-white rounded-xl p-6 border border-slate-200 shadow-sm"
          >
            <UserTypeDetails userType={selectedType} />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

// User Type Details Component
const UserTypeDetails = ({ userType }) => {
  const config = userTypeConfigs[userType];
  if (!config) return null;

  const Icon = config.icon;

  return (
    <div>
      <div className="flex items-center space-x-3 mb-6">
        <div className={`p-2 bg-${config.color}-100 rounded-lg`}>
          <Icon className={`w-6 h-6 text-${config.color}-600`} />
        </div>
        <div>
          <h3 className="text-lg font-semibold text-slate-900">
            {config.label} Experience
          </h3>
          <p className="text-slate-600">{config.description}</p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Available Features */}
        <div>
          <h4 className="font-medium text-slate-900 mb-3 flex items-center">
            <Zap className="w-4 h-4 mr-2" />
            Available Features
          </h4>
          <div className="space-y-2">
            {Object.entries(config.features)
              .filter(([_, enabled]) => enabled)
              .map(([feature, _]) => (
                <div key={feature} className="flex items-center text-sm">
                  <div className={`w-2 h-2 bg-${config.color}-500 rounded-full mr-3`} />
                  <span className="text-slate-700">
                    {feature.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}
                  </span>
                </div>
              ))}
          </div>
        </div>

        {/* Interface Customizations */}
        <div>
          <h4 className="font-medium text-slate-900 mb-3 flex items-center">
            <Settings className="w-4 h-4 mr-2" />
            Interface Customizations
          </h4>
          <div className="space-y-2">
            <div className="flex items-center text-sm">
              <div className={`w-2 h-2 bg-${config.color}-500 rounded-full mr-3`} />
              <span className="text-slate-700">
                {config.defaultSettings.showTooltips ? 'Helpful tooltips' : 'Minimal tooltips'}
              </span>
            </div>
            <div className="flex items-center text-sm">
              <div className={`w-2 h-2 bg-${config.color}-500 rounded-full mr-3`} />
              <span className="text-slate-700">
                {config.defaultSettings.simplifiedResults ? 'Simplified results' : 'Detailed results'}
              </span>
            </div>
            <div className="flex items-center text-sm">
              <div className={`w-2 h-2 bg-${config.color}-500 rounded-full mr-3`} />
              <span className="text-slate-700">
                {config.defaultSettings.guidedWorkflow ? 'Guided workflow' : 'Expert workflow'}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Onboarding info */}
      {config.onboarding.required && (
        <div className="mt-6 p-4 bg-slate-50 rounded-lg">
          <div className="flex items-center space-x-2 mb-2">
            <BookOpen className="w-4 h-4 text-slate-600" />
            <span className="font-medium text-slate-900">Getting Started</span>
          </div>
          <p className="text-sm text-slate-600">
            We'll guide you through a {config.onboarding.steps}-step setup process to customize your experience.
          </p>
        </div>
      )}
    </div>
  );
};

// Adaptive Component Wrapper
export const AdaptiveComponent = ({ 
  component: Component, 
  requiredFeature,
  fallback = null,
  children,
  ...props 
}) => {
  const { hasFeature, config } = useUserType();

  if (requiredFeature && !hasFeature(requiredFeature)) {
    return fallback;
  }

  if (Component) {
    return <Component {...props} userType={config} />;
  }

  return children;
};

// User Type Badge Component
export const UserTypeBadge = ({ className = "" }) => {
  const { config } = useUserType();
  const Icon = config.icon;

  return (
    <div className={`inline-flex items-center space-x-2 px-3 py-1 bg-${config.color}-100 text-${config.color}-700 rounded-full text-sm font-medium ${className}`}>
      <Icon className="w-4 h-4" />
      <span>{config.label}</span>
    </div>
  );
};

// Terms Helper Hook
export const useTerms = () => {
  const { getTerm } = useUserType();
  return getTerm;
};

export default UserTypeAdapter;