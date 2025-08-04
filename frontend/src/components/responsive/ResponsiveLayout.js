/**
 * ResponsiveLayout - Adaptive layout system for multi-model deepfake detection
 * Optimizes experience across mobile, tablet, and desktop devices
 */

import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Menu, 
  X, 
  ChevronDown, 
  ChevronUp,
  Smartphone,
  Tablet,
  Monitor,
  Eye,
  EyeOff,
  Maximize2,
  Minimize2
} from 'lucide-react';

import MobileAnalysisInterface from './MobileAnalysisInterface';
import TabletAnalysisInterface from './TabletAnalysisInterface';
import DesktopAnalysisInterface from './DesktopAnalysisInterface';
import QuickActionsPanel from './QuickActionsPanel';
import CollapsibleSection from './CollapsibleSection';

const ResponsiveLayout = ({ 
  children, 
  userType = 'general',
  analysisState,
  onLayoutChange,
  className = ""
}) => {
  const [screenSize, setScreenSize] = useState('desktop');
  const [orientation, setOrientation] = useState('landscape');
  const [viewportDimensions, setViewportDimensions] = useState({ width: 0, height: 0 });
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [compactMode, setCompactMode] = useState(false);
  const [hiddenSections, setHiddenSections] = useState(new Set());
  const [fullscreenComponent, setFullscreenComponent] = useState(null);
  
  // Responsive breakpoints
  const breakpoints = {
    mobile: 768,
    tablet: 1024,
    desktop: 1200
  };

  // Update screen size and orientation
  useEffect(() => {
    const updateDimensions = () => {
      const width = window.innerWidth;
      const height = window.innerHeight;
      
      setViewportDimensions({ width, height });
      
      // Determine screen size
      if (width < breakpoints.mobile) {
        setScreenSize('mobile');
      } else if (width < breakpoints.tablet) {
        setScreenSize('tablet');
      } else {
        setScreenSize('desktop');
      }
      
      // Determine orientation
      setOrientation(width > height ? 'landscape' : 'portrait');
      
      // Auto-enable compact mode on small screens
      if (width < breakpoints.mobile) {
        setCompactMode(true);
      }
    };

    updateDimensions();
    window.addEventListener('resize', updateDimensions);
    
    return () => window.removeEventListener('resize', updateDimensions);
  }, []);

  // Handle device orientation change
  useEffect(() => {
    const handleOrientationChange = () => {
      setTimeout(updateDimensions, 100); // Delay to get accurate dimensions
    };

    window.addEventListener('orientationchange', handleOrientationChange);
    return () => window.removeEventListener('orientationchange', handleOrientationChange);
  }, []);

  // Layout configurations for different devices
  const layoutConfigs = {
    mobile: {
      sidebar: false,
      tabbed: true,
      stackedComponents: true,
      compactControls: true,
      virtualScrolling: true,
      gestureNavigation: true
    },
    tablet: {
      sidebar: orientation === 'landscape',
      tabbed: true,
      stackedComponents: orientation === 'portrait',
      compactControls: false,
      virtualScrolling: false,
      gestureNavigation: true
    },
    desktop: {
      sidebar: true,
      tabbed: false,
      stackedComponents: false,
      compactControls: false,
      virtualScrolling: false,
      gestureNavigation: false
    }
  };

  const currentConfig = layoutConfigs[screenSize];

  // Toggle section visibility
  const toggleSection = (sectionId) => {
    const newHidden = new Set(hiddenSections);
    if (newHidden.has(sectionId)) {
      newHidden.delete(sectionId);
    } else {
      newHidden.add(sectionId);
    }
    setHiddenSections(newHidden);
  };

  // Fullscreen component handler
  const toggleFullscreen = (componentId) => {
    setFullscreenComponent(fullscreenComponent === componentId ? null : componentId);
  };

  // Adaptive component sizing
  const getComponentSize = (component, defaultSize) => {
    if (fullscreenComponent === component) {
      return 'w-full h-full fixed inset-0 z-50';
    }
    
    if (compactMode || screenSize === 'mobile') {
      return 'w-full min-h-64';
    }
    
    return defaultSize;
  };

  // Render different layouts based on screen size
  const renderMobileLayout = () => (
    <div className="min-h-screen bg-slate-50">
      {/* Mobile Header */}
      <div className="bg-white border-b border-slate-200 sticky top-0 z-40">
        <div className="flex items-center justify-between p-4">
          <h1 className="text-lg font-semibold text-slate-900">Deepfake Detection</h1>
          <button
            onClick={() => setIsMenuOpen(!isMenuOpen)}
            className="p-2 text-slate-600 hover:text-slate-900"
          >
            {isMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
          </button>
        </div>
        
        {/* Mobile Menu */}
        <AnimatePresence>
          {isMenuOpen && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              className="border-t border-slate-200 bg-white"
            >
              <div className="p-4 space-y-2">
                <button className="w-full text-left py-2 px-3 rounded-lg text-slate-700 hover:bg-slate-100">
                  Upload Image
                </button>
                <button className="w-full text-left py-2 px-3 rounded-lg text-slate-700 hover:bg-slate-100">
                  View Results
                </button>
                <button className="w-full text-left py-2 px-3 rounded-lg text-slate-700 hover:bg-slate-100">
                  Settings
                </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Mobile Content */}
      <div className="p-4 space-y-4">
        <MobileAnalysisInterface
          analysisState={analysisState}
          userType={userType}
          orientation={orientation}
          onComponentToggle={toggleFullscreen}
        />
      </div>

      {/* Mobile Quick Actions */}
      <div className="fixed bottom-0 left-0 right-0 bg-white border-t border-slate-200 p-4">
        <QuickActionsPanel 
          mode="mobile"
          analysisState={analysisState}
          compact={true}
        />
      </div>
    </div>
  );

  const renderTabletLayout = () => (
    <div className="min-h-screen bg-slate-50">
      <div className={`flex ${orientation === 'portrait' ? 'flex-col' : 'flex-row'}`}>
        {/* Tablet Sidebar (landscape only) */}
        {orientation === 'landscape' && (
          <div className="w-64 bg-white border-r border-slate-200 flex-shrink-0">
            <div className="p-4">
              <h1 className="text-xl font-semibold text-slate-900 mb-4">
                Deepfake Detection
              </h1>
              
              <nav className="space-y-2">
                <button className="w-full text-left py-2 px-3 rounded-lg text-slate-700 hover:bg-slate-100">
                  Analysis
                </button>
                <button className="w-full text-left py-2 px-3 rounded-lg text-slate-700 hover:bg-slate-100">
                  Comparison
                </button>
                <button className="w-full text-left py-2 px-3 rounded-lg text-slate-700 hover:bg-slate-100">
                  Results
                </button>
              </nav>
            </div>
          </div>
        )}

        {/* Main Content */}
        <div className="flex-1 overflow-hidden">
          {orientation === 'portrait' && (
            <div className="bg-white border-b border-slate-200 p-4">
              <h1 className="text-xl font-semibold text-slate-900">
                Deepfake Detection
              </h1>
            </div>
          )}
          
          <div className="h-full overflow-y-auto p-4">
            <TabletAnalysisInterface
              analysisState={analysisState}
              userType={userType}
              orientation={orientation}
              onComponentToggle={toggleFullscreen}
            />
          </div>
        </div>
      </div>
    </div>
  );

  const renderDesktopLayout = () => (
    <div className="min-h-screen bg-slate-50">
      <div className="flex">
        {/* Desktop Sidebar */}
        <motion.div
          initial={false}
          animate={{ width: compactMode ? 64 : 256 }}
          className="bg-white border-r border-slate-200 flex-shrink-0 transition-all duration-300"
        >
          <div className="p-4">
            {!compactMode && (
              <h1 className="text-xl font-semibold text-slate-900 mb-4">
                Deepfake Detection
              </h1>
            )}
            
            <button
              onClick={() => setCompactMode(!compactMode)}
              className="w-full p-2 text-slate-600 hover:text-slate-900 hover:bg-slate-100 rounded-lg"
              title={compactMode ? 'Expand sidebar' : 'Collapse sidebar'}
            >
              {compactMode ? <Maximize2 className="w-5 h-5 mx-auto" /> : <Minimize2 className="w-5 h-5" />}
            </button>
            
            <nav className="mt-4 space-y-2">
              {[
                { id: 'analysis', label: 'Analysis', icon: Monitor },
                { id: 'comparison', label: 'Comparison', icon: Eye },
                { id: 'results', label: 'Results', icon: ChevronDown }
              ].map((item) => {
                const Icon = item.icon;
                return (
                  <button
                    key={item.id}
                    className="w-full flex items-center space-x-3 py-2 px-3 rounded-lg text-slate-700 hover:bg-slate-100"
                  >
                    <Icon className="w-5 h-5" />
                    {!compactMode && <span>{item.label}</span>}
                  </button>
                );
              })}
            </nav>
          </div>
        </motion.div>

        {/* Main Content */}
        <div className="flex-1 overflow-hidden">
          <div className="h-full overflow-y-auto p-6">
            <DesktopAnalysisInterface
              analysisState={analysisState}
              userType={userType}
              compactMode={compactMode}
              hiddenSections={hiddenSections}
              onSectionToggle={toggleSection}
              onComponentToggle={toggleFullscreen}
            />
          </div>
        </div>

        {/* Desktop Right Panel */}
        {!compactMode && userType !== 'general' && (
          <div className="w-80 bg-white border-l border-slate-200 flex-shrink-0">
            <div className="p-4">
              <h3 className="font-semibold text-slate-900 mb-4">Quick Insights</h3>
              <QuickActionsPanel 
                mode="desktop"
                analysisState={analysisState}
                compact={false}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );

  // Device-specific optimizations
  const renderDeviceOptimizations = () => {
    if (screenSize === 'mobile') {
      return (
        <style jsx>{`
          /* Mobile-specific optimizations */
          .analysis-image {
            max-height: 50vh;
            object-fit: contain;
          }
          
          .heatmap-overlay {
            pointer-events: none;
          }
          
          .touch-friendly {
            min-height: 44px;
            min-width: 44px;
          }
          
          /* Reduce animation complexity on mobile */
          .reduced-motion {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
          }
        `}</style>
      );
    }
    
    return null;
  };

  // Performance monitoring for different devices
  useEffect(() => {
    if (typeof window !== 'undefined' && 'performance' in window) {
      const observer = new PerformanceObserver((list) => {
        const entries = list.getEntries();
        entries.forEach((entry) => {
          // Log performance metrics for optimization
          if (entry.entryType === 'paint' && entry.name === 'first-contentful-paint') {
            console.log(`FCP on ${screenSize}: ${entry.startTime}ms`);
          }
        });
      });
      
      observer.observe({ entryTypes: ['paint', 'navigation'] });
      
      return () => observer.disconnect();
    }
  }, [screenSize]);

  // Render appropriate layout based on screen size
  const renderLayout = () => {
    switch (screenSize) {
      case 'mobile':
        return renderMobileLayout();
      case 'tablet':
        return renderTabletLayout();
      case 'desktop':
      default:
        return renderDesktopLayout();
    }
  };

  return (
    <div className={`responsive-layout ${screenSize} ${orientation} ${className}`}>
      {renderDeviceOptimizations()}
      
      {/* Fullscreen overlay */}
      <AnimatePresence>
        {fullscreenComponent && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4"
            onClick={() => setFullscreenComponent(null)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="bg-white rounded-lg max-w-full max-h-full overflow-auto"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="p-4">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold">Fullscreen View</h3>
                  <button
                    onClick={() => setFullscreenComponent(null)}
                    className="p-2 text-slate-500 hover:text-slate-700"
                  >
                    <X className="w-5 h-5" />
                  </button>
                </div>
                {/* Fullscreen component content would go here */}
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Device indicator (development only) */}
      {process.env.NODE_ENV === 'development' && (
        <div className="fixed bottom-4 left-4 bg-black text-white px-2 py-1 rounded text-xs z-50">
          {screenSize} - {orientation} - {viewportDimensions.width}x{viewportDimensions.height}
        </div>
      )}

      {renderLayout()}
    </div>
  );
};

export default ResponsiveLayout;