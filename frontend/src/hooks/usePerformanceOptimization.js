/**
 * Performance Optimization Hook
 * Implements comprehensive performance strategies for the multi-model deepfake detection interface
 */

import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { debounce, throttle } from 'lodash';

// Web Workers for heavy computations
const createWorker = (workerFunction) => {
  const blob = new Blob([`(${workerFunction.toString()})()`], { type: 'application/javascript' });
  return new Worker(URL.createObjectURL(blob));
};

// Image processing worker
const imageProcessingWorker = () => {
  self.onmessage = function(e) {
    const { imageData, operation, params } = e.data;
    
    switch (operation) {
      case 'resize':
        // Implement image resizing logic
        break;
      case 'compress':
        // Implement image compression logic
        break;
      case 'heatmapOverlay':
        // Implement heatmap overlay computation
        break;
      default:
        break;
    }
  };
};

export const usePerformanceOptimization = () => {
  const [performanceMetrics, setPerformanceMetrics] = useState({
    renderTime: 0,
    memoryUsage: 0,
    fps: 60,
    componentCount: 0
  });
  
  const [optimizationSettings, setOptimizationSettings] = useState({
    enableVirtualization: true,
    lazyLoadImages: true,
    debounceDelay: 300,
    throttleDelay: 16,
    maxConcurrentOperations: 3,
    enableWebWorkers: true,
    enableMemoization: true,
    reduceAnimations: false
  });

  const workerRef = useRef(null);
  const performanceObserverRef = useRef(null);
  const renderTimeRef = useRef(0);
  const componentCountRef = useRef(0);

  // Initialize Web Worker
  useEffect(() => {
    if (optimizationSettings.enableWebWorkers && !workerRef.current) {
      try {
        workerRef.current = createWorker(imageProcessingWorker);
      } catch (error) {
        console.warn('Web Workers not supported:', error);
        setOptimizationSettings(prev => ({ ...prev, enableWebWorkers: false }));
      }
    }

    return () => {
      if (workerRef.current) {
        workerRef.current.terminate();
        workerRef.current = null;
      }
    };
  }, [optimizationSettings.enableWebWorkers]);

  // Performance monitoring
  useEffect(() => {
    if ('PerformanceObserver' in window) {
      performanceObserverRef.current = new PerformanceObserver((list) => {
        const entries = list.getEntries();
        
        entries.forEach((entry) => {
          if (entry.entryType === 'measure') {
            setPerformanceMetrics(prev => ({
              ...prev,
              renderTime: entry.duration
            }));
          }
          
          if (entry.entryType === 'paint') {
            console.log(`${entry.name}: ${entry.startTime}ms`);
          }
        });
      });

      performanceObserverRef.current.observe({ 
        entryTypes: ['measure', 'paint', 'navigation'] 
      });
    }

    // Memory usage monitoring
    const memoryInterval = setInterval(() => {
      if ('memory' in performance) {
        setPerformanceMetrics(prev => ({
          ...prev,
          memoryUsage: performance.memory.usedJSHeapSize / 1024 / 1024 // MB
        }));
      }
    }, 5000);

    return () => {
      if (performanceObserverRef.current) {
        performanceObserverRef.current.disconnect();
      }
      clearInterval(memoryInterval);
    };
  }, []);

  // FPS monitoring
  useEffect(() => {
    let frameCount = 0;
    let lastTime = performance.now();
    
    const measureFPS = () => {
      frameCount++;
      const currentTime = performance.now();
      
      if (currentTime - lastTime >= 1000) {
        setPerformanceMetrics(prev => ({
          ...prev,
          fps: Math.round((frameCount * 1000) / (currentTime - lastTime))
        }));
        
        frameCount = 0;
        lastTime = currentTime;
      }
      
      requestAnimationFrame(measureFPS);
    };
    
    requestAnimationFrame(measureFPS);
  }, []);

  // Adaptive optimization based on device capabilities
  useEffect(() => {
    const deviceMemory = navigator.deviceMemory || 4; // GB
    const hardwareConcurrency = navigator.hardwareConcurrency || 4;
    const connection = navigator.connection;
    
    const isLowEndDevice = deviceMemory <= 2 || hardwareConcurrency <= 2;
    const isSlowConnection = connection && (connection.effectiveType === 'slow-2g' || connection.effectiveType === '2g');
    
    if (isLowEndDevice || isSlowConnection) {
      setOptimizationSettings(prev => ({
        ...prev,
        reduceAnimations: true,
        maxConcurrentOperations: 1,
        debounceDelay: 500,
        enableVirtualization: true
      }));
    }
  }, []);

  // Optimized debounce hook
  const useOptimizedDebounce = useCallback((callback, delay = optimizationSettings.debounceDelay) => {
    return useMemo(
      () => debounce(callback, delay, { leading: false, trailing: true }),
      [callback, delay]
    );
  }, [optimizationSettings.debounceDelay]);

  // Optimized throttle hook
  const useOptimizedThrottle = useCallback((callback, delay = optimizationSettings.throttleDelay) => {
    return useMemo(
      () => throttle(callback, delay, { leading: true, trailing: false }),
      [callback, delay]
    );
  }, [optimizationSettings.throttleDelay]);

  // Image optimization utilities
  const optimizeImage = useCallback(async (imageFile, options = {}) => {
    const {
      maxWidth = 1920,
      maxHeight = 1080,
      quality = 0.8,
      format = 'webp'
    } = options;

    return new Promise((resolve) => {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      const img = new Image();

      img.onload = () => {
        // Calculate optimal dimensions
        let { width, height } = img;
        
        if (width > maxWidth) {
          height = (height * maxWidth) / width;
          width = maxWidth;
        }
        
        if (height > maxHeight) {
          width = (width * maxHeight) / height;
          height = maxHeight;
        }

        canvas.width = width;
        canvas.height = height;

        // Draw and compress
        ctx.drawImage(img, 0, 0, width, height);
        
        canvas.toBlob(resolve, `image/${format}`, quality);
      };

      img.src = URL.createObjectURL(imageFile);
    });
  }, []);

  // Lazy loading with Intersection Observer
  const useLazyLoading = useCallback((ref, options = {}) => {
    const [isIntersecting, setIsIntersecting] = useState(false);
    const [isLoaded, setIsLoaded] = useState(false);

    useEffect(() => {
      if (!optimizationSettings.lazyLoadImages || !ref.current) return;

      const observer = new IntersectionObserver(
        ([entry]) => {
          if (entry.isIntersecting) {
            setIsIntersecting(true);
            observer.unobserve(entry.target);
          }
        },
        {
          rootMargin: '50px',
          threshold: 0.1,
          ...options
        }
      );

      observer.observe(ref.current);

      return () => observer.disconnect();
    }, [ref, options]);

    return { isIntersecting, isLoaded, setIsLoaded };
  }, [optimizationSettings.lazyLoadImages]);

  // Virtual scrolling for large lists
  const useVirtualScrolling = useCallback((items, itemHeight = 100, containerHeight = 400) => {
    const [scrollTop, setScrollTop] = useState(0);
    
    const visibleItems = useMemo(() => {
      if (!optimizationSettings.enableVirtualization) {
        return items;
      }

      const startIndex = Math.floor(scrollTop / itemHeight);
      const endIndex = Math.min(
        startIndex + Math.ceil(containerHeight / itemHeight) + 1,
        items.length
      );

      return items.slice(startIndex, endIndex).map((item, index) => ({
        ...item,
        index: startIndex + index,
        offsetY: (startIndex + index) * itemHeight
      }));
    }, [items, scrollTop, itemHeight, containerHeight, optimizationSettings.enableVirtualization]);

    const totalHeight = items.length * itemHeight;

    const handleScroll = useOptimizedThrottle((e) => {
      setScrollTop(e.target.scrollTop);
    });

    return {
      visibleItems,
      totalHeight,
      handleScroll
    };
  }, [optimizationSettings.enableVirtualization, useOptimizedThrottle]);

  // Memoized component wrapper
  const MemoizedComponent = useCallback((Component) => {
    if (!optimizationSettings.enableMemoization) {
      return Component;
    }

    return React.memo(Component, (prevProps, nextProps) => {
      // Custom comparison logic for deep equality
      return JSON.stringify(prevProps) === JSON.stringify(nextProps);
    });
  }, [optimizationSettings.enableMemoization]);

  // Bundle optimization utilities
  const preloadCriticalResources = useCallback(() => {
    const criticalResources = [
      '/api/models/info',
      '/static/js/analysis-worker.js',
      '/static/css/critical.css'
    ];

    criticalResources.forEach(resource => {
      const link = document.createElement('link');
      link.rel = 'preload';
      link.href = resource;
      link.as = resource.endsWith('.js') ? 'script' : 
               resource.endsWith('.css') ? 'style' : 'fetch';
      document.head.appendChild(link);
    });
  }, []);

  // Code splitting utilities
  const loadComponentAsync = useCallback((importFunction) => {
    return React.lazy(() => 
      importFunction().catch(() => ({
        default: () => React.createElement('div', null, 'Failed to load component')
      }))
    );
  }, []);

  // Cache management
  const useCacheManager = useCallback(() => {
    const cache = useRef(new Map());
    const maxCacheSize = 100;

    const get = (key) => cache.current.get(key);
    
    const set = (key, value) => {
      if (cache.current.size >= maxCacheSize) {
        const firstKey = cache.current.keys().next().value;
        cache.current.delete(firstKey);
      }
      cache.current.set(key, value);
    };

    const clear = () => cache.current.clear();

    return { get, set, clear };
  }, []);

  // Performance measurement utilities
  const measurePerformance = useCallback((name, fn) => {
    performance.mark(`${name}-start`);
    const result = fn();
    performance.mark(`${name}-end`);
    performance.measure(name, `${name}-start`, `${name}-end`);
    return result;
  }, []);

  // Component render tracking
  const useRenderTracking = useCallback((componentName) => {
    const renderCount = useRef(0);
    
    useEffect(() => {
      renderCount.current += 1;
      componentCountRef.current += 1;
      
      setPerformanceMetrics(prev => ({
        ...prev,
        componentCount: componentCountRef.current
      }));

      if (process.env.NODE_ENV === 'development') {
        console.log(`${componentName} rendered ${renderCount.current} times`);
      }
    });

    return renderCount.current;
  }, []);

  // Batch state updates
  const useBatchedUpdates = useCallback(() => {
    const pendingUpdates = useRef([]);
    const batchTimeout = useRef(null);

    const batchUpdate = (updateFn) => {
      pendingUpdates.current.push(updateFn);
      
      if (batchTimeout.current) {
        clearTimeout(batchTimeout.current);
      }

      batchTimeout.current = setTimeout(() => {
        React.unstable_batchedUpdates(() => {
          pendingUpdates.current.forEach(fn => fn());
          pendingUpdates.current = [];
        });
      }, 0);
    };

    return batchUpdate;
  }, []);

  // WebWorker communication helper
  const useWebWorker = useCallback((operation, data) => {
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    useEffect(() => {
      if (!workerRef.current || !optimizationSettings.enableWebWorkers) {
        return;
      }

      setLoading(true);
      setError(null);

      const handleMessage = (e) => {
        setResult(e.data);
        setLoading(false);
      };

      const handleError = (e) => {
        setError(e.error);
        setLoading(false);
      };

      workerRef.current.onmessage = handleMessage;
      workerRef.current.onerror = handleError;

      workerRef.current.postMessage({ operation, data });

      return () => {
        if (workerRef.current) {
          workerRef.current.onmessage = null;
          workerRef.current.onerror = null;
        }
      };
    }, [operation, data]);

    return { result, loading, error };
  }, [optimizationSettings.enableWebWorkers]);

  return {
    // Metrics
    performanceMetrics,
    
    // Settings
    optimizationSettings,
    setOptimizationSettings,
    
    // Optimization hooks
    useOptimizedDebounce,
    useOptimizedThrottle,
    useLazyLoading,
    useVirtualScrolling,
    useCacheManager,
    useRenderTracking,
    useBatchedUpdates,
    useWebWorker,
    
    // Utility functions
    optimizeImage,
    measurePerformance,
    preloadCriticalResources,
    loadComponentAsync,
    MemoizedComponent
  };
};

// Performance monitoring component
export const PerformanceMonitor = ({ visible = false }) => {
  const { performanceMetrics } = usePerformanceOptimization();

  if (!visible || process.env.NODE_ENV !== 'development') {
    return null;
  }

  return (
    <div className="fixed top-4 right-4 bg-black text-white p-4 rounded-lg text-sm font-mono z-50">
      <div>FPS: {performanceMetrics.fps}</div>
      <div>Render: {performanceMetrics.renderTime.toFixed(2)}ms</div>
      <div>Memory: {performanceMetrics.memoryUsage.toFixed(1)}MB</div>
      <div>Components: {performanceMetrics.componentCount}</div>
    </div>
  );
};

export default usePerformanceOptimization;