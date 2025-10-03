import { useState, useCallback } from 'react';

export const useAnalysis = () => {
  const [analysisState, setAnalysisState] = useState('idle');
  const [analysisResults, setAnalysisResults] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState(null);

  const startAnalysis = useCallback(() => {
    setIsAnalyzing(true);
    setAnalysisState('running');
    setProgress(0);
    setError(null);
  }, []);

  const pauseAnalysis = useCallback(() => {
    setIsAnalyzing(false);
    setAnalysisState('paused');
  }, []);

  const resetAnalysis = useCallback(() => {
    setIsAnalyzing(false);
    setAnalysisState('idle');
    setProgress(0);
    setError(null);
    setAnalysisResults(null);
  }, []);

  return {
    analysisState,
    startAnalysis,
    pauseAnalysis,
    resetAnalysis,
    analysisResults,
    isAnalyzing,
    progress,
    error
  };
};





