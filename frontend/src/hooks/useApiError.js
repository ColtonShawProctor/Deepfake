import { useState, useCallback } from 'react';
import { apiUtils } from '../services/api';

export const useApiError = () => {
  const [error, setError] = useState('');

  const handleError = useCallback((err) => {
    const errorMessage = apiUtils.handleError(err);
    setError(errorMessage);
    return errorMessage;
  }, []);

  const clearError = useCallback(() => {
    setError('');
  }, []);

  return {
    error,
    setError,
    handleError,
    clearError
  };
}; 