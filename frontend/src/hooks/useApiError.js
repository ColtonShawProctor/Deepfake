import { useState } from 'react';
import { apiUtils } from '../services/api';

export const useApiError = () => {
  const [error, setError] = useState('');

  const handleError = (err) => {
    const errorMessage = apiUtils.handleError(err);
    setError(errorMessage);
    return errorMessage;
  };

  const clearError = () => {
    setError('');
  };

  return {
    error,
    setError,
    handleError,
    clearError
  };
}; 