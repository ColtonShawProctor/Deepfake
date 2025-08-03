// API Configuration
export const API_CONFIG = {
  // Base URL for API requests
  BASE_URL: process.env.REACT_APP_API_URL || 'http://localhost:8000',
  
  // Request timeout in milliseconds
  TIMEOUT: 30000,
  
  // File upload settings
  UPLOAD: {
    MAX_FILE_SIZE: 50 * 1024 * 1024, // 50MB
    ALLOWED_TYPES: [
      'image/jpeg',
      'image/png', 
      'image/jpg',
      'video/mp4',
      'video/avi',
      'video/mov'
    ]
  },
  
  // Authentication settings
  AUTH: {
    TOKEN_KEY: 'token',
    USER_KEY: 'user',
    REFRESH_THRESHOLD: 5 * 60 * 1000 // 5 minutes before token expires
  },
  
  // Pagination settings
  PAGINATION: {
    DEFAULT_PAGE_SIZE: 10,
    MAX_PAGE_SIZE: 100
  }
};

// Environment-specific settings
export const getApiConfig = () => {
  const env = process.env.NODE_ENV;
  
  switch (env) {
    case 'production':
      return {
        ...API_CONFIG,
        BASE_URL: process.env.REACT_APP_API_URL || 'https://api.deepfake-detection.com'
      };
    case 'development':
      return {
        ...API_CONFIG,
        BASE_URL: process.env.REACT_APP_API_URL || 'http://localhost:8000'
      };
    default:
      return API_CONFIG;
  }
}; 