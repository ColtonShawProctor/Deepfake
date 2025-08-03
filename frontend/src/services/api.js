import axios from 'axios';
import { getApiConfig } from '../config/api';

// Get API configuration
const config = getApiConfig() || {
  BASE_URL: 'http://localhost:8000',
  TIMEOUT: 30000,
  AUTH: {
    TOKEN_KEY: 'token',
    USER_KEY: 'user'
  }
};
console.log('API Config loaded:', config);

// Create axios instance with base configuration
const api = axios.create({
  baseURL: config.BASE_URL,
  timeout: config.TIMEOUT,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor to add auth token to all requests
api.interceptors.request.use(
  (requestConfig) => {
    const token = localStorage.getItem(config.AUTH.TOKEN_KEY);
    if (token) {
      requestConfig.headers.Authorization = `Bearer ${token}`;
    }
    
    // Add CSRF protection headers
    requestConfig.headers['X-Requested-With'] = 'XMLHttpRequest';
    
    return requestConfig;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor to handle common errors
api.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    // Handle common HTTP errors
    if (error.response) {
      const { status, data } = error.response;
      
      switch (status) {
        case 401:
          // Unauthorized - clear token and redirect to login
          localStorage.removeItem(config.AUTH.TOKEN_KEY);
          localStorage.removeItem(config.AUTH.USER_KEY);
          window.location.href = '/login';
          break;
        case 403:
          console.error('Access forbidden:', data);
          break;
        case 404:
          console.error('Resource not found:', data);
          break;
        case 422:
          console.error('Validation error:', data);
          break;
        case 500:
          console.error('Server error:', data);
          break;
        default:
          console.error(`HTTP ${status} error:`, data);
      }
    } else if (error.request) {
      // Network error
      console.error('Network error:', error.request);
    } else {
      // Other error
      console.error('Error:', error.message);
    }
    
    return Promise.reject(error);
  }
);

// Authentication API functions
export const authAPI = {
  // Register new user
  register: async (email, password) => {
    try {
      const response = await api.post('/auth/register', {
        email,
        password
      });
      
      // Store token and user data if registration includes auto-login
      if (response.data.access_token) {
        localStorage.setItem(config.AUTH.TOKEN_KEY, response.data.access_token);
        localStorage.setItem(config.AUTH.USER_KEY, JSON.stringify(response.data.user));
      }
      
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  // Login user
  login: async (email, password) => {
    try {
      // Convert to form data format expected by FastAPI OAuth2PasswordRequestForm
      const formData = new URLSearchParams();
      formData.append('username', email); // FastAPI expects 'username' field
      formData.append('password', password);
      
      const response = await api.post('/auth/login', formData, {
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
      });
      
      // Store token and user data
      if (response.data.access_token) {
        localStorage.setItem(config.AUTH.TOKEN_KEY, response.data.access_token);
        localStorage.setItem(config.AUTH.USER_KEY, JSON.stringify(response.data.user));
      }
      
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  // Get current user info
  getCurrentUser: async () => {
    try {
      const response = await api.get('/auth/me');
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  // Logout user
  logout: async () => {
    try {
      await api.post('/auth/logout');
    } catch (error) {
      // Even if logout fails, clear local storage
      console.error('Logout error:', error);
    } finally {
      // Clear local storage regardless of API response
      localStorage.removeItem('token');
      localStorage.removeItem('user');
    }
  },

  // Refresh token
  refreshToken: async () => {
    try {
      const response = await api.post('/auth/refresh');
      if (response.data.token) {
        localStorage.setItem(config.AUTH.TOKEN_KEY, response.data.token);
      }
      return response.data;
    } catch (error) {
      throw error;
    }
  }
};

// File upload API functions
export const fileAPI = {
  // Upload file
  uploadFile: async (file, onProgress = null) => {
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await api.post('/api/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          if (onProgress) {
            const percentCompleted = Math.round(
              (progressEvent.loaded * 100) / progressEvent.total
            );
            onProgress(percentCompleted);
          }
        },
      });
      
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  // Get upload status
  getUploadStatus: async (fileId) => {
    try {
      const response = await api.get(`/api/upload/${fileId}/status`);
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  // Delete uploaded file
  deleteFile: async (fileId) => {
    try {
      const response = await api.delete(`/api/upload/${fileId}`);
      return response.data;
    } catch (error) {
      throw error;
    }
  }
};

// Analysis API functions
export const analysisAPI = {
  // Analyze file
  analyzeFile: async (fileId, options = {}) => {
    try {
      const response = await api.post(`/api/analysis/analyze/${fileId}`, options);
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  // Get analysis status
  getAnalysisStatus: async (analysisId) => {
    try {
      const response = await api.get(`/analysis/${analysisId}/status`);
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  // Get analysis results
  getResults: async (fileId) => {
    try {
      const response = await api.get(`/api/analysis/results/${fileId}`);
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  // Get all user's analysis results
  getAllResults: async (page = 1, limit = 10) => {
    try {
      const response = await api.get('/api/analysis/history', {
        params: { page, limit }
      });
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  // Delete analysis result
  deleteAnalysis: async (analysisId) => {
    try {
      const response = await api.delete(`/analysis/${analysisId}`);
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  // Get analysis statistics
  getStats: async () => {
    try {
      const response = await api.get('/api/analysis/stats');
      return response.data;
    } catch (error) {
      throw error;
    }
  }
};

// User profile API functions
export const userAPI = {
  // Get user profile
  getProfile: async () => {
    try {
      const response = await api.get('/user/profile');
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  // Update user profile
  updateProfile: async (profileData) => {
    try {
      const response = await api.put('/user/profile', profileData);
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  // Change password
  changePassword: async (currentPassword, newPassword) => {
    try {
      const response = await api.put('/user/password', {
        current_password: currentPassword,
        new_password: newPassword
      });
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  // Get user statistics
  getStats: async () => {
    try {
      const response = await api.get('/user/stats');
      return response.data;
    } catch (error) {
      throw error;
    }
  }
};

// Utility functions
export const apiUtils = {
  // Check if user is authenticated
  isAuthenticated: () => {
    return !!localStorage.getItem(config.AUTH.TOKEN_KEY);
  },

  // Get stored token
  getToken: () => {
    return localStorage.getItem(config.AUTH.TOKEN_KEY);
  },

  // Get stored user data
  getUser: () => {
    const user = localStorage.getItem(config.AUTH.USER_KEY);
    return user ? JSON.parse(user) : null;
  },

  // Clear all stored data
  clearAuth: () => {
    localStorage.removeItem(config.AUTH.TOKEN_KEY);
    localStorage.removeItem(config.AUTH.USER_KEY);
  },

  // Handle API errors with user-friendly messages
  handleError: (error) => {
    if (error.response?.data?.detail) {
      return error.response.data.detail;
    } else if (error.response?.data?.message) {
      return error.response.data.message;
    } else if (error.message) {
      return error.message;
    } else {
      return 'An unexpected error occurred';
    }
  }
};

// Export the main api instance for custom requests
export default api; 