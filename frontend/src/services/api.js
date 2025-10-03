import axios from 'axios';
import { getApiConfig } from '../config/api';
import requestThrottle from '../utils/requestThrottle';

// Get API configuration
const config = getApiConfig();
console.log('API Config loaded:', config);
console.log('Token key:', config?.AUTH?.TOKEN_KEY);
console.log('User key:', config?.AUTH?.USER_KEY);

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
    const token = localStorage.getItem('token');
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
          // Only clear token and redirect if it's not a login/register request
          // This prevents infinite redirect loops during authentication
          const isAuthRequest = error.config?.url?.includes('/auth/login') || 
                               error.config?.url?.includes('/auth/register');
          
          if (!isAuthRequest) {
            // Unauthorized - clear token and redirect to login
            localStorage.removeItem('token');
            localStorage.removeItem('user');
            window.location.href = '/login';
          }
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
        // Store token directly in localStorage for consistency
        localStorage.setItem('token', response.data.access_token);
        localStorage.setItem('user', JSON.stringify(response.data.user));
        
        // Verify token was stored
        const storedToken = localStorage.getItem('token');
        if (!storedToken) {
          throw new Error('Token storage failed');
        }
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
        console.log('Storing token:', response.data.access_token.substring(0, 20) + '...');
        console.log('Storing user:', response.data.user);
        
        // Store token directly in localStorage for consistency
        localStorage.setItem('token', response.data.access_token);
        localStorage.setItem('user', JSON.stringify(response.data.user));
        
        // Verify token was stored
        const storedToken = localStorage.getItem('token');
        console.log('Token stored successfully:', storedToken ? 'Yes' : 'No');
        console.log('Stored token value:', storedToken ? storedToken.substring(0, 20) + '...' : 'None');
        
        if (!storedToken) {
          throw new Error('Token storage failed');
        }
      } else {
        console.error('No access_token in response:', response.data);
        throw new Error('No access token received');
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
      if (response.data.access_token) {
        localStorage.setItem('token', response.data.access_token);
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
  // Analyze file using Detection API
  analyzeFile: async (fileId, options = {}) => {
    const requestKey = `analyzeFile-${fileId}`;
    
    return requestThrottle.throttle(requestKey, async () => {
      try {
        // Use the detection analysis endpoint for all files
        const response = await api.post(`/api/detection/analyze/${fileId}`, options);
        return response.data;
      } catch (error) {
        throw error;
      }
    });
  },

  // Analyze file directly using Single Model API (for immediate analysis)
  analyzeFileDirect: async (file) => {
    try {
      const formData = new FormData();
      formData.append('file', file);

      // First upload the file
      const uploadResponse = await api.post('/api/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      
      // Check if upload was successful by looking for file_id
      if (uploadResponse.data.file_id) {
        const fileId = uploadResponse.data.file_id;
        
        // Then analyze the uploaded file
        const analysisResponse = await api.post(`/api/detection/analyze/${fileId}`);
        return analysisResponse.data;
      } else {
        throw new Error('File upload failed');
      }
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
    const requestKey = `getResults-${fileId}`;
    
    return requestThrottle.throttle(requestKey, async () => {
      try {
        // Use the detection router endpoint consistently
        const response = await api.get(`/api/detection/results/${fileId}`);
        return response.data;
      } catch (error) {
        throw error;
      }
    });
  },

    // Get all user's analysis results
  getAllResults: async (page = 1, limit = 10) => {
    try {
      // Use the detection router endpoint consistently
      const response = await api.get('/api/detection/results', {
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
  },

  // Video-specific API functions
    // Get video analysis results
  getVideoResults: async (fileId) => {
    try {
      const response = await api.get(`/api/video/results/${fileId}`);
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  // Get video analysis progress
  getVideoProgress: async (taskId) => {
    try {
      const response = await api.get(`/api/video/progress/${taskId}`);
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  // List user's video files
  listUserVideos: async () => {
    try {
      const response = await api.get('/api/video/list');
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
    return !!localStorage.getItem('token');
  },

  // Get stored token
  getToken: () => {
    return localStorage.getItem('token');
  },

  // Get stored user data
  getUser: () => {
    const user = localStorage.getItem('user');
    return user ? JSON.parse(user) : null;
  },

  // Clear all stored data
  clearAuth: () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
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