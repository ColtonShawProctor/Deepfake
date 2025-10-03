import React, { createContext, useContext, useState, useEffect } from 'react';
import { authAPI, apiUtils } from '../services/api';

const AuthContext = createContext();

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [authChecked, setAuthChecked] = useState(false);

  useEffect(() => {
    // Check if user is authenticated and validate token
    const checkAuth = async () => {
      try {
        console.log('AuthContext: Starting auth check...');
        const token = localStorage.getItem('token');
        console.log('AuthContext: Found token:', token ? 'Yes' : 'No');
        
        if (token) {
          console.log('AuthContext: Validating token with backend...');
          const userData = await authAPI.getCurrentUser();
          console.log('AuthContext: Token valid, user data:', userData);
          setUser(userData);
        } else {
          console.log('AuthContext: No token found, user not authenticated');
          setUser(null);
        }
      } catch (error) {
        console.log('AuthContext: Auth check error:', error);
        // Only clear auth data if it's a 401 error (invalid token)
        // Don't clear on network errors or other issues
        if (error.response?.status === 401) {
          console.log('Token invalid, clearing auth data:', error);
          apiUtils.clearAuth();
          setUser(null);
        } else {
          console.log('Auth check failed but not clearing auth data:', error);
          // Keep existing auth data on non-401 errors
        }
      } finally {
        console.log('AuthContext: Auth check completed');
        setLoading(false);
        setAuthChecked(true);
      }
    };

    checkAuth();
  }, []);

  const login = async (email, password) => {
    try {
      const response = await authAPI.login(email, password);
      console.log('AuthContext login response:', response);
      
      // Store token and user data first
      if (response.access_token) {
        localStorage.setItem('token', response.access_token);
        localStorage.setItem('user', JSON.stringify(response.user));
        
        // Verify token was stored
        const storedToken = localStorage.getItem('token');
        if (!storedToken) {
          throw new Error('Token storage failed');
        }
        
        // Update user state immediately after successful token storage
        setUser(response.user);
        console.log('AuthContext user set to:', response.user);
        
        // Small delay to ensure state update is processed
        await new Promise(resolve => setTimeout(resolve, 100));
        
        return response;
      } else {
        throw new Error('No access token in response');
      }
    } catch (error) {
      console.error('Login error in AuthContext:', error);
      throw error;
    }
  };

  const register = async (email, password) => {
    try {
      const response = await authAPI.register(email, password);
      console.log('AuthContext register response:', response);
      // Auto-login after successful registration
      if (response.access_token) {
        // Store token and user data
        localStorage.setItem('token', response.access_token);
        localStorage.setItem('user', JSON.stringify(response.user));
        
        // Update user state
        setUser(response.user);
        console.log('AuthContext user set to:', response.user);
        
        // Small delay to ensure state update is processed
        await new Promise(resolve => setTimeout(resolve, 100));
      }
      return response;
    } catch (error) {
      throw error;
    }
  };

  const logout = async () => {
    try {
      await authAPI.logout();
    } catch (error) {
      console.error('Logout error:', error);
    } finally {
      setUser(null);
      apiUtils.clearAuth();
    }
  };

  // Improved isAuthenticated logic with better timing
  const isAuthenticated = user !== null && authChecked && !loading;

  const value = {
    user,
    login,
    register,
    logout,
    loading,
    authChecked,
    isAuthenticated
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}; 