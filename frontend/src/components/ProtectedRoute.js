import React, { useEffect } from 'react';
import { Navigate } from 'react-router-dom';
import { useAuth } from './AuthContext';

const ProtectedRoute = ({ children }) => {
  const { user, loading, isAuthenticated, authChecked } = useAuth();
  
  console.log('ProtectedRoute - user:', user, 'loading:', loading, 'isAuthenticated:', isAuthenticated, 'authChecked:', authChecked);

  // Add effect to log state changes
  useEffect(() => {
    console.log('ProtectedRoute state changed:', {
      user: !!user,
      loading,
      isAuthenticated,
      authChecked
    });
  }, [user, loading, isAuthenticated, authChecked]);

  // Show loading spinner while checking authentication
  if (loading || !authChecked) {
    return (
      <div className="d-flex justify-content-center align-items-center" style={{ height: '100vh' }}>
        <div className="spinner-border text-primary" role="status">
          <span className="visually-hidden">Loading...</span>
        </div>
      </div>
    );
  }

  // Redirect to login if not authenticated
  if (!isAuthenticated) {
    console.log('ProtectedRoute: Redirecting to login - user not authenticated');
    return <Navigate to="/login" replace />;
  }

  // User is authenticated, render protected content
  console.log('ProtectedRoute: Rendering protected content for user:', user?.email);
  return children;
};

export default ProtectedRoute; 