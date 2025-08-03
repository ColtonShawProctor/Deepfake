import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { useApiError } from '../hooks/useApiError';

const ForgotPassword = () => {
  const [email, setEmail] = useState('');
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);
  const [validationError, setValidationError] = useState('');
  
  const { error, handleError, clearError } = useApiError();

  // Clear errors when email changes
  useEffect(() => {
    if (validationError) {
      setValidationError('');
    }
    if (error) {
      clearError();
    }
  }, [email, validationError, error, clearError]);

  const validateEmail = (email) => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!email.trim()) {
      setValidationError('Email is required');
      return;
    }

    if (!validateEmail(email)) {
      setValidationError('Please enter a valid email address');
      return;
    }

    setLoading(true);
    clearError();
    setValidationError('');

    try {
      // This would call your backend API for password reset
      // await authAPI.forgotPassword(email);
      
      // For demo purposes, simulate API call
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      setSuccess(true);
    } catch (err) {
      handleError(err);
    } finally {
      setLoading(false);
    }
  };

  const handleEmailChange = (e) => {
    setEmail(e.target.value);
  };

  if (success) {
    return (
      <div className="container mt-5">
        <div className="row justify-content-center">
          <div className="col-md-6 col-lg-4">
            <div className="card shadow-lg border-0">
              <div className="card-header bg-success text-white text-center py-3">
                <h3 className="mb-0">
                  <i className="fas fa-check-circle me-2"></i>
                  Email Sent
                </h3>
              </div>
              
              <div className="card-body p-4 text-center">
                <div className="mb-4">
                  <i className="fas fa-envelope-open fa-3x text-success mb-3"></i>
                  <h5>Check Your Email</h5>
                  <p className="text-muted">
                    We've sent a password reset link to <strong>{email}</strong>
                  </p>
                </div>
                
                <div className="alert alert-info">
                  <small>
                    <i className="fas fa-info-circle me-2"></i>
                    If you don't see the email, check your spam folder.
                  </small>
                </div>
                
                <div className="d-grid gap-2">
                  <Link to="/login" className="btn btn-primary">
                    <i className="fas fa-arrow-left me-2"></i>
                    Back to Login
                  </Link>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="container mt-5">
      <div className="row justify-content-center">
        <div className="col-md-6 col-lg-4">
          <div className="card shadow-lg border-0">
            <div className="card-header bg-warning text-dark text-center py-3">
              <h3 className="mb-0">
                <i className="fas fa-key me-2"></i>
                Reset Password
              </h3>
            </div>
            
            <div className="card-body p-4">
              <p className="text-muted text-center mb-4">
                Enter your email address and we'll send you a link to reset your password.
              </p>

              {error && (
                <div className="alert alert-danger alert-dismissible fade show" role="alert">
                  <i className="fas fa-exclamation-triangle me-2"></i>
                  {error}
                  <button 
                    type="button" 
                    className="btn-close" 
                    onClick={clearError}
                    aria-label="Close"
                  ></button>
                </div>
              )}

              {validationError && (
                <div className="alert alert-warning alert-dismissible fade show" role="alert">
                  <i className="fas fa-exclamation-triangle me-2"></i>
                  {validationError}
                  <button 
                    type="button" 
                    className="btn-close" 
                    onClick={() => setValidationError('')}
                    aria-label="Close"
                  ></button>
                </div>
              )}

              <form onSubmit={handleSubmit} noValidate>
                <div className="mb-3">
                  <label htmlFor="email" className="form-label">
                    <i className="fas fa-envelope me-2"></i>
                    Email Address
                  </label>
                  <input
                    type="email"
                    className="form-control"
                    id="email"
                    name="email"
                    value={email}
                    onChange={handleEmailChange}
                    placeholder="Enter your email address"
                    disabled={loading}
                    autoComplete="email"
                    required
                  />
                </div>

                <div className="d-grid mb-3">
                  <button
                    type="submit"
                    className="btn btn-warning btn-lg"
                    disabled={loading || !email.trim()}
                  >
                    {loading ? (
                      <>
                        <span className="spinner-border spinner-border-sm me-2" role="status"></span>
                        Sending...
                      </>
                    ) : (
                      <>
                        <i className="fas fa-paper-plane me-2"></i>
                        Send Reset Link
                      </>
                    )}
                  </button>
                </div>
              </form>
            </div>

            <div className="card-footer text-center py-3 bg-light">
              <p className="mb-0">
                Remember your password?{' '}
                <Link to="/login" className="text-decoration-none fw-bold">
                  <i className="fas fa-sign-in-alt me-1"></i>
                  Sign in here
                </Link>
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ForgotPassword; 