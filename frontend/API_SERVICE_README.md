# API Service Documentation

## Overview

The API service provides a comprehensive interface for communicating with the deepfake detection backend. It includes authentication, file upload, analysis, and user management functionality.

## Features

- ✅ **Authentication Management**: Login, register, logout, token refresh
- ✅ **File Upload**: Progress tracking, validation, status checking
- ✅ **Analysis**: Start analysis, check status, retrieve results
- ✅ **User Management**: Profile management, statistics
- ✅ **Error Handling**: Comprehensive error handling with user-friendly messages
- ✅ **Token Management**: Automatic token inclusion, refresh, and cleanup
- ✅ **Configuration**: Environment-based configuration

## File Structure

```
frontend/src/
├── services/
│   └── api.js              # Main API service
├── config/
│   └── api.js              # API configuration
├── hooks/
│   └── useApiError.js      # Error handling hook
└── components/
    └── AuthContext.js      # Authentication context
```

## API Endpoints

### Authentication
- `POST /auth/register` - Register new user
- `POST /auth/login` - Login user
- `GET /auth/me` - Get current user info
- `POST /auth/logout` - Logout user
- `POST /auth/refresh` - Refresh access token

### File Upload
- `POST /upload` - Upload file
- `GET /upload/{fileId}/status` - Get upload status
- `DELETE /upload/{fileId}` - Delete uploaded file

### Analysis
- `POST /analysis/{fileId}` - Start analysis
- `GET /analysis/{analysisId}/status` - Get analysis status
- `GET /analysis/{analysisId}/results` - Get analysis results
- `GET /analysis/results` - Get all user results
- `DELETE /analysis/{analysisId}` - Delete analysis

### User Management
- `GET /user/profile` - Get user profile
- `PUT /user/profile` - Update user profile
- `PUT /user/password` - Change password
- `GET /user/stats` - Get user statistics

## Usage Examples

### Authentication

```javascript
import { authAPI } from '../services/api';

// Register
const registerUser = async () => {
  try {
    const response = await authAPI.register('user@example.com', 'password123', 'username');
    console.log('Registration successful:', response);
  } catch (error) {
    console.error('Registration failed:', error);
  }
};

// Login
const loginUser = async () => {
  try {
    const response = await authAPI.login('user@example.com', 'password123');
    console.log('Login successful:', response);
  } catch (error) {
    console.error('Login failed:', error);
  }
};
```

### File Upload

```javascript
import { fileAPI } from '../services/api';

const uploadFile = async (file) => {
  try {
    const response = await fileAPI.uploadFile(file, (progress) => {
      console.log(`Upload progress: ${progress}%`);
    });
    console.log('Upload successful:', response);
  } catch (error) {
    console.error('Upload failed:', error);
  }
};
```

### Analysis

```javascript
import { analysisAPI } from '../services/api';

const analyzeFile = async (fileId) => {
  try {
    // Start analysis
    const analysis = await analysisAPI.analyzeFile(fileId);
    
    // Check status
    const status = await analysisAPI.getAnalysisStatus(analysis.id);
    
    // Get results
    const results = await analysisAPI.getResults(analysis.id);
    
    console.log('Analysis results:', results);
  } catch (error) {
    console.error('Analysis failed:', error);
  }
};
```

### Error Handling

```javascript
import { useApiError } from '../hooks/useApiError';

const MyComponent = () => {
  const { error, handleError, clearError } = useApiError();

  const handleApiCall = async () => {
    try {
      await someApiCall();
    } catch (err) {
      handleError(err);
    }
  };

  return (
    <div>
      {error && <div className="alert alert-danger">{error}</div>}
      <button onClick={handleApiCall}>Make API Call</button>
    </div>
  );
};
```

## Configuration

### Environment Variables

Create a `.env` file in the frontend directory:

```env
REACT_APP_API_URL=http://localhost:8000
```

### API Configuration

The configuration is managed in `src/config/api.js`:

```javascript
export const API_CONFIG = {
  BASE_URL: process.env.REACT_APP_API_URL || 'http://localhost:8000',
  TIMEOUT: 30000,
  UPLOAD: {
    MAX_FILE_SIZE: 50 * 1024 * 1024, // 50MB
    ALLOWED_TYPES: ['image/jpeg', 'image/png', 'video/mp4']
  },
  AUTH: {
    TOKEN_KEY: 'token',
    USER_KEY: 'user'
  }
};
```

## Error Handling

The API service includes comprehensive error handling:

- **401 Unauthorized**: Automatically clears tokens and redirects to login
- **403 Forbidden**: Logs access denied errors
- **404 Not Found**: Logs resource not found errors
- **422 Validation Error**: Logs validation errors
- **500 Server Error**: Logs server errors
- **Network Errors**: Handles connection issues

## Token Management

- **Automatic Inclusion**: Tokens are automatically added to request headers
- **Storage**: Tokens are stored in localStorage with configurable keys
- **Refresh**: Support for token refresh functionality
- **Cleanup**: Automatic cleanup on logout or token expiration

## File Upload Features

- **Progress Tracking**: Real-time upload progress
- **File Validation**: Type and size validation
- **FormData**: Proper multipart/form-data handling
- **Status Checking**: Upload status monitoring

## Integration with React Components

The API service is integrated with React components through:

1. **AuthContext**: Manages authentication state
2. **ProtectedRoute**: Protects routes based on authentication
3. **Custom Hooks**: Provides error handling utilities
4. **Component Integration**: Direct API calls in components

## Testing

To test the API service:

1. Start the backend server on `http://localhost:8000`
2. Start the React development server: `npm start`
3. Test authentication flows
4. Test file upload functionality
5. Test analysis workflows

## Security Considerations

- Tokens are stored in localStorage (consider httpOnly cookies for production)
- Automatic token cleanup on logout
- Error messages don't expose sensitive information
- File type and size validation
- HTTPS enforcement in production

## Future Enhancements

- [ ] Token refresh automation
- [ ] Request/response caching
- [ ] Offline support
- [ ] Request queuing
- [ ] Retry logic for failed requests
- [ ] Request/response logging
- [ ] API rate limiting handling 