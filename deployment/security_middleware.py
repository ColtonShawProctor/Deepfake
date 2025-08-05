"""
Security Middleware for Deepfake Detection System

This module provides security middleware including rate limiting,
authentication, request validation, and security headers.
"""

import time
import hashlib
import hmac
import logging
from typing import Optional, Dict, Any
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.base import BaseHTTPMiddleware
import redis
from deployment.production_config import config, security_config

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware using Redis."""
    
    def __init__(self, app, redis_client: redis.Redis):
        super().__init__(app)
        self.redis_client = redis_client
        self.rate_limit_per_minute = config.RATE_LIMIT_PER_MINUTE
        self.rate_limit_burst = config.RATE_LIMIT_BURST
    
    async def dispatch(self, request: Request, call_next):
        # Get client identifier (IP address or API key)
        client_id = self._get_client_id(request)
        
        # Check rate limit
        if not self._check_rate_limit(client_id):
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Maximum {self.rate_limit_per_minute} requests per minute allowed"
                }
            )
        
        response = await call_next(request)
        return response
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting."""
        # Try to get API key first
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"api:{api_key}"
        
        # Fall back to IP address
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return f"ip:{forwarded_for.split(',')[0].strip()}"
        
        return f"ip:{request.client.host}"
    
    def _check_rate_limit(self, client_id: str) -> bool:
        """Check if client has exceeded rate limit."""
        try:
            current_time = int(time.time())
            window_start = current_time - 60  # 1 minute window
            
            # Use Redis sorted set for sliding window rate limiting
            key = f"rate_limit:{client_id}"
            
            # Remove old entries
            self.redis_client.zremrangebyscore(key, 0, window_start)
            
            # Count current requests
            current_requests = self.redis_client.zcard(key)
            
            if current_requests >= self.rate_limit_per_minute:
                return False
            
            # Add current request
            self.redis_client.zadd(key, {str(current_time): current_time})
            self.redis_client.expire(key, 120)  # Expire after 2 minutes
            
            return True
            
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            return True  # Allow request if rate limiting fails


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Add security headers
        for header, value in config.SECURITY_HEADERS.items():
            response.headers[header] = value
        
        return response


class RequestValidationMiddleware(BaseHTTPMiddleware):
    """Validate incoming requests for security."""
    
    async def dispatch(self, request: Request, call_next):
        # Validate request size
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > config.MAX_FILE_SIZE:
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={
                    "error": "File too large",
                    "message": f"Maximum file size is {config.MAX_FILE_SIZE // (1024*1024)}MB"
                }
            )
        
        # Validate content type for uploads
        if request.url.path.startswith("/upload"):
            content_type = request.headers.get("content-type", "")
            if not any(allowed_type in content_type for allowed_type in config.ALLOWED_FILE_TYPES):
                return JSONResponse(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    content={
                        "error": "Unsupported file type",
                        "message": f"Allowed types: {', '.join(config.ALLOWED_FILE_TYPES)}"
                    }
                )
        
        response = await call_next(request)
        return response


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Authentication middleware for protected endpoints."""
    
    def __init__(self, app, redis_client: redis.Redis):
        super().__init__(app)
        self.redis_client = redis_client
    
    async def dispatch(self, request: Request, call_next):
        # Skip authentication for public endpoints
        if self._is_public_endpoint(request.url.path):
            response = await call_next(request)
            return response
        
        # Validate authentication
        auth_header = request.headers.get("Authorization")
        api_key = request.headers.get("X-API-Key")
        
        if not auth_header and not api_key:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": "Authentication required",
                    "message": "Provide Authorization header or X-API-Key"
                }
            )
        
        # Validate token or API key
        if not await self._validate_auth(auth_header, api_key):
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": "Invalid authentication",
                    "message": "Invalid token or API key"
                }
            )
        
        response = await call_next(request)
        return response
    
    def _is_public_endpoint(self, path: str) -> bool:
        """Check if endpoint is public (no authentication required)."""
        public_paths = [
            "/health",
            "/docs",
            "/openapi.json",
            "/login",
            "/register",
            "/"
        ]
        return any(path.startswith(public_path) for public_path in public_paths)
    
    async def _validate_auth(self, auth_header: Optional[str], api_key: Optional[str]) -> bool:
        """Validate authentication token or API key."""
        try:
            if api_key:
                return await self._validate_api_key(api_key)
            elif auth_header and auth_header.startswith("Bearer "):
                token = auth_header.split(" ")[1]
                return await self._validate_jwt_token(token)
            return False
        except Exception as e:
            logger.error(f"Authentication validation failed: {e}")
            return False
    
    async def _validate_api_key(self, api_key: str) -> bool:
        """Validate API key."""
        try:
            # Check if API key exists and is not expired
            key_data = self.redis_client.get(f"api_key:{api_key}")
            if not key_data:
                return False
            
            # TODO: Implement API key validation logic
            return True
        except Exception as e:
            logger.error(f"API key validation failed: {e}")
            return False
    
    async def _validate_jwt_token(self, token: str) -> bool:
        """Validate JWT token."""
        try:
            # TODO: Implement JWT token validation
            # This would typically involve decoding and verifying the token
            return True
        except Exception as e:
            logger.error(f"JWT token validation failed: {e}")
            return False


class LoggingMiddleware(BaseHTTPMiddleware):
    """Logging middleware for request/response tracking."""
    
    def __init__(self, app, redis_client: redis.Redis):
        super().__init__(app)
        self.redis_client = redis_client
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Log request
        logger.info(f"Request: {request.method} {request.url.path} from {request.client.host}")
        
        try:
            response = await call_next(request)
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Log response
            logger.info(f"Response: {response.status_code} in {response_time:.3f}s")
            
            # Store metrics
            await self._store_metrics(request, response, response_time)
            
            return response
            
        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"Request failed: {request.method} {request.url.path} in {response_time:.3f}s - {e}")
            raise
    
    async def _store_metrics(self, request: Request, response, response_time: float):
        """Store request metrics for monitoring."""
        try:
            metrics = {
                "timestamp": time.time(),
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "response_time": response_time,
                "client_ip": request.client.host,
                "user_agent": request.headers.get("user-agent", ""),
                "content_length": request.headers.get("content-length", 0)
            }
            
            # Store in Redis for real-time metrics
            self.redis_client.lpush("api_metrics", str(metrics))
            self.redis_client.ltrim("api_metrics", 0, 999)  # Keep last 1000 metrics
            
        except Exception as e:
            logger.error(f"Failed to store metrics: {e}")


class FileSecurityMiddleware(BaseHTTPMiddleware):
    """File security middleware for upload validation."""
    
    async def dispatch(self, request: Request, call_next):
        if request.url.path.startswith("/upload"):
            # Validate file upload security
            if not await self._validate_file_upload(request):
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={
                        "error": "File upload validation failed",
                        "message": "File appears to be malicious or invalid"
                    }
                )
        
        response = await call_next(request)
        return response
    
    async def _validate_file_upload(self, request: Request) -> bool:
        """Validate file upload for security."""
        try:
            # Check file size
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > config.MAX_FILE_SIZE:
                return False
            
            # Check content type
            content_type = request.headers.get("content-type", "")
            if not any(allowed_type in content_type for allowed_type in config.ALLOWED_FILE_TYPES):
                return False
            
            # TODO: Implement additional file validation
            # - File signature validation
            # - Malware scanning
            # - File content analysis
            
            return True
            
        except Exception as e:
            logger.error(f"File upload validation failed: {e}")
            return False


def create_security_middleware(app, redis_client: redis.Redis):
    """Create and configure all security middleware."""
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RequestValidationMiddleware)
    app.add_middleware(RateLimitMiddleware, redis_client=redis_client)
    app.add_middleware(AuthenticationMiddleware, redis_client=redis_client)
    app.add_middleware(LoggingMiddleware, redis_client=redis_client)
    app.add_middleware(FileSecurityMiddleware)
    
    logger.info("Security middleware configured successfully") 