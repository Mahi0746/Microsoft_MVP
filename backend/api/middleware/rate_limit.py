# HealthSync AI - Rate Limiting Middleware
import time
from typing import Dict, Optional
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
import structlog
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from config_flexible import settings
from services.db_service import redis_client


logger = structlog.get_logger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware using Redis for distributed rate limiting."""
    
    def __init__(self, app):
        super().__init__(app)
        
        # Rate limit configurations per endpoint type
        self.rate_limits = {
            "voice": {
                "limit": settings.rate_limit_voice,
                "window": 60,  # 1 minute
                "paths": ["/api/v1/voice/"]
            },
            "image": {
                "limit": settings.rate_limit_image,
                "window": 60,
                "paths": ["/api/v1/scan/", "/api/v1/future/"]
            },
            "ml": {
                "limit": settings.rate_limit_ml,
                "window": 60,
                "paths": ["/api/v1/health/predictions", "/api/v1/health/train-model"]
            },
            "general": {
                "limit": settings.rate_limit_general,
                "window": 60,
                "paths": ["/api/v1/"]
            }
        }
    
    async def dispatch(self, request: Request, call_next):
        """Apply rate limiting based on endpoint and user."""
        
        # Skip rate limiting for public endpoints
        if self._is_exempt_path(request.url.path):
            return await call_next(request)
        
        try:
            # Get user identifier
            user_id = self._get_user_identifier(request)
            
            # Determine rate limit type
            limit_type = self._get_limit_type(request.url.path)
            
            # Check rate limit
            allowed = await self._check_rate_limit(user_id, limit_type, request.url.path)
            
            if not allowed:
                # Get remaining limit info
                remaining = await self._get_remaining_limit(user_id, limit_type)
                reset_time = await self._get_reset_time(user_id, limit_type)
                
                logger.warning(
                    "Rate limit exceeded",
                    user_id=user_id,
                    limit_type=limit_type,
                    path=request.url.path,
                    remaining=remaining,
                    reset_time=reset_time
                )
                
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded",
                    headers={
                        "X-RateLimit-Limit": str(self.rate_limits[limit_type]["limit"]),
                        "X-RateLimit-Remaining": str(remaining),
                        "X-RateLimit-Reset": str(reset_time),
                        "Retry-After": str(self.rate_limits[limit_type]["window"])
                    }
                )
            
            # Process request
            response = await call_next(request)
            
            # Add rate limit headers to response
            remaining = await self._get_remaining_limit(user_id, limit_type)
            response.headers["X-RateLimit-Limit"] = str(self.rate_limits[limit_type]["limit"])
            response.headers["X-RateLimit-Remaining"] = str(remaining)
            
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Rate limiting error", error=str(e))
            # Continue without rate limiting on error
            return await call_next(request)
    
    def _is_exempt_path(self, path: str) -> bool:
        """Check if path is exempt from rate limiting."""
        
        exempt_paths = [
            "/health",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/",
            "/favicon.ico"
        ]
        
        exempt_prefixes = [
            "/static/",
            "/ws/"  # WebSocket connections
        ]
        
        # Exact matches
        if path in exempt_paths:
            return True
        
        # Prefix matches
        for prefix in exempt_prefixes:
            if path.startswith(prefix):
                return True
        
        return False
    
    def _get_user_identifier(self, request: Request) -> str:
        """Get user identifier for rate limiting."""
        
        # Use authenticated user ID if available
        if hasattr(request.state, "user") and request.state.user:
            return f"user:{request.state.user['user_id']}"
        
        # Fall back to IP address for unauthenticated requests
        client_ip = get_remote_address(request)
        return f"ip:{client_ip}"
    
    def _get_limit_type(self, path: str) -> str:
        """Determine rate limit type based on path."""
        
        # Check specific endpoint types first
        for limit_type, config in self.rate_limits.items():
            if limit_type == "general":
                continue
            
            for endpoint_path in config["paths"]:
                if path.startswith(endpoint_path):
                    return limit_type
        
        # Default to general rate limit
        return "general"
    
    async def _check_rate_limit(self, user_id: str, limit_type: str, path: str) -> bool:
        """Check if request is within rate limit."""
        
        try:
            config = self.rate_limits[limit_type]
            key = f"rate_limit:{limit_type}:{user_id}"
            
            # Use Redis sliding window rate limiting
            current_time = int(time.time())
            window_start = current_time - config["window"]
            
            # Remove old entries
            await redis_client.zremrangebyscore(key, 0, window_start)
            
            # Count current requests in window
            current_count = await redis_client.zcard(key)
            
            if current_count >= config["limit"]:
                return False
            
            # Add current request
            await redis_client.zadd(key, {str(current_time): current_time})
            
            # Set expiry for cleanup
            await redis_client.expire(key, config["window"])
            
            logger.debug(
                "Rate limit check",
                user_id=user_id,
                limit_type=limit_type,
                current_count=current_count,
                limit=config["limit"],
                path=path
            )
            
            return True
            
        except Exception as e:
            logger.error("Rate limit check failed", error=str(e))
            # Allow request on Redis failure
            return True
    
    async def _get_remaining_limit(self, user_id: str, limit_type: str) -> int:
        """Get remaining requests in current window."""
        
        try:
            config = self.rate_limits[limit_type]
            key = f"rate_limit:{limit_type}:{user_id}"
            
            current_count = await redis_client.zcard(key)
            remaining = max(0, config["limit"] - current_count)
            
            return remaining
            
        except Exception as e:
            logger.error("Failed to get remaining limit", error=str(e))
            return 0
    
    async def _get_reset_time(self, user_id: str, limit_type: str) -> int:
        """Get timestamp when rate limit resets."""
        
        try:
            config = self.rate_limits[limit_type]
            return int(time.time()) + config["window"]
            
        except Exception as e:
            logger.error("Failed to get reset time", error=str(e))
            return int(time.time()) + 60


# =============================================================================
# SLOWAPI INTEGRATION (Alternative Implementation)
# =============================================================================

# Create limiter instance for decorator-based rate limiting
limiter = Limiter(key_func=get_remote_address)


def create_limiter_with_redis():
    """Create Slowapi limiter with Redis backend."""
    
    def get_user_id_or_ip(request: Request) -> str:
        """Get user ID or IP for rate limiting key."""
        
        if hasattr(request.state, "user") and request.state.user:
            return request.state.user["user_id"]
        
        return get_remote_address(request)
    
    return Limiter(
        key_func=get_user_id_or_ip,
        storage_uri=settings.redis_url
    )


# Enhanced limiter with Redis
redis_limiter = create_limiter_with_redis()


# =============================================================================
# RATE LIMIT DECORATORS
# =============================================================================

def rate_limit_voice(func):
    """Decorator for voice endpoint rate limiting."""
    return redis_limiter.limit(f"{settings.rate_limit_voice}/minute")(func)


def rate_limit_image(func):
    """Decorator for image endpoint rate limiting."""
    return redis_limiter.limit(f"{settings.rate_limit_image}/minute")(func)


def rate_limit_ml(func):
    """Decorator for ML endpoint rate limiting."""
    return redis_limiter.limit(f"{settings.rate_limit_ml}/minute")(func)


def rate_limit_general(func):
    """Decorator for general endpoint rate limiting."""
    return redis_limiter.limit(f"{settings.rate_limit_general}/minute")(func)


# =============================================================================
# RATE LIMIT UTILITIES
# =============================================================================

async def get_rate_limit_status(user_id: str, limit_type: str) -> Dict[str, int]:
    """Get current rate limit status for user."""
    
    try:
        # Create middleware instance to access methods
        middleware = RateLimitMiddleware(None)
        
        config = middleware.rate_limits[limit_type]
        key = f"rate_limit:{limit_type}:{user_id}"
        
        # Clean old entries
        current_time = int(time.time())
        window_start = current_time - config["window"]
        await redis_client.zremrangebyscore(key, 0, window_start)
        
        # Get current usage
        current_count = await redis_client.zcard(key)
        remaining = max(0, config["limit"] - current_count)
        reset_time = current_time + config["window"]
        
        return {
            "limit": config["limit"],
            "used": current_count,
            "remaining": remaining,
            "reset_time": reset_time,
            "window_seconds": config["window"]
        }
        
    except Exception as e:
        logger.error("Failed to get rate limit status", error=str(e))
        return {
            "limit": 0,
            "used": 0,
            "remaining": 0,
            "reset_time": int(time.time()),
            "window_seconds": 60
        }


async def reset_rate_limit(user_id: str, limit_type: Optional[str] = None):
    """Reset rate limit for user (admin function)."""
    
    try:
        if limit_type:
            # Reset specific limit type
            key = f"rate_limit:{limit_type}:{user_id}"
            await redis_client.delete(key)
            logger.info("Rate limit reset", user_id=user_id, limit_type=limit_type)
        else:
            # Reset all limit types for user
            middleware = RateLimitMiddleware(None)
            for lt in middleware.rate_limits.keys():
                key = f"rate_limit:{lt}:{user_id}"
                await redis_client.delete(key)
            logger.info("All rate limits reset", user_id=user_id)
            
    except Exception as e:
        logger.error("Failed to reset rate limit", user_id=user_id, error=str(e))
        raise


# =============================================================================
# RATE LIMIT MONITORING
# =============================================================================

async def get_rate_limit_metrics() -> Dict[str, any]:
    """Get rate limiting metrics for monitoring."""
    
    try:
        metrics = {
            "total_keys": 0,
            "by_type": {},
            "top_users": []
        }
        
        # Scan for rate limit keys
        pattern = "rate_limit:*"
        keys = []
        
        cursor = 0
        while True:
            cursor, batch = await redis_client.scan(cursor, match=pattern, count=100)
            keys.extend(batch)
            if cursor == 0:
                break
        
        metrics["total_keys"] = len(keys)
        
        # Analyze by type
        type_counts = {}
        user_counts = {}
        
        for key in keys:
            parts = key.split(":")
            if len(parts) >= 3:
                limit_type = parts[1]
                user_id = parts[2]
                
                # Count by type
                if limit_type not in type_counts:
                    type_counts[limit_type] = 0
                type_counts[limit_type] += 1
                
                # Count by user
                count = await redis_client.zcard(key)
                if user_id not in user_counts:
                    user_counts[user_id] = 0
                user_counts[user_id] += count
        
        metrics["by_type"] = type_counts
        
        # Top users by request count
        sorted_users = sorted(user_counts.items(), key=lambda x: x[1], reverse=True)
        metrics["top_users"] = sorted_users[:10]
        
        return metrics
        
    except Exception as e:
        logger.error("Failed to get rate limit metrics", error=str(e))
        return {"error": str(e)}