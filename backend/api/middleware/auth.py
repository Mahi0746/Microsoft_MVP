# HealthSync AI - Authentication Middleware
import uuid
from typing import Optional, List
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
import structlog
from jose import JWTError, jwt
from datetime import datetime, timedelta

from config import settings
from services.db_service import DatabaseService


logger = structlog.get_logger(__name__)
security = HTTPBearer(auto_error=False)


class AuthMiddleware(BaseHTTPMiddleware):
    """Authentication middleware for JWT token validation."""
    
    # Public endpoints that don't require authentication
    PUBLIC_PATHS = {
        "/",
        "/health",
        "/health/detailed",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/api/v1/auth/signup",
        "/api/v1/auth/login",
        "/api/v1/auth/refresh",
        # Legacy paths (without v1) for backward compatibility
        "/api/auth/signup",
        "/api/auth/login",
        "/api/auth/refresh",
    }
    
    # Paths that start with these prefixes are public
    PUBLIC_PREFIXES = [
        "/static/",
        "/favicon.ico",
        "/dev/" if settings.is_development else None,
        # Allow voice endpoints in both versioned and legacy prefixes to be public (e.g., /api/voice/, /api/v1/voice/)
        f"{settings.api_prefix}/voice/",
        "/api/voice/",
        # Allow therapy game endpoints (versioned + legacy)
        f"{settings.api_prefix}/therapy-game/",
        "/api/therapy-game/",
        # Allow AR scanner endpoints (versioned + legacy)
        f"{settings.api_prefix}/ar-scanner/",
        "/api/ar-scanner/",
        # Allow doctors marketplace endpoints (versioned + legacy)
        f"{settings.api_prefix}/doctors/",
        "/api/doctors/",
        # Allow future simulator endpoints (versioned + legacy)
        f"{settings.api_prefix}/future-simulator/",
        "/api/future-simulator/"
    ]
    
    def __init__(self, app):
        super().__init__(app)
        self.public_prefixes = [p for p in self.PUBLIC_PREFIXES if p is not None]
        # Log the public paths for debugging
        logger.info("Auth middleware initialized", 
                   public_paths=list(self.PUBLIC_PATHS),
                   api_prefix=settings.api_prefix)
    
    async def dispatch(self, request: Request, call_next):
        """Process request through authentication middleware."""
        
        # Add request ID for tracing
        request.state.request_id = str(uuid.uuid4())
        
        # Always allow OPTIONS requests (CORS preflight)
        if request.method == "OPTIONS":
            return await call_next(request)
        
        # Check if path is public. For public paths, allow unauthenticated access,
        # but if an Authorization header is present attempt to validate the token
        # so handlers that optionally use `request.state.user` can work.
        if self._is_public_path(request.url.path):
            authorization = request.headers.get("Authorization")
            if not authorization:
                return await call_next(request)
            # If Authorization header exists, fall through and validate token
        
        # Extract and validate JWT token
        try:
            token = await self._extract_token(request)
            if not token:
                logger.warning("Authentication required: no token provided", path=request.url.path, request_id=request.state.request_id)
                resp = JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED, content={
                    "error": True,
                    "message": "Authentication required",
                    "status_code": status.HTTP_401_UNAUTHORIZED,
                    "path": request.url.path,
                    "request_id": request.state.request_id
                })
                if settings.is_development:
                    resp.headers["Access-Control-Allow-Origin"] = "*"
                    resp.headers["Access-Control-Allow-Credentials"] = "true"
                    resp.headers["Access-Control-Allow-Headers"] = "*, Authorization, Content-Type"
                    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, PATCH"
                return resp
            
            # Validate token and get user info
            user_info = await self._validate_token(token)
            request.state.user = user_info
            
            logger.debug(
                "Request authenticated",
                user_id=user_info["user_id"],
                role=user_info["role"],
                path=request.url.path,
                request_id=request.state.request_id
            )
            
        except HTTPException as he:
            # Return a JSON response instead of re-raising to avoid it bubbling out of middleware
            logger.warning("Authentication failed", error=str(he.detail), path=request.url.path, request_id=request.state.request_id)
            resp = JSONResponse(status_code=he.status_code, content={
                "error": True,
                "message": he.detail,
                "status_code": he.status_code,
                "path": request.url.path,
                "request_id": request.state.request_id
            })
            if settings.is_development:
                resp.headers["Access-Control-Allow-Origin"] = "*"
                resp.headers["Access-Control-Allow-Credentials"] = "true"
                resp.headers["Access-Control-Allow-Headers"] = "*, Authorization, Content-Type"
                resp.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, PATCH"
            return resp
        except Exception as e:
            logger.error(
                "Authentication error",
                error=str(e),
                path=request.url.path,
                request_id=request.state.request_id
            )
            resp = JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED, content={
                "error": True,
                "message": "Invalid authentication credentials",
                "status_code": status.HTTP_401_UNAUTHORIZED,
                "path": request.url.path,
                "request_id": request.state.request_id
            })
            if settings.is_development:
                resp.headers["Access-Control-Allow-Origin"] = "*"
                resp.headers["Access-Control-Allow-Credentials"] = "true"
                resp.headers["Access-Control-Allow-Headers"] = "*, Authorization, Content-Type"
                resp.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, PATCH"
            return resp
        
        return await call_next(request)
    
    def _is_public_path(self, path: str) -> bool:
        """Check if the path is public (doesn't require authentication)."""
        
        # Log the path being checked for debugging
        logger.debug("Checking path", path=path, public_paths=list(self.PUBLIC_PATHS))
        
        # Exact path matches
        if path in self.PUBLIC_PATHS:
            logger.debug("Path is public (exact match)", path=path)
            return True
        
        # Prefix matches
        for prefix in self.public_prefixes:
            if path.startswith(prefix):
                logger.debug("Path is public (prefix match)", path=path, prefix=prefix)
                return True
        
        logger.debug("Path is not public", path=path)
        return False
    
    async def _extract_token(self, request: Request) -> Optional[str]:
        """Extract JWT token from request headers."""
        
        # Try Authorization header first
        authorization = request.headers.get("Authorization")
        if authorization and authorization.startswith("Bearer "):
            # Log only a short prefix for debugging (do not log full tokens)
            try:
                logger.debug("Authorization header received", prefix=authorization[:30])
            except Exception:
                pass
            return authorization.split(" ")[1]
        
        # Try cookie as fallback
        token = request.cookies.get("access_token")
        if token:
            try:
                logger.debug("Token found in cookies", prefix=str(token)[:10])
            except Exception:
                pass
        if token:
            return token
        
        return None
    
    async def _validate_token(self, token: str) -> dict:
        """Validate JWT token and return user information."""
        
        try:
            # Decode JWT token
            payload = jwt.decode(
                token,
                settings.secret_key,
                algorithms=[settings.jwt_algorithm]
            )
            
            # Extract user information
            user_id = payload.get("sub")
            if not user_id:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token: missing user ID"
                )
            
            # Check token expiration
            exp = payload.get("exp")
            if exp and datetime.utcnow().timestamp() > exp:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has expired"
                )
            
            # Get user details from database
            user_details = await self._get_user_details(user_id)
            if not user_details:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found"
                )
            
            # Check if user is active
            if not user_details.get("is_active", True):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User account is disabled"
                )
            
            return {
                "user_id": user_id,
                "email": user_details["email"],
                "role": user_details["role"],
                "token_type": payload.get("type", "access"),
                "issued_at": payload.get("iat"),
                "expires_at": payload.get("exp")
            }
            
        except JWTError as e:
            logger.warning("JWT validation failed", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    
    async def _get_user_details(self, user_id: str) -> Optional[dict]:
        """Get user details from database."""
        
        try:
            # Get user from MongoDB
            user = await DatabaseService.mongodb_find_one(
                "users",
                {"user_id": user_id}
            )
            
            if user:
                return {
                    "id": str(user.get("user_id") or user.get("_id")),
                    "email": user["email"],
                    "role": user["role"],
                    "is_active": user.get("is_active", True),
                    "created_at": user.get("created_at"),
                    "updated_at": user.get("updated_at")
                }
            
            return None
            
        except Exception as e:
            logger.error("Failed to get user details", user_id=user_id, error=str(e))
            return None


# =============================================================================
# AUTHENTICATION UTILITIES
# =============================================================================

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token."""
    
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.jwt_access_token_expire_minutes)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access"
    })
    
    encoded_jwt = jwt.encode(
        to_encode,
        settings.secret_key,
        algorithm=settings.jwt_algorithm
    )
    
    return encoded_jwt


def create_refresh_token(data: dict) -> str:
    """Create JWT refresh token."""
    
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=settings.jwt_refresh_token_expire_days)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "refresh"
    })
    
    encoded_jwt = jwt.encode(
        to_encode,
        settings.secret_key,
        algorithm=settings.jwt_algorithm
    )
    
    return encoded_jwt


def verify_token(token: str) -> dict:
    """Verify and decode JWT token."""
    
    try:
        payload = jwt.decode(
            token,
            settings.secret_key,
            algorithms=[settings.jwt_algorithm]
        )
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )


# =============================================================================
# AUTHORIZATION DECORATORS
# =============================================================================

from functools import wraps
from fastapi import Depends

def get_current_user(request: Request) -> dict:
    """Dependency to get current authenticated user."""
    
    if not hasattr(request.state, "user"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    return request.state.user


def require_role(allowed_roles: List[str]):
    """Decorator to require specific user roles."""
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get request from args or kwargs (FastAPI may inject as kwarg)
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            if not request:
                # check kwargs
                for v in kwargs.values():
                    if isinstance(v, Request):
                        request = v
                        break

            if not request:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Request object not found"
                )
            
            user = get_current_user(request)
            
            if user["role"] not in allowed_roles:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Access denied. Required roles: {', '.join(allowed_roles)}"
                )
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


def require_patient(func):
    """Decorator to require patient role."""
    return require_role(["patient"])(func)


def require_doctor(func):
    """Decorator to require doctor role."""
    return require_role(["doctor", "admin"])(func)


def require_admin(func):
    """Decorator to require admin role."""
    return require_role(["admin"])(func)


# =============================================================================
# PERMISSION UTILITIES
# =============================================================================

async def can_access_user_data(current_user: dict, target_user_id: str) -> bool:
    """Check if current user can access target user's data."""
    
    # Users can access their own data
    if current_user["user_id"] == target_user_id:
        return True
    
    # Admins can access all data
    if current_user["role"] == "admin":
        return True
    
    # Doctors can access their patients' data
    if current_user["role"] == "doctor":
        return await _doctor_has_patient_access(current_user["user_id"], target_user_id)
    
    return False


async def _doctor_has_patient_access(doctor_user_id: str, patient_user_id: str) -> bool:
    """Check if doctor has access to patient's data through appointments."""
    
    try:
        # Get appointments from MongoDB
        appointments = await DatabaseService.mongodb_find_many(
            "appointments",
            {
                "doctor_user_id": doctor_user_id,
                "patient_user_id": patient_user_id,
                "status": {"$in": ["confirmed", "completed"]}
            },
            limit=1
        )
        
        return len(appointments) > 0
        
    except Exception as e:
        logger.error(
            "Failed to check doctor-patient access",
            doctor_id=doctor_user_id,
            patient_id=patient_user_id,
            error=str(e)
        )
        return False