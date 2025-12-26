# HealthSync AI - FastAPI Main Application
import sys
import os

# Fix Windows console encoding for emoji support
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import structlog

from config import settings, configure_logging, APIDocsConfig
from api.middleware.auth import AuthMiddleware
from api.middleware.rate_limit import RateLimitMiddleware
from api.middleware.logging import LoggingMiddleware
from api.routes import (
    auth, health, voice, doctors, websocket, ar_scanner, therapy_game, future_simulator
)
from api.routes import crm
from services.db_service import DatabaseService
from services.ai_service import AIService


# Configure logging
configure_logging()
# Silence noisy watchfiles debug events during reload
logging.getLogger("watchfiles").setLevel(logging.WARNING)
logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    logger.info("Starting HealthSync AI Backend", version=APIDocsConfig.version)
    
    # Initialize services
    try:
        # Initialize database connections
        await DatabaseService.initialize()
        logger.info("Database connections initialized")
        
        # Initialize AI services
        await AIService.initialize()
        logger.info("AI services initialized")
        
        # Initialize ML models (optional)
        try:
            from services.ml_service import MLModelService
            await MLModelService.initialize_models()
            logger.info("ML models initialized")
        except Exception as e:
            logger.warning("ML models initialization failed", error=str(e))
        
        # Initialize MediaPipe for therapy game (optional)
        if settings.enable_therapy_game:
            try:
                from services.therapy_game_service import TherapyGameService
                await TherapyGameService.initialize_mediapipe()
                logger.info("MediaPipe models initialized")
            except Exception as e:
                logger.warning("MediaPipe initialization failed, therapy game disabled", error=str(e))
                # Disable therapy game feature if MediaPipe fails
                settings.enable_therapy_game = False
        
        # Start background tasks
        asyncio.create_task(background_health_monitor())
        logger.info("Background tasks started")
        
        logger.info("HealthSync AI Backend started successfully")
        
    except Exception as e:
        logger.error("Failed to initialize application", error=str(e))
        raise
    
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down HealthSync AI Backend")
    
    try:
        await DatabaseService.close()
        await AIService.cleanup()
        logger.info("Services cleaned up successfully")
        
    except Exception as e:
        logger.error("Error during shutdown", error=str(e))


# Create FastAPI application
app = FastAPI(
    title=APIDocsConfig.title,
    version=APIDocsConfig.version,
    contact=APIDocsConfig.contact,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    lifespan=lifespan
)

# DEV HELP: ensure CORS headers are always present so browser error responses aren't blocked by CORS
# This is safe for development and makes debugging 500s easier. Remove in production.
if settings.is_development:
    @app.middleware("http")
    async def add_cors_headers(request: Request, call_next):
        response = await call_next(request)
        origin = request.headers.get("origin")
        # If origin is in allowed list, use it; otherwise use first allowed origin
        if origin in settings.allowed_origins:
            response.headers["Access-Control-Allow-Origin"] = origin
        elif settings.allowed_origins:
            response.headers["Access-Control-Allow-Origin"] = settings.allowed_origins[0]
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, PATCH"
        return response


# Helper to add CORS headers to programmatically-created responses (e.g., exception handlers)
def _add_cors_headers(response: JSONResponse, origin: str = None):
    try:
        if settings.is_development:
            if origin in settings.allowed_origins:
                response.headers["Access-Control-Allow-Origin"] = origin
            elif settings.allowed_origins:
                response.headers["Access-Control-Allow-Origin"] = settings.allowed_origins[0]
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, PATCH"
    except Exception:
        # Be defensive; don't let CORS header injection mask the original error
        pass
    return response


# =============================================================================
# MIDDLEWARE CONFIGURATION
# =============================================================================

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,  # Use specific origins instead of wildcard with credentials
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
    expose_headers=["X-Request-ID", "X-Rate-Limit-Remaining"],
    max_age=3600,  # Cache preflight requests for 1 hour
)


# Trusted Host Middleware (Security)
if settings.is_production:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["healthsync.ai", "*.healthsync.ai", "api.healthsync.ai"]
    )

# Custom Middleware
app.add_middleware(LoggingMiddleware)
# app.add_middleware(RateLimitMiddleware)  # Temporarily disabled due to Redis issues
app.add_middleware(AuthMiddleware)


# =============================================================================
# EXCEPTION HANDLERS
# =============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with structured logging."""
    logger.warning(
        "HTTP exception occurred",
        status_code=exc.status_code,
        detail=exc.detail,
        path=request.url.path,
        method=request.method,
        client_ip=request.client.host if request.client else None
    )
    
    resp = JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "message": exc.detail,
            "status_code": exc.status_code,
            "timestamp": "2025-12-17T10:30:00Z",
            "path": request.url.path,
            "request_id": getattr(request.state, "request_id", None)
        }
    )
    _add_cors_headers(resp)
    return resp


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors."""
    logger.warning(
        "Validation error occurred",
        errors=[str(error) for error in exc.errors()],  # Convert to strings
        path=request.url.path,
        method=request.method
    )
    
    # Convert errors to JSON-serializable format
    serializable_errors = []
    for error in exc.errors():
        serializable_error = {
            "type": error.get("type"),
            "loc": error.get("loc"),
            "msg": error.get("msg"),
            "input": str(error.get("input")) if error.get("input") is not None else None
        }
        serializable_errors.append(serializable_error)
    
    resp = JSONResponse(
        status_code=422,
        content={
            "error": True,
            "message": "Validation error",
            "details": serializable_errors,
            "status_code": 422,
            "timestamp": "2025-12-17T10:30:00Z",
            "path": request.url.path,
            "request_id": getattr(request.state, "request_id", None)
        }
    )
    _add_cors_headers(resp)
    return resp


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error(
        "Unhandled exception occurred",
        error=str(exc),
        error_type=type(exc).__name__,
        path=request.url.path,
        method=request.method,
        exc_info=True
    )
    
    # Don't expose internal errors in production
    if settings.is_production:
        message = "Internal server error"
        details = None
    else:
        message = str(exc)
        details = {
            "type": type(exc).__name__,
            "traceback": str(exc)
        }
    
    resp = JSONResponse(
        status_code=500,
        content={
            "error": True,
            "message": message,
            "details": details,
            "status_code": 500,
            "timestamp": "2025-12-17T10:30:00Z",
            "path": request.url.path,
            "request_id": getattr(request.state, "request_id", None)
        }
    )
    _add_cors_headers(resp)
    return resp


# =============================================================================
# ROUTE REGISTRATION
# =============================================================================

# Health check (no auth required)
@app.get("/health", tags=["System"])
async def health_check():
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": "2025-12-17T10:30:00Z",
        "version": APIDocsConfig.version,
        "environment": settings.environment
    }


@app.get("/health/detailed", tags=["System"])
async def detailed_health_check():
    """Detailed health check with service status."""
    try:
        checks = await perform_health_checks()
        
        overall_status = "healthy" if all(
            check["status"] == "healthy" for check in checks.values()
        ) else "unhealthy"
        
        return {
            "status": overall_status,
            "timestamp": "2025-12-17T10:30:00Z",
            "version": APIDocsConfig.version,
            "environment": settings.environment,
            "services": checks
        }
        
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "timestamp": "2025-12-17T10:30:00Z",
                "error": "Health check failed"
            }
        )


# API Routes
app.include_router(
    auth.router,
    prefix=f"{settings.api_prefix}/auth",
    tags=["Authentication"]
)

# Legacy API Routes (without v1) for backward compatibility
app.include_router(
    auth.router,
    prefix="/api/auth",
    tags=["Authentication (Legacy)"]
)

app.include_router(
    health.router,
    prefix=f"{settings.api_prefix}/health",
    tags=["Health"]
)

# Legacy Health routes (without version)
app.include_router(
    health.router,
    prefix="/api/health",
    tags=["Health (Legacy)"]
)

app.include_router(
    voice.router,
    prefix=f"{settings.api_prefix}/voice",
    tags=["Voice"]
)

# Legacy API Routes (without version) for backward compatibility
app.include_router(
    voice.router,
    prefix="/api/voice",
    tags=["Voice (Legacy)"]
)

app.include_router(
    doctors.router,
    prefix=f"{settings.api_prefix}/doctors",
    tags=["Doctors"]
)

# Legacy Doctors routes (without version)
app.include_router(
    doctors.router,
    prefix="/api/doctors",
    tags=["Doctors (Legacy)"]
)

app.include_router(
    ar_scanner.router,
    prefix=f"{settings.api_prefix}/ar-scanner",
    tags=["AR Scanner"]
)

# Legacy AR Scanner routes (without version)
app.include_router(
    ar_scanner.router,
    prefix="/api/ar-scanner",
    tags=["AR Scanner (Legacy)"]
)

app.include_router(
    therapy_game.router,
    prefix=f"{settings.api_prefix}/therapy-game",
    tags=["Therapy Game"]
)

# Legacy Therapy Game routes (without version)
app.include_router(
    therapy_game.router,
    prefix="/api/therapy-game",
    tags=["Therapy Game (Legacy)"]
)

app.include_router(
    future_simulator.router,
    prefix=f"{settings.api_prefix}/future-simulator",
    tags=["Future Simulator"]
)

# CRM admin routes (users, appointments)
app.include_router(
    crm.router,
    prefix=f"{settings.api_prefix}/crm",
    tags=["CRM"]
)

# Legacy CRM routes
app.include_router(
    crm.router,
    prefix="/api/crm",
    tags=["CRM (Legacy)"]
)

# Legacy Future Simulator routes (without version)
app.include_router(
    future_simulator.router,
    prefix="/api/future-simulator",
    tags=["Future Simulator (Legacy)"]
)

# WebSocket routes
app.include_router(
    websocket.router,
    prefix="/ws"
)


# =============================================================================
# BACKGROUND TASKS
# =============================================================================

async def background_health_monitor():
    """Background task to monitor system health."""
    while True:
        try:
            await asyncio.sleep(settings.health_check_interval)
            
            # Perform health checks
            checks = await perform_health_checks()
            
            # Log unhealthy services
            for service, status in checks.items():
                if status["status"] != "healthy":
                    logger.warning(
                        "Service unhealthy",
                        service=service,
                        status=status["status"],
                        error=status.get("error")
                    )
            
            # Update metrics (if monitoring service is available)
            # await update_health_metrics(checks)
            
        except Exception as e:
            logger.error("Health monitor error", error=str(e))


async def perform_health_checks() -> Dict[str, Dict[str, Any]]:
    """Perform health checks on all services."""
    checks = {}
    
    # Database health check
    try:
        await DatabaseService.health_check()
        checks["database"] = {"status": "healthy", "response_time_ms": 50}
    except Exception as e:
        checks["database"] = {"status": "unhealthy", "error": str(e)}
    
    # Redis health check (only when configured)
    if settings.redis_url:
        try:
            # Use DatabaseService redis client if available
            client = None
            try:
                client = DatabaseService.get_redis_client()
            except Exception:
                client = None

            if client:
                pong = await client.ping()
                checks["redis"] = {"status": "healthy" if pong else "unhealthy", "response_time_ms": 10}
            else:
                checks["redis"] = {"status": "unhealthy", "error": "Redis client not initialized"}
        except Exception as e:
            checks["redis"] = {"status": "unhealthy", "error": str(e)}
    else:
        checks["redis"] = {"status": "not_configured"}
    
    # AI services health check
    try:
        await AIService.health_check()
        checks["ai_services"] = {"status": "healthy", "response_time_ms": 200}
    except Exception as e:
        checks["ai_services"] = {"status": "unhealthy", "error": str(e)}
    
    return checks


# =============================================================================
# STARTUP EVENTS
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Additional startup tasks."""
    logger.info("Performing startup tasks")
    
    # Warm up AI models
    if settings.enable_voice_analysis:
        try:
            await AIService.warm_up_models()
            logger.info("AI models warmed up")
        except Exception as e:
            logger.warning("Failed to warm up AI models", error=str(e))
    
    # Initialize caches
    try:
        from services.db_service import redis_client
        await redis_client.set("startup_time", "2025-12-17T10:30:00Z")
        logger.info("Cache initialized")
    except Exception as e:
        logger.warning("Failed to initialize cache", error=str(e))


# =============================================================================
# DEVELOPMENT HELPERS
# =============================================================================

if settings.is_development:
    
    @app.get("/dev/info", tags=["Development"])
    async def dev_info():
        """Development information endpoint."""
        return {
            "environment": settings.environment,
            "debug": settings.debug,
            "api_prefix": settings.api_prefix,
            "database_url": settings.mongodb_url.split("@")[-1],  # Hide credentials
            "features": {
                "voice_analysis": settings.enable_voice_analysis,
                "ar_scanner": settings.enable_ar_scanner,
                "therapy_game": settings.enable_therapy_game,
                "future_simulator": settings.enable_future_simulator,
                "doctor_marketplace": settings.enable_doctor_marketplace
            },
            "rate_limits": {
                "voice": settings.rate_limit_voice,
                "image": settings.rate_limit_image,
                "ml": settings.rate_limit_ml,
                "general": settings.rate_limit_general
            }
        }
    
    @app.post("/dev/reset-cache", tags=["Development"])
    async def reset_cache():
        """Reset all caches (development only)."""
        try:
            from services.db_service import redis_client
            await redis_client.flushdb()
            return {"message": "Cache reset successfully"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to reset cache: {str(e)}")


# =============================================================================
# APPLICATION METADATA
# =============================================================================

@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint with API information."""
    return {
        "name": "HealthSync AI API",
        "version": APIDocsConfig.version,
        "description": "AI-powered healthcare platform API",
        "docs_url": "/docs" if settings.debug else None,
        "health_check": "/health",
        "api_prefix": settings.api_prefix,
        "timestamp": "2025-12-17T10:30:00Z"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        access_log=True
    )