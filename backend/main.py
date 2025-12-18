# HealthSync AI - FastAPI Main Application
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

from config_flexible import settings, configure_logging, APIDocsConfig
from api.middleware.auth import AuthMiddleware
from api.middleware.rate_limit import RateLimitMiddleware
from api.middleware.logging import LoggingMiddleware
from api.routes import (
    auth, health, voice, doctors, websocket, ar_scanner, therapy_game, future_simulator
)
from services.db_service import DatabaseService
from services.ai_service import AIService


# Configure logging
configure_logging()
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
        
        # Initialize ML models
        from services.ml_service import MLModelService
        await MLModelService.initialize_models()
        logger.info("ML models initialized")
        
        # Initialize MediaPipe for therapy game
        if settings.enable_therapy_game:
            from services.therapy_game_service import TherapyGameService
            await TherapyGameService.initialize_mediapipe()
            logger.info("MediaPipe models initialized")
        
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
    description=APIDocsConfig.description,
    version=APIDocsConfig.version,
    contact=APIDocsConfig.contact,
    license_info=APIDocsConfig.license_info,
    openapi_tags=APIDocsConfig.tags_metadata,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    lifespan=lifespan
)


# =============================================================================
# MIDDLEWARE CONFIGURATION
# =============================================================================

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Rate-Limit-Remaining"]
)

# Trusted Host Middleware (Security)
if settings.is_production:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["healthsync.ai", "*.healthsync.ai", "api.healthsync.ai"]
    )

# Custom Middleware
app.add_middleware(LoggingMiddleware)
app.add_middleware(RateLimitMiddleware)
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
    
    return JSONResponse(
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


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors."""
    logger.warning(
        "Validation error occurred",
        errors=exc.errors(),
        path=request.url.path,
        method=request.method
    )
    
    return JSONResponse(
        status_code=422,
        content={
            "error": True,
            "message": "Validation error",
            "details": exc.errors(),
            "status_code": 422,
            "timestamp": "2025-12-17T10:30:00Z",
            "path": request.url.path,
            "request_id": getattr(request.state, "request_id", None)
        }
    )


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
    
    return JSONResponse(
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

app.include_router(
    health.router,
    prefix=f"{settings.api_prefix}/health",
    tags=["Health"]
)

app.include_router(
    voice.router,
    prefix=f"{settings.api_prefix}/voice",
    tags=["Voice"]
)

app.include_router(
    doctors.router,
    prefix=f"{settings.api_prefix}/doctors",
    tags=["Doctors"]
)

app.include_router(
    ar_scanner.router,
    prefix=f"{settings.api_prefix}/ar-scanner",
    tags=["AR Scanner"]
)

app.include_router(
    therapy_game.router,
    prefix=f"{settings.api_prefix}/therapy-game",
    tags=["Therapy Game"]
)

app.include_router(
    future_simulator.router,
    prefix=f"{settings.api_prefix}/future-simulator",
    tags=["Future Simulator"]
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
    
    # Redis health check
    try:
        from services.db_service import redis_client
        await redis_client.ping()
        checks["redis"] = {"status": "healthy", "response_time_ms": 10}
    except Exception as e:
        checks["redis"] = {"status": "unhealthy", "error": str(e)}
    
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
            "database_url": settings.database_url.split("@")[-1],  # Hide credentials
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