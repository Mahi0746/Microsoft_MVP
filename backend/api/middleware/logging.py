# HealthSync AI - Logging Middleware
import time
import uuid
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import structlog

from config import settings


logger = structlog.get_logger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Structured logging middleware for request/response tracking."""
    
    def __init__(self, app):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Log request and response details with structured data."""
        
        # Generate request ID if not already set
        if not hasattr(request.state, "request_id"):
            request.state.request_id = str(uuid.uuid4())
        
        # Start timing
        start_time = time.time()
        
        # Extract request details
        request_details = {
            "request_id": request.state.request_id,
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "client_ip": self._get_client_ip(request),
            "user_agent": request.headers.get("user-agent", ""),
            "content_type": request.headers.get("content-type", ""),
            "content_length": request.headers.get("content-length", 0)
        }
        
        # Add user info if authenticated
        if hasattr(request.state, "user") and request.state.user:
            request_details.update({
                "user_id": request.state.user["user_id"],
                "user_role": request.state.user["role"]
            })
        
        # Log incoming request
        logger.info(
            "Request started",
            **request_details
        )
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Extract response details
            response_details = {
                "request_id": request.state.request_id,
                "status_code": response.status_code,
                "process_time_ms": round(process_time * 1000, 2),
                "response_size": len(response.body) if hasattr(response, 'body') else 0
            }
            
            # Add custom headers
            response.headers["X-Request-ID"] = request.state.request_id
            response.headers["X-Process-Time"] = str(response_details["process_time_ms"])
            
            # Log response based on status code
            if response.status_code >= 500:
                logger.error(
                    "Request completed with server error",
                    **{**request_details, **response_details}
                )
            elif response.status_code >= 400:
                logger.warning(
                    "Request completed with client error",
                    **{**request_details, **response_details}
                )
            else:
                logger.info(
                    "Request completed successfully",
                    **{**request_details, **response_details}
                )
            
            # Log slow requests
            if process_time > 5.0:  # 5 seconds threshold
                logger.warning(
                    "Slow request detected",
                    **{**request_details, **response_details},
                    threshold_seconds=5.0
                )
            
            return response
            
        except Exception as e:
            # Calculate processing time for failed requests
            process_time = time.time() - start_time
            
            # Log exception
            logger.error(
                "Request failed with exception",
                **request_details,
                process_time_ms=round(process_time * 1000, 2),
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True
            )
            
            raise
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        
        # Check for forwarded headers (behind proxy/load balancer)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Take the first IP in the chain
            return forwarded_for.split(",")[0].strip()
        
        # Check for real IP header
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fall back to direct client IP
        if request.client:
            return request.client.host
        
        return "unknown"


# =============================================================================
# STRUCTURED LOGGING UTILITIES
# =============================================================================

def log_api_usage(
    user_id: str,
    endpoint: str,
    method: str,
    status_code: int,
    response_time_ms: float,
    ai_service_used: str = None,
    tokens_used: int = None,
    cost_estimate: float = None
):
    """Log API usage for analytics and billing."""
    
    usage_data = {
        "event_type": "api_usage",
        "user_id": user_id,
        "endpoint": endpoint,
        "method": method,
        "status_code": status_code,
        "response_time_ms": response_time_ms,
        "timestamp": time.time()
    }
    
    if ai_service_used:
        usage_data["ai_service_used"] = ai_service_used
    
    if tokens_used:
        usage_data["tokens_used"] = tokens_used
    
    if cost_estimate:
        usage_data["cost_estimate"] = cost_estimate
    
    logger.info("API usage recorded", **usage_data)


def log_ai_service_call(
    service_name: str,
    operation: str,
    user_id: str,
    input_size: int,
    output_size: int,
    processing_time_ms: float,
    success: bool,
    error: str = None,
    cost: float = None
):
    """Log AI service API calls for monitoring and cost tracking."""
    
    call_data = {
        "event_type": "ai_service_call",
        "service_name": service_name,
        "operation": operation,
        "user_id": user_id,
        "input_size": input_size,
        "output_size": output_size,
        "processing_time_ms": processing_time_ms,
        "success": success,
        "timestamp": time.time()
    }
    
    if error:
        call_data["error"] = error
    
    if cost:
        call_data["cost"] = cost
    
    if success:
        logger.info("AI service call completed", **call_data)
    else:
        logger.error("AI service call failed", **call_data)


def log_health_event(
    user_id: str,
    event_type: str,
    event_data: dict,
    risk_level: str = None,
    ai_confidence: float = None
):
    """Log health-related events for medical tracking."""
    
    health_data = {
        "event_type": "health_event",
        "user_id": user_id,
        "health_event_type": event_type,
        "event_data": event_data,
        "timestamp": time.time()
    }
    
    if risk_level:
        health_data["risk_level"] = risk_level
    
    if ai_confidence:
        health_data["ai_confidence"] = ai_confidence
    
    logger.info("Health event recorded", **health_data)


def log_security_event(
    event_type: str,
    user_id: str = None,
    ip_address: str = None,
    details: dict = None,
    severity: str = "info"
):
    """Log security-related events for monitoring."""
    
    security_data = {
        "event_type": "security_event",
        "security_event_type": event_type,
        "timestamp": time.time(),
        "severity": severity
    }
    
    if user_id:
        security_data["user_id"] = user_id
    
    if ip_address:
        security_data["ip_address"] = ip_address
    
    if details:
        security_data["details"] = details
    
    if severity == "critical":
        logger.critical("Security event", **security_data)
    elif severity == "warning":
        logger.warning("Security event", **security_data)
    else:
        logger.info("Security event", **security_data)


# =============================================================================
# PERFORMANCE MONITORING
# =============================================================================

class PerformanceMonitor:
    """Monitor and log performance metrics."""
    
    @staticmethod
    def log_database_query(
        query_type: str,
        table_name: str,
        execution_time_ms: float,
        rows_affected: int = None,
        user_id: str = None
    ):
        """Log database query performance."""
        
        perf_data = {
            "event_type": "database_query",
            "query_type": query_type,
            "table_name": table_name,
            "execution_time_ms": execution_time_ms,
            "timestamp": time.time()
        }
        
        if rows_affected is not None:
            perf_data["rows_affected"] = rows_affected
        
        if user_id:
            perf_data["user_id"] = user_id
        
        # Log slow queries as warnings
        if execution_time_ms > 1000:  # 1 second threshold
            logger.warning("Slow database query", **perf_data)
        else:
            logger.debug("Database query completed", **perf_data)
    
    @staticmethod
    def log_cache_operation(
        operation: str,
        key: str,
        hit: bool = None,
        execution_time_ms: float = None
    ):
        """Log cache operations for monitoring."""
        
        cache_data = {
            "event_type": "cache_operation",
            "operation": operation,
            "key": key,
            "timestamp": time.time()
        }
        
        if hit is not None:
            cache_data["hit"] = hit
        
        if execution_time_ms is not None:
            cache_data["execution_time_ms"] = execution_time_ms
        
        logger.debug("Cache operation", **cache_data)
    
    @staticmethod
    def log_ml_inference(
        model_name: str,
        input_features: int,
        prediction_time_ms: float,
        confidence_score: float = None,
        user_id: str = None
    ):
        """Log ML model inference performance."""
        
        ml_data = {
            "event_type": "ml_inference",
            "model_name": model_name,
            "input_features": input_features,
            "prediction_time_ms": prediction_time_ms,
            "timestamp": time.time()
        }
        
        if confidence_score is not None:
            ml_data["confidence_score"] = confidence_score
        
        if user_id:
            ml_data["user_id"] = user_id
        
        logger.info("ML inference completed", **ml_data)


# =============================================================================
# LOG AGGREGATION UTILITIES
# =============================================================================

async def get_request_metrics(time_window_hours: int = 24) -> dict:
    """Get aggregated request metrics for monitoring dashboard."""
    
    # This would typically query a log aggregation service
    # For now, return mock data structure
    
    return {
        "time_window_hours": time_window_hours,
        "total_requests": 0,
        "requests_by_status": {
            "2xx": 0,
            "4xx": 0,
            "5xx": 0
        },
        "average_response_time_ms": 0,
        "slowest_endpoints": [],
        "most_active_users": [],
        "error_rate_percentage": 0
    }


async def get_ai_service_metrics(time_window_hours: int = 24) -> dict:
    """Get AI service usage metrics."""
    
    return {
        "time_window_hours": time_window_hours,
        "total_calls": 0,
        "calls_by_service": {},
        "total_cost": 0,
        "cost_by_service": {},
        "average_response_time_ms": 0,
        "success_rate_percentage": 0,
        "quota_usage": {}
    }