# HealthSync AI - Configuration Management
import os
from typing import List, Optional
from pydantic import validator
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # =============================================================================
    # SERVER CONFIGURATION
    # =============================================================================
    environment: str = "development"
    debug: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    api_version: str = "v1"
    
    # Security
    secret_key: str
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30
    jwt_refresh_token_expire_days: int = 7
    
    # CORS
    allowed_origins: List[str] = [
        "http://localhost:3000",
        "http://localhost:19006",
        "exp://192.168.1.100:19000"
    ]
    
    @validator('allowed_origins', pre=True)
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',')]
        return v
    
    # =============================================================================
    # DATABASE CONFIGURATION
    # =============================================================================
    
    # MongoDB Atlas (Primary Database)
    mongodb_url: str
    mongodb_database: str = "healthsync"
    
    # Redis Cache (Optional)
    redis_url: Optional[str] = None
    
    # =============================================================================
    # AI/ML SERVICES
    # =============================================================================
    
    # Groq API (Llama 3.1)
    groq_api_key: str
    groq_model: str = "llama-3.1-70b-versatile"
    
    # Replicate API
    replicate_api_token: str
    
    # Hugging Face
    huggingface_api_key: str
    
    # OpenAI (optional)
    openai_api_key: Optional[str] = None
    
    # =============================================================================
    # EXTERNAL SERVICES
    # =============================================================================
    
    # Email Service (Resend)
    resend_api_key: str
    from_email: str = "noreply@healthsync.ai"
    
    # File Upload Limits
    max_file_size_mb: int = 10
    allowed_image_types: List[str] = ["image/jpeg", "image/png", "image/webp"]
    allowed_audio_types: List[str] = ["audio/wav", "audio/mp3", "audio/webm"]
    
    @validator('allowed_image_types', 'allowed_audio_types', pre=True)
    def parse_file_types(cls, v):
        if isinstance(v, str):
            return [t.strip() for t in v.split(',')]
        return v
    
    # =============================================================================
    # RATE LIMITING & PERFORMANCE
    # =============================================================================
    
    # API Rate Limits (per minute)
    rate_limit_voice: int = 10
    rate_limit_image: int = 5
    rate_limit_ml: int = 20
    rate_limit_general: int = 100
    
    # Cache TTL (seconds)
    cache_ttl_predictions: int = 3600
    cache_ttl_doctors: int = 1800
    cache_ttl_health_data: int = 900
    
    # =============================================================================
    # MONITORING & LOGGING
    # =============================================================================
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/healthsync.log"
    log_rotation: str = "daily"
    
    # Health Check
    health_check_interval: int = 30
    health_check_timeout: int = 10
    
    # Sentry (Error Monitoring)
    sentry_dsn: Optional[str] = None
    
    # =============================================================================
    # AI MODEL CONFIGURATION
    # =============================================================================
    
    # Voice Analysis
    voice_sample_rate: int = 16000
    voice_chunk_duration: float = 1.0  # seconds
    voice_max_duration: int = 300  # 5 minutes
    
    # Image Processing
    image_max_width: int = 2048
    image_max_height: int = 2048
    image_quality: int = 85
    
    # ML Models
    ml_model_cache_size: int = 10
    ml_prediction_batch_size: int = 32
    
    # =============================================================================
    # FEATURE FLAGS
    # =============================================================================
    
    # Enable/disable features
    enable_voice_analysis: bool = True
    enable_ar_scanner: bool = True
    enable_therapy_game: bool = True
    enable_future_simulator: bool = True
    enable_doctor_marketplace: bool = True
    
    # AI Service Fallbacks
    enable_groq_fallback: bool = True
    enable_local_whisper: bool = False
    enable_offline_mode: bool = False
    
    # =============================================================================
    # COMPUTED PROPERTIES
    # =============================================================================
    
    @property
    def api_prefix(self) -> str:
        """API URL prefix."""
        return f"/api/{self.api_version}"
    
    @property
    def max_file_size_bytes(self) -> int:
        """Convert MB to bytes."""
        return self.max_file_size_mb * 1024 * 1024
    
    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment.lower() == "development"
    
    # =============================================================================
    # VALIDATION
    # =============================================================================
    
    @validator('secret_key')
    def validate_secret_key(cls, v):
        if len(v) < 32:
            raise ValueError('Secret key must be at least 32 characters long')
        return v
    
    @validator('jwt_access_token_expire_minutes')
    def validate_token_expiry(cls, v):
        if v < 5 or v > 1440:  # 5 minutes to 24 hours
            raise ValueError('Token expiry must be between 5 and 1440 minutes')
        return v
    
    @validator('port')
    def validate_port(cls, v):
        if v < 1024 or v > 65535:
            raise ValueError('Port must be between 1024 and 65535')
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        
        # Environment variable prefixes
        env_prefix = ""
        
        # Validation
        validate_assignment = True
        use_enum_values = True


class DevelopmentSettings(Settings):
    """Development-specific settings."""
    environment: str = "development"
    debug: bool = True
    log_level: str = "DEBUG"
    
    # Relaxed rate limits for development
    rate_limit_voice: int = 100
    rate_limit_image: int = 50
    rate_limit_ml: int = 200
    rate_limit_general: int = 1000


class ProductionSettings(Settings):
    """Production-specific settings."""
    environment: str = "production"
    debug: bool = False
    log_level: str = "INFO"
    
    # Strict security in production
    allowed_origins: List[str] = [
        "https://healthsync.vercel.app",
        "https://admin.healthsync.ai"
    ]
    
    # Enhanced monitoring
    health_check_interval: int = 15
    
    @validator('secret_key')
    def validate_production_secret(cls, v):
        if len(v) < 64:
            raise ValueError('Production secret key must be at least 64 characters')
        return v


class TestSettings(Settings):
    """Test-specific settings."""
    environment: str = "test"
    debug: bool = True
    
    # Test database
    mongodb_database: str = "healthsync_test"
    
    # Disable external services in tests
    enable_voice_analysis: bool = False
    enable_ar_scanner: bool = False
    
    # Fast cache expiry for tests
    cache_ttl_predictions: int = 1
    cache_ttl_doctors: int = 1
    cache_ttl_health_data: int = 1


@lru_cache()
def get_settings() -> Settings:
    """Get application settings based on environment."""
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    if env == "production":
        return ProductionSettings()
    elif env == "test":
        return TestSettings()
    else:
        return DevelopmentSettings()


# Global settings instance
settings = get_settings()


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

import logging
import structlog
from pathlib import Path

def configure_logging():
    """Configure structured logging."""
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format="%(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(settings.log_file)
        ]
    )


# =============================================================================
# HEALTH CHECK CONFIGURATION
# =============================================================================

class HealthCheckConfig:
    """Health check endpoints configuration."""
    
    @staticmethod
    def get_checks():
        """Get list of health checks to perform."""
        return [
            {
                "name": "database",
                "check": "check_database_connection",
                "timeout": settings.health_check_timeout
            },
            {
                "name": "redis",
                "check": "check_redis_connection", 
                "timeout": settings.health_check_timeout
            },
            {
                "name": "mongodb",
                "check": "check_mongodb_connection",
                "timeout": settings.health_check_timeout
            },
            {
                "name": "groq_api",
                "check": "check_groq_api",
                "timeout": settings.health_check_timeout
            }
        ]


# =============================================================================
# API DOCUMENTATION CONFIGURATION
# =============================================================================

class APIDocsConfig:
    """API documentation configuration."""
    
    title = "HealthSync AI API"
    description = """
    HealthSync AI is a comprehensive healthcare platform with AI-powered features:
    
    ## Features
    * **Voice AI Doctor** - Real-time voice analysis with stress detection
    * **Health Twin** - Personal disease prediction with ML models  
    * **AR Medical Scanner** - OCR and image analysis for prescriptions
    * **Pain-to-Game Therapy** - Gamified rehabilitation exercises
    * **Doctor Marketplace** - Specialist matching and bidding
    * **Future-You Simulator** - Age progression with health projections
    * **Family Health Graph** - Inherited disease risk analysis
    
    ## Authentication
    All endpoints require JWT authentication except public health checks.
    
    ## Rate Limiting
    API calls are rate limited per user and endpoint type.
    """
    
    version = "1.0.0"
    contact = {
        "name": "HealthSync AI Team",
        "email": "api@healthsync.ai",
        "url": "https://healthsync.ai"
    }
    
    license_info = {
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT"
    }
    
    tags_metadata = [
        {
            "name": "Authentication",
            "description": "User authentication and authorization"
        },
        {
            "name": "Health",
            "description": "Health metrics and predictions"
        },
        {
            "name": "Voice",
            "description": "Voice analysis and AI doctor"
        },
        {
            "name": "AR Scanner", 
            "description": "Augmented reality medical scanning"
        },
        {
            "name": "Therapy",
            "description": "Gamified therapy and rehabilitation"
        },
        {
            "name": "Doctors",
            "description": "Doctor marketplace and appointments"
        },
        {
            "name": "Future Simulator",
            "description": "Future health simulation and age progression"
        },
        {
            "name": "System",
            "description": "System health and monitoring"
        }
    ]