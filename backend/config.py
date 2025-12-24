# HealthSync AI - Configuration Management

import os
from typing import List, Optional
from functools import lru_cache
from pathlib import Path
import logging
import structlog

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic.functional_validators import field_validator


# =============================================================================
# BASE SETTINGS
# =============================================================================

class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        validate_assignment=True,
        extra="ignore",
    )

    # =============================================================================
    # SERVER CONFIGURATION
    # =============================================================================
    environment: str = "development"
    debug: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    api_version: str = "v1"

    # =============================================================================
    # SECURITY
    # =============================================================================
    secret_key: str
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30
    jwt_refresh_token_expire_days: int = 7

    # =============================================================================
    # CORS
    # =============================================================================
    allowed_origins: List[str] = [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "http://localhost:19006",
        "http://localhost:8081",
        "exp://192.168.1.100:19000",
    ]

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [o.strip() for o in v.split(",")]
        return v

    # =============================================================================
    # DATABASE
    # =============================================================================
    mongodb_url: str
    mongodb_database: str = "healthsync"
    redis_url: Optional[str] = None

    # =============================================================================
    # AI / ML SERVICES
    # =============================================================================
    groq_api_key: str
    groq_model: str = "llama-3.1-70b-versatile"

    replicate_api_token: str
    huggingface_api_key: str
    openai_api_key: Optional[str] = None

    # =============================================================================
    # EXTERNAL SERVICES
    # =============================================================================
    resend_api_key: str
    from_email: str = "noreply@healthsync.ai"

    # =============================================================================
    # FILE UPLOAD LIMITS
    # =============================================================================
    max_file_size_mb: int = 10
    allowed_image_types: List[str] = ["image/jpeg", "image/png", "image/webp"]
    allowed_audio_types: List[str] = ["audio/wav", "audio/mp3", "audio/webm"]

    @field_validator(
        "allowed_image_types",
        "allowed_audio_types",
        mode="before",
    )
    @classmethod
    def parse_file_types(cls, v):
        if isinstance(v, str):
            return [t.strip() for t in v.split(",")]
        return v

    # =============================================================================
    # RATE LIMITING & PERFORMANCE
    # =============================================================================
    rate_limit_voice: int = 10
    rate_limit_image: int = 5
    rate_limit_ml: int = 20
    rate_limit_general: int = 100

    cache_ttl_predictions: int = 3600
    cache_ttl_doctors: int = 1800
    cache_ttl_health_data: int = 900

    # =============================================================================
    # MONITORING & LOGGING
    # =============================================================================
    log_level: str = "INFO"
    log_file: str = "logs/healthsync.log"
    log_rotation: str = "daily"

    health_check_interval: int = 30
    health_check_timeout: int = 10

    sentry_dsn: Optional[str] = None

    # =============================================================================
    # AI MODEL CONFIG
    # =============================================================================
    voice_sample_rate: int = 16000
    voice_chunk_duration: float = 1.0
    voice_max_duration: int = 300

    image_max_width: int = 2048
    image_max_height: int = 2048
    image_quality: int = 85

    ml_model_cache_size: int = 10
    ml_prediction_batch_size: int = 32

    # =============================================================================
    # FEATURE FLAGS
    # =============================================================================
    enable_voice_analysis: bool = True
    enable_ar_scanner: bool = True
    enable_therapy_game: bool = True
    enable_future_simulator: bool = True
    enable_doctor_marketplace: bool = True

    enable_groq_fallback: bool = True
    enable_local_whisper: bool = False
    enable_offline_mode: bool = False

    # =============================================================================
    # COMPUTED
    # =============================================================================
    @property
    def api_prefix(self) -> str:
        return f"/api/{self.api_version}"

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"

    @property
    def is_development(self) -> bool:
        return self.environment.lower() == "development"

    # =============================================================================
    # VALIDATION
    # =============================================================================
    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, v):
        if len(v) < 32:
            raise ValueError("Secret key must be at least 32 characters")
        return v

    @field_validator("jwt_access_token_expire_minutes")
    @classmethod
    def validate_token_expiry(cls, v):
        if not 5 <= v <= 1440:
            raise ValueError("Token expiry must be 5–1440 minutes")
        return v

    @field_validator("port")
    @classmethod
    def validate_port(cls, v):
        if not 1024 <= v <= 65535:
            raise ValueError("Port must be 1024–65535")
        return v


# =============================================================================
# ENV-SPECIFIC SETTINGS
# =============================================================================

class DevelopmentSettings(Settings):
    environment: str = "development"
    debug: bool = True
    log_level: str = "DEBUG"

    rate_limit_voice: int = 100
    rate_limit_image: int = 50
    rate_limit_ml: int = 200
    rate_limit_general: int = 1000


class ProductionSettings(Settings):
    environment: str = "production"
    debug: bool = False
    log_level: str = "INFO"

    allowed_origins: List[str] = [
        "https://healthsync.vercel.app",
        "https://admin.healthsync.ai",
    ]

    health_check_interval: int = 15

    @field_validator("secret_key")
    @classmethod
    def validate_production_secret(cls, v):
        if len(v) < 64:
            raise ValueError("Production secret key must be ≥ 64 characters")
        return v


class TestSettings(Settings):
    environment: str = "test"
    debug: bool = True

    mongodb_database: str = "healthsync_test"

    enable_voice_analysis: bool = False
    enable_ar_scanner: bool = False

    cache_ttl_predictions: int = 1
    cache_ttl_doctors: int = 1
    cache_ttl_health_data: int = 1


# =============================================================================
# SETTINGS LOADER
# =============================================================================

@lru_cache
def get_settings() -> Settings:
    env = os.getenv("ENVIRONMENT", "development").lower()

    if env == "production":
        return ProductionSettings()
    if env == "test":
        return TestSettings()
    return DevelopmentSettings()


settings = get_settings()


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

def configure_logging():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(settings.log_file),
        ],
    )


# =============================================================================
# HEALTH CHECK CONFIGURATION
# =============================================================================

class HealthCheckConfig:
    @staticmethod
    def get_checks():
        return [
            {"name": "database", "timeout": settings.health_check_timeout},
            {"name": "redis", "timeout": settings.health_check_timeout},
            {"name": "mongodb", "timeout": settings.health_check_timeout},
            {"name": "groq_api", "timeout": settings.health_check_timeout},
        ]


# =============================================================================
# API DOCUMENTATION CONFIGURATION
# =============================================================================

class APIDocsConfig:
    title = "HealthSync AI API"
    version = "1.0.0"
    contact = {
        "name": "HealthSync AI Team",
        "email": "api@healthsync.ai",
        "url": "https://healthsync.ai",
    }
