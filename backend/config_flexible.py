# HealthSync AI - Flexible Configuration (Handles Missing API Keys)

from typing import List, Optional
from functools import lru_cache
from pathlib import Path
import logging

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic.functional_validators import field_validator

# -----------------------------------------------------------------------------
# Base directory (for .env resolution)
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent


class FlexibleSettings(BaseSettings):
    """Application settings that work with or without API keys."""

    # =============================================================================
    # Pydantic v2 Configuration
    # =============================================================================
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_ignore_empty=True,
        extra="ignore",
        case_sensitive=False,
        validate_assignment=True,
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
    secret_key: str = (
        "healthsync_dev_secret_key_change_in_production_12345678901234567890"
    )
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30
    jwt_refresh_token_expire_days: int = 7

    # =============================================================================
    # CORS
    # =============================================================================
    allowed_origins: List[str] = [
        "http://localhost:3000",
        "http://localhost:19006",
        "exp://192.168.1.100:19000",
    ]

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    # =============================================================================
    # DATABASE CONFIGURATION
    # =============================================================================
    mongodb_url: str = "mongodb://localhost:27017/healthsync_dev"
    mongodb_database: str = "healthsync"
    redis_url: str = "redis://localhost:6379/0"

    # =============================================================================
    # AI / ML SERVICES (WITH DEMO DEFAULTS)
    # =============================================================================
    groq_api_key: str = "demo_groq_key_get_real_key_from_console_groq_com"
    groq_model: str = "llama-3.1-70b-versatile"

    replicate_api_token: str = (
        "demo_replicate_token_get_real_token_from_replicate_com"
    )

    huggingface_api_key: str = (
        "demo_hf_key_get_real_key_from_huggingface_co"
    )

    # OpenAI (optional – NOT required for startup)
    openai_api_key: Optional[str] = None

    # =============================================================================
    # EXTERNAL SERVICES
    # =============================================================================
    resend_api_key: str = "demo_resend_key_get_real_key_from_resend_com"
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
    # HELPER METHODS (SAFE CHECKS)
    # =============================================================================
    def has_real_groq_key(self) -> bool:
        return self.groq_api_key.startswith("gsk_")

    def has_real_replicate_token(self) -> bool:
        return self.replicate_api_token.startswith("r8_")

    def has_real_huggingface_key(self) -> bool:
        return self.huggingface_api_key.startswith("hf_")

    def has_mongodb_atlas(self) -> bool:
        return "mongodb+srv://" in self.mongodb_url

    def get_ai_status(self) -> dict:
        return {
            "groq": self.has_real_groq_key(),
            "replicate": self.has_real_replicate_token(),
            "huggingface": self.has_real_huggingface_key(),
            "mongodb_atlas": self.has_mongodb_atlas(),
        }

    def is_demo_mode(self) -> bool:
        return not any(self.get_ai_status().values())

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
    # FEATURE FLAGS
    # =============================================================================
    enable_voice_analysis: bool = True
    enable_ar_scanner: bool = True
    enable_therapy_game: bool = True
    enable_future_simulator: bool = True
    enable_doctor_marketplace: bool = True

    ar_scanner_cache_ttl: int = 1800
    ar_scanner_max_image_size: int = 10
    ar_scanner_performance_monitoring: bool = True
    ar_scanner_fallback_enabled: bool = True
    ar_scanner_dynamic_analysis: bool = True
    ar_scanner_ensemble_models: bool = True
    ar_scanner_real_ocr: bool = True

    enable_groq_fallback: bool = True
    enable_local_whisper: bool = False
    enable_offline_mode: bool = False

    # =============================================================================
    # COMPUTED PROPERTIES
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
    # RELAXED VALIDATION (DEV SAFE)
    # =============================================================================
    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, v):
        if len(v) < 16:
            return (
                "healthsync_dev_secret_key_change_in_production_12345678901234567890"
            )
        return v

    @field_validator("jwt_access_token_expire_minutes")
    @classmethod
    def validate_token_expiry(cls, v):
        if v < 5 or v > 1440:
            return 30
        return v

    @field_validator("port")
    @classmethod
    def validate_port(cls, v):
        if v < 1024 or v > 65535:
            return 8000
        return v


# -----------------------------------------------------------------------------
# Cached settings instance
# -----------------------------------------------------------------------------
@lru_cache
def get_flexible_settings() -> FlexibleSettings:
    return FlexibleSettings()


settings = get_flexible_settings()

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
def configure_logging():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "healthsync.log"),
        ],
    )


# =============================================================================
# API DOCUMENTATION CONFIGURATION
# =============================================================================
class APIDocsConfig:
    title = "HealthSync AI API"
    version = "1.0.0"

    description = """
    HealthSync AI is a comprehensive healthcare platform with AI-powered features.

    ## Demo Mode
    The API runs without external API keys.
    Enable real services by providing keys via environment variables.
    """

    contact = {
        "name": "HealthSync AI Team",
        "email": "support@healthsync.ai",
    }

    tags_metadata = [
        {"name": "Authentication", "description": "User authentication"},
        {"name": "Health", "description": "Health metrics and predictions"},
        {"name": "Voice", "description": "Voice AI doctor consultations"},
        {"name": "AR Scanner", "description": "Medical document scanning"},
        {"name": "Therapy", "description": "Gamified therapy sessions"},
        {"name": "Doctors", "description": "Doctor marketplace"},
        {"name": "Future Simulator", "description": "Future health simulation"},
        {"name": "System", "description": "System health and status"},
    ]
