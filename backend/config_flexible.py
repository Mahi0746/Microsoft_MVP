# HealthSync AI - Flexible Configuration (Handles Missing API Keys)
import os
from typing import List, Optional
from pydantic import validator, Field
from pydantic_settings import BaseSettings
from functools import lru_cache


class FlexibleSettings(BaseSettings):
    """Application settings that work with or without API keys."""
    
    # =============================================================================
    # SERVER CONFIGURATION
    # =============================================================================
    environment: str = "development"
    debug: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    api_version: str = "v1"
    
    # Security - with defaults
    secret_key: str = "healthsync_dev_secret_key_change_in_production_12345678901234567890"
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
    # DATABASE CONFIGURATION (WITH DEFAULTS)
    # =============================================================================
    
    # MongoDB Atlas (Primary Database)
    mongodb_url: str = "mongodb://localhost:27017/healthsync_dev"
    mongodb_database: str = "healthsync"
    
    # Redis Cache (Optional - for caching)
    redis_url: str = "redis://localhost:6379/0"
    
    # =============================================================================
    # AI/ML SERVICES (WITH DEMO DEFAULTS)
    # =============================================================================
    
    # Groq API (with demo default)
    groq_api_key: str = "demo_groq_key_get_real_key_from_console_groq_com"
    groq_model: str = "llama-3.1-70b-versatile"
    
    # Replicate API (with demo default)
    replicate_api_token: str = "demo_replicate_token_get_real_token_from_replicate_com"
    
    # Hugging Face (with demo default)
    huggingface_api_key: str = "demo_hf_key_get_real_key_from_huggingface_co"
    
    # OpenAI (optional)
    openai_api_key: Optional[str] = None
    
    # =============================================================================
    # EXTERNAL SERVICES (WITH DEMO DEFAULTS)
    # =============================================================================
    
    # Email Service (with demo default)
    resend_api_key: str = "demo_resend_key_get_real_key_from_resend_com"
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
    # HELPER METHODS TO CHECK REAL API KEYS
    # =============================================================================
    
    def has_real_groq_key(self) -> bool:
        """Check if real Groq API key is configured."""
        return self.groq_api_key.startswith("gsk_")
    
    def has_real_replicate_token(self) -> bool:
        """Check if real Replicate token is configured."""
        return self.replicate_api_token.startswith("r8_")
    
    def has_real_huggingface_key(self) -> bool:
        """Check if real Hugging Face key is configured."""
        return self.huggingface_api_key.startswith("hf_")
    
    def has_mongodb_atlas(self) -> bool:
        """Check if MongoDB Atlas is configured."""
        return "mongodb+srv://" in self.mongodb_url and "cluster" in self.mongodb_url
    
    def get_ai_status(self) -> dict:
        """Get status of all AI services."""
        return {
            "groq": self.has_real_groq_key(),
            "replicate": self.has_real_replicate_token(),
            "huggingface": self.has_real_huggingface_key(),
            "mongodb_atlas": self.has_mongodb_atlas()
        }
    
    def is_demo_mode(self) -> bool:
        """Check if running in demo mode (no real API keys)."""
        ai_status = self.get_ai_status()
        return not any(ai_status.values())
    
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
    # FEATURE FLAGS
    # =============================================================================
    
    # Enable/disable features
    enable_voice_analysis: bool = True
    enable_ar_scanner: bool = True
    enable_therapy_game: bool = True
    enable_future_simulator: bool = True
    enable_doctor_marketplace: bool = True
    
    # AR Scanner Configuration
    ar_scanner_cache_ttl: int = 1800  # 30 minutes
    ar_scanner_max_image_size: int = 10  # MB
    ar_scanner_performance_monitoring: bool = True
    ar_scanner_fallback_enabled: bool = True
    ar_scanner_dynamic_analysis: bool = True  # Enable real dynamic analysis
    ar_scanner_ensemble_models: bool = True  # Use multiple AI models
    ar_scanner_real_ocr: bool = True  # Enable real OCR processing
    
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
    # RELAXED VALIDATION (NO STRICT REQUIREMENTS)
    # =============================================================================
    
    @validator('secret_key')
    def validate_secret_key(cls, v):
        # More relaxed validation for development
        if len(v) < 16:
            return "healthsync_dev_secret_key_change_in_production_12345678901234567890"
        return v
    
    @validator('jwt_access_token_expire_minutes')
    def validate_token_expiry(cls, v):
        if v < 5 or v > 1440:  # 5 minutes to 24 hours
            return 30  # Default to 30 minutes
        return v
    
    @validator('port')
    def validate_port(cls, v):
        if v < 1024 or v > 65535:
            return 8000  # Default port
        return v
    
    class Config:
        # Look for .env in parent directory (root) and current directory
        import os
        env_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
        case_sensitive = False
        
        # Don't fail on missing env vars
        env_ignore_empty = True
        
        # Allow extra fields from .env
        extra = "ignore"
        
        # Validation
        validate_assignment = True
        use_enum_values = True


@lru_cache()
def get_flexible_settings() -> FlexibleSettings:
    """Get flexible application settings."""
    return FlexibleSettings()


# Global flexible settings instance
settings = get_flexible_settings()


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

import logging
from pathlib import Path

def configure_logging():
    """Configure basic logging."""
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure standard logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/healthsync.log")
        ]
    )


# =============================================================================
# API DOCUMENTATION CONFIGURATION
# =============================================================================

class APIDocsConfig:
    """API documentation configuration."""
    
    title = "HealthSync AI API"
    description = """
    HealthSync AI is a comprehensive healthcare platform with AI-powered features:
    
    ## Features
    * **Voice AI Doctor** - Real-time voice analysis and consultations
    * **AR Medical Scanner** - OCR and image analysis for medical documents
    * **Pain-to-Game Therapy** - Gamified rehabilitation exercises
    * **Doctor Marketplace** - Specialist matching and appointment booking
    * **Future-You Simulator** - Age progression with health projections
    * **Health Twin + Family Graph** - Disease prediction and family health tracking
    
    ## Demo Mode
    This API works in demo mode without API keys, providing sample responses.
    Connect real AI services for full functionality.
    
    ## Authentication
    Basic JWT authentication for user management.
    """
    
    version = "1.0.0"
    contact = {
        "name": "HealthSync AI Team",
        "email": "support@healthsync.ai"
    }
    
    tags_metadata = [
        {"name": "Authentication", "description": "User authentication"},
        {"name": "Health", "description": "Health metrics and predictions"},
        {"name": "Voice", "description": "Voice AI doctor consultations"},
        {"name": "AR Scanner", "description": "Medical document scanning"},
        {"name": "Therapy", "description": "Gamified therapy sessions"},
        {"name": "Doctors", "description": "Doctor marketplace"},
        {"name": "Future Simulator", "description": "Future health simulation"},
        {"name": "System", "description": "System health and status"}
    ]