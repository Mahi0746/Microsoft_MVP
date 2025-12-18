# HealthSync AI - Test Configuration
import os
from typing import Optional
from pydantic import BaseSettings

class TestSettings(BaseSettings):
    """Test configuration settings"""
    
    # Basic settings
    secret_key: str = "test_secret_key_for_testing_only"
    jwt_secret_key: str = "test_jwt_secret_key_for_testing_only"
    environment: str = "test"
    debug: bool = True
    
    # Database URLs (Mock)
    database_url: str = "sqlite:///./test.db"
    mongodb_url: str = "mongodb://localhost:27017/healthsync_test"
    redis_url: str = "redis://localhost:6379/0"
    
    # Supabase (Mock)
    supabase_url: str = "https://test.supabase.co"
    supabase_anon_key: str = "test_anon_key"
    supabase_service_role_key: str = "test_service_role_key"
    
    # AI Services (Mock)
    groq_api_key: str = "test_groq_api_key"
    replicate_api_token: str = "test_replicate_token"
    huggingface_api_key: str = "test_huggingface_key"
    openai_api_key: Optional[str] = "test_openai_key"
    
    # Email Service (Mock)
    resend_api_key: str = "test_resend_key"
    
    # Server settings
    host: str = "127.0.0.1"
    port: int = 8000
    
    # CORS
    allowed_origins: list = ["http://localhost:3000", "http://localhost:19006"]
    
    # File upload settings
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_image_types: list = ["image/jpeg", "image/png", "image/webp"]
    allowed_audio_types: list = ["audio/wav", "audio/mp3", "audio/m4a"]
    
    # JWT settings
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30
    jwt_refresh_token_expire_days: int = 7
    
    class Config:
        env_file = ".env.test"
        case_sensitive = False

# Global test settings instance
test_settings = TestSettings()