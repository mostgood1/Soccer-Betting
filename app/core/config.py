from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    # Pydantic v2 configuration
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    # Database
    database_url: str = "postgresql://user:password@localhost:5432/soccer_betting"

    # Security
    secret_key: str = "your-secret-key-here"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    # API Keys
    football_api_key: Optional[str] = None
    rapid_api_key: Optional[str] = None
    football_data_api_key: Optional[str] = None  # Football-Data.org API key

    # Redis
    redis_url: str = "redis://localhost:6379"

    # Environment
    environment: str = "development"
    debug: bool = True
    log_level: str = "INFO"

    # API Configuration
    api_host: str = "127.0.0.1"
    api_port: str = "8000"

    # Backward compatibility note:
    # Extra env vars like odds_api_key, football_data_api_token, ml_skip_startup_train,
    # uvicorn_no_reload are tolerated (ignored) to avoid ValidationError in tests.


settings = Settings()
