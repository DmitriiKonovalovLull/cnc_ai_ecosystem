import os
from pathlib import Path
from typing import List, Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "CNC AI Ecosystem"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")

    # Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://cnc_user:cnc_password@localhost:5432/cnc_ecosystem"
    )
    SYNC_DATABASE_URL: str = DATABASE_URL.replace("asyncpg", "psycopg2")

    # Redis
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # Vector Database
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY")
    CHROMA_PERSIST_DIR: str = str(BASE_DIR / "data" / "vector_db" / "chroma")

    # AI Models
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    INTENT_MODEL: str = os.getenv("INTENT_MODEL", "cointegrated/rubert-tiny")
    NER_MODEL: str = os.getenv("NER_MODEL", "DeepPavlov/rubert-base-cased-ner-ontonotes")

    # API Keys
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    SANDVIK_API_KEY: Optional[str] = os.getenv("SANDVIK_API_KEY")
    ISCAR_API_KEY: Optional[str] = os.getenv("ISCAR_API_KEY")

    # Security
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-me")
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # File Storage
    UPLOAD_DIR: Path = BASE_DIR / "storage" / "uploads"
    TEMP_DIR: Path = BASE_DIR / "storage" / "temp"
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB

    # Scraping
    USER_AGENTS: List[str] = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
        "Googlebot/2.1 (+http://www.google.com/bot.html)"
    ]
    REQUEST_DELAY: tuple = (1.0, 5.0)
    MAX_RETRIES: int = 3
    PROXY_LIST: Optional[List[str]] = None

    # Training
    FEEDBACK_BATCH_SIZE: int = 50
    TRAINING_EPOCHS: int = 10
    LEARNING_RATE: float = 1e-5

    # Monitoring
    SENTRY_DSN: Optional[str] = os.getenv("SENTRY_DSN")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

# Create directories
for directory in [settings.UPLOAD_DIR, settings.TEMP_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
