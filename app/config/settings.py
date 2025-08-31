# app/config/settings.py
import os
import logging
import logging.config
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    # --- OpenAI / LLM ---
    OPENAI_MODEL: str = "gpt-3.5-turbo"  # default from your project overview
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    
    # --- Paths / data ---
    DATA_DIR: str = Field(default="app/data")
    CACHE_DIR: str = Field(default="app/data/cache")
    SP100_FILE: str = Field(default="app/data/sp100.json")
    PDF_DIR: str = Field(default="app/data/pdfs/")

    # Vector DB (Chroma)
    CHROMA_DIR: str = Field(default="app/data/vectordb")
    CHROMA_COLLECTION: str = Field(default="ta_pdfs")

    # Embeddings / LLM
    EMBEDDING_MODEL: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    OPENAI_API_KEY: str = Field(default="")

    # Market data (FMP/Finnhub only now)
    HISTORY_YEARS: int = Field(default=1)             # 1-year history
    MAX_TICKERS: int = Field(default=100)             # S&P 100 cap
    CACHE_TTL_MARKET: int = Field(default=60 * 60)    # 1 hour

    # Providers (keep for future switches)
    MARKET_PROVIDER: str = Field(default="auto")
    FMP_API_KEY: str = Field(default="")
    FINNHUB_API_KEY: str = Field(default="")

    # Misc
    DEBUG: bool = Field(default=False)

    # --- Extra keys you had in .env (now supported) ---
    LOGGING_CONFIG: str = Field(default="app/config/logging.conf", alias="logging_config")
    LOG_LEVEL: str = Field(default="INFO", alias="log_level")

    CACHE_TTL_EMBEDDINGS: int = Field(default=86400, alias="cache_ttl_embeddings")
    VECTOR_BACKEND: str = Field(default="chroma", alias="vector_backend")
    EMBED_PROVIDER: str = Field(default="openai", alias="embed_provider")
    OPENAI_EMBED_MODEL: str = Field(default="text-embedding-3-small", alias="openai_embed_model")
    EMBED_MAX_PAGE_CHARS: int = Field(default=150000, alias="embed_max_page_chars")
    EMBED_CHUNK_SIZE: int = Field(default=800, alias="embed_chunk_size")
    EMBED_CHUNK_OVERLAP: int = Field(default=120, alias="embed_chunk_overlap")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",             # ignore any other unexpected env keys
        case_sensitive=False,       # allow LOG_LEVEL vs log_level, etc.
        populate_by_name=True,      # accept both field name and alias
    )


def init_logging() -> None:
    """
    Configure logging from settings.LOGGING_CONFIG if present,
    otherwise fall back to basicConfig with settings.LOG_LEVEL (or INFO).
    """
    cfg_path = settings.LOGGING_CONFIG if settings.LOGGING_CONFIG else os.path.join("app", "config", "logging.conf")
    if cfg_path and os.path.exists(cfg_path):
        logging.config.fileConfig(cfg_path, disable_existing_loggers=False)
    else:
        logging.basicConfig(
            level=getattr(logging, (settings.LOG_LEVEL or "INFO").upper(), logging.INFO),
            format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        )


def _ensure_dirs(s: "Settings") -> None:
    os.makedirs(s.DATA_DIR, exist_ok=True)
    os.makedirs(s.CACHE_DIR, exist_ok=True)
    os.makedirs(s.CHROMA_DIR, exist_ok=True)
    os.makedirs(os.path.join(s.CACHE_DIR, "market"), exist_ok=True)


# Singleton settings instance
settings = Settings()
_ensure_dirs(settings)
