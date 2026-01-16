"""Configuration management using Pydantic Settings."""

from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # LLM Settings
    ollama_host: str = "http://localhost:11434"
    main_model: str = "qwen3:14b"
    embedding_model: str = "nomic-embed-text"
    
    # Database Paths
    lancedb_path: str = "./data/lancedb"
    conversations_path: str = "./data/conversations"
    falkordb_host: str = "localhost"
    falkordb_port: int = 6379
    
    # Memory Settings
    max_context_tokens: int = 3000
    sliding_window_turns: int = 10
    vector_search_top_k: int = 5
    
    # Logging
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()


def get_data_dir() -> Path:
    """Ensure data directories exist and return base path."""
    base = Path(settings.lancedb_path).parent
    base.mkdir(parents=True, exist_ok=True)
    Path(settings.conversations_path).mkdir(parents=True, exist_ok=True)
    return base
