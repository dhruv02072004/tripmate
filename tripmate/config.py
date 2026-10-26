from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    google_api_key: str = ""
    llm_model: str = "gemini-3.5-flash-lite"   
    llm_timeout_s: int = 30

    # RAG
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    chroma_dir: Path = Path("./.chroma")
    data_dir: Path = Path("./data/destinations")
    collection_name: str = "destination_guides"
    rag_top_k: int = 3
    rag_max_distance: float = 0.398

    # Weather
    weather_timeout_s: float = 8.0
    weather_cache_ttl_s: int = 3600

    # Agent / safety
    max_query_chars: int = 1000
    max_agent_steps: int = 8

    # Logging
    log_level: str = "INFO"
    log_file: Path = Path("logs/trace.jsonl")


@lru_cache
def get_settings() -> Settings:
    return Settings()