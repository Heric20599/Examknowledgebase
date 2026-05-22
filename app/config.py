from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    openai_api_key: str
    openai_embed_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"

    pinecone_api_key: str
    pinecone_index: str = "schoolknowledgebase"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    embed_dim: int = 1536

    chunk_size: int = 800
    chunk_overlap: int = 120
    max_pdf_mb: int = 50

    # In-process keep-alive loop. If keepalive_url is set, the app pings it on
    # an interval from inside the FastAPI lifespan. Leave blank to disable.
    keepalive_url: str | None = None
    keepalive_interval_seconds: int = 600


@lru_cache
def get_settings() -> Settings:
    return Settings()
