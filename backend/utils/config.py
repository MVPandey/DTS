from pydantic import Field, PostgresDsn
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    openai_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="Base URL for LLM API",
    )
    openai_api_key: str = Field(
        ...,
        description="API key for LLM service",
    )
    llm_name: str = Field(default="minimax/minimax-m2.1", description="LLM model name")
    embedding_base_url: str = Field(
        default="https://openrouter.ai/api/v1", description="Base URL for embedding API"
    )
    embedding_api_key: str = Field(
        default="",
        description="API key for embedding service (defaults to OPENAI_API_KEY if not set)",
    )
    embedding_model_name: str = Field(
        default="openai/text-embedding-3-small", description="Embedding model name"
    )
    llm_provider: str | None = Field(default="fireworks", description="LLM provider")
    logging_level: str = Field(default="INFO", description="Logging level")
    debug: bool = Field(default=False, description="Debug mode")
    server_port: int = Field(default=8000, ge=1, le=65535, description="Server port")
    server_host: str = Field(default="localhost", description="Server host")
    database_url: PostgresDsn | None = Field(
        default=None, env="DATABASE_URL", description="PostgreSQL database URL"
    )


config = Config()
