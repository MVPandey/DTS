from typing import Self

from pydantic import Field, PostgresDsn, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="Base URL for LLM API",
    )
    openai_api_key: str = Field(
        ...,
        description="API key for LLM service",
    )

    openrouter_api_key: str = Field(
        default="",
        description="API key for OpenRouter API",
    )

    # LLM client settings
    llm_timeout: float = Field(
        default=120.0,
        description="Request timeout in seconds for LLM calls",
    )
    llm_max_retries: int = Field(
        default=2,
        description="Maximum retries for failed LLM requests",
    )
    max_concurrency: int = Field(
        default=16,
        description="Maximum concurrent LLM calls",
    )

    @model_validator(mode="after")
    def set_openrouter_fallback(self) -> Self:
        """Fall back to openai_api_key if openrouter_api_key is not set."""
        if not self.openrouter_api_key:
            self.openrouter_api_key = self.openai_api_key
        return self

    llm_name: str = Field(default="minimax/minimax-m2.1", description="LLM model name")

    # --- These are for deep researcher ---
    fast_llm: str = Field(
        default="openrouter:minimax/minimax-m2.1",
        description="Fast LLM model name",
    )
    smart_llm: str = Field(
        default="openrouter:minimax/minimax-m2.1",
        description="Smart LLM model name",
    )
    strategic_llm: str = Field(
        default="openrouter:minimax/minimax-m2.1",
        description="Strategic LLM model name",
    )
    reasoning_enabled: bool = Field(
        default=False,
        description="Enable reasoning tokens for LLM calls (increases cost but may improve quality)",
    )

    firecrawl_api_key: str = Field(
        default="",
        description="API key for Firecrawl API",
    )

    scraper: str = Field(
        default="firecrawl",
        description="Web scraper to use (firecrawl, tavily, etc.)",
    )
    max_scraper_workers: int = Field(
        default=5,
        description="Maximum concurrent scraper workers for deep research",
    )

    # Deep research parameters
    deep_research_breadth: int = Field(
        default=6,
        description="Number of parallel research queries per depth level",
    )
    deep_research_depth: int = Field(
        default=4,
        description="Maximum depth of research tree exploration",
    )
    deep_research_concurrency: int = Field(
        default=4,
        description="Maximum concurrent research tasks",
    )
    total_words: int = Field(
        default=12000,
        description="Target word count for research report (higher = more comprehensive)",
    )
    max_subtopics: int = Field(
        default=8,
        description="Maximum number of subtopics for detailed reports",
    )
    max_iterations: int = Field(
        default=5,
        description="Maximum research iterations per subtopic",
    )
    max_search_results: int = Field(
        default=10,
        description="Maximum search results per query",
    )
    report_format: str = Field(
        default="markdown",
        description="Report format (markdown, APA, etc.)",
    )
    smart_token_limit: int = Field(
        default=32000,
        description="Token limit for smart LLM model in gpt-researcher",
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Embedding model for deep research",
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
