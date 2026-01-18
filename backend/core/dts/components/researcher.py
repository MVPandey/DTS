"""Deep research component using GPT Researcher."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from backend.core.dts.utils import create_event_emitter, log_phase
from backend.llm.types import Message
from backend.utils.config import config
from backend.utils.logging import logger

if TYPE_CHECKING:
    from collections.abc import Awaitable, Coroutine

    from backend.llm.client import LLM

    EventCallback = Callable[
        [str, dict[str, Any]], Awaitable[None] | Coroutine[Any, Any, None] | None
    ]


class DeepResearcher:
    """
    Conducts deep research using gpt-researcher package.

    Provides domain context for strategy generation by researching
    the conversation goal and initial message.
    """

    QUERY_DISTILL_PROMPT = """Distill the following conversation goal and opening message into a single, focused research query. The query should capture the essential topic and context needed for web research.

Goal: {goal}
First message: {first_message}

Write a single sentence research query that will help gather relevant domain knowledge, tactics, and context. Output ONLY the query, nothing else."""

    def __init__(
        self,
        llm: LLM,
        model: str | None = None,
        cache_dir: str = ".cache/research",
        max_concurrent_research: int = 5,
        on_cost: Callable[[float], None] | None = None,
        on_event: EventCallback | None = None,
    ) -> None:
        """
        Initialize the researcher.

        Args:
            llm: LLM client for query generation.
            model: Model to use for query distillation.
            cache_dir: Directory for caching research results.
            max_concurrent_research: Maximum concurrent research requests.
            on_cost: Callback for tracking external USD costs.
            on_event: Async callback for emitting events to UI.
        """
        self.llm = llm
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._sem = asyncio.Semaphore(max_concurrent_research)
        self._on_cost = on_cost
        self._emit = create_event_emitter(on_event, logger)

    async def research(
        self,
        goal: str,
        first_message: str,
        report_type: str = "deep",
    ) -> str:
        """
        Conduct research on the conversation goal and context.

        Args:
            goal: Conversation goal/objective.
            first_message: Initial user message for context.
            report_type: GPT Researcher report type.

        Returns:
            Research report as string context.

        Raises:
            ValueError: If required API keys are missing.
            RuntimeError: If gpt-researcher is not installed.
        """
        # Check cache first
        cache_key = self._get_cache_key(goal, first_message)
        cached = self._load_cache(cache_key)
        if cached:
            log_phase(logger, "RESEARCH", f"Cache hit: {cache_key[:8]}...", indent=1)
            self._emit(
                "research_log",
                {"message": "Using cached research results", "type": "cache_hit"},
            )
            return cached

        # Validate dependencies and setup environment
        self._validate_requirements()
        self._setup_environment()

        # Generate research query via LLM
        self._emit(
            "research_log",
            {"message": "Generating research query...", "type": "progress"},
        )
        query = await self._generate_query(goal, first_message)
        log_phase(logger, "RESEARCH", f"Generated query: {query}", indent=1)

        # Run research with rate limiting
        try:
            from gpt_researcher import GPTResearcher

            log_phase(
                logger,
                "RESEARCH",
                f"Starting deep research for: {goal[:50]}...",
                indent=1,
            )
            self._emit(
                "research_log",
                {"message": f"Researching: {goal[:80]}...", "type": "start"},
            )

            researcher = GPTResearcher(
                query=query,
                report_type=report_type,
            )

            # Rate limit concurrent research requests
            async with self._sem:
                # Conduct research with progress updates
                self._emit(
                    "research_log",
                    {
                        "message": "Searching for relevant sources...",
                        "type": "progress",
                    },
                )
                await researcher.conduct_research()

                self._emit(
                    "research_log",
                    {"message": "Writing research report...", "type": "progress"},
                )
                report = await researcher.write_report()

            # Track external cost
            cost = researcher.get_costs()
            if self._on_cost and cost:
                self._on_cost(cost)
                log_phase(logger, "RESEARCH", f"Research cost: ${cost:.4f}", indent=1)

            # Cache result
            self._save_cache(cache_key, report)
            log_phase(
                logger,
                "RESEARCH",
                f"Research complete, cached as {cache_key[:8]}...",
                indent=1,
            )
            self._emit(
                "research_log",
                {"message": "Research complete", "type": "complete"},
            )

            return report

        except ImportError:
            raise RuntimeError(
                "gpt-researcher not installed. Run: pip install gpt-researcher"
            ) from None

    def _setup_environment(self) -> None:
        """
        Inject config values into os.environ for gpt-researcher.

        GPT Researcher reads directly from environment variables,
        so we need to bridge our Pydantic config to os.environ.
        """
        logger.debug("Setting up environment for GPT Researcher")
        logger.debug(
            f"API keys present - OpenAI: {bool(config.openai_api_key)}, Firecrawl: {bool(config.firecrawl_api_key)}"
        )

        # OpenRouter API key - gpt-researcher needs OPENAI_API_KEY set
        if config.openai_api_key:
            os.environ["OPENAI_API_KEY"] = config.openai_api_key
            os.environ["OPENROUTER_API_KEY"] = config.openrouter_api_key
            os.environ["OPENAI_BASE_URL"] = config.openai_base_url
            logger.debug(f"Set OPENAI_BASE_URL: {config.openai_base_url}")

        # LLM configurations for gpt-researcher
        if config.fast_llm:
            os.environ["FAST_LLM"] = config.fast_llm
        if config.smart_llm:
            os.environ["SMART_LLM"] = config.smart_llm
        if config.strategic_llm:
            os.environ["STRATEGIC_LLM"] = config.strategic_llm
        os.environ["SMART_TOKEN_LIMIT"] = str(config.smart_token_limit)

        # Web scraper configuration
        if config.scraper:
            os.environ["SCRAPER"] = config.scraper
        if config.firecrawl_api_key:
            os.environ["FIRECRAWL_API_KEY"] = config.firecrawl_api_key
        os.environ["MAX_SCRAPER_WORKERS"] = str(config.max_scraper_workers)

        # Embedding config - use custom provider to route through OpenRouter
        os.environ["EMBEDDING"] = f"custom:{config.embedding_model}"

        # Deep research parameters
        os.environ["BREADTH"] = str(config.deep_research_breadth)
        os.environ["DEPTH"] = str(config.deep_research_depth)
        os.environ["CONCURRENCY"] = str(config.deep_research_concurrency)
        os.environ["TOTAL_WORDS"] = str(config.total_words)

        # Comprehensive report parameters
        os.environ["MAX_SUBTOPICS"] = str(config.max_subtopics)
        os.environ["MAX_ITERATIONS"] = str(config.max_iterations)
        os.environ["MAX_SEARCH_RESULTS_PER_QUERY"] = str(config.max_search_results)
        os.environ["REPORT_FORMAT"] = config.report_format

        logger.debug("Environment setup complete")

    def _validate_requirements(self) -> None:
        """Validate required API keys are present."""
        if not config.openai_api_key:
            raise ValueError("OPENAI_API_KEY required for deep_research=True.")
        if not config.firecrawl_api_key:
            raise ValueError(
                "FIRECRAWL_API_KEY required for deep_research=True. "
                "Get one at https://firecrawl.dev"
            )

    async def _generate_query(self, goal: str, first_message: str) -> str:
        """Generate a focused research query using the LLM."""
        prompt = self.QUERY_DISTILL_PROMPT.format(
            goal=goal,
            first_message=first_message,
        )

        try:
            completion = await self.llm.complete(
                [Message.user(prompt)],
                model=self.model,
                temperature=0.3,
            )
            query = (completion.message.content or "").strip()
            if query:
                return query
        except Exception as e:
            logger.warning(f"Query generation failed, using fallback: {e}")

        # Fallback to simple concatenation
        return f"{goal} - {first_message}"

    def _get_cache_key(self, goal: str, first_message: str) -> str:
        """Generate cache key from inputs."""
        composite = f"{goal}::{first_message}"
        return hashlib.sha256(composite.encode()).hexdigest()

    def _load_cache(self, key: str) -> str | None:
        """Load cached research result."""
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            try:
                data = json.loads(path.read_text())
                return data.get("report")
            except Exception:
                return None
        return None

    def _save_cache(self, key: str, report: str) -> None:
        """Save research result to cache."""
        path = self.cache_dir / f"{key}.json"
        try:
            path.write_text(json.dumps({"report": report}))
        except Exception as e:
            logger.warning(f"Failed to cache research: {e}")
