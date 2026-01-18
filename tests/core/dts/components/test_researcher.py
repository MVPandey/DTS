"""Tests for backend/core/dts/components/researcher.py."""

import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.core.dts.components.researcher import DeepResearcher
from backend.llm.types import Completion, Message

# -----------------------------------------------------------------------------
# DeepResearcher Initialization Tests
# -----------------------------------------------------------------------------


class TestDeepResearcherInit:
    """Tests for DeepResearcher initialization."""

    def test_init_with_defaults(self) -> None:
        """Test initialization with default values."""
        mock_llm = MagicMock()
        with tempfile.TemporaryDirectory() as tmp_dir:
            researcher = DeepResearcher(llm=mock_llm, cache_dir=tmp_dir)

            assert researcher.llm is mock_llm
            assert researcher.model is None
            assert researcher.cache_dir == Path(tmp_dir)

    def test_init_with_custom_values(self) -> None:
        """Test initialization with custom values."""
        mock_llm = MagicMock()
        on_cost = MagicMock()
        on_event = AsyncMock()

        with tempfile.TemporaryDirectory() as tmp_dir:
            researcher = DeepResearcher(
                llm=mock_llm,
                model="gpt-4",
                cache_dir=tmp_dir,
                max_concurrent_research=3,
                on_cost=on_cost,
                on_event=on_event,
            )

            assert researcher.model == "gpt-4"
            assert researcher._on_cost is on_cost

    def test_init_creates_cache_dir(self) -> None:
        """Test that cache directory is created."""
        mock_llm = MagicMock()
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "new_cache"
            DeepResearcher(llm=mock_llm, cache_dir=str(cache_path))

            assert cache_path.exists()


# -----------------------------------------------------------------------------
# Cache Key Generation Tests
# -----------------------------------------------------------------------------


class TestCacheKey:
    """Tests for cache key generation."""

    def test_get_cache_key_deterministic(self) -> None:
        """Test cache key is deterministic."""
        mock_llm = MagicMock()
        with tempfile.TemporaryDirectory() as tmp_dir:
            researcher = DeepResearcher(llm=mock_llm, cache_dir=tmp_dir)

            key1 = researcher._get_cache_key("goal", "message")
            key2 = researcher._get_cache_key("goal", "message")

            assert key1 == key2

    def test_get_cache_key_unique_for_different_inputs(self) -> None:
        """Test cache key is unique for different inputs."""
        mock_llm = MagicMock()
        with tempfile.TemporaryDirectory() as tmp_dir:
            researcher = DeepResearcher(llm=mock_llm, cache_dir=tmp_dir)

            key1 = researcher._get_cache_key("goal1", "message")
            key2 = researcher._get_cache_key("goal2", "message")

            assert key1 != key2


# -----------------------------------------------------------------------------
# Cache Load/Save Tests
# -----------------------------------------------------------------------------


class TestCacheOperations:
    """Tests for cache load and save operations."""

    def test_load_cache_returns_none_when_missing(self) -> None:
        """Test loading non-existent cache returns None."""
        mock_llm = MagicMock()
        with tempfile.TemporaryDirectory() as tmp_dir:
            researcher = DeepResearcher(llm=mock_llm, cache_dir=tmp_dir)

            result = researcher._load_cache("nonexistent_key")
            assert result is None

    def test_save_and_load_cache(self) -> None:
        """Test saving and loading cache."""
        mock_llm = MagicMock()
        with tempfile.TemporaryDirectory() as tmp_dir:
            researcher = DeepResearcher(llm=mock_llm, cache_dir=tmp_dir)

            key = "test_key"
            report = "Test research report content"

            researcher._save_cache(key, report)
            result = researcher._load_cache(key)

            assert result == report

    def test_load_cache_handles_invalid_json(self) -> None:
        """Test loading invalid JSON returns None."""
        mock_llm = MagicMock()
        with tempfile.TemporaryDirectory() as tmp_dir:
            researcher = DeepResearcher(llm=mock_llm, cache_dir=tmp_dir)

            # Write invalid JSON
            cache_path = Path(tmp_dir) / "test_key.json"
            cache_path.write_text("invalid json {{{")

            result = researcher._load_cache("test_key")
            assert result is None

    def test_save_cache_handles_write_error(self) -> None:
        """Test save cache handles write errors gracefully."""
        mock_llm = MagicMock()
        with tempfile.TemporaryDirectory() as tmp_dir:
            researcher = DeepResearcher(llm=mock_llm, cache_dir=tmp_dir)

            # Make cache dir read-only
            os.chmod(tmp_dir, 0o444)

            try:
                # Should not raise
                researcher._save_cache("key", "report")
            finally:
                # Restore permissions for cleanup
                os.chmod(tmp_dir, 0o755)


# -----------------------------------------------------------------------------
# Query Generation Tests
# -----------------------------------------------------------------------------


class TestGenerateQuery:
    """Tests for query generation."""

    @pytest.mark.asyncio
    async def test_generate_query_success(self) -> None:
        """Test successful query generation."""
        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(
            return_value=Completion(
                message=Message.assistant("How to debug Python async code effectively")
            )
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            researcher = DeepResearcher(llm=mock_llm, cache_dir=tmp_dir)

            query = await researcher._generate_query(
                goal="Help debug async code",
                first_message="My async function is hanging",
            )

            assert query == "How to debug Python async code effectively"
            mock_llm.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_query_fallback_on_error(self) -> None:
        """Test fallback when LLM fails."""
        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(side_effect=Exception("LLM Error"))

        with tempfile.TemporaryDirectory() as tmp_dir:
            researcher = DeepResearcher(llm=mock_llm, cache_dir=tmp_dir)

            query = await researcher._generate_query(
                goal="Test goal",
                first_message="Test message",
            )

            assert query == "Test goal - Test message"

    @pytest.mark.asyncio
    async def test_generate_query_fallback_on_empty(self) -> None:
        """Test fallback when LLM returns empty."""
        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(return_value=Completion(message=Message.assistant("")))

        with tempfile.TemporaryDirectory() as tmp_dir:
            researcher = DeepResearcher(llm=mock_llm, cache_dir=tmp_dir)

            query = await researcher._generate_query(
                goal="Test goal",
                first_message="Test message",
            )

            assert query == "Test goal - Test message"


# -----------------------------------------------------------------------------
# Validate Requirements Tests
# -----------------------------------------------------------------------------


class TestValidateRequirements:
    """Tests for requirement validation."""

    def test_validate_missing_openai_key(self) -> None:
        """Test validation fails without OpenAI key."""
        mock_llm = MagicMock()
        with tempfile.TemporaryDirectory() as tmp_dir:
            researcher = DeepResearcher(llm=mock_llm, cache_dir=tmp_dir)

            with patch("backend.core.dts.components.researcher.config") as mock_config:
                mock_config.openai_api_key = None
                mock_config.firecrawl_api_key = "test-key"

                with pytest.raises(ValueError) as exc_info:
                    researcher._validate_requirements()

                assert "OPENAI_API_KEY required" in str(exc_info.value)

    def test_validate_missing_firecrawl_key(self) -> None:
        """Test validation fails without Firecrawl key."""
        mock_llm = MagicMock()
        with tempfile.TemporaryDirectory() as tmp_dir:
            researcher = DeepResearcher(llm=mock_llm, cache_dir=tmp_dir)

            with patch("backend.core.dts.components.researcher.config") as mock_config:
                mock_config.openai_api_key = "test-key"
                mock_config.firecrawl_api_key = None

                with pytest.raises(ValueError) as exc_info:
                    researcher._validate_requirements()

                assert "FIRECRAWL_API_KEY required" in str(exc_info.value)


# -----------------------------------------------------------------------------
# Setup Environment Tests
# -----------------------------------------------------------------------------


class TestSetupEnvironment:
    """Tests for environment setup."""

    def test_setup_environment_sets_vars(self) -> None:
        """Test environment variables are set."""
        mock_llm = MagicMock()
        with tempfile.TemporaryDirectory() as tmp_dir:
            researcher = DeepResearcher(llm=mock_llm, cache_dir=tmp_dir)

            with patch("backend.core.dts.components.researcher.config") as mock_config:
                mock_config.openai_api_key = "test-openai-key"
                mock_config.openrouter_api_key = "test-openrouter-key"
                mock_config.openai_base_url = "https://api.test.com"
                mock_config.firecrawl_api_key = "test-firecrawl-key"
                mock_config.fast_llm = "gpt-3.5"
                mock_config.smart_llm = "gpt-4"
                mock_config.strategic_llm = "gpt-4"
                mock_config.smart_token_limit = 8000
                mock_config.scraper = "firecrawl"
                mock_config.max_scraper_workers = 4
                mock_config.embedding_model = "text-embedding-ada"
                mock_config.deep_research_breadth = 3
                mock_config.deep_research_depth = 2
                mock_config.deep_research_concurrency = 5
                mock_config.total_words = 5000
                mock_config.max_subtopics = 3
                mock_config.max_iterations = 3
                mock_config.max_search_results = 5
                mock_config.report_format = "markdown"

                researcher._setup_environment()

                assert os.environ.get("OPENAI_API_KEY") == "test-openai-key"
                assert os.environ.get("OPENROUTER_API_KEY") == "test-openrouter-key"
                assert os.environ.get("FIRECRAWL_API_KEY") == "test-firecrawl-key"


# -----------------------------------------------------------------------------
# Research Method Tests
# -----------------------------------------------------------------------------


class TestResearch:
    """Tests for the main research method."""

    @pytest.mark.asyncio
    async def test_research_returns_cached(self) -> None:
        """Test research returns cached result."""
        mock_llm = MagicMock()
        with tempfile.TemporaryDirectory() as tmp_dir:
            researcher = DeepResearcher(llm=mock_llm, cache_dir=tmp_dir)

            # Pre-populate cache
            key = researcher._get_cache_key("Test goal", "Test message")
            researcher._save_cache(key, "Cached research report")

            result = await researcher.research(
                goal="Test goal",
                first_message="Test message",
            )

            assert result == "Cached research report"
            # LLM should not be called
            mock_llm.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_research_calls_gpt_researcher(self) -> None:
        """Test research calls GPT Researcher when no cache."""
        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(
            return_value=Completion(message=Message.assistant("Research query"))
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            researcher = DeepResearcher(llm=mock_llm, cache_dir=tmp_dir)

            with (
                patch("backend.core.dts.components.researcher.config") as mock_config,
                patch("gpt_researcher.GPTResearcher") as mock_gpt_researcher,
            ):
                # Set up config
                mock_config.openai_api_key = "test-key"
                mock_config.firecrawl_api_key = "test-firecrawl"
                mock_config.openrouter_api_key = "test-key"
                mock_config.openai_base_url = "https://api.test.com"
                mock_config.fast_llm = None
                mock_config.smart_llm = None
                mock_config.strategic_llm = None
                mock_config.smart_token_limit = 8000
                mock_config.scraper = None
                mock_config.max_scraper_workers = 4
                mock_config.embedding_model = "ada"
                mock_config.deep_research_breadth = 3
                mock_config.deep_research_depth = 2
                mock_config.deep_research_concurrency = 5
                mock_config.total_words = 5000
                mock_config.max_subtopics = 3
                mock_config.max_iterations = 3
                mock_config.max_search_results = 5
                mock_config.report_format = "markdown"

                # Mock GPT Researcher
                mock_researcher_instance = MagicMock()
                mock_researcher_instance.conduct_research = AsyncMock()
                mock_researcher_instance.write_report = AsyncMock(return_value="Research findings")
                mock_researcher_instance.get_costs.return_value = 0.05
                mock_gpt_researcher.return_value = mock_researcher_instance

                result = await researcher.research(
                    goal="Test goal",
                    first_message="Test message",
                )

                assert result == "Research findings"
                mock_researcher_instance.conduct_research.assert_called_once()
                mock_researcher_instance.write_report.assert_called_once()

    @pytest.mark.asyncio
    async def test_research_tracks_cost(self) -> None:
        """Test research tracks cost via callback."""
        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(return_value=Completion(message=Message.assistant("Query")))
        cost_callback = MagicMock()

        with tempfile.TemporaryDirectory() as tmp_dir:
            researcher = DeepResearcher(
                llm=mock_llm,
                cache_dir=tmp_dir,
                on_cost=cost_callback,
            )

            with (
                patch("backend.core.dts.components.researcher.config") as mock_config,
                patch("gpt_researcher.GPTResearcher") as mock_gpt_researcher,
            ):
                # Setup mocks
                mock_config.openai_api_key = "test"
                mock_config.firecrawl_api_key = "test"
                mock_config.openrouter_api_key = "test"
                mock_config.openai_base_url = "https://api.test.com"
                mock_config.fast_llm = None
                mock_config.smart_llm = None
                mock_config.strategic_llm = None
                mock_config.smart_token_limit = 8000
                mock_config.scraper = None
                mock_config.max_scraper_workers = 4
                mock_config.embedding_model = "ada"
                mock_config.deep_research_breadth = 3
                mock_config.deep_research_depth = 2
                mock_config.deep_research_concurrency = 5
                mock_config.total_words = 5000
                mock_config.max_subtopics = 3
                mock_config.max_iterations = 3
                mock_config.max_search_results = 5
                mock_config.report_format = "markdown"

                mock_researcher_instance = MagicMock()
                mock_researcher_instance.conduct_research = AsyncMock()
                mock_researcher_instance.write_report = AsyncMock(return_value="Report")
                mock_researcher_instance.get_costs.return_value = 0.10
                mock_gpt_researcher.return_value = mock_researcher_instance

                await researcher.research("Goal", "Message")

                cost_callback.assert_called_once_with(0.10)

    @pytest.mark.asyncio
    async def test_research_raises_on_missing_gpt_researcher(self) -> None:
        """Test research raises when gpt-researcher not installed."""
        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(return_value=Completion(message=Message.assistant("Query")))

        with tempfile.TemporaryDirectory() as tmp_dir:
            researcher = DeepResearcher(llm=mock_llm, cache_dir=tmp_dir)

            with patch("backend.core.dts.components.researcher.config") as mock_config:
                mock_config.openai_api_key = "test"
                mock_config.firecrawl_api_key = "test"
                mock_config.openrouter_api_key = "test"
                mock_config.openai_base_url = "https://api.test.com"
                mock_config.fast_llm = None
                mock_config.smart_llm = None
                mock_config.strategic_llm = None
                mock_config.smart_token_limit = 8000
                mock_config.scraper = None
                mock_config.max_scraper_workers = 4
                mock_config.embedding_model = "ada"
                mock_config.deep_research_breadth = 3
                mock_config.deep_research_depth = 2
                mock_config.deep_research_concurrency = 5
                mock_config.total_words = 5000
                mock_config.max_subtopics = 3
                mock_config.max_iterations = 3
                mock_config.max_search_results = 5
                mock_config.report_format = "markdown"

                # Simulate import error
                import builtins

                original_import = builtins.__import__

                def mock_import(name, *args, **kwargs):
                    if name == "gpt_researcher":
                        raise ImportError("No module named 'gpt_researcher'")
                    return original_import(name, *args, **kwargs)

                with patch.object(builtins, "__import__", side_effect=mock_import):
                    with pytest.raises(RuntimeError) as exc_info:
                        await researcher.research("Goal", "Message")

                    assert "gpt-researcher not installed" in str(exc_info.value)
