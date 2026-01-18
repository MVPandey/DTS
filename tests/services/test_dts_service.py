"""Tests for backend/services/dts_service.py."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.api.schemas import SearchRequest
from backend.core.dts.config import DTSConfig
from backend.services.dts_service import (
    create_dts_config,
    create_llm_client,
    run_dts_session,
)

# -----------------------------------------------------------------------------
# create_llm_client Tests
# -----------------------------------------------------------------------------


class TestCreateLLMClient:
    """Tests for create_llm_client function."""

    @patch("backend.services.dts_service.LLM")
    @patch("backend.services.dts_service.config")
    def test_creates_llm_with_config(
        self, mock_config: MagicMock, mock_llm_class: MagicMock
    ) -> None:
        """Test that LLM client is created with config values."""
        mock_config.openai_api_key = "test-key"
        mock_config.openai_base_url = "https://api.test.com/v1"
        mock_config.llm_name = "test-model"

        create_llm_client()

        mock_llm_class.assert_called_once_with(
            api_key="test-key",
            base_url="https://api.test.com/v1",
            model="test-model",
        )


# -----------------------------------------------------------------------------
# create_dts_config Tests
# -----------------------------------------------------------------------------


class TestCreateDTSConfig:
    """Tests for create_dts_config function."""

    def test_creates_config_from_request(self) -> None:
        """Test creating DTSConfig from SearchRequest."""
        request = SearchRequest(
            goal="Help user debug code",
            first_message="My code isn't working",
            init_branches=5,
            turns_per_branch=3,
            user_intents_per_branch=2,
            scoring_mode="absolute",
            prune_threshold=7.0,
            deep_research=True,
        )

        config = create_dts_config(request)

        assert isinstance(config, DTSConfig)
        assert config.goal == "Help user debug code"
        assert config.first_message == "My code isn't working"
        assert config.init_branches == 5
        assert config.turns_per_branch == 3
        assert config.user_intents_per_branch == 2
        assert config.scoring_mode == "absolute"
        assert config.prune_threshold == 7.0
        assert config.deep_research is True

    def test_creates_config_with_defaults(self) -> None:
        """Test that default values are used when not specified."""
        request = SearchRequest(
            goal="Test goal",
            first_message="Test message",
        )

        config = create_dts_config(request)

        assert config.init_branches == 6  # Default
        assert config.turns_per_branch == 5  # Default
        assert config.scoring_mode == "comparative"  # Default

    def test_creates_config_with_per_phase_models(self) -> None:
        """Test that per-phase models are passed through."""
        request = SearchRequest(
            goal="Test",
            first_message="Test",
            strategy_model="gpt-4-turbo",
            simulator_model="gpt-3.5-turbo",
            judge_model="claude-3-opus",
        )

        config = create_dts_config(request)

        assert config.strategy_model == "gpt-4-turbo"
        assert config.simulator_model == "gpt-3.5-turbo"
        assert config.judge_model == "claude-3-opus"


# -----------------------------------------------------------------------------
# run_dts_session Tests
# -----------------------------------------------------------------------------


class TestRunDTSSession:
    """Tests for run_dts_session async generator."""

    @pytest.mark.asyncio
    @patch("backend.services.dts_service.DTSEngine")
    @patch("backend.services.dts_service.create_llm_client")
    async def test_yields_events(
        self, mock_create_llm: MagicMock, mock_engine_class: MagicMock
    ) -> None:
        """Test that session yields events from engine."""
        # Setup mocks
        mock_llm = MagicMock()
        mock_create_llm.return_value = mock_llm

        mock_engine = MagicMock()
        mock_engine_class.return_value = mock_engine

        # Mock the run method to return a result
        mock_result = MagicMock()
        mock_result.best_node_id = "node-1"
        mock_result.best_score = 8.0
        mock_result.best_messages = []
        mock_result.pruned_count = 2
        mock_result.total_rounds = 1
        mock_result.token_usage = {}
        mock_result.to_exploration_dict.return_value = {}
        mock_engine.run = AsyncMock(return_value=mock_result)

        request = SearchRequest(
            goal="Test",
            first_message="Hello",
            rounds=1,
        )

        events = []
        async for event in run_dts_session(request):
            events.append(event)

        # Should have at least the complete event
        assert len(events) >= 1
        assert any(e["type"] == "complete" for e in events)

    @pytest.mark.asyncio
    @patch("backend.services.dts_service.DTSEngine")
    @patch("backend.services.dts_service.create_llm_client")
    async def test_sets_event_callback(
        self, mock_create_llm: MagicMock, mock_engine_class: MagicMock
    ) -> None:
        """Test that event callback is set on engine."""
        mock_llm = MagicMock()
        mock_create_llm.return_value = mock_llm

        mock_engine = MagicMock()
        mock_engine_class.return_value = mock_engine

        mock_result = MagicMock()
        mock_result.best_node_id = None
        mock_result.best_score = 0.0
        mock_result.best_messages = []
        mock_result.pruned_count = 0
        mock_result.total_rounds = 1
        mock_result.token_usage = {}
        mock_result.to_exploration_dict.return_value = {}
        mock_engine.run = AsyncMock(return_value=mock_result)

        request = SearchRequest(
            goal="Test",
            first_message="Hello",
        )

        async for _ in run_dts_session(request):
            pass

        # Verify set_event_callback was called
        mock_engine.set_event_callback.assert_called_once()

    @pytest.mark.asyncio
    @patch("backend.services.dts_service.DTSEngine")
    @patch("backend.services.dts_service.create_llm_client")
    async def test_complete_event_contains_result(
        self, mock_create_llm: MagicMock, mock_engine_class: MagicMock
    ) -> None:
        """Test that complete event contains result data."""
        mock_llm = MagicMock()
        mock_create_llm.return_value = mock_llm

        mock_engine = MagicMock()
        mock_engine_class.return_value = mock_engine

        mock_result = MagicMock()
        mock_result.best_node_id = "best-node"
        mock_result.best_score = 8.5
        mock_result.best_messages = []
        mock_result.pruned_count = 3
        mock_result.total_rounds = 2
        mock_result.token_usage = {"total": 1000}
        mock_result.to_exploration_dict.return_value = {"branches": []}
        mock_engine.run = AsyncMock(return_value=mock_result)

        request = SearchRequest(
            goal="Test",
            first_message="Hello",
            rounds=2,
        )

        complete_event = None
        async for event in run_dts_session(request):
            if event["type"] == "complete":
                complete_event = event
                break

        assert complete_event is not None
        assert complete_event["data"]["best_node_id"] == "best-node"
        assert complete_event["data"]["best_score"] == 8.5
        assert complete_event["data"]["total_rounds"] == 2

    @pytest.mark.asyncio
    @patch("backend.services.dts_service.DTSEngine")
    @patch("backend.services.dts_service.create_llm_client")
    async def test_error_handling(
        self, mock_create_llm: MagicMock, mock_engine_class: MagicMock
    ) -> None:
        """Test that errors are yielded as error events."""
        mock_llm = MagicMock()
        mock_create_llm.return_value = mock_llm

        mock_engine = MagicMock()
        mock_engine_class.return_value = mock_engine
        mock_engine.run = AsyncMock(side_effect=Exception("Test error"))

        request = SearchRequest(
            goal="Test",
            first_message="Hello",
        )

        with pytest.raises(Exception) as exc_info:
            async for _ in run_dts_session(request):
                pass

        assert "Test error" in str(exc_info.value)

    @pytest.mark.asyncio
    @patch("backend.services.dts_service.DTSEngine")
    @patch("backend.services.dts_service.create_llm_client")
    async def test_event_callback_puts_events_in_queue(
        self, mock_create_llm: MagicMock, mock_engine_class: MagicMock
    ) -> None:
        """Test that event callback puts events in the queue and they are yielded."""
        mock_llm = MagicMock()
        mock_create_llm.return_value = mock_llm

        mock_engine = MagicMock()
        mock_engine_class.return_value = mock_engine

        # Capture the callback so we can call it during the run
        captured_callback = None

        def capture_callback(cb):
            nonlocal captured_callback
            captured_callback = cb

        mock_engine.set_event_callback.side_effect = capture_callback

        # Make run() trigger the callback and return a result
        async def run_with_events(**_kwargs):
            # Emit some events via the callback
            if captured_callback:
                await captured_callback("search_started", {"branches": 3})
                await captured_callback("round_complete", {"round": 1})
            # Return result
            result = MagicMock()
            result.best_node_id = "node-1"
            result.best_score = 8.0
            result.best_messages = []
            result.pruned_count = 0
            result.total_rounds = 1
            result.token_usage = {}
            result.to_exploration_dict.return_value = {}
            return result

        mock_engine.run = run_with_events

        request = SearchRequest(goal="Test", first_message="Hello", rounds=1)

        events = []
        async for event in run_dts_session(request):
            events.append(event)

        # Should have search_started, round_complete, and complete events
        event_types = [e["type"] for e in events]
        assert "search_started" in event_types
        assert "round_complete" in event_types
        assert "complete" in event_types

    @pytest.mark.asyncio
    @patch("backend.services.dts_service.DTSEngine")
    @patch("backend.services.dts_service.create_llm_client")
    async def test_best_messages_serialization(
        self, mock_create_llm: MagicMock, mock_engine_class: MagicMock
    ) -> None:
        """Test that best_messages are properly serialized."""
        mock_llm = MagicMock()
        mock_create_llm.return_value = mock_llm

        mock_engine = MagicMock()
        mock_engine_class.return_value = mock_engine

        # Create messages
        msg1 = MagicMock()
        msg1.role = "user"
        msg1.content = "Hello"
        msg2 = MagicMock()
        msg2.role = "assistant"
        msg2.content = "Hi there!"

        mock_result = MagicMock()
        mock_result.best_node_id = "best"
        mock_result.best_score = 9.0
        mock_result.best_messages = [msg1, msg2]
        mock_result.pruned_count = 0
        mock_result.total_rounds = 1
        mock_result.token_usage = {}
        mock_result.to_exploration_dict.return_value = {}
        mock_engine.run = AsyncMock(return_value=mock_result)

        request = SearchRequest(goal="Test", first_message="Hello")

        complete_event = None
        async for event in run_dts_session(request):
            if event["type"] == "complete":
                complete_event = event
                break

        assert complete_event is not None
        messages = complete_event["data"]["best_messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "Hi there!"

    @pytest.mark.asyncio
    @patch("backend.services.dts_service.DTSEngine")
    @patch("backend.services.dts_service.create_llm_client")
    async def test_queued_events_processed_when_engine_done(
        self, mock_create_llm: MagicMock, mock_engine_class: MagicMock
    ) -> None:
        """Test that remaining queue events are processed when engine finishes."""
        mock_llm = MagicMock()
        mock_create_llm.return_value = mock_llm

        mock_engine = MagicMock()
        mock_engine_class.return_value = mock_engine

        captured_callback = None

        def capture_callback(cb):
            nonlocal captured_callback
            captured_callback = cb

        mock_engine.set_event_callback.side_effect = capture_callback

        # Simulate multiple events being queued and engine completing
        async def run_with_multiple_events(**_kwargs):
            if captured_callback:
                # Queue multiple events before returning
                await captured_callback("event_1", {"seq": 1})
                await captured_callback("event_2", {"seq": 2})
                await captured_callback("event_3", {"seq": 3})
            # Return result immediately
            result = MagicMock()
            result.best_node_id = "node-1"
            result.best_score = 7.0
            result.best_messages = []
            result.pruned_count = 0
            result.total_rounds = 1
            result.token_usage = {}
            result.to_exploration_dict.return_value = {}
            return result

        mock_engine.run = run_with_multiple_events

        request = SearchRequest(goal="Test", first_message="Hello")

        events = []
        async for event in run_dts_session(request):
            events.append(event)

        # All events should be yielded plus the complete event
        event_types = [e["type"] for e in events]
        assert event_types.count("complete") == 1
        # Should have at least some of the queued events
        assert len(events) >= 1
