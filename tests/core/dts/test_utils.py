"""Tests for backend/core/dts/utils.py and retry.py."""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.core.dts.retry import llm_retry
from backend.core.dts.utils import (
    create_event_emitter,
    emit_event,
    format_message_history,
    log_phase,
)
from backend.llm.errors import JSONParseError, RateLimitError, ServerError
from backend.llm.types import Message

# -----------------------------------------------------------------------------
# log_phase Tests
# -----------------------------------------------------------------------------


class TestLogPhase:
    """Tests for log_phase function."""

    def test_log_phase_basic(self, caplog) -> None:
        """Test basic phase logging."""
        logger = logging.getLogger("test")
        with caplog.at_level(logging.INFO):
            log_phase(logger, "INIT", "Starting initialization")

        assert "[DTS:INIT]" in caplog.text
        assert "Starting initialization" in caplog.text

    def test_log_phase_with_indent(self, caplog) -> None:
        """Test phase logging with indentation."""
        logger = logging.getLogger("test")
        with caplog.at_level(logging.INFO):
            log_phase(logger, "EXPAND", "Processing node", indent=2)

        assert "[DTS:EXPAND]" in caplog.text
        assert "    Processing node" in caplog.text  # 4 spaces (2 * 2)

    def test_log_phase_different_phases(self, caplog) -> None:
        """Test logging different phases."""
        logger = logging.getLogger("test")
        phases = ["INIT", "EXPAND", "SCORE", "PRUNE", "DONE"]

        with caplog.at_level(logging.INFO):
            for phase in phases:
                log_phase(logger, phase, f"Phase {phase}")

        for phase in phases:
            assert f"[DTS:{phase}]" in caplog.text


# -----------------------------------------------------------------------------
# format_message_history Tests
# -----------------------------------------------------------------------------


class TestFormatMessageHistory:
    """Tests for format_message_history function."""

    def test_format_single_message(self) -> None:
        """Test formatting a single message."""
        messages = [Message.user("Hello")]
        result = format_message_history(messages)

        assert "User: Hello" in result

    def test_format_multiple_messages(self) -> None:
        """Test formatting multiple messages."""
        messages = [
            Message.user("Hello"),
            Message.assistant("Hi there!"),
            Message.user("How are you?"),
        ]
        result = format_message_history(messages)

        assert "User: Hello" in result
        assert "Assistant: Hi there!" in result
        assert "User: How are you?" in result

    def test_format_with_system_message(self) -> None:
        """Test formatting with system message."""
        messages = [
            Message.system("You are helpful"),
            Message.user("Hello"),
        ]
        result = format_message_history(messages)

        assert "System: You are helpful" in result
        assert "User: Hello" in result

    def test_format_empty_messages(self) -> None:
        """Test formatting empty message list."""
        result = format_message_history([])
        assert result == ""

    def test_format_message_with_none_content(self) -> None:
        """Test formatting message with None content."""
        messages = [Message(role="assistant", content=None)]
        result = format_message_history(messages)

        assert "Assistant:" in result

    def test_format_preserves_order(self) -> None:
        """Test that message order is preserved."""
        messages = [
            Message.user("First"),
            Message.assistant("Second"),
            Message.user("Third"),
        ]
        result = format_message_history(messages)

        first_pos = result.find("First")
        second_pos = result.find("Second")
        third_pos = result.find("Third")

        assert first_pos < second_pos < third_pos


# -----------------------------------------------------------------------------
# emit_event Tests
# -----------------------------------------------------------------------------


class TestEmitEvent:
    """Tests for emit_event function."""

    @pytest.mark.asyncio
    async def test_emit_with_async_callback(self) -> None:
        """Test emitting event with async callback."""
        callback = AsyncMock()
        await emit_event(callback, "test_event", {"key": "value"})

        callback.assert_called_once_with("test_event", {"key": "value"})

    @pytest.mark.asyncio
    async def test_emit_with_sync_callback(self) -> None:
        """Test emitting event with sync callback."""
        callback = MagicMock()
        await emit_event(callback, "test_event", {"key": "value"})

        callback.assert_called_once_with("test_event", {"key": "value"})

    @pytest.mark.asyncio
    async def test_emit_with_none_callback(self) -> None:
        """Test emitting event with None callback does nothing."""
        # Should not raise
        await emit_event(None, "test_event", {"key": "value"})

    @pytest.mark.asyncio
    async def test_emit_handles_callback_error(self, caplog) -> None:
        """Test that callback errors are caught and logged."""
        callback = MagicMock(side_effect=Exception("Callback failed"))
        logger = logging.getLogger("test")

        with caplog.at_level(logging.WARNING):
            await emit_event(callback, "test_event", {}, logger)

        assert "Event callback error" in caplog.text

    @pytest.mark.asyncio
    async def test_emit_handles_async_callback_error(self, caplog) -> None:
        """Test that async callback errors are caught and logged."""
        callback = AsyncMock(side_effect=Exception("Async callback failed"))
        logger = logging.getLogger("test")

        with caplog.at_level(logging.WARNING):
            await emit_event(callback, "test_event", {}, logger)

        assert "Event callback error" in caplog.text


# -----------------------------------------------------------------------------
# create_event_emitter Tests
# -----------------------------------------------------------------------------


class TestCreateEventEmitter:
    """Tests for create_event_emitter function."""

    @pytest.mark.asyncio
    async def test_creates_working_emitter(self) -> None:
        """Test that created emitter works."""
        callback = AsyncMock()
        logger = logging.getLogger("test")
        emitter = create_event_emitter(callback, logger)

        emitter("test_event", {"data": 123})

        # Yield to event loop to let fire-and-forget task run (no actual delay)
        await asyncio.sleep(0)

        callback.assert_called_once_with("test_event", {"data": 123})

    @pytest.mark.asyncio
    async def test_emitter_with_none_callback(self) -> None:
        """Test emitter does nothing with None callback."""
        logger = logging.getLogger("test")
        emitter = create_event_emitter(None, logger)

        # Should not raise
        emitter("test_event", {"data": 123})

    @pytest.mark.asyncio
    async def test_emitter_is_fire_and_forget(self) -> None:
        """Test that emitter doesn't block."""
        callback = AsyncMock()
        logger = logging.getLogger("test")
        emitter = create_event_emitter(callback, logger)

        # Should return immediately
        emitter("event1", {})
        emitter("event2", {})

        # Yield to event loop to let fire-and-forget tasks run (no actual delay)
        await asyncio.sleep(0)

        assert callback.call_count == 2


# -----------------------------------------------------------------------------
# llm_retry Tests
# -----------------------------------------------------------------------------


class TestLLMRetry:
    """Tests for llm_retry decorator."""

    @pytest.mark.asyncio
    async def test_retry_on_rate_limit(self) -> None:
        """Test retry on RateLimitError."""
        call_count = 0

        @llm_retry(max_attempts=3)
        async def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RateLimitError("Rate limited")
            return "success"

        result = await flaky_function()

        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_on_server_error(self) -> None:
        """Test retry on ServerError."""
        call_count = 0

        @llm_retry(max_attempts=3)
        async def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ServerError("Server error", status_code=500)
            return "success"

        result = await flaky_function()

        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_retry_on_json_parse_error(self) -> None:
        """Test retry on JSONParseError."""
        call_count = 0

        @llm_retry(max_attempts=3)
        async def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise JSONParseError("Invalid JSON")
            return "success"

        result = await flaky_function()

        assert result == "success"

    @pytest.mark.asyncio
    async def test_retry_exhausted(self) -> None:
        """Test that error is raised after max attempts."""

        @llm_retry(max_attempts=2)
        async def always_fails():
            raise RateLimitError("Always rate limited")

        with pytest.raises(RateLimitError):
            await always_fails()

    @pytest.mark.asyncio
    async def test_no_retry_on_other_errors(self) -> None:
        """Test that non-retryable errors are not retried."""
        call_count = 0

        @llm_retry(max_attempts=3)
        async def raises_value_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("Not retryable")

        with pytest.raises(ValueError):
            await raises_value_error()

        assert call_count == 1  # Only called once

    @pytest.mark.asyncio
    async def test_success_on_first_try(self) -> None:
        """Test successful function doesn't retry."""
        call_count = 0

        @llm_retry(max_attempts=3)
        async def always_succeeds():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await always_succeeds()

        assert result == "success"
        assert call_count == 1
