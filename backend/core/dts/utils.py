"""Shared utilities for the DTS module."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Callable, Coroutine

if TYPE_CHECKING:
    from backend.llm.types import Message


def log_phase(
    logger: logging.Logger,
    phase: str,
    message: str,
    indent: int = 0,
) -> None:
    """
    Log a formatted DTS phase message.

    Args:
        logger: The logger instance to use.
        phase: The phase name (e.g., "INIT", "EXPAND", "SCORE").
        message: The log message.
        indent: Number of indentation levels (2 spaces each).
    """
    prefix = "  " * indent
    logger.info(f"[DTS:{phase}] {prefix}{message}")


def format_message_history(messages: list[Message]) -> str:
    """
    Format conversation messages into a readable string.

    Args:
        messages: List of Message objects with role and content.

    Returns:
        Formatted string with each message on a new line.
    """
    lines = []
    for msg in messages:
        role = msg.role.capitalize()
        content = msg.content or ""
        lines.append(f"{role}: {content}")
    return "\n\n".join(lines)


async def emit_event(
    callback: Callable[..., Coroutine[Any, Any, None] | Any] | None,
    event_type: str,
    data: dict[str, Any],
    logger: logging.Logger | None = None,
) -> None:
    """
    Safely emit an event via callback if provided.

    Handles both sync and async callbacks. Logs warnings on errors.

    Args:
        callback: Optional event callback function.
        event_type: Type of event being emitted.
        data: Event data dictionary.
        logger: Optional logger for error reporting.
    """
    if callback is None:
        return

    try:
        result = callback(event_type, data)
        if asyncio.iscoroutine(result):
            await result
    except Exception as e:
        if logger:
            logger.warning(f"Event callback error: {e}")


def create_event_emitter(
    callback: Callable[..., Coroutine[Any, Any, None] | Any] | None,
    logger: logging.Logger,
) -> Callable[[str, dict[str, Any]], None]:
    """
    Create a fire-and-forget event emitter function.

    Used by components to emit events without awaiting. The returned function
    creates an asyncio task to handle the event asynchronously.

    Args:
        callback: Optional async event callback function.
        logger: Logger for error reporting.

    Returns:
        A sync function that emits events via asyncio.create_task.
    """

    def emit(event_type: str, data: dict[str, Any]) -> None:
        if callback is not None:
            asyncio.create_task(emit_event(callback, event_type, data, logger))

    return emit
