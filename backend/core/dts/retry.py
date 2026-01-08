"""Shared retry logic for DTS LLM calls."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import logging

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from backend.llm.errors import (
    ConnectionError,
    JSONParseError,
    RateLimitError,
    ServerError,
    TimeoutError,
)
from backend.utils.logging import logger


# -----------------------------------------------------------------------------
# Retry Decorator
# -----------------------------------------------------------------------------
def llm_retry(max_attempts: int = 3):
    """
    Standard retry decorator for LLM calls.

    Uses exponential backoff for transient errors:
    - RateLimitError (429)
    - ServerError (5xx)
    - TimeoutError
    - ConnectionError
    - JSONParseError (empty or malformed responses)

    Args:
        max_attempts: Maximum number of attempts before giving up.

    Returns:
        A tenacity retry decorator configured for LLM calls.
    """
    return retry(
        retry=retry_if_exception_type(
            (RateLimitError, ServerError, TimeoutError, ConnectionError, JSONParseError)
        ),
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
