class LLMError(Exception):
    """Base exception for LLM errors."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class AuthenticationError(LLMError):
    """Invalid or missing API key."""

    pass


class RateLimitError(LLMError):
    """Rate limit exceeded."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message, status_code=429)
        self.retry_after = retry_after


class InvalidRequestError(LLMError):
    """Invalid request parameters."""

    pass


class ModelNotFoundError(LLMError):
    """Requested model does not exist."""

    pass


class ContentFilterError(LLMError):
    """Content was filtered by safety systems."""

    pass


class ContextLengthError(LLMError):
    """Input exceeded model's context length."""

    pass


class JSONParseError(LLMError):
    """Response content could not be parsed as valid JSON."""

    pass
