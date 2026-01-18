"""Tests for backend/llm/errors.py."""

from backend.llm.errors import (
    AuthenticationError,
    ConnectionError,
    ContentFilterError,
    ContextLengthError,
    InvalidRequestError,
    JSONParseError,
    LLMError,
    ModelNotFoundError,
    RateLimitError,
    ServerError,
    TimeoutError,
)

# -----------------------------------------------------------------------------
# LLMError Tests
# -----------------------------------------------------------------------------


class TestLLMError:
    """Tests for the base LLMError class."""

    def test_llm_error_message_only(self) -> None:
        """Test LLMError with message only."""
        error = LLMError("Something went wrong")
        assert str(error) == "Something went wrong"
        assert error.message == "Something went wrong"
        assert error.status_code is None

    def test_llm_error_with_status_code(self) -> None:
        """Test LLMError with status code."""
        error = LLMError("Server error", status_code=500)
        assert error.message == "Server error"
        assert error.status_code == 500

    def test_llm_error_inheritance(self) -> None:
        """Test LLMError inherits from Exception."""
        error = LLMError("Test")
        assert isinstance(error, Exception)


# -----------------------------------------------------------------------------
# Authentication Error Tests
# -----------------------------------------------------------------------------


class TestAuthenticationError:
    """Tests for AuthenticationError."""

    def test_authentication_error(self) -> None:
        """Test AuthenticationError creation."""
        error = AuthenticationError("Invalid API key")
        assert str(error) == "Invalid API key"
        assert isinstance(error, LLMError)

    def test_authentication_error_with_status(self) -> None:
        """Test AuthenticationError with status code."""
        error = AuthenticationError("Unauthorized", status_code=401)
        assert error.status_code == 401


# -----------------------------------------------------------------------------
# Rate Limit Error Tests
# -----------------------------------------------------------------------------


class TestRateLimitError:
    """Tests for RateLimitError."""

    def test_rate_limit_error_basic(self) -> None:
        """Test RateLimitError with message only."""
        error = RateLimitError("Rate limit exceeded")
        assert str(error) == "Rate limit exceeded"
        assert error.status_code == 429
        assert error.retry_after is None

    def test_rate_limit_error_with_retry(self) -> None:
        """Test RateLimitError with retry_after."""
        error = RateLimitError("Too many requests", retry_after=30.0)
        assert error.retry_after == 30.0
        assert error.status_code == 429


# -----------------------------------------------------------------------------
# Invalid Request Error Tests
# -----------------------------------------------------------------------------


class TestInvalidRequestError:
    """Tests for InvalidRequestError."""

    def test_invalid_request_error(self) -> None:
        """Test InvalidRequestError creation."""
        error = InvalidRequestError("Bad parameters")
        assert str(error) == "Bad parameters"
        assert isinstance(error, LLMError)


# -----------------------------------------------------------------------------
# Model Not Found Error Tests
# -----------------------------------------------------------------------------


class TestModelNotFoundError:
    """Tests for ModelNotFoundError."""

    def test_model_not_found_error(self) -> None:
        """Test ModelNotFoundError creation."""
        error = ModelNotFoundError("Model 'gpt-5' not found", status_code=404)
        assert "gpt-5" in str(error)
        assert error.status_code == 404


# -----------------------------------------------------------------------------
# Content Filter Error Tests
# -----------------------------------------------------------------------------


class TestContentFilterError:
    """Tests for ContentFilterError."""

    def test_content_filter_error(self) -> None:
        """Test ContentFilterError creation."""
        error = ContentFilterError("Content blocked by safety systems")
        assert "blocked" in str(error)
        assert isinstance(error, LLMError)


# -----------------------------------------------------------------------------
# Context Length Error Tests
# -----------------------------------------------------------------------------


class TestContextLengthError:
    """Tests for ContextLengthError."""

    def test_context_length_error(self) -> None:
        """Test ContextLengthError creation."""
        error = ContextLengthError("Input exceeds 128k tokens", status_code=400)
        assert "128k" in str(error)
        assert error.status_code == 400


# -----------------------------------------------------------------------------
# JSON Parse Error Tests
# -----------------------------------------------------------------------------


class TestJSONParseError:
    """Tests for JSONParseError."""

    def test_json_parse_error(self) -> None:
        """Test JSONParseError creation."""
        error = JSONParseError("Invalid JSON: missing closing brace")
        assert "Invalid JSON" in str(error)
        assert isinstance(error, LLMError)


# -----------------------------------------------------------------------------
# Server Error Tests
# -----------------------------------------------------------------------------


class TestServerError:
    """Tests for ServerError."""

    def test_server_error(self) -> None:
        """Test ServerError creation."""
        error = ServerError("Internal server error", status_code=500)
        assert error.status_code == 500
        assert isinstance(error, LLMError)


# -----------------------------------------------------------------------------
# Timeout Error Tests
# -----------------------------------------------------------------------------


class TestTimeoutError:
    """Tests for TimeoutError."""

    def test_timeout_error(self) -> None:
        """Test TimeoutError creation."""
        error = TimeoutError("Request timed out after 60s")
        assert "60s" in str(error)
        assert isinstance(error, LLMError)


# -----------------------------------------------------------------------------
# Connection Error Tests
# -----------------------------------------------------------------------------


class TestConnectionError:
    """Tests for ConnectionError."""

    def test_connection_error(self) -> None:
        """Test ConnectionError creation."""
        error = ConnectionError("Failed to connect to server")
        assert "connect" in str(error)
        assert isinstance(error, LLMError)


# -----------------------------------------------------------------------------
# Error Hierarchy Tests
# -----------------------------------------------------------------------------


class TestErrorHierarchy:
    """Test the error class hierarchy."""

    def test_all_errors_inherit_from_llm_error(self) -> None:
        """Verify all error types inherit from LLMError."""
        errors = [
            AuthenticationError("test"),
            RateLimitError("test"),
            InvalidRequestError("test"),
            ModelNotFoundError("test"),
            ContentFilterError("test"),
            ContextLengthError("test"),
            JSONParseError("test"),
            ServerError("test"),
            TimeoutError("test"),
            ConnectionError("test"),
        ]
        for error in errors:
            assert isinstance(error, LLMError)
            assert isinstance(error, Exception)

    def test_catching_llm_error_catches_all(self) -> None:
        """Verify catching LLMError catches all subtypes."""
        try:
            raise RateLimitError("Test")
        except LLMError as e:
            assert str(e) == "Test"

        try:
            raise JSONParseError("Parse failed")
        except LLMError as e:
            assert "Parse failed" in str(e)
