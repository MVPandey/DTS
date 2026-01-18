"""Tests for backend/llm/client.py."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import APIError, AsyncOpenAI

from backend.llm.client import LLM
from backend.llm.errors import (
    AuthenticationError,
    ContentFilterError,
    ContextLengthError,
    InvalidRequestError,
    JSONParseError,
    LLMError,
    ModelNotFoundError,
    RateLimitError,
    ServerError,
)
from backend.llm.tools import Tool, ToolRegistry
from backend.llm.types import Completion, Message

# -----------------------------------------------------------------------------
# Mock Response Helpers
# -----------------------------------------------------------------------------


def create_mock_response(
    content: str | None = "Test response",
    tool_calls: list | None = None,
    finish_reason: str = "stop",
    model: str = "test-model",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> MagicMock:
    """Create a mock OpenAI response."""
    mock_message = MagicMock()
    mock_message.content = content
    mock_message.tool_calls = tool_calls

    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_choice.finish_reason = finish_reason

    mock_usage = MagicMock()
    mock_usage.prompt_tokens = prompt_tokens
    mock_usage.completion_tokens = completion_tokens
    mock_usage.total_tokens = prompt_tokens + completion_tokens

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = mock_usage
    mock_response.model = model

    return mock_response


def create_mock_tool_call(
    id: str = "tc_123",
    name: str = "test_tool",
    arguments: str = "{}",
) -> MagicMock:
    """Create a mock tool call."""
    mock_function = MagicMock()
    mock_function.name = name
    mock_function.arguments = arguments

    mock_tc = MagicMock()
    mock_tc.id = id
    mock_tc.type = "function"
    mock_tc.function = mock_function

    return mock_tc


# -----------------------------------------------------------------------------
# LLM Initialization Tests
# -----------------------------------------------------------------------------


class TestLLMInit:
    """Tests for LLM initialization."""

    @patch.object(AsyncOpenAI, "__init__", return_value=None)
    def test_llm_init_with_defaults(self, mock_init: MagicMock) -> None:
        """Test LLM initialization with default values."""
        LLM(api_key="test-key")
        mock_init.assert_called_once()

    @patch.object(AsyncOpenAI, "__init__", return_value=None)
    def test_llm_init_with_custom_values(self, _mock_init: MagicMock) -> None:
        """Test LLM initialization with custom values."""
        llm = LLM(
            api_key="test-key",
            base_url="https://custom.api.com/v1",
            model="custom-model",
            timeout=60.0,
            max_retries=5,
        )
        assert llm._default_model == "custom-model"


# -----------------------------------------------------------------------------
# LLM Complete Tests
# -----------------------------------------------------------------------------


class TestLLMComplete:
    """Tests for LLM.complete() method."""

    @pytest.fixture
    def mock_client(self) -> MagicMock:
        """Create a mock AsyncOpenAI client."""
        return MagicMock()

    @pytest.fixture
    def llm_with_mock(self, mock_client: MagicMock) -> LLM:
        """Create LLM with mocked client."""
        with patch.object(AsyncOpenAI, "__init__", return_value=None):
            llm = LLM(api_key="test-key", model="test-model")
            llm._client = mock_client
            return llm

    @pytest.mark.asyncio
    async def test_complete_with_string_message(self, llm_with_mock: LLM) -> None:
        """Test complete with a simple string message."""
        llm_with_mock._client.chat.completions.create = AsyncMock(
            return_value=create_mock_response(content="Hello!")
        )

        result = await llm_with_mock.complete("Hi there")

        assert isinstance(result, Completion)
        assert result.content == "Hello!"
        assert result.model == "test-model"

    @pytest.mark.asyncio
    async def test_complete_with_message_object(self, llm_with_mock: LLM) -> None:
        """Test complete with a Message object."""
        llm_with_mock._client.chat.completions.create = AsyncMock(
            return_value=create_mock_response()
        )

        result = await llm_with_mock.complete(Message.user("Hello"))

        assert isinstance(result, Completion)

    @pytest.mark.asyncio
    async def test_complete_with_message_list(self, llm_with_mock: LLM) -> None:
        """Test complete with a list of messages."""
        llm_with_mock._client.chat.completions.create = AsyncMock(
            return_value=create_mock_response()
        )

        messages = [
            Message.system("You are helpful"),
            Message.user("Hi"),
        ]
        result = await llm_with_mock.complete(messages)

        assert isinstance(result, Completion)

    @pytest.mark.asyncio
    async def test_complete_no_model_raises(self, llm_with_mock: LLM) -> None:
        """Test complete raises when no model specified."""
        llm_with_mock._default_model = None

        with pytest.raises(InvalidRequestError) as exc_info:
            await llm_with_mock.complete("Hi")

        assert "No model specified" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_complete_with_structured_output(self, llm_with_mock: LLM) -> None:
        """Test complete with structured JSON output."""
        json_response = '{"name": "John", "age": 30}'
        llm_with_mock._client.chat.completions.create = AsyncMock(
            return_value=create_mock_response(content=json_response)
        )

        result = await llm_with_mock.complete("Get user info", structured_output=True)

        assert result.data == {"name": "John", "age": 30}

    @pytest.mark.asyncio
    async def test_complete_structured_output_from_markdown(self, llm_with_mock: LLM) -> None:
        """Test extracting JSON from markdown code blocks."""
        markdown_response = '```json\n{"key": "value"}\n```'
        llm_with_mock._client.chat.completions.create = AsyncMock(
            return_value=create_mock_response(content=markdown_response)
        )

        result = await llm_with_mock.complete("Get data", structured_output=True)

        assert result.data == {"key": "value"}

    @pytest.mark.asyncio
    async def test_complete_structured_output_retries(self, llm_with_mock: LLM) -> None:
        """Test retries on invalid JSON."""
        responses = [
            create_mock_response(content="not json"),
            create_mock_response(content="still not json"),
            create_mock_response(content='{"valid": true}'),
        ]
        llm_with_mock._client.chat.completions.create = AsyncMock(side_effect=responses)

        result = await llm_with_mock.complete(
            "Get data", structured_output=True, max_json_retries=3
        )

        assert result.data == {"valid": True}

    @pytest.mark.asyncio
    async def test_complete_structured_output_empty_fails(self, llm_with_mock: LLM) -> None:
        """Test that empty response raises JSONParseError."""
        llm_with_mock._client.chat.completions.create = AsyncMock(
            return_value=create_mock_response(content="")
        )

        with pytest.raises(JSONParseError):
            await llm_with_mock.complete("Get data", structured_output=True, max_json_retries=1)

    @pytest.mark.asyncio
    async def test_complete_with_tools(self, llm_with_mock: LLM) -> None:
        """Test complete with tool calls."""
        tool_call = create_mock_tool_call(
            id="tc_1",
            name="get_weather",
            arguments='{"city": "London"}',
        )
        llm_with_mock._client.chat.completions.create = AsyncMock(
            return_value=create_mock_response(content=None, tool_calls=[tool_call])
        )

        tools = [{"type": "function", "function": {"name": "get_weather"}}]
        result = await llm_with_mock.complete("What's the weather?", tools=tools)

        assert result.has_tool_calls
        assert result.message.tool_calls is not None
        assert result.message.tool_calls[0].function.name == "get_weather"


# -----------------------------------------------------------------------------
# LLM Stream Tests
# -----------------------------------------------------------------------------


class TestLLMStream:
    """Tests for LLM.stream() method."""

    @pytest.fixture
    def llm_with_mock(self) -> LLM:
        """Create LLM with mocked client."""
        with patch.object(AsyncOpenAI, "__init__", return_value=None):
            llm = LLM(api_key="test-key", model="test-model")
            llm._client = MagicMock()
            return llm

    @pytest.mark.asyncio
    async def test_stream_yields_content(self, llm_with_mock: LLM) -> None:
        """Test streaming yields content chunks."""

        async def mock_stream():
            for chunk_content in ["Hello", " ", "World"]:
                chunk = MagicMock()
                chunk.choices = [MagicMock()]
                chunk.choices[0].delta.content = chunk_content
                yield chunk

        llm_with_mock._client.chat.completions.create = AsyncMock(return_value=mock_stream())

        chunks = []
        async for chunk in llm_with_mock.stream("Hi"):
            chunks.append(chunk)

        assert chunks == ["Hello", " ", "World"]

    @pytest.mark.asyncio
    async def test_stream_no_model_raises(self, llm_with_mock: LLM) -> None:
        """Test stream raises when no model specified."""
        llm_with_mock._default_model = None

        with pytest.raises(InvalidRequestError):
            async for _ in llm_with_mock.stream("Hi"):
                pass


# -----------------------------------------------------------------------------
# LLM Run Tests
# -----------------------------------------------------------------------------


class TestLLMRun:
    """Tests for LLM.run() method with automatic tool execution."""

    @pytest.fixture
    def llm_with_mock(self) -> LLM:
        """Create LLM with mocked client."""
        with patch.object(AsyncOpenAI, "__init__", return_value=None):
            llm = LLM(api_key="test-key", model="test-model")
            llm._client = MagicMock()
            return llm

    @pytest.mark.asyncio
    async def test_run_without_tools(self, llm_with_mock: LLM) -> None:
        """Test run without tools just calls complete."""
        llm_with_mock._client.chat.completions.create = AsyncMock(
            return_value=create_mock_response(content="Done")
        )

        result = await llm_with_mock.run("Do something")

        assert result.content == "Done"

    @pytest.mark.asyncio
    async def test_run_with_tool_execution(self, llm_with_mock: LLM) -> None:
        """Test run executes tools and continues conversation."""

        def get_time() -> str:
            """Get the current time."""
            return "12:00 PM"

        registry = ToolRegistry()
        registry.add(Tool(get_time))

        # First call returns tool call, second returns final response
        tool_call = create_mock_tool_call(id="tc_1", name="get_time", arguments="{}")
        responses = [
            create_mock_response(content=None, tool_calls=[tool_call]),
            create_mock_response(content="The time is 12:00 PM"),
        ]
        llm_with_mock._client.chat.completions.create = AsyncMock(side_effect=responses)

        result = await llm_with_mock.run("What time is it?", tools=registry)

        assert result.content == "The time is 12:00 PM"
        assert llm_with_mock._client.chat.completions.create.call_count == 2


# -----------------------------------------------------------------------------
# Error Mapping Tests
# -----------------------------------------------------------------------------


class TestLLMErrorMapping:
    """Tests for error mapping from OpenAI errors to custom errors."""

    @pytest.fixture
    def llm_with_mock(self) -> LLM:
        """Create LLM with mocked client."""
        with patch.object(AsyncOpenAI, "__init__", return_value=None):
            llm = LLM(api_key="test-key", model="test-model")
            llm._client = MagicMock()
            return llm

    @pytest.mark.asyncio
    async def test_authentication_error_mapping(self, llm_with_mock: LLM) -> None:
        """Test OpenAI AuthenticationError maps correctly."""
        from openai import AuthenticationError as OpenAIAuthError

        llm_with_mock._client.chat.completions.create = AsyncMock(
            side_effect=OpenAIAuthError(
                message="Invalid API key",
                response=MagicMock(status_code=401),
                body=None,
            )
        )

        with pytest.raises(AuthenticationError):
            await llm_with_mock.complete("Hi")

    @pytest.mark.asyncio
    async def test_rate_limit_error_mapping(self, llm_with_mock: LLM) -> None:
        """Test OpenAI RateLimitError maps correctly."""
        from openai import RateLimitError as OpenAIRateLimitError

        llm_with_mock._client.chat.completions.create = AsyncMock(
            side_effect=OpenAIRateLimitError(
                message="Rate limit exceeded",
                response=MagicMock(status_code=429),
                body=None,
            )
        )

        with pytest.raises(RateLimitError):
            await llm_with_mock.complete("Hi")

    @pytest.mark.asyncio
    async def test_api_error_404_maps_to_model_not_found(self, llm_with_mock: LLM) -> None:
        """Test 404 APIError maps to ModelNotFoundError."""
        error = APIError(
            message="Model not found",
            request=MagicMock(),
            body=None,
        )
        error.status_code = 404  # type: ignore[attr-defined]

        llm_with_mock._client.chat.completions.create = AsyncMock(side_effect=error)

        with pytest.raises(ModelNotFoundError):
            await llm_with_mock.complete("Hi")

    @pytest.mark.asyncio
    async def test_api_error_400_context_length(self, llm_with_mock: LLM) -> None:
        """Test 400 with context_length maps correctly."""
        error = APIError(
            message="context_length exceeded",
            request=MagicMock(),
            body=None,
        )
        error.status_code = 400  # type: ignore[attr-defined]

        llm_with_mock._client.chat.completions.create = AsyncMock(side_effect=error)

        with pytest.raises(ContextLengthError):
            await llm_with_mock.complete("Hi")

    @pytest.mark.asyncio
    async def test_api_error_400_content_filter(self, llm_with_mock: LLM) -> None:
        """Test 400 with safety maps to ContentFilterError."""
        error = APIError(
            message="content blocked by safety systems",
            request=MagicMock(),
            body=None,
        )
        error.status_code = 400  # type: ignore[attr-defined]

        llm_with_mock._client.chat.completions.create = AsyncMock(side_effect=error)

        with pytest.raises(ContentFilterError):
            await llm_with_mock.complete("Hi")

    @pytest.mark.asyncio
    async def test_api_error_500_maps_to_server_error(self, llm_with_mock: LLM) -> None:
        """Test 5xx APIError maps to ServerError."""
        error = APIError(
            message="Internal server error",
            request=MagicMock(),
            body=None,
        )
        error.status_code = 500  # type: ignore[attr-defined]

        llm_with_mock._client.chat.completions.create = AsyncMock(side_effect=error)

        with pytest.raises(ServerError):
            await llm_with_mock.complete("Hi")


# -----------------------------------------------------------------------------
# Helper Method Tests
# -----------------------------------------------------------------------------


class TestLLMHelperMethods:
    """Tests for LLM helper methods."""

    @pytest.fixture
    def llm(self) -> LLM:
        """Create LLM instance."""
        with patch.object(AsyncOpenAI, "__init__", return_value=None):
            return LLM(api_key="test-key", model="test-model")

    def test_strip_reasoning_tags(self, llm: LLM) -> None:
        """Test stripping reasoning tags from content."""
        content = "Hello <think>internal reasoning</think> World"
        result = llm._strip_reasoning_tags(content)
        assert result == "Hello  World"

    def test_strip_multiple_reasoning_tags(self, llm: LLM) -> None:
        """Test stripping multiple reasoning tags."""
        content = "<think>first</think>text<reasoning>second</reasoning>end"
        result = llm._strip_reasoning_tags(content)
        assert result == "textend"

    def test_extract_json_from_code_block(self, llm: LLM) -> None:
        """Test extracting JSON from markdown code block."""
        content = '```json\n{"key": "value"}\n```'
        result = llm._extract_json(content)
        assert result == '{"key": "value"}'

    def test_extract_json_plain(self, llm: LLM) -> None:
        """Test extracting JSON that's already plain."""
        content = '{"key": "value"}'
        result = llm._extract_json(content)
        assert result == '{"key": "value"}'

    def test_extract_json_with_surrounding_text(self, llm: LLM) -> None:
        """Test extracting JSON with surrounding text."""
        content = 'Here is the result: {"data": [1, 2, 3]} that you requested'
        result = llm._extract_json(content)
        assert result == '{"data": [1, 2, 3]}'

    def test_build_extra_body_with_provider(self, llm: LLM) -> None:
        """Test building extra_body with provider."""
        result = llm._build_extra_body(provider="Fireworks", reasoning_enabled=None)
        assert result == {"provider": {"order": ["Fireworks"], "allow_fallbacks": True}}

    def test_build_extra_body_with_reasoning(self, llm: LLM) -> None:
        """Test building extra_body with reasoning enabled."""
        result = llm._build_extra_body(provider=None, reasoning_enabled=True)
        assert result == {"reasoning": {"enabled": True}}

    def test_prepare_messages_from_string(self, llm: LLM) -> None:
        """Test preparing messages from string."""
        result = llm._prepare_messages("Hello")
        assert result == [{"role": "user", "content": "Hello"}]

    def test_prepare_messages_from_message(self, llm: LLM) -> None:
        """Test preparing messages from Message object."""
        msg = Message.user("Hi")
        result = llm._prepare_messages(msg)
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_prepare_messages_from_list(self, llm: LLM) -> None:
        """Test preparing messages from list."""
        messages = [Message.system("Be helpful"), Message.user("Hi")]
        result = llm._prepare_messages(messages)
        assert len(result) == 2

    def test_parse_response_empty_choices_raises(self, llm: LLM) -> None:
        """Test parsing response with no choices raises."""
        response = MagicMock()
        response.choices = []

        with pytest.raises(LLMError) as exc_info:
            llm._parse_response(response)
        assert "no choices" in str(exc_info.value)

    def test_parse_response_with_usage(self, llm: LLM) -> None:
        """Test parsing response includes usage."""
        response = create_mock_response(prompt_tokens=100, completion_tokens=50)
        result = llm._parse_response(response)

        assert result.usage is not None
        assert result.usage.prompt_tokens == 100
        assert result.usage.completion_tokens == 50

    def test_parse_response_without_usage(self, llm: LLM) -> None:
        """Test parsing response without usage info."""
        response = create_mock_response()
        response.usage = None
        result = llm._parse_response(response)

        assert result.usage is None

    def test_build_extra_body_with_provider_list(self, llm: LLM) -> None:
        """Test building extra_body with provider list."""
        result = llm._build_extra_body(provider=["Fireworks", "Together"], reasoning_enabled=None)
        assert result == {"provider": {"order": ["Fireworks", "Together"], "allow_fallbacks": True}}

    def test_build_extra_body_with_existing(self, llm: LLM) -> None:
        """Test building extra_body merges with existing."""
        existing = {"custom_key": "value"}
        result = llm._build_extra_body(
            provider="Fireworks", reasoning_enabled=True, existing=existing
        )
        assert result["custom_key"] == "value"
        assert "provider" in result
        assert "reasoning" in result

    def test_build_request_params_all_options(self, llm: LLM) -> None:
        """Test building request params with all options."""
        result = llm._build_request_params(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.7,
            max_tokens=100,
            tools=[{"type": "function", "function": {"name": "test"}}],
            tool_choice="auto",
            stop=["END"],
            stream=True,
        )

        assert result["model"] == "test-model"
        assert result["temperature"] == 0.7
        assert result["max_tokens"] == 100
        assert result["tools"] is not None
        assert result["tool_choice"] == "auto"
        assert result["stop"] == ["END"]
        assert result["stream"] is True

    def test_map_api_error_401(self, llm: LLM) -> None:
        """Test mapping 401 APIError."""
        error = APIError(message="Unauthorized", request=MagicMock(), body=None)
        error.status_code = 401  # type: ignore[attr-defined]
        result = llm._map_api_error(error)
        assert isinstance(result, AuthenticationError)

    def test_map_api_error_429(self, llm: LLM) -> None:
        """Test mapping 429 APIError."""
        error = APIError(message="Rate limited", request=MagicMock(), body=None)
        error.status_code = 429  # type: ignore[attr-defined]
        result = llm._map_api_error(error)
        assert isinstance(result, RateLimitError)

    def test_map_api_error_400_generic(self, llm: LLM) -> None:
        """Test mapping 400 APIError without specific message."""
        error = APIError(message="Bad request", request=MagicMock(), body=None)
        error.status_code = 400  # type: ignore[attr-defined]
        result = llm._map_api_error(error)
        assert isinstance(result, InvalidRequestError)

    def test_map_api_error_unknown(self, llm: LLM) -> None:
        """Test mapping unknown APIError."""
        error = APIError(message="Unknown error", request=MagicMock(), body=None)
        error.status_code = 418  # type: ignore[attr-defined]
        result = llm._map_api_error(error)
        assert isinstance(result, LLMError)

    def test_map_api_error_no_status(self, llm: LLM) -> None:
        """Test mapping APIError without status code."""
        error = APIError(message="Unknown error", request=MagicMock(), body=None)
        # No status_code attribute
        result = llm._map_api_error(error)
        assert isinstance(result, LLMError)


# -----------------------------------------------------------------------------
# Additional Coverage Tests
# -----------------------------------------------------------------------------


class TestLLMAdditionalCoverage:
    """Additional tests to improve coverage."""

    @pytest.fixture
    def llm_with_mock(self) -> LLM:
        """Create LLM with mocked client."""
        with patch.object(AsyncOpenAI, "__init__", return_value=None):
            llm = LLM(api_key="test-key", model="test-model")
            llm._client = MagicMock()
            return llm

    @pytest.mark.asyncio
    async def test_complete_with_extra_body(self, llm_with_mock: LLM) -> None:
        """Test complete passes extra_body correctly."""
        llm_with_mock._client.chat.completions.create = AsyncMock(
            return_value=create_mock_response()
        )

        await llm_with_mock.complete("Hi", provider="Fireworks", reasoning_enabled=True)

        call_kwargs = llm_with_mock._client.chat.completions.create.call_args.kwargs
        assert "extra_body" in call_kwargs

    @pytest.mark.asyncio
    async def test_stream_with_extra_body(self, llm_with_mock: LLM) -> None:
        """Test stream passes extra_body correctly."""

        async def mock_stream():
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = "Hello"
            yield chunk

        llm_with_mock._client.chat.completions.create = AsyncMock(return_value=mock_stream())

        chunks = []
        async for chunk in llm_with_mock.stream("Hi", provider="Fireworks"):
            chunks.append(chunk)

        call_kwargs = llm_with_mock._client.chat.completions.create.call_args.kwargs
        assert "extra_body" in call_kwargs

    @pytest.mark.asyncio
    async def test_stream_auth_error(self, llm_with_mock: LLM) -> None:
        """Test stream handles authentication errors."""
        from openai import AuthenticationError as OpenAIAuthError

        llm_with_mock._client.chat.completions.create = AsyncMock(
            side_effect=OpenAIAuthError(
                message="Invalid key",
                response=MagicMock(status_code=401),
                body=None,
            )
        )

        with pytest.raises(AuthenticationError):
            async for _ in llm_with_mock.stream("Hi"):
                pass

    @pytest.mark.asyncio
    async def test_stream_rate_limit_error(self, llm_with_mock: LLM) -> None:
        """Test stream handles rate limit errors."""
        from openai import RateLimitError as OpenAIRateLimitError

        llm_with_mock._client.chat.completions.create = AsyncMock(
            side_effect=OpenAIRateLimitError(
                message="Rate limited",
                response=MagicMock(status_code=429),
                body=None,
            )
        )

        with pytest.raises(RateLimitError):
            async for _ in llm_with_mock.stream("Hi"):
                pass

    @pytest.mark.asyncio
    async def test_stream_api_error(self, llm_with_mock: LLM) -> None:
        """Test stream handles APIError."""
        error = APIError(message="Error", request=MagicMock(), body=None)
        error.status_code = 500  # type: ignore[attr-defined]

        llm_with_mock._client.chat.completions.create = AsyncMock(side_effect=error)

        with pytest.raises(ServerError):
            async for _ in llm_with_mock.stream("Hi"):
                pass

    @pytest.mark.asyncio
    async def test_run_with_message_object(self, llm_with_mock: LLM) -> None:
        """Test run with Message object."""
        llm_with_mock._client.chat.completions.create = AsyncMock(
            return_value=create_mock_response(content="Done")
        )

        result = await llm_with_mock.run(Message.user("Do something"))
        assert result.content == "Done"

    @pytest.mark.asyncio
    async def test_run_with_message_list(self, llm_with_mock: LLM) -> None:
        """Test run with list of messages."""
        llm_with_mock._client.chat.completions.create = AsyncMock(
            return_value=create_mock_response(content="Done")
        )

        messages = [Message.system("Be helpful"), Message.user("Hi")]
        result = await llm_with_mock.run(messages)
        assert result.content == "Done"

    @pytest.mark.asyncio
    async def test_run_with_tool_list(self, llm_with_mock: LLM) -> None:
        """Test run with list of Tools instead of ToolRegistry."""

        def get_time() -> str:
            """Get the current time."""
            return "12:00 PM"

        tools = [Tool(get_time)]

        # No tool calls, just return response
        llm_with_mock._client.chat.completions.create = AsyncMock(
            return_value=create_mock_response(content="Done")
        )

        result = await llm_with_mock.run("Hi", tools=tools)
        assert result.content == "Done"

    @pytest.mark.asyncio
    async def test_run_max_iterations(self, llm_with_mock: LLM) -> None:
        """Test run respects max_iterations."""

        def get_time() -> str:
            """Get the current time."""
            return "12:00 PM"

        registry = ToolRegistry()
        registry.add(Tool(get_time))

        # Always return tool calls, never a final response
        tool_call = create_mock_tool_call(id="tc_1", name="get_time", arguments="{}")
        llm_with_mock._client.chat.completions.create = AsyncMock(
            return_value=create_mock_response(content=None, tool_calls=[tool_call])
        )

        # Should stop after max_iterations
        await llm_with_mock.run("Hi", tools=registry, max_iterations=3)

        # Should have called complete 3 times
        assert llm_with_mock._client.chat.completions.create.call_count == 3

    @pytest.mark.asyncio
    async def test_structured_output_retry_then_success(self, llm_with_mock: LLM) -> None:
        """Test structured output retries on JSON error then succeeds."""
        responses = [
            create_mock_response(content="invalid json"),
            create_mock_response(content='{"valid": true}'),
        ]
        llm_with_mock._client.chat.completions.create = AsyncMock(side_effect=responses)

        result = await llm_with_mock.complete(
            "Get data", structured_output=True, max_json_retries=2
        )

        assert result.data == {"valid": True}
        assert llm_with_mock._client.chat.completions.create.call_count == 2

    @pytest.mark.asyncio
    async def test_structured_output_all_retries_fail(self, llm_with_mock: LLM) -> None:
        """Test structured output raises after all retries fail."""
        llm_with_mock._client.chat.completions.create = AsyncMock(
            return_value=create_mock_response(content="not json at all")
        )

        with pytest.raises(JSONParseError) as exc_info:
            await llm_with_mock.complete("Get data", structured_output=True, max_json_retries=2)

        assert "Invalid JSON" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_structured_output_empty_retries(self, llm_with_mock: LLM) -> None:
        """Test structured output retries on empty content."""
        responses = [
            create_mock_response(content=""),
            create_mock_response(content='{"data": 1}'),
        ]
        llm_with_mock._client.chat.completions.create = AsyncMock(side_effect=responses)

        result = await llm_with_mock.complete(
            "Get data", structured_output=True, max_json_retries=2
        )

        assert result.data == {"data": 1}

    @pytest.mark.asyncio
    async def test_stream_empty_chunk(self, llm_with_mock: LLM) -> None:
        """Test stream skips chunks without content."""

        async def mock_stream():
            # First chunk has no content
            chunk1 = MagicMock()
            chunk1.choices = [MagicMock()]
            chunk1.choices[0].delta.content = None
            yield chunk1

            # Second chunk has no choices
            chunk2 = MagicMock()
            chunk2.choices = []
            yield chunk2

            # Third chunk has content
            chunk3 = MagicMock()
            chunk3.choices = [MagicMock()]
            chunk3.choices[0].delta.content = "Hello"
            yield chunk3

        llm_with_mock._client.chat.completions.create = AsyncMock(return_value=mock_stream())

        chunks = []
        async for chunk in llm_with_mock.stream("Hi"):
            chunks.append(chunk)

        assert chunks == ["Hello"]

    def test_extract_json_array(self) -> None:
        """Test extracting JSON array from content."""
        with patch.object(AsyncOpenAI, "__init__", return_value=None):
            llm = LLM(api_key="test-key", model="test-model")

        content = "[1, 2, 3]"
        result = llm._extract_json(content)
        assert result == "[1, 2, 3]"

    def test_extract_json_embedded_array(self) -> None:
        """Test extracting embedded JSON array from content."""
        with patch.object(AsyncOpenAI, "__init__", return_value=None):
            llm = LLM(api_key="test-key", model="test-model")

        content = "Here is the data: [1, 2, 3] as requested"
        result = llm._extract_json(content)
        assert result == "[1, 2, 3]"
