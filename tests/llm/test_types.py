"""Tests for backend/llm/types.py."""

from backend.llm.types import Completion, Function, Message, ToolCall, Usage

# -----------------------------------------------------------------------------
# Message Tests
# -----------------------------------------------------------------------------


class TestMessage:
    """Tests for the Message class."""

    def test_system_message_creation(self) -> None:
        """Test creating a system message."""
        msg = Message.system("You are a helpful assistant.")
        assert msg.role == "system"
        assert msg.content == "You are a helpful assistant."
        assert msg.tool_calls is None
        assert msg.tool_call_id is None

    def test_user_message_creation(self) -> None:
        """Test creating a user message."""
        msg = Message.user("Hello, how are you?")
        assert msg.role == "user"
        assert msg.content == "Hello, how are you?"

    def test_assistant_message_creation(self) -> None:
        """Test creating an assistant message."""
        msg = Message.assistant("I'm doing well, thank you!")
        assert msg.role == "assistant"
        assert msg.content == "I'm doing well, thank you!"
        assert msg.tool_calls is None

    def test_assistant_message_with_tool_calls(self) -> None:
        """Test creating an assistant message with tool calls."""
        tool_call = ToolCall(
            id="tc_123",
            type="function",
            function=Function(name="get_weather", arguments='{"city": "London"}'),
        )
        msg = Message.assistant(content=None, tool_calls=[tool_call])
        assert msg.role == "assistant"
        assert msg.content is None
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].function.name == "get_weather"

    def test_tool_message_creation(self) -> None:
        """Test creating a tool message."""
        msg = Message.tool(content="Weather in London: 15°C", tool_call_id="tc_123")
        assert msg.role == "tool"
        assert msg.content == "Weather in London: 15°C"
        assert msg.tool_call_id == "tc_123"

    def test_message_with_none_content(self) -> None:
        """Test message with None content."""
        msg = Message(role="assistant", content=None)
        assert msg.content is None

    def test_message_model_dump(self) -> None:
        """Test message serialization."""
        msg = Message.user("Hello")
        dumped = msg.model_dump()
        assert dumped["role"] == "user"
        assert dumped["content"] == "Hello"


# -----------------------------------------------------------------------------
# Usage Tests
# -----------------------------------------------------------------------------


class TestUsage:
    """Tests for the Usage class."""

    def test_usage_default_values(self) -> None:
        """Test Usage with default values."""
        usage = Usage()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0

    def test_usage_with_values(self) -> None:
        """Test Usage with custom values."""
        usage = Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150

    def test_usage_model_dump(self) -> None:
        """Test usage serialization."""
        usage = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        dumped = usage.model_dump()
        assert dumped == {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }


# -----------------------------------------------------------------------------
# Function Tests
# -----------------------------------------------------------------------------


class TestFunction:
    """Tests for the Function class."""

    def test_function_creation(self) -> None:
        """Test Function creation."""
        func = Function(name="get_weather", arguments='{"city": "Paris"}')
        assert func.name == "get_weather"
        assert func.arguments == '{"city": "Paris"}'

    def test_function_empty_arguments(self) -> None:
        """Test Function with empty arguments."""
        func = Function(name="no_args", arguments="{}")
        assert func.name == "no_args"
        assert func.arguments == "{}"


# -----------------------------------------------------------------------------
# ToolCall Tests
# -----------------------------------------------------------------------------


class TestToolCall:
    """Tests for the ToolCall class."""

    def test_tool_call_creation(self) -> None:
        """Test ToolCall creation."""
        tc = ToolCall(
            id="call_123",
            type="function",
            function=Function(name="search", arguments='{"query": "python"}'),
        )
        assert tc.id == "call_123"
        assert tc.type == "function"
        assert tc.function.name == "search"

    def test_tool_call_default_type(self) -> None:
        """Test ToolCall default type is 'function'."""
        tc = ToolCall(
            id="call_456",
            function=Function(name="test", arguments="{}"),
        )
        assert tc.type == "function"


# -----------------------------------------------------------------------------
# Completion Tests
# -----------------------------------------------------------------------------


class TestCompletion:
    """Tests for the Completion class."""

    def test_completion_creation(self) -> None:
        """Test basic Completion creation."""
        msg = Message.assistant("Hello!")
        completion = Completion(
            message=msg,
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            model="gpt-4",
            finish_reason="stop",
        )
        assert completion.message.content == "Hello!"
        assert completion.model == "gpt-4"
        assert completion.finish_reason == "stop"

    def test_completion_has_tool_calls_false(self) -> None:
        """Test has_tool_calls when no tool calls."""
        msg = Message.assistant("No tools here")
        completion = Completion(message=msg)
        assert completion.has_tool_calls is False

    def test_completion_has_tool_calls_true(self) -> None:
        """Test has_tool_calls when tool calls present."""
        tool_call = ToolCall(
            id="tc_1",
            function=Function(name="test", arguments="{}"),
        )
        msg = Message.assistant(content=None, tool_calls=[tool_call])
        completion = Completion(message=msg)
        assert completion.has_tool_calls is True

    def test_completion_content_property(self) -> None:
        """Test content property returns message content."""
        msg = Message.assistant("Test content")
        completion = Completion(message=msg)
        assert completion.content == "Test content"

    def test_completion_with_data(self) -> None:
        """Test Completion with parsed data."""
        msg = Message.assistant('{"key": "value"}')
        completion = Completion(message=msg, data={"key": "value"})
        assert completion.data == {"key": "value"}

    def test_completion_optional_fields(self) -> None:
        """Test Completion with minimal fields."""
        msg = Message.assistant("Hi")
        completion = Completion(message=msg)
        assert completion.usage is None
        assert completion.model is None
        assert completion.finish_reason is None
        assert completion.data is None
