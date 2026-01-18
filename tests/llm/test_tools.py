"""Tests for backend/llm/tools.py."""

import json
from typing import Any

import pytest
from pydantic import BaseModel

from backend.llm.tools import Tool, ToolRegistry
from backend.llm.types import Function, ToolCall

# -----------------------------------------------------------------------------
# Sample Functions for Testing
# -----------------------------------------------------------------------------


def sync_function(city: str, unit: str = "celsius") -> str:
    """Get the weather for a city."""
    return f"Weather in {city}: 22{unit[0].upper()}"


async def async_function(query: str) -> str:
    """Search for something."""
    return f"Results for: {query}"


def function_with_list(items: list[str]) -> str:
    """Process a list of items."""
    return ", ".join(items)


def function_with_dict(data: dict) -> str:
    """Process a dictionary."""
    return str(data)


def function_no_args() -> str:
    """A function with no arguments."""
    return "no args"


class ResponseModel(BaseModel):
    """A Pydantic model for responses."""

    result: str
    count: int


def function_returns_model() -> ResponseModel:
    """Return a Pydantic model."""
    return ResponseModel(result="success", count=42)


def function_returns_dict() -> dict[str, Any]:
    """Return a dictionary."""
    return {"key": "value", "number": 123}


# -----------------------------------------------------------------------------
# Tool Tests
# -----------------------------------------------------------------------------


class TestTool:
    """Tests for the Tool class."""

    def test_tool_creation_from_function(self) -> None:
        """Test creating a Tool from a function."""
        tool = Tool(sync_function)
        assert tool.name == "sync_function"
        assert "weather" in tool.description.lower()

    def test_tool_with_custom_name(self) -> None:
        """Test creating a Tool with custom name."""
        tool = Tool(sync_function, name="get_weather")
        assert tool.name == "get_weather"

    def test_tool_with_custom_description(self) -> None:
        """Test creating a Tool with custom description."""
        tool = Tool(sync_function, description="Custom description")
        assert tool.description == "Custom description"

    def test_tool_schema_structure(self) -> None:
        """Test the generated schema structure."""
        tool = Tool(sync_function)
        schema = tool.schema

        assert schema["type"] == "function"
        assert "function" in schema
        assert schema["function"]["name"] == "sync_function"
        assert "parameters" in schema["function"]
        assert schema["function"]["parameters"]["type"] == "object"

    def test_tool_schema_required_params(self) -> None:
        """Test that required params are identified correctly."""
        tool = Tool(sync_function)
        schema = tool.schema

        required = schema["function"]["parameters"]["required"]
        properties = schema["function"]["parameters"]["properties"]

        assert "city" in required
        assert "unit" not in required  # Has default value
        assert "city" in properties
        assert "unit" in properties

    def test_tool_schema_type_mapping(self) -> None:
        """Test type mapping in schema."""

        def typed_func(_s: str, _i: int, _f: float, _b: bool) -> str:
            return "test"

        tool = Tool(typed_func)
        props = tool.schema["function"]["parameters"]["properties"]

        assert props["_s"]["type"] == "string"
        assert props["_i"]["type"] == "integer"
        assert props["_f"]["type"] == "number"
        assert props["_b"]["type"] == "boolean"

    def test_tool_schema_list_type(self) -> None:
        """Test list type handling in schema."""
        tool = Tool(function_with_list)
        props = tool.schema["function"]["parameters"]["properties"]

        assert props["items"]["type"] == "array"
        assert props["items"]["items"]["type"] == "string"

    def test_tool_schema_dict_type(self) -> None:
        """Test dict type handling in schema."""
        tool = Tool(function_with_dict)
        props = tool.schema["function"]["parameters"]["properties"]

        assert props["data"]["type"] == "object"

    @pytest.mark.asyncio
    async def test_tool_execute_sync_function(self) -> None:
        """Test executing a sync function."""
        tool = Tool(sync_function)
        result = await tool.execute({"city": "London"})
        assert "London" in result
        assert "22" in result

    @pytest.mark.asyncio
    async def test_tool_execute_async_function(self) -> None:
        """Test executing an async function."""
        tool = Tool(async_function)
        result = await tool.execute({"query": "python"})
        assert "python" in result

    @pytest.mark.asyncio
    async def test_tool_execute_with_json_string(self) -> None:
        """Test executing with JSON string arguments."""
        tool = Tool(sync_function)
        result = await tool.execute('{"city": "Paris", "unit": "fahrenheit"}')
        assert "Paris" in result
        assert "F" in result

    @pytest.mark.asyncio
    async def test_tool_execute_returns_json_for_dict(self) -> None:
        """Test that dict return values are serialized to JSON."""
        tool = Tool(function_returns_dict)
        result = await tool.execute({})
        parsed = json.loads(result)
        assert parsed["key"] == "value"
        assert parsed["number"] == 123

    @pytest.mark.asyncio
    async def test_tool_execute_returns_json_for_model(self) -> None:
        """Test that Pydantic model return values are serialized."""
        tool = Tool(function_returns_model)
        result = await tool.execute({})
        parsed = json.loads(result)
        assert parsed["result"] == "success"
        assert parsed["count"] == 42

    def test_tool_direct_call(self) -> None:
        """Test calling tool directly."""
        tool = Tool(sync_function)
        result = tool("London", unit="fahrenheit")
        assert "London" in result

    def test_tool_is_async_detection(self) -> None:
        """Test async function detection."""
        sync_tool = Tool(sync_function)
        async_tool = Tool(async_function)

        assert sync_tool._is_async is False
        assert async_tool._is_async is True


# -----------------------------------------------------------------------------
# ToolRegistry Tests
# -----------------------------------------------------------------------------


class TestToolRegistry:
    """Tests for the ToolRegistry class."""

    def test_registry_register_decorator(self) -> None:
        """Test using register as a decorator."""
        registry = ToolRegistry()

        @registry.register
        def my_tool(x: str) -> str:
            """Do something."""
            return x

        assert len(registry) == 1
        assert registry.get("my_tool") is not None

    def test_registry_register_with_name(self) -> None:
        """Test register decorator with custom name."""
        registry = ToolRegistry()

        @registry.register(name="custom_name")
        def another_tool(x: str) -> str:
            """Another tool."""
            return x

        assert registry.get("custom_name") is not None
        assert registry.get("another_tool") is None

    def test_registry_add_tool(self) -> None:
        """Test adding an existing Tool instance."""
        registry = ToolRegistry()
        tool = Tool(sync_function)
        registry.add(tool)

        assert len(registry) == 1
        assert registry.get("sync_function") is tool

    def test_registry_get_nonexistent(self) -> None:
        """Test getting a non-existent tool returns None."""
        registry = ToolRegistry()
        assert registry.get("nonexistent") is None

    def test_registry_schemas(self) -> None:
        """Test getting all schemas."""
        registry = ToolRegistry()
        registry.add(Tool(sync_function))
        registry.add(Tool(async_function))

        schemas = registry.schemas
        assert len(schemas) == 2
        assert all(s["type"] == "function" for s in schemas)

    @pytest.mark.asyncio
    async def test_registry_execute_tool_call(self) -> None:
        """Test executing a tool call."""
        registry = ToolRegistry()
        registry.add(Tool(sync_function))

        tool_call = ToolCall(
            id="tc_123",
            function=Function(name="sync_function", arguments='{"city": "Tokyo"}'),
        )

        result = await registry.execute(tool_call)
        assert result.role == "tool"
        assert result.tool_call_id == "tc_123"
        assert result.content is not None
        assert "Tokyo" in result.content

    @pytest.mark.asyncio
    async def test_registry_execute_unknown_tool(self) -> None:
        """Test executing unknown tool raises KeyError."""
        registry = ToolRegistry()
        tool_call = ToolCall(
            id="tc_999",
            function=Function(name="unknown_tool", arguments="{}"),
        )

        with pytest.raises(KeyError) as exc_info:
            await registry.execute(tool_call)
        assert "unknown_tool" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_registry_execute_all(self) -> None:
        """Test executing multiple tool calls."""
        registry = ToolRegistry()
        registry.add(Tool(sync_function))

        tool_calls = [
            ToolCall(
                id="tc_1",
                function=Function(name="sync_function", arguments='{"city": "A"}'),
            ),
            ToolCall(
                id="tc_2",
                function=Function(name="sync_function", arguments='{"city": "B"}'),
            ),
        ]

        results = await registry.execute_all(tool_calls)
        assert len(results) == 2
        assert all(r.role == "tool" for r in results)
        assert results[0].tool_call_id == "tc_1"
        assert results[1].tool_call_id == "tc_2"

    def test_registry_iteration(self) -> None:
        """Test iterating over registry."""
        registry = ToolRegistry()
        registry.add(Tool(sync_function))
        registry.add(Tool(async_function))

        tools = list(registry)
        assert len(tools) == 2

    def test_registry_len(self) -> None:
        """Test len() on registry."""
        registry = ToolRegistry()
        assert len(registry) == 0

        registry.add(Tool(sync_function))
        assert len(registry) == 1


# -----------------------------------------------------------------------------
# Edge Cases
# -----------------------------------------------------------------------------


class TestToolEdgeCases:
    """Test edge cases for Tool and ToolRegistry."""

    @pytest.mark.asyncio
    async def test_malformed_json_fix(self) -> None:
        """Test that malformed JSON is fixed when possible."""
        tool = Tool(sync_function)
        # Simulate malformed JSON that some models produce
        malformed = '{"city": "London"}{"unit": "celsius"}'
        # This should be fixed by the tool
        result = await tool.execute(malformed)
        assert "London" in result

    def test_tool_without_docstring(self) -> None:
        """Test tool creation when function has no docstring."""

        def no_docs(x: str) -> str:
            return x

        tool = Tool(no_docs)
        assert tool.description == ""
        assert tool.name == "no_docs"

    def test_tool_schema_cached(self) -> None:
        """Test that schema is cached after first access."""
        tool = Tool(sync_function)
        schema1 = tool.schema
        schema2 = tool.schema
        assert schema1 is schema2

    def test_tool_with_no_type_hints(self) -> None:
        """Test tool with no type hints uses string as default."""

        def untyped(x):  # noqa: ANN001, ANN202
            return str(x)

        tool = Tool(untyped)
        props = tool.schema["function"]["parameters"]["properties"]
        assert props["x"]["type"] == "string"

    def test_tool_with_union_type(self) -> None:
        """Test tool with Union/Optional type."""

        def union_func(x: str | None) -> str:
            return x or "default"

        tool = Tool(union_func)
        props = tool.schema["function"]["parameters"]["properties"]
        # Union with None should just use the non-None type
        assert props["x"]["type"] == "string"

    def test_tool_with_multi_union_type(self) -> None:
        """Test tool with multi-type Union."""

        def multi_union_func(x: str | int) -> str:
            return str(x)

        tool = Tool(multi_union_func)
        props = tool.schema["function"]["parameters"]["properties"]
        # Should create anyOf schema
        assert "anyOf" in props["x"]

    def test_tool_with_literal_type(self) -> None:
        """Test tool with Literal type."""
        from typing import Literal

        def literal_func(mode: Literal["fast", "slow"]) -> str:
            return mode

        tool = Tool(literal_func)
        props = tool.schema["function"]["parameters"]["properties"]
        assert "enum" in props["mode"]
        assert set(props["mode"]["enum"]) == {"fast", "slow"}

    def test_tool_with_pydantic_model(self) -> None:
        """Test tool with Pydantic model parameter."""
        from pydantic import BaseModel

        class InputModel(BaseModel):
            name: str
            value: int

        def model_func(data: InputModel) -> str:
            return data.name

        tool = Tool(model_func)
        props = tool.schema["function"]["parameters"]["properties"]
        # Should have the model schema
        assert "properties" in props["data"]

    def test_tool_with_method(self) -> None:
        """Test tool creation from class method."""

        class Helper:
            def method(self, x: str) -> str:
                return x.upper()

        helper = Helper()
        tool = Tool(helper.method)
        props = tool.schema["function"]["parameters"]["properties"]
        # self should be excluded
        assert "self" not in props
        assert "x" in props
