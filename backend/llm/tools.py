import inspect
import json
from collections.abc import Awaitable, Callable
from typing import Any, get_args, get_origin

from pydantic import BaseModel

from .types import Message, ToolCall

_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


class Tool:
    """
    Wraps a function as a callable tool for LLM function calling.

    Usage:
        @Tool
        def get_weather(city: str, unit: str = "celsius") -> str:
            '''Get the current weather for a city.'''
            return f"Weather in {city}: 22{unit[0].upper()}"

        # Or without decorator
        tool = Tool(get_weather)

        # Get OpenAI-compatible schema
        schema = tool.schema

        # Execute with parsed arguments
        result = await tool.execute({"city": "London"})
    """

    def __init__(
        self,
        fn: Callable[..., Any] | Callable[..., Awaitable[Any]],
        *,
        name: str | None = None,
        description: str | None = None,
    ):
        self.fn = fn
        self.name = name or getattr(fn, "__name__", "tool")
        self.description = description or getattr(fn, "__doc__", "") or ""
        self._is_async = inspect.iscoroutinefunction(fn)
        self._schema: dict[str, Any] | None = None

    @property
    def schema(self) -> dict[str, Any]:
        """Generate OpenAI-compatible tool schema."""
        if self._schema is None:
            self._schema = self._build_schema()
        return self._schema

    def _build_schema(self) -> dict[str, Any]:
        """Build JSON Schema from function signature."""
        sig = inspect.signature(self.fn)
        hints = getattr(self.fn, "__annotations__", {})

        properties: dict[str, Any] = {}
        required: list[str] = []

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            param_type = hints.get(param_name, str)
            prop_schema = self._type_to_schema(param_type)

            properties[param_name] = prop_schema

            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description.strip(),
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def _type_to_schema(self, t: type) -> dict[str, Any]:
        """Convert Python type to JSON Schema."""
        origin = get_origin(t)

        if origin is type(None):
            return {"type": "null"}

        if origin is list:
            args = get_args(t)
            items_schema = self._type_to_schema(args[0]) if args else {"type": "string"}
            return {"type": "array", "items": items_schema}

        if origin is dict:
            return {"type": "object"}

        if origin is type(int | str):
            args = get_args(t)
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1:
                return self._type_to_schema(non_none[0])
            return {"anyOf": [self._type_to_schema(a) for a in non_none]}

        if origin is type(None) or str(origin) == "typing.Literal":
            args = get_args(t)
            if args:
                return {"type": "string", "enum": list(args)}

        if isinstance(t, type) and issubclass(t, BaseModel):
            return t.model_json_schema()

        json_type = _TYPE_MAP.get(t, "string")
        return {"type": json_type}

    async def execute(self, arguments: dict[str, Any] | str) -> str:
        """
        Execute the tool with given arguments.

        Args:
            arguments: Dict of arguments or JSON string.

        Returns:
            String result (for use in tool response message).
        """
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                # Some models return malformed JSON, try to fix common issues
                # e.g., {"a": 1}{"b": 2} -> {"a": 1, "b": 2}
                import re

                fixed = re.sub(r"\}\s*\{", ", ", arguments)
                arguments = json.loads(fixed)

        if self._is_async:
            result = await self.fn(**arguments)
        else:
            result = self.fn(**arguments)

        if isinstance(result, str):
            return result
        if isinstance(result, BaseModel):
            return result.model_dump_json()
        return json.dumps(result)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Allow direct calling of the wrapped function."""
        return self.fn(*args, **kwargs)


class ToolRegistry:
    """
    Registry for managing multiple tools.

    Usage:
        tools = ToolRegistry()

        @tools.register
        def get_weather(city: str) -> str:
            return f"Sunny in {city}"

        @tools.register
        def search(query: str) -> str:
            return f"Results for {query}"

        # Get all schemas for LLM
        schemas = tools.schemas

        # Execute a tool call
        result = await tools.execute(tool_call)
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(
        self,
        fn: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> Callable[..., Any] | Tool:
        """
        Register a function as a tool.

        Can be used as decorator with or without arguments:
            @tools.register
            def my_func(): ...

            @tools.register(name="custom_name")
            def my_func(): ...
        """

        def decorator(f: Callable[..., Any]) -> Tool:
            tool = Tool(f, name=name, description=description)
            self._tools[tool.name] = tool
            return tool

        if fn is not None:
            return decorator(fn)
        return decorator

    def add(self, tool: Tool) -> None:
        """Add an existing Tool instance."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    @property
    def schemas(self) -> list[dict[str, Any]]:
        """Get all tool schemas for LLM."""
        return [tool.schema for tool in self._tools.values()]

    async def execute(self, tool_call: ToolCall) -> Message:
        """
        Execute a tool call and return a tool response message.

        Args:
            tool_call: The tool call from the LLM.

        Returns:
            Message with role="tool" containing the result.

        Raises:
            KeyError: If tool not found.
        """
        tool = self._tools.get(tool_call.function.name)
        if not tool:
            raise KeyError(f"Tool not found: {tool_call.function.name}")

        result = await tool.execute(tool_call.function.arguments)
        return Message.tool(content=result, tool_call_id=tool_call.id)

    async def execute_all(self, tool_calls: list[ToolCall]) -> list[Message]:
        """Execute multiple tool calls and return all response messages."""
        return [await self.execute(tc) for tc in tool_calls]

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self):
        return iter(self._tools.values())
