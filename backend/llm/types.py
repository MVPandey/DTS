from typing import Any, Literal

from pydantic import BaseModel

Role = Literal["system", "user", "assistant", "tool"]


class Function(BaseModel):
    """Function definition for tool calls."""

    name: str
    arguments: str  # JSON string


class ToolCall(BaseModel):
    """A tool call requested by the model."""

    id: str
    type: Literal["function"] = "function"
    function: Function


class Message(BaseModel):
    """A chat message."""

    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None

    @classmethod
    def system(cls, content: str) -> "Message":
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> "Message":
        return cls(role="user", content=content)

    @classmethod
    def assistant(
        cls, content: str | None = None, tool_calls: list[ToolCall] | None = None
    ) -> "Message":
        return cls(role="assistant", content=content, tool_calls=tool_calls)

    @classmethod
    def tool(cls, content: str, tool_call_id: str) -> "Message":
        return cls(role="tool", content=content, tool_call_id=tool_call_id)


class Usage(BaseModel):
    """Token usage information."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class Completion(BaseModel):
    """A completion response from the LLM."""

    message: Message
    usage: Usage | None = None
    model: str | None = None
    finish_reason: str | None = None
    data: dict[str, Any] | None = None  # Parsed JSON when structured_output=True

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.message.tool_calls)

    @property
    def content(self) -> str | None:
        return self.message.content
