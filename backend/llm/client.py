import json
from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI, APIError, AuthenticationError as OpenAIAuthError
from openai import RateLimitError as OpenAIRateLimitError

from ..utils.config import config

from .errors import (
    AuthenticationError,
    ContentFilterError,
    ContextLengthError,
    InvalidRequestError,
    JSONParseError,
    LLMError,
    ModelNotFoundError,
    RateLimitError,
)

from .tools import Tool, ToolRegistry
from .types import Completion, Function, Message, ToolCall, Usage


class LLM:
    """
    A lightweight LLM client with OpenAI-compatible API support.

    Supports any provider that implements the OpenAI API format:
    OpenAI, OpenRouter, Fireworks, Together, local models via Ollama, etc.

    Usage:
        llm = LLM(api_key="sk-...", base_url="https://api.openai.com/v1")
        response = await llm.complete("Hello!")
        print(response.message.content)
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = config.openai_base_url,
        model: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 2,
    ):
        """
        Initialize the LLM client.

        Args:
            api_key: API key for authentication.
            base_url: Base URL for the API (OpenAI-compatible).
            model: Default model to use. Can be overridden per request.
            timeout: Request timeout in seconds.
            max_retries: Number of retries for failed requests.
        """
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
        self._default_model = model

    async def complete(
        self,
        messages: list[Message] | Message | str,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        stop: list[str] | str | None = None,
        structured_output: bool = False,
        max_json_retries: int = 3,
        **kwargs: Any,
    ) -> Completion:
        """
        Generate a completion.

        Args:
            messages: Input messages. Can be a list, single Message, or string.
            model: Model to use. Falls back to default if not specified.
            temperature: Sampling temperature (0-2).
            max_tokens: Maximum tokens to generate.
            tools: Tool definitions for function calling.
            tool_choice: How to select tools ("auto", "none", or specific tool).
            stop: Stop sequences.
            structured_output: If True, enforces JSON output and parses to dict.
            max_json_retries: Retries on JSON parse failure (default: 3).
            **kwargs: Additional provider-specific parameters.

        Returns:
            Completion object with message and metadata.
            If structured_output=True, the .data field contains the parsed dict.

        Raises:
            LLMError: On API errors.
            JSONParseError: If structured_output=True and JSON parsing fails.
        """
        model = model or self._default_model
        if not model:
            raise InvalidRequestError("No model specified and no default model set")

        prepared_messages = self._prepare_messages(messages)

        request_params: dict[str, Any] = {
            "model": model,
            "messages": prepared_messages,
        }

        if temperature is not None:
            request_params["temperature"] = temperature
        if max_tokens is not None:
            request_params["max_tokens"] = max_tokens
        if tools is not None:
            request_params["tools"] = tools
        if tool_choice is not None:
            request_params["tool_choice"] = tool_choice
        if stop is not None:
            request_params["stop"] = stop

        if structured_output:
            kwargs["response_format"] = {"type": "json_object"}
            json_hint = {"role": "system", "content": "You must output valid JSON."}
            request_params["messages"] = list[dict[str, Any]](prepared_messages) + [
                json_hint
            ]

        request_params.update(kwargs)

        attempts = max_json_retries if structured_output else 1
        last_error: JSONParseError | None = None

        for attempt in range(attempts):
            try:
                response = await self._client.chat.completions.create(
                    **request_params,
                    extra_body={
                        "reasoning": {"enabled": False},
                        "provider": {"order": ["fireworks"], "allow_fallbacks": True},
                    },
                )
            except OpenAIAuthError as e:
                raise AuthenticationError(str(e)) from e
            except OpenAIRateLimitError as e:
                raise RateLimitError(str(e)) from e
            except APIError as e:
                raise self._map_api_error(e) from e

            completion = self._parse_response(response)

            if structured_output:
                content = completion.message.content
                if not content:
                    last_error = JSONParseError("Empty response content")
                    if attempt < attempts - 1:
                        continue
                    raise last_error

                try:
                    completion.data = json.loads(content)
                except json.JSONDecodeError as e:
                    last_error = JSONParseError(f"Invalid JSON: {e}")
                    if attempt < attempts - 1:
                        continue
                    raise last_error from e

            return completion

    async def stream(
        self,
        messages: list[Message] | Message | str,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        stop: list[str] | str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Stream a completion, yielding content chunks.

        Args:
            messages: Input messages.
            model: Model to use.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens.
            tools: Tool definitions (note: tool calls in streaming are complex).
            tool_choice: Tool selection mode.
            stop: Stop sequences.
            **kwargs: Additional parameters.

        Yields:
            Content chunks as strings.
        """
        model = model or self._default_model
        if not model:
            raise InvalidRequestError("No model specified and no default model set")

        prepared_messages = self._prepare_messages(messages)

        request_params: dict[str, Any] = {
            "model": model,
            "messages": prepared_messages,
            "stream": True,
        }

        if temperature is not None:
            request_params["temperature"] = temperature
        if max_tokens is not None:
            request_params["max_tokens"] = max_tokens
        if tools is not None:
            request_params["tools"] = tools
        if tool_choice is not None:
            request_params["tool_choice"] = tool_choice
        if stop is not None:
            request_params["stop"] = stop

        request_params.update(kwargs)

        try:
            stream = await self._client.chat.completions.create(**request_params)
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except OpenAIAuthError as e:
            raise AuthenticationError(str(e)) from e
        except OpenAIRateLimitError as e:
            raise RateLimitError(str(e)) from e
        except APIError as e:
            raise self._map_api_error(e) from e

    async def run(
        self,
        messages: list[Message] | Message | str,
        tools: ToolRegistry | list[Tool] | None = None,
        *,
        model: str | None = None,
        max_iterations: int = 100,
        **kwargs: Any,
    ) -> Completion:
        """
        Run a completion with automatic tool execution.

        Automatically executes tool calls and continues until the model
        returns a final response (no more tool calls).

        Args:
            messages: Input messages.
            tools: ToolRegistry or list of Tool instances.
            model: Model to use.
            max_iterations: Max tool call rounds to prevent infinite loops.
            **kwargs: Additional parameters passed to complete().

        Returns:
            Final Completion after all tool calls are resolved.
        """
        if isinstance(messages, str):
            message_list = [Message.user(messages)]
        elif isinstance(messages, Message):
            message_list = [messages]
        else:
            message_list = list[Message](messages)

        if tools is None:
            return await self.complete(message_list, model=model, **kwargs)

        if isinstance(tools, list):
            registry = ToolRegistry()
            for t in tools:
                registry.add(t)
        else:
            registry = tools

        tool_schemas = registry.schemas

        for _ in range(max_iterations):
            response = await self.complete(
                message_list, model=model, tools=tool_schemas, **kwargs
            )

            if not response.has_tool_calls:
                return response

            message_list.append(response.message)

            tool_messages = await registry.execute_all(response.message.tool_calls)
            message_list.extend(tool_messages)

        return response

    def _prepare_messages(
        self, messages: list[Message] | Message | str
    ) -> list[dict[str, Any]]:
        """Convert input to list of message dicts."""
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]
        if isinstance(messages, Message):
            return [messages.model_dump(exclude_none=True)]
        return [m.model_dump(exclude_none=True) for m in messages]

    def _parse_response(self, response: Any) -> Completion:
        """Parse OpenAI response into Completion."""
        choice = response.choices[0]
        msg = choice.message

        tool_calls = None
        if msg.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    type="function",
                    function=Function(
                        name=tc.function.name,
                        arguments=tc.function.arguments,
                    ),
                )
                for tc in msg.tool_calls
            ]

        message = Message(
            role="assistant",
            content=msg.content,
            tool_calls=tool_calls,
        )

        usage = None
        if response.usage:
            usage = Usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )

        return Completion(
            message=message,
            usage=usage,
            model=response.model,
            finish_reason=choice.finish_reason,
        )

    def _map_api_error(self, error: APIError) -> LLMError:
        """Map OpenAI API errors to our error types."""
        message = str(error)
        status = getattr(error, "status_code", None)

        if status == 401:
            return AuthenticationError(message, status)
        if status == 429:
            return RateLimitError(message)
        if status == 404:
            return ModelNotFoundError(message, status)
        if status == 400:
            if "context_length" in message.lower():
                return ContextLengthError(message, status)
            if "content_filter" in message.lower() or "safety" in message.lower():
                return ContentFilterError(message, status)
            return InvalidRequestError(message, status)

        return LLMError(message, status)
