import json
import re
from collections.abc import AsyncIterator
from typing import Any

from openai import (
    APIError,
    AsyncOpenAI,
    AuthenticationError as OpenAIAuthError,
    RateLimitError as OpenAIRateLimitError,
)

from backend.utils.logging import logger

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
    ServerError,
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

    # --- Initialization ---

    def __init__(
        self,
        api_key: str,
        base_url: str = config.openai_base_url,
        model: str | None = None,
        timeout: float = config.llm_timeout,
        max_retries: int = config.llm_max_retries,
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

    # --- Public Methods ---

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
        provider: str | list[str] | None = None,
        reasoning_enabled: bool | None = None,
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
            provider: Provider preference for OpenRouter (e.g., "Fireworks" or ["Fireworks", "Together"]).
            reasoning_enabled: Enable/disable reasoning tokens (OpenRouter). None = don't specify.
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

        request_params = self._build_request_params(
            model=model,
            messages=prepared_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            stop=stop,
        )

        extra_body = self._build_extra_body(
            provider, reasoning_enabled, kwargs.pop("extra_body", {})
        )
        if extra_body:
            kwargs["extra_body"] = extra_body

        if structured_output:
            # Skip response_format for reasoning models as it may conflict
            # Instead, rely on prompt instructions to get JSON output
            if not reasoning_enabled:
                kwargs["response_format"] = {"type": "json_object"}
            # Note: JSON output instructions are now included in system prompts,
            # so no additional hint message is needed

        request_params.update(kwargs)

        attempts = max_json_retries if structured_output else 1
        last_error: JSONParseError | None = None

        for attempt in range(attempts):
            try:
                response = await self._client.chat.completions.create(
                    **request_params,
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

                # Check for reasoning field if content is empty (some models put content there)
                if not content:
                    # Try to get content from reasoning field for reasoning models
                    choice = response.choices[0] if response.choices else None
                    if choice and hasattr(choice.message, "reasoning"):
                        reasoning = choice.message.reasoning
                        if reasoning:
                            logger.warning(
                                f"Content empty but found reasoning: {reasoning[:200]}..."
                            )

                    last_error = JSONParseError(
                        f"Empty response content. Model: {response.model}, "
                        f"Finish reason: {completion.finish_reason}"
                    )
                    if attempt < attempts - 1:
                        continue
                    raise last_error

                # Strip reasoning tags (e.g., <think>...</think>) from content
                content = self._strip_reasoning_tags(content)

                # Try to extract JSON from content (handle markdown code blocks)
                content = self._extract_json(content)

                try:
                    completion.data = json.loads(content)
                except json.JSONDecodeError as e:
                    last_error = JSONParseError(
                        f"Invalid JSON: {e}\nContent: {content[:500]}"
                    )
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
        provider: str | list[str] | None = None,
        reasoning_enabled: bool | None = None,
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
            provider: Provider preference for OpenRouter (e.g., "Fireworks" or ["Fireworks", "Together"]).
            reasoning_enabled: Enable/disable reasoning tokens (OpenRouter). None = don't specify.
            **kwargs: Additional parameters.

        Yields:
            Content chunks as strings.
        """
        model = model or self._default_model
        if not model:
            raise InvalidRequestError("No model specified and no default model set")

        prepared_messages = self._prepare_messages(messages)

        request_params = self._build_request_params(
            model=model,
            messages=prepared_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            stop=stop,
            stream=True,
        )

        extra_body = self._build_extra_body(
            provider, reasoning_enabled, kwargs.pop("extra_body", {})
        )
        if extra_body:
            kwargs["extra_body"] = extra_body

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

    # --- Private Methods ---

    def _build_request_params(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        stop: list[str] | str | None = None,
        stream: bool = False,
    ) -> dict[str, Any]:
        """Build request parameters for OpenAI API call."""
        params: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if stream:
            params["stream"] = True
        if temperature is not None:
            params["temperature"] = temperature
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if tools is not None:
            params["tools"] = tools
        if tool_choice is not None:
            params["tool_choice"] = tool_choice
        if stop is not None:
            params["stop"] = stop
        return params

    def _build_extra_body(
        self,
        provider: str | list[str] | None,
        reasoning_enabled: bool | None,
        existing: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build extra_body dict for OpenRouter-specific options."""
        extra_body = existing or {}
        if provider is not None:
            order = [provider] if isinstance(provider, str) else provider
            extra_body["provider"] = {"order": order, "allow_fallbacks": True}
        if reasoning_enabled:
            extra_body["reasoning"] = {"enabled": True}
        return extra_body

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
        if not response.choices:
            raise LLMError("Empty response from API: no choices returned")
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

        # 5xx server errors are retryable
        if status and 500 <= status < 600:
            return ServerError(message, status)

        return LLMError(message, status)

    def _strip_reasoning_tags(self, content: str) -> str:
        """Strip reasoning tags (e.g., <think>...</think>) from content."""
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
        content = re.sub(r"<reasoning>.*?</reasoning>", "", content, flags=re.DOTALL)
        return content.strip()

    def _extract_json(self, content: str) -> str:
        """Extract JSON from content, handling markdown code blocks."""
        # Try to extract from markdown code blocks first
        # Match ```json ... ``` or ``` ... ```
        json_block = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
        if json_block:
            return json_block.group(1).strip()

        # Try to find JSON object or array directly
        # Look for content starting with { or [
        content = content.strip()
        if content.startswith("{") or content.startswith("["):
            return content

        # Try to find JSON anywhere in the content
        json_match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", content)
        if json_match:
            return json_match.group(1)

        return content
