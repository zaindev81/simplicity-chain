"""Model backends. ``OpenAIServerModel`` talks to any OpenAI-compatible HTTP endpoint."""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from typing import Any

import httpx

from .errors import ModelError
from .types import ChatMessage, TokenUsage

DEFAULT_API_BASE = "https://api.openai.com/v1"
_RETRYABLE_STATUS_CODES = frozenset({408, 409, 429, 500, 502, 503, 504})


class Model(ABC):
    """Minimal interface every model backend must implement."""

    model_id: str

    @abstractmethod
    def generate(
        self,
        messages: list[ChatMessage],
        *,
        stop_sequences: list[str] | None = None,
    ) -> ChatMessage:
        """Return the next assistant message for the given conversation."""

    def to_dict(self) -> dict[str, Any]:
        return {"type": type(self).__name__, "model_id": self.model_id}


class OpenAIServerModel(Model):
    """Call any OpenAI-compatible ``/chat/completions`` endpoint.

    Works with OpenAI, DeepSeek, OpenRouter, Together, Ollama (``api_key="none"``) and
    local vLLM servers. The API base defaults to ``$OPENAI_BASE_URL`` and the key to
    ``$OPENAI_API_KEY``.
    """

    def __init__(
        self,
        model_id: str,
        *,
        api_base: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        timeout: float = 60.0,
        max_retries: int = 2,
        retry_backoff: float = 0.5,
        http_client: httpx.Client | None = None,
    ) -> None:
        if not model_id:
            raise ValueError("model_id must be a non-empty string.")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0.")
        self.model_id = model_id
        self.api_base = (api_base or os.environ.get("OPENAI_BASE_URL") or DEFAULT_API_BASE).rstrip(
            "/"
        )
        resolved_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ModelError(
                "No API key provided. Pass api_key=... or set the OPENAI_API_KEY "
                "environment variable (use api_key='none' for local servers)."
            )
        self.api_key = resolved_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self._owns_client = http_client is None
        self._client = http_client or httpx.Client(timeout=timeout)

    def generate(
        self,
        messages: list[ChatMessage],
        *,
        stop_sequences: list[str] | None = None,
    ) -> ChatMessage:
        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": [message.to_dict() for message in messages],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if stop_sequences:
            payload["stop"] = stop_sequences

        data = self._post("/chat/completions", payload)
        choices = data.get("choices") or []
        if not choices:
            raise ModelError(f"Model response contained no choices: {_short(json.dumps(data))}")
        message = choices[0].get("message") or {}
        usage_data = data.get("usage") or {}
        token_usage = None
        if usage_data:
            token_usage = TokenUsage(
                input_tokens=int(usage_data.get("prompt_tokens") or 0),
                output_tokens=int(usage_data.get("completion_tokens") or 0),
            )
        return ChatMessage(
            role="assistant",
            content=_extract_content(message.get("content")),
            token_usage=token_usage,
            raw=data,
        )

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.api_base}{path}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_error: str | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.post(url, headers=headers, json=payload)
            except httpx.HTTPError as error:
                last_error = f"Request to {url} failed: {error}"
                if attempt < self.max_retries:
                    self._sleep(attempt)
                    continue
                raise ModelError(last_error) from error

            if response.status_code >= 400:
                last_error = (
                    f"Model request failed with HTTP {response.status_code}: "
                    f"{_short(response.text)}"
                )
                if response.status_code in _RETRYABLE_STATUS_CODES and attempt < self.max_retries:
                    self._sleep(attempt)
                    continue
                raise ModelError(last_error)

            try:
                data = response.json()
            except ValueError as error:
                raise ModelError(
                    f"Model endpoint returned invalid JSON: {_short(response.text)}"
                ) from error
            if not isinstance(data, dict):
                raise ModelError("Model endpoint returned an unexpected JSON payload.")
            return data

        raise ModelError(last_error or "Model request failed.")

    def _sleep(self, attempt: int) -> None:
        if self.retry_backoff > 0:
            time.sleep(self.retry_backoff * (2**attempt))

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> OpenAIServerModel:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


def _extract_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    return str(content)


def _short(text: Any, limit: int = 500) -> str:
    value = str(text)
    if len(value) <= limit:
        return value
    return f"{value[:limit]}..."
