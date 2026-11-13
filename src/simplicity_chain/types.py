"""Core data types shared across the library."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

MessageRole: TypeAlias = Literal["system", "user", "assistant", "tool"]
JSONDict: TypeAlias = dict[str, Any]


@dataclass
class TokenUsage:
    """Token counts reported by a model provider."""

    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> JSONDict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class ChatMessage:
    """A single chat message, compatible with the OpenAI chat format."""

    role: MessageRole
    content: str | None = None
    tool_calls: list[JSONDict] | None = None
    token_usage: TokenUsage | None = None
    raw: Any = field(default=None, repr=False)

    def to_dict(self) -> JSONDict:
        message: JSONDict = {"role": self.role, "content": self.content or ""}
        if self.tool_calls:
            message["tool_calls"] = self.tool_calls
        return message


@dataclass
class Timing:
    """Wall-clock timing for a run or a step."""

    start_time: float = field(default_factory=time.time)
    end_time: float | None = None

    @property
    def duration(self) -> float:
        end = self.end_time if self.end_time is not None else time.time()
        return end - self.start_time

    def to_dict(self) -> JSONDict:
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
        }


def json_safe(value: Any) -> Any:
    """Best-effort conversion of arbitrary Python values to JSON-serializable data."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        pass
    else:
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    return repr(value)


@dataclass
class RunResult:
    """Extended information about an agent run."""

    output: Any
    steps: list[JSONDict] = field(default_factory=list)
    token_usage: TokenUsage | None = None
    timing: Timing = field(default_factory=Timing)
    state: str = "success"

    def to_dict(self) -> JSONDict:
        return {
            "output": json_safe(self.output),
            "steps": self.steps,
            "token_usage": self.token_usage.to_dict() if self.token_usage is not None else None,
            "timing": self.timing.to_dict(),
            "state": self.state,
        }
