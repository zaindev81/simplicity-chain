"""Shared test helpers: a scripted model and a couple of tools."""

from __future__ import annotations

from typing import Any

from simplicity_chain import ChatMessage, Model, TokenUsage, tool


class ScriptedModel(Model):
    """A fake model that returns pre-scripted outputs and records each call."""

    model_id = "scripted"

    def __init__(
        self,
        outputs: list[str],
        *,
        token_usage: TokenUsage | None = None,
        error: BaseException | None = None,
    ) -> None:
        self.outputs = list(outputs)
        self.token_usage = TokenUsage(10, 5) if token_usage is None else token_usage
        self.error = error
        self.calls: list[list[ChatMessage]] = []

    def generate(
        self,
        messages: list[ChatMessage],
        *,
        stop_sequences: list[str] | None = None,
    ) -> ChatMessage:
        self.calls.append(list(messages))
        if self.error is not None:
            raise self.error
        if not self.outputs:
            raise AssertionError("ScriptedModel ran out of scripted outputs")
        return ChatMessage(
            role="assistant",
            content=self.outputs.pop(0),
            token_usage=self.token_usage,
        )


def code_block(code: str) -> str:
    return f"```python\n{code}\n```"


@tool
def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: The first number.
        b: The second number.
    """
    return a + b


@tool
def word_count(text: str) -> int:
    """Count the words in a string."""
    return len(text.split())


def make_tool(name: str, func: Any, **overrides: Any) -> Any:
    """Build a Tool directly, for validation-edge tests."""
    from simplicity_chain import Tool

    defaults = {
        "name": name,
        "description": "",
        "inputs": {},
        "output_type": "any",
        "func": func,
    }
    defaults.update(overrides)
    return Tool(**defaults)
