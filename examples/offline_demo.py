"""A complete agent loop with no network access.

Run it with::

    uv run python examples/offline_demo.py
"""

from __future__ import annotations

from simplicity_chain import ChatMessage, CodeAgent, Model, TokenUsage, tool


class ScriptedModel(Model):
    """Pretends to be an LLM by replaying two canned code actions."""

    model_id = "offline-script"

    def __init__(self) -> None:
        self.outputs = [
            "```python\nprint(add(21, 21))\n```",
            "```python\nfinal_answer(multiply(add(21, 21), 2))\n```",
        ]

    def generate(self, messages: list[ChatMessage], *, stop_sequences=None) -> ChatMessage:
        return ChatMessage(
            role="assistant",
            content=self.outputs.pop(0),
            token_usage=TokenUsage(input_tokens=50, output_tokens=20),
        )


@tool
def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: The first number.
        b: The second number.
    """
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers.

    Args:
        a: The first number.
        b: The second number.
    """
    return a * b


def main() -> None:
    model = ScriptedModel()
    agent = CodeAgent(tools=[add, multiply], model=model, max_steps=5)
    answer = agent.run("What is (21 + 21) * 2? Use the provided tools.")
    print(f"\nAnswer: {answer}")


if __name__ == "__main__":
    main()
