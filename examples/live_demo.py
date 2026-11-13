"""Talk to a real OpenAI-compatible endpoint.

Examples::

    export OPENAI_API_KEY=sk-...
    uv run python examples/live_demo.py

    # Ollama, no key required:
    export OPENAI_BASE_URL=http://localhost:11434/v1
    export OPENAI_API_KEY=none
    export SIMPLICITY_CHAIN_MODEL=qwen3:8b
    uv run python examples/live_demo.py
"""

from __future__ import annotations

import os

from simplicity_chain import CodeAgent, OpenAIServerModel, tool


@tool
def word_count(text: str) -> int:
    """Count the number of words in a piece of text."""
    return len(text.split())


def main() -> None:
    model = OpenAIServerModel(os.environ.get("SIMPLICITY_CHAIN_MODEL", "gpt-4o-mini"))
    agent = CodeAgent(tools=[word_count], model=model, max_steps=6)
    answer = agent.run(
        "How many words are in the sentence 'the quick brown fox jumps over the lazy dog'? "
        "Use the word_count tool and finish with final_answer."
    )
    print(f"\nFinal answer: {answer}")


if __name__ == "__main__":
    main()
