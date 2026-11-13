# simplicity-chain

Tiny agents that act by writing Python code, inspired by
[smolagents](https://github.com/huggingface/smolagents). The core stays readable:
**CodeAgent**, **`@tool`**, `final_answer`, and one HTTP model backend. One runtime
dependency (`httpx`), Python 3.10+.

## Install

```bash
git clone <this repo> && cd simplicity-chain
uv sync        # or: python -m venv .venv && pip install -e ".[dev]"
```

## Quick start

```python
from simplicity_chain import CodeAgent, OpenAIServerModel, tool


@tool
def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: The first number.
        b: The second number.
    """
    return a + b


model = OpenAIServerModel("gpt-4o-mini")  # reads OPENAI_API_KEY
agent = CodeAgent(tools=[add], model=model, max_steps=5)
print(agent.run("What is 19 + 23? Use the tool."))
```

The model replies with a fenced Python block. Code runs in a persistent session where
tools are plain functions; whatever it prints, plus the last expression, becomes the
observation. Calling `final_answer(...)` ends the run. Everything is inspectable in
`agent.memory.steps`, or request `return_full_result=True` to get a `RunResult` with
`output`, `steps`, `token_usage`, and `timing` (`to_dict()` is JSON-serializable).

## Providers

`OpenAIServerModel` speaks `/chat/completions`, so it works with any compatible server:

| Provider | Configuration |
| --- | --- |
| OpenAI | `OpenAIServerModel("gpt-4o-mini")` + `OPENAI_API_KEY` |
| DeepSeek | `OpenAIServerModel("deepseek-chat", api_base="https://api.deepseek.com/v1")` |
| OpenRouter | `OpenAIServerModel("openai/gpt-4o", api_base="https://openrouter.ai/api/v1")` |
| Ollama | `OpenAIServerModel("qwen3:8b", api_base="http://localhost:11434/v1", api_key="none")` |

Plug in anything else by subclassing `simplicity_chain.Model`.

## Execution and security

The executor allows a small stdlib allowlist (`math`, `json`, `re`, `datetime`, ...) plus
`additional_authorized_imports`, and blocks `open`, `eval`, `exec`, `getattr`/`setattr`,
dangerous dunder attributes, and other imports. A best-effort timeout stops runaway loops.

> **Warning:** `LocalPythonExecutor` is **not a security sandbox**. It prevents common
> mistakes, not a motivated escape. For untrusted models or inputs, use a real container
> or VM; remote executors are on the roadmap.

Parsing and execution errors are not fatal: the error becomes the next observation so the
agent can fix its code. Three consecutive parsing failures, or exhausting `max_steps`,
abort the run with an explicit exception.

## CLI and examples

```bash
export OPENAI_API_KEY=sk-...
uv run simplicity-chain "What is (21 + 21) * 2?" --max-steps 6
uv run simplicity-chain "List 3 facts about Mars" --model qwen3:8b \
  --api-base http://localhost:11434/v1 --api-key none   # local Ollama
```

[`examples/offline_demo.py`](examples/offline_demo.py) runs a full two-step loop with a
scripted model and no network. [`examples/live_demo.py`](examples/live_demo.py) calls a real
endpoint. `python -m simplicity_chain` is the same entry point as the console script.

## Development

```bash
uv sync
uv run pytest -q                    # 75 tests, fully offline
uv run pytest --cov=simplicity_chain  # ~87% line coverage
uv run ruff check . && uv run ruff format --check .
```

## Not included (yet)

Streaming, planning steps, managed agents, JSON tool-calling agents, save/load, MCP, and
remote sandboxes are deliberately deferred so the core stays small.

## License

MIT. An independent implementation inspired by smolagents (Apache-2.0).
