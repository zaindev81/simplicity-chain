"""simplicity-chain: tiny agents that act by writing Python code.

Example::

    from simplicity_chain import CodeAgent, OpenAIServerModel, tool

    @tool
    def add(a: int, b: int) -> int:
        "Add two numbers."
        return a + b

    agent = CodeAgent(tools=[add], model=OpenAIServerModel("gpt-4o-mini"))
    agent.run("What is 21 + 21?")
"""

from .agents import CodeAgent, MultiStepAgent
from .errors import (
    AgentError,
    AgentExecutionError,
    AgentGenerationError,
    AgentMaxStepsError,
    AgentParsingError,
    AgentToolError,
    ModelError,
)
from .executor import CodeOutput, LocalPythonExecutor
from .memory import (
    ActionStep,
    AgentMemory,
    FinalAnswerStep,
    SystemPromptStep,
    TaskStep,
)
from .models import Model, OpenAIServerModel
from .parser import fix_final_answer_code, parse_code_blobs
from .tools import Tool, tool
from .types import ChatMessage, RunResult, Timing, TokenUsage

__version__ = "0.1.0"

__all__ = [
    "ActionStep",
    "AgentError",
    "AgentExecutionError",
    "AgentGenerationError",
    "AgentMaxStepsError",
    "AgentMemory",
    "AgentParsingError",
    "AgentToolError",
    "ChatMessage",
    "CodeAgent",
    "CodeOutput",
    "FinalAnswerStep",
    "LocalPythonExecutor",
    "Model",
    "ModelError",
    "MultiStepAgent",
    "OpenAIServerModel",
    "RunResult",
    "SystemPromptStep",
    "TaskStep",
    "Timing",
    "TokenUsage",
    "Tool",
    "__version__",
    "fix_final_answer_code",
    "parse_code_blobs",
    "tool",
]
