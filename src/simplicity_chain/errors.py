"""Exception hierarchy for simplicity-chain."""

from __future__ import annotations

from typing import Any


class AgentError(Exception):
    """Base class for every error raised by the agent loop."""

    def __init__(self, message: str, *, agent: Any | None = None) -> None:
        super().__init__(message)
        self.agent = agent


class ModelError(AgentError):
    """The model backend could not be configured or called."""


class AgentGenerationError(AgentError):
    """The model failed while generating an action. This is fatal for a run."""


class AgentParsingError(AgentError):
    """The model output could not be parsed into an executable action."""


class AgentExecutionError(AgentError):
    """The generated action raised an error while executing."""

    def __init__(self, message: str, *, logs: str = "", agent: Any | None = None) -> None:
        super().__init__(message, agent=agent)
        self.logs = logs


class AgentToolError(AgentError):
    """A tool was called with invalid arguments."""


class AgentMaxStepsError(AgentError):
    """The agent used all its steps without producing a final answer."""

    def __init__(self, message: str, *, memory: Any | None = None) -> None:
        super().__init__(message)
        self.memory = memory
