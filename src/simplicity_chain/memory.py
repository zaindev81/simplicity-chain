"""Agent memory: the step history that becomes the model's chat context."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .types import ChatMessage, Timing, TokenUsage, json_safe


@dataclass
class SystemPromptStep:
    system_prompt: str

    def to_messages(self) -> list[ChatMessage]:
        return [ChatMessage(role="system", content=self.system_prompt)]

    def to_dict(self) -> dict[str, Any]:
        return {"type": "system_prompt", "system_prompt": self.system_prompt}


@dataclass
class TaskStep:
    task: str

    def to_messages(self) -> list[ChatMessage]:
        return [ChatMessage(role="user", content=self.task)]

    def to_dict(self) -> dict[str, Any]:
        return {"type": "task", "task": self.task}


@dataclass
class ActionStep:
    step_number: int
    timing: Timing = field(default_factory=Timing)
    model_input_messages: list[ChatMessage] | None = None
    model_output: str | None = None
    code_action: str | None = None
    observations: str | None = None
    action_output: Any = None
    is_final_answer: bool = False
    error: BaseException | None = None
    token_usage: TokenUsage | None = None

    def to_messages(self) -> list[ChatMessage]:
        messages: list[ChatMessage] = []
        if self.model_output:
            messages.append(ChatMessage(role="assistant", content=self.model_output))
        if self.observations:
            messages.append(ChatMessage(role="user", content=f"Observation:\n{self.observations}"))
        return messages

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "action",
            "step_number": self.step_number,
            "model_input_messages": (
                [message.to_dict() for message in self.model_input_messages]
                if self.model_input_messages is not None
                else None
            ),
            "model_output": self.model_output,
            "code_action": self.code_action,
            "observations": self.observations,
            "action_output": json_safe(self.action_output),
            "is_final_answer": self.is_final_answer,
            "error": f"{type(self.error).__name__}: {self.error}" if self.error else None,
            "token_usage": (self.token_usage.to_dict() if self.token_usage is not None else None),
            "timing": self.timing.to_dict(),
        }


@dataclass
class FinalAnswerStep:
    output: Any

    def to_messages(self) -> list[ChatMessage]:
        return []

    def to_dict(self) -> dict[str, Any]:
        return {"type": "final_answer", "output": json_safe(self.output)}


MemoryStep = SystemPromptStep | TaskStep | ActionStep | FinalAnswerStep


class AgentMemory:
    """Ordered list of steps taken by an agent."""

    def __init__(self, system_prompt: str = "") -> None:
        self.system_prompt = system_prompt
        self.steps: list[MemoryStep] = []

    def reset(self) -> None:
        self.system_prompt = ""
        self.steps = []

    def write_memory_to_messages(self) -> list[ChatMessage]:
        """Convert every step into the chat messages passed to the model."""
        messages: list[ChatMessage] = []
        for step in self.steps:
            messages.extend(step.to_messages())
        return messages

    def get_full_steps(self) -> list[dict[str, Any]]:
        return [step.to_dict() for step in self.steps]
