"""Multi-step agents: the ReAct loop and the code-writing ``CodeAgent``."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from .errors import (
    AgentError,
    AgentExecutionError,
    AgentGenerationError,
    AgentMaxStepsError,
    AgentParsingError,
)
from .executor import LocalPythonExecutor
from .memory import ActionStep, AgentMemory, FinalAnswerStep, SystemPromptStep, TaskStep
from .models import Model
from .parser import fix_final_answer_code, parse_code_blobs
from .prompts import build_system_prompt
from .tools import Tool
from .types import RunResult, Timing, TokenUsage

MAX_CONSECUTIVE_PARSING_ERRORS = 3


@dataclass
class ActionOutput:
    """What one agent step produced."""

    output: Any = None
    is_final_answer: bool = False


class MultiStepAgent(ABC):
    """Base class for agents that solve a task over several steps."""

    def __init__(
        self,
        tools: list[Tool] | Tool | None,
        model: Model,
        *,
        instructions: str | None = None,
        max_steps: int = 20,
        verbose: bool = True,
    ) -> None:
        if not isinstance(model, Model):
            raise TypeError("model must be an instance of simplicity_chain.Model.")
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1.")
        tool_list = [tools] if isinstance(tools, Tool) else list(tools or [])
        if not all(isinstance(tool, Tool) for tool in tool_list):
            raise TypeError("All tools must be instances of simplicity_chain.Tool (use @tool).")
        names = [tool.name for tool in tool_list]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Duplicate tool names: {duplicates}.")
        if "final_answer" in names:
            raise ValueError("'final_answer' is reserved and cannot be used as a tool name.")

        self.tools: dict[str, Tool] = {tool.name: tool for tool in tool_list}
        self.model = model
        self.instructions = instructions
        self.max_steps = max_steps
        self.verbose = verbose
        self.memory = AgentMemory()
        self.step_number = 0
        self.task: str | None = None

    # ------------------------------------------------------------- interface
    @property
    def system_prompt(self) -> str:
        return self.initialize_system_prompt()

    @abstractmethod
    def initialize_system_prompt(self) -> str:
        """Build the system prompt describing available tools and rules."""

    @abstractmethod
    def _step(self, action_step: ActionStep) -> ActionOutput:
        """Run a single think/act/observe step."""

    def _reset_executor(self) -> None:
        """Reset any execution environment between runs."""

    # ------------------------------------------------------------------- run
    def run(
        self,
        task: str,
        *,
        reset: bool = True,
        max_steps: int | None = None,
        return_full_result: bool = False,
    ) -> Any | RunResult:
        """Run the agent until it calls ``final_answer`` or runs out of steps.

        Args:
            task: The task to solve.
            reset: Clear memory and execution state before starting.
            max_steps: Override the agent's configured step limit for this run.
            return_full_result: Return a :class:`RunResult` instead of the raw output.

        Raises:
            AgentMaxStepsError: if no final answer is produced within ``max_steps``.
        """
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be a non-empty string.")
        task = task.strip()
        steps_limit = self.max_steps if max_steps is None else max_steps
        if steps_limit < 1:
            raise ValueError("max_steps must be >= 1.")

        if reset:
            self.memory.reset()
            self.step_number = 0
            self._reset_executor()

        self.task = task
        self.memory.system_prompt = self.initialize_system_prompt()
        self._sync_system_prompt(self.memory.system_prompt)
        self.memory.steps.append(TaskStep(task=task))

        run_timing = Timing()
        parsing_failures = 0
        action_output: ActionOutput | None = None

        for step_number in range(1, steps_limit + 1):
            self.step_number = step_number
            action_step = ActionStep(step_number=step_number)
            self._log(f"\n--- Step {step_number} ---")
            try:
                action_output = self._step(action_step)
            except AgentGenerationError as error:
                action_step.error = error
                action_step.observations = f"Error while generating output: {error}"
            except AgentParsingError as error:
                parsing_failures += 1
                action_step.error = error
                action_step.observations = f"Error in code parsing: {error}"
            except AgentExecutionError as error:
                parsing_failures = 0
                action_step.error = error
                logs = f"Execution logs:\n{error.logs}\n" if error.logs else ""
                action_step.observations = f"{logs}Error while executing code: {error}"
            else:
                parsing_failures = 0
                action_step.action_output = action_output.output
                action_step.is_final_answer = action_output.is_final_answer
            finally:
                action_step.timing.end_time = time.time()
                self.memory.steps.append(action_step)
                self._log_action(action_step)

            if action_step.error is not None and isinstance(
                action_step.error, AgentGenerationError
            ):
                raise action_step.error
            if action_step.error is not None:
                if parsing_failures >= MAX_CONSECUTIVE_PARSING_ERRORS:
                    raise AgentParsingError(
                        f"Stopped after {MAX_CONSECUTIVE_PARSING_ERRORS} consecutive parsing "
                        f"errors. Last error: {action_step.error}"
                    )
                continue
            if action_output is not None and action_output.is_final_answer:
                self.memory.steps.append(FinalAnswerStep(output=action_output.output))
                return self._build_result(action_output.output, run_timing, return_full_result)

        raise AgentMaxStepsError(
            f"Reached max_steps={steps_limit} without calling final_answer.",
            memory=self.memory,
        )

    # --------------------------------------------------------------- helpers
    def _sync_system_prompt(self, prompt: str) -> None:
        if self.memory.steps and isinstance(self.memory.steps[0], SystemPromptStep):
            self.memory.steps[0] = SystemPromptStep(system_prompt=prompt)
        else:
            self.memory.steps.insert(0, SystemPromptStep(system_prompt=prompt))

    def _build_result(
        self, output: Any, timing: Timing, return_full_result: bool
    ) -> Any | RunResult:
        if not return_full_result:
            return output
        timing.end_time = time.time()
        return RunResult(
            output=output,
            steps=self.memory.get_full_steps(),
            token_usage=self._total_token_usage(),
            timing=timing,
        )

    def _total_token_usage(self) -> TokenUsage | None:
        action_steps = [step for step in self.memory.steps if isinstance(step, ActionStep)]
        if not action_steps or any(step.token_usage is None for step in action_steps):
            return None
        total = TokenUsage()
        for step in action_steps:
            if step.token_usage is not None:
                total = total + step.token_usage
        return total

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _log_action(self, action_step: ActionStep) -> None:
        if not self.verbose:
            return
        if action_step.code_action:
            print(f"Code:\n{action_step.code_action}")
        if action_step.observations:
            print(f"Observation:\n{_shorten(action_step.observations, 2_000)}")
        if action_step.is_final_answer:
            print(f"Final answer: {action_step.action_output}")


class CodeAgent(MultiStepAgent):
    """An agent that acts by writing Python code, smolagents-style.

    The model sees the conversation history, replies with a fenced Python code block,
    and the code runs in a persistent, lightly restricted Python session. Tool calls are
    ordinary function calls; ``final_answer(...)`` ends the run.
    """

    def __init__(
        self,
        tools: list[Tool] | Tool | None,
        model: Model,
        *,
        instructions: str | None = None,
        max_steps: int = 20,
        additional_authorized_imports: list[str] | None = None,
        executor: LocalPythonExecutor | None = None,
        max_print_outputs_length: int = 20_000,
        execution_timeout: float | None = 30.0,
        verbose: bool = True,
    ) -> None:
        super().__init__(
            tools,
            model,
            instructions=instructions,
            max_steps=max_steps,
            verbose=verbose,
        )
        self.additional_authorized_imports = list(additional_authorized_imports or [])
        self.max_print_outputs_length = max_print_outputs_length
        self.code_block_tags = ("```python", "```")
        self.stop_sequences = ["Observation:", "Calling tools:"]
        self.python_executor = executor or LocalPythonExecutor(
            self.additional_authorized_imports,
            max_print_outputs_length=max_print_outputs_length,
            execution_timeout=execution_timeout,
        )
        self._send_tools_to_executor()

    @property
    def authorized_imports(self) -> list[str]:
        return list(
            getattr(
                self.python_executor,
                "authorized_imports",
                self.additional_authorized_imports,
            )
        )

    def initialize_system_prompt(self) -> str:
        return build_system_prompt(
            self.tools,
            authorized_imports=self.authorized_imports,
            instructions=self.instructions,
        )

    def _send_tools_to_executor(self) -> None:
        send_tools = getattr(self.python_executor, "send_tools", None)
        if send_tools is not None:
            send_tools(self.tools)

    def _reset_executor(self) -> None:
        reset = getattr(self.python_executor, "reset", None)
        if reset is not None:
            reset()
        self._send_tools_to_executor()

    def _step(self, action_step: ActionStep) -> ActionOutput:
        messages = self.memory.write_memory_to_messages()
        action_step.model_input_messages = messages
        try:
            chat_message = self.model.generate(
                messages,
                stop_sequences=list(self.stop_sequences),
            )
        except AgentError:
            raise
        except Exception as error:
            raise AgentGenerationError(f"Error while generating model output: {error}") from error

        action_step.model_output = chat_message.content or ""
        action_step.token_usage = chat_message.token_usage

        code_action = parse_code_blobs(action_step.model_output, self.code_block_tags)
        code_action = fix_final_answer_code(code_action)
        action_step.code_action = code_action

        try:
            code_output = self.python_executor(code_action)
        except AgentExecutionError:
            raise
        except Exception as error:
            raise AgentExecutionError(f"{type(error).__name__}: {error}") from error

        action_step.observations = self._build_observation(code_output)
        action_step.action_output = code_output.output
        action_step.is_final_answer = code_output.is_final_answer
        return ActionOutput(
            output=code_output.output,
            is_final_answer=code_output.is_final_answer,
        )

    def _build_observation(self, code_output: Any) -> str:
        logs = code_output.logs.strip()
        lines = ["Execution logs:", logs or "<no output>"]
        if not code_output.is_final_answer:
            output = _stringify(code_output.output)
            if output:
                lines.extend(["Last output from code snippet:", output])
        return "\n".join(lines)


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    return _shorten(str(value), 20_000)


def _shorten(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n... [truncated {len(text) - limit} characters]"
