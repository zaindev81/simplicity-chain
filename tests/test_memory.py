from __future__ import annotations

import json

from simplicity_chain import ActionStep, AgentMemory, FinalAnswerStep, SystemPromptStep, TaskStep
from tests.support import code_block


def build_memory() -> AgentMemory:
    memory = AgentMemory()
    memory.steps.extend(
        [
            SystemPromptStep(system_prompt="You are an agent."),
            TaskStep(task="Compute 1 + 1."),
            ActionStep(
                step_number=1,
                model_output=code_block("print(1 + 1)"),
                observations="Execution logs:\n2",
            ),
            FinalAnswerStep(output=2),
        ]
    )
    return memory


def test_memory_converts_to_chat_messages():
    messages = build_memory().write_memory_to_messages()
    assert [message.role for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert messages[0].content == "You are an agent."
    assert messages[1].content == "Compute 1 + 1."
    assert messages[2].content == "```python\nprint(1 + 1)\n```"
    assert messages[3].content == "Observation:\nExecution logs:\n2"


def test_final_answer_step_is_not_sent_to_the_model():
    assert FinalAnswerStep(output=2).to_messages() == []


def test_action_step_without_output_only_sends_observation():
    step = ActionStep(step_number=1, observations="Error in code parsing: nope")
    messages = step.to_messages()
    assert len(messages) == 1
    assert messages[0].role == "user"


def test_full_steps_are_serializable():
    memory = build_memory()
    steps = memory.get_full_steps()
    assert [step["type"] for step in steps] == [
        "system_prompt",
        "task",
        "action",
        "final_answer",
    ]
    json.dumps(steps)
    assert steps[2]["code_action"] is None
    assert steps[2]["observations"] == "Execution logs:\n2"


def test_reset_clears_memory():
    memory = build_memory()
    memory.reset()
    assert memory.steps == []
    assert memory.system_prompt == ""
