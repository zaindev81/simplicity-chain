from __future__ import annotations

import json

import pytest

from simplicity_chain import (
    ActionStep,
    AgentGenerationError,
    AgentMaxStepsError,
    AgentParsingError,
    ChatMessage,
    CodeAgent,
    RunResult,
    Tool,
)
from tests.support import ScriptedModel, add, code_block, word_count


def make_agent(outputs, *, tools=None, **kwargs) -> tuple[CodeAgent, ScriptedModel]:
    model = kwargs.pop("model", None) or ScriptedModel(outputs)
    agent = CodeAgent(
        tools=[add, word_count] if tools is None else tools,
        model=model,
        verbose=False,
        **kwargs,
    )
    return agent, model


def action_steps(agent: CodeAgent) -> list[ActionStep]:
    return [step for step in agent.memory.steps if isinstance(step, ActionStep)]


def test_happy_path_two_steps():
    agent, model = make_agent(
        [
            code_block("print(add(2, 3))"),
            code_block("final_answer(add(2, 3))"),
        ]
    )
    assert agent.run("add 2 and 3") == 5
    assert len(agent.memory.steps) == 5
    steps = action_steps(agent)
    assert len(steps) == 2
    assert "5" in (steps[0].observations or "")
    assert steps[1].is_final_answer
    assert model.calls[1][-1].role == "user"
    assert (model.calls[1][-1].content or "").startswith("Observation:")


def test_tool_call_from_code():
    agent, _ = make_agent(
        [code_block('final_answer(word_count("one two three"))')],
        tools=[word_count],
    )
    assert agent.run("count words") == 3


def test_single_tool_can_be_passed_directly():
    agent, _ = make_agent(
        [code_block("final_answer(add(1, 1))")],
        tools=add,
    )
    assert agent.run("one plus one") == 2


def test_parsing_error_is_recovered():
    agent, _ = make_agent(
        [
            "I should compute this, but I forgot the code block.",
            code_block("final_answer(4)"),
        ]
    )
    assert agent.run("what is 2 + 2?") == 4
    first = action_steps(agent)[0]
    assert isinstance(first.error, AgentParsingError)
    assert "Error in code parsing" in (first.observations or "")


def test_three_consecutive_parsing_errors_abort():
    model = ScriptedModel(["no code", "still no code", "nope"])
    agent = CodeAgent(tools=[], model=model, verbose=False)
    with pytest.raises(AgentParsingError, match="consecutive"):
        agent.run("do something")
    assert len(model.calls) == 3
    assert len(action_steps(agent)) == 3


def test_execution_error_is_recovered_and_logs_are_kept():
    agent, _ = make_agent(
        [
            code_block("print('before')\n1 / 0"),
            code_block("final_answer('recovered')"),
        ]
    )
    assert agent.run("break then recover") == "recovered"
    first = action_steps(agent)[0]
    assert "ZeroDivisionError" in (first.observations or "")
    assert "before" in (first.observations or "")


def test_generation_error_is_fatal():
    model = ScriptedModel([], error=RuntimeError("upstream boom"))
    agent = CodeAgent(tools=[], model=model, verbose=False)
    with pytest.raises(AgentGenerationError, match="upstream boom"):
        agent.run("anything")
    assert len(agent.memory.steps) == 3


def test_max_steps_raises_with_memory():
    agent, _ = make_agent(
        [code_block("x = 1"), code_block("x = x + 1")],
        max_steps=2,
    )
    with pytest.raises(AgentMaxStepsError) as info:
        agent.run("loop forever")
    assert info.value.memory is agent.memory
    assert len(action_steps(agent)) == 2
    assert agent.step_number == 2


def test_run_result_contains_usage_and_serializable_steps():
    agent, _ = make_agent([code_block("final_answer(7)")])
    result = agent.run("return seven", return_full_result=True)
    assert isinstance(result, RunResult)
    assert result.output == 7
    assert result.state == "success"
    assert result.token_usage is not None
    assert result.token_usage.input_tokens == 10
    assert result.token_usage.output_tokens == 5
    assert len(result.steps) == 4
    assert result.steps[0]["type"] == "system_prompt"
    json.dumps(result.to_dict())


class NoUsageModel(ScriptedModel):
    def generate(self, messages, *, stop_sequences=None) -> ChatMessage:
        message = super().generate(messages, stop_sequences=stop_sequences)
        message.token_usage = None
        return message


def test_run_result_token_usage_is_none_when_any_step_lacks_it():
    model = NoUsageModel([code_block("final_answer(1)")])
    agent = CodeAgent(tools=[], model=model, verbose=False)
    result = agent.run("one", return_full_result=True)
    assert isinstance(result, RunResult)
    assert result.token_usage is None


def test_reset_clears_memory_and_executor_state():
    agent, _ = make_agent(
        [
            code_block("secret = 41"),
            code_block("final_answer(secret)"),
            code_block("final_answer(secret)"),
            code_block("final_answer('ok')"),
        ]
    )
    assert agent.run("set a secret") == 41
    assert agent.run("read the secret") == "ok"
    assert "secret" not in agent.python_executor.state
    assert len(agent.memory.steps) == 5


def test_run_without_reset_keeps_history():
    agent, model = make_agent([code_block("final_answer(1)"), code_block("final_answer(2)")])
    assert agent.run("first task") == 1
    assert agent.run("second task", reset=False) == 2
    assert len(agent.memory.steps) == 7
    assert any(message.content == "second task" for message in model.calls[1])
    assert len(model.calls) == 2


def test_system_prompt_lists_tools_and_rules():
    agent, _ = make_agent([code_block("final_answer(1)")])
    prompt = agent.system_prompt
    assert "add(a: integer, b: integer) -> integer" in prompt
    assert "final_answer" in prompt
    assert "Authorized imports" in prompt
    assert "```python" in prompt


def test_additional_imports_reach_prompt_and_executor():
    agent, _ = make_agent(
        [code_block("final_answer(1)")],
        tools=[],
        additional_authorized_imports=["pandas"],
    )
    assert "pandas" in agent.system_prompt
    assert "pandas" in agent.authorized_imports


def test_instructions_reach_system_prompt():
    agent, _ = make_agent(
        [code_block("final_answer(1)")],
        tools=[],
        instructions="Always answer in French.",
    )
    assert "Always answer in French." in agent.system_prompt


def test_verbose_mode_prints_steps(capsys):
    agent, _ = make_agent([code_block("final_answer(7)")])
    agent.verbose = True
    agent.run("seven")
    output = capsys.readouterr().out
    assert "--- Step 1 ---" in output
    assert "Code:" in output
    assert "Final answer: 7" in output


def test_validation_errors():
    model = ScriptedModel([])
    with pytest.raises(ValueError, match="Duplicate"):
        CodeAgent(tools=[add, add], model=model, verbose=False)
    with pytest.raises(TypeError, match="Tool"):
        CodeAgent(tools=[lambda x: x], model=model, verbose=False)
    with pytest.raises(TypeError, match="Model"):
        CodeAgent(tools=[], model=object(), verbose=False)
    with pytest.raises(ValueError, match="max_steps"):
        CodeAgent(tools=[], model=model, max_steps=0, verbose=False)

    reserved = Tool(
        name="final_answer",
        description="",
        inputs={},
        output_type="any",
        func=lambda value=None: value,
    )
    with pytest.raises(ValueError, match="reserved"):
        CodeAgent(tools=[reserved], model=model, verbose=False)

    agent = CodeAgent(tools=[], model=model, verbose=False)
    with pytest.raises(ValueError, match="task"):
        agent.run("   ")
