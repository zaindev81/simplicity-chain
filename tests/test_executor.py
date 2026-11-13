from __future__ import annotations

import signal

import pytest

from simplicity_chain import LocalPythonExecutor
from simplicity_chain.errors import AgentExecutionError
from tests.support import add

needs_sigalrm = pytest.mark.skipif(
    not hasattr(signal, "SIGALRM"), reason="timeout requires SIGALRM"
)


def make_executor(**kwargs):
    kwargs.setdefault("execution_timeout", None)
    return LocalPythonExecutor(**kwargs)


def test_final_answer_returns_value():
    executor = make_executor()
    output = executor("final_answer(42)")
    assert output.is_final_answer
    assert output.output == 42
    assert output.logs == ""


def test_prints_are_captured():
    executor = make_executor()
    output = executor("print('hello')\nprint('world')\nfinal_answer(1)")
    assert output.logs == "hello\nworld\n"
    assert output.output == 1


def test_last_expression_is_returned():
    executor = make_executor()
    assert executor("2 + 3").output == 5
    assert executor("x = 1").output is None


def test_state_persists_between_calls():
    executor = make_executor()
    executor("counter = 5")
    assert executor("final_answer(counter * 2)").output == 10


def test_reset_clears_state():
    executor = make_executor()
    executor("secret = 1")
    executor.reset()
    with pytest.raises(AgentExecutionError, match="NameError"):
        executor("final_answer(secret)")


def test_send_variables():
    executor = make_executor()
    executor.send_variables({"value": 10})
    assert executor("final_answer(value + 1)").output == 11


def test_tools_are_callable_from_code():
    executor = make_executor()
    executor.send_tools({"add": add})
    assert executor("final_answer(add(2, 3))").output == 5


def test_invalid_tool_arguments_surface_as_execution_error():
    executor = make_executor()
    executor.send_tools({"add": add})
    with pytest.raises(AgentExecutionError, match="expected integer"):
        executor("add('2', 3)")


def test_authorized_imports():
    executor = make_executor()
    assert "math" in executor.authorized_imports
    assert executor("import math\nfinal_answer(math.floor(2.7))").output == 2


def test_additional_authorized_imports():
    executor = make_executor(additional_authorized_imports=["statistics"])
    output = executor("import statistics\nfinal_answer(statistics.mean([2, 4]))")
    assert output.output == 3


def test_forbidden_import_is_reported_with_guidance():
    executor = make_executor()
    with pytest.raises(AgentExecutionError, match="additional_authorized_imports"):
        executor("import os")


def test_forbidden_builtin_is_blocked():
    executor = make_executor()
    with pytest.raises(AgentExecutionError, match="not allowed"):
        executor("open('secrets.txt')")


def test_forbidden_attribute_is_blocked():
    executor = make_executor()
    with pytest.raises(AgentExecutionError, match="__class__"):
        executor("final_answer((1).__class__)")


def test_syntax_error_is_reported():
    executor = make_executor()
    with pytest.raises(AgentExecutionError, match="SyntaxError"):
        executor("def broken(")


def test_empty_code_is_rejected():
    executor = make_executor()
    with pytest.raises(AgentExecutionError, match="No code"):
        executor("   ")


def test_runtime_errors_are_wrapped_with_logs():
    executor = make_executor()
    with pytest.raises(AgentExecutionError, match="ZeroDivisionError") as info:
        executor("print('before')\n1 / 0")
    assert "before" in info.value.logs


def test_print_output_is_truncated():
    executor = make_executor(max_print_outputs_length=10)
    output = executor("print('abcdefghijklmnop')\nfinal_answer(1)")
    assert "truncated" in output.logs


def test_relative_imports_are_rejected():
    executor = make_executor()
    with pytest.raises(AgentExecutionError, match="Relative imports"):
        executor("from . import thing")


@needs_sigalrm
def test_infinite_loop_times_out():
    executor = LocalPythonExecutor(execution_timeout=0.5)
    with pytest.raises(AgentExecutionError, match="exceeded"):
        executor("while True:\n    pass")
