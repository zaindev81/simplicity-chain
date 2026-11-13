from __future__ import annotations

from typing import Literal

import pytest

from simplicity_chain import Tool, tool
from simplicity_chain.errors import AgentToolError
from tests.support import add, make_tool


def test_decorator_derives_schema_from_hints_and_docstring():
    assert add.name == "add"
    assert add.description == "Add two integers."
    assert add.inputs["a"] == {"description": "The first number.", "type": "integer"}
    assert add.inputs["b"] == {"description": "The second number.", "type": "integer"}
    assert add.output_type == "integer"


def test_call_and_argument_validation():
    assert add(2, 3) == 5
    assert add(a=1, b=2) == 3
    with pytest.raises(AgentToolError):
        add("2", 3)
    with pytest.raises(AgentToolError):
        add(2)
    with pytest.raises(AgentToolError):
        add(2, 3, 4)
    with pytest.raises(AgentToolError):
        add(2, 3, c=4)


def test_defaults_make_arguments_optional():
    @tool
    def greet(name: str, greeting: str = "hello") -> str:
        "Greet a person."
        return f"{greeting} {name}"

    assert greet("Ada") == "hello Ada"
    assert greet.inputs["greeting"]["type"] == "string"
    with pytest.raises(AgentToolError):
        greet(1)


def test_custom_name_and_description():
    @tool(name="plus", description="Add two floats.")
    def add_floats(x: float, y: float) -> float:
        return x + y

    assert add_floats.name == "plus"
    assert add_floats.description == "Add two floats."
    assert add_floats.inputs["x"]["type"] == "number"
    assert add_floats.output_type == "number"


def test_nullable_literal_and_collection_annotations():
    @tool
    def puzzle(
        maybe: int | None,
        names: list[str],
        mapping: dict[str, int],
        mode: Literal["fast", "slow"],
    ) -> list[str]:
        return names

    assert puzzle.inputs["maybe"] == {"description": "", "type": "integer", "nullable": True}
    assert puzzle.inputs["names"] == {
        "description": "",
        "type": "array",
        "items": {"type": "string"},
    }
    assert puzzle.inputs["mapping"] == {
        "description": "",
        "type": "object",
        "additionalProperties": {"type": "integer"},
    }
    assert puzzle.inputs["mode"]["enum"] == ["fast", "slow"]
    assert puzzle.output_type == "array"


def test_optional_argument_accepts_none():
    @tool
    def maybe(value: int | None = None) -> str:
        return "none" if value is None else str(value)

    assert maybe() == "none"
    assert maybe(None) == "none"
    assert maybe(3) == "3"


def test_unsupported_annotations_fail_fast():
    with pytest.raises(TypeError):

        @tool
        def bad(x: tuple[int, int]) -> int:
            return 0

    with pytest.raises(TypeError):

        @tool
        def bad_union(x: int | str) -> int:
            return 0


def test_varargs_are_rejected():
    with pytest.raises(TypeError):

        @tool
        def bad(*args: int) -> int:
            return 0


def test_invalid_names_are_rejected():
    with pytest.raises(ValueError):

        @tool(name="bad name")
        def valid_function(x: int) -> int:
            return x

    with pytest.raises(ValueError):

        @tool(name="class")
        def another(x: int) -> int:
            return x


def test_description_fallback_and_to_dict():
    @tool
    def no_doc(value: str) -> str:
        return value

    assert no_doc.description == "Tool no doc."
    payload = no_doc.to_dict()
    assert payload["name"] == "no_doc"
    assert payload["inputs"]["value"]["type"] == "string"


def test_manual_tool_validation():
    with pytest.raises(ValueError):
        make_tool("broken", lambda: 1, inputs={"nope": {"type": "string"}})
    with pytest.raises(TypeError):
        make_tool("broken", "not callable")


def test_tool_decorator_without_parentheses_returns_tool():
    @tool
    def identity(value: int) -> int:
        "Return the value."
        return value

    assert isinstance(identity, Tool)
    assert identity(5) == 5
