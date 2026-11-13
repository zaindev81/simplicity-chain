"""Tools: ordinary Python functions with a JSON-schema-like description."""

from __future__ import annotations

import inspect
import keyword
import re
import types
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

from .errors import AgentToolError

_PY_TO_JSON: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    type(None): "null",
}

_SECTION_HEADER = ("Args", "Arguments", "Parameters")
_STOP_HEADERS = ("Returns", "Return", "Raises", "Yields", "Examples", "Example", "Notes", "Note")


def _annotation_to_schema(annotation: Any) -> dict[str, Any]:
    """Convert a Python type annotation into a small JSON-schema-like dict."""
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {"type": "any"}
    if annotation is None:
        return {"type": "null"}
    if annotation in _PY_TO_JSON:
        return {"type": _PY_TO_JSON[annotation]}

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is Literal:
        values = list(args)
        schema: dict[str, Any] = {"enum": values}
        if all(isinstance(value, bool) for value in values):
            schema["type"] = "boolean"
        elif all(isinstance(value, str) for value in values):
            schema["type"] = "string"
        elif all(isinstance(value, int) for value in values):
            schema["type"] = "integer"
        return schema

    if origin is Union or origin is types.UnionType:
        non_none = [arg for arg in args if arg is not type(None)]
        if len(non_none) == 1:
            schema = dict(_annotation_to_schema(non_none[0]))
            if len(non_none) != len(args):
                schema["nullable"] = True
            return schema
        raise TypeError(f"Union types are not supported for tool arguments: {annotation!r}")

    if annotation is list or origin is list:
        item_schema = _annotation_to_schema(args[0]) if args else {"type": "any"}
        return {"type": "array", "items": item_schema}

    if annotation is dict or origin is dict:
        value_schema = _annotation_to_schema(args[1]) if len(args) > 1 else {"type": "any"}
        return {"type": "object", "additionalProperties": value_schema}

    raise TypeError(f"Unsupported type annotation for a tool argument: {annotation!r}")


def _matches_schema(value: Any, schema: dict[str, Any]) -> bool:
    expected = schema.get("type", "any")
    if expected == "any":
        return True
    if value is None:
        return expected == "null" or bool(schema.get("nullable"))
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "array":
        return isinstance(value, (list, tuple))
    if expected == "object":
        return isinstance(value, dict)
    if expected == "null":
        return value is None
    return True


def _parse_docstring(docstring: str | None) -> tuple[str, dict[str, str]]:
    """Return (description, {argument: description}) from a Google-style docstring."""
    if not docstring:
        return "", {}

    lines = inspect.cleandoc(docstring).splitlines()
    description_lines: list[str] = []
    argument_docs: dict[str, str] = {}
    in_arguments = False
    current_argument: str | None = None

    for line in lines:
        stripped = line.strip()
        if stripped.rstrip(":") in _SECTION_HEADER and stripped.endswith(":"):
            in_arguments = True
            current_argument = None
            continue

        if in_arguments:
            if not stripped:
                continue
            if not line[0].isspace():
                break
            match = re.match(r"([A-Za-z_]\w*)\s*(?:\([^)]*\))?\s*:\s*(.*)", stripped)
            if match:
                current_argument = match.group(1)
                argument_docs[current_argument] = match.group(2).strip()
            elif current_argument:
                argument_docs[current_argument] = (
                    f"{argument_docs[current_argument]} {stripped}".strip()
                )
            continue

        if stripped.rstrip(":") in _STOP_HEADERS and stripped.endswith(":"):
            break
        description_lines.append(line)

    description = "\n".join(description_lines).strip()
    return description, argument_docs


@dataclass
class Tool:
    """A callable tool the agent can invoke from generated code.

    Usually created with the :func:`tool` decorator, which derives ``inputs`` from the
    function signature and docstring.
    """

    name: str
    description: str
    inputs: dict[str, dict[str, Any]]
    output_type: str
    func: Callable[..., Any]
    signature: inspect.Signature = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or not self.name.isidentifier()
            or keyword.iskeyword(self.name)
        ):
            raise ValueError(
                f"Tool name {self.name!r} must be a valid Python identifier and not a keyword."
            )
        if not callable(self.func):
            raise TypeError(f"Tool {self.name!r}: 'func' must be callable.")
        self.signature = inspect.signature(self.func)
        unknown = set(self.inputs) - set(self.signature.parameters)
        if unknown:
            raise ValueError(f"Tool {self.name!r}: inputs {sorted(unknown)} are not parameters.")
        self.inputs = {name: {"description": "", **schema} for name, schema in self.inputs.items()}

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.validate_arguments(args, kwargs)
        return self.func(*args, **kwargs)

    def validate_arguments(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        """Validate a call against the tool signature and input schema."""
        try:
            bound = self.signature.bind(*args, **kwargs)
        except TypeError as exc:
            raise AgentToolError(f"Invalid arguments for tool {self.name!r}: {exc}") from exc
        for argument_name, value in bound.arguments.items():
            schema = self.inputs.get(argument_name)
            if schema is None:
                continue
            if value is None and schema.get("nullable"):
                continue
            if not _matches_schema(value, schema):
                raise AgentToolError(
                    f"Tool {self.name!r} argument {argument_name!r} expected "
                    f"{schema.get('type', 'any')} but got {type(value).__name__}."
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputs": self.inputs,
            "output_type": self.output_type,
        }

    @classmethod
    def from_function(
        cls,
        func: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> Tool:
        if not callable(func):
            raise TypeError(f"@tool expects a callable, got {type(func).__name__}.")
        signature = inspect.signature(func)
        try:
            hints = get_type_hints(func)
        except Exception:  # pragma: no cover - unresolved local forward references
            hints = getattr(func, "__annotations__", {})

        doc_description, argument_docs = _parse_docstring(func.__doc__)
        inputs: dict[str, dict[str, Any]] = {}
        for parameter_name, parameter in signature.parameters.items():
            if parameter.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                raise TypeError(
                    f"Tool {func.__name__!r} uses *args/**kwargs, which are not supported."
                )
            annotation = hints.get(parameter_name, parameter.annotation)
            schema = _annotation_to_schema(annotation)
            if argument_docs.get(parameter_name):
                schema["description"] = argument_docs[parameter_name]
            inputs[parameter_name] = schema

        return_annotation = hints.get("return", signature.return_annotation)
        output_schema = _annotation_to_schema(return_annotation)
        output_type = output_schema.get("type", "any")
        if output_type == "null" and "nullable" in output_schema:
            output_type = "null"

        tool_name = name or func.__name__
        tool_description = description or doc_description or f"Tool {tool_name.replace('_', ' ')}."
        return cls(
            name=tool_name,
            description=tool_description,
            inputs=inputs,
            output_type=output_type,
            func=func,
        )


def tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Any:
    """Decorator turning a function into a :class:`Tool`.

    Usage::

        @tool
        def add(a: int, b: int) -> int:
            "Add two integers."
            return a + b

        @tool(name="search", description="Search the web.")
        def web_search(query: str) -> str: ...
    """

    def decorate(function: Callable[..., Any]) -> Tool:
        return Tool.from_function(function, name=name, description=description)

    if func is None:
        return decorate
    if isinstance(func, Tool):
        return func
    return decorate(func)
