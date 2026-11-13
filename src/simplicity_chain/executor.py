"""A best-effort restricted Python executor for agent-generated code.

This is **not a security sandbox**. It filters the most common mistakes and dangerous
builtins, but determined code can escape it. Run untrusted models inside a real
container or VM if isolation matters.
"""

from __future__ import annotations

import ast
import builtins
import contextlib
import io
import signal
import threading
import traceback
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from .errors import AgentExecutionError
from .tools import Tool

BASE_BUILTIN_MODULES: tuple[str, ...] = (
    "base64",
    "collections",
    "csv",
    "datetime",
    "decimal",
    "fractions",
    "functools",
    "hashlib",
    "itertools",
    "json",
    "math",
    "random",
    "re",
    "statistics",
    "string",
    "textwrap",
    "time",
    "uuid",
)

ALLOWED_BUILTINS: tuple[str, ...] = (
    "ArithmeticError",
    "AssertionError",
    "AttributeError",
    "BaseException",
    "Exception",
    "ImportError",
    "IndexError",
    "KeyError",
    "LookupError",
    "NameError",
    "NotImplementedError",
    "OverflowError",
    "RuntimeError",
    "StopIteration",
    "TimeoutError",
    "TypeError",
    "ValueError",
    "ZeroDivisionError",
    "abs",
    "all",
    "any",
    "ascii",
    "bin",
    "bool",
    "bytearray",
    "bytes",
    "callable",
    "chr",
    "classmethod",
    "complex",
    "dict",
    "divmod",
    "enumerate",
    "filter",
    "float",
    "format",
    "frozenset",
    "hash",
    "hex",
    "int",
    "isinstance",
    "issubclass",
    "iter",
    "len",
    "list",
    "map",
    "max",
    "min",
    "next",
    "object",
    "oct",
    "ord",
    "pow",
    "property",
    "range",
    "repr",
    "reversed",
    "round",
    "set",
    "slice",
    "sorted",
    "staticmethod",
    "str",
    "sum",
    "super",
    "tuple",
    "type",
    "zip",
)

DENIED_BUILTIN_NAMES: frozenset[str] = frozenset(
    {
        "__builtins__",
        "__import__",
        "breakpoint",
        "compile",
        "delattr",
        "dir",
        "eval",
        "exec",
        "exit",
        "getattr",
        "globals",
        "help",
        "input",
        "locals",
        "open",
        "quit",
        "setattr",
        "vars",
    }
)

DENIED_ATTRIBUTES: frozenset[str] = frozenset(
    {
        "__bases__",
        "__builtins__",
        "__class__",
        "__closure__",
        "__code__",
        "__dict__",
        "__func__",
        "__getattr__",
        "__getattribute__",
        "__globals__",
        "__init_subclass__",
        "__loader__",
        "__mro__",
        "__reduce__",
        "__reduce_ex__",
        "__self__",
        "__setattr__",
        "__spec__",
        "__subclasses__",
        "__traceback__",
        "ag_frame",
        "cr_frame",
        "f_back",
        "f_globals",
        "f_locals",
        "gi_frame",
        "tb_frame",
    }
)


class FinalAnswerSignal(Exception):
    """Internal signal used by ``final_answer(...)`` to stop code execution."""

    def __init__(self, value: Any) -> None:
        super().__init__("final_answer was called")
        self.value = value


@dataclass
class CodeOutput:
    """Result of executing one code action."""

    output: Any = None
    logs: str = ""
    is_final_answer: bool = False


class LocalPythonExecutor:
    """Execute code in-process with filtered builtins, imports and attributes.

    Variables persist in ``state`` between calls, so the agent can split a task over
    multiple steps.
    """

    def __init__(
        self,
        additional_authorized_imports: list[str] | None = None,
        *,
        max_print_outputs_length: int = 20_000,
        execution_timeout: float | None = 30.0,
    ) -> None:
        self.additional_authorized_imports = list(additional_authorized_imports or [])
        if "*" in self.additional_authorized_imports:
            warnings.warn(
                "All imports are authorized: import filtering is effectively disabled.",
                UserWarning,
                stacklevel=2,
            )
        self.authorized_imports = sorted(
            set(BASE_BUILTIN_MODULES) | set(self.additional_authorized_imports)
        )
        self.max_print_outputs_length = max_print_outputs_length
        self.execution_timeout = execution_timeout
        self.tools: dict[str, Tool] = {}
        self.state: dict[str, Any] = {}
        self._buffer: io.StringIO | None = None
        self.reset()

    # ------------------------------------------------------------------ setup
    def reset(self) -> None:
        """Clear all variables and restore the base namespace."""
        self.state = {
            "__name__": "__simplicity_chain__",
            "__builtins__": self._make_builtins(),
            "final_answer": self._final_answer,
        }
        self.state.update(self.tools)

    def send_tools(self, tools: dict[str, Tool]) -> None:
        self.tools = dict(tools)
        self.state.update(self.tools)

    def send_variables(self, variables: dict[str, Any]) -> None:
        self.state.update(variables)

    # -------------------------------------------------------------- execution
    def __call__(self, code: str) -> CodeOutput:
        """Validate and execute one code action."""
        tree = self._validate(code)
        self._buffer = io.StringIO()
        try:
            output = self._run(tree)
            return CodeOutput(output=output, logs=self._get_logs())
        except FinalAnswerSignal as signal_error:
            return CodeOutput(
                output=signal_error.value,
                logs=self._get_logs(),
                is_final_answer=True,
            )
        except Exception as error:
            raise AgentExecutionError(_format_error(error), logs=self._get_logs()) from None
        finally:
            self._buffer = None

    def _run(self, tree: ast.Module) -> Any:
        with self._time_limit():
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                body = ast.Module(body=tree.body[:-1], type_ignores=[])
                last = ast.Expression(body=tree.body[-1].value)
                if body.body:
                    exec(compile(body, "<agent_code>", "exec"), self.state)
                return eval(compile(last, "<agent_code>", "eval"), self.state)
            exec(compile(tree, "<agent_code>", "exec"), self.state)
            return None

    def _validate(self, code: str) -> ast.Module:
        if not isinstance(code, str) or not code.strip():
            raise AgentExecutionError("No code to execute.")
        try:
            tree = ast.parse(code)
        except SyntaxError as error:
            location = f" (line {error.lineno})" if error.lineno else ""
            raise AgentExecutionError(f"SyntaxError: {error.msg}{location}") from None

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self._check_import(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    raise AgentExecutionError("Relative imports are not allowed.")
                if node.module:
                    self._check_import(node.module)
            elif isinstance(node, ast.Name) and node.id in DENIED_BUILTIN_NAMES:
                raise AgentExecutionError(
                    f"Use of {node.id!r} is not allowed. Write plain Python using the "
                    "available tools and authorized imports."
                )
            elif isinstance(node, ast.Attribute) and node.attr in DENIED_ATTRIBUTES:
                raise AgentExecutionError(f"Access to attribute {node.attr!r} is not allowed.")
        return tree

    def _check_import(self, module_name: str) -> None:
        root = module_name.split(".")[0]
        if root not in self.authorized_imports and "*" not in self.authorized_imports:
            authorized = ", ".join(name for name in self.authorized_imports if name != "*")
            raise AgentExecutionError(
                f"Import of {root!r} is not allowed. Authorized imports: {authorized}. "
                "Add it with additional_authorized_imports=[...] when creating the agent."
            )

    # ---------------------------------------------------------------- helpers
    def _final_answer(self, value: Any) -> None:
        raise FinalAnswerSignal(value)

    def _make_builtins(self) -> dict[str, Any]:
        allowed = {
            name: getattr(builtins, name) for name in ALLOWED_BUILTINS if hasattr(builtins, name)
        }
        allowed["__build_class__"] = builtins.__build_class__
        allowed["__import__"] = self._safe_import
        allowed["print"] = self._print
        return allowed

    def _safe_import(
        self,
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        root = name.split(".")[0]
        if root not in self.authorized_imports and "*" not in self.authorized_imports:
            raise ImportError(f"Import of {root!r} is not allowed.")
        return builtins.__import__(name, globals, locals, fromlist, level)

    def _print(self, *args: Any, **kwargs: Any) -> None:
        kwargs.pop("file", None)
        builtins.print(*args, file=self._buffer, **kwargs)

    def _get_logs(self) -> str:
        if self._buffer is None:
            return ""
        logs = self._buffer.getvalue()
        limit = self.max_print_outputs_length
        if len(logs) <= limit:
            return logs
        truncated = len(logs) - limit
        return f"{logs[:limit]}\n... [truncated {truncated} characters]"

    @contextlib.contextmanager
    def _time_limit(self) -> Iterator[None]:
        timeout = self.execution_timeout
        if not timeout or timeout <= 0:
            yield
            return
        if threading.current_thread() is not threading.main_thread() or not hasattr(
            signal, "SIGALRM"
        ):
            warnings.warn(
                "execution_timeout is only enforced on the main thread with SIGALRM; "
                "it was skipped for this execution.",
                UserWarning,
                stacklevel=3,
            )
            yield
            return

        def _raise_timeout(signum: int, frame: Any) -> None:
            raise TimeoutError(f"Execution exceeded execution_timeout={timeout}s")

        previous_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, _raise_timeout)
        signal.setitimer(signal.ITIMER_REAL, float(timeout))
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)


def _format_error(error: BaseException) -> str:
    if isinstance(error, TimeoutError):
        return str(error)
    return "".join(traceback.format_exception_only(type(error), error)).strip()
