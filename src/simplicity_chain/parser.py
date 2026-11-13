"""Extraction of executable code from model output."""

from __future__ import annotations

import ast
import re

from .errors import AgentParsingError

DEFAULT_CODE_BLOCK_TAGS: tuple[str, str] = ("```python", "```")
_DEFAULT_PATTERN = r"```(?:python|py)?[ \t]*\r?\n(.*?)```"


def parse_code_blobs(
    text: str,
    code_block_tags: tuple[str, str] = DEFAULT_CODE_BLOCK_TAGS,
) -> str:
    """Extract and join all fenced code blocks from a model response.

    Raises:
        AgentParsingError: if the response contains no (or only empty) code blocks.
    """
    if not text or not text.strip():
        raise AgentParsingError(
            "The model returned an empty response. Reply with exactly one Python code block."
        )
    if code_block_tags == DEFAULT_CODE_BLOCK_TAGS:
        pattern = _DEFAULT_PATTERN
        flags = re.DOTALL | re.IGNORECASE
    else:
        opening, closing = code_block_tags
        pattern = re.escape(opening) + r"(.*?)" + re.escape(closing)
        flags = re.DOTALL

    matches = re.findall(pattern, text, flags=flags)
    if not matches:
        raise AgentParsingError(
            "No code block found in the model response. Reply with exactly one Python "
            f"code block delimited by {code_block_tags[0]!r} and {code_block_tags[1]!r}."
        )
    code = "\n".join(match.strip("\n") for match in matches).strip()
    if not code:
        raise AgentParsingError("The code block in the model response was empty.")
    return code


def fix_final_answer_code(code: str) -> str:
    """Rewrite a trailing ``final_answer = expr`` assignment into ``final_answer(expr)``."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    if not tree.body:
        return code
    last_statement = tree.body[-1]
    if not (
        isinstance(last_statement, ast.Assign)
        and len(last_statement.targets) == 1
        and isinstance(last_statement.targets[0], ast.Name)
        and last_statement.targets[0].id == "final_answer"
    ):
        return code
    call = ast.Call(
        func=ast.Name(id="final_answer", ctx=ast.Load()),
        args=[last_statement.value],
        keywords=[],
    )
    tree.body[-1] = ast.Expr(value=call)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)
