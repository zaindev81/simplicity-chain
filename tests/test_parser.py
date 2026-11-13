from __future__ import annotations

import pytest

from simplicity_chain.errors import AgentParsingError
from simplicity_chain.parser import fix_final_answer_code, parse_code_blobs


def test_parses_single_block():
    assert parse_code_blobs("```python\nx = 1\n```") == "x = 1"
    assert parse_code_blobs("```py\nx = 1\n```") == "x = 1"


def test_ignores_prose_around_block():
    text = "I will compute this.\n```python\nprint(1)\n```\nDone."
    assert parse_code_blobs(text) == "print(1)"


def test_joins_multiple_blocks():
    text = "```python\na = 1\n```\nand\n```python\nb = 2\n```"
    assert parse_code_blobs(text) == "a = 1\nb = 2"


def test_custom_tags():
    text = "<code>x = 1</code>"
    assert parse_code_blobs(text, ("<code>", "</code>")) == "x = 1"


def test_missing_or_empty_block_raises():
    with pytest.raises(AgentParsingError):
        parse_code_blobs("I think the answer is 42.")
    with pytest.raises(AgentParsingError):
        parse_code_blobs("")
    with pytest.raises(AgentParsingError):
        parse_code_blobs("```python\n\n```")


def test_fix_final_answer_assignment():
    fixed = fix_final_answer_code("total = 2 + 3\nfinal_answer = total")
    assert "final_answer(total)" in fixed


def test_fix_leaves_correct_code_alone():
    code = "final_answer(3)"
    assert fix_final_answer_code(code) == code


def test_fix_leaves_invalid_or_non_final_code_alone():
    assert fix_final_answer_code("def broken(") == "def broken("
    code = "result = 1\nprint(result)"
    assert fix_final_answer_code(code) == code
