from __future__ import annotations

from simplicity_chain.cli import build_parser, main


def test_cli_reports_missing_api_key(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    assert main(["what is 2 + 2?"]) == 1
    assert "No API key" in capsys.readouterr().err


def test_cli_parser_defaults():
    args = build_parser().parse_args(["some task"])
    assert args.task == "some task"
    assert args.max_steps == 8
    assert args.authorized_imports == ""
    assert args.quiet is False
