"""Allow ``python -m simplicity_chain``."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
