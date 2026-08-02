"""Allow ``python -m flatbuild`` to invoke the CLI."""

from flatbuild.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
