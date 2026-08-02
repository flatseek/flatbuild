"""CLI wrapper around ``scripts/generate_demo_large.py``.

Importable entry points:

- :func:`main` — Click handler factory used by the CLI.
- :func:`run` — Module-level Python helper.
"""

from __future__ import annotations

import sys
from pathlib import Path

from flatbuild.utils import get_logger

logger = get_logger(__name__)


def run(n: int, out: Path | str, seed: int = 7) -> Path:
    """Generate ``n`` samples to ``out``.

    Args:
        n: Number of samples.
        out: Destination JSONL file path.
        seed: RNG seed for reproducibility.

    Returns:
        The output file path.
    """
    # Import lazily so the package starts up even if the generator is in flux.
    # ``generate_data.py`` lives at ``src/flatbuild/generate_data.py``,
    # so ``parent.parent.parent`` is the repo root.
    repo_root = Path(__file__).resolve().parent.parent.parent
    script = repo_root / "scripts" / "generate_demo_large.py"
    if not script.exists():
        raise FileNotFoundError(f"Generator script not found: {script}")

    import importlib.util

    spec = importlib.util.spec_from_file_location("flatbuild_generator", script)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    sys.argv = [str(script), "--n", str(n), "--out", str(out), "--seed", str(seed)]
    rc = mod.main()
    if rc != 0:
        raise RuntimeError(f"Generator exited with code {rc}")
    return Path(out)


def main() -> None:
    """Click handler — invoked by ``flatbuild generate-data``."""
    import click

    @click.command("generate-data")
    @click.option(
        "--n",
        default=100_000,
        show_default=True,
        type=int,
        help="Number of conversation samples.",
    )
    @click.option(
        "--out",
        default="data/demo_large/dataset.jsonl",
        show_default=True,
        type=click.Path(),
        help="Output JSONL path.",
    )
    @click.option("--seed", default=7, show_default=True, type=int)
    def _cmd(n: int, out: str, seed: int) -> None:
        out_path = Path(out)
        run(n=n, out=out_path, seed=seed)
        click.echo(f"Wrote {n:,} samples to {out_path}")

    _cmd()


if __name__ == "__main__":
    main()
