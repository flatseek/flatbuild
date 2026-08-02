"""Flatbuild command-line interface.

Top-level subcommands:

- ``train`` — train a model from a YAML config.
- ``resume`` — resume training from a saved checkpoint.
- ``evaluate`` — run evaluation on a saved checkpoint.
- ``export`` — export checkpoint to SafeTensors / HuggingFace format.
- ``generate`` — generate text from a checkpoint (interactive).
- ``inspect`` — print checkpoint metadata.
- ``benchmark`` — numeric benchmark over a small batch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import click

from flatbuild import __version__
from flatbuild.checkpoint.manager import CheckpointManager, CheckpointState
from flatbuild.config import ExportFormat, FlatBuildConfig
from flatbuild.exporters.huggingface import HuggingFaceExporter
from flatbuild.exporters.safetensors import SafeTensorsExporter
from flatbuild.models import FlatbuildModel
from flatbuild.tokenizers.bpe import BPETokenizer
from flatbuild.trainer.trainer import FlatbuildTrainer, build_callbacks
from flatbuild.utils import (
    create_run_folder,
    get_logger,
    setup_logging,
    write_json,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_checkpoint_bundle(path: Path) -> tuple[FlatbuildModel, FlatBuildConfig, CheckpointState, BPETokenizer]:
    """Load a checkpoint and return ``(model, config, state, tokenizer)``."""
    bundle = CheckpointManager.load(path)
    cfg: FlatBuildConfig = bundle["config"]
    if cfg is None:
        raise ValueError(f"No config.yaml found in checkpoint dir: {path}")
    model = FlatbuildModel(cfg.model)
    model.load_state_dict_llama(bundle["model_state_dict"], strict=False)
    tok: BPETokenizer | None = None
    if bundle["tokenizer_dir"] is not None:
        tok = BPETokenizer.load(bundle["tokenizer_dir"])
    return model, cfg, bundle["state"], tok


def _inspect_fwg(path: Path) -> None:
    """Print metadata about a ``.fwg`` archive."""
    from flatbuild.exporters._fwg_reader import load_fwg_state_dict

    sd, meta = load_fwg_state_dict(path)
    n_params = sum(int(t.numel()) for t in sd.values())
    click.echo(json.dumps({
        "checkpoint_dir": str(path),
        "format": "fwg",
        "tile_size": meta["tile_size"],
        "tensor_count": meta["tensor_count"],
        "page_count": meta["page_count"],
        "storage_mode": meta["storage_mode"],
        "params": n_params,
        "sample_tensors": sorted(list(sd.keys()))[:8],
    }, indent=2))


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(version=__version__)
def cli() -> None:
    """Flatbuild - Train conversational language models from scratch."""
    pass


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("config_file", type=click.Path(exists=True))
@click.option(
    "--profile",
    is_flag=True,
    help="Measure per-phase timing and emit a final performance report.",
)
def train(config_file: str, profile: bool) -> None:
    """Train a model from a YAML configuration file."""
    setup_logging("train")
    cfg = FlatBuildConfig.from_yaml(config_file)
    logger.info(f"Loaded config from {config_file}")
    logger.info(f"Project: {cfg.name}")

    run_dir = create_run_folder(cfg.name, output_dir=cfg.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_yaml(run_dir / "config.yaml")

    # 1. Load the dataset.
    from flatbuild.datasets.loader import build_train_val_test, load_dataset

    all_samples = list(load_dataset(cfg.dataset))
    logger.info(f"Loaded {len(all_samples)} samples")
    if not all_samples:
        raise click.ClickException("Dataset is empty — check config.dataset.path")

    splits = build_train_val_test(
        all_samples,
        train_split=cfg.dataset.train_split,
        val_split=cfg.dataset.val_split,
        test_split=cfg.dataset.test_split,
        seed=cfg.dataset.seed,
    )
    logger.info(f"Splits: {splits.sizes()}")

    # 2. Train the tokenizer.
    from flatbuild.tokenizers.bpe import BPETokenizer
    from flatbuild.config import TokenizerSource

    tok_dir = run_dir / "tokenizer"
    if cfg.tokenizer.source == TokenizerSource.TRAIN:
        tok = BPETokenizer.train(
            (s.text if hasattr(s, "text") else _render_for_bpe(s, cfg) for s in splits.train),
            vocab_size=cfg.tokenizer.vocab_size,
            min_frequency=cfg.tokenizer.min_frequency,
            added_tokens=cfg.tokenizer.added_tokens,
            save_to=tok_dir,
        )
    else:
        if not cfg.tokenizer.path:
            raise click.ClickException("tokenizer.source=load requires tokenizer.path")
        tok = BPETokenizer.load(cfg.tokenizer.path)

    cfg.model.vocab_size = tok.vocab_size
    if cfg.model.vocab_size <= 0:
        raise click.ClickException("Tokenizer has zero vocab — corpus too small?")

    # 3. Build model + trainer + train.
    model = FlatbuildModel(cfg.model)
    callbacks = build_callbacks(cfg, run_dir)
    trainer = FlatbuildTrainer(
        config=cfg,
        model=model,
        tokenizer=tok,
        samples=splits.train,
        val_samples=splits.val,
        run_dir=run_dir,
        callbacks=callbacks,
        profile=profile,
    )
    artifacts = trainer.train()

    # 4. Persist run metadata.
    write_json(run_dir / "metrics.json", artifacts.metrics)
    write_json(
        run_dir / "train_metadata.json",
        {
            "status": "completed" if not artifacts.early_stopped else "early_stopped",
            "timestamp": __import__("datetime").datetime.utcnow().isoformat() + "Z",
            "config": cfg.to_dict(),
            "splits": splits.sizes(),
        },
    )
    click.echo(f"[train] Run completed. Artifacts at {run_dir}")


# ---------------------------------------------------------------------------
# resume
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("checkpoint", type=click.Path(exists=True))
@click.option(
    "--config",
    "config_file",
    type=click.Path(exists=True),
    default=None,
    help="Config to override the saved one.",
)
def resume(checkpoint: str, config_file: Optional[str]) -> None:
    """Resume training from a saved checkpoint."""
    setup_logging("resume")
    bundle = CheckpointManager.load(checkpoint)
    cfg: FlatBuildConfig = bundle["config"] or (
        FlatBuildConfig.from_yaml(config_file) if config_file else None
    )
    if cfg is None:
        raise click.ClickException("No config.yaml in checkpoint; pass --config")

    tok = BPETokenizer.load(bundle["tokenizer_dir"]) if bundle["tokenizer_dir"] else None
    if tok is None:
        raise click.ClickException("Checkpoint has no tokenizer — cannot resume.")

    run_dir = Path(checkpoint).parent
    model = FlatbuildModel(cfg.model)
    model.load_state_dict_llama(bundle["model_state_dict"], strict=False)

    # For resume we use the same splits as the original run; the user
    # can re-specify via --config. Falls back to a fresh load.
    from flatbuild.datasets.loader import build_train_val_test, load_dataset

    all_samples = list(load_dataset(cfg.dataset))
    splits = build_train_val_test(
        all_samples,
        train_split=cfg.dataset.train_split,
        val_split=cfg.dataset.val_split,
        test_split=cfg.dataset.test_split,
        seed=cfg.dataset.seed,
    )
    callbacks = build_callbacks(cfg, run_dir)
    trainer = FlatbuildTrainer(
        config=cfg,
        model=model,
        tokenizer=tok,
        samples=splits.train,
        val_samples=splits.val,
        run_dir=run_dir,
        callbacks=callbacks,
    )
    trainer.global_step = bundle["state"].global_step
    trainer.epoch_index = bundle["state"].epoch_index

    if bundle["optimizer_state"] is not None:
        opt_state = bundle["optimizer_state"].get("optimizer")
        if opt_state is not None:
            trainer.optimizer.load_state_dict(opt_state)

    artifacts = trainer.train()
    write_json(run_dir / "resume_metadata.json", artifacts.metrics)
    click.echo(f"[resume] Run completed. Artifacts at {run_dir}")


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("checkpoint", type=click.Path(exists=True))
@click.option("--config", "config_file", type=click.Path(exists=True), default=None)
def evaluate(checkpoint: str, config_file: Optional[str]) -> None:
    """Evaluate a checkpoint against the configured validation split."""
    setup_logging("evaluate")
    model, cfg, _state, tok = _load_checkpoint_bundle(Path(checkpoint))
    if tok is None:
        raise click.ClickException("Checkpoint has no tokenizer")

    from flatbuild.datasets.loader import build_train_val_test, load_dataset

    all_samples = list(load_dataset(cfg.dataset))
    splits = build_train_val_test(
        all_samples,
        train_split=cfg.dataset.train_split,
        val_split=cfg.dataset.val_split,
        test_split=cfg.dataset.test_split,
        seed=cfg.dataset.seed,
    )

    trainer = FlatbuildTrainer(
        config=cfg,
        model=model,
        tokenizer=tok,
        samples=[],
        val_samples=splits.val,
        run_dir=Path(checkpoint).parent,
    )
    metrics = trainer.evaluate()
    click.echo(json.dumps(metrics, indent=2))


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("checkpoint", type=click.Path(exists=True))
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["safetensors", "huggingface", "gguf", "fwg"]),
    default="safetensors",
    show_default=True,
)
@click.option("--output", "output_dir", type=click.Path(), default=None)
def export(checkpoint: str, fmt: str, output_dir: Optional[str]) -> None:
    """Export a checkpoint to SafeTensors, HuggingFace, GGUF, or Flatweight (.fwg) format.

    The ``gguf`` format requires ``pip install -e '.[gguf]'``; the ``fwg``
    format requires ``pip install -e '.[fwg]'``.
    """
    setup_logging("export")
    model, cfg, _state, tok = _load_checkpoint_bundle(Path(checkpoint))
    ckpt = Path(checkpoint)

    target = Path(output_dir) if output_dir else ckpt.parent / f"export_{fmt}"
    target.mkdir(parents=True, exist_ok=True)

    if fmt == "safetensors":
        exporter = SafeTensorsExporter(copy_tokenizer=True)
    elif fmt == "huggingface":
        exporter = HuggingFaceExporter(copy_tokenizer=True)
    elif fmt == "gguf":
        from flatbuild.exporters.gguf import GGUFExporter

        exporter = GGUFExporter(copy_tokenizer=True)
    elif fmt == "fwg":
        from flatbuild.exporters.fwg import FWGExporter

        exporter = FWGExporter(copy_tokenizer=True)
    else:  # pragma: no cover - guarded by click.Choice
        raise click.ClickException(f"Unknown export format: {fmt}")

    cfg_export = cfg.export
    # If the saved config didn't carry a tokenizer_path, point at the one
    # that was saved inside the checkpoint.
    if tok is not None:
        setattr(cfg_export, "tokenizer_path", str(ckpt / "tokenizer"))
    out = exporter.export(model, target, config=cfg_export)
    click.echo(f"[export] Wrote {out}")


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("checkpoint", type=click.Path(exists=True))
@click.option("--prompt", default="Hello", show_default=True)
@click.option("--max-new", "max_new_tokens", default=64, show_default=True)
@click.option("--temperature", default=0.8, show_default=True)
@click.option("--top-k", default=50, show_default=True)
@click.option("--top-p", default=0.95, show_default=True)
@click.option("--no-sample", "no_sample", is_flag=True, help="Use greedy decoding.")
@click.option(
    "--chat",
    "use_chat_template",
    is_flag=True,
    help="Render the prompt through the chat template.",
)
def generate(
    checkpoint: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    no_sample: bool,
    use_chat_template: bool,
) -> None:
    """Generate text from a checkpoint."""
    import torch

    setup_logging("generate")
    model, cfg, _state, tok = _load_checkpoint_bundle(Path(checkpoint))
    if tok is None:
        raise click.ClickException("Checkpoint has no tokenizer")

    model.eval()
    device = next(model.parameters()).device

    if use_chat_template:
        from flatbuild.tokenizers.template import build_chat_template

        tmpl = build_chat_template(cfg.chat_template)
        # Always include the system prompt when one is configured —
        # this matches what tokenize_sample feeds during training.
        render_messages: list[tuple[str, str]] = []
        if cfg.chat_template.system:
            render_messages.append(("system", cfg.chat_template.system))
        render_messages.append(("user", prompt))
        rendered = tmpl.render(render_messages, add_generation_prompt=True)
    else:
        rendered = prompt

    ids = tok.encode(rendered) or [tok.eos_token_id]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=not no_sample,
        eos_token_id=tok.eos_token_id,
    )
    text = tok.decode([int(i) for i in out[0].tolist()])
    click.echo(text)


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("checkpoint", type=click.Path(exists=True))
@click.option(
    "--from-fwg",
    "from_fwg",
    is_flag=True,
    help="Read directly from a .fwg Flatweight archive (no decoding needed).",
)
def inspect(checkpoint: str, from_fwg: bool) -> None:
    """Print metadata and parameter counts of a checkpoint."""
    setup_logging("inspect")

    if from_fwg or checkpoint.endswith(".fwg"):
        _inspect_fwg(Path(checkpoint))
        return

    bundle = CheckpointManager.load(checkpoint)
    cfg = bundle["config"]
    state = bundle["state"]

    n_params = 0
    for tensor in bundle["model_state_dict"].values():
        n_params += int(tensor.numel())

    click.echo(json.dumps(
        {
            "checkpoint_dir": str(checkpoint),
            "project": cfg.name if cfg else None,
            "global_step": state.global_step,
            "epoch": state.epoch_index,
            "last_loss": state.last_loss,
            "params": n_params,
            "vocab_size": cfg.model.vocab_size if cfg else None,
            "n_layers": cfg.model.n_layers if cfg else None,
            "hidden_dim": cfg.model.hidden_dim if cfg else None,
            "n_heads": cfg.model.n_heads if cfg else None,
            "n_kv_heads": cfg.model.n_kv_heads if cfg else None,
            "activation": cfg.model.activation.value if cfg else None,
            "norm": cfg.model.norm.value if cfg else None,
        },
        indent=2,
    ))


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("checkpoint", type=click.Path(exists=True))
@click.option("--iters", default=20, show_default=True)
def benchmark(checkpoint: str, iters: int) -> None:
    """Benchmark forward + generation latency on a tiny batch."""
    import time

    import torch

    setup_logging("benchmark")
    model, cfg, _state, tok = _load_checkpoint_bundle(Path(checkpoint))
    model.eval()
    device = next(model.parameters()).device
    seq_len = cfg.model.context_length

    prompt = [tok.bos_token_id or tok.eos_token_id] * seq_len
    input_ids = torch.tensor([prompt], dtype=torch.long, device=device)

    # Warmup
    with torch.no_grad():
        model(input_ids)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        with torch.no_grad():
            model(input_ids)
        times.append(time.perf_counter() - t0)
    avg_ms = sum(times) / len(times) * 1000
    click.echo(json.dumps({
        "iters": iters,
        "seq_len": seq_len,
        "avg_forward_ms": round(avg_ms, 2),
    }, indent=2))


# ---------------------------------------------------------------------------
# generate-data
# ---------------------------------------------------------------------------


@cli.command("generate-data")
@click.option(
    "--n",
    default=100_000,
    show_default=True,
    type=int,
    help="Number of conversation samples to generate.",
)
@click.option(
    "--out",
    default="data/demo_large/dataset.jsonl",
    show_default=True,
    type=click.Path(),
    help="Output JSONL path.",
)
@click.option("--seed", default=7, show_default=True, type=int)
def generate_data_cmd(n: int, out: str, seed: int) -> None:
    """Generate the demo conversational dataset (default 100,000 samples)."""
    from flatbuild.generate_data import run as run_generator

    setup_logging("generate-data")
    run_generator(n=n, out=Path(out), seed=seed)
    click.echo(f"Wrote {n:,} samples to {out}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Click entry point — also referenced from pyproject as ``flatbuild``."""
    cli.main(args=argv, standalone_mode=False)
    return 0


def _render_for_bpe(sample, cfg):
    """Best-effort textual rendering of a sample for BPE training."""
    from flatbuild.datasets.base import PretrainingSample

    if isinstance(sample, PretrainingSample):
        return sample.text
    from flatbuild.tokenizers.template import build_chat_template

    return build_chat_template(cfg.chat_template).render_sample(sample)


if __name__ == "__main__":
    raise SystemExit(main())
