"""Training loop for Flatbuild.

This module is the orchestrator. The heavy lifting is delegated to:

- :class:`flatbuild.trainer.progress.ProgressReporter` — per-epoch tqdm.
- :class:`flatbuild.trainer.profiler.PerformanceProfiler` — phase timers.
- :class:`flatbuild.trainer.validation.ValidationRunner` — eval pass.
- :mod:`flatbuild.trainer.datamodule` — tokenization caching + DataLoader.

Callbacks remain the public API for users wanting additional hooks.
Major user-visible events (train started/finished, validation, checkpoint
saved, early stopping) are routed to ``logging``; everything else is
shown in tqdm's postfix to avoid console spam.
"""

from __future__ import annotations

import json
import math
import multiprocessing
import os
import sys
import time
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from flatbuild.callbacks.base import (
    Callback,
    CallbackContext,
)
from flatbuild.checkpoint.manager import CheckpointManager, CheckpointState
from flatbuild.config import FlatBuildConfig, Precision
from flatbuild.datasets.base import Sample
from flatbuild.models import FlatbuildModel
from flatbuild.optimizers.factory import build_optimizer
from flatbuild.schedulers.factory import build_scheduler
from flatbuild.tokenizers.bpe import Tokenizer
from flatbuild.tokenizers.template import ChatTemplate, build_chat_template
from flatbuild.trainer.datamodule import DataLoaderConfig, build_dataloader
from flatbuild.trainer.profiler import PerformanceProfiler
from flatbuild.trainer.progress import ProgressReporter
from flatbuild.trainer.validation import make_validation_runner
from flatbuild.utils import get_logger, set_seed, write_json

logger = get_logger(__name__)


@dataclass
class TrainArtifacts:
    """Bundle returned by :meth:`FlatbuildTrainer.train`."""

    run_dir: Path
    checkpoint_state: CheckpointState
    metrics: dict[str, Any]
    early_stopped: bool = False



# ---------------------------------------------------------------------------
# Binary tokenized-cache I/O
# ---------------------------------------------------------------------------


def _save_tokenized_cache(
    data: list[tuple[list[int], list[int]]],
    bin_path: Path,
    meta_path: Path,
    max_length: int,
) -> None:
    """Write tokenized data to a binary cache file.

    Format:
        uint32 n_samples
        uint32[n_samples] lengths
        uint64[n_samples+1] ids_offsets  (byte offsets into ids_flat)
        uint64[n_samples+1] labels_offsets  (byte offsets into labels_flat)
        int32[total_ids] ids_flat
        int32[total_labels] labels_flat
    """
    import json as _json

    import numpy as np

    n = len(data)
    lens = np.array([len(ids) for ids, _ in data], dtype=np.uint32)

    ids_flat_list: list[int] = []
    labels_flat_list: list[int] = []
    ids_offsets = np.zeros(n + 1, dtype=np.uint64)
    labels_offsets = np.zeros(n + 1, dtype=np.uint64)

    ids_off = 0
    labels_off = 0
    for i, (ids, labels) in enumerate(data):
        ids_offsets[i] = ids_off
        labels_offsets[i] = labels_off
        ids_flat_list.extend(ids)
        labels_flat_list.extend(labels)
        ids_off += len(ids)
        labels_off += len(labels)
    ids_offsets[n] = ids_off
    labels_offsets[n] = labels_off

    ids_flat = np.array(ids_flat_list, dtype=np.int32)
    labels_flat = np.array(labels_flat_list, dtype=np.int32)

    with open(bin_path, "wb") as f:
        np.array(n, dtype=np.uint32).tofile(f)
        lens.tofile(f)
        ids_offsets.tofile(f)
        labels_offsets.tofile(f)
        ids_flat.tofile(f)
        labels_flat.tofile(f)

    with open(meta_path, "w", encoding="utf-8") as f:
        _json.dump({"max_length": max_length, "n_samples": n, "version": 1}, f)


def _load_tokenized_cache(
    bin_path: Path,
    meta_path: Path,
    max_length: int,
) -> list[tuple[list[int], list[int]]] | None:
    """Load tokenized data from binary cache. Returns None if cache is missing or stale."""
    import json as _json

    import numpy as np

    if not bin_path.exists() or not meta_path.exists():
        return None
    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = _json.load(f)
        if meta.get("max_length") != max_length or meta.get("version") != 1:
            return None
    except Exception:
        return None

    try:
        n = meta["n_samples"]
        with open(bin_path, "rb") as f:
            _ = np.fromfile(f, dtype=np.uint32, count=1)[0]
            lens = np.fromfile(f, dtype=np.uint32, count=n)
            ids_offsets = np.fromfile(f, dtype=np.uint64, count=n + 1)
            labels_offsets = np.fromfile(f, dtype=np.uint64, count=n + 1)
            ids_flat = np.fromfile(f, dtype=np.int32)
            labels_flat = np.fromfile(f, dtype=np.int32)

        results: list[tuple[list[int], list[int]]] = []
        for i in range(n):
            ids_start = int(ids_offsets[i])
            ids_end = int(ids_offsets[i + 1])
            labels_start = int(labels_offsets[i])
            labels_end = int(labels_offsets[i + 1])
            ids = ids_flat[ids_start:ids_end].tolist()
            labels = labels_flat[labels_start:labels_end].tolist()
            results.append((ids, labels))
        return results
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Parallel tokenization worker (module-level for ProcessPoolExecutor pickle)
# ---------------------------------------------------------------------------


def _tokenize_chunk(
    chunk: list[dict],
    tokenizer_path: str,
    template: "ChatTemplate",
    max_length: int,
) -> list[tuple[list[int], list[int]]]:
    """Tokenize a chunk of samples in a worker process.

    Each worker loads its own tokenizer copy from disk, so the
    ``_inner`` Rust object never needs to be pickled.
    """
    from flatbuild.tokenizers.bpe import BPETokenizer
    from flatbuild.trainer.tokenize import tokenize_sample
    from flatbuild.datasets.base import normalize_sample

    tokenizer = BPETokenizer.load(tokenizer_path)
    results: list[tuple[list[int], list[int]]] = []
    for raw in chunk:
        try:
            sample = normalize_sample(raw)
            ids, labels = tokenize_sample(
                sample, tokenizer, template, max_length=max_length
            )
            if ids:
                results.append((ids, labels))
        except Exception:
            continue
    return results


class FlatbuildTrainer:
    """Orchestrates training: tokenizer + chat template + model + optim + sched.

    Owns the trainer loop, callbacks, optimizer, scheduler, mixed-
    precision context, checkpoint manager, progress reporter, and
    performance profiler. Delegates data, validation, and progress UI
    out to dedicated modules.
    """

    def __init__(
        self,
        config: FlatBuildConfig,
        model: FlatbuildModel,
        tokenizer: Tokenizer,
        samples: list[Sample],
        val_samples: list[Sample] | None = None,
        *,
        run_dir: Path,
        callbacks: Iterable[Callback] | None = None,
        device: str | torch.device | None = None,
        profile: bool = False,
    ) -> None:
        """Initialize the trainer.

        Args:
            config: Top-level flatbuild config.
            model: Initialized :class:`FlatbuildModel`.
            tokenizer: Trained tokenizer.
            samples: Training samples.
            val_samples: Optional evaluation samples.
            run_dir: Output directory for artifacts.
            callbacks: Iterable of callback instances.
            device: Force a device; default → cuda if available else cpu.
            profile: When ``True``, enable :class:`PerformanceProfiler`
                so the trainer prints a breakdown at the end.
        """
        self.config = config
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = tokenizer
        self.samples = list(samples)
        self.val_samples = list(val_samples) if val_samples is not None else []
        self.template = build_chat_template(config.chat_template)

        # Resolve device.
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.model = model.to(self.device)

        # State owned by the trainer.
        self.callbacks: list[Callback] = list(callbacks or [])
        self.callback_ctx_state: dict[str, Any] = {}
        self.profiler = PerformanceProfiler(enabled=profile)

        # Optimizer + scheduler.
        self.optimizer = build_optimizer(config.optimizer, self.model.parameters())
        total_optimizer_steps = self._estimate_total_steps()
        self.scheduler = build_scheduler(
            config.scheduler, self.optimizer, total_steps=total_optimizer_steps
        )

        # Mixed precision: autocast + GradScaler only when requested.
        self.precision = config.trainer.precision
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.precision == Precision.FP16)
        self._amp_enabled = self.precision != Precision.FP32
        self._amp_dtype = (
            {Precision.FP16: torch.float16, Precision.BF16: torch.bfloat16}[self.precision]
            if self._amp_enabled
            else torch.float32
        )

        # Tokenize + cache samples once (small CPU cost, faster epochs).
        self.train_tokenized = self._tokenize_all(self.samples)
        self.val_tokenized = self._tokenize_all(self.val_samples) if self.val_samples else []

        # DataLoader config.
        self.dl_config = DataLoaderConfig(
            num_workers=int(config.trainer.num_workers),
            pin_memory=bool(config.trainer.pin_memory and self.device.type == "cuda"),
            persistent_workers=bool(config.trainer.persistent_workers and config.trainer.num_workers > 0),
            prefetch_factor=int(config.trainer.prefetch_factor),
            drop_last=bool(config.trainer.drop_last),
        )

        # Checkpoint + state.
        self.checkpoint_manager = CheckpointManager(
            self.run_dir, max_to_keep=config.checkpoint.keep_last
        )
        # Async checkpoint writes if requested.
        self._ckpt_executor: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="ckpt")
            if bool(config.checkpoint.async_write)
            else None
        )

        self.global_step = 0
        self.epoch_index = 0
        self.last_loss: float | None = None
        self._stop_training = False
        self.state = CheckpointState()

        # Trackers for the final summary.
        self._max_cpu_mem_bytes = 0
        self._max_gpu_mem_bytes = 0
        self._total_samples = 0
        self._total_tokens = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> TrainArtifacts:
        """Run the training loop.

        Returns:
            :class:`TrainArtifacts` with paths and metrics.
        """
        # Snapshot peak memory before training so the deltas are meaningful.
        self._sample_peak_memory()

        set_seed(self.config.trainer.seed)
        self._fire("on_train_begin")
        progress = ProgressReporter()
        history: list[dict[str, Any]] = []
        wall_start = time.perf_counter()

        try:
            if self.epoch_index > 0 or self.global_step > 0:
                start_epoch = self.epoch_index + 1  # resume from next epoch
            else:
                start_epoch = 0
            for epoch in range(start_epoch, max(1, int(self.config.trainer.epochs))):
                if self._stop_training:
                    break
                self.epoch_index = epoch
                self._train_one_epoch(epoch, progress, history)
                # End-of-epoch full validation (always).
                if self.val_tokenized:
                    try:
                        metrics = self._run_validation()
                    except Exception as exc:
                        logger.warning(f"Validation failed: {exc}. Skipping.")
                        metrics = {"loss": 0.0, "perplexity": 1.0, "accuracy": 0.0}
                    self.callback_ctx_state.update(
                        val_loss=metrics["loss"],
                        val_perplexity=metrics["perplexity"],
                        val_accuracy=metrics["accuracy"],
                    )
                    self._fire("on_eval_end")
                    history.append(
                        {
                            "epoch": epoch,
                            "global_step": self.global_step,
                            **metrics,
                        }
                    )
                if self.config.checkpoint.save_final:
                    self._save_final()
                if self._stop_training:
                    break
            if self.config.checkpoint.save_final:
                self._save_final()
        finally:
            self._fire("on_train_end")
            self._write_history(history)
            self._shutdown_executor()
            elapsed = time.perf_counter() - wall_start
            progress.close()

            # Print performance breakdown if profiling.
            if self.profiler.enabled:
                self.profiler.print_summary(
                    step_count=self.global_step,
                    elapsed=elapsed,
                )
                self._print_throughput_summary(elapsed)
                self._print_peak_memory()

        metrics = self._final_metrics(history, elapsed)
        write_json(self.run_dir / "metrics.json", metrics)
        return TrainArtifacts(
            run_dir=self.run_dir,
            checkpoint_state=self.state,
            metrics=metrics,
            early_stopped=self.callback_ctx_state.get("early_stopped", False),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _train_one_epoch(
        self,
        epoch: int,
        progress: ProgressReporter,
        history: list[dict[str, Any]],
    ) -> None:
        """Run one epoch of training, mutating ``history`` in-place."""
        cfg = self.config
        batch_size = max(1, int(cfg.trainer.batch_size))
        grad_accum = max(1, int(cfg.trainer.gradient_accumulation))
        max_norm = float(cfg.trainer.max_grad_norm)

        # Build a fresh DataLoader for the epoch (shuffled) — avoids
        # carrying state between epochs and keeps worker lifecycle
        # simple across ``persistent_workers``.
        loader = build_dataloader(
            self.train_tokenized,
            batch_size=batch_size,
            shuffle=True,
            pad_token_id=int(self.tokenizer.pad_token_id or 0),
            config=self.dl_config,
        )

        # Cap epochs at ``max_steps``.
        steps_per_epoch = max(1, len(loader) // grad_accum)
        max_steps = cfg.trainer.max_steps
        stop_at_step = max_steps if max_steps is not None else None

        progress.start_epoch(
            total=min(steps_per_epoch, stop_at_step) if stop_at_step else steps_per_epoch,
            epoch=epoch + 1,
            total_epochs=cfg.trainer.epochs,
        )

        # Pre-allocate accumulators outside the step loop.
        accum_loss = 0.0
        accum_count = 0
        epoch_metrics: list[dict[str, Any]] = []
        self.model.train()

        try:
            for micro_idx, (input_ids, labels) in enumerate(loader):
                if stop_at_step is not None and self.global_step >= stop_at_step:
                    self._stop_training = True
                    break
                if self._stop_training:
                    break

                # -------- Data: copy micro-batch onto device --------
                with self.profiler.measure("data"):
                    input_ids = input_ids.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                    bsz = input_ids.size(0)
                    tok_count = int((labels != -100).sum().item())

                # -------- Forward + loss + autocast --------
                step_loss: float = 0.0
                ctx_amp = torch.amp.autocast(
                    device_type=self.device.type,
                    enabled=self._amp_enabled,
                    dtype=self._amp_dtype,
                )
                with self.profiler.measure("forward"), ctx_amp:
                    out = self.model(input_ids, labels=labels)
                    loss = out.loss / grad_accum
                    step_loss = float(loss.detach().item())
                with self.profiler.measure("backward"):
                    if self.precision == Precision.FP16:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                accum_loss += step_loss * grad_accum
                accum_count += 1
                self._total_samples += bsz
                self._total_tokens += tok_count

                # -------- Optimizer / scheduler step (every grad_accum) --------
                if accum_count % grad_accum == 0:
                    with self.profiler.measure("optim"):
                        if self.precision == Precision.FP16:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
                            self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad(set_to_none=True)

                    self.global_step += 1
                    self.profiler.tick(samples=bsz, tokens=tok_count)
                    self._sample_peak_memory()

                    avg_loss = accum_loss / max(1, accum_count)
                    accum_loss = 0.0
                    accum_count = 0
                    self.last_loss = avg_loss
                    lr_now = float(self.optimizer.param_groups[0]["lr"])
                    self.callback_ctx_state["loss"] = avg_loss
                    self.callback_ctx_state["lr"] = lr_now
                    self._fire("on_step_end")

                    # -------- Periodic inline evaluation --------
                    if (
                        cfg.validation.every_steps is not None
                        and self.global_step % cfg.validation.every_steps == 0
                        and self.val_tokenized
                    ):
                        self._fire("on_eval_begin")
                        metrics = self._run_validation(
                            max_batches=cfg.validation.max_batches
                        )
                        self.callback_ctx_state.update(
                            val_loss=metrics["loss"],
                            val_perplexity=metrics["perplexity"],
                            val_accuracy=metrics["accuracy"],
                        )
                        self._fire("on_eval_end")
                        self.model.train()
                        history.append(
                            {
                                "epoch": epoch,
                                "global_step": self.global_step,
                                **metrics,
                            }
                        )

                    # -------- Periodic checkpoint --------
                    if self.global_step % max(1, cfg.checkpoint.every_n_steps) == 0:
                        self._save_step()
                        self._fire("on_checkpoint_saved")

                    progress.update(
                        step=self.global_step,
                        loss=avg_loss,
                        lr=lr_now,
                        samples=bsz,
                        tokens=tok_count,
                    )
        finally:
            progress.close()

    def _run_validation(self, *, max_batches: int | None = None) -> dict[str, float]:
        """Run a validation pass via :class:`ValidationRunner`.

        Args:
            max_batches: Optional cap; ``None`` evaluates the full set.

        Returns:
            ``{loss, perplexity, accuracy}``.
        """
        cfg = self.config
        runner = make_validation_runner(
            self.model,
            self.tokenizer,
            self.val_tokenized,
            batch_size=max(1, int(cfg.trainer.batch_size)),
            max_batches=max_batches,
            pad_token_id=int(self.tokenizer.pad_token_id or 0),
            pin_memory=self.dl_config.pin_memory,
            device=self.device,
        )
        with self.profiler.measure("valid"):
            metrics = runner.run()
        if max_batches is not None and max_batches < max(1, len(self.val_tokenized) // max(1, int(cfg.trainer.batch_size))):
            logger.info(
                f"Validation (capped @ {max_batches} batches): "
                f"loss={metrics['loss']:.4f} ppl={metrics['perplexity']:.2f} acc={metrics['accuracy']:.3f}"
            )
        else:
            logger.info(
                f"Validation: loss={metrics['loss']:.4f} ppl={metrics['perplexity']:.2f} "
                f"acc={metrics['accuracy']:.3f}"
            )
        return metrics

    # ----- Checkpoint helpers -----

    def _save_final(self) -> None:
        """Snapshot the final weights to ``checkpoints/final/`` (always retained)."""
        self.state.global_step = self.global_step
        self.state.epoch_index = self.epoch_index
        self.state.last_loss = self.last_loss
        path = self.checkpoint_manager.save_final(
            model=self.model,
            optimizer=self.optimizer,
            tokenizer_dir=self.run_dir / "tokenizer",
            config=self.config,
            state=self.state,
        )
        logger.info(f"Saved final checkpoint to {path}")

    def _save_step(self) -> None:
        """Save a numbered step checkpoint (optionally async)."""
        self.state.global_step = self.global_step
        self.state.epoch_index = self.epoch_index
        self.state.last_loss = self.last_loss
        if self._ckpt_executor is not None:
            self._ckpt_executor.submit(self._do_save_step_sync)
        else:
            with self.profiler.measure("ckpt"):
                self._do_save_step_sync()

    def _do_save_step_sync(self) -> None:
        """Synchronous checkpoint write body — measure & move head only on caller."""
        with self.profiler.measure("ckpt"):
            self.checkpoint_manager.save_step(
                step=self.global_step,
                model=self.model,
                optimizer=self.optimizer,
                tokenizer_dir=self.run_dir / "tokenizer",
                config=self.config,
                state=self.state,
            )

    def _shutdown_executor(self) -> None:
        """Wait for any in-flight async checkpoint writes before returning."""
        if self._ckpt_executor is not None:
            self._ckpt_executor.shutdown(wait=True)
            self._ckpt_executor = None

    # ----- Eval API (used by CLI evaluate command) -----

    def evaluate(self) -> dict[str, float]:
        """Run a full evaluation pass.

        Returns:
            ``{loss, perplexity, accuracy}``.
        """
        if not self.val_tokenized:
            return {"loss": 0.0, "perplexity": 1.0, "accuracy": 0.0}
        runner = make_validation_runner(
            self.model,
            self.tokenizer,
            self.val_tokenized,
            batch_size=max(1, int(self.config.trainer.batch_size)),
            max_batches=None,  # full eval
            pad_token_id=int(self.tokenizer.pad_token_id or 0),
            device=self.device,
        )
        return runner.run()

    # ----- Step estimation -----

    def _estimate_total_steps(self) -> int:
        """Estimate optimizer step count (after gradient accumulation)."""
        batch_size = max(1, int(self.config.trainer.batch_size))
        grad_accum = max(1, int(self.config.trainer.gradient_accumulation))
        epochs = max(1, int(self.config.trainer.epochs))
        steps_per_epoch = max(1, math.ceil(len(self.samples) / (batch_size * grad_accum)))
        if self.config.trainer.max_steps is not None:
            return int(self.config.trainer.max_steps)
        return steps_per_epoch * epochs

    # ----- Tokenization (one-shot, cached) -----

    def _tokenize_all(self, samples: list[Sample]) -> list[tuple[list[int], list[int]]]:
        """Render every sample to ``(input_ids, labels)`` once using multiprocessing.

        Results are cached to a binary file for near-instant reload on
        subsequent runs. The cache is invalidated when ``max_length`` changes.
        """
        from flatbuild.trainer.tokenize import tokenize_sample

        n_samples = len(samples)
        if n_samples == 0:
            return []

        tok_path = str(self.run_dir / "tokenizer")
        max_len = self.config.dataset.max_length
        bin_path = self.run_dir / "tokenized.bin"
        meta_path = self.run_dir / "tokenized.meta.json"

        cached = _load_tokenized_cache(bin_path, meta_path, max_len)
        if cached is not None:
            logger.info(f"Loaded {len(cached)} tokenized samples from cache ({bin_path.name})")
            return cached

        results: list[tuple[list[int], list[int]]] = []
        if Path(tok_path).exists():
            # Use more workers for tokenization (CPU-bound, doesn't compete with GPU training)
            # Cap at 8 to avoid excessive memory usage
            n_workers = min(8, max(2, os.cpu_count() or 4))
            chunk_size = max(100, math.ceil(n_samples / n_workers))
            chunks: list[list[dict]] = []
            for i in range(0, n_samples, chunk_size):
                chunk_dicts = []
                for sample in samples[i : i + chunk_size]:
                    if hasattr(sample, "__dict__"):
                        chunk_dicts.append(sample.__dict__)
                    else:
                        chunk_dicts.append(dict(sample))  # type: ignore[arg-type]
                chunks.append(chunk_dicts)

            logger.info(f"Tokenizing {n_samples} samples with {n_workers} workers (chunk size ~{chunk_size})...")
            # Use 'fork' on Linux for faster worker startup (avoids pickle overhead)
            ctx = None
            if sys.platform != "darwin":
                ctx = multiprocessing.get_context("fork")
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
                futures = [
                    executor.submit(_tokenize_chunk, chunk, tok_path, self.template, max_len)
                    for chunk in chunks
                ]
                for future in futures:
                    results.extend(future.result())
            logger.info(f"Tokenization complete: {len(results)} samples ready")
        else:
            logger.info(f"Tokenizing {n_samples} samples (single-threaded fallback — tokenizer not saved to disk)...")
            out: list[tuple[list[int], list[int]]] = []
            for sample in samples:
                try:
                    ids, labels = tokenize_sample(
                        sample,
                        self.tokenizer,
                        self.template,
                        max_length=max_len,
                    )
                except Exception as exc:
                    logger.warning(f"Failed to tokenize sample: {exc}")
                    continue
                if ids:
                    out.append((ids, labels))
            results = out
            logger.info(f"Tokenization complete: {len(results)} samples ready")

        _save_tokenized_cache(results, bin_path, meta_path, max_len)
        return results

    # ----- Callback helpers -----

    def _fire(self, hook_name: str) -> None:
        """Invoke a callback hook by name with the trainer context."""
        valid_hooks = (
            "on_train_begin",
            "on_train_end",
            "on_epoch_begin",
            "on_epoch_end",
            "on_step_begin",
            "on_step_end",
            "on_eval_begin",
            "on_eval_end",
            "on_early_stop",
            "on_checkpoint_saved",
        )
        if hook_name not in valid_hooks:
            raise ValueError(f"Unknown hook: {hook_name}")
        ctx = self._ctx()
        for cb in self.callbacks:
            getattr(cb, hook_name)(ctx)

    def _ctx(self) -> CallbackContext:
        """Build a callback context with the current trainer state."""
        return CallbackContext(trainer=self, state=self.callback_ctx_state)

    def _write_history(self, history: list[dict]) -> None:
        """Persist the per-eval history next to the run folder."""
        with open(self.run_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    def _final_metrics(self, history: list[dict], elapsed: float) -> dict[str, Any]:
        """Build the final metrics dictionary for the CLI / API."""
        best: dict[str, float | None] = {
            "loss": None,
            "perplexity": None,
            "accuracy": None,
        }
        for entry in history:
            for key in ("loss", "perplexity", "accuracy"):
                v = entry.get(key)
                if v is None:
                    continue
                cur = best[key]
                if (
                    cur is None
                    or (key == "perplexity" and v < cur)
                    or (key == "loss" and v < cur)
                    or (key == "accuracy" and v > cur)
                ):
                    best[key] = v

        if self.val_tokenized:
            try:
                summary = self.evaluate()
            except Exception as exc:
                logger.warning(f"Final evaluation failed: {exc}. Skipping.")
                summary = {
                    "loss": self.last_loss or 0.0,
                    "perplexity": math.exp(self.last_loss) if self.last_loss else 1.0,
                    "accuracy": 0.0,
                }
        else:
            summary = {
                "loss": self.last_loss or 0.0,
                "perplexity": math.exp(self.last_loss) if self.last_loss else 1.0,
                "accuracy": 0.0,
            }
        return {
            "elapsed_seconds": round(elapsed, 1),
            "epochs": self.epoch_index + 1,
            "global_step": self.global_step,
            "samples_per_sec": round(self._total_samples / elapsed, 1) if elapsed else 0.0,
            "tokens_per_sec": round(self._total_tokens / elapsed, 1) if elapsed else 0.0,
            "final": summary,
            "best": {f"val_{k}": v for k, v in best.items() if v is not None},
        }

    # ----- Peak-memory tracking -----

    def _sample_peak_memory(self) -> None:
        """Track CPU + GPU peak working-set across the run (best-effort)."""
        try:
            import resource

            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            self._max_cpu_mem_bytes = max(self._max_cpu_mem_bytes, rss)
        except (ImportError, OSError, ValueError):
            pass
        try:
            import torch

            if torch.cuda.is_available():
                # ``max_memory_allocated`` is the per-tensor high-water mark
                # for *active* allocations. Combined with reserved for
                # cache we get a complete picture.
                alloc = torch.cuda.max_memory_allocated()
                self._max_gpu_mem_bytes = max(self._max_gpu_mem_bytes, alloc)
        except Exception:  # pragma: no cover
            pass

    def _print_throughput_summary(self, elapsed: float) -> None:
        """Print samples/sec and tokens/sec final numbers."""
        if elapsed <= 0:
            return
        s_s = self._total_samples / elapsed
        t_s = self._total_tokens / elapsed
        logger.info(f"Samples/sec........ {s_s:.1f}")
        logger.info(f"Tokens/sec......... {t_s:.1f}")

    def _print_peak_memory(self) -> None:
        """Print peak CPU/GPU memory."""
        if self._max_cpu_mem_bytes:
            # ``ru_maxrss``: bytes on macOS, kilobytes on Linux.
            mb = self._max_cpu_mem_bytes / 1024 / 1024
            if mb < 10:
                mb = self._max_cpu_mem_bytes / 1024
            logger.info(f"Peak CPU RAM........ {mb:.1f}MB")
        if self._max_gpu_mem_bytes:
            mb = self._max_gpu_mem_bytes / 1024 / 1024
            logger.info(f"Peak GPU RAM........ {mb:.1f}MB")


def build_callbacks(config: FlatBuildConfig, run_dir: Path) -> list[Callback]:
    """Default callback chain: gradient clipping + (optional) early stopping.

    The previous default also wired in a logging callback. The
    refactored trainer reports progress through tqdm, so the logger
    callback was removed in favor of a single point of control.
    Early-stopping is added when configured.
    """
    from flatbuild.callbacks import EarlyStoppingCallback, GradientClipCallback

    chain: list[Callback] = [
        GradientClipCallback(max_norm=config.trainer.max_grad_norm),
    ]
    if config.trainer.early_stopping.enabled:
        chain.append(
            EarlyStoppingCallback(
                patience=config.trainer.early_stopping.patience,
                min_delta=config.trainer.early_stopping.min_delta,
                monitor=config.trainer.early_stopping.monitor,
            )
        )
    return chain


def train(
    config: FlatBuildConfig,
    run_dir: Path,
    *,
    samples: list[Sample],
    val_samples: list[Sample] | None = None,
    tokenizer: Tokenizer,
    model: FlatbuildModel | None = None,
    load_checkpoint_dir: Path | str | None = None,
    profile: bool = False,
) -> TrainArtifacts:
    """One-shot helper that builds the trainer and runs training.

    Args:
        config: Top-level flatbuild config.
        run_dir: Output directory for this run.
        samples: Training samples.
        val_samples: Optional validation samples.
        tokenizer: Trained tokenizer.
        model: Optional pre-built model.
        load_checkpoint_dir: Optional checkpoint to load weights from.
        profile: Enable :class:`PerformanceProfiler` for the run.

    Returns:
        :class:`TrainArtifacts` populated with final metrics.
    """
    if model is None:
        model = FlatbuildModel(config.model)
    if load_checkpoint_dir is not None:
        loaded = CheckpointManager.load(load_checkpoint_dir)
        sd = loaded["model_state_dict"]
        model.load_state_dict_llama(sd, strict=False)
    callbacks = build_callbacks(config, run_dir)
    trainer = FlatbuildTrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        samples=samples,
        val_samples=val_samples,
        run_dir=run_dir,
        callbacks=callbacks,
        profile=profile,
    )
    return trainer.train()
