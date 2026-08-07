# Changelog

All notable changes to Flatbuild will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- (filled in by future releases)

## [0.1.2] - 2026-08-03

### Fixed
- **GGUF export**: Q/K weight permute for interleaved RoPE layout (matching
  `convert_hf_to_gguf.py`), resolving garbage output in llama.cpp.
- **Quantization**: call `gguf.quants.quantize()` explicitly; raw_dtype was
  metadata only. K-quants (Q2_K, Q4_K, etc.) now fail fast with clear error.

### Added
- **System-prompt injection**: default system prompt is now embedded into
  `chat_template` at export time, so callers don't need `--system` flag.
  Works in both Flatrun's restricted Jinja and LM Studio.
- `flatbuild quantize` CLI command for quantizing existing GGUF files.
- Demo iteration 3: `demo_chat_10m.yaml` (6L/384, 10.6M params) with
  uncertainty-aware system prompt.

### Refactored
- `ExportConfig` gained `chat_template` field; CLI propagates top-level config.
- `to_flatrun_jinja()` guards against duplicate system injection.
- Repo cleanup: `scripts/`, `test.gguf/`, `test.hf/`, `main.log` removed from
  tracking; added to `.gitignore`.

## [0.1.1] - 2026-08-02

### Added
- Flatseek pinned to 0.1.1 in dev dependencies.

## [0.1.0] - Initial

### Added
- First public release of Flatbuild.
- Decoder-only Transformer with GQA, RoPE, RMSNorm, SwiGLU.
- BPE tokenizer training on a corpus.
- Configurable chat template.
- Dataset loaders for JSONL, Parquet, and HuggingFace `datasets`.
- YAML-driven training configuration.
- Trainer with mixed precision, gradient accumulation, AdamW,
  cosine LR schedule with linear warmup, validation, early stopping.
- Callbacks: structured logging, gradient clipping, checkpoint saving,
  early stopping.
- Checkpoint save/load/resume.
- Exporters: SafeTensors and HuggingFace Transformers format.
- CLI: `train`, `resume`, `evaluate`, `export`, `generate`,
  `inspect`, `benchmark`.
- Built-in demo conversational dataset that trains in under a minute
  on CPU and exports to a working demo checkpoint.
