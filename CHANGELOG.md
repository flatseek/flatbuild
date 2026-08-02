# Changelog

All notable changes to Flatbuild will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Initial scaffold: project layout, packaging, CLI entry point, Makefile.
- (filled in by future releases)

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
