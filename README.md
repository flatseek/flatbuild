<div align="center">

<img src="logo.svg" alt="Flatbuild" width="64" height="64">

# Flatbuild

**Build conversational language models from scratch.**

<p align="center">
  <em>
An open-source framework for training compact decoder-only language models
from random initialization using a modular, configuration-driven pipeline.
  </em>
</p>

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Tests](https://github.com/flatseek/flatbuild/workflows/Test/badge.svg)](https://github.com/flatseek/flatbuild/actions)
[![PyPI version](https://img.shields.io/pypi/v/flatbuild.svg)](https://pypi.org/project/flatbuild/)

**GitHub:** https://github.com/flatseek/flatbuild
&nbsp;&middot;&nbsp;
**Organization:** https://github.com/flatseek

<br>

**Part of the Flatseek ecosystem**

[Flatseek](https://github.com/flatseek/flatseek) (Keyword Search) •
[Flatvec](https://github.com/flatseek/flatvec) (Vector Search) •
[Flatask](https://github.com/flatseek/flatask) (RAG Runtime) •
[Flatlens](https://github.com/flatseek/flatlens) (Data Visualization) •
[Flatbuild](https://github.com/flatseek/flatbuild) (LLM Training) •
[Flattune](https://github.com/flatseek/flattune) (LLM Fine-Tuning) •
[Flatrun](https://github.com/flatseek/flatrun) (LLM Inference)

</div>

---
# See It In Action

Flatbuild includes everything required to train a language model from scratch—from conversational datasets and tokenizer training to checkpoint export and inference.

The repository includes a small conversational dataset (`data/demo_chat`) that demonstrates the complete workflow.

Example conversation:

```json
{"messages":[{"role":"system","content":"You are Flatbot, a friendly conversational assistant. You help users understand things, solve problems, and have natural conversations. Be clear, helpful, and concise."},{"role":"user","content":"What gas do plants absorb?"},{"role":"assistant","content":"Plants absorb carbon dioxide from the atmosphere."}]}
```

Train the demonstration model:

```bash
flatbuild train configs/demo_chat.yaml
```

Example training output:

```text
Loaded config: configs/demo_chat.yaml
Project: demo-chat
Loaded 2500 conversations
Train: 2375  Validation: 125

Epoch 1/12  loss=4.9630  val_loss=5.2337  ppl=187.49  acc=36.4%
...
Epoch 12/12 loss=0.8245  val_loss=1.4638  ppl=4.32    acc=72.8%

Run completed.

Artifacts:
outputs/demo-chat/20260802T014922Z
```

Export the trained checkpoint:

```bash
flatbuild export \
    outputs/demo-chat/20260802T014922Z/checkpoints/final \
    --format gguf
```

```text
Wrote model.gguf

outputs/demo-chat/20260802T014922Z/checkpoints/export_gguf
```

The same training pipeline is used to produce **[Flatbot-micro-4M](https://huggingface.co/flatseek/flatbot-micro-4M)**, the official demonstration model of the Flatseek ecosystem.

Run it directly with **[Flatrun](https://github.com/flatseek/flatrun)**:

```bash
pip install flatrun
```

```bash
flatrun chat \
    --model flatbot-micro-4M.gguf \
    --temp 0.2
```

Example session:

```text
Detected format: gguf
Building tokenizer from GGUF metadata (flatbot-micro-4M.gguf) ...
Tokenizer vocab: 516
Loaded model in 0.01 s; layers=6

You: Who are you?

Assistant:
Sure — I'm Flatbot — a small conversational assistant trained from scratch using Flatbuild.
```

**Flatbot-micro-4M** is a compact (~4M parameter) decoder-only transformer trained entirely from random initialization using Flatbuild. It exists as a reproducible end-to-end demonstration of the Flatseek ecosystem, showing how a modern conversational language model can be built—from dataset generation to deployment—without relying on pretrained weights.

While the model is intentionally small and not intended to compete with large language models, the same training and export pipeline scales directly to larger datasets and larger transformer architectures.

---

# Overview

**A modular training framework for building compact language models.**

Flatbuild is an open-source Python framework for training decoder-only language models entirely from random initialization.

Instead of requiring a large pre-trained checkpoint, Flatbuild provides every component of the training pipeline—including datasets, tokenizer training, model architecture, optimization, checkpointing, evaluation, and exporting—inside a single configurable framework.

Every training run is driven by a YAML configuration, making experiments reproducible, portable, and easy to automate.

---

# Why Flatbuild?

Most modern training frameworks assume you already have a base model.

Flatbuild starts one layer lower.

It provides a complete reference implementation of the entire language-model training pipeline where every module can be inspected, replaced, extended, or optimized independently.

The resulting checkpoints can be exported to SafeTensors or HuggingFace Transformers and used directly by Flatrun or any compatible inference runtime.

---

# Highlights

- Train decoder-only Transformers from scratch
- YAML-driven reproducible experiments
- Built-in tokenizer training
- Grouped Query Attention (GQA)
- RoPE positional embeddings
- RMSNorm and SwiGLU
- Gradient accumulation
- Mixed precision (FP32 / FP16 / BF16)
- Configurable chat templates
- JSONL, Parquet, and HuggingFace datasets
- SafeTensors and HuggingFace export
- Modular architecture for research and experimentation

---

# Installation

```bash
pip install -e ".[dev]"
```

Development dependencies include:

- pytest
- ruff
- mypy

PyTorch is required at runtime.

Install the build matching your platform (CPU, CUDA, or Apple MPS).

---

# Configuration

Training is controlled entirely through a YAML file.

Example:

```yaml
name: demo-chat

dataset:
  type: conversation
  path: data/demo_chat/dataset.jsonl
  max_length: 384

tokenizer:
  source: train
  vocab_size: 1024

model:
  hidden_dim: 256
  n_layers: 4
  n_heads: 4
  n_kv_heads: 2
  context_length: 256

optimizer:
  type: adamw
  lr: 3e-4

trainer:
  epochs: 1
  batch_size: 8
```

See `configs/demo_chat.yaml` for the complete configuration.

---

# CLI

```bash
flatbuild train
flatbuild resume
flatbuild evaluate
flatbuild export
flatbuild generate
flatbuild inspect
flatbuild benchmark
```

Run:

```bash
flatbuild --help
```

for the complete command reference.

---

# Training Pipeline

```text
Dataset
    │
    ▼
Tokenizer
    │
    ▼
Model
    │
    ▼
Training Loop
    │
    ▼
Checkpoint
    │
    ▼
Evaluation
    │
    ▼
Export
```

Every stage is configurable and replaceable.

---

# Current Features

Flatbuild currently supports:

- Decoder-only Transformer
- Grouped Query Attention (GQA)
- Rotary Position Embeddings (RoPE)
- RMSNorm
- SwiGLU
- AdamW optimizer
- Cosine learning-rate scheduler
- Linear warmup
- Early stopping
- Gradient accumulation
- Mixed precision training
- BPE tokenizer training
- Chat template formatting
- JSONL datasets
- Parquet datasets
- HuggingFace datasets
- SafeTensors export
- HuggingFace Transformers export

---

# Roadmap

Future releases are planned to include:

- Supervised Fine-Tuning (SFT)
- LoRA
- QLoRA
- DPO
- MoE architectures
- Vision-language models
- Distributed training
- Multi-node training
- Advanced evaluation benchmarks

---
# Quick Start

Train the bundled demo model:

```bash
flatbuild train configs/demo_chat.yaml
```

A typical run will:

- Load the demo conversational dataset
- Train a tokenizer (or reuse an existing one)
- Train a decoder-only Transformer from scratch
- Save checkpoints, metrics, and metadata
- Export the final model

Outputs are written to:

```text
outputs/demo/<run-id>/
```

---

# License

Apache License 2.0

See `LICENSE`.
