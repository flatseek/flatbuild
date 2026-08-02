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

The repository includes a small conversational dataset that demonstrates the complete Flatbuild workflow—from dataset to training, export, and inference.

Example dataset (`data/demo/dataset.jsonl`):

```json
{"messages":[{"role":"system","content":"You are Flatbot, a helpful, friendly, and concise conversational assistant."},{"role":"user","content":"What gas do plants absorb?"},{"role":"assistant","content":"Plants absorb carbon dioxide from the atmosphere."}]}
```

Train the model:

```bash
flatbuild train configs/demo.yaml
```

Example output:

```text
Loaded config: configs/demo.yaml
Project: demo
Loaded 1000 samples
Train: 850  Validation: 150

Epoch 1/4  loss=4.9630  val_loss=5.2337  ppl=187.49  acc=36.4%
Epoch 2/4  loss=4.2745  val_loss=4.0206  ppl=55.73   acc=46.3%
Epoch 3/4  loss=2.6530  val_loss=3.2722  ppl=26.37   acc=49.6%
Epoch 4/4  loss=1.9804  val_loss=2.8162  ppl=16.71   acc=54.2%

Run completed.

Artifacts:
outputs/demo/20260802T014922Z
```

Export the trained checkpoint:

```bash
flatbuild export \
    outputs/demo/20260802T014922Z/checkpoints/final \
    --format safetensors
```

```text
Wrote model.safetensors
Copied tokenizer files

outputs/demo/20260802T014922Z/checkpoints/export_safetensors
```

Run the exported model with **[Flatrun](https://github.com/flatseek/flatrun)**.
```bash
pip install flatrun
```

Then run the exported checkpoint:

```bash
flatrun \
    --model outputs/demo-large/20260802T014922Z/checkpoints/export_safetensors \
    --prompt "who are you?"
```

Example output:

```text
Detected format: safetensors
Tokenizer vocab: 2603
Chat template: Qwen2 ChatML
Loaded model in 0.01 s

Prompt:
who are you?

Generated:
'I I I 1 1 1 Ocean classic...... c solar that that that'
```

This tiny demonstration trains a decoder-only language model from scratch using **1,000 conversational examples** and showcases the complete Flat pipeline:

```text
Conversation Dataset
        │
        ▼
 Flatbuild Training
        │
        ▼
   Model Checkpoint
        │
        ▼
 SafeTensors Export
        │
        ▼
 Flatrun Inference
```

The bundled demo model is intentionally small and only trained for a few epochs, so the generated text is not expected to be meaningful. Its purpose is to demonstrate the complete end-to-end workflow. The same pipeline scales to larger datasets and larger language models without changing the training or deployment process.

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
name: demo

dataset:
  type: conversation
  path: data/demo/dataset.jsonl
  max_length: 256

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

See `configs/demo.yaml` for the complete configuration.

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
flatbuild train configs/demo.yaml
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
