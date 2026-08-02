# Flatbuild

**Build conversational language models from scratch.**

Flatbuild is an open-source framework for training language models from random initialization.

It is part of the Flat ecosystem:

- Flatseek — Keyword Search
- Flatvec — Vector Search
- Flatask — RAG Runtime
- Flatrun — Streaming Inference Runtime
- Flatweight — AI Model Storage (future native format)
- Flattune — Fine-tuning Framework
- Flatbuild — Language Model Training Framework

---

# Vision

Flatbuild aims to make training language models as simple, modular, and reproducible as building modern software.

Rather than focusing solely on benchmark performance, Flatbuild emphasizes practical language models that are easy to build, customize, deploy, and integrate into real-world applications.

The long-term vision is to support language models of any scale, from compact assistants to large foundation models.

---

# Initial Focus

The first milestone is intentionally modest.

Train compact conversational language models that can:

- Hold natural conversations
- Follow user instructions
- Answer general questions
- Work seamlessly with Retrieval-Augmented Generation (RAG)
- Serve as the conversational layer for AI products

The objective is not to build the smartest model.

The objective is to build reliable conversational models that developers can own, customize, and deploy.

---

# Target Users

Flatbuild is designed for developers and organizations that want to build their own conversational AI.

Examples include:

- Internal company assistants
- Customer support chatbots
- Knowledge base assistants
- Documentation assistants
- Healthcare assistants
- Educational tutors
- Legal assistants
- Financial assistants
- AI agents
- Domain-specific language models

Flatbuild provides the conversational intelligence, while external systems such as RAG supply up-to-date knowledge.

---

# Design Principles

Prioritize:

- Simplicity
- Readability
- Reproducibility
- Modularity
- Extensibility
- Deterministic training
- Configuration-driven workflows

Avoid unnecessary abstractions.

Favor small independent components over monolithic trainer classes.

---

# Architecture

Organize the project into independent modules.

flatbuild/

    datasets/
    tokenizers/
    models/
    layers/
    optimizers/
    schedulers/
    losses/
    metrics/
    callbacks/
    checkpoint/
    exporters/
    trainer/
    cli/

Every module should be replaceable.

---

# Configuration

Training should be driven by YAML configuration files.

Example:

config.yaml

- dataset
- tokenizer
- model
- optimizer
- scheduler
- trainer
- evaluation
- checkpoint
- export

Users should rarely need to modify Python code.

---

# Dataset Support

Support multiple dataset formats.

Examples:

JSONL

Parquet

HuggingFace Dataset

Streaming datasets

All datasets should be normalized into a common internal format.

---

# Dataset Types

## Pretraining

{
    "text":"..."
}

---

## Instruction

{
    "instruction":"...",
    "input":"...",
    "output":"..."
}

---

## Conversation

{
    "messages":[
        {
            "role":"system",
            "content":"..."
        },
        {
            "role":"user",
            "content":"..."
        },
        {
            "role":"assistant",
            "content":"..."
        }
    ]
}

---

Internally every sample should become a standardized conversation before tokenization.

---

# Built-in Demo Dataset

Include a small demonstration dataset that allows users to train their first conversational model immediately.

The dataset should teach fundamental conversational behaviors instead of domain knowledge.

Examples include:

Greetings

Identity

Question answering

Instruction following

Summarization

Rewriting

Classification

Simple reasoning

Code generation

Polite refusals

Context switching

Follow-up questions

Conversation flow

The resulting demo model should already feel like a basic chatbot instead of a text completion model.

---

# Chat Template

Use configurable chat templates.

Example:

<|system|>

...

<|user|>

...

<|assistant|>

...

The template should not be hardcoded.

---

# Model

Initially support decoder-only Transformer architectures.

Features include:

- Multi-head Attention
- Grouped Query Attention
- RoPE
- RMSNorm
- SwiGLU
- KV Cache compatibility
- Configurable vocabulary
- Configurable context length

The architecture should be defined by configuration rather than hardcoded implementations.

---

# Training

Support:

- Mixed Precision
- Gradient Accumulation
- Checkpoint Resume
- Validation
- Evaluation
- Early Stopping
- Learning Rate Scheduling
- AdamW

The implementation should prioritize correctness and readability before optimization.

---

# Export

Initially support exporting to:

- SafeTensors
- HuggingFace Transformers

GGUF and Flatweight support may be added in future releases.

---

# CLI

Examples:

flatbuild train config.yaml

flatbuild evaluate checkpoint/

flatbuild resume checkpoint/

flatbuild export checkpoint/

flatbuild generate

flatbuild inspect checkpoint/

flatbuild benchmark

---

# Coding Standards

Write clean production-quality Python.

Use:

- Type hints
- Docstrings
- Unit tests
- Small focused classes
- Small functions

Avoid hidden side effects.

Keep code easy to understand and modify.

---

# Future Roadmap

Design the architecture so future capabilities can be added without major refactoring.

Examples include:

- Continued Pretraining
- Supervised Fine-tuning (SFT)
- LoRA
- QLoRA
- DPO
- RLHF
- Mixture of Experts
- Vision Models
- Multimodal Models
- Distributed Training
- Flatweight Native Checkpoints
- Streaming Checkpoints
- Out-of-Core Training

Do not implement these features yet.

Instead, ensure the architecture remains flexible enough to support them naturally as the project evolves.

---

# Success Criteria

The first public release should enable a developer to:

1. Prepare a conversational dataset.
2. Train a compact language model from scratch.
3. Export the model to SafeTensors.
4. Load it using Hugging Face or Flatrun (via conversion if needed).
5. Build a conversational AI application with RAG on top of it.

Flatbuild should be approachable for individual developers while providing an architecture capable of scaling toward larger foundation model training in future releases.
