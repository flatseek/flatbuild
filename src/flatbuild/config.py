"""YAML-driven configuration for Flatbuild.

A single :class:`FlatBuildConfig` dataclass holds every setting used by
the trainer, model, tokenizer, dataset, exporters and CLI commands.

Configs are loaded with ``FlatBuildConfig.from_yaml(path)``. Saving back
out (for reproducibility snapshots) uses :meth:`FlatBuildConfig.to_yaml`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Precision(str, Enum):
    """Training numeric precision."""

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"


class SampleType(str, Enum):
    """How a raw dataset sample should be normalized."""

    PRETRAINING = "pretraining"
    INSTRUCTION = "instruction"
    CONVERSATION = "conversation"


class TokenizerSource(str, Enum):
    """Where the BPE tokenizer comes from."""

    TRAIN = "train"
    LOAD = "load"


class NormType(str, Enum):
    """Normalization layer."""

    RMSNORM = "rmsnorm"
    LAYERNORM = "layernorm"


class ActivationType(str, Enum):
    """Feed-forward activation."""

    SWIGLU = "swiglu"
    RELU2 = "relu2"
    GELU = "gelu"


class ExportFormat(str, Enum):
    """Checkpoint export formats."""

    SAFETENSORS = "safetensors"
    HUGGINGFACE = "huggingface"


class DatasetFormat(str, Enum):
    """Source dataset file format."""

    JSONL = "jsonl"
    PARQUET = "parquet"
    HF = "hf"


# ---------------------------------------------------------------------------
# Component configs
# ---------------------------------------------------------------------------


@dataclass
class DatasetConfig:
    """Dataset loader settings."""

    format: DatasetFormat = DatasetFormat.JSONL
    path: str = ""
    field_mapping: dict[str, str] = field(default_factory=dict)
    train_split: float = 0.9
    val_split: float = 0.05
    test_split: float = 0.05
    max_samples: int | None = None
    max_length: int = 512
    seed: int = 42
    sample_type: SampleType = SampleType.CONVERSATION
    hf_kwargs: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["format"] = self.format.value
        data["sample_type"] = self.sample_type.value
        return data


@dataclass
class TokenizerConfig:
    """Tokenizer settings."""

    source: TokenizerSource = TokenizerSource.TRAIN
    path: str | None = None
    vocab_size: int = 4096
    min_frequency: int = 2
    added_tokens: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["source"] = self.source.value
        return data


@dataclass
class ChatTemplateConfig:
    """How multi-turn messages get rendered to a single string."""

    system: str | None = "You are a helpful assistant."
    user_prefix: str = "<|user|>\n"
    assistant_prefix: str = "<|assistant|>\n"
    end_of_turn: str = "<|endoftext|>"
    separator: str = "\n\n"
    name: str = "flatbuild-default"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ModelConfig:
    """Transformer architecture hyperparameters."""

    vocab_size: int = 4096
    n_layers: int = 4
    n_heads: int = 4
    n_kv_heads: int | None = None
    hidden_dim: int = 256
    ffn_dim: int | None = None
    context_length: int = 256
    rope_theta: float = 10000.0
    rope_scaling: dict | None = None
    norm: NormType = NormType.RMSNORM
    activation: ActivationType = ActivationType.SWIGLU
    tie_embeddings: bool = True
    attention_dropout: float = 0.0
    residual_dropout: float = 0.0
    embedding_dropout: float = 0.0
    initializer_range: float = 0.02

    def __post_init__(self) -> None:
        # Default GQA ratio: half the heads for KV when not set.
        if self.n_kv_heads is None:
            self.n_kv_heads = max(1, self.n_heads // 2)
        # FFN roughly 8/3 * hidden_dim (round) for SwiGLU — Llama convention.
        if self.ffn_dim is None:
            mult = 8 // 3
            self.ffn_dim = int(round(self.hidden_dim * mult / 32) * 32)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["norm"] = self.norm.value
        data["activation"] = self.activation.value
        return data


@dataclass
class OptimizerConfig:
    """Optimizer hyperparameters."""

    type: str = "adamw"
    lr: float = 3.0e-4
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1.0e-8
    weight_decay: float = 0.1

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SchedulerConfig:
    """Learning-rate schedule."""

    type: str = "cosine"
    warmup_steps: int = 50
    min_lr_ratio: float = 0.1

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EarlyStoppingConfig:
    """Early-stopping knobs."""

    enabled: bool = False
    patience: int = 3
    min_delta: float = 0.0
    monitor: str = "val_loss"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainerConfig:
    """Training loop settings."""

    epochs: int = 1
    batch_size: int = 8
    gradient_accumulation: int = 4
    max_steps: int | None = None
    precision: Precision = Precision.BF16
    gradient_checkpointing: bool = False
    seed: int = 42
    eval_every_n_steps: int | None = None
    log_every_n_steps: int = 10
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    max_grad_norm: float = 1.0
    # Validation cadence (kept here for runtime use; the YAML key is
    # ``validation`` — see ``ValidationConfig`` below).
    # DataLoader knobs.
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: int = 2
    drop_last: bool = True

    def to_dict(self) -> dict:
        data = asdict(self)
        data["precision"] = self.precision.value
        return data


@dataclass
class ValidationConfig:
    """Validation loop behavior."""

    every_steps: int | None = None  # None → never inline; full eval at epoch end only
    max_batches: int | None = None  # None → all batches; integer → cap
    full_at_epoch_end: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CheckpointConfig:
    """Checkpoint saving / resuming."""

    every_n_steps: int = 200
    keep_last: int = 3
    save_final: bool = True
    resume_from: str | None = None
    async_write: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExportConfig:
    """Post-training export."""

    format: ExportFormat = ExportFormat.SAFETENSORS
    output_dir: str | None = None
    quant: str = "f16"
    publisher: str | None = None
    model_name: str | None = None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["format"] = self.format.value
        return data


@dataclass
class GenerateConfig:
    """Text generation defaults."""

    prompt: str = "Hello"
    max_new_tokens: int = 64
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 50
    do_sample: bool = True
    seed: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


@dataclass
class FlatBuildConfig:
    """Aggregate configuration. One per training run."""

    name: str = "flatbuild-run"
    description: str | None = None
    output_dir: str = "outputs"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    chat_template: ChatTemplateConfig = field(default_factory=ChatTemplateConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    generate: GenerateConfig = field(default_factory=GenerateConfig)

    # ------------------------------------------------------------------
    # (De)serialization)
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FlatBuildConfig":
        """Load config from a YAML file on disk.

        Args:
            path: YAML file path.

        Returns:
            A populated :class:`FlatBuildConfig`.
        """
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if data is None:
            raise ValueError(f"Empty YAML file: {path}")
        if not isinstance(data, dict):
            raise TypeError(
                f"Top-level YAML in {path} must be a mapping, got {type(data)}."
            )
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "FlatBuildConfig":
        """Build a config from an in-memory dictionary.

        Args:
            data: Configuration dictionary.

        Returns:
            A populated :class:`FlatBuildConfig`.
        """
        out = cls()

        for key in ("name", "description", "output_dir"):
            if key in data:
                setattr(out, key, data[key])

        if "dataset" in data and data["dataset"] is not None:
            ds = data["dataset"]
            _check_unknown_keys(DatasetConfig, ds, owner="dataset")
            out.dataset = DatasetConfig(**_strings_to_enums(ds, {"format": DatasetFormat, "sample_type": SampleType}))

        if "tokenizer" in data and data["tokenizer"] is not None:
            tk = data["tokenizer"]
            _check_unknown_keys(TokenizerConfig, tk, owner="tokenizer")
            out.tokenizer = TokenizerConfig(**_strings_to_enums(tk, {"source": TokenizerSource}))

        if "chat_template" in data and data["chat_template"] is not None:
            ct = data["chat_template"]
            _check_unknown_keys(ChatTemplateConfig, ct, owner="chat_template")
            out.chat_template = ChatTemplateConfig(**ct)

        if "model" in data and data["model"] is not None:
            md = data["model"]
            _check_unknown_keys(ModelConfig, md, owner="model")
            out.model = ModelConfig(**_strings_to_enums(md, {"norm": NormType, "activation": ActivationType}))

        if "optimizer" in data and data["optimizer"] is not None:
            op = data["optimizer"]
            _check_unknown_keys(OptimizerConfig, op, owner="optimizer")
            out.optimizer = OptimizerConfig(**op)

        if "scheduler" in data and data["scheduler"] is not None:
            sc = data["scheduler"]
            _check_unknown_keys(SchedulerConfig, sc, owner="scheduler")
            out.scheduler = SchedulerConfig(**sc)

        if "trainer" in data and data["trainer"] is not None:
            trainer_data = dict(data["trainer"])
            _check_unknown_keys(TrainerConfig, trainer_data, owner="trainer")
            if "precision" in trainer_data:
                trainer_data["precision"] = Precision(trainer_data["precision"])
            if "early_stopping" in trainer_data and trainer_data["early_stopping"] is not None:
                es = dict(trainer_data["early_stopping"])
                _check_unknown_keys(EarlyStoppingConfig, es, owner="trainer.early_stopping")
                trainer_data["early_stopping"] = EarlyStoppingConfig(**es)
            out.trainer = TrainerConfig(**trainer_data)

        if "checkpoint" in data and data["checkpoint"] is not None:
            ck = data["checkpoint"]
            _check_unknown_keys(CheckpointConfig, ck, owner="checkpoint")
            out.checkpoint = CheckpointConfig(**ck)

        if "validation" in data and data["validation"] is not None:
            vd = data["validation"]
            _check_unknown_keys(ValidationConfig, vd, owner="validation")
            out.validation = ValidationConfig(**vd)

        if "export" in data and data["export"] is not None:
            export_data = dict(data["export"])
            _check_unknown_keys(ExportConfig, export_data, owner="export")
            if "format" in export_data:
                export_data["format"] = ExportFormat(export_data["format"])
            out.export = ExportConfig(**export_data)

        if "generate" in data and data["generate"] is not None:
            out.generate = GenerateConfig(**data["generate"])

        return out

    def to_dict(self) -> dict:
        """Return a plain-dictionary snapshot of the config."""
        return {
            "name": self.name,
            "description": self.description,
            "output_dir": self.output_dir,
            "dataset": self.dataset.to_dict(),
            "tokenizer": self.tokenizer.to_dict(),
            "chat_template": self.chat_template.to_dict(),
            "model": self.model.to_dict(),
            "optimizer": self.optimizer.to_dict(),
            "scheduler": self.scheduler.to_dict(),
            "trainer": self.trainer.to_dict(),
            "validation": self.validation.to_dict(),
            "checkpoint": self.checkpoint.to_dict(),
            "export": self.export.to_dict(),
            "generate": self.generate.to_dict(),
        }

    def to_yaml(self, path: str | Path) -> None:
        """Write the config to ``path`` as pretty-printed YAML."""
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strings_to_enums(data: dict, mapping: dict) -> dict:
    """Best-effort ``str -> Enum`` coercion for config dicts.

    Args:
        data: Source dictionary.
        mapping: ``field-name -> EnumClass`` map.

    Returns:
        New dictionary with enum-valued keys converted.
    """
    new = dict(data)
    for key, enum_cls in mapping.items():
        if key in new and isinstance(new[key], str):
            new[key] = enum_cls(new[key])
    return new


def _check_unknown_keys(klass: type, data: dict, *, owner: str) -> None:
    """Raise :class:`ValueError` listing unknown keys in ``data``.

    Args:
        klass: Dataclass type whose ``__dataclass_fields__`` is the
            canonical set of accepted keys.
        data: Candidate field dictionary.
        owner: Path in the top-level config (used in the error message).
    """
    valid = set(getattr(klass, "__dataclass_fields__", {}).keys())
    extras = [k for k in data if k not in valid]
    if extras:
        joined = ", ".join(sorted(extras))
        allowed = ", ".join(sorted(valid))
        raise ValueError(
            f"Unknown key(s) in {owner}: {joined}. "
            f"Allowed keys: {allowed}"
        )
