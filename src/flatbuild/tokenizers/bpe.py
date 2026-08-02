"""BPE tokenizer built on top of HuggingFace ``tokenizers``.

We pick HuggingFace ``tokenizers`` because it is a well-tested Rust-backed
implementation, but we hide it behind a minimal interface so the rest of
the codebase stays portable.

A :class:`BPETokenizer` can be trained on a corpus, saved to a directory,
loaded back, and exposes ``encode`` / ``decode`` methods over a list of
integers. Special tokens always include ``<|endoftext|>``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol


class Tokenizer(Protocol):
    """Minimal tokenizer contract used by Flatbuild's training pipeline."""

    @property
    def vocab_size(self) -> int:
        """Size of the vocabulary (excluding special tokens)."""

    @property
    def eos_token_id(self) -> int:
        """End-of-sequence token id."""

    @property
    def bos_token_id(self) -> int | None:
        """Beginning-of-sequence token id, if any."""

    @property
    def pad_token_id(self) -> int | None:
        """Padding token id, if any."""

    def encode(self, text: str) -> list[int]:
        """Encode text into a list of token ids."""

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token ids back to text."""

    def save(self, directory: str | Path) -> None:
        """Persist tokenizer files to ``directory``."""

    @classmethod
    def load(cls, directory: str | Path) -> "Tokenizer":
        """Load a previously saved tokenizer from ``directory``."""


@dataclass
class BPETokenizer:
    """Byte-level BPE wrapper over HuggingFace ``tokenizers``."""

    _inner: object  # tokenizers.Tokenizer instance
    eos_token: str = "<|endoftext|>"
    pad_token: str = "<|pad|>"
    unk_token: str = "<|unk|>"
    bos_token: str = "<|bos|>"

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        *,
        vocab_size: int = 4096,
        min_frequency: int = 2,
        added_tokens: Iterable[str] | None = None,
        save_to: str | Path | None = None,
    ) -> "BPETokenizer":
        """Train a new BPE tokenizer on an iterable of texts.

        Args:
            texts: Iterable of training strings.
            vocab_size: Target vocabulary size (excluding special tokens).
            min_frequency: Minimum frequency for a token to be kept.
            added_tokens: Tokens to force into the vocabulary.
            save_to: If given, save the resulting tokenizer to this directory.

        Returns:
            A trained :class:`BPETokenizer`.
        """
        try:
            from tokenizers import Tokenizer as _Tokenizer
            from tokenizers.models import BPE
            from tokenizers.pre_tokenizers import ByteLevel as _ByteLevel
            from tokenizers.decoders import ByteLevel as _ByteLevelDec
            from tokenizers.trainers import BpeTrainer
        except ImportError as exc:
            raise ImportError(
                "BPE training requires `pip install tokenizers`"
            ) from exc

        added = [cls.eos_token, cls.pad_token, cls.bos_token]
        if added_tokens:
            added.extend(t for t in added_tokens if t not in added)
        # Always reserve an unk for robustness.
        if cls.unk_token not in added:
            added.append(cls.unk_token)

        tokenizer = _Tokenizer(BPE(unk_token=cls.unk_token))
        tokenizer.pre_tokenizer = _ByteLevel(add_prefix_space=False)
        tokenizer.decoder = _ByteLevelDec()
        trainer = BpeTrainer(
            vocab_size=vocab_size + len(added),
            min_frequency=min_frequency,
            special_tokens=added,
            initial_alphabet=ByteLevel_alphabet(),
            show_progress=False,
        )

        def text_iterator():
            for chunk in texts:
                if chunk:
                    yield chunk

        tokenizer.train_from_iterator(text_iterator(), trainer=trainer)

        instance = cls(_inner=tokenizer)
        if save_to is not None:
            instance.save(save_to)
        return instance

    @classmethod
    def load(cls, directory: str | Path) -> "BPETokenizer":
        """Load a tokenizer previously saved with :meth:`save`."""
        directory = Path(directory)
        try:
            from tokenizers import Tokenizer as _Tokenizer
        except ImportError as exc:
            raise ImportError("BPE loading requires `pip install tokenizers`") from exc

        inner = _Tokenizer.from_file(str(directory / "tokenizer.json"))
        return cls(_inner=inner)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        inner = self._inner
        return int(inner.get_vocab_size())

    @property
    def eos_token_id(self) -> int:
        return int(self._inner.token_to_id(self.eos_token))

    @property
    def bos_token_id(self) -> int | None:
        return int(self._inner.token_to_id(self.bos_token))

    @property
    def pad_token_id(self) -> int | None:
        return int(self._inner.token_to_id(self.pad_token))

    @property
    def unk_token_id(self) -> int:
        return int(self._inner.token_to_id(self.unk_token))

    # ------------------------------------------------------------------
    # Encode / decode
    # ------------------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """Encode a piece of text into a list of token ids."""
        if not text:
            return []
        return [int(t) for t in self._inner.encode(text).ids]

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token ids back to text."""
        if not ids:
            return ""
        # Default to KEEPING special tokens so a decode/encode round-trip
        # is lossless (otherwise HF defaults strip them silently).
        return str(self._inner.decode([int(i) for i in ids], skip_special_tokens=False))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str | Path) -> None:
        """Persist the tokenizer to ``directory``.

        Writes the following files:

        - ``tokenizer.json`` — main tokenizer file.
        - ``tokenizer_config.json`` — metadata + special-token mappings.
        - ``chat_template.json`` — descriptive metadata.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        # Inner save
        if hasattr(self._inner, "save"):
            self._inner.save(str(directory / "tokenizer.json"))
        # Companion config
        config = {
            "tokenizer_class": "FlatbuildBPETokenizer",
            "eos_token": self.eos_token,
            "bos_token": self.bos_token,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "vocab_size": self.vocab_size,
        }
        with open(directory / "tokenizer_config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        # Small manifest so external tools can discover the format.
        with open(directory / "chat_template.json", "w", encoding="utf-8") as f:
            json.dump({"name": "flatbuild-default"}, f, indent=2)


def ByteLevel_alphabet() -> list[str]:
    """Return the GPT-2 byte-level alphabet (256 byte tokens).

    Returns:
        A list of single-character strings covering all possible bytes.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return [chr(c) for c in cs]
