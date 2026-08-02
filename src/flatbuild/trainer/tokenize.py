"""Tokenization helpers: render samples to token ids + label masks.

This module is the **training-time** consumer of :class:`ChatTemplate`.
It MUST stay byte-for-byte aligned with what the inference path
produces via :meth:`ChatTemplate.render` — see the audit report in
``scripts/audit_preprocessing.py``.

Implementation notes:

- Each role's contribution is rendered exactly via
  :meth:`ChatTemplate.render_message` (single source of truth).
- Inter-message separators are encoded separately and inserted
  between messages; labels for the separator are ``-100`` so they
  contribute no loss.
- The final user-turn separator (``separator + assistant_prefix``)
  IS included in input ids with ``-100`` labels, so the model sees
  the ``<|assistant|>\\n`` cue during training. This matches the
  inference ``add_generation_prompt=True`` rendering exactly.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

import torch

from flatbuild.datasets.base import (
    ConversationSample,
    InstructionSample,
    PretrainingSample,
    Sample,
)
from flatbuild.tokenizers.bpe import Tokenizer
from flatbuild.tokenizers.template import ChatTemplate


def tokenize_sample(
    sample: Sample,
    tokenizer: Tokenizer,
    template: ChatTemplate,
    *,
    max_length: int,
) -> Tuple[List[int], List[int]]:
    """Render and tokenize one sample.

    Args:
        sample: Normalized sample (pretraining / instruction / conversation).
        tokenizer: Tokenizer with ``encode`` / ``eos_token_id``.
        template: Chat template renderer.
        max_length: Max sequence length.

    Returns:
        ``(input_ids, labels)`` where ``input_ids`` is the full token
        sequence and ``labels`` matches ``input_ids`` except that
        ``-100`` masks any token the model is *not* expected to
        predict (system content, the separator between messages, the
        user prompt and the assistant prefix that the model sees but
        does not score).

        The token stream is byte-for-byte the same as if you ran
        ``tokenizer.encode(template.render(messages))`` and split it
        at the same per-message boundaries.
    """
    if isinstance(sample, PretrainingSample):
        ids = tokenizer.encode(sample.text)[:max_length]
        return ids, list(ids)

    messages = sample.messages if isinstance(sample, ConversationSample) else _to_messages(sample)

    boundary_input: List[int] = []
    boundary_label: List[int] = []
    for i, (role, content) in enumerate(messages):
        # Inter-message separator (same byte sequence as
        # ChatTemplate.render uses between messages).
        if i > 0:
            sep_ids = tokenizer.encode(template.separator)
            boundary_input.extend(sep_ids)
            boundary_label.extend([-100] * len(sep_ids))

        if role == "assistant":
            # Score ONLY the assistant content + EOS. The
            # ``<|assistant>\n`` prefix is part of the prompt and
            # must be masked so the model never learns to predict it.
            prefix_ids = tokenizer.encode(template.assistant_prefix)
            content_ids = tokenizer.encode(content)
            eos_ids = tokenizer.encode(template.end_of_turn)
            boundary_input.extend(prefix_ids + content_ids + eos_ids)
            boundary_label.extend(
                [-100] * len(prefix_ids) + content_ids + eos_ids
            )
        else:
            role_text = template.render_message(role, content)
            role_ids = tokenizer.encode(role_text)
            boundary_input.extend(role_ids)
            boundary_label.extend([-100] * len(role_ids))

    # If the conversation has no assistant, fall back to pretraining-style
    # loss over the full token sequence.
    if not boundary_label or all(l == -100 for l in boundary_label):
        ids = boundary_input[:max_length]
        return ids, list(ids)

    if len(boundary_input) > max_length:
        boundary_input = boundary_input[:max_length]
        boundary_label = boundary_label[:max_length]
    return boundary_input, boundary_label


def _to_messages(sample: InstructionSample) -> List[Tuple[str, str]]:
    """Convert an :class:`InstructionSample` into a ``(role, content)`` list."""
    user_turn = sample.instruction
    if sample.input:
        user_turn = f"{user_turn}\n\n{sample.input}"
    return [("user", user_turn), ("assistant", sample.output)]


def batch_samples(
    tokenized: Iterable[Tuple[List[int], List[int]]],
    *,
    pad_token_id: int,
    pad_to: int | None = None,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Pad a list of tokenized samples into batched tensors.

    Args:
        tokenized: Iterable of ``(input_ids, labels)`` pairs.
        pad_token_id: Padding token id.
        pad_to: If given, pad every row up to this length.

    Returns:
        ``(input_ids, labels)`` tensors of shape ``(B, T)``.
    """
    rows = [pair for pair in tokenized]
    if not rows:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
    max_len = max(len(r[0]) for r in rows)
    if pad_to is not None:
        max_len = max(max_len, pad_to)
    inputs = []
    labels = []
    for inp, lab in rows:
        pad_len = max_len - len(inp)
        if pad_len < 0:
            inp = inp[:max_len]
            lab = lab[:max_len]
            pad_len = 0
        inputs.append(inp + [pad_token_id] * pad_len)
        labels.append(lab + [-100] * pad_len)
    return torch.tensor(inputs, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
