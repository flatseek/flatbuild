"""Regression tests for the preprocessing pipeline.

These tests pin the contract that ``ChatTemplate.render`` and
``tokenize_sample`` produce byte-for-byte identical strings (modulo the
separators between roles, which both encode the same way).

If these tests fail, training and inference are out of sync —
that is the bug pattern reported by users who see "loss collapses
but generation is gibberish".
"""

from __future__ import annotations

import pytest
import torch

from flatbuild.config import ChatTemplateConfig, TokenizerConfig
from flatbuild.datasets.base import ConversationSample
from flatbuild.tokenizers.bpe import BPETokenizer
from flatbuild.tokenizers.template import ChatTemplate, build_chat_template
from flatbuild.trainer.tokenize import tokenize_sample


def _make_template() -> ChatTemplate:
    return build_chat_template(
        ChatTemplateConfig(
            system="You are Flatbot, a helpful assistant.",
            user_prefix="<|user|>\n",
            assistant_prefix="<|assistant|>\n",
            end_of_turn="<|endoftext|>",
            separator="\n\n",
        )
    )


def _make_tokenizer() -> BPETokenizer:
    corpus = [
        "You are Flatbot, a helpful assistant.<|endoftext|>",
        "<|user|>\nHello<|endoftext|>",
        "<|assistant>\nHi!<|endoftext|>",
        "How many days are in April?\n\n<|assistant>\nApril has 30 days.<|endoftext|>",
        "Translate hello to Spanish.\n\n<|assistant>\n'hola'.<|endoftext|>",
    ]
    return BPETokenizer.train(corpus, vocab_size=128, min_frequency=1)


@pytest.mark.parametrize(
    "sample",
    [
        ConversationSample(
            messages=(
                ("system", "You are Flatbot, a helpful assistant."),
                ("user", "How many days are in April?"),
                ("assistant", "April has 30 days."),
            )
        ),
        ConversationSample(
            messages=(
                ("user", "Hi"),
                ("assistant", "Hello! How can I help?"),
            )
        ),
        ConversationSample(
            messages=(
                ("system", "You are a translator."),
                ("user", "Translate 'hello' to Spanish."),
                ("assistant", "'hola' in Spanish."),
            )
        ),
        ConversationSample(
            messages=(
                ("system", "System one."),
                ("user", "First user turn."),
                ("assistant", "First reply."),
                ("user", "Second user turn."),
                ("assistant", "Second reply."),
            )
        ),
    ],
    ids=["basic-sys", "no-sys", "translator", "multi-turn"],
)
def test_tokenize_sample_matches_chat_template_render(sample: ConversationSample):
    """Training text (from tokenize_sample) == inference text (from ChatTemplate.render)."""
    template = _make_template()
    tokenizer = _make_tokenizer()

    ids, labels = tokenize_sample(sample, tokenizer, template, max_length=512)
    training_text = tokenizer.decode(ids)
    inference_complete = template.render(sample.messages)
    assert training_text == inference_complete, (
        f"tokenize_sample produces text that is NOT byte-identical to\n"
        f"ChatTemplate.render(sample.messages):\n"
        f"  training : {training_text!r}\n"
        f"  inference: {inference_complete!r}"
    )


def test_inference_prompt_matches_training_prefix():
    """The --chat inference prompt is the prefix that ``tokenize_sample``
    feeds up to (and including) the assistant prefix — also byte-identical."""
    template = _make_template()
    tokenizer = _make_tokenizer()

    sample = ConversationSample(
        messages=(
            ("system", "You are Flatbot, a helpful assistant."),
            ("user", "How many days are in April?"),
            ("assistant", "April has 30 days."),
        )
    )

    inference_messages: list[tuple[str, str]] = []
    if template.system:
        inference_messages.append(("system", template.system))
    inference_messages.append(("user", "How many days are in April?"))
    inference_text = template.render(inference_messages, add_generation_prompt=True)

    expected_prefix = tokenizer.encode(inference_text)
    ids, _ = tokenize_sample(sample, tokenizer, template, max_length=512)

    assert ids[: len(expected_prefix)] == expected_prefix, (
        f"Inference prompt tokens are not the prefix of training tokens.\n"
        f"  rendered inference text: {inference_text!r}\n"
        f"  expected first N training tokens: {expected_prefix[:8]}\n"
        f"  actual first N training tokens  : {ids[:8]}"
    )


def test_assistant_prefix_is_present_and_masked_in_training():
    """The trailing ``<|assistant|>\\n`` at the end of the user turn
    must be present in input_ids and masked in labels — same bytes as
    inference will see as the start of its reply."""
    template = _make_template()
    tokenizer = _make_tokenizer()
    sample = ConversationSample(
        messages=(
            ("user", "Hello"),
            ("assistant", "Hi!"),
        )
    )
    ids, labels = tokenize_sample(sample, tokenizer, template, max_length=512)

    inference_prompt = template.render(
        [("user", "Hello")], add_generation_prompt=True
    )
    expected_prefix = tokenizer.encode(inference_prompt)

    assert ids[: len(expected_prefix)] == expected_prefix, (
        f"Training does not start with the inference prompt prefix.\n"
        f"  inference prompt: {inference_prompt!r}\n"
        f"  training prefix ids: {ids[:len(expected_prefix)]}\n"
        f"  expected prefix ids: {expected_prefix}"
    )
    assert all(l == -100 for l in labels[: len(expected_prefix)]), (
        f"Prefix tokens at the start of training must be masked.\n"
        f"  labels prefix: {labels[:len(expected_prefix)]}"
    )


def test_only_assistant_tokens_compute_loss():
    """Loss is computed only on assistant content + EOS (not the assistant prefix)."""
    template = _make_template()
    tokenizer = _make_tokenizer()
    sample = ConversationSample(
        messages=(
            ("system", "You are Flatbot."),
            ("user", "Hi"),
            ("assistant", "Hello!"),
        )
    )
    ids, labels = tokenize_sample(sample, tokenizer, template, max_length=512)
    first_loss_idx = next((i for i, l in enumerate(labels) if l != -100), -1)
    assert first_loss_idx >= 0, "no assistant labels found in tokenized sample"
    decoded_from_loss = tokenizer.decode(ids[first_loss_idx:])
    assert decoded_from_loss.startswith("Hello!"), (
        f"Loss tokens should start with assistant content, but decode is:\n"
        f"  {decoded_from_loss!r}"
    )


def test_eos_loss_signal_present():
    """The trailing ``<|endoftext|>`` token after the assistant reply IS
    a loss target — the model must learn when to stop."""
    template = _make_template()
    tokenizer = _make_tokenizer()
    sample = ConversationSample(
        messages=(
            ("user", "Hi"),
            ("assistant", "Hello!"),
        )
    )
    ids, labels = tokenize_sample(sample, tokenizer, template, max_length=512)
    assert labels[-1] != -100, "Final label is masked — model cannot learn EOS."
    assert labels[-1] == tokenizer.eos_token_id, (
        f"Final label token id ({labels[-1]}) is not eos_token_id "
        f"({tokenizer.eos_token_id})."
    )
