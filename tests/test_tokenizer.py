"""Tests for the BPE tokenizer and chat template."""

from __future__ import annotations

import json

import pytest

from flatbuild.tokenizers.bpe import BPETokenizer
from flatbuild.tokenizers.template import (
    to_flatrun_jinja,
)


def test_chat_template_renders_simple_chat():
    """System + user + assistant produces a single concatenated string."""
    from flatbuild.tokenizers.template import ChatTemplate

    tmpl = ChatTemplate(
        system="You are Flatbot.",
        user_prefix="<|user|>\n",
        assistant_prefix="<|assistant|>\n",
        end_of_turn="<|endoftext|>",
        separator="\n\n",
        name="test",
    )
    rendered = tmpl.render(
        [
            ("system", "You are Flatbot."),
            ("user", "Hi"),
            ("assistant", "Hello!"),
        ]
    )
    assert "You are Flatbot." in rendered
    assert "<|user|>\nHi" in rendered
    assert "<|assistant|>\nHello!" in rendered
    assert rendered.endswith("<|endoftext|>")


def test_chat_template_add_generation_prompt():
    """add_generation_prompt emits an assistant prefix at the end."""
    from flatbuild.tokenizers.template import ChatTemplate

    tmpl = ChatTemplate(
        system=None,
        user_prefix="<|user|>\n",
        assistant_prefix="<|assistant|>\n",
        end_of_turn="<|endoftext|>",
        separator="\n\n",
        name="test",
    )
    rendered = tmpl.render([("user", "Hi")], add_generation_prompt=True)
    assert rendered.rstrip().endswith("<|assistant|>")


def test_bpe_train_encode_decode(tmp_path):
    """Round-trip encode/decode of a small corpus."""
    corpus = [
        "Hello world!",
        "Goodbye world!",
        "Transformers are great.",
        "Hello again.",
        "Once upon a time.",
    ] * 4
    tok = BPETokenizer.train(corpus, vocab_size=64, min_frequency=1)
    assert tok.vocab_size > 8
    ids = tok.encode("Hello world")
    assert all(isinstance(i, int) for i in ids)
    decoded = tok.decode(ids)
    assert decoded.replace(" ", "").lower().startswith("helloworld".replace(" ", ""))
    tok.save(tmp_path / "tok")
    loaded = BPETokenizer.load(tmp_path / "tok")
    assert loaded.vocab_size == tok.vocab_size
    assert loaded.encode("Hello world") == tok.encode("Hello world")


def test_bpe_special_tokens(tmp_path):
    """Special tokens are present in the trained tokenizer."""
    tok = BPETokenizer.train(["a b c", "b c d"], vocab_size=32, min_frequency=1)
    assert tok.eos_token_id >= 0
    assert tok.bos_token_id >= 0
    assert tok.pad_token_id >= 0


def test_bpe_save_writes_chat_template(tmp_path):
    """save() persists the chat_template into tokenizer_config.json."""
    tok = BPETokenizer.train(["hello world"], vocab_size=32, min_frequency=1)
    jinja = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    tok.save(tmp_path / "tok", chat_template=jinja)
    with open(tmp_path / "tok" / "tokenizer_config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    assert cfg["chat_template"] == jinja
    # Loading keeps it available for downstream consumers via the file.
    loaded = BPETokenizer.load(tmp_path / "tok")
    assert loaded is not None


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"separator": "\n\n"},
        {"system": None},
    ],
)
def test_to_flatrun_jinja_produces_valid_template(overrides):
    """Generated Jinja renders byte-identically to ChatTemplate.render.

    Runs the generated template through Flatrun's own (restricted) Jinja
    renderer when the sibling ``flatrun`` package is importable. Skipped
    otherwise.
    """
    fr = pytest.importorskip("flatrun.tokenizer.bpe")
    from flatbuild.tokenizers.template import ChatTemplate

    kwargs = dict(
        system="You are Flatbot, a helpful assistant.",
        user_prefix="\n\n<|user|>\n",
        assistant_prefix="\n\n<|assistant|>\n",
        end_of_turn="<|endoftext|>",
        separator="",
        name="test",
    )
    kwargs.update(overrides)
    tmpl = ChatTemplate(**kwargs)
    jinja = to_flatrun_jinja(tmpl)
    fr_tok = fr.BPETokenizer(vocab={}, merges=[], chat_template=jinja)

    messages = [
        ("system", "You are Flatbot, a helpful assistant."),
        ("user", "Hi there"),
        ("assistant", "Hello! What can I do for you?"),
        ("user", "Explain machine learning simply."),
    ]
    for add_gen in (True, False):
        fb_text = tmpl.render(messages, add_generation_prompt=add_gen)
        fr_msgs = [{"role": r, "content": c} for r, c in messages]
        fr_text = fr_tok.apply_chat_template(fr_msgs, add_generation_prompt=add_gen)
        assert fb_text == fr_text
