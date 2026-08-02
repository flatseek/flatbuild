"""Verify Flatbuild <-> Flatrun interop assumptions before building the demo.

Checks two things empirically:

1. Chat-template byte-identity: flatrun's tiny-Jinja renderer (fed the
   ``chat_template`` string we plan to write into ``tokenizer_config.json``)
   must produce the exact same prompt bytes as Flatbuild's
   ``ChatTemplate.render`` for the same message list.
2. Tokenizer ID-identity: Flatbuild's HuggingFace ``tokenizers`` BPE and
   Flatrun's pure-Python BPE must produce identical token IDs for the
   rendered prompt strings (so the model sees the same context).

Run from the repo root:

    python scripts/verify_flatrun_interop.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, "/Users/judotens/Works/Codes/tenslab/flatseek/flatrun/src")

from flatbuild.config import ChatTemplateConfig  # noqa: E402
from flatbuild.tokenizers.bpe import BPETokenizer  # noqa: E402
from flatbuild.tokenizers.template import (  # noqa: E402
    build_chat_template,
    to_flatrun_jinja,
)

from flatrun.tokenizer.bpe import BPETokenizer as FlatrunBPE  # noqa: E402

# --- chat template config we plan to use for the demo ---------------------
# The ``\n\n`` turn separator is folded into the prefixes so the stream is
# byte-identical to ``separator="\n\n"`` while staying renderable by
# flatrun's restricted Jinja (no loop vars / no literal text between tags).
CFG = ChatTemplateConfig(
    name="flatbuild-demo",
    system=(
        "You are Flatbot, a friendly conversational assistant. "
        "You help users understand things, solve problems, and have natural conversations. "
        "Be clear, helpful, and concise."
    ),
    user_prefix="\n\n<|user|>\n",
    assistant_prefix="\n\n<|assistant|>\n",
    end_of_turn="<|endoftext|>",
    separator="",
)


def messages_fixtures() -> list[tuple[str, str]]:
    return [
        ("system", CFG.system),
        ("user", "Hello"),
        ("assistant", "Hi there! What can I help you with today?"),
        ("user", "What can you do?"),
    ]


def main() -> int:
    fb_tmpl = build_chat_template(CFG)
    JINJA = to_flatrun_jinja(fb_tmpl)
    fr_tok = FlatrunBPE(vocab={}, merges=[], chat_template=JINJA)

    print(f"Generated Jinja template ({len(JINJA)} chars):")
    print(f"  {JINJA!r}")
    print()

    print("=== Chat-template byte-identity ===")
    ok = True
    cases = [
        ("single user, gen prompt",
         [("user", "Hello")], True),
        ("system + user, gen prompt (flatrun run default)",
         [("system", CFG.system), ("user", "What is machine learning?")], True),
        ("full multi-turn, gen prompt (flatrun chat)",
         messages_fixtures(), True),
        ("full multi-turn, no gen prompt (training continuation)",
         messages_fixtures(), False),
        ("chat-mode empty-assistant placeholder",
         messages_fixtures() + [("assistant", "")], True),
    ]
    for label, msgs, gen in cases:
        if label == "chat-mode empty-assistant placeholder":
            # flatrun chat appends an empty assistant placeholder before
            # add_generation_prompt; the template must skip it so the
            # rendered prompt equals flatbuild's render of the *same
            # conversation* (no placeholder) with generation prompt.
            fr_msgs = [{"role": r, "content": c} for r, c in msgs]
            fr_text = fr_tok.apply_chat_template(fr_msgs, add_generation_prompt=gen)
            real_msgs = [m for m in msgs if not (m[0] == "assistant" and m[1] == "")]
            fb_text = fb_tmpl.render(real_msgs, add_generation_prompt=True)
        else:
            fb_text = fb_tmpl.render(msgs, add_generation_prompt=gen)
            fr_msgs = [{"role": r, "content": c} for r, c in msgs]
            fr_text = fr_tok.apply_chat_template(fr_msgs, add_generation_prompt=gen)
        match = fb_text == fr_text
        ok = ok and match
        print(f"[{'OK ' if match else 'FAIL'}] {label}")
        if not match:
            print(f"  flatbuild: {fb_text!r}")
            print(f"  flatrun  : {fr_text!r}")

    print()
    print("=== Tokenizer ID-identity ===")
    corpus = [
        "hello world how are you today",
        "machine learning is the study of data",
        "The quick brown fox jumps over the lazy dog!",
        "Paris is the capital of France.",
        "def square(n):\n    return n * n",
        "What is the difference between Python and JavaScript?",
        "plan a weekend trip to the mountains",
    ]
    corpus += [fb_tmpl.render(messages_fixtures())]
    fb = BPETokenizer.train(corpus, vocab_size=512, min_frequency=1)
    fb.save(ROOT / "logs" / "interop_tok")

    from flatrun.tokenizer import auto_load
    fr2 = auto_load(ROOT / "logs" / "interop_tok")
    total = 0
    same = 0
    for text in corpus:
        a = fb.encode(text)
        b = fr2.encode(text)
        total += 1
        if a == b:
            same += 1
        else:
            print(f"[DIFF] {text!r}")
            print(f"  flatbuild: {a}")
            print(f"  flatrun  : {b}")
    print(f"  identical tokenization for {same}/{total} strings")
    print(f"  flatbuild decode: {fb.decode(fb.encode('hello world'))!r}")
    print(f"  flatrun   decode: {fr2.decode(fr2.encode('hello world'))!r}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
