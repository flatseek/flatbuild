"""Generate ``data/demo_chat/dataset.jsonl`` — the flagship Flatbot demo dataset.

A conversational dataset for a general-purpose "Flatbot" assistant:

* 2,000+ conversations (target range 1,000–5,000).
* At least 50% multi-turn — the model must learn to keep context.
* Mixed topics: facts, geography, science, math, code, translation,
  summarization, rewrites, tips, books, haiku, jokes, refusals,
  small-talk, identity, and follow-up/context-switch chains.
* Every conversation opens with the canonical Flatbot system prompt so
  training bytes match the flatrun chat template exactly.

The topic pools and single/multi-turn sample builders are reused from
``scripts/generate_demo_large.py``; this script only swaps in the
``demo_chat`` system prompt and enforces the multi-turn ratio.

Run::

    python scripts/generate_demo_chat.py --out data/demo_chat/dataset.jsonl --n 2500
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
_LARGE = _SCRIPTS / "generate_demo_large.py"


def _load_large_module():
    """Import ``generate_demo_large`` as ``gdl`` for its pools/builders."""
    spec = importlib.util.spec_from_file_location("gdl", _LARGE)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


# Canonical Flatbot system prompt (byte-for-byte identical to
# ``configs/demo_chat.yaml`` chat_template.system).
SYSTEM = (
    "You are Flatbot, a friendly conversational assistant. "
    "You help users understand things, solve problems, and have natural conversations. "
    "Be clear, helpful, and concise."
)


def make_acknowledge_sample(rng, gdl):
    """Two turns: answer, then a polite closing (thanks / bye)."""
    q, a = rng.choice(gdl.SCIENCE_FACTS + gdl.GEOGRAPHY_FACTS + gdl.HISTORY_FACTS)
    closers_u = [
        "Thanks, that's really helpful.",
        "Got it, thank you!",
        "Thanks a lot.",
        "That makes sense. Thanks!",
        "Awesome, thanks for explaining.",
        "Perfect, thanks!",
        "Okay, got it. Thanks!",
        "Thanks! That was useful.",
    ]
    closers_a = [
        "You're welcome! Happy to help.",
        "Anytime. Glad it helped.",
        "You're welcome. Let me know if you have more questions.",
        "No problem at all. Anything else?",
        "Glad to help. Have a great day!",
        "Happy to help anytime.",
        "You're welcome! Feel free to ask more.",
        "Sure thing. Take care!",
    ]
    return {
        "messages": gdl._conv(
            ("system", SYSTEM),
            ("user", gdl._wrap_user(rng, q)),
            ("assistant", gdl._wrap_assistant(rng, a)),
            ("user", gdl._wrap_user(rng, rng.choice(closers_u))),
            ("assistant", rng.choice(closers_a)),
        ),
        "metadata": {"generator": "acknowledge"},
    }


def make_two_question_sample(rng, gdl):
    """Two turns: two related factual questions in a row."""
    first = rng.choice(gdl.SCIENCE_FACTS + gdl.GEOGRAPHY_FACTS + gdl.HISTORY_FACTS)
    second = rng.choice(gdl.SCIENCE_FACTS + gdl.GEOGRAPHY_FACTS + gdl.HISTORY_FACTS)
    return {
        "messages": gdl._conv(
            ("system", SYSTEM),
            ("user", gdl._wrap_user(rng, first[0])),
            ("assistant", gdl._wrap_assistant(rng, first[1])),
            ("user", gdl._wrap_user(rng, second[0])),
            ("assistant", gdl._wrap_assistant(rng, second[1])),
        ),
        "metadata": {"generator": "two_question"},
    }


def make_greeting_chain_sample(rng, gdl):
    """Greeting -> question -> answer -> close."""
    q, a = rng.choice(
        gdl.SCIENCE_FACTS + gdl.GEOGRAPHY_FACTS + gdl.HISTORY_FACTS
        + [("What can you do?", gdl.IDENTITY_ANSWERS[0])]
    )
    greetings = [
        ("Hi", "Hello! What can I help you with today?"),
        ("Hello", "Hey there. What's on your mind?"),
        ("Hey", "Hey! How can I help?"),
        ("Good morning", "Good morning! What can I do for you?"),
        ("Hi there", "Hi! I'm happy to help."),
    ]
    g, ga = rng.choice(greetings)
    return {
        "messages": gdl._conv(
            ("system", SYSTEM),
            ("user", gdl._wrap_user(rng, g)),
            ("assistant", gdl._wrap_assistant(rng, ga)),
            ("user", gdl._wrap_user(rng, q)),
            ("assistant", gdl._wrap_assistant(rng, a)),
        ),
        "metadata": {"generator": "greeting_chain"},
    }


# Generators that produce >= 2 assistant turns (multi-turn).
MULTI_TURN = [
    ("followup", lambda rng, gdl: gdl.make_followup_sample(rng)),
    ("context_switch", lambda rng, gdl: gdl.make_context_switch_sample(rng)),
    ("three_turn", lambda rng, gdl: gdl.make_three_turn_sample(rng)),
    ("acknowledge", make_acknowledge_sample),
    ("two_question", make_two_question_sample),
    ("greeting_chain", make_greeting_chain_sample),
]

# Single-turn generators (name -> weight).
SINGLE_TURN = [
    ("greeting", "make_greeting_sample", 6),
    ("identity", "make_identity_sample", 6),
    ("capital", "make_capital_sample", 10),
    ("animal_fact", "make_animal_fact_sample", 5),
    ("fact", "make_fact_sample", 15),
    ("keyword_explain", "make_keyword_sample", 4),
    ("translate", "make_translate_sample", 6),
    ("translate_compound", "make_compound_translate_sample", 3),
    ("summarize", "make_summarize_sample", 3),
    ("rewrite_polite", "make_politer_rewrite_sample", 3),
    ("sentiment", "make_sentiment_sample", 4),
    ("tips", "make_tips_sample", 4),
    ("code", "make_code_sample", 6),
    ("math", "make_math_sample", 7),
    ("next_day", "make_next_day_sample", 2),
    ("month_days", "make_month_days_sample", 2),
    ("counting", "make_counting_sample", 3),
    ("book", "make_book_recommendation_sample", 2),
    ("haiku", "make_haiku_sample", 2),
    ("joke", "make_joke_sample", 2),
    ("refusal", "make_refusal_sample", 2),
]


def _weighted_choice(rng, entries):
    total = sum(w for _, _, w in entries)
    pick = rng.uniform(0, total)
    cum = 0
    for entry in entries:
        cum += entry[2]
        if pick <= cum:
            return entry
    return entries[-1]


def generate(n: int, seed: int = 42, multi_ratio: float = 0.55) -> list[dict]:
    """Return ``n`` conversation samples with at least ``multi_ratio`` multi-turn."""
    gdl = _load_large_module()
    gdl.SYSTEM = SYSTEM
    rng = random.Random(seed)

    n_multi = int(round(n * multi_ratio))
    n_single = n - n_multi

    samples: list[dict] = []
    for _ in range(n_multi):
        name, maker = rng.choice(MULTI_TURN)
        samples.append(maker(rng, gdl))
    for _ in range(n_single):
        name, fn_name, _w = _weighted_choice(rng, SINGLE_TURN)
        samples.append(getattr(gdl, fn_name)(rng))

    rng.shuffle(samples)
    for i, s in enumerate(samples):
        s.setdefault("metadata", {})["sample_index"] = i
    return samples


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/demo_chat/dataset.jsonl"))
    parser.add_argument("--n", type=int, default=2500, help="Number of conversations.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--multi-ratio", type=float, default=0.55)
    args = parser.parse_args(argv)

    samples = generate(args.n, args.seed, args.multi_ratio)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    counts: dict[str, int] = {}
    for s in samples:
        counts[s["metadata"]["generator"]] = counts.get(s["metadata"]["generator"], 0) + 1
    n_multi = sum(1 for s in samples if len(s["messages"]) >= 5)
    print(f"Wrote {len(samples):,} samples to {args.out}")
    print(f"Multi-turn (>=5 messages): {n_multi:,} ({n_multi / len(samples):.0%})")
    print("Generator breakdown (top 12):")
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {k:>18}: {v:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
