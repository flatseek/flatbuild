"""Chat-template rendering for Flatbuild.

The chat template converts a multi-turn message list (system / user /
assistant) into a single string the language model is trained on. By
default it uses a ChatML-style scheme with explicit user/assistant
prefixes, but every prefix and separator is configurable so the same
trained model can speak several dialects (``ChatML``, ``Alpaca``,
``Llama-3``, ``Qwen``, etc.) without retraining.

Canonical rendering contract (single source of truth)
-------------------------------------------------------

:meth:`ChatTemplate.render_message` produces the bytes for a single
role. :meth:`ChatTemplate.render` is just a glue method that
concatenates ``render_message(role)`` outputs with ``separator``
between them. :func:`tokenize_sample` (in the trainer) consumes the
same :meth:`render_message` outputs directly. Training and inference
produce byte-for-byte identical strings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from flatbuild.config import ChatTemplateConfig
from flatbuild.datasets.base import ConversationSample, Sample


@dataclass(frozen=True)
class ChatTemplate:
    """Concrete chat template with explicit pieces."""

    system: str | None
    user_prefix: str
    assistant_prefix: str
    end_of_turn: str
    separator: str
    name: str

    # ------------------------------------------------------------------
    # Canonical per-role renderer — single source of truth.
    # ------------------------------------------------------------------

    def render_message(self, role: str, content: str) -> str:
        """Return the text fragment for one role *without* separators.

        Args:
            role: One of ``"system"``, ``"user"``, ``"assistant"``,
                or any other role (rendered generically).
            content: Raw role content.

        Returns:
            The role's text contribution. Does not include any
            inter-role separators — those are inserted by :meth:`render`.
            Encodes the canonical convention that:

            - ``system`` content is followed by ``end_of_turn``.
            - ``user`` content follows ``user_prefix`` only.
            - ``assistant`` content is wrapped by ``assistant_prefix``
              *and* ``end_of_turn``.
        """
        if role == "system":
            return f"{content}{self.end_of_turn}"
        if role == "user":
            return f"{self.user_prefix}{content}"
        if role == "assistant":
            return f"{self.assistant_prefix}{content}{self.end_of_turn}"
        # Unknown roles get a generic prefix so we never silently drop.
        return f"[{role}] {content}{self.end_of_turn}"

    # ------------------------------------------------------------------
    # Multi-turn renderer.
    # ------------------------------------------------------------------

    def render(
        self,
        messages: Sequence[tuple[str, str]],
        *,
        add_generation_prompt: bool = False,
    ) -> str:
        """Render a conversation (list of (role, content)) into a string.

        Args:
            messages: Sequence of ``(role, content)`` tuples.
            add_generation_prompt: When ``True``, append the assistant
                prefix at the end (useful for inference).

        Returns:
            The rendered prompt string. Identical bytes to what
            ``tokenize_sample`` would tokenize role-by-role.
        """
        parts: list[str] = []
        for i, (role, content) in enumerate(messages):
            if i > 0:
                parts.append(self.separator)
            parts.append(self.render_message(role, content))
        if add_generation_prompt:
            # Multi-turn rendering already inserts separator between
            # every pair of messages. After the LAST message we still
            # need a separator + assistant_prefix — same as what
            # ``tokenize_sample`` encodes as the user→assistant
            # boundary in training.
            parts.append(self.separator)
            parts.append(self.assistant_prefix)
        return "".join(parts)

    # ------------------------------------------------------------------
    # Sample helpers.
    # ------------------------------------------------------------------

    def render_sample(self, sample: Sample) -> str:
        """Render a normalized :class:`Sample` to a training string.

        Args:
            sample: Pretraining / instruction / conversation sample.

        Returns:
            A single string suitable for tokenization.
        """
        from flatbuild.datasets.base import InstructionSample, PretrainingSample

        if isinstance(sample, ConversationSample):
            return self.render(sample.messages)
        if isinstance(sample, InstructionSample):
            messages: list[tuple[str, str]] = []
            if self.system:
                messages.append(("system", self.system))
            user_turn = sample.instruction
            if sample.input:
                user_turn = f"{user_turn}\n\n{sample.input}"
            messages.append(("user", user_turn))
            messages.append(("assistant", sample.output))
            return self.render(messages)
        if isinstance(sample, PretrainingSample):
            return sample.text
        raise TypeError(f"Cannot render sample of type {type(sample).__name__}")


def build_chat_template(cfg: ChatTemplateConfig) -> ChatTemplate:
    """Build a :class:`ChatTemplate` from a :class:`ChatTemplateConfig`.

    Args:
        cfg: Chat-template configuration.

    Returns:
        A new :class:`ChatTemplate`.
    """
    return ChatTemplate(
        system=cfg.system,
        user_prefix=cfg.user_prefix,
        assistant_prefix=cfg.assistant_prefix,
        end_of_turn=cfg.end_of_turn,
        separator=cfg.separator,
        name=cfg.name,
    )


def render_conversation(
    sample: Sample,
    template: ChatTemplate,
    *,
    add_generation_prompt: bool = False,
) -> str:
    """Convenience wrapper around :meth:`ChatTemplate.render`."""
    if isinstance(sample, ConversationSample):
        return template.render(sample.messages, add_generation_prompt=add_generation_prompt)
    return template.render_sample(sample)


__all__ = [
    "ChatTemplate",
    "build_chat_template",
    "render_conversation",
]
