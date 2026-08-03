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


def _jinja_str(s: str) -> str:
    """Escape a string so it can live inside a single-quoted Python literal
    embedded in a Jinja ``{{ ... }}`` expression (as evaluated by Flatrun's
    restricted renderer)."""
    return (
        s.replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("'", "\\'")
    )


def to_flatrun_jinja(template: ChatTemplate) -> str:
    """Render ``template`` as a Jinja string Flatrun can execute.

    Flatrun's bundled template renderer is a deliberately small subset of
    Jinja: it supports ``for``/``if``/``else`` tags and ``{{ ... }}``
    expressions, but *no* loop variables (so ``loop.index`` is out) and it
    cannot scan past literal text when locating a matching ``endfor`` /
    ``endif``. This generator therefore emits a template that:

    - puts every byte of output inside ``{{ ... }}`` expressions (no
      literal text between tags), and
    - folds the inter-turn ``separator`` into the per-message fragment via
      ``messages.index(message)`` so ``\n\n``-separated templates stay
      byte-identical to :meth:`ChatTemplate.render`.

    The empty assistant placeholder that Flatrun's chat REPL appends before
    ``add_generation_prompt`` renders as nothing, so a turn prompt is
    byte-identical to the training stream.

    Args:
        template: The Flatbuild chat template to translate.

    Returns:
        A Jinja chat-template string that reproduces :meth:`render` output
        byte-for-byte when run through Flatrun's ``apply_chat_template``.
    """
    eot = _jinja_str(template.end_of_turn)
    user_p = _jinja_str(template.user_prefix)
    asst_p = _jinja_str(template.assistant_prefix)
    sep = _jinja_str(template.separator)

    # When a default system prompt is configured, prepend it as the
    # first turn whenever the caller omits a system message.  The
    # guard uses only constructs supported by Flatrun's restricted
    # Jinja subset (no loop variables, no set) and by full Jinja
    # (LM Studio, llama.cpp).
    injection = ""
    if template.system:
        sys_literal = _jinja_str(template.system + template.end_of_turn)
        injection = (
            "{% if messages and messages[0]['role'] != 'system' %}"
            f"{{{{ '{sys_literal}' }}}}"
            "{% endif %}"
        )

    # Per-message fragment: empty assistant -> '', system -> content+eot,
    # user -> prefix+content, assistant -> prefix+content+eot.
    frag_body = (
        f"message['content'] + '{eot}' if message['role']=='system' "
        f"else ('{user_p}' + message['content'] if message['role']=='user' "
        f"else '{asst_p}' + message['content'] + '{eot}')"
    )
    empty_check = (
        "('' if (message['role']=='assistant' and message['content']=='') "
        "else "
    )
    if template.separator:
        # ``messages.index(message)`` is position in the rendered list —
        # the best available stand-in for ``loop.index`` in Flatrun's
        # renderer. Duplicate (role, content) pairs resolve to the first
        # occurrence, which is the documented best-effort limitation.
        frag = (
            empty_check
            + f"('{sep}' if messages.index(message) > 0 else '') + ({frag_body}))"
        )
        gen_frag = f"('{sep}{asst_p}')"
    else:
        frag = empty_check + f"({frag_body}))"
        gen_frag = f"('{asst_p}')"

    return (
        injection
        + "{% for message in messages %}"
        f"{{{{ {frag} }}}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        f"{{{{ {gen_frag} }}}}"
        "{% endif %}"
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
    "to_flatrun_jinja",
]
