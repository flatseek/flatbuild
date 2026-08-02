"""Decoder-only Transformer model for Flatbuild."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn

from flatbuild.config import ModelConfig
from flatbuild.models.layers.block import DecoderBlock


@dataclass
class StateDictPaths:
    """Standard HF/Llama-compatible state-dict key prefixes used by Flatbuild."""

    embed: str = "model.embed_tokens.weight"
    layers_prefix: str = "model.layers"
    final_norm: str = "model.norm.weight"
    lm_head: str = "lm_head.weight"


class FlatbuildModel(nn.Module):
    """Decoder-only Transformer with GQA, RoPE, RMSNorm, SwiGLU.

    Parameter naming follows the standard HuggingFace / flatrun convention
    so the produced checkpoints are interchangeable with the rest of the
    Flat ecosystem (notably ``flatrun``):

    - ``model.embed_tokens.weight``
    - ``model.layers.{i}.self_attn.{q,k,v,o}_proj.weight``
    - ``model.layers.{i}.input_layernorm.weight``     (alias of norm_1)
    - ``model.layers.{i}.post_attention_layernorm.weight``  (alias of norm_2)
    - ``model.layers.{i}.mlp.{gate,up,down}_proj.weight``
    - ``model.norm.weight``
    - ``lm_head.weight``
    """

    def __init__(self, config: ModelConfig) -> None:
        """Initialize the model.

        Args:
            config: Architecture configuration.
        """
        super().__init__()
        self.config = config
        if config.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")

        head_dim = config.hidden_dim // config.n_heads
        if head_dim * config.n_heads != config.hidden_dim:
            raise ValueError("hidden_dim must be divisible by n_heads")

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.embed_dropout = nn.Dropout(config.embedding_dropout)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    hidden_dim=config.hidden_dim,
                    n_heads=config.n_heads,
                    n_kv_heads=config.n_kv_heads or max(1, config.n_heads // 2),
                    head_dim=head_dim,
                    ffn_dim=config.ffn_dim or int(8 * config.hidden_dim / 3),
                    rope_theta=config.rope_theta,
                    max_seq_len=config.context_length,
                    rope_scaling=config.rope_scaling,
                    norm=config.norm.value,
                    activation=config.activation.value,
                    dropout=config.attention_dropout,
                )
                for _ in range(config.n_layers)
            ]
        )
        # ``norm`` is final RMSNorm — keep attribute name ``norm`` so it
        # serializes as ``model.norm.weight``.
        from flatbuild.models.layers.rmsnorm import RMSNorm

        self.norm = RMSNorm(config.hidden_dim)

        # LM head — optionally tied to the embedding matrix.
        self.lm_head = nn.Linear(
            config.hidden_dim,
            config.vocab_size,
            bias=False,
        )
        if config.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # Initialize weights.
        self.apply(self._init_weights)
        # Rescale residual projection weights per GPT-2 / Llama convention.
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) and name.endswith("o_proj"):
                module.weight.data.mul_(1.0 / math.sqrt(2.0 * len(self.layers)))

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights with normal-distribution small init."""
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_kv: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> "ModelOutput":
        """Run the full forward pass.

        Args:
            input_ids: Integer tensor of shape ``(B, T)``.
            labels: Optional label tensor of shape ``(B, T)`` used to
                compute the cross-entropy loss when provided.
            attention_mask: Optional additive mask.
            past_kv: Optional per-layer KV cache.

        Returns:
            :class:`ModelOutput` containing ``logits`` (and optional
            ``loss``) plus the updated ``present_kv``.
        """
        B, T = input_ids.shape

        x = self.embed_tokens(input_ids)
        x = self.embed_dropout(x)

        present_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i, layer in enumerate(self.layers):
            layer_cache = past_kv[i] if past_kv is not None else None
            x, present = layer(x, past_kv=layer_cache, attention_mask=attention_mask)
            present_kv.append(present)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = self._compute_loss(logits, labels)

        return ModelOutput(logits=logits, loss=loss, present_kv=present_kv)

    # ------------------------------------------------------------------
    # Generation utilities
    # ------------------------------------------------------------------

    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Greedy / sampling autoregressive generation.

        Args:
            input_ids: Prompt token ids of shape ``(1, T_prompt)``.
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature. ``1.0`` = no rescaling.
            top_k: Top-K filtering. ``0`` disables.
            top_p: Nucleus filter. ``1.0`` disables.
            do_sample: ``True`` to sample; ``False`` for greedy.
            eos_token_id: When generated, stop early.

        Returns:
            Tensor of shape ``(1, T_prompt + T_new)``.
        """
        was_training = self.training
        self.eval()
        try:
            device = input_ids.device
            generated = input_ids
            past_kv: list[tuple[torch.Tensor, torch.Tensor]] | None = None
            for _ in range(max_new_tokens):
                if past_kv is None:
                    input_step = generated
                else:
                    input_step = generated[:, -1:]
                with torch.no_grad():
                    out = self.forward(
                        input_step,
                        past_kv=past_kv,
                    )
                logits = out.logits[:, -1, :]
                past_kv = out.present_kv
                if do_sample:
                    logits = logits / max(1e-6, temperature)
                    if top_k and top_k > 0:
                        values, _ = torch.topk(logits, top_k)
                        kth = values[:, -1:]
                        logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                        probs = torch.softmax(sorted_logits, dim=-1)
                        cum = torch.cumsum(probs, dim=-1)
                        remove = cum > top_p
                        # Always keep at least one token.
                        remove[..., 1:] = remove[..., :-1].clone()
                        remove[..., 0] = False
                        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                        logits = torch.zeros_like(logits).scatter(
                            -1, sorted_indices, sorted_logits
                        )
                    probs = torch.softmax(logits, dim=-1)
                    next_id = torch.multinomial(probs, num_samples=1)
                else:
                    next_id = torch.argmax(logits, dim=-1, keepdim=True)
                generated = torch.cat([generated, next_id], dim=1)
                if eos_token_id is not None and int(next_id[0, 0]) == eos_token_id:
                    break
            return generated
        finally:
            if was_training:
                self.train()

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy over shifted logits.

        Args:
            logits: ``(B, T, V)`` logits.
            labels: ``(B, T)`` target token ids (``-100`` to ignore).

        Returns:
            Scalar loss.
        """
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        return loss

    # ------------------------------------------------------------------
    # State dict helpers (Llama/HF naming)
    # ------------------------------------------------------------------

    def state_dict_llama(self) -> dict[str, torch.Tensor]:
        """Return state-dict with Llama/HF-style parameter names.

        Returns:
            Ordered mapping ``name -> torch.Tensor``.
        """
        out: dict[str, torch.Tensor] = {}
        # Embeddings (also acts as lm_head when tied).
        out["model.embed_tokens.weight"] = self.embed_tokens.weight.detach().clone()
        # Layers.
        for i, block in enumerate(self.layers):
            prefix = f"model.layers.{i}"
            attn = block.attn
            out[f"{prefix}.self_attn.q_proj.weight"] = attn.q_proj.weight.detach().clone()
            out[f"{prefix}.self_attn.k_proj.weight"] = attn.k_proj.weight.detach().clone()
            out[f"{prefix}.self_attn.v_proj.weight"] = attn.v_proj.weight.detach().clone()
            out[f"{prefix}.self_attn.o_proj.weight"] = attn.o_proj.weight.detach().clone()
            out[f"{prefix}.input_layernorm.weight"] = block.norm_1.weight.detach().clone()
            out[f"{prefix}.post_attention_layernorm.weight"] = block.norm_2.weight.detach().clone()
            mlp = block.mlp
            # swiglu/relu2/gelu variants all expose .up_proj + .down_proj.
            # swiglu also has .gate_proj.
            if hasattr(mlp, "gate_proj"):
                out[f"{prefix}.mlp.gate_proj.weight"] = mlp.gate_proj.weight.detach().clone()
            out[f"{prefix}.mlp.up_proj.weight"] = mlp.up_proj.weight.detach().clone()
            out[f"{prefix}.mlp.down_proj.weight"] = mlp.down_proj.weight.detach().clone()
        # Final norm.
        out["model.norm.weight"] = self.norm.weight.detach().clone()
        # LM head — when tied, point at the embedding tensor for safety.
        if self.config.tie_embeddings:
            out["lm_head.weight"] = self.embed_tokens.weight.detach().clone()
        else:
            out["lm_head.weight"] = self.lm_head.weight.detach().clone()
        return out

    def load_state_dict_llama(self, state_dict: dict[str, torch.Tensor], strict: bool = True):
        """Load Llama-style weights into this model.

        Args:
            state_dict: Llama-keyed weight dict.
            strict: When ``True`` (default), require exact key match.

        Returns:
            The result of :meth:`torch.nn.Module.load_state_dict`.
        """
        renamed: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            # Strip the ``model.`` prefix when matching internal layers.
            if key.startswith("model.layers."):
                # Keep the full key — block layer hooks accept this.
                renamed[key] = value
            else:
                renamed[key] = value
        # Final norm + lm_head → our ``norm`` and ``lm_head``.
        if "model.norm.weight" in renamed:
            renamed["norm.weight"] = renamed.pop("model.norm.weight")
        if "lm_head.weight" in renamed:
            renamed["lm_head.weight"] = renamed.pop("lm_head.weight")
        # Stitch the embed/lm_head tied tensor into both.
        embed = renamed.get("model.embed_tokens.weight")
        if embed is None:
            embed = renamed.get("embed_tokens.weight")
        if embed is not None and self.config.tie_embeddings:
            renamed["lm_head.weight"] = embed
        return self.load_state_dict(renamed, strict=strict)


@dataclass
class ModelOutput:
    """Output container returned by :meth:`FlatbuildModel.forward`."""

    logits: torch.Tensor
    loss: torch.Tensor | None
    present_kv: list[tuple[torch.Tensor, torch.Tensor]]

    def __len__(self) -> int:
        return len(self.present_kv)
