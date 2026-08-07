"""Default callbacks for training."""

from pathlib import Path

from flatbuild.callbacks.base import Callback, CallbackContext
from flatbuild.config import FlatBuildConfig
from flatbuild.models import FlatbuildModel


class GenerationCallback(Callback):
    """Generate sample text after each epoch for inspection."""

    def __init__(
        self,
        model: FlatbuildModel,
        tokenizer,
        prompts: list[str],
        max_new_tokens: int = 64,
        temperature: float = 0.8,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def on_epoch_end(self, ctx: CallbackContext) -> None:
        """Generate and print sample text."""
        import torch
        print("\n" + "=" * 60)
        print("Sample generations:")
        print("=" * 60)
        for prompt in self.prompts[:3]:
            input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
            input_tensor = torch.tensor([input_ids], dtype=torch.long)
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_tensor,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                )
            generated = self.tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)
            print(f"\n[Prompt]: {prompt}")
            print(f"[Output]: {generated[:200]}...")
        print("=" * 60 + "\n")


def build_callbacks(config: FlatBuildConfig, run_dir: Path) -> list[Callback]:
    """Default callback chain: gradient clipping + (optional) early stopping."""
    from flatbuild.callbacks import EarlyStoppingCallback, GradientClipCallback

    chain: list[Callback] = [
        GradientClipCallback(max_norm=config.trainer.max_grad_norm),
    ]
    if config.trainer.early_stopping.enabled:
        chain.append(
            EarlyStoppingCallback(
                patience=config.trainer.early_stopping.patience,
                min_delta=config.trainer.early_stopping.min_delta,
                monitor=config.trainer.early_stopping.monitor,
            )
        )
    return chain
