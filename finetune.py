"""CLI for the 500-step self-consistency fine-tuning stage.

Self-consistency starts from a reconstructed checkpoint, freezes the geometry,
and optimizes the augmented-view consistency objective for a few hundred steps.
It is the paper's contribution, so it gets its own entry point rather than a
flag on ``train.py``.
"""

from __future__ import annotations

import torch

from config import build_training_parser, load_config
from training import train
from utils import seed_everything

DEFAULT_STEPS = 500


def build_parser():
    return build_training_parser(
        "Fine-tune a reconstructed P-CORE model with the self-consistency objective",
        default_steps=DEFAULT_STEPS,
    )


def main(argv: list[str] | None = None) -> None:
    cli = build_parser().parse_args(argv)
    resolved = load_config(
        cli.opt,
        phase="self_consistency",
        overrides=cli.overrides,
        max_steps=cli.steps,
        scheduler_total_steps=cli.scheduler_steps,
        checkpoint=cli.checkpoint,
    )
    if resolved.max_steps <= 0:
        raise ValueError("--steps must be positive")
    if cli.save_every is not None and cli.save_every <= 0:
        raise ValueError("--save-every must be positive")
    if cli.checkpoint_step is not None and cli.checkpoint is None:
        raise ValueError("--checkpoint-step requires --checkpoint")
    seed_everything(resolved.values.seed)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    checkpoint = train(
        resolved,
        cli.output_dir,
        device=cli.device,
        resume=cli.resume,
        save_every=cli.save_every,
        checkpoint_step=cli.checkpoint_step,
    )
    print(f"Saved {checkpoint}")


if __name__ == "__main__":
    main()
