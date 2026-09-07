"""CLI for strict checkpoint audit and paper-protocol evaluation."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from checkpoint import load_checkpoint_strict, read_checkpoint
from config import load_config
from evaluation import evaluate
from models import PAPR
from utils import seed_everything


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a released P-CORE checkpoint")
    parser.add_argument("--opt", required=True, type=Path, help="Scene YAML")
    parser.add_argument(
        "--phase",
        choices=("reconstruction", "self-consistency"),
        default="reconstruction",
        help="Use self-consistency when evaluating a fine-tuned checkpoint",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        default=None,
        help="Required when a custom bare legacy checkpoint filename has no step suffix",
    )
    parser.add_argument("--points", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument(
        "--checkpoint-only",
        action="store_true",
        help="Validate exact state loading without rendering",
    )
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> None:
    cli = build_parser().parse_args(argv)
    resolved = load_config(
        cli.opt,
        phase=cli.phase.replace("-", "_"),
        overrides=cli.overrides,
        checkpoint=cli.checkpoint,
    )
    args = resolved.values
    seed_everything(args.seed)
    default_checkpoint = Path(args.release.checkpoint.path)
    checkpoint = Path(cli.checkpoint) if cli.checkpoint else default_checkpoint
    expected_sha = args.release.checkpoint.sha256 if checkpoint.resolve() == default_checkpoint.resolve() else None
    expected_step = int(args.release.checkpoint.step)

    # A user-supplied release checkpoint carries its own local phase step. Do
    # not incorrectly require the paper checkpoint's absolute reconstruction
    # step when evaluating a newly fine-tuned checkpoint.
    _, checkpoint_step, _ = read_checkpoint(
        checkpoint,
        expected_step=(
            cli.checkpoint_step
            if cli.checkpoint_step is not None
            else expected_step if expected_sha else None
        ),
    )

    if cli.checkpoint_only:
        init_args = copy.deepcopy(args)
        init_args.geoms.points.load_path = ""
        model = PAPR(init_args, device=torch.device(cli.device)).to(cli.device)
        report = load_checkpoint_strict(
            model,
            checkpoint,
            expected_step=checkpoint_step,
            expected_sha256=expected_sha,
        )
        print(json.dumps(report.as_dict(), indent=2))
        return

    points = cli.points or Path(args.release.evaluation_points)
    dataset = cli.dataset or Path(args.release.evaluation.dataset)
    result = evaluate(
        args,
        checkpoint=checkpoint,
        checkpoint_step=checkpoint_step,
        checkpoint_sha256=expected_sha,
        points=points,
        dataset_path=dataset,
        protocol=args.release.evaluation,
        output_dir=cli.output_dir,
        device=cli.device,
        max_frames=cli.max_frames,
        save_images=cli.save_images,
    )
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
