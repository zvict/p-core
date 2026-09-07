"""Configuration loading and phase resolution."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

from utils import ConfigNode, to_config_node


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings without mutating either input."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def _set_dotted(config: dict[str, Any], expression: str) -> None:
    if "=" not in expression:
        raise ValueError(f"Override must be KEY=VALUE, got {expression!r}")
    dotted, raw = expression.split("=", 1)
    keys = dotted.split(".")
    target = config
    for key in keys[:-1]:
        target = target.setdefault(key, {})
        if not isinstance(target, dict):
            raise ValueError(f"Cannot set {dotted!r}: {key!r} is not a mapping")
    target[keys[-1]] = yaml.safe_load(raw)


def _effective_training_fingerprint(
    config: dict[str, Any],
    *,
    phase: str,
    phase_config: dict[str, Any],
) -> str:
    """Hash the portable, merged configuration that can affect continuation.

    This is computed after defaults, scene values, phase overrides, derived
    settings, and CLI ``--set`` expressions have been applied, but before path
    resolution.  Initialization source, output location, and target run length
    do not affect continuation from a full-state checkpoint.
    """
    snapshot = copy.deepcopy(config)
    snapshot.pop("load_path", None)
    snapshot.pop("save_dir", None)
    snapshot["phases"] = {
        "selected": phase,
        "scheduler_total_steps": int(phase_config["scheduler_total_steps"]),
        "freeze_geometry": bool(phase_config.get("freeze_geometry", False)),
        "reset_accumulators": bool(phase_config.get("reset_accumulators", False)),
    }
    canonical = json.dumps(
        snapshot,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


@dataclass(frozen=True)
class ResolvedConfig:
    values: ConfigNode
    phase: str
    max_steps: int
    scheduler_total_steps: int
    checkpoint: Path | None
    expected_checkpoint_sha256: str | None
    expected_checkpoint_step: int | None
    freeze_geometry: bool
    reset_accumulators: bool
    effective_fingerprint: str
    source: Path


def load_config(
    path: str | Path,
    *,
    phase: str,
    overrides: Iterable[str] = (),
    max_steps: int | None = None,
    scheduler_total_steps: int | None = None,
    checkpoint: str | Path | None = None,
) -> ResolvedConfig:
    """Resolve archived model options and one of the two release phases."""
    if phase not in {"reconstruction", "self_consistency"}:
        raise ValueError(f"Unknown phase: {phase}")
    path = Path(path).resolve()
    defaults_path = Path(__file__).resolve().parent / "configs" / "default.yaml"
    defaults = _load_yaml(defaults_path)
    scene = _load_yaml(path)
    config = deep_merge(defaults, scene)
    phase_config = config["phases"][phase]
    if phase == "self_consistency":
        config = deep_merge(config, phase_config.get("overrides", {}))
        config["edit"]["train_with_augmentation"] = True
    else:
        config["edit"]["train_with_augmentation"] = False

    scheduler_total = int(
        scheduler_total_steps
        if scheduler_total_steps is not None
        else phase_config["scheduler_total_steps"]
    )
    if scheduler_total <= 0:
        raise ValueError("scheduler_total_steps must be positive")
    config["training"]["steps"] = scheduler_total
    config["dataset"]["bkg_sphere_radius"] = config["geoms"]["points"][
        "bkg_points_init_scale"
    ][0]
    for expression in overrides:
        _set_dotted(config, expression)

    effective_fingerprint = _effective_training_fingerprint(
        config,
        phase=phase,
        phase_config=phase_config,
    )

    # Config paths are plain repo-relative strings and are used exactly as written.
    expected_checkpoint_sha256: str | None = None
    expected_checkpoint_step: int | None = None

    chosen_checkpoint = checkpoint
    if chosen_checkpoint is None:
        chosen_checkpoint = phase_config.get("init_checkpoint")
        if chosen_checkpoint is not None:
            # A bare legacy checkpoint carries no step of its own, so the phase has
            # to supply it; without one, read_checkpoint cannot date the weights.
            expected_checkpoint_step = phase_config.get("init_checkpoint_step")
            expected_checkpoint_sha256 = phase_config.get("init_checkpoint_sha256")
        elif phase == "reconstruction":
            chosen_checkpoint = config.get("load_path") or None

    release_checkpoint = config.get("release", {}).get("checkpoint", {})
    if chosen_checkpoint is not None and str(chosen_checkpoint) == release_checkpoint.get("path"):
        expected_checkpoint_step = release_checkpoint.get("step")
        expected_checkpoint_sha256 = release_checkpoint.get("sha256")

    if expected_checkpoint_step is not None:
        expected_checkpoint_step = int(expected_checkpoint_step)
    config["load_path"] = str(chosen_checkpoint or "")

    return ResolvedConfig(
        values=to_config_node(config),
        phase=phase,
        max_steps=int(max_steps if max_steps is not None else phase_config["max_steps"]),
        scheduler_total_steps=scheduler_total,
        checkpoint=Path(chosen_checkpoint) if chosen_checkpoint else None,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_checkpoint_step=expected_checkpoint_step,
        freeze_geometry=bool(phase_config.get("freeze_geometry", False)),
        reset_accumulators=bool(phase_config.get("reset_accumulators", False)),
        effective_fingerprint=effective_fingerprint,
        source=path,
    )


def build_training_parser(
    description: str,
    *,
    default_steps: int | None = None,
) -> argparse.ArgumentParser:
    """Build the argument parser shared by ``train.py`` and ``finetune.py``.

    Each entry point hard-codes its phase, so the phase is not a flag; every
    other training option is identical between the two stages.  ``default_steps``
    lets a stage pin its own target step (self-consistency runs 500) while the
    reconstruction script falls through to the phase default in the config.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--opt", required=True, type=Path, help="Scene YAML")
    parser.add_argument("--output-dir", required=True, type=Path)
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Initialize model weights only; optimizer and RNG state start fresh",
    )
    initialization.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume model, optimizer, scheduler, scaler, RNG, and sampler state",
    )
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        default=None,
        metavar="STEP",
        help="Step of a bare legacy --checkpoint whose filename has no step suffix",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=default_steps,
        help="Total target step (not additional updates when resuming)",
    )
    parser.add_argument(
        "--scheduler-steps",
        type=int,
        default=None,
        metavar="N",
        help="Extend the cosine schedule horizon; required to train past the phase default",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=None,
        metavar="N",
        help="Also save a resumable checkpoint every N total steps",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Repeatable dotted YAML override",
    )
    return parser
