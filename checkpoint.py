"""Strict, audited loading for the five historical checkpoint schemas."""

from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn


_DYNAMIC_PARAMETERS = {
    "points",
    "points_influ_scores",
    "points_scaler",
    "points_last_grad",
    "points_acc_grad",
    "points_acc_grad_norm",
    "points_grad_cnt",
    "points_density",
    "pc_feats",
    "points_alpha",
    "bkg_points_pc_feats",
    "bkg_points_influ_scores",
    "bkg_points",
}
_DERIVABLE_MISSING = {
    "points_scaler",
    "points_density",
}


@dataclass(frozen=True)
class LoadReport:
    path: str
    sha256: str
    source_schema: str
    step: int
    migrations: tuple[str, ...]
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]
    shape_mismatch: tuple[str, ...]
    unequal: tuple[str, ...]

    @property
    def exact(self) -> bool:
        return not (self.missing or self.unexpected or self.shape_mismatch or self.unequal)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self) | {"exact": self.exact}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_checkpoint(
    path: str | Path, *, expected_step: int | None = None
) -> tuple[dict[str, torch.Tensor], int, str]:
    """Unwrap both legacy and release checkpoint containers."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, Mapping) and payload.get("schema") in {"pcore-1", "pcore-2"}:
        step = int(payload["step"])
        if expected_step is not None and step != expected_step:
            raise ValueError(f"Checkpoint step is {step}, expected {expected_step}")
        return dict(payload["model"]), step, str(payload["schema"])
    if isinstance(payload, Mapping) and payload and all(torch.is_tensor(v) for v in payload.values()):
        step = expected_step
        if step is None:
            stem = Path(path).stem
            suffix = stem.rsplit("_", 1)[-1]
            if not suffix.isdigit():
                raise ValueError("A bare legacy checkpoint requires expected_step")
            step = int(suffix)
        return dict(payload), int(step), "legacy-bare"
    if isinstance(payload, Mapping) and len(payload) == 1:
        label, state = next(iter(payload.items()))
        if str(label).isdigit() and isinstance(state, Mapping):
            step = int(label)
            if expected_step is not None and step != expected_step:
                raise ValueError(f"Checkpoint step is {step}, expected {expected_step}")
            return dict(state), step, "legacy-wrapped"
    raise ValueError(f"Unsupported checkpoint container: {path}")


def _materialize_dynamic_parameters(model: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    current = dict(model.named_parameters())
    for name in _DYNAMIC_PARAMETERS.intersection(state):
        if "." in name or not hasattr(model, name):
            continue
        old = current.get(name)
        requires_grad = old.requires_grad if old is not None else False
        tensor = state[name].detach().to(model.device)
        setattr(model, name, nn.Parameter(tensor.clone(), requires_grad=requires_grad))

    count = state["points"].shape[0]
    if "points_scaler" not in state and hasattr(model, "points_scaler"):
        old = model.points_scaler
        value = torch.ones((count, 1), device=model.device, dtype=model.points.dtype)
        model.points_scaler = nn.Parameter(value, requires_grad=old.requires_grad)
    if "points_density" not in state and hasattr(model, "points_density"):
        value = torch.ones(count, device=model.device, dtype=model.points.dtype)
        model.points_density = nn.Parameter(value, requires_grad=False)


def load_checkpoint_strict(
    model: nn.Module,
    path: str | Path,
    *,
    expected_step: int | None = None,
    expected_sha256: str | None = None,
) -> LoadReport:
    """Load a legacy checkpoint, allowing only named, lossless migrations."""
    path = Path(path)
    actual_sha = sha256_file(path)
    if expected_sha256 and actual_sha != expected_sha256:
        raise ValueError(f"SHA256 mismatch for {path}: {actual_sha} != {expected_sha256}")
    source, step, schema = read_checkpoint(path, expected_step=expected_step)
    state = {name: tensor.detach().clone() for name, tensor in source.items()}
    migrations: list[str] = []

    key = "append_bkg_points_embedv"
    current = model.state_dict()
    if key in state and key in current and state[key].ndim == 1 and current[key].ndim == 2:
        state[key] = state[key].unsqueeze(0)
        migrations.append("legacy_flat_append_embedv:unsqueeze(0)")

    _materialize_dynamic_parameters(model, state)
    current = model.state_dict()
    shape_mismatch = tuple(
        sorted(name for name in state if name in current and current[name].shape != state[name].shape)
    )
    if shape_mismatch:
        raise RuntimeError(f"Checkpoint shape mismatches: {shape_mismatch}")

    incompatible = model.load_state_dict(state, strict=False)
    missing = tuple(sorted(set(incompatible.missing_keys) - _DERIVABLE_MISSING))
    unexpected = tuple(sorted(incompatible.unexpected_keys))
    loaded = model.state_dict()
    unequal = tuple(
        sorted(
            name
            for name, tensor in state.items()
            if name in loaded
            and not torch.allclose(
                loaded[name].detach().cpu(),
                tensor.detach().cpu(),
                rtol=0,
                atol=0,
                equal_nan=True,
            )
        )
    )
    report = LoadReport(
        path=str(path.resolve()),
        sha256=actual_sha,
        source_schema=schema,
        step=step,
        migrations=tuple(migrations),
        missing=missing,
        unexpected=unexpected,
        shape_mismatch=shape_mismatch,
        unequal=unequal,
    )
    if not report.exact:
        raise RuntimeError(f"Non-exact checkpoint load: {report.as_dict()}")
    return report


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    step: int,
    phase: str,
    config_id: str,
    config_sha256: str,
    config_fingerprint: str,
    sampler,
    device: str | torch.device,
) -> Path:
    """Write a portable, fully resumable ``pcore-2`` checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "pcore-2",
            "step": int(step),
            "phase": phase,
            "config": {
                "id": config_id,
                "sha256": config_sha256,
                "effective_sha256": config_fingerprint,
            },
            "model": model.state_dict(),
            "training": _capture_training_state(model, sampler=sampler, device=device),
        },
        path,
    )
    return path


def _capture_rng_state(device: str | torch.device) -> dict[str, Any]:
    """Capture only weights-only-safe values for exact stochastic continuation."""
    bit_generator, state, position, has_gauss, cached_gaussian = np.random.get_state()
    device = torch.device(device)
    cuda_state = None
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Cannot capture CUDA RNG state without CUDA")
        cuda_state = torch.cuda.get_rng_state(device).cpu()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": bit_generator,
            "state": torch.from_numpy(state.copy()),
            "position": int(position),
            "has_gauss": int(has_gauss),
            "cached_gaussian": float(cached_gaussian),
        },
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": cuda_state,
    }


def _restore_rng_state(state: Mapping[str, Any], device: str | torch.device) -> None:
    python_state = state.get("python")
    numpy_state = state.get("numpy")
    torch_cpu = state.get("torch_cpu")
    if python_state is None or not isinstance(numpy_state, Mapping):
        raise ValueError("Checkpoint RNG state is incomplete")
    numpy_values = numpy_state.get("state")
    if not torch.is_tensor(numpy_values) or not torch.is_tensor(torch_cpu):
        raise ValueError("Checkpoint NumPy and torch RNG states must be tensors")

    random.setstate(python_state)
    np.random.set_state(
        (
            str(numpy_state["bit_generator"]),
            numpy_values.detach().cpu().numpy().astype(np.uint32, copy=True),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(torch_cpu.detach().cpu())

    device = torch.device(device)
    cuda_state = state.get("torch_cuda")
    if device.type == "cuda":
        if not torch.is_tensor(cuda_state):
            raise ValueError("Checkpoint has no CUDA RNG state for a CUDA resume")
        torch.cuda.set_rng_state(cuda_state.detach().cpu(), device=device)


def _capture_training_state(model: nn.Module, *, sampler, device) -> dict[str, Any]:
    optimizers = getattr(model, "optimizers", None)
    schedulers = getattr(model, "schedulers", None)
    scaler = getattr(model, "scaler", None)
    if not isinstance(optimizers, dict) or not isinstance(schedulers, dict) or scaler is None:
        raise TypeError("Model does not expose optimizers, schedulers, and GradScaler state")
    if not hasattr(sampler, "state_dict"):
        raise TypeError("Training sampler does not expose state_dict()")
    runtime = {
        "pruned_points": bool(getattr(model, "pruned_points", False)),
        "added_points": bool(getattr(model, "added_points", False)),
        "select_k": getattr(model, "select_k", None),
        "pixel_width": getattr(model, "pixel_width", None),
        "pixel_height": getattr(model, "pixel_height", None),
    }
    for name in ("pixel_width", "pixel_height"):
        if torch.is_tensor(runtime[name]):
            runtime[name] = runtime[name].detach().cpu().clone()
    return {
        "optimizers": {
            name: optimizer.state_dict() if optimizer is not None else None
            for name, optimizer in optimizers.items()
        },
        "schedulers": {
            name: scheduler.state_dict() if scheduler is not None else None
            for name, scheduler in schedulers.items()
        },
        "scaler": scaler.state_dict(),
        "rng": _capture_rng_state(device),
        "sampler": sampler.state_dict(),
        "model_runtime": runtime,
    }


def restore_training_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    sampler,
    expected_phase: str,
    expected_config_id: str,
    expected_config_fingerprint: str,
    device: str | torch.device,
) -> int:
    """Restore all non-model state from a strict ``pcore-2`` checkpoint."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or payload.get("schema") != "pcore-2":
        schema = payload.get("schema") if isinstance(payload, Mapping) else None
        raise ValueError(f"--resume requires a pcore-2 checkpoint, got {schema!r}")
    if payload.get("phase") != expected_phase:
        raise ValueError(
            f"Resume phase is {payload.get('phase')!r}, expected {expected_phase!r}"
        )
    config = payload.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("Resume checkpoint has no portable config metadata")
    if config.get("id") != expected_config_id:
        raise ValueError(
            f"Resume config is {config.get('id')!r}, expected {expected_config_id!r}"
        )
    if config.get("effective_sha256") != expected_config_fingerprint:
        raise ValueError(
            "Resume effective training configuration does not match the current "
            "defaults, selected phase, scene YAML, and CLI overrides: "
            f"{config.get('effective_sha256')} != {expected_config_fingerprint}"
        )

    training = payload.get("training")
    if not isinstance(training, Mapping):
        raise ValueError("Resume checkpoint has no training state")
    saved_optimizers = training.get("optimizers")
    saved_schedulers = training.get("schedulers")
    if not isinstance(saved_optimizers, Mapping) or not isinstance(saved_schedulers, Mapping):
        raise ValueError("Resume checkpoint has invalid optimizer or scheduler state")
    optimizers = getattr(model, "optimizers", None)
    schedulers = getattr(model, "schedulers", None)
    if not isinstance(optimizers, dict) or not isinstance(schedulers, dict):
        raise TypeError("Model does not expose optimizer and scheduler dictionaries")
    if set(saved_optimizers) != set(optimizers):
        raise ValueError(
            "Resume optimizer groups do not match: "
            f"saved={sorted(saved_optimizers)}, current={sorted(optimizers)}"
        )
    if set(saved_schedulers) != set(schedulers):
        raise ValueError(
            "Resume scheduler groups do not match: "
            f"saved={sorted(saved_schedulers)}, current={sorted(schedulers)}"
        )
    for name, optimizer in optimizers.items():
        saved = saved_optimizers[name]
        if optimizer is None:
            if saved is not None:
                raise ValueError(f"Optimizer {name!r} is disabled but has saved state")
        else:
            if saved is None:
                raise ValueError(f"Optimizer {name!r} has no saved state")
            optimizer.load_state_dict(saved)
    for name, scheduler in schedulers.items():
        saved = saved_schedulers[name]
        if scheduler is None:
            if saved is not None:
                raise ValueError(f"Scheduler {name!r} is disabled but has saved state")
        else:
            if saved is None:
                raise ValueError(f"Scheduler {name!r} has no saved state")
            scheduler.load_state_dict(saved)

    scaler = getattr(model, "scaler", None)
    scaler_state = training.get("scaler")
    if scaler is None or not isinstance(scaler_state, Mapping):
        raise ValueError("Resume checkpoint has invalid GradScaler state")
    scaler.load_state_dict(dict(scaler_state))

    sampler_state = training.get("sampler")
    if not isinstance(sampler_state, Mapping) or not hasattr(sampler, "load_state_dict"):
        raise ValueError("Resume checkpoint has invalid sampler state")
    sampler.load_state_dict(sampler_state)

    runtime = training.get("model_runtime")
    if not isinstance(runtime, Mapping):
        raise ValueError("Resume checkpoint has invalid model runtime state")
    model.pruned_points = bool(runtime.get("pruned_points", False))
    model.added_points = bool(runtime.get("added_points", False))
    if runtime.get("select_k") is not None:
        model.select_k = int(runtime["select_k"])
    for name in ("pixel_width", "pixel_height"):
        value = runtime.get(name)
        if value is not None and not torch.is_tensor(value):
            raise ValueError(f"Resume model runtime {name} must be a tensor or None")
        setattr(model, name, value.to(device) if torch.is_tensor(value) else None)

    rng = training.get("rng")
    if not isinstance(rng, Mapping):
        raise ValueError("Resume checkpoint has invalid RNG state")
    # Restore RNG last: loading model/optimizer/sampler state must not perturb
    # the exact next stochastic operation in the resumed run.
    _restore_rng_state(rng, device)
    return int(payload["step"])


def freeze_loaded_geometry(model: nn.Module, *, reset_accumulators: bool) -> None:
    """Make the historical phase-2 optimizer behavior explicit and reproducible."""
    frozen = ("points", "points_influ_scores", "points_scaler", "pc_feats")
    for name in frozen:
        parameter = getattr(model, name, None)
        if isinstance(parameter, nn.Parameter):
            parameter.requires_grad_(False)
            parameter.grad = None
    # Dynamic checkpoint tensors replace their initialization-time Parameters.
    # Removing the now-stale optimizer groups makes the historical no-update
    # behavior explicit and avoids advancing schedulers that can never step.
    for collection_name in ("optimizers", "schedulers"):
        collection = getattr(model, collection_name, None)
        if isinstance(collection, dict):
            for name in frozen:
                collection.pop(name, None)
    if reset_accumulators:
        for name in (
            "points_last_grad",
            "points_acc_grad",
            "points_acc_grad_norm",
            "points_grad_cnt",
        ):
            value = getattr(model, name, None)
            if torch.is_tensor(value):
                value.data.zero_()
