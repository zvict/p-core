"""Unified reconstruction and optional self-consistency training loop."""

from __future__ import annotations

import bisect
import json
import time
from pathlib import Path

import torch
from torch import autocast

from checkpoint import (
    freeze_loaded_geometry,
    load_checkpoint_strict,
    restore_training_checkpoint,
    save_checkpoint,
    sha256_file,
)
from config import ResolvedConfig
from dataset import RayImageDataset, build_loader
from models import PAPR, get_loss

from .self_consistency import compute_consistency_loss


def _attach_camera(model: PAPR, dataset: RayImageDataset) -> None:
    model.fx, model.fy = dataset.focal_x, dataset.focal_y
    model.cx, model.cy = dataset.cx, dataset.cy
    model.H, model.W = dataset.H, dataset.W


def _validate_resume_target(resume_step: int, target_step: int) -> None:
    if resume_step >= target_step:
        raise ValueError(
            f"--steps is the total target step and must exceed resume step {resume_step}; "
            f"got {target_step}"
        )


def _train_batch(step, model, dataset, batch, loss_fn, args):
    device = model.device
    target = batch["image"].to(device)
    mask = batch["mask"].to(device)
    rays_d = batch["rayd"].to(device)
    rays_d_raw = batch["rays_d_no_norm"].to(device)
    rays_o = batch["rayo"].to(device)
    pixel_coords = batch["pix_coords"].to(device)
    pixels = batch["pixels"].to(device)
    c2w = dataset.get_c2w(batch["idx"]).to(device)
    surface_points = batch.get("surface_points")
    if surface_points is not None:
        surface_points = surface_points.to(device)

    background = model.bkg_feats * model.bkg_scaler.squeeze(0)
    if args.rnd_background:
        denominator = args.rnd_background_stop - args.rnd_background_start
        shift = (
            (step - args.rnd_background_start) / denominator
            if denominator > 0 and step >= args.rnd_background_start
            else 0.0
        ) + args.rnd_background_shift
        random_background = (torch.rand(3, device=device) + shift) * model.bkg_scaler.squeeze(0)
        target = target.clone()
        target[mask.squeeze(-1) < 0.5] = random_background
        background = random_background[None]

    model.clear_grad()
    with autocast(device_type=device.type, dtype=model.amp_dtype, enabled=model.use_amp):
        (
            output,
            _,
            grid,
            regularizers,
            hit_prediction,
            attention,
            _,
            selected_indices,
            _,
        ) = model(
            rays_o,
            rays_d,
            c2w,
            pixel_coords,
            pixels,
            background,
            mask,
            step,
            surface_points=surface_points,
            rays_d_no_norm=rays_d_raw,
        )
        output = model.last_act(output)
        if grid is not None:
            target = torch.nn.functional.grid_sample(
                target.permute(0, 3, 1, 2), grid, align_corners=True
            ).permute(0, 2, 3, 1)
        rgb_loss = loss_fn(output, target)
        hit_loss = torch.zeros((), device=device)
        if hit_prediction is not None:
            hit_loss = -(
                mask.squeeze(-1) * torch.log(hit_prediction + 1e-8)
                + (1 - mask.squeeze(-1)) * torch.log(1 - hit_prediction + 1e-8)
            ).mean()
        total = rgb_loss + hit_loss * args.training.hit_loss_weight
        for name, (value, weight) in regularizers.items():
            if weight > 0:
                if args.training.get(name + "_scale_by_rgb_loss", False) and value.item() > rgb_loss.item():
                    value = value * (rgb_loss.item() / value.item())
                total = total + value * weight

        consistency = None
        if args.edit.train_with_augmentation:
            consistency = compute_consistency_loss(
                model,
                loss_fn,
                args,
                step=step,
                rays_o=rays_o,
                rays_d=rays_d,
                mask=mask,
                pixel_coords=pixel_coords,
                pixels=pixels,
                rays_d_raw=rays_d_raw,
                c2w=c2w,
                target_rgb=target,
                background=background,
                canonical_attention=attention,
                canonical_indices=selected_indices,
            )
            total = total + consistency.geometry * args.edit.aug_loss_weight
            total = total + consistency.texture * args.edit.aug_texture_loss_weight

    model.scaler.scale(total).backward()
    with torch.no_grad():
        threshold = args.training.prune_thresh
        if (
            args.training.prune_steps > 0
            and args.training.prune_start <= step < args.training.prune_stop
        ):
            if args.training.prune_steps_list:
                threshold = args.training.prune_thresh_list[
                    bisect.bisect_left(args.training.prune_steps_list, step)
                ]
            if step % args.training.prune_steps == 0:
                model.prune_points(threshold, step=step)
                model.pruned_points = True
        if model.pruned_points and step in args.training.add_steps_list:
            position = args.training.add_steps_list.index(step)
            count = args.training.add_num_list[position]
            if args.max_num_pts > 0:
                count = min(count, args.max_num_pts - model.points.shape[0])
            if count > 0:
                model.add_points(count, step=step, prune_thresh=threshold)
        elif (
            model.pruned_points
            and args.training.add_steps > 0
            and args.training.add_start <= step < args.training.add_stop
            and step % args.training.add_steps == 0
        ):
            count = args.training.add_num
            if args.max_num_pts > 0:
                count = min(count, args.max_num_pts - model.points.shape[0])
            if count > 0:
                model.add_points(count, step=step, prune_thresh=threshold)
        model.step(
            step,
            grad_clip_norm=args.training.get("grad_clip_norm"),
            grad_clip_value=args.training.get("grad_clip_value"),
        )
        model.scaler.update()

    metrics = {
        "total": float(total.detach()),
        "rgb": float(rgb_loss.detach()),
        "hit": float(hit_loss.detach()),
        "points_lr": float(model.pts_lr),
        "attention_lr": float(model.attn_lr),
    }
    if consistency is not None:
        metrics["consistency_geometry"] = float(consistency.geometry.detach())
        metrics["consistency_texture"] = float(consistency.texture.detach())
    return metrics


def train(
    resolved: ResolvedConfig,
    output_dir: str | Path,
    *,
    device: str | torch.device = "cuda",
    resume: str | Path | None = None,
    save_every: int | None = None,
    checkpoint_step: int | None = None,
) -> Path:
    """Run one resolved phase and return its final checkpoint path."""
    args = resolved.values
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the released P-CORE kernels")
    if save_every is not None and save_every <= 0:
        raise ValueError("--save-every must be positive")
    if resolved.max_steps > resolved.scheduler_total_steps:
        # Past its horizon a CosineAnnealingLR mirrors back upward, so every
        # learning rate would climb toward its base value instead of staying at
        # the annealed floor.  Refuse rather than silently destroy the model.
        raise ValueError(
            f"Target step {resolved.max_steps} exceeds the phase's scheduler horizon "
            f"{resolved.scheduler_total_steps}; the cosine schedule would restart and "
            f"raise every learning rate. Re-run with "
            f"--scheduler-steps {resolved.max_steps} to extend the schedule deliberately."
        )

    config_id = f"{args.release.benchmark}/{args.release.scene}"
    config_sha256 = sha256_file(resolved.source)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # The archived `save_dir` is a workspace-relative research path.  Point every
    # auxiliary dump (currently the added-point clouds) at the run's own output
    # directory so nothing is written relative to the current directory.
    args.save_dir = str(output_dir)
    dataset = RayImageDataset(args.dataset, split="train", device=device)
    model = PAPR(args, device=device).to(device)
    _attach_camera(model, dataset)

    load_report = None
    loaded_checkpoint = Path(resume) if resume is not None else resolved.checkpoint
    if loaded_checkpoint is not None:
        # The resolver already derived the manifest-backed expectations for the
        # phase's initialization checkpoint, including the reconstruction parent
        # checkpoints whose filename carries no step (`base_model.pth`).  A
        # `--resume` file is a self-describing pcore-2 state instead, so it is
        # verified by `restore_training_checkpoint` rather than here.
        expected_sha = resolved.expected_checkpoint_sha256 if resume is None else None
        expected_step = resolved.expected_checkpoint_step if resume is None else None
        if checkpoint_step is not None:
            if resume is not None:
                raise ValueError("--checkpoint-step applies to --checkpoint, not --resume")
            # A user-supplied bare legacy file carries no step in its name.
            expected_step = checkpoint_step
        load_report = load_checkpoint_strict(
            model,
            loaded_checkpoint,
            expected_step=expected_step,
            expected_sha256=expected_sha,
        )
        # Strict loading materializes dynamically sized Parameters.  Rebuild all
        # optimizer references before either fresh fine-tuning or state restore.
        model.init_optimizers(total_steps=0)
        model.pruned_points = True
    if resolved.freeze_geometry:
        freeze_loaded_geometry(
            model,
            # Accumulators are reset only when entering phase 2 from a paper
            # checkpoint, never when continuing a pcore-2 phase-2 run.
            reset_accumulators=resolved.reset_accumulators and resume is None,
        )

    loader = build_loader(dataset, args.dataset)
    sampler = loader.batch_sampler
    # Construct every potentially stochastic module before restoring RNG.  In
    # particular, LPIPS/VGG construction can initialize module parameters even
    # though pretrained weights are loaded immediately afterwards.
    loss_fn = get_loss(args.training.losses).to(device)
    step = 0
    if resume is not None:
        step = restore_training_checkpoint(
            resume,
            model,
            sampler=sampler,
            expected_phase=resolved.phase,
            expected_config_id=config_id,
            expected_config_fingerprint=resolved.effective_fingerprint,
            device=device,
        )
        _validate_resume_target(step, resolved.max_steps)

    checkpoint_load = load_report.as_dict() if load_report else None
    if checkpoint_load is not None:
        checkpoint_load["path"] = Path(checkpoint_load["path"]).name
    metadata = {
        "phase": resolved.phase,
        "max_steps": resolved.max_steps,
        "target_step": resolved.max_steps,
        "start_step": step,
        "scheduler_total_steps": resolved.scheduler_total_steps,
        "config": {
            "id": config_id,
            "sha256": config_sha256,
            "effective_sha256": resolved.effective_fingerprint,
        },
        "checkpoint_load": checkpoint_load,
        "resume": Path(resume).name if resume is not None else None,
        "save_every": save_every,
    }
    (output_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    def write_checkpoint(current_step: int) -> Path:
        return save_checkpoint(
            output_dir / f"checkpoint_{current_step:06d}.pt",
            model,
            step=current_step,
            phase=resolved.phase,
            config_id=config_id,
            config_sha256=config_sha256,
            config_fingerprint=resolved.effective_fingerprint,
            sampler=sampler,
            device=device,
        )

    started = time.monotonic()
    last_saved_step: int | None = None
    while step < resolved.max_steps:
        for batch in loader:
            metrics = _train_batch(step, model, dataset, batch, loss_fn, args)
            step += 1
            if step == 1 or step % 10 == 0 or step == resolved.max_steps:
                elapsed = time.monotonic() - started
                print(
                    f"[{resolved.phase}] {step:06d}/{resolved.max_steps:06d} "
                    f"loss={metrics['total']:.6f} rgb={metrics['rgb']:.6f} "
                    f"elapsed={elapsed:.1f}s"
                )
            if save_every is not None and step % save_every == 0:
                write_checkpoint(step)
                last_saved_step = step
            if step >= resolved.max_steps:
                break

    destination = output_dir / f"checkpoint_{step:06d}.pt"
    if last_saved_step != step:
        destination = write_checkpoint(step)
    return destination
