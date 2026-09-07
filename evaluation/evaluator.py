"""Paper-protocol metric evaluation."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import imageio.v2 as imageio
import lpips
import numpy as np
import open3d as o3d
import piq
import torch

from checkpoint import load_checkpoint_strict
from dataset import RayImageDataset
from models import PAPR

from .render import render_image


def _rotation_z(degrees: float, *, device, dtype) -> torch.Tensor:
    radians = np.deg2rad(degrees)
    cosine, sine = np.cos(radians), np.sin(radians)
    return torch.tensor(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    )


def _rotate_sample(sample, c2w, rotation):
    sample = dict(sample)
    for key in ("rayo", "rayd", "rays_d_no_norm"):
        sample[key] = sample[key] @ rotation.T
    transform = torch.eye(4, device=c2w.device, dtype=c2w.dtype)
    transform[:3, :3] = rotation
    return sample, transform.unsqueeze(0) @ c2w


def _load_points(path: str | Path) -> torch.Tensor:
    path = Path(path)
    if path.suffix == ".npy":
        points = np.load(path)
    else:
        points = np.asarray(o3d.io.read_point_cloud(str(path)).points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected an N x 3 point cloud in {path}")
    return torch.from_numpy(points.astype(np.float32))


def evaluate(
    args,
    *,
    checkpoint: str | Path,
    checkpoint_step: int,
    checkpoint_sha256: str | None,
    points: str | Path,
    dataset_path: str | Path,
    protocol,
    output_dir: str | Path,
    device: str | torch.device = "cuda",
    max_frames: int | None = None,
    save_images: bool = False,
) -> dict:
    """Render edited points and return per-frame and aggregate paper metrics."""
    device = torch.device(device)
    init_args = copy.deepcopy(args)
    # Checkpoint loading materializes all dynamic tensors; the initial PLY is not
    # needed for evaluation and would make checkpoint assets depend on training assets.
    init_args.geoms.points.load_path = ""
    model = PAPR(init_args, device=device).to(device)
    report = load_checkpoint_strict(
        model,
        checkpoint,
        expected_step=checkpoint_step,
        expected_sha256=checkpoint_sha256,
    )

    data_args = copy.deepcopy(args.dataset)
    data_args.path = str(dataset_path)
    data_args.factor = 1
    data_args.extract_patch = False
    data_args.load_gt_depth = False
    dataset = RayImageDataset(data_args, split=protocol.split, device=device)
    model.fx, model.fy = dataset.focal_x, dataset.focal_y
    model.cx, model.cy = dataset.cx, dataset.cy
    model.H, model.W = dataset.H, dataset.W

    edited = _load_points(points).to(device) * model.coord_scale
    rotation = None
    if abs(protocol.world_rotation_z_degrees) > 1e-8:
        rotation = _rotation_z(
            protocol.world_rotation_z_degrees, device=device, dtype=edited.dtype
        )
        edited = edited @ rotation.T
    if edited.shape != model.points.shape:
        raise ValueError(
            f"Edited point shape {tuple(edited.shape)} does not match checkpoint "
            f"{tuple(model.points.shape)}"
        )
    model.points.data = edited

    lpips_vgg = lpips.LPIPS(net="vgg", version="0.1").to(device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_count = min(protocol.frames, len(dataset))
    if max_frames is not None:
        frame_count = min(frame_count, max_frames)
    per_frame = []
    for index in range(frame_count):
        sample = dataset.get_full_img(index)
        target = sample["image"].to(device).float()
        c2w = dataset.get_c2w([index]).to(device)
        if rotation is not None:
            sample, c2w = _rotate_sample(sample, c2w, rotation)
        prediction = render_image(
            model,
            sample,
            c2w,
            step=checkpoint_step,
            tile=args.test.max_height,
        )
        if protocol.mask_background:
            mask = sample["mask"].to(device)
            prediction = prediction * mask + (1 - mask)
        nchw_prediction = prediction.permute(0, 3, 1, 2)
        nchw_target = target.permute(0, 3, 1, 2)
        metrics = {
            "frame": index,
            "psnr": float(piq.psnr(nchw_prediction, nchw_target).squeeze().detach()),
            "ssim": float(piq.ssim(nchw_prediction, nchw_target).squeeze().detach()),
            "lpips_vgg": float(
                lpips_vgg(nchw_prediction, nchw_target).squeeze().detach()
            ),
        }
        per_frame.append(metrics)
        print(
            f"frame {index:04d}: PSNR={metrics['psnr']:.4f} "
            f"SSIM={metrics['ssim']:.4f} LPIPS-VGG={metrics['lpips_vgg']:.4f}"
        )
        if save_images:
            imageio.imwrite(
                output_dir / f"frame_{index:04d}.png",
                (prediction.squeeze(0).cpu().numpy() * 255).astype(np.uint8),
            )

    if not per_frame:
        raise ValueError("Evaluation selected zero frames")
    aggregate = {
        name: float(np.mean([row[name] for row in per_frame]))
        for name in ("psnr", "ssim", "lpips_vgg")
    }
    result = {
        "checkpoint": report.as_dict(),
        "frames": frame_count,
        "metrics": aggregate,
        "per_frame": per_frame,
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
