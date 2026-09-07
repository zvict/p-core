"""In-memory ray/image dataset shared by reconstruction and consistency training."""

from __future__ import annotations

import glob
import re
from collections.abc import Iterator, Mapping
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from utils import get_ray_sphere_intersection

from .blender import load_blender_data
from .rays import make_rays


class RayImageDataset(Dataset):
    """Blender images, masks, cameras, and randomly sampled image patches."""

    def __init__(self, args, split: str = "train", device: str | torch.device = "cuda"):
        if args.type != "synthetic":
            raise ValueError("The release supports Blender-format synthetic datasets only")
        images, poses, (height, width, focal), paths = load_blender_data(
            args.path, split=split, factor=args.factor
        )
        masks = images[..., -1:] if images.shape[-1] == 4 else np.ones_like(images[..., :1])
        if images.shape[-1] == 4:
            rgb = images[..., :3]
            if args.white_bg:
                rgb = rgb * masks + (1.0 - masks) * np.asarray(args.bg_color).reshape(1, 1, 1, 3)
            images = rgb

        device = torch.device(device)
        c2w = torch.from_numpy(poses)
        if args.coord_scale != 1:
            scale = torch.diag(torch.tensor([args.coord_scale] * 3 + [1.0]))
            c2w = scale @ c2w

        self.args = args
        self.device = device
        self.H, self.W = height, width
        self.focal_x = self.focal_y = focal
        self.cx, self.cy = width / 2, height / 2
        self.image_paths = paths
        self.num_imgs = len(paths)
        self.num_rays = self.num_imgs * height * width
        self.c2w = c2w.to(device)
        self.images = torch.from_numpy(images).float().to(device)
        self.masks = torch.from_numpy(masks).float().to(device)
        px, py = torch.meshgrid(
            torch.arange(width, dtype=torch.float32) + 0.5,
            torch.arange(height, dtype=torch.float32) + 0.5,
            indexing="xy",
        )
        self.pix_coords = torch.stack([px, py], dim=-1).to(device)
        self.foreground_coords = torch.nonzero(self.masks.squeeze(-1) > 0.5)
        origins, directions, pixels, directions_raw = make_rays(
            height, width, focal, focal, c2w
        )
        self.rayo = origins.to(device)
        self.rayd = directions.to(device)
        self.pixels = pixels.to(device)
        self.rays_d_no_norm = directions_raw.to(device)

        self.gt_depths = self._load_depths(split) if args.load_gt_depth else None
        self.gt_surface_points = None
        if self.gt_depths is not None:
            self.gt_surface_points = self._depth_to_surface(self.gt_depths)

    def _load_depths(self, split: str) -> torch.Tensor:
        directory = str(self.args.gt_depth_dir)
        if "nerf_synthetic" in directory or "m360" in directory:
            directory = directory.replace("train", split).replace("test", split)
        loaded: list[torch.Tensor] = []
        for index in range(self.num_imgs):
            selected = None
            for suffix in (".npy", ".tiff", ".png"):
                for candidate in sorted(glob.glob(str(Path(directory) / f"*{index}*{suffix}"))):
                    if index in [int(value) for value in re.findall(r"\d+", Path(candidate).name)]:
                        selected = candidate
                        break
                if selected:
                    break
            if selected is None:
                raise FileNotFoundError(f"No depth map for frame {index} in {directory}")
            depth = np.load(selected) if selected.endswith(".npy") else imageio.imread(selected)
            depth = torch.from_numpy(np.asarray(depth[..., 0] if depth.ndim == 3 else depth)).float()
            if depth.shape != (self.H, self.W):
                depth = torch.nn.functional.interpolate(
                    depth[None, None], size=(self.H, self.W), mode="bilinear", align_corners=False
                )[0, 0]
            loaded.append(depth * self.args.coord_scale)
        return torch.stack(loaded).to(self.device)

    def _depth_to_surface(self, depths: torch.Tensor) -> torch.Tensor:
        points = self.rayo[:, None, None, :] + self.rays_d_no_norm * (
            depths / self.args.coord_scale
        )[..., None]
        background = self.masks.squeeze(-1) <= 0.5
        sphere = get_ray_sphere_intersection(
            self.rayo,
            self.rays_d_no_norm,
            sphere_radius=self.args.bkg_sphere_radius * self.args.coord_scale,
        )
        return torch.where(background[..., None], sphere, points)

    def __len__(self) -> int:
        return self.num_imgs

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        if self.args.extract_patch:
            patch_h, patch_w = self.args.patches.height, self.args.patches.width
            top = np.random.randint(0, self.H - patch_h)
            left = np.random.randint(0, self.W - patch_w)
            region = np.s_[top : top + patch_h, left : left + patch_w]
        else:
            region = np.s_[:, :]
        result = {
            "idx": index,
            "patch_idx": 0,
            "image": self.images[index][region],
            "mask": self.masks[index][region],
            "rayd": self.rayd[index][region],
            "rays_d_no_norm": self.rays_d_no_norm[index][region],
            "rayo": self.rayo[index],
            "pix_coords": self.pix_coords[region],
            "pixels": self.pixels[index][region],
        }
        if self.gt_surface_points is not None:
            result["depth"] = self.gt_depths[index][region]
            result["surface_points"] = self.gt_surface_points[index][region]
        return result

    def get_full_img(self, index: int) -> dict[str, torch.Tensor]:
        result = {
            "idx": torch.as_tensor(index),
            "image": self.images[index][None],
            "mask": self.masks[index][None],
            "rayd": self.rayd[index][None],
            "rays_d_no_norm": self.rays_d_no_norm[index][None],
            "rayo": self.rayo[index][None],
            "pix_coords": self.pix_coords,
            "pixels": self.pixels[index][None],
        }
        if self.gt_surface_points is not None:
            result["surface_points"] = self.gt_surface_points[index][None]
            result["depth"] = self.gt_depths[index][None]
        return result

    def get_c2w(self, indices) -> torch.Tensor:
        selected = self.c2w[indices]
        return selected[None] if selected.ndim == 2 else selected

    def get_new_rays(self, c2w: torch.Tensor):
        return make_rays(self.H, self.W, self.focal_x, self.focal_y, c2w)[:3]


class ResumableBatchSampler(Sampler[list[int]]):
    """Batch sampler whose permutation and cursor can be checkpointed exactly.

    Training uses ``num_workers=0``, so advancing the cursor immediately before
    yielding a batch records exactly the examples consumed by the caller.  The
    generator state alone is insufficient for a mid-epoch resume because a
    DataLoader creates the entire shuffled permutation at iterator creation.
    """

    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        *,
        shuffle: bool,
        seed: int,
        drop_last: bool = False,
    ) -> None:
        if dataset_size <= 0:
            raise ValueError("Cannot train with an empty dataset")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.dataset_size = int(dataset_size)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(seed))
        self.order: torch.Tensor | None = None
        self.cursor = 0
        self.epoch = 0

    def _start_epoch(self) -> None:
        if self.shuffle:
            self.order = torch.randperm(self.dataset_size, generator=self.generator)
        else:
            self.order = torch.arange(self.dataset_size)
        self.cursor = 0
        self.epoch += 1

    def __iter__(self) -> Iterator[list[int]]:
        if self.order is None or self.cursor >= self.dataset_size:
            self._start_epoch()
        assert self.order is not None
        while self.cursor < self.dataset_size:
            stop = min(self.cursor + self.batch_size, self.dataset_size)
            if self.drop_last and stop - self.cursor < self.batch_size:
                self.cursor = self.dataset_size
                break
            batch = self.order[self.cursor : stop].tolist()
            # Advance before yielding so a checkpoint written after consuming
            # the batch points at the first unseen example.
            self.cursor = stop
            yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return self.dataset_size // self.batch_size
        return (self.dataset_size + self.batch_size - 1) // self.batch_size

    def state_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "dataset_size": self.dataset_size,
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "drop_last": self.drop_last,
            "generator_state": self.generator.get_state().clone(),
            "has_order": self.order is not None,
            "order": (
                self.order.clone()
                if self.order is not None
                else torch.empty(0, dtype=torch.long)
            ),
            "cursor": self.cursor,
            "epoch": self.epoch,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        expected = {
            "dataset_size": self.dataset_size,
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "drop_last": self.drop_last,
        }
        for name, value in expected.items():
            if state.get(name) != value:
                raise ValueError(
                    f"Sampler {name} is {state.get(name)!r}, expected {value!r}"
                )
        if int(state.get("version", -1)) != 1:
            raise ValueError(f"Unsupported sampler state version: {state.get('version')!r}")

        generator_state = state.get("generator_state")
        order = state.get("order")
        if not torch.is_tensor(generator_state) or not torch.is_tensor(order):
            raise ValueError("Sampler generator_state and order must be tensors")
        self.generator.set_state(generator_state.detach().cpu())

        has_order = bool(state.get("has_order", False))
        restored_order = order.detach().cpu().to(torch.long).clone()
        if has_order:
            if restored_order.shape != (self.dataset_size,):
                raise ValueError(
                    f"Sampler order has shape {tuple(restored_order.shape)}, "
                    f"expected {(self.dataset_size,)}"
                )
            if not torch.equal(
                torch.sort(restored_order).values, torch.arange(self.dataset_size)
            ):
                raise ValueError("Sampler order is not a dataset permutation")
            self.order = restored_order
        else:
            if restored_order.numel() != 0:
                raise ValueError("Sampler without an active order must store an empty tensor")
            self.order = None

        cursor = int(state.get("cursor", -1))
        if not 0 <= cursor <= self.dataset_size:
            raise ValueError(f"Sampler cursor is out of range: {cursor}")
        if self.order is None and cursor != 0:
            raise ValueError("Sampler without an active order must have cursor 0")
        epoch = int(state.get("epoch", -1))
        if epoch < 0:
            raise ValueError(f"Sampler epoch is invalid: {epoch}")
        self.cursor = cursor
        self.epoch = epoch


def build_loader(dataset: RayImageDataset, args) -> DataLoader:
    batch_sampler = ResumableBatchSampler(
        len(dataset),
        args.batch_size,
        shuffle=args.shuffle,
        seed=torch.initial_seed(),
    )
    # DataLoader draws an iterator base seed even with num_workers=0.  Keep
    # that bookkeeping on a private generator so recreating an iterator for a
    # mid-epoch resume cannot perturb the model's restored torch RNG stream.
    loader_generator = torch.Generator(device="cpu")
    loader_generator.manual_seed(0)
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=0,
        generator=loader_generator,
    )
