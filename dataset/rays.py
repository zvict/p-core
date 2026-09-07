"""Camera and ray helpers."""

from __future__ import annotations

import torch


def _transform(coords: torch.Tensor, matrix: torch.Tensor, *, vector: bool) -> torch.Tensor:
    homogeneous = torch.cat(
        [coords, torch.zeros_like(coords[..., :1]) if vector else torch.ones_like(coords[..., :1])],
        dim=-1,
    )
    if homogeneous.ndim == 4:
        result = torch.sum(homogeneous.unsqueeze(-2) * matrix[:, None, None, :, :], dim=-1)
    elif homogeneous.ndim == 3:
        result = torch.sum(homogeneous.unsqueeze(-2) * matrix[None, None, :, :], dim=-1)
    else:
        raise ValueError(f"Unsupported coordinate shape: {tuple(coords.shape)}")
    return result[..., :3]


def camera_to_world(coords: torch.Tensor, c2w: torch.Tensor) -> torch.Tensor:
    return _transform(coords, c2w, vector=True)


def world_to_camera(coords: torch.Tensor, c2w: torch.Tensor) -> torch.Tensor:
    return _transform(coords, torch.linalg.inv(c2w), vector=False)


def make_rays(
    height: int,
    width: int,
    focal_x: float,
    focal_y: float,
    c2w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create the normalized and unnormalized rays used by legacy checkpoints."""
    x_axis = torch.linspace(0, width / focal_x, width + 1, device=c2w.device)
    y_axis = torch.linspace(0, height / focal_y, height + 1, device=c2w.device)
    y, x = torch.meshgrid(y_axis, x_axis, indexing="ij")
    x = (x - width / focal_x / 2 + (x_axis[1] - x_axis[0]) / 2)[:-1, :-1]
    y = -(y - height / focal_y / 2 + (y_axis[1] - y_axis[0]) / 2)[:-1, :-1]
    camera_directions = torch.stack([x, y, -torch.ones_like(x)], dim=-1)
    directions = camera_to_world(camera_directions.unsqueeze(0), c2w)
    origins = c2w[:, :3, -1]
    pixels = world_to_camera(origins[:, None, None, :] + directions, c2w)
    normalized = directions / torch.linalg.vector_norm(directions, dim=-1, keepdim=True)
    return origins, normalized, pixels, directions
