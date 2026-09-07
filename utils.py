"""Small runtime utilities shared by data loading and rendering."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


class ConfigNode(dict):
    """Dictionary with recursive attribute access used by the legacy renderer."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return to_config_node(value)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = to_config_node(value)

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def to_config_node(value: Any) -> Any:
    if isinstance(value, ConfigNode):
        return value
    if isinstance(value, dict):
        return ConfigNode({key: to_config_node(item) for key, item in value.items()})
    if isinstance(value, list):
        return [to_config_node(item) for item in value]
    return value


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_ray_sphere_intersection(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    sphere_center: torch.Tensor | list[float] | tuple[float, float, float] | float | None = None,
    sphere_radius: torch.Tensor | float = 1.0,
) -> torch.Tensor:
    """Return the forward intersection of each ray with a sphere."""
    original_ndim = rays_d.ndim
    if original_ndim == 3:
        rays_d = rays_d.unsqueeze(0)

    batch, height, width, _ = rays_d.shape
    if rays_o.ndim == 1:
        rays_o = rays_o.reshape(1, 1, 1, 3).expand(batch, height, width, -1)
    elif rays_o.ndim == 2:
        rays_o = rays_o.reshape(batch, 1, 1, 3).expand(-1, height, width, -1)
    elif rays_o.ndim == 3:
        rays_o = rays_o.unsqueeze(0).expand(batch, -1, -1, -1)

    directions = F.normalize(rays_d, dim=-1, eps=1e-8)
    if sphere_center is None or (
        isinstance(sphere_center, (int, float)) and sphere_center == 0
    ):
        center = torch.zeros((1, 1, 1, 3), device=rays_d.device, dtype=rays_d.dtype)
    else:
        center = torch.as_tensor(sphere_center, device=rays_d.device, dtype=rays_d.dtype)
        center = center.reshape(1, 1, 1, 3)
    radius = torch.as_tensor(sphere_radius, device=rays_d.device, dtype=rays_d.dtype)

    offset = rays_o - center
    offset_dot_direction = torch.sum(offset * directions, dim=-1)
    discriminant = (
        offset_dot_direction.square() - torch.sum(offset.square(), dim=-1) + radius.square()
    )
    distance = -offset_dot_direction + torch.sqrt(torch.clamp(discriminant, min=0.0))
    points = rays_o + torch.clamp(distance, min=0.0).unsqueeze(-1) * directions
    return points.squeeze(0) if original_ndim == 3 else points
