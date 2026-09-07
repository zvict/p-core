"""The fixed-dilation surface-consistency recipe used for the release."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from pytorch3d.ops import knn_points

from models.utils import erode_mask_by_pixel_count


@dataclass(frozen=True)
class ConsistencyLoss:
    geometry: torch.Tensor
    texture: torch.Tensor


def _dilate(points: torch.Tensor, center: torch.Tensor, mask: torch.Tensor | None = None):
    """Apply the historical exact 2x centroid dilation."""
    output = points.clone()
    if mask is None:
        return (output - center) * 2.0 + center
    output[mask] = (output[mask] - center) * 2.0 + center
    return output


def compute_consistency_loss(
    model,
    loss_fn,
    args,
    *,
    step: int,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    mask: torch.Tensor,
    pixel_coords: torch.Tensor,
    pixels: torch.Tensor,
    rays_d_raw: torch.Tensor,
    c2w: torch.Tensor,
    target_rgb: torch.Tensor,
    background: torch.Tensor,
    canonical_attention: torch.Tensor,
    canonical_indices: torch.Tensor,
) -> ConsistencyLoss:
    """Compute the two geometry branches and their GT-pixel texture losses.

    This intentionally follows the archived implementation, including unsquared
    Euclidean surface distance, summed branches, and GT RGB texture targets.
    """
    batch, height, width, _ = rays_d.shape
    rays_o_full = rays_o.reshape(batch, 1, 1, 3).expand(-1, height, width, -1)

    with torch.no_grad():
        canonical_points = model.points.detach().clone()
        canonical_surface = model.get_surface_points(
            canonical_indices.detach(),
            model.get_points(canonical_points, drop=False),
            rays_o_full,
            rays_d,
            canonical_attention.detach(),
            "position-sphere",
        ).squeeze(-2)
        if args.texture.divide_by_coord_scale:
            canonical_surface = canonical_surface * model.coord_scale

        flat_surface = canonical_surface.reshape(1, -1, 3)
        distances = knn_points(flat_surface, canonical_points.unsqueeze(0), K=1).dists
        threshold = args.edit.aug_fg_nn_dist_threshold * model.coord_scale
        foreground = distances.squeeze(0).squeeze(-1).sqrt().reshape(batch, height, width) < threshold
        if not foreground.any():
            zero = torch.zeros((), device=rays_d.device)
            return ConsistencyLoss(zero, zero)

        center = canonical_points.mean(dim=0, keepdim=True)
        deformed_points = _dilate(canonical_points, center)
        deformed_surface = _dilate(canonical_surface, center, foreground)

        depth = torch.linalg.vector_norm(canonical_surface - rays_o_full, dim=-1, keepdim=True)
        normalized_rays = rays_d / torch.linalg.vector_norm(rays_d, dim=-1, keepdim=True)
        termination = rays_o_full + depth * normalized_rays
        shifted_origins = rays_o_full + (depth - args.edit.aug_how_far_to_sp) * normalized_rays

        # Points-only branch: keep origins fixed, deform foreground endpoints.
        po_termination = _dilate(termination, center, foreground)
        po_directions = rays_d.clone()
        po_direction_fg = po_termination[foreground] - rays_o_full[foreground]
        po_directions[foreground] = po_direction_fg / torch.linalg.vector_norm(
            po_direction_fg, dim=-1, keepdim=True
        )

        # Rays-and-points branch: apply the identical field to origins/endpoints.
        rap_origins = _dilate(shifted_origins, center, foreground)
        rap_termination = _dilate(termination, center, foreground)
        rap_directions = (rap_termination - rap_origins) / torch.linalg.vector_norm(
            rap_termination - rap_origins, dim=-1, keepdim=True
        )

        foreground = erode_mask_by_pixel_count(
            foreground, pixels_to_erode=args.edit.aug_erode_pixels
        )
        if not foreground.any():
            zero = torch.zeros((), device=rays_d.device)
            return ConsistencyLoss(zero, zero)

    def branch(branch_origins: torch.Tensor, branch_directions: torch.Tensor):
        rgb, _, _, _, _, attention, _, indices, _ = model(
            branch_origins,
            branch_directions,
            c2w,
            pixel_coords,
            pixels,
            background,
            mask,
            # The consistency branches render the same scene at the same
            # training step as the canonical forward, so they must see the real
            # step: schedules keyed on it (notably the spherical-harmonic degree
            # ramp) are otherwise evaluated at a sentinel value.
            step,
            rays_d_no_norm=rays_d_raw,
            deformed_points=deformed_points.detach(),
            log=False,
            drop=False,
        )
        predicted_surface = model.get_surface_points(
            indices,
            model.get_points(deformed_points=deformed_points, drop=False).detach(),
            branch_origins,
            branch_directions,
            attention,
            "position-sphere",
        ).squeeze(-2)
        if args.texture.divide_by_coord_scale:
            predicted_surface = predicted_surface * model.coord_scale
        factor = model.coord_scale if args.texture.divide_by_coord_scale else 1.0
        geometry = torch.linalg.vector_norm(
            (deformed_surface[foreground].detach() - predicted_surface[foreground]) / factor,
            dim=-1,
        ).mean()
        texture = loss_fn(rgb[foreground].reshape(-1, 3), target_rgb[foreground].reshape(-1, 3))
        return geometry, texture

    po_geometry, po_texture = branch(rays_o_full, po_directions)
    rap_geometry, rap_texture = branch(rap_origins, rap_directions)
    return ConsistencyLoss(po_geometry + rap_geometry, po_texture + rap_texture)
