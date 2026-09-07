"""Chunked legacy-compatible PAPR rendering."""

from __future__ import annotations

import torch


def _selection_width(model) -> int:
    width = model.select_k
    args = model.args
    if args.geoms.points.select_k_rnd:
        width = max(width, args.geoms.points.select_k_max)
    if args.models.attn.select_prop_scores:
        width = min(width, args.models.attn.select_prop_scores_topk)
    if args.models.attn.select_v_scores:
        width = min(width, args.models.attn.select_v_scores_topk)
        if args.models.attn.select_v_scores_rnd:
            width = max(width, args.models.attn.select_v_scores_topk_max)
    return width


@torch.no_grad()
def render_image(model, sample: dict[str, torch.Tensor], c2w: torch.Tensor, *, step: int, tile: int):
    """Render one full image, matching the archived tiled feature-map path."""
    rays_o = sample["rayo"].to(model.device)
    rays_d = sample["rayd"].to(model.device)
    rays_d_raw = sample["rays_d_no_norm"].to(model.device)
    pixel_coords = sample["pix_coords"].to(model.device)
    if pixel_coords.ndim == 3:
        pixel_coords = pixel_coords.unsqueeze(0)
    c2w = c2w.to(model.device)
    batch, height, width, _ = rays_d.shape
    selected_width = _selection_width(model)
    args = model.args

    background_sequence = 0
    if (
        not args.geoms.no_additional_bkg
        and model.bkg_feats is not None
        and not args.geoms.points.use_bkg_points
        and not args.models.attn.append_bkg_points
    ):
        background_sequence = model.bkg_feats.shape[0]
    appended = int(args.models.attn.append_bkg_points and not args.geoms.no_additional_bkg)
    features = torch.zeros(batch, height, width, 1, model.feat_map_dim, device=model.device)
    attention = torch.zeros(
        batch,
        height,
        width,
        selected_width + appended + background_sequence,
        1,
        device=model.device,
    )
    min_d2r = torch.zeros(batch, height, width, device=model.device)
    selected_indices = torch.empty(
        batch, height, width, selected_width, dtype=torch.long, device=model.device
    )

    for top in range(0, height, tile):
        for left in range(0, width, tile):
            bottom, right = min(top + tile, height), min(left + tile, width)
            region = (slice(None), slice(top, bottom), slice(left, right))
            tile_features, tile_attention, _, _ = model.evaluate(
                rays_o,
                rays_d[region],
                c2w,
                pixel_coords[region],
                step=step,
                rays_d_no_norm=rays_d_raw[region],
            )
            features[region] = tile_features
            attention[region] = tile_attention
            selected_indices[region] = model.select_k_ind
            min_d2r[region] = model.proximity_attn.d2r.min(dim=-2).values.squeeze(-1)

    if args.models.unet.use:
        if args.models.unet.double_channel:
            features = torch.cat([features, features], dim=-1)
        foreground = model.unet(features.squeeze(-2).permute(0, 3, 1, 2))
        foreground = foreground.permute(0, 2, 3, 1).unsqueeze(-2)
    elif args.models.fused_feature_mlp.use:
        encoded = model.fused_feature_encoder(features.reshape(batch * height * width, -1))
        foreground = model.fused_feature_mlp(encoded).reshape(batch, height, width, 1, 3)
    else:
        foreground = features

    if args.geoms.no_additional_bkg:
        rgb = foreground.squeeze(-2)
    elif model.bkg_feats is not None and not args.geoms.points.use_bkg_points and not args.models.attn.append_bkg_points:
        background_attention = attention[..., -1:, :]
        if args.geoms.background.use_dumb_constant:
            background_attention = (background_attention * model.dumb_constant).clamp(0, 1)
        background_attention = model.bkg_attn_act(
            (background_attention + args.models.bkg_attn_shift) * args.models.bkg_attn_scale
        )
        if args.mask_bkg_rays_thresh > 0:
            foreground_mask = min_d2r[..., None] < args.mask_bkg_rays_thresh
            masked = torch.ones_like(background_attention)
            masked[foreground_mask] = background_attention[foreground_mask]
            background_attention = masked
        if args.models.bkg_attn_deno_thresh > 0:
            denominator = background_attention.detach().clone()
            denominator[denominator < args.models.bkg_attn_deno_thresh] = 1
            background_attention = background_attention / denominator
        if args.models.bkg_attn_sigmoid_scale > 0:
            background_attention = torch.sigmoid(
                background_attention * args.models.bkg_attn_sigmoid_scale
            )
        background = model.bkg_feats * model.bkg_scaler.squeeze(0)
        cubemap = model.get_cubemap_background_color(rays_o, rays_d, cur_step=-1)
        background_rgb = (
            cubemap.unsqueeze(-2)
            if cubemap is not None
            else background.expand(batch, height, width, -1, -1)
        )
        rgb = (foreground * (1 - background_attention) + background_rgb * background_attention).squeeze(-2)
    else:
        rgb = foreground.squeeze(-2)
        background = model.bkg_feats * model.bkg_scaler.squeeze(0)
        cubemap = model.get_cubemap_background_color(rays_o, rays_d, cur_step=-1)
        background_rgb = cubemap if cubemap is not None else background.expand(batch, height, width, -1)
        if args.geoms.points.use_bkg_points and args.models.attn.append_bkg_points:
            topk_attention = attention[..., :-1, :]
            background_attention = attention[..., -1, :]
            selected_background = model.bkg_points_mask[selected_indices].expand(-1, -1, -1, -1, 1)
            background_attention = (
                topk_attention * (1 - selected_background.float())
            ).sum(dim=3) + background_attention
            if args.geoms.points.bkg_points_alpha_blend or args.models.attn.append_bkg_points_alpha_blend:
                rgb = rgb * (1 - background_attention) + background_rgb * background_attention
        elif args.geoms.points.use_bkg_points:
            selected_background = model.bkg_points_mask[selected_indices].expand(-1, -1, -1, -1, 1)
            background_attention = (attention * (1 - selected_background.float())).sum(dim=3)
            if args.geoms.points.bkg_points_alpha_blend:
                rgb = rgb * (1 - background_attention) + background_rgb * background_attention
        elif args.models.attn.append_bkg_points:
            background_attention = attention[..., -1, :]
            if args.models.attn.append_bkg_points_alpha_blend:
                rgb = rgb * (1 - background_attention) + background_rgb * background_attention

    return model.last_act(rgb).clamp(0, 1).float()
