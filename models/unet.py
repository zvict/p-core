"""U-Net feature decoder used by the released P-CORE renderers.

An independent implementation of the standard U-Net encoder/decoder with skip
connections (Ronneberger, Fischer and Brox, *U-Net: Convolutional Networks for
Biomedical Image Segmentation*, MICCAI 2015), written to the fixed three-level
depth and channel widths the released checkpoints were trained with.

The submodule names below (``inc``, ``down1``, ``down2``, ``up1``, ``up2``,
``outc``, and the ``double_conv`` / ``maxpool_conv`` / ``up`` / ``conv``
children inside them) are the keys the released checkpoints serialize, so they
are part of the on-disk state-dict format rather than a stylistic choice.

Released configuration
----------------------
Every shipped scene that enables the U-Net uses one configuration:
``single=True`` (one convolution per block), ``norm='none'``, ``act='relu'``,
``bilinear=False`` (learned transposed-convolution upsampling), ``groups=1``,
``channel_factor=1``, ``use_outc=True``, ``last_act='none'`` and
``inp_scale=1.0``.  The other branches exist because the archived configuration
schema exposes them.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast

from .utils import activation_func


_NORMALIZATIONS = {
    "none": None,
    "batch": nn.BatchNorm2d,
    "instance": nn.InstanceNorm2d,
    "group": lambda channels: nn.GroupNorm(1, channels),
}


def _conv_stage(in_channels: int, out_channels: int, *, norm: str, act: str, groups: int) -> list:
    """One same-resolution stage: 3x3 convolution, optional norm, activation."""
    if norm not in _NORMALIZATIONS:
        raise NotImplementedError(
            f"Unknown U-Net normalization {norm!r}; expected one of {sorted(_NORMALIZATIONS)}"
        )
    stage = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=groups)]
    normalization = _NORMALIZATIONS[norm]
    if normalization is not None:
        stage.append(normalization(out_channels))
    stage.append(activation_func(act, inplace=True))
    return stage


class ConvBlock(nn.Module):
    """One or two same-resolution convolution stages.

    A one-stage block emits ``mid_channels`` and a two-stage block widens to
    ``mid_channels`` before emitting ``out_channels``; both default
    ``mid_channels`` to ``out_channels``.  Keeping the single-stage output tied
    to ``mid_channels`` preserves the shapes the released checkpoints store.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int | None = None,
        *,
        stages: int = 1,
        norm: str = "none",
        act: str = "relu",
        groups: int = 1,
    ) -> None:
        super().__init__()
        if stages not in (1, 2):
            raise NotImplementedError(f"A U-Net block has one or two stages, not {stages}")
        if not mid_channels:
            mid_channels = out_channels
        widths = [(in_channels, mid_channels)]
        if stages == 2:
            widths.append((mid_channels, out_channels))
        layers: list[nn.Module] = []
        for stage_in, stage_out in widths:
            layers.extend(_conv_stage(stage_in, stage_out, norm=norm, act=act, groups=groups))
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Halve the resolution with max pooling, then convolve."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        single: bool = False,
        norm: str = "none",
        act: str = "relu",
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(
                in_channels,
                out_channels,
                stages=1 if single else 2,
                norm=norm,
                act=act,
                groups=groups,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Double the resolution, concatenate the skip connection, then convolve."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bilinear: bool = True,
        single: bool = False,
        norm: str = "none",
        act: str = "relu",
        groups: int = 1,
    ) -> None:
        super().__init__()
        if bilinear:
            # Parameter-free upsampling keeps all `in_channels`, so concatenating
            # the `in_channels // 2` skip gives 1.5x the width the following
            # block expects.  The archived code had the same mismatch and would
            # raise a shape error inside the convolution; no released config
            # selects it, so refuse it up front instead.
            raise NotImplementedError(
                "bilinear upsampling is not supported by this channel layout; "
                "the released configuration uses models.unet.bilinear: false"
            )
        stages = 1 if single else 2
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2, groups=groups
        )
        self.conv = ConvBlock(
            in_channels, out_channels, stages=stages, norm=norm, act=act, groups=groups
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # An odd resolution loses a pixel on the pooled path, so the upsampled
        # map can come back one row or column short of its skip connection.
        # Centre it inside the skip, putting any leftover pixel on the bottom
        # and the right, before concatenating along the channel axis.
        missing_rows = skip.shape[-2] - x.shape[-2]
        missing_columns = skip.shape[-1] - x.shape[-1]
        x = F.pad(
            x,
            (
                missing_columns // 2,
                missing_columns - missing_columns // 2,
                missing_rows // 2,
                missing_rows - missing_rows // 2,
            ),
        )
        return self.conv(torch.cat((skip, x), dim=1))


class OutConv(nn.Module):
    """Pointwise projection from decoder features to output channels."""

    def __init__(self, in_channels: int, out_channels: int, groups: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """Three-level U-Net: 128 -> 256 -> 512 channels and back.

    ``gamma`` and ``beta`` are accepted by :meth:`forward` because the renderer
    passes its shading-code affine parameters positionally; they are unused, as
    in the archived implementation the released checkpoints were trained with.
    """

    def __init__(
        self,
        n_channels,
        n_classes,
        bilinear=False,
        single=True,
        norm="none",
        last_act="none",
        act="relu",
        inp_scale=1.0,
        use_amp=False,
        amp_dtype=torch.float16,
        affine_layer=-1,
        use_outc=True,
        groups=1,
        group_last=False,
        channel_factor=1,
    ):
        super().__init__()
        if groups != 1:
            raise NotImplementedError(
                "Grouped U-Net convolutions are not part of the released configuration"
            )
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype
        self.affine_layer = affine_layer
        self.inp_scale = inp_scale
        self.use_outc = use_outc
        self.groups = groups
        self.channel_factor = channel_factor
        self.group_last = group_last

        narrow = 128 * channel_factor
        middle = 256 * channel_factor
        wide = 512 * channel_factor
        shape = {"norm": norm, "act": act, "groups": groups}

        self.inc = ConvBlock(n_channels, narrow, stages=1, **shape)
        self.down1 = Down(narrow, middle, single=single, **shape)
        self.down2 = Down(middle, wide, single=single, **shape)
        self.up1 = Up(wide, middle, bilinear, single=single, **shape)
        if use_outc:
            self.up2 = Up(middle, narrow, bilinear, single=single, **shape)
            self.outc = OutConv(narrow, n_classes, groups=groups if group_last else 1)
        else:
            # Without the pointwise head the last decoder block emits the
            # output channels directly.
            self.up2 = Up(middle, n_classes, bilinear, single=single, **shape)

        self.last_act = activation_func(last_act)

    def forward(self, x, log=False, gamma=None, beta=None):
        with autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp):
            x = x * self.inp_scale

            level1 = self.inc(x)          # (N, 128 * channel_factor, H,   W)
            level2 = self.down1(level1)   # (N, 256 * channel_factor, H/2, W/2)
            level3 = self.down2(level2)   # (N, 512 * channel_factor, H/4, W/4)

            decoded = self.up1(level3, level2)   # (N, 256 * channel_factor, H/2, W/2)
            decoded = self.up2(decoded, level1)  # (N, 128 * channel_factor, H,   W)
            if self.use_outc:
                decoded = self.outc(decoded)     # (N, n_classes, H, W)

            return self.last_act(decoded)
