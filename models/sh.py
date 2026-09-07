#  Copyright 2021 The PlenOctree Authors.
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#  this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.

import torch
import torch.nn.functional as F

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]   


def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result -
                C1 * y * sh[..., 1] +
                C1 * z * sh[..., 2] -
                C1 * x * sh[..., 3])

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                    C2[0] * xy * sh[..., 4] +
                    C2[1] * yz * sh[..., 5] +
                    C2[2] * (2.0 * zz - xx - yy) * sh[..., 6] +
                    C2[3] * xz * sh[..., 7] +
                    C2[4] * (xx - yy) * sh[..., 8])

            if deg > 2:
                result = (result +
                C3[0] * y * (3 * xx - yy) * sh[..., 9] +
                C3[1] * xy * z * sh[..., 10] +
                C3[2] * y * (4 * zz - xx - yy)* sh[..., 11] +
                C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] +
                C3[5] * z * (xx - yy) * sh[..., 14] +
                C3[6] * x * (xx - 3 * yy) * sh[..., 15])

                if deg > 3:
                    result = (result + C4[0] * xy * (xx - yy) * sh[..., 16] +
                            C4[1] * yz * (3 * xx - yy) * sh[..., 17] +
                            C4[2] * xy * (7 * zz - 1) * sh[..., 18] +
                            C4[3] * yz * (7 * zz - 3) * sh[..., 19] +
                            C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                            C4[5] * xz * (7 * zz - 3) * sh[..., 21] +
                            C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                            C4[7] * xz * (xx - 3 * yy) * sh[..., 23] +
                            C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])
    # return SH2RGB(result)
    # return torch.sigmoid(result)
    return result


def _eval_sh_bases_fast(basis_dim: int, dirs: torch.Tensor):
    """
    Evaluate spherical harmonics bases at unit direction for high orders
    using approach described by
    Efficient Spherical Harmonic Evaluation, Peter-Pike Sloan, JCGT 2013
    https://jcgt.org/published/0002/02/06/


    :param basis_dim: int SH basis dim. Currently, only 1-25 square numbers supported
    :param dirs: torch.Tensor (..., 3) unit directions

    :return: torch.Tensor (..., basis_dim)

    See reference C++ code in https://jcgt.org/published/0002/02/06/code.zip
    """
    result = torch.empty(
        (*dirs.shape[:-1], basis_dim), dtype=dirs.dtype, device=dirs.device
    )

    result[..., 0] = 0.2820947917738781

    if basis_dim <= 1:
        return result

    x, y, z = dirs.unbind(-1)

    fTmpA = -0.48860251190292
    result[..., 2] = -fTmpA * z
    result[..., 3] = fTmpA * x
    result[..., 1] = fTmpA * y

    if basis_dim <= 4:
        return result

    z2 = z * z
    fTmpB = -1.092548430592079 * z
    fTmpA = 0.5462742152960395
    fC1 = x * x - y * y
    fS1 = 2 * x * y
    result[..., 6] = 0.9461746957575601 * z2 - 0.3153915652525201
    result[..., 7] = fTmpB * x
    result[..., 5] = fTmpB * y
    result[..., 8] = fTmpA * fC1
    result[..., 4] = fTmpA * fS1

    if basis_dim <= 9:
        return result

    fTmpC = -2.285228997322329 * z2 + 0.4570457994644658
    fTmpB = 1.445305721320277 * z
    fTmpA = -0.5900435899266435
    fC2 = x * fC1 - y * fS1
    fS2 = x * fS1 + y * fC1
    result[..., 12] = z * (1.865881662950577 * z2 - 1.119528997770346)
    result[..., 13] = fTmpC * x
    result[..., 11] = fTmpC * y
    result[..., 14] = fTmpB * fC1
    result[..., 10] = fTmpB * fS1
    result[..., 15] = fTmpA * fC2
    result[..., 9] = fTmpA * fS2

    if basis_dim <= 16:
        return result

    fTmpD = z * (-4.683325804901025 * z2 + 2.007139630671868)
    fTmpC = 3.31161143515146 * z2 - 0.47308734787878
    fTmpB = -1.770130769779931 * z
    fTmpA = 0.6258357354491763
    fC3 = x * fC2 - y * fS2
    fS3 = x * fS2 + y * fC2
    result[..., 20] = 1.984313483298443 * z2 * (
        1.865881662950577 * z2 - 1.119528997770346
    ) + -1.006230589874905 * (0.9461746957575601 * z2 - 0.3153915652525201)
    result[..., 21] = fTmpD * x
    result[..., 19] = fTmpD * y
    result[..., 22] = fTmpC * fC1
    result[..., 18] = fTmpC * fS1
    result[..., 23] = fTmpB * fC2
    result[..., 17] = fTmpB * fS2
    result[..., 24] = fTmpA * fC3
    result[..., 16] = fTmpA * fS3
    return result


def spherical_harmonics(
    degrees_to_use: int,
    dirs: torch.Tensor,  # [..., 3]
    coeffs: torch.Tensor,  # [..., K, D]
):
    """Pytorch implementation of `gsplat.cuda._wrapper.spherical_harmonics()`."""
    assert (degrees_to_use + 1) ** 2 <= coeffs.shape[-2], coeffs.shape
    batch_dims = dirs.shape[:-1]
    assert dirs.shape == batch_dims + (3,), dirs.shape
    # Support any feature channel dimension D (not just RGB=3):
    # coeffs shape is [..., num_bases, D], output is [..., D].
    assert (
        (len(coeffs.shape) == len(batch_dims) + 2)
        and coeffs.shape[:-2] == batch_dims
    ), coeffs.shape
    dirs = F.normalize(dirs, p=2, dim=-1)
    num_bases = (degrees_to_use + 1) ** 2
    bases = torch.zeros_like(coeffs[..., 0])
    bases[..., :num_bases] = _eval_sh_bases_fast(num_bases, dirs)
    return (bases[..., None] * coeffs).sum(dim=-2)


def eval_sh_bases(basis_dim : int, dirs : torch.Tensor):
    """
    Evaluate spherical harmonics bases at unit directions,
    without taking linear combination.
    At each point, the final result may the be
    obtained through simple multiplication.

    :param basis_dim: int SH basis dim. Currently, 1-25 square numbers supported
    :param dirs: torch.Tensor (..., 3) unit directions

    :return: torch.Tensor (..., basis_dim)
    """
    result = torch.empty((*dirs.shape[:-1], basis_dim), dtype=dirs.dtype, device=dirs.device)
    result[..., 0] = C0
    if basis_dim > 1:
        x, y, z = dirs.unbind(-1)
        result[..., 1] = -C1 * y;
        result[..., 2] = C1 * z;
        result[..., 3] = -C1 * x;
        if basis_dim > 4:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result[..., 4] = C2[0] * xy;
            result[..., 5] = C2[1] * yz;
            result[..., 6] = C2[2] * (2.0 * zz - xx - yy);
            result[..., 7] = C2[3] * xz;
            result[..., 8] = C2[4] * (xx - yy);

            if basis_dim > 9:
                result[..., 9] = C3[0] * y * (3 * xx - yy);
                result[..., 10] = C3[1] * xy * z;
                result[..., 11] = C3[2] * y * (4 * zz - xx - yy);
                result[..., 12] = C3[3] * z * (2 * zz - 3 * xx - 3 * yy);
                result[..., 13] = C3[4] * x * (4 * zz - xx - yy);
                result[..., 14] = C3[5] * z * (xx - yy);
                result[..., 15] = C3[6] * x * (xx - 3 * yy);

                if basis_dim > 16:
                    result[..., 16] = C4[0] * xy * (xx - yy);
                    result[..., 17] = C4[1] * yz * (3 * xx - yy);
                    result[..., 18] = C4[2] * xy * (7 * zz - 1);
                    result[..., 19] = C4[3] * yz * (7 * zz - 3);
                    result[..., 20] = C4[4] * (zz * (35 * zz - 30) + 3);
                    result[..., 21] = C4[5] * xz * (7 * zz - 3);
                    result[..., 22] = C4[6] * (xx - yy) * (7 * zz - 1);
                    result[..., 23] = C4[7] * xz * (xx - 3 * yy);
                    result[..., 24] = C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy));
    return result


def RGB2SH(rgb):
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    return sh * C0 + 0.5
