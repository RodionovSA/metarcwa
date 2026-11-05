# metarcwa/solver/pml.py
"""Stretched in-plane wavevector operators of a lateral PML.

Complex coordinate stretching replaces ``∂x → (1/s_x(x)) ∂x``, which in
Fourier space turns the diagonal ``Kx`` into ``K̃x = [[1/s_x]] Kx`` (and
likewise for y). See ``docs/PML.md`` "Fourier-space form".
"""

import torch

from metarcwa.model.pml import PMLSpec
from metarcwa.solver.blockmatrix import Block


def stretched_k(kx: torch.Tensor, ky: torch.Tensor, pml: PMLSpec,
                m_flat: torch.Tensor, n_flat: torch.Tensor,
                factorization: str = "laurent") -> tuple[Block, Block]:
    """Build the stretched wavevector operators ``K̃x``, ``K̃y``.

    Parameters
    ----------
    kx, ky : torch.Tensor
        In-plane wavevector components (k0-normalized), shape ``[..., Nh]``.
    pml : PMLSpec
        Stretch profiles ``s_x`` (``[Nx]``) and ``s_y`` (``[Ny]``); ``None``
        for a disabled axis.
    m_flat, n_flat : torch.Tensor
        Integer harmonic indices along x and y, shape ``[Nh]``.
    factorization : str
        ``"laurent"`` builds ``[[1/s]]``; ``"inverse"`` builds ``[[s]]⁻¹``.

    Returns
    -------
    Kx, Ky : Block
        DENSE ``[..., Nh, Nh]`` along a stretched axis, DIAG ``[..., Nh]``
        (unchanged) along a disabled one.
    """
    out = []
    for k, s, own, other in ((kx, pml.sx, m_flat, n_flat), (ky, pml.sy, n_flat, m_flat)):
        K = Block(Block.DIAG, k)
        if s is None:
            out.append(K)
            continue
        # Toeplitz matrix of a function of one coordinate: couples the
        # harmonic index along that axis only, identity in the other.
        coeff = torch.fft.fft(1 / s if factorization == "laurent" else s, norm="forward")
        T = coeff[(own[:, None] - own[None, :]) % s.shape[-1]] * (other[:, None] == other[None, :])
        if factorization == "inverse":
            T = torch.linalg.inv(T)
        out.append(Block(Block.DENSE, T) @ K)
    return out[0], out[1]
