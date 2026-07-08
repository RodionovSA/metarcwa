# metarcwa/_dtypes.py
"""
_dtypes — shared real/complex dtype helpers
=============================================

Lives at the top of the package (not under ``model/`` or ``solver/``) so both
can import from it without either depending on the other — the ``model``
package originally defined these in ``model/utils.py``, but the ``solver``
package also needs them (e.g. promoting a real permittivity grid to complex
before an eigendecomposition), and ``solver`` must not import from ``model``
(one-way ``model → solver`` dependency direction; see CLAUDE.md).
"""

import warnings
import torch

# Maps a real floating dtype to its matching complex dtype.
_REAL_TO_COMPLEX = {
    torch.float16: torch.complex32,
    torch.float32: torch.complex64,
    torch.float64: torch.complex128,
}


def to_complex(t: torch.Tensor) -> torch.Tensor:
    """Return ``t`` as a complex tensor, promoting if necessary.

    If ``t`` is already complex, it is returned unchanged (no copy). If it is
    real, it is cast to the matching complex dtype (e.g. float32 → complex64).
    """
    if t.is_complex():
        return t
    return t.to(_REAL_TO_COMPLEX.get(t.dtype, torch.complex64))


def to_real(t: torch.Tensor, *, name: str = "tensor", atol: float = 1e-8) -> torch.Tensor:
    """Return ``t`` as a real tensor, dropping the imaginary part if needed.

    If ``t`` is already real, it is returned unchanged. If it is complex and
    has a non-negligible imaginary part (``max|imag| > atol``), a
    ``UserWarning`` is emitted before dropping the imaginary part.

    Parameters
    ----------
    t : Tensor
        Input tensor.
    name : str
        Human-readable name used in the warning message.
    atol : float
        Threshold above which the imaginary part is considered non-negligible.
    """
    if not t.is_complex():
        return t
    if t.imag.abs().max() > atol:
        warnings.warn(
            f"{name} has a non-negligible imaginary part "
            f"(max |imag| = {t.imag.abs().max().item():.3g}); "
            "casting to real (medium treated as lossless).",
            UserWarning,
            stacklevel=2,
        )
    return t.real
