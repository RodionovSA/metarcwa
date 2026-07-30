# metarcwa/solver/config.py
"""
config — solver hyperparameters for the RCWA pipeline
======================================================

Two public dataclasses:

  Factorization   — TVF (Tangent Vector Field) factorization settings.
  Config          — Top-level solver configuration.

Both are plain ``@dataclass`` types; pass a ``Config`` instance to
``LayerSolver`` to control grid resolution, harmonic truncation,
factorization method, and eigenvalue solver behaviour.
"""

import yaml
import torch
import warnings
from dataclasses import dataclass, field
from pathlib import Path

_DTYPE_TO_STR: dict[torch.dtype, str] = {
    torch.float32:    "float32",
    torch.float64:    "float64",
    torch.complex64:  "complex64",
    torch.complex128: "complex128",
}
_STR_TO_DTYPE: dict[str, torch.dtype] = {v: k for k, v in _DTYPE_TO_STR.items()}


@dataclass
class Factorization:
    """
    Settings for the TVF (Tangent Vector Field) Li-factorization rule.

    Controls how the permittivity Fourier convolution matrix is computed
    for patterned layers.  The TVF smoothly interpolates the factorization
    direction field across material interfaces, improving convergence of the
    Fourier series.

    Attributes
    ----------
    method : str
        TVF algorithm.  One of ``"Jones"``, ``"Pol"``, ``"Normal"``,
        ``"Jones_direct"``.  Default ``"Jones"``.
    optimizer : str
        Optimiser used to fit the TVF field.  Default ``"newton"``.
    steps : int
        Number of optimiser steps.  Default ``1`` (exact for the Newton
        quadratic, sufficient in most cases).
    alpha : float
        Alignment loss weight (gradient-field alignment term).  Default ``1.0``.
    beta : float
        Fourier regularisation weight (band-limit smoothness).  Default ``0.05``.
    gamma : float
        Smoothness loss weight (spatial smoothness).  Default ``0.05``.
    newton_chunk_size : int or None
        Column-chunk size for the exact-Newton optimizer's Hessian assembly
        (``NewtonExact`` only; ignored by other optimizers). The Newton
        solve builds the Hessian via a ``vmap`` over ``(2*m+1)*(2*n+1)*4``
        basis directions, each triggering a full-grid ``[B, Ny, Nx, 2]``
        forward/tangent evaluation; done all at once this is the dominant
        peak-memory cost of TVF (can reach several GB at ``m=n=10`` and
        ``nx=ny=256``). Setting this to a positive int assembles the
        Hessian in chunks of that many columns instead, cutting peak memory
        by roughly ``flat / newton_chunk_size`` at a modest runtime cost —
        the result is mathematically identical (same Hessian, same solve).
        Default ``64``, which keeps peak memory low on consumer GPUs.  Set
        to ``None`` to disable chunking (single-shot, original, fastest but
        highest-memory behavior).
    newton_cg_max_iter : int or None
        Max CG iterations for the matrix-free Newton-CG optimizer
        (``optimizer="newton_cg"`` only; ignored otherwise). Unlike
        ``NewtonExact``, ``NewtonCG`` never materializes the dense
        ``[B, flat, flat]`` Hessian or its ``O(flat**3)`` solve — each CG
        iteration is one Hessian-vector product, so memory stays
        ``O(flat + grid)`` regardless of harmonic count. Prefer this
        optimizer over ``"newton"`` (with or without ``newton_chunk_size``)
        at high truncation, where the dense Hessian/solve dominate. Default
        ``None`` -> ``2 * flat`` (``flat = (2*m+1)*(2*n+1)*4``), generous
        headroom since exact arithmetic converges within ``flat`` iterations.
    newton_cg_tol : float
        Relative-residual stop tolerance for Newton-CG:
        ``||r|| <= newton_cg_tol * ||g||``. Default ``1e-8``. Use
        ``dtype=torch.float64`` for tight tolerances; ``float32`` will
        typically not converge much below ``~1e-6`` regardless of this
        setting. A ``RuntimeWarning`` is raised if ``newton_cg_max_iter`` is
        exhausted before convergence.
    """

    method: str                    = "Jones"    # "Normal" | "Pol" | "Jones" | "Jones_direct"
    optimizer: str                 = "newton"
    steps: int                     = 1
    alpha: float                   = 1.0
    beta: float                    = 0.005
    gamma: float                   = 0.0
    newton_chunk_size: int | None  = 64
    newton_cg_max_iter: int | None = None
    newton_cg_tol: float           = 1e-8

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "optimizer": self.optimizer,
            "steps": self.steps,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "newton_chunk_size": self.newton_chunk_size,
            "newton_cg_max_iter": self.newton_cg_max_iter,
            "newton_cg_tol": self.newton_cg_tol,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Factorization":
        return cls(**d)


@dataclass
class Config:
    """
    Top-level solver configuration for the RCWA pipeline.

    Pass an instance of this class to ``LayerSolver`` to control grid
    resolution, Fourier truncation, TVF factorization, and eigenvalue
    solver behaviour.

    Attributes
    ----------
    dtype : torch.dtype
        Floating-point precision for all computations.  Default
        ``torch.float32``; use ``torch.float64`` for higher accuracy.
    device : str or torch.device
        Target device, e.g. ``"cpu"`` or ``"cuda"``.  Default ``"cpu"``.
    nx : int
        Real-space grid resolution along the **a1** lattice direction
        (number of pixels).  Default ``128``.
    ny : int
        Real-space grid resolution along the **a2** lattice direction.
        Default ``128``.
    m : int
        Number of retained Fourier harmonics along **a1**.  The total
        harmonic count along a1 is ``2m + 1``.  Default ``12``.
    n : int
        Number of retained Fourier harmonics along **a2**.  Default ``12``.
    truncation : str
        Harmonic truncation scheme.  ``"circular"`` keeps harmonics inside
        an ellipse (smoother convergence); ``"rectangular"`` keeps all
        ``(2m+1)×(2n+1)`` harmonics.  Default ``"circular"``.
    factorization : Factorization or None
        TVF Li-factorization settings.  ``None`` disables TVF and uses the
        plain Laurent convolution rule.  Default ``Factorization()``.
    modesolver : str
        Mode-solving strategy for patterned layers.  One of ``"eig"`` (full
        eigendecomposition via :func:`eigsolver`) or ``"matexp"`` (matrix
        exponential of the first-order system via
        :mod:`~metarcwa.solver.layersolver.matexpsolver`, see
        ``docs/matrixexp.md``).  Default ``"eig"``.  The ``matexp_*``
        settings below apply only to ``"matexp"``; ``eigsolver_stable`` and
        ``checkpoint_eig`` apply only to ``"eig"`` (inert, not an error,
        under the other solver).  Note the cost model inverts relative to
        ``"eig"``: ``matexp`` makes ``LayerSolver.prepare`` cheaper (no
        eigendecomposition, just ``P``/``Q``) and ``smatrix``/``Solver.run``
        more expensive.
    eigsolver_stable : bool
        If ``True`` (default), use :class:`Eig` with Lorentzian-broadened
        gradients for stability near degenerate eigenvalues.  Set to
        ``False`` to use ``torch.linalg.eig`` directly (faster but
        gradients can be NaN near degeneracies).
    checkpoint_eig : bool
        If ``True``, gradient-checkpoint the patterned-layer eigendecomposition
        (:func:`eigsolver`) instead of keeping its saved-for-backward tensors
        alive for the whole solve. Cuts peak memory (the eigenvector stack is
        recomputed once per patterned layer during backward instead of held
        from construction to backward) at the cost of one extra eigensolver
        forward pass per patterned layer in backward. Default ``False`` —
        leave off for compute-bound / small-batch runs where the recompute
        cost isn't worth it.
    grazing_eps_reg : float
        Minimum imaginary part (infinitesimal material loss) added to every
        permittivity used in mode solving — media, homogeneous layers,
        patterned-layer grids, and the internal vacuum reference. Regularizes
        the exact-grazing degeneracy (``kz² → 0`` at normal-incidence or
        critical-angle points) where the ``Q`` operator becomes rank-deficient
        and the boundary S-matrix solve turns singular, most visibly at exactly
        the total-internal-reflection critical angle. Default ``1e-8``; set to
        ``0`` to disable (may raise ``torch.linalg.solve`` singular-matrix
        errors at exact grazing incidence, especially in ``float32``).
    matexp_slicing : bool
        Master on/off switch for slicing a ``"matexp"`` patterned layer into
        several thin sub-layers before exponentiating. A matrix exponential
        of the full first-order system has entries spanning
        ``exp(+-|lam|*k0*d)``; for thick layers / high harmonics these
        overflow and swamp the small entries that carry the physical S-matrix
        (docs/smatrix.md's stated reason for preferring S- over T-matrices).
        Slicing into ``n`` thin sub-layers keeps each ``expm`` argument
        bounded, converts each slice to an S-matrix immediately, and
        recombines via ``n-1`` Redheffer star products (:math:`O(\\log n)` via
        repeated squaring) — exact, not approximate. Default ``True``. Set to
        ``False`` to force a single unsliced ``expm`` (fast, but unstable for
        thick/high-harmonic layers — a ``RuntimeWarning`` is raised when the
        estimated exponent exceeds ``matexp_max_exponent``).
    matexp_slices : int or None
        Explicit slice count for ``"matexp"`` patterned layers, overriding
        automatic estimation. ``None`` (default) estimates ``n`` from
        ``k0*d*max|lam|`` and ``matexp_max_exponent`` (see
        :func:`~metarcwa.solver.layersolver.matexpsolver.slice_count`).
        Ignored when ``matexp_slicing=False``.
    matexp_max_slices : int
        Cap on the automatically estimated slice count (never applies to an
        explicit ``matexp_slices``). Default ``512``. A ``RuntimeWarning``
        is raised if the estimate exceeds this cap; the count is clamped to
        it, which may leave the per-slice exponent above
        ``matexp_max_exponent``.
    matexp_max_exponent : float or None
        Per-slice exponent budget ``max(k0*d*|lam|)`` used to estimate the
        automatic slice count. ``None`` (default) resolves to ``8.0`` for
        ``complex128`` (``dtype=torch.float64``, retained relative accuracy
        ``~ machine_eps * exp(2*8) ~ 2e-9``) or ``3.0`` for ``complex64``
        (``dtype=torch.float32``, ``~ 5e-5``). Lower is more conservative
        (more slices, more star products); ignored when
        ``matexp_slices`` is set or ``matexp_slicing=False``.
    """

    dtype:                torch.dtype         = torch.float32
    device:               torch.device        = "cpu"
    nx:                   int                 = 128
    ny:                   int                 = 128
    m:                    int                 = 12
    n:                    int                 = 12
    truncation:           str                 = "circular"    # "circular" | "rectangular"
    factorization:        Factorization|None  = field(default_factory=Factorization)
    modesolver:           str                 = "eig"         # "eig" | "matexp"
    eigsolver_stable:     bool                = True
    checkpoint_eig:       bool                = False
    grazing_eps_reg:      float               = 1e-8
    matexp_slicing:       bool                = True
    matexp_slices:        int | None          = None
    matexp_max_slices:    int                 = 512
    matexp_max_exponent:  float | None        = None

    _MODESOLVERS = ("eig", "matexp")

    def __post_init__(self) -> None:
        if not isinstance(self.device, torch.device):
            self.device = torch.device(self.device)

        if self.modesolver not in Config._MODESOLVERS:
            raise ValueError(
                f"modesolver={self.modesolver!r} not supported; must be one "
                f"of {Config._MODESOLVERS}."
            )

        # Harmonic truncation vs. real-space grid: the convolution matrix
        # (convolution.py) indexes eps_hat modulo (nx, ny). Once the harmonic
        # span 2*m+1 (or 2*n+1) exceeds the grid, distinct harmonics alias
        # onto the same Fourier bin and the convolution matrix becomes
        # exactly singular -- surfaces downstream as a cryptic
        # torch.linalg.inv/solve "singular matrix" error deep in the modal
        # solve. Catch it here instead, at Config construction time (this
        # also fires on every dataclasses.replace(...)).
        for label, m_max, grid in (("m", self.m, self.nx), ("n", self.n, self.ny)):
            span = 2 * m_max + 1
            if span > grid:
                other = "nx" if label == "m" else "ny"
                raise ValueError(
                    f"{other}={grid} too small for {label}={m_max}: truncation "
                    f"spans {label}=[-{m_max}, {m_max}] ({span} harmonics), which "
                    f"aliases modulo {other} and makes the convolution matrix "
                    f"singular; need {other} >= {span} (>= {4 * m_max + 1} to also "
                    "avoid Laurent-rule aliasing)."
                )
            if 4 * m_max + 1 > grid:
                other = "nx" if label == "m" else "ny"
                warnings.warn(
                    f"{other}={grid} is below the Laurent-rule sampling "
                    f"requirement for {label}={m_max} (need {other} >= "
                    f"{4 * m_max + 1}); Fourier coefficients up to order "
                    f"2*{label}={2 * m_max} will be aliased, degrading accuracy.",
                    RuntimeWarning,
                    stacklevel=2,
                )

    def to_dict(self) -> dict:
        return {
            "dtype": _DTYPE_TO_STR[self.dtype],
            "device": str(self.device),
            "nx": self.nx,
            "ny": self.ny,
            "m": self.m,
            "n": self.n,
            "truncation": self.truncation,
            "factorization": self.factorization.to_dict() if self.factorization is not None else None,
            "modesolver": self.modesolver,
            "eigsolver_stable": self.eigsolver_stable,
            "checkpoint_eig": self.checkpoint_eig,
            "grazing_eps_reg": self.grazing_eps_reg,
            "matexp_slicing": self.matexp_slicing,
            "matexp_slices": self.matexp_slices,
            "matexp_max_slices": self.matexp_max_slices,
            "matexp_max_exponent": self.matexp_max_exponent,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        d = dict(d)
        d["dtype"] = _STR_TO_DTYPE[d["dtype"]]
        d["device"] = torch.device(d["device"])
        if d.get("factorization") is not None:
            d["factorization"] = Factorization.from_dict(d["factorization"])
        return cls(**d)

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))
