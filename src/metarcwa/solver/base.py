# metarcwa/solver/base.py
"""
Solver — top-level RCWA solver
==============================

Binds a :class:`~metarcwa.model.base.Model` and a :class:`Config` into a
ready-to-run simulation.  At construction the model is moved to the requested
device/dtype, harmonics and the optional TVF are pre-computed, and a
:class:`LayerSolver` is initialised.  The single :meth:`Solver.solve` method
assembles the full-stack S-matrix via Redheffer star products.
"""

from metarcwa.model.base import Model
from metarcwa.solver.layersolver.base import LayerSolver
from metarcwa.solver.tvf import TVF
from metarcwa.solver.config import Config
from metarcwa.solver.harmonics import compute_kxy, harmonic_index_map
from metarcwa.solver.blockmatrix import Block2x2


class Solver:
    """Top-level RCWA solver.

    Holds a fully-resolved model snapshot and a pre-initialised
    :class:`LayerSolver`.  Constructing a ``Solver`` is the expensive step
    (device transfer, harmonic pre-computation, TVF setup); calling
    :meth:`solve` is then cheap.

    Attributes
    ----------
    model : Model
        The simulation model, cast to the dtype/device given in ``config``.
    config : Config
        Solver hyperparameters.
    layersolver : LayerSolver
        Pre-initialised layer-level S-matrix dispatcher.
    """

    def __init__(self, model: Model, config: Config) -> None:
        """
        Parameters
        ----------
        model : Model
            Stack + source description.  Moved to ``config.dtype`` /
            ``config.device`` in-place.
        config : Config
            Solver hyperparameters (grid resolution, harmonic truncation,
            TVF factorization, eigensolver settings).
        """
        self.model = model.to(dtype=config.dtype, device=config.device)
        self.config = config

        self.model_spec = self.model.spec(config.nx, config.ny)

        m_flat, n_flat = harmonic_index_map(
            config.m, config.n, config.truncation == "circular", config.device
        )
        kx, ky = compute_kxy(
            self.model_spec.kx0, self.model_spec.ky0,
            self.model_spec.a1,  self.model_spec.a2,
            m_flat, n_flat,
        )

        if config.factorization is not None:
            f = config.factorization
            tvf = TVF(self.model_spec.a1, self.model_spec.a2,
                      config.m, config.n,
                      f.method, f.optimizer,
                      f.alpha, f.beta, f.gamma, f.steps)
        else:
            tvf = None

        self.layersolver = LayerSolver(
            config, self.model_spec.wavelength, kx, ky, m_flat, n_flat, tvf
        )

    def solve(self) -> Block2x2:
        """Compute the full-stack S-matrix.

        Assembles the S-matrix by star-multiplying the incidence boundary,
        every finite layer in order, and the transmission boundary.

        Returns
        -------
        S : Block2x2
            Full-stack scattering matrix.  Off-diagonal blocks carry
            transmission amplitudes; diagonal blocks carry reflection.
        """
        S = self.layersolver.solve(self.model_spec.incidence, left=True)
        for layer in self.model_spec.layers:
            S = S.star(self.layersolver.solve(layer))
        S = S.star(self.layersolver.solve(self.model_spec.transmission, left=False))
        return S
