# metarcwa/solver/base.py
"""
Solver — top-level RCWA solver
==============================

Binds a :class:`~metarcwa.model.base.Model` and a :class:`Config` into a
ready-to-run simulation.  At construction the model is moved to the requested
device/dtype, harmonics and the optional TVF are pre-computed, a
:class:`LayerSolver` is initialised, and every stack element's modal
eigenproblem is solved once via :meth:`LayerSolver.prepare` — this is what
makes construction the expensive step. The single :meth:`Solver.solve`
method is then genuinely cheap: pure Redheffer star-product composition of
the precomputed :class:`~metarcwa.solver.layersolver.base.LayerOperator`
objects, with no TVF, convolution, or eigendecomposition work.

Because the pattern/geometry is resolved once at construction
(``model.spec()`` in ``__init__``), an inverse-design loop must construct a
new ``Solver`` every optimization step whenever the pattern changes — this
was already required before this precompute split, so the workflow is
unchanged. Thickness gradients/updates are the one exception: operators hold
a *reference* to each layer's thickness tensor and read it inside
:meth:`solve`, so autograd through thickness (and in-place ``nn.Parameter``
thickness updates) work without rebuilding the ``Solver``.
"""

from metarcwa.model.base import Model
from metarcwa.solver.layersolver.base import LayerSolver
from metarcwa.solver.tvf import TVF
from metarcwa.solver.config import Config
from metarcwa.solver.harmonics import compute_kxy, harmonic_index_map
from metarcwa.solver.blockmatrix import Block2x2


class Solver:
    """Top-level RCWA solver.

    Holds a fully-resolved model snapshot, a pre-initialised
    :class:`LayerSolver`, and the precomputed
    :class:`~metarcwa.solver.layersolver.base.LayerOperator` for every stack
    element (incidence boundary, each finite layer, transmission boundary).
    Constructing a ``Solver`` is the expensive step (device transfer,
    harmonic pre-computation, TVF setup, and one modal eigensolve per
    element); calling :meth:`solve` is then genuinely cheap — pure
    Redheffer star-product composition, no eigendecomposition.

    Attributes
    ----------
    model : Model
        The simulation model, cast to the dtype/device given in ``config``.
    config : Config
        Solver hyperparameters.
    layersolver : LayerSolver
        Pre-initialised layer-level modal solver / S-matrix assembler.
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

        # Solve the modal eigenproblem for every stack element once, here.
        # solve() then only does cheap Redheffer star-product composition.
        self._ops = [
            self.layersolver.prepare(self.model_spec.incidence),
            *(self.layersolver.prepare(layer) for layer in self.model_spec.layers),
            self.layersolver.prepare(self.model_spec.transmission),
        ]

    def solve(self) -> Block2x2:
        """Compute the full-stack S-matrix.

        Assembles the S-matrix by star-multiplying the incidence boundary,
        every finite layer in order, and the transmission boundary, using
        the :class:`LayerOperator` objects precomputed in ``__init__``.

        Returns
        -------
        S : Block2x2
            Full-stack scattering matrix.  Off-diagonal blocks carry
            transmission amplitudes; diagonal blocks carry reflection.
        """
        ls = self.layersolver
        S = ls.smatrix(self._ops[0], left=True)
        for op in self._ops[1:-1]:
            S = S.star(ls.smatrix(op))
        return S.star(ls.smatrix(self._ops[-1], left=False))
