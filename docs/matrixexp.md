# Matrix exponential

## Overview

The coupled first-order system (Eqs. (37)–(38) in [RCWA Core](rcwa_core.md))

$$
\begin{align}
\frac{1}{k_0}\frac{\partial s}{\partial z} = Pu, \qquad
\frac{1}{k_0}\frac{\partial u}{\partial z} = Qs,
\end{align}
$$

can be solved two ways. [Eigenvalue problem](eigenproblem.md) eliminates $u$, reduces to the algebraic eigenproblem $PQ\,W=W\Lambda$, and reconstructs $\psi(z)$ from the eigenbasis. This document derives the alternative: propagate the combined vector $\psi=(s,u)$ directly with a matrix exponential, without ever diagonalizing anything.

The appeal is what it *removes*, not new physics: no eigendecomposition (the dominant cost of the eig path, and a known GPU bottleneck — `torch.linalg.eig` synchronizes CPU↔GPU), and `torch.linalg.matrix_exp` has an exact built-in autograd formula, so none of the `Eig` Lorentzian-broadening machinery that [Eigenvalue problem](eigenproblem.md) needs for near-degenerate eigenvalues is required at all. `P` and `Q` themselves are unchanged — same `compute_isotropic` call, same TVF correction — only the propagation step differs.

---

## The system matrix

Stack $s=([S_x],[S_y])$ and $u=([U_x],[U_y])$ into $\psi=(s,u)$, a vector of length $4N_h$ ($N_h$ = retained harmonic count). Equations (37)–(38) become a single first-order linear ODE with a $z$-independent coefficient matrix:

$$
\begin{align}
\frac{\partial \psi}{\partial z} = k_0\,A\,\psi, \qquad
A = \begin{pmatrix} 0 & P \\ Q & 0 \end{pmatrix},
\end{align}
$$

where $P$, $Q$ are each $2N_h\times2N_h$ (identical to the operators used in [Eigenvalue problem](eigenproblem.md)), so $A$ is $4N_h\times4N_h$. Since $A$ does not depend on $z$, the solution over a distance $d$ is the matrix exponential

$$
\begin{align}
\psi(d) = T\,\psi(0), \qquad T = \exp\!\bigl(A\,k_0\,d\bigr).
\end{align}
$$

$T$ is the layer's **transfer matrix** in the field basis $(s,u)$ — it propagates $\psi$ directly, unlike the eigenbasis route which first changes to modal amplitudes $(c^+,c^-)$.

**Consistency check.** $A$'s eigenpairs are exactly $(\pm\lambda_i, (W_{\cdot i},\pm V_{\cdot i}))$ for the same $\lambda$, $W$, $V$ the eigenvalue-problem route computes ($\Omega^2=PQ$, $V=QW\Lambda^{-1/2}$): writing $\Phi=\bigl(\begin{smallmatrix}W&W\\V&-V\end{smallmatrix}\bigr)$ (the gap matrix, [Eigenvalue problem](eigenproblem.md)), $A\Phi=\Phi\,\mathrm{diag}(\lambda,-\lambda)$, so $T=\exp(Ak_0d)=\Phi\,\mathrm{diag}\bigl(e^{\lambda k_0 d}, e^{-\lambda k_0 d}\bigr)\Phi^{-1}$ — the two routes compute the same propagator, just via different means (diagonalize-then-exponentiate vs. exponentiate directly).

---

## From transfer matrix to S-matrix

[S-matrix algebra](smatrix.md) explains why an S-matrix is preferred over a T-matrix: a T-matrix's entries span $\exp(\pm|\lambda|k_0d)$, so for a layer with any appreciable evanescent content the large entries overflow and swamp the small ones, destroying exactly the information the S-matrix needs. That objection does not disappear here — it is the reason this route needs the slicing scheme below — but it does not rule out using $T$ as an *intermediate* quantity, converted to an S-matrix immediately, one thin slice at a time.

Both faces of a slice sit in the same vacuum background, so the field vector at each face is $\psi=\Phi_0\,(c^+,c^-)^\top$ with the vacuum gap matrix $\Phi_0=\bigl(\begin{smallmatrix}W_0&W_0\\V_0&-V_0\end{smallmatrix}\bigr)$ ($W_0=I$, $V_0$ the vacuum H-mode matrix). Substituting into $\psi(d)=T\,\psi(0)$ and grouping outgoing amplitudes $(c_L^-,c_R^+)$ on one side, incoming $(c_L^+,c_R^-)$ on the other, gives exactly the same *shape* of linear system `S_boundary` solves in [S-matrix algebra](smatrix.md) — just built from $T$ (partitioned to match the $(s,u)$ structure of $A$, so $T_{11},T_{12},T_{21},T_{22}$ below are $T$'s own $2N_h\times2N_h$ blocks) instead of plain continuity:

$$
\begin{align}
\underbrace{\begin{pmatrix} W_0 & -(T_{11}W_0-T_{12}V_0) \\ V_0 & -(T_{21}W_0-T_{22}V_0) \end{pmatrix}}_{\text{left}}
\begin{pmatrix} c_R^+ \\ c_L^- \end{pmatrix}
=
\underbrace{\begin{pmatrix} T_{11}W_0+T_{12}V_0 & -W_0 \\ T_{21}W_0+T_{22}V_0 & V_0 \end{pmatrix}}_{\text{right}}
\begin{pmatrix} c_L^+ \\ c_R^- \end{pmatrix}.
\end{align}
$$

One `Block2x2.solve` (`left.solve(right)`) gives $(c_R^+,c_L^-)$ in terms of $(c_L^+,c_R^-)$; swapping the two output rows puts it in the $(c_L^-,c_R^+)$ order the S-matrix convention (Eq. (1) in [S-matrix algebra](smatrix.md)) expects.

**Why not $M=\Phi_0^{-1}T\Phi_0$?** That is algebraically equivalent but numerically worse: at `dtype=torch.float32` it produced physically invalid $T>1$ on a real patterned-grating regression, and — the signature of a genuine conditioning problem rather than under-slicing — the error grew *worse*, not better, as the slice count increased. $\Phi_0$ mixes harmonics with very different magnitudes (evanescent harmonics can have large $|k_z|$), so inverting it directly is ill-conditioned exactly where `S_boundary` avoids the analogous risk by never inverting a monolithic gap matrix. The direct-solve form above has the same conditioning profile as `S_boundary` itself, because it *is* the same derivation. `tests/solver/test_matexpsolver.py::TestTransferToSmatrix` and the real-structure benchmark below both target this regression specifically.

**Sanity check.** For a vacuum layer, $\Phi_0$ diagonalizes $A$ exactly (it *is* the eigenbasis, with $\lambda=1j k_z$ the vacuum dispersion), so the system above collapses to $S_{11}=S_{22}=0$, $S_{12}=S_{21}=X_d$ with $X_d=\exp(\lambda k_0 d)$ — exactly $S_l$ from [S-matrix algebra](smatrix.md). `tests/solver/test_matexpsolver.py::TestTransferToSmatrix::test_vacuum_slab_matches_s_prop` checks this bit-for-bit.

---

## Stability: slicing

A single unsliced $T=\exp(Ak_0d)$ reintroduces exactly the instability an S-matrix exists to avoid: entries spanning $\exp(\pm|\lambda|k_0d)$ overflow (or, on some LAPACK/cuSOLVER backends, the subsequent `M22` solve simply reports the matrix as exactly singular) once $|\lambda|k_0d$ exceeds a few hundred — trivially reached at oblique incidence, high harmonic truncation, or ordinary layer thicknesses in the evanescent regime. The retained relative accuracy of the whole computation is bounded by

$$
\varepsilon_\text{rel} \sim \varepsilon_\text{machine}\cdot e^{2\cdot\max|\lambda|k_0d},
$$

so the exponent must stay bounded, not just finite.

The fix keeps the transfer-matrix route intact: split the layer into $n$ identical thin sub-layers of thickness $d/n$, exponentiate and convert **one** slice, and recombine via the Redheffer star product:

$$
S_\text{layer} = \underbrace{S_\text{slice}\star S_\text{slice}\star\cdots\star S_\text{slice}}_{n\text{ times}}.
$$

Every slice is a genuine vacuum-embedded S-matrix (the derivation above holds for any thickness), so this recombination is **exact**, not an approximation — shrinking $d/n$ only shrinks the per-slice exponent, and $S_\text{slice}$ is exponentiated once and reused $n$ times via repeated squaring ($O(\log n)$ star products, since every slice is identical), not recomputed $n$ times.

**Choosing $n$.** No eigenvalues are available (that's the point of this solver), so $n$ is sized from a cheap upper bound on the modal exponent. For the isotropic system $\lambda^2\approx k_x^2+k_y^2-\varepsilon$, so

$$
\max|\lambda| \lesssim \sqrt{\max_h(k_x^2+k_y^2) + \max|\varepsilon|}, \qquad
n = \left\lceil \frac{k_0\,d\,\max|\lambda|}{\text{budget}} \right\rceil,
$$

reduced to a detached scalar (`.max()`, `.detach()`) since $n$ controls a Python loop count and cannot vary per batch element. Gradients through the resulting S-matrix are exact for any fixed $n$ — only the *choice* of $n$ is non-differentiable, which is correct, not approximate (a design variable crossing the threshold where $n$ changes moves to a different, equally valid computation graph rather than perturbing an otherwise-smooth function).

The default budget is $8.0$ for `complex128` ($\varepsilon_\text{rel}\sim2\times10^{-9}$) or $3.0$ for `complex64` ($\varepsilon_\text{rel}\sim5\times10^{-5}$). `Config` exposes full control: `matexp_slicing` (master on/off), `matexp_slices` (explicit override, wins over estimation), `matexp_max_slices` (cap on the automatic estimate), `matexp_max_exponent` (the budget itself).

---

## Cost model

Unlike the eig path, where `LayerSolver.prepare()` is the expensive step (eigendecomposition) and `smatrix()` is cheap (pure algebra), the matexp path **inverts** this: `prepare()` only needs $P$, $Q$ (no eigendecomposition — the cheap half of `compute_isotropic` + `eigsolver`), and the matrix exponential moves into `smatrix()`, because it depends on `thickness` and must stay responsive to `dataclasses.replace(op, thickness=...)` — computing it in `prepare()` would silently break the documented late-binding of thickness (a thickness-only sweep or optimization step must not require re-preparing).

Memory: the system matrix $A$ is $4N_h\times4N_h$ densified once per slice-exponentiation, versus $2N_h\times2N_h$ for $\Omega^2=PQ$ on the eig path — roughly $4\times$ the dense footprint per patterned layer.

---

## Implementation

Implemented in `src/metarcwa/solver/layersolver/matexpsolver.py`. All mode matrices are `Block2x2` structured operators (see [Block matrices](blockmatrix.md)).

```python
def system_matrix(P, Q):              # A = [[0, P], [Q, 0]]
    Z = P.zeros_like()
    return Block2x2(Z, P, Q, Z)

def transfer_matrix(A, Nh, k0, d):     # T = expm(A * k0 * d)
    A_dense = A.to_dense(Nh)                      # [..., 4Nh, 4Nh]
    T_dense = torch.linalg.matrix_exp(A_dense * (k0 * d)[..., None, None])
    return Block2x2.from_dense(T_dense, A)

def transfer_to_smatrix(T, background):        # never forms Phi0^-1 (see above)
    W0, V0 = background.W0, background.V0
    TW0_s, TW0_u = T.a @ W0, T.c @ W0           # T11 W0, T21 W0
    TV0_s, TV0_u = T.b @ V0, T.d @ V0           # T12 V0, T22 V0
    left  = Block2x2(W0, -(TW0_s - TV0_s), V0, -(TW0_u - TV0_u))
    right = Block2x2(TW0_s + TV0_s, -W0, TW0_u + TV0_u, V0)
    M = left.solve(right)                       # [c_R+; c_L-] = M [c_L+; c_R-]
    return Block2x2(M.c, M.d, M.a, M.b)         # swap rows -> [c_L-; c_R+]

def star_power(S, n):                  # S composed with itself n times, O(log n)
    result, base = None, S
    while n:
        if n & 1:
            result = base if result is None else result.star(base)
        n >>= 1
        if n:
            base = base.star(base)
    return result
```

`Block2x2.from_dense` (the inverse of `.to_dense()`) re-embeds a plain dense tensor into the `Block2x2` tree, matching a template's nesting — the counterpart operation `matrix_exp` needs, since it has no structured/batched-sparse form and must densify.

`star_power` deliberately does **not** seed from `Block2x2.star_identity()`: that returns `SCALAR` leaves, which cannot compose with `S`'s `DENSE`-leaf structure via `.star()`. It accumulates from the first set bit of `n` instead.

`TransferOperator` (the `LayerOperator` implementation for this path — `LayerOperator` itself is a structural contract, not one concrete type, satisfied by both `ModalOperator` (the eig path) and `TransferOperator`) carries `P`, `Q`, the harmonic count, a detached `lam_bound`, `Config`, and `thickness`; its `smatrix()` runs the slice → exponentiate → convert → `star_power` pipeline above, and its `transfer(background, z)` — the propagator $\psi(0)\to\psi(z)$ needed for interior-field evaluation — is `transfer_matrix` applied directly with no slicing (intended for single-depth evaluation, not cascading; large $z$ faces the same conditioning limits as an unsliced `smatrix()`).

---

## Real-structure benchmark

Measured, not assumed: `examples/2d_grating.ipynb`'s stack (ellipse-patterned metasurface, $m=n=8$, $128\times128$ real-space grid, 61-point wavelength sweep 300–900 nm), solved with `modesolver="eig"` vs. `"matexp"` (default slicing), reflectance/transmittance compared for both polarizations:

| device | dtype | eig | matexp | max\|ΔR\| | max\|ΔT\| |
|---|---|---|---|---|---|
| CPU | `float64` | 13.7 s | 13.1 s | $2\times10^{-13}$ | $1\times10^{-12}$ |
| CPU | `float32` | 5.1 s | 5.3 s | $5\times10^{-5}$ | $8\times10^{-5}$ |
| CUDA | `float32` | 2.79 s (peak 1.76 GB) | **0.36 s** (peak 4.07 GB) | $6\times10^{-5}$ | $1\times10^{-4}$ |

The premise this solver exists for — avoiding `torch.linalg.eig`'s CPU↔GPU synchronization — holds up: **7.8× faster on GPU**, at the cost of ~2.3× peak memory (consistent with the $4N_h\times4N_h$ vs. $2N_h\times2N_h$ dense footprint above) and no CPU advantage (CPU `eig` isn't bottlenecked by the same synchronization, so the two paths land close together there). Accuracy matches the analysis above: near machine precision at `float64`, ~$10^{-4}$ at `float32` — both easily adequate for design/optimization loops, in line with `eigsolver`'s own `float32` behavior.

---

## Known limitations

- **Anisotropic media are unsupported**, matching `eigsolver`'s current coverage — both solvers dispatch from the same `compute_isotropic` path.
- **Accuracy is bounded** by $\varepsilon_\text{machine}\cdot e^{2\cdot\text{budget}}$; `float32` is materially worse here than on the eig path, where accuracy is set by the eigendecomposition's own conditioning rather than an explicit, user-tunable exponent budget. Measured at ~$10^{-4}$ on a real structure (above) — fine for design work, but prefer `float64` for tight-tolerance validation.
- **`Solver.run()` is no longer cheap** under `modesolver="matexp"` — a thickness sweep or a single-layer optimization step now pays one (sliced) matrix exponential per evaluation, instead of reusing a cached eigendecomposition and only re-cascading star products.
- **Peak memory is higher** (~2.3× measured above) — the $4N_h\times4N_h$ system matrix, densified once per slice-exponentiation, dominates.
- **`transfer()` is not yet wired into `Observables`** — the primitive exists and is cross-validated against the eig path's own `transfer()` (`tests/solver/test_matexpsolver.py::TestFieldParity`), but interior-field reconstruction itself is future work (`FieldSolution` in `solver/base.py`, currently unimplemented).
