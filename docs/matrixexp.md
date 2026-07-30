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

Both faces of a slice sit in the same **reference medium** — some homogeneous medium, not necessarily vacuum (see "Accuracy and conditioning" below for why a per-layer *gap medium* is used by default) — so the field vector at each face is $\psi=\Phi\,(c^+,c^-)^\top$ with that medium's gap matrix $\Phi=\bigl(\begin{smallmatrix}W&W\\V&-V\end{smallmatrix}\bigr)$ ($W=I$ for any isotropic homogeneous medium, $V$ its H-mode matrix). Substituting into $\psi(d)=T\,\psi(0)$ and grouping outgoing amplitudes $(c_L^-,c_R^+)$ on one side, incoming $(c_L^+,c_R^-)$ on the other, gives exactly the same *shape* of linear system `S_boundary` solves in [S-matrix algebra](smatrix.md) — just built from $T$ (partitioned to match the $(s,u)$ structure of $A$, so $T_{11},T_{12},T_{21},T_{22}$ below are $T$'s own $2N_h\times2N_h$ blocks) instead of plain continuity:

$$
\begin{align}
\underbrace{\begin{pmatrix} W & -(T_{11}W-T_{12}V) \\ V & -(T_{21}W-T_{22}V) \end{pmatrix}}_{\text{left}}
\begin{pmatrix} c_R^+ \\ c_L^- \end{pmatrix}
=
\underbrace{\begin{pmatrix} T_{11}W+T_{12}V & -W \\ T_{21}W+T_{22}V & V \end{pmatrix}}_{\text{right}}
\begin{pmatrix} c_L^+ \\ c_R^- \end{pmatrix}.
\end{align}
$$

One `Block2x2.solve` (`left.solve(right)`) gives $(c_R^+,c_L^-)$ in terms of $(c_L^+,c_R^-)$; swapping the two output rows puts it in the $(c_L^-,c_R^+)$ order the S-matrix convention (Eq. (1) in [S-matrix algebra](smatrix.md)) expects.

**Numerical form.** The direct-solve form above avoids materializing $\Phi^{-1}$. Forming $M=\Phi^{-1}T\Phi$ explicitly is algebraically equivalent but ill-conditioned at `dtype=torch.float32`: $\Phi$ mixes harmonics of very different magnitude (evanescent harmonics can have large $|k_z|$), so its inverse is poorly scaled. On a real patterned-grating regression the naive route produced a physically invalid $T>1$ that grew worse with more slices, the signature of a conditioning problem rather than under-slicing. The solve-based form shares `S_boundary`'s conditioning instead, since it's the same derivation. Regression: `tests/solver/test_matexpsolver.py::TestTransferToSmatrix`.

**Sanity check.** For a layer matching the reference medium exactly, $\Phi$ diagonalizes $A$ exactly (it *is* the eigenbasis, with $\lambda=1j k_z$ that medium's own dispersion), so the system above collapses to $S_{11}=S_{22}=0$, $S_{12}=S_{21}=X_d$ with $X_d=\exp(\lambda k_0 d)$ — exactly $S_l$ from [S-matrix algebra](smatrix.md). `tests/solver/test_matexpsolver.py::TestTransferToSmatrix::test_vacuum_slab_matches_s_prop` checks this bit-for-bit for the vacuum case.

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

Every slice is exponentiated once and reused $n$ times via repeated squaring ($O(\log n)$ star products, not $n-1$), and the recombination is exact for any thickness and any homogeneous reference medium — shrinking $d/n$ only shrinks the per-slice exponent.

The reference medium for this internal slicing is a per-layer **gap medium** (`Config.matexp_gap`, default the layer's own mean permittivity), not the stack's vacuum background. See "Accuracy and conditioning" below. $S_\text{layer}$ is transitioned back to the vacuum background afterward by two boundary S-matrices, which are exact for the same reason `S_boundary` is.

**Choosing $n$.** No eigenvalues are available, so $n$ is sized from a cheap upper bound on the modal exponent. For the isotropic system $\lambda^2\approx k_x^2+k_y^2-\varepsilon$:

$$
\max|\lambda| \lesssim \sqrt{\max_h(k_x^2+k_y^2) + \max|\varepsilon|}, \qquad
n = \left\lceil \frac{k_0\,d\,\max|\lambda|}{\text{budget}} \right\rceil.
$$

$n$ is a detached scalar (`.max()`, `.detach()`): it controls a Python loop count, so it can't vary per batch element or enter autograd. Gradients through the S-matrix are still exact for any fixed $n$; only the choice of $n$ itself is non-differentiable.

The default budget is $8.0$ for `complex128` ($\varepsilon_\text{rel}\sim2\times10^{-9}$) or $3.0$ for `complex64` ($\varepsilon_\text{rel}\sim5\times10^{-5}$). `Config` exposes full control: `matexp_slicing` (master on/off), `matexp_slices` (explicit override, wins over estimation), `matexp_max_slices` (cap on the automatic estimate), `matexp_max_exponent` (the budget itself).

---

## Accuracy and conditioning

The exponent budget bounds `matrix_exp`'s own error, not the error of the whole `"matexp"` solve. Measured on `examples/compare_solvers.ipynb` (ellipse-patterned metasurface, $m=n=10$, $N_h=317$, 200 nm layer, $n=3.8$ in air, 300–900 nm sweep), `float32` `matexp` with vacuum embedding and the automatic slice count reached $\max|\Delta R|=7\times10^{-3}$ against an `eig`/`float64` reference, 20× worse than `eig`/`float32`'s own $3\times10^{-4}$. The error appeared only at isolated wavelengths and depended on `matexp_slices` alone, with no change to the physical geometry.

**Mechanism.** Each slice's S-matrix embeds a thin sub-slab of the patterned layer's material — a fictitious construct valid only for the whole layer, not for an arbitrary thickness $d/n$ of it. Under vacuum embedding, harmonics evanescent in vacuum but propagating inside a high-index sub-slab produce a sharp, wavelength-dependent resonance at specific sub-slab thicknesses. Scanning sub-slab thickness in isolation at 560 nm (`float64`, exact arithmetic) shows the spike:

| $t$ (nm) | 20 | 22 | 23 | 24 | **25** | 26 | 27 | 28 | 30 |
|---|---|---|---|---|---|---|---|---|---|
| $\max|S_\text{slice}|$ | 1.6 | 2.0 | 2.5 | 3.8 | **28** | 3.9 | 2.0 | 1.4 | 1.0 |

At $t=25\,\text{nm}=d/8$, the Redheffer star product's internal solve — $(I-S_{22}^BS_{11}^A)$ in `Block2x2.star` — has condition number $\sim2\times10^5$, independent of $n$ or `dtype`. That costs $\sim5\times10^{-11}$ relative error at `complex128` and a few percent at `complex64`.

**Dyadic slice counts.** `star_power` composes $n$ slices by repeated squaring, visiting intermediate thicknesses $(d/n)\cdot2^k$. For even $n$, several of those are exact dyadic fractions of $d$ (e.g. $n=8,16,32,\dots$ all pass through $d/8=25\,\text{nm}$ on this structure), which can land on the resonance above. An odd $n>1$ never revisits an exact dyadic fraction of $d$. Measured error in $R$ at 560 nm vs. $n$ (`float32`, vacuum embedding; `float64` stays below $10^{-11}$ for every $n$ shown):

| $n$ | 1 | 2 | 3 | 4 | 5 | 8 | 11 | 16 | 32 | 48 | 64 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| err | 2e-1 | 4e-2 | 3e-5 | 6e-4 | 1e-6 | 1e-1 | 3e-5 | **1.2** | 1e-1 | 4e-5 | 4e-2 |

Every power of two $\geq8$ is bad; every odd $n$ shown lands at `eig`/`float32` parity ($\sim10^{-5}$). $n=1$ fails for a different reason — an unsliced exponent this large overflows.

**Mitigations applied by default, independent of `matexp_gap`:**

- `slice_count` bumps its automatic estimate to the next odd integer whenever it lands even.
- `star_power` checks each intermediate S-matrix's largest-magnitude entry against a `dtype`-aware threshold ($50$ for `complex64`, $10^4$ for `complex128`) and raises a `RuntimeWarning` if exceeded. This catches gross under-slicing but not every case — a resonance narrow enough relative to the accumulated error can corrupt the result without the magnitude leaving a plausible range. If it fires, try a different `matexp_slices` or `matexp_gap`, or `dtype=torch.float64`.
- `Block2x2.star` solves rather than inverts (`(I-P)^{-1}@rhs` via `Entry.solve`, never a materialized `(I-P)^{-1}`), which is better conditioned near a near-pole and benefits every star product in the solver.

**Gap medium.** These mitigations reduce the chance of hitting the resonance; they don't change why it exists, which is that every slice is referenced to vacuum regardless of the layer's actual index. `TransferOperator.smatrix()` instead references each slice to a per-layer gap medium (`Config.matexp_gap`), then transitions back to the stack's vacuum background with two boundary S-matrices at the layer's outer faces — the same $S_\text{in}\star S_\text{prop}\star S_\text{out}$ sandwich [S-matrix algebra](smatrix.md) uses on the `"eig"` path, with the gap medium standing in for the layer's own eigenbasis. The mirror trick used for the second boundary,

$$
S_\text{out} = J \cdot S_\text{in} \cdot J = \begin{pmatrix}(S_\text{in})_{22} & (S_\text{in})_{21}\\(S_\text{in})_{12} & (S_\text{in})_{11}\end{pmatrix}, \qquad J=\begin{pmatrix}0&I\\I&0\end{pmatrix},
$$

is an identity of `S_boundary` for any two media, not only equal ones. The whole sandwich is exact for any gap medium — the choice affects conditioning, not the answer. `tests/solver/test_matexpsolver.py::TestGapMediumInvariance` checks this directly (three different gap media, same `smatrix()` output to $\sim10^{-9}$ relative).

`Config.matexp_gap` values:

- **`"mean"`** (default) — the layer's spatial-mean permittivity: the DC/harmonic-0 Fourier coefficient, and the fill-fraction-weighted average of the solid/void materials.
- **`"max"`** — the componentwise max (real and imaginary parts separately) over the layer's permittivity grid.
- **`"vacuum"`** — plain vacuum, the behavior before this option existed.

Re-measured on the same benchmark (`float32`, full spectrum, $\max|\Delta R|$ against `eig`/`float64`; `eig`/`float32`'s own error is $2.96\times10^{-4}$):

| `matexp_gap` | $\max|\Delta R|$ | rms $|\Delta R|$ | resonance-guard warnings |
|---|---|---|---|
| `"vacuum"` | $1.52\times10^{-3}$ | $2.41\times10^{-4}$ | 1 |
| `"mean"` | $6.68\times10^{-4}$ | $9.98\times10^{-5}$ | 0 |
| `"max"` | $3.12\times10^{-4}$ | $5.47\times10^{-5}$ | 0 |

Both `"mean"` and `"max"` land at or inside `eig`/`float32`'s own error, with no resonance-guard warnings over the full sweep. The sub-slab thickness scan at 600 nm (this benchmark's worst `"vacuum"` wavelength) shows the same thing directly: `"vacuum"` peaks at $4.2$ at $t=30\,\text{nm}$, while `"mean"`/`"max"` stay flat and below $1.0$ across the same range:

| $t$ (nm) | 5 | 15 | 22 | 25 | **30** | 35 | 40 | 50 |
|---|---|---|---|---|---|---|---|---|
| `"vacuum"` | 1.03 | 1.11 | 1.28 | 1.49 | **4.19** | 1.29 | 0.89 | 0.93 |
| `"mean"` | 1.00 | 0.97 | 0.94 | 0.93 | 0.91 | 0.89 | 0.87 | 0.83 |
| `"max"` | 1.00 | 0.97 | 0.94 | 0.92 | 0.89 | 0.86 | 0.82 | 0.77 |

`"mean"` reduces the index contrast that creates the resonance but doesn't remove it: an inhomogeneous layer has regions with $\varepsilon>\varepsilon_\text{gap}$ that can still guide relative to the mean. `"max"` references above every local index in the layer, so nothing can guide relative to it — the measurements above are consistent with this removing the mechanism entirely, though that has only been checked on the structures here, not proven in general. The dyadic-ladder and magnitude-guard mitigations stay in place regardless of `matexp_gap`.

**Cost.** The two extra boundary S-matrices use `DIAG`/`SCALAR` mode matrices (any isotropic homogeneous medium's `W=I`, `V` diagonal), so they hit `Block2x2.solve`'s cheap per-harmonic path ($O(N_h)$, not $O(N_h^3)$), and the `star()` calls that fold them in are cheaper than a typical `star_power` squaring step. Measured overhead on the benchmark above was within run-to-run noise (~0.3%).

---

## Cost model

Unlike the eig path, where `LayerSolver.prepare()` is the expensive step (eigendecomposition) and `smatrix()` is cheap (pure algebra), the matexp path **inverts** this: `prepare()` only needs $P$, $Q$ (no eigendecomposition — the cheap half of `compute_isotropic` + `eigsolver`), and the matrix exponential moves into `smatrix()`, because it depends on `thickness` and must stay responsive to `dataclasses.replace(op, thickness=...)` — computing it in `prepare()` would silently break the documented late-binding of thickness (a thickness-only sweep or optimization step must not require re-preparing).

Memory: the system matrix $A$ is $4N_h\times4N_h$ densified once per slice-exponentiation, versus $2N_h\times2N_h$ for $\Omega^2=PQ$ on the eig path — roughly $4\times$ the dense footprint per patterned layer.

**`float64` GPU throughput.** Consumer GPUs (measured: RTX 4090) run `complex128` GEMM/`matrix_exp` at roughly $1/40$ their `complex64` throughput (`torch.linalg.matrix_exp` measured $36\times$ slower, plain `X@X` $42\times$) — `float64` isn't a first-class datapath on that silicon. `torch.linalg.eig` is latency/reduction-bound (`geev`) rather than GEMM-bound, so it only slows $\sim4\times$ going to `complex128`. The eigendecomposition matexp removes is the least `dtype`-sensitive part of the eig path; the `matrix_exp` it adds is the most sensitive part of its own. Same 20-wavelength benchmark, split by `Solver.__init__`/`run()`:

| dtype | solver | init | run | total | slices |
|---|---|---|---|---|---|
| `float32` | eig | 3.24 s | 0.13 s | 3.37 s | – |
| `float32` | matexp | 0.77 s | 0.29 s | **1.06 s** | 32 |
| `float64` | eig | 6.57 s | 1.74 s | 8.30 s | – |
| `float64` | matexp | 0.80 s | 6.21 s | **7.01 s** | 12 |

matexp's $3\times$ `float32` advantage nearly vanishes at `float64` on this GPU (the larger `float64` exponent budget, 8.0 vs 3.0, needing fewer slices — 12 vs 32 — is the only reason it doesn't reverse outright). On a datacenter GPU with a 1:2 `float64:float32` throughput ratio (A100/H100) rather than a consumer card's $\sim$1:40–1:64, matexp should keep its advantage at `float64` too — this is a property of the deployment GPU, not of the algorithm.

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

def transfer_to_smatrix(T, background):        # never forms Phi^-1 (see above);
    W, V = background.W0, background.V0        # "background" here = the gap medium
    TW_s, TW_u = T.a @ W, T.c @ W               # T11 W, T21 W
    TV_s, TV_u = T.b @ V, T.d @ V               # T12 V, T22 V
    left  = Block2x2(W, -(TW_s - TV_s), V, -(TW_u - TV_u))
    right = Block2x2(TW_s + TV_s, -W, TW_u + TV_u, V)
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

# TransferOperator.smatrix(): slice, exponentiate, convert to an S-matrix
# referenced to the gap medium, recombine, then sandwich back to vacuum --
# mirrors S_boundary(vacuum,layer) . S_prop . mirror(...) on the eig path,
# with S_gap (a star_power-composed multi-slice S-matrix) standing in for
# S_prop, and the gap medium standing in for the layer's own eigenbasis.
def smatrix(self, background):
    n = slice_count(self.lam_bound, k0, self.thickness, self.config)
    A = system_matrix(self.P, self.Q)
    T_slice = transfer_matrix(A, self.Nh, k0, self.thickness / n)
    S_slice = transfer_to_smatrix(T_slice, self.gap_background)   # not `background`
    S_gap = star_power(S_slice, n)

    S_in = S_boundary(background.W0, background.V0,
                       self.gap_background.W0, self.gap_background.V0)
    S_out = Block2x2(S_in.d, S_in.c, S_in.b, S_in.a)   # mirror trick
    return S_in.star(S_gap).star(S_out)
```

`Block2x2.from_dense` (the inverse of `.to_dense()`) re-embeds a plain dense tensor into the `Block2x2` tree, matching a template's nesting — the counterpart operation `matrix_exp` needs, since it has no structured/batched-sparse form and must densify.

`star_power` doesn't seed from `Block2x2.star_identity()`, since that returns `SCALAR` leaves that can't compose with `S`'s `DENSE`-leaf structure. It accumulates from the first set bit of `n` instead.

The snippet omits two pieces covered in "Accuracy and conditioning": `slice_count`'s odd-`n` nudge, and `star_power`'s magnitude check on each intermediate. The mirror trick used for `S_out` is also derived there.

`TransferOperator` is the `LayerOperator` implementation for this path (`ModalOperator` is the other, for `"eig"`). It carries `P`, `Q`, the harmonic count, a detached `lam_bound`, `Config`, `gap_background`, and `thickness`. `transfer(background, z)` — the propagator $\psi(0)\to\psi(z)$ for interior-field evaluation — is `transfer_matrix` applied directly with no slicing and no gap-medium embedding, since it returns a field-basis propagator rather than an S-matrix. It's meant for single-depth evaluation, not cascading; large $z$ hits the same conditioning limits as an unsliced `smatrix()`.

---

## Real-structure benchmark

Measured on `examples/2d_grating.ipynb`'s stack (ellipse-patterned metasurface, $m=n=8$, $128\times128$ real-space grid, 61-point wavelength sweep 300–900 nm), `modesolver="eig"` vs. `"matexp"` with default slicing:

| device | dtype | eig | matexp | max\|ΔR\| | max\|ΔT\| |
|---|---|---|---|---|---|
| CPU | `float64` | 13.7 s | 13.1 s | $2\times10^{-13}$ | $1\times10^{-12}$ |
| CPU | `float32` | 5.1 s | 5.3 s | $5\times10^{-5}$ | $8\times10^{-5}$ |
| CUDA | `float32` | 2.79 s (peak 1.76 GB) | **0.36 s** (peak 4.07 GB) | $6\times10^{-5}$ | $1\times10^{-4}$ |

Avoiding `torch.linalg.eig`'s CPU↔GPU synchronization gives a 7.8× speedup on GPU, at the cost of ~2.3× peak memory (consistent with the $4N_h\times4N_h$ vs. $2N_h\times2N_h$ dense footprint above). CPU `eig` isn't bottlenecked by the same synchronization, so the two paths land close together there. Accuracy is near machine precision at `float64` and ~$10^{-4}$ at `float32`, in line with `eigsolver`'s own `float32` behavior.

This table predates `Config.matexp_gap` (still vacuum embedding). The `compare_solvers.ipynb` measurements above show the timing conclusions hold either way; `float32` accuracy on this structure is not re-measured under the current `"mean"` default.

---

## Known limitations

- **Anisotropic media are unsupported**, matching `eigsolver`'s current coverage — both solvers dispatch from the same `compute_isotropic` path.
- **`float32` accuracy is dominated by sub-slab resonance conditioning, not the exponent budget** — see "Accuracy and conditioning". `Config.matexp_gap` addresses this; `"max"` measured at `eig`/`float32` parity on the one structure tested. Prefer `float64` for tight-tolerance validation.
- **`Solver.run()` is no longer cheap** under `modesolver="matexp"` — a thickness sweep or a single-layer optimization step now pays one (sliced) matrix exponential per evaluation, instead of reusing a cached eigendecomposition and only re-cascading star products.
- **Peak memory is higher** (~2.3× measured above) — the $4N_h\times4N_h$ system matrix, densified once per slice-exponentiation, dominates.
- **`transfer()` is not yet wired into `Observables`** — the primitive exists and is cross-validated against the eig path's own `transfer()` (`tests/solver/test_matexpsolver.py::TestFieldParity`), but interior-field reconstruction itself is future work (`FieldSolution` in `solver/base.py`, currently unimplemented).
