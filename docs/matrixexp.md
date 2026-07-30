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

## Accuracy and conditioning

The exponent budget above bounds `matrix_exp`'s *own* error. It does not bound the error of the whole `"matexp"` solve — measured on `examples/compare_solvers.ipynb` (an ellipse-patterned metasurface, $m=n=10$, $N_h=317$, 200 nm layer, $n=3.8$ in air, 300–900 nm sweep), `float32` `matexp` with the (pre-fix) automatic slice count reached $\max|\Delta R|=7\times10^{-3}$ against an `eig`/`float64` reference — 20× worse than `eig`/`float32`'s own $3\times10^{-4}$ — at specific, isolated wavelengths, and the error at a given wavelength could appear or vanish depending on `matexp_slices` alone, with no change to the physical geometry.

**Mechanism.** Every slice's S-matrix is built by embedding a *thin, fictitious* sub-slab of the patterned layer's material in the vacuum reference (`transfer_to_smatrix`, above) — a real physical construct only for the whole layer, not for an arbitrary thickness $d/n$ of it. That fictitious sub-slab has its own S-matrix poles: harmonics evanescent in the vacuum reference but propagating inside a high-index sub-slab produce a sharp, wavelength-dependent near-resonance at specific sub-slab thicknesses. On the benchmark structure at 560 nm, scanning sub-slab thickness in isolation (in `float64`, so the numbers below are exact, not roundoff) shows a spike:

| $t$ (nm) | 20 | 22 | 23 | 24 | **25** | 26 | 27 | 28 | 30 |
|---|---|---|---|---|---|---|---|---|---|
| $\max|S_\text{slice}|$ | 1.6 | 2.0 | 2.5 | 3.8 | **28** | 3.9 | 2.0 | 1.4 | 1.0 |

At $t=25\,\text{nm}=d/8$, the Redheffer star product's internal solve — $(I-S_{22}^BS_{11}^A)$ in `Block2x2.star` — has condition number $\sim2\times10^5$, independent of $n$ or `dtype`. At `complex128` ($\varepsilon_\text{machine}\sim2\times10^{-16}$) that costs $\sim5\times10^{-11}$ relative error, negligible; at `complex64` ($\varepsilon_\text{machine}\sim1.2\times10^{-7}$) it costs a few **percent**.

**Why it appears/disappears with $n$.** `star_power` composes $n$ slices by repeated squaring, so its intermediate thicknesses are exactly $(d/n)\cdot2^k$ for the $k$ visited on the way to $n$. For any even $n$, several of those are exact dyadic fractions of $d$ (e.g. $n=8,16,32,\dots$ all pass through $d/8=25\,\text{nm}$ on this structure); an odd $n>1$ never can, since $2^k/n$ is dyadic only if $n$ divides $2^k$, impossible for odd $n>1$. Measured error in $R$ at 560 nm vs. $n$ (`float32`; `float64` stays at $\lesssim10^{-11}$ for every $n$ shown, confirming the mechanism is precision-limited, not algorithmic):

| $n$ | 1 | 2 | 3 | 4 | 5 | 8 | 11 | 16 | 32 | 48 | 64 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| err | 2e-1 | 4e-2 | 3e-5 | 6e-4 | 1e-6 | 1e-1 | 3e-5 | **1.2** | 1e-1 | 4e-5 | 4e-2 |

Every power of two $\geq8$ is bad; every odd $n$ shown lands at `eig`/`float32` parity ($\sim10^{-5}$). ($n=1$ fails for the *modeled* reason instead — an unsliced exponent this large simply overflows.)

**Mitigations, applied by default:**

- `slice_count` nudges its automatic estimate to the next odd integer whenever it lands even, keeping the squaring ladder off exact dyadic fractions of $d$. This is the primary fix — on the benchmark it drops full-spectrum $\max|\Delta R|$ from $7\times10^{-3}$ to $1.4\times10^{-3}$, back in `eig`/`float32`'s own range.
- `star_power` checks each intermediate S-matrix's largest-magnitude entry against a `dtype`-aware threshold ($50$ for `complex64`, $10^4$ for `complex128`) and raises a `RuntimeWarning` if exceeded. This is a **heuristic backstop, not a certificate**: it reliably catches gross under-slicing (the $n=1$ case above), but a resonance narrow enough relative to the accumulated error can corrupt the result by a few percent without the magnitude itself leaving a physically plausible range — an odd `n` chosen right next to a resonance could still, in principle, be unlucky. If the warning fires, try a different `matexp_slices` or switch to `dtype=torch.float64`.
- `Block2x2.star` itself now solves rather than inverts (`(I-P)^{-1}@rhs` via `Entry.solve`, never a materialized `(I-P)^{-1}`) — repo convention, and the better-conditioned of the two equivalent formulations near a near-pole; it also benefits every other star product in the solver, not just this path.

**Not fixed by any of the above** (future work): the fictitious-slab resonance itself, which is an artifact of referencing every slice to *vacuum* regardless of the layer's actual index. Referencing each slice to a "gap medium" close to the layer's own mean permittivity instead — at the cost of two extra boundary S-matrices at the layer's outer faces — would shrink the index contrast that creates the poles in the first place, addressing the mechanism rather than dodging it. Not implemented.

---

## Cost model

Unlike the eig path, where `LayerSolver.prepare()` is the expensive step (eigendecomposition) and `smatrix()` is cheap (pure algebra), the matexp path **inverts** this: `prepare()` only needs $P$, $Q$ (no eigendecomposition — the cheap half of `compute_isotropic` + `eigsolver`), and the matrix exponential moves into `smatrix()`, because it depends on `thickness` and must stay responsive to `dataclasses.replace(op, thickness=...)` — computing it in `prepare()` would silently break the documented late-binding of thickness (a thickness-only sweep or optimization step must not require re-preparing).

Memory: the system matrix $A$ is $4N_h\times4N_h$ densified once per slice-exponentiation, versus $2N_h\times2N_h$ for $\Omega^2=PQ$ on the eig path — roughly $4\times$ the dense footprint per patterned layer.

**`float64` erodes matexp's GPU speed advantage — a hardware effect, not an algorithmic one.** Consumer GPUs (measured: RTX 4090) run `complex128` GEMM/`matrix_exp` at roughly $1/40$ their `complex64` throughput (`torch.linalg.matrix_exp` measured $36\times$ slower, plain `X@X` $42\times$), because `float64` isn't a first-class datapath on that silicon. `torch.linalg.eig`, by contrast, is latency/reduction-bound (`geev`), not GEMM-bound, so it only slows $\sim4\times$ going to `complex128`. The part matexp *removes* (eigendecomposition) is the part least sensitive to `dtype`; the part it *adds* (dense `matrix_exp`) is the part most sensitive. Net effect, same 20-wavelength benchmark structure as above, split by `Solver.__init__`/`run()`:

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

The snippet above omits two pieces added for the reasons in "Accuracy and conditioning": `slice_count` nudges its automatic `n` to the next odd integer, and `star_power` checks each intermediate's magnitude against a `dtype`-aware threshold, warning if a fictitious-slab resonance looks like it was hit.

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
- **`float32` accuracy is dominated by fictitious-slab resonance conditioning, not the exponent budget** — see "Accuracy and conditioning" above. The default mitigations (odd auto slice count, `star_power`'s magnitude guard) bring it back to `eig`/`float32`'s own ballpark ($\sim10^{-4}$–$10^{-3}$) on the structures measured so far, but the guard is a heuristic, not a certificate; prefer `float64` for tight-tolerance validation, keeping in mind its GPU cost (see "Cost model" above).
- **`Solver.run()` is no longer cheap** under `modesolver="matexp"` — a thickness sweep or a single-layer optimization step now pays one (sliced) matrix exponential per evaluation, instead of reusing a cached eigendecomposition and only re-cascading star products.
- **Peak memory is higher** (~2.3× measured above) — the $4N_h\times4N_h$ system matrix, densified once per slice-exponentiation, dominates.
- **`transfer()` is not yet wired into `Observables`** — the primitive exists and is cross-validated against the eig path's own `transfer()` (`tests/solver/test_matexpsolver.py::TestFieldParity`), but interior-field reconstruction itself is future work (`FieldSolution` in `solver/base.py`, currently unimplemented).
