# Tangential Vector Field (TVF) — Concept and Implementation

## Motivation

Rigorous Coupled-Wave Analysis (RCWA) expresses electromagnetic fields and
material properties as truncated Fourier series. When two such series are
multiplied — as occurs constantly when enforcing Maxwell's equations inside a
patterned layer — the result is not simply the convolution of the two spectra.
For *discontinuous* functions (step-like permittivity distributions), naive
multiplication in Fourier space converges slowly and can give systematically
wrong results. Li (1996) established that the correct factorization rule depends
on the local continuity of the two factors: if both are simultaneously
discontinuous at the same point, the *inverse rule* must be applied; otherwise,
the standard *Laurent rule* applies. The full statement of these rules and their
derivation are given in [Factorization rules](factorization.md).

The practical consequence is that the field components parallel to a material
boundary must be factorized differently from those normal to it. To apply this
selectively, the orientation of every material boundary must be encoded at every
grid point — which is exactly what the **Tangential Vector Field** (TVF)
encodes. A TVF is a 2-D vector field $\mathbf{T}(\mathbf{r}) = (T_x, T_y)$
that is smooth, periodic, and everywhere tangent (parallel) to the nearest
material boundary. Given $\mathbf{T}$, the permittivity tensor can be split
into a tangential part (factorized with the inverse rule) and a normal part
(factorized with the Laurent rule), yielding the correction matrices
$[[A_{ij}]]$ that appear in the $Q$ operator (see [RCWA Core](rcwa_core.md)).
The resulting Fourier convolution matrices converge at the expected exponential
rate, in contrast to the power-law convergence of naive truncation.

Four published methods for constructing a TVF are supported here:

| Key | Reference |
|-----|-----------|
| `Jones` | Antos 2009, *Opt. Express* 17, 7269 |
| `Pol` | S4, Liu & Fan 2012, *Comput. Phys. Commun.* 183, 2233 |
| `Normal` | Schuster 2007, *J. Opt. Soc. Am. A* 24, 2880 |
| `Jones_direct` | FMMax, Schubert et al. 2023, *Opt. Express* 31, 503481 |

They differ in how the final vector field is normalized and whether it is
real-valued or complex (Jones).

---

## Grid and Lattice Convention

All fields in this module are 3-D tensors of shape **`[B, D0, D1]`**, where
`B` is a batch dimension and `D0 × D1` is the spatial grid over the unit cell.
The two axes map to the two lattice vectors as follows:

- **axis −2** (size `D0`): lattice vector **a₂**, fractional coordinate `f₂ = i/D₀`, harmonic order `n`.
- **axis −1** (size `D1`): lattice vector **a₁**, fractional coordinate `f₁ = j/D₁`, harmonic order `m`.

The Cartesian position of grid point `(i, j)` is `f₂ a₂ + f₁ a₁`. The
corresponding **reciprocal lattice vectors** `b₁`, `b₂` satisfy

```
bᵢ · aⱼ = 2π δᵢⱼ
```

and are computed explicitly via the 2-D cross product:

```python
det = a1x * a2y - a1y * a2x
factor = 2 * torch.pi / det
b1 = factor * torch.stack([ a2y, -a2x])
b2 = factor * torch.stack([-a1y,  a1x])
```

This convention is shared across the whole solver so that TVF output can feed directly into the Fourier
convolution stage.

---

## The Pipeline End-to-End

A single call to `TVF.compute(eps)` runs the following sequence:

```
TVF.compute(eps)
  └─ _optimize(Re(eps).detach())
       ├─ _prepare_field(field_real)
       │    ├─ periodic Cartesian gradient  (_grad_periodic)
       │    ├─ low-pass filter to [2M+1]×[2N+1] harmonics
       │    ├─ rotate −90° → tangent-to-boundary target field
       │    ├─ [Jones_direct] convert to Jones field
       │    └─ fftshift(fft2(...)) → initial Fourier coefficients + boundary weights
       ├─ gather in-band coeffs → params [B, num_terms, 2, 2]  (hard band limit)
       ├─ optimize params = [Re F̂, Im F̂] under total_loss (exact Newton)
       ├─ scatter back → full [B,D0,D1,2] FFT grid
       └─ ifft2(ifftshift(...))   → spatial field
  └─ method-specific final normalization → Tx [B,D0,D1], Ty [B,D0,D1]
```

The geometric idea is: compute the gradient of the *real part* of the
permittivity (which points perpendicular to material boundaries), band-limit it
to the Fourier harmonics actually retained in the simulation, then rotate it
90° to obtain a first guess for the tangential direction. An optimization step
then smooths this field — finding the nearest smooth, periodic, band-limited
vector field that still aligns with the boundaries. The optimization variable
is **only the `(2M+1)(2N+1)` in-band coefficients** (a hard constraint enforced
by gather/scatter around the optimizer), so out-of-band harmonics are
structurally zero throughout. The result is detached from the autograd graph;
TVF is treated as a precomputed geometry quantity, not differentiated through
during training.

---

## Building Blocks

### Periodic gradient on an oblique lattice

`_grad_periodic(s, a1, a2)` computes the 2-D Cartesian gradient of a scalar
field `s` with periodic boundary conditions on an arbitrary Bravais lattice.
Central differences are evaluated in *fractional* coordinates and converted to
Cartesian via the chain rule:

```
∇s = (∂s/∂f₁) · b₁/(2π)  +  (∂s/∂f₂) · b₂/(2π)
```

In code:

```python
ds_df2 = 0.5 * D0 * (torch.roll(s, -1, dims=-2) - torch.roll(s, 1, dims=-2))
ds_df1 = 0.5 * D1 * (torch.roll(s, -1, dims=-1) - torch.roll(s, 1, dims=-1))
two_pi = 2.0 * torch.pi
gradx = (ds_df1 * b1[0] + ds_df2 * b2[0]) / two_pi
grady = (ds_df1 * b1[1] + ds_df2 * b2[1]) / two_pi
```

The factor `0.5 * Dᵢ` is exactly `1 / (2 Δfᵢ)` — the standard central
difference denominator, because the fractional step size is `Δfᵢ = 1/Dᵢ`.
`torch.roll` enforces periodic wrapping at the cell boundaries, so the
gradient is consistent with the periodicity of the lattice for any Bravais
geometry, including oblique and hexagonal cells.

### Forward-difference gradient for the smoothness loss

The **smoothness loss** measures how quickly the TVF varies spatially. It is
computed via `_grad_forward_periodic(s, a1, a2)`, which uses *forward*
differences instead of central differences:

```python
ds_df2 = D0 * (torch.roll(s, shifts=-1, dims=-2) - s)   # forward along a2
ds_df1 = D1 * (torch.roll(s, shifts=-1, dims=-1) - s)   # forward along a1
gradx = (ds_df1 * b1[0] + ds_df2 * b2[0]) / two_pi
grady = (ds_df1 * b1[1] + ds_df2 * b2[1]) / two_pi
```

**Why forward, not central?** Central differences `0.5(f[+1]−f[−1])` have a
null space at the Nyquist / checkerboard frequency: any perfectly alternating
`+1, −1, +1, −1, …` pattern produces *zero* derivative at every point, making
the smoothness loss completely blind to grid-scale noise. Forward differences
`f[+1]−f` do not have this null space (the checkerboard gives a constant
`−2` everywhere), so they correctly penalise all frequencies up to Nyquist.

Note that the **array gradient** (used to detect material boundaries and build
the target field) is still computed with `_grad_periodic` (central differences),
which is more accurate for that role. Only the smoothness penalty inside the
loss switches to forward differences.

### Fourier low-pass filtering

Before anything else, the gradient is band-limited to the `(2M+1) × (2N+1)`
harmonics that the simulation actually resolves. This is essential: the TVF
must live in the same Fourier space as the electromagnetic fields, otherwise
the convolution matrices it informs would contain aliased content.

`low_pass_mask(D0, D1, M, N)` builds a boolean mask in the *centred* Fourier
domain by shifting harmonic indices to `[−Dᵢ//2, …, Dᵢ//2−1]` and keeping
those with `|n| ≤ N` and `|m| ≤ M`:

```python
n = torch.arange(D0, device=device) - D0 // 2   # a2 orders
m = torch.arange(D1, device=device) - D1 // 2   # a1 orders
N_grid, M_grid = torch.meshgrid(n, m, indexing="ij")
mask = (N_grid.abs() <= N) & (M_grid.abs() <= M)
```

`low_pass_filter` then applies it in-place on the gradient vector field:

```python
gradx_fft = torch.fft.fftshift(torch.fft.fft2(gradx, dim=(-2,-1)), dim=(-2,-1))
filteredx  = torch.fft.ifft2(torch.fft.ifftshift(gradx_fft * mask, ...), ...).real
```

The `.real` at the end discards the round-off imaginary part — the input
gradient is real, so the output should be too.

The band limit is enforced **twice**: once on the initial gradient (above), and
once as a **hard structural constraint** on the optimization variable itself.
In `_optimize`, the centred FFT coefficients are *gathered* to just the
`num_terms = (2M+1)(2N+1)` in-band positions before being handed to the
optimizer, and *scattered* back into the full `D0×D1` grid (with zeros outside
the band) before every loss evaluation. This means the optimizer cannot
accidentally push energy into out-of-band harmonics.

### Magnitudes and normalizations

A recurring pattern throughout is computing the per-pixel magnitude of a
2-component field while keeping autograd safe at pixels where the field is
exactly zero (where `sqrt` would produce a NaN gradient):

```python
mag_sq = torch.sum(torch.abs(field) ** 2, dim=-1, keepdim=True)
is_zero = mag_sq == 0
mag_sq_safe = torch.where(is_zero, torch.ones_like(mag_sq), mag_sq)
mag = torch.where(is_zero, torch.zeros_like(mag_sq), torch.sqrt(mag_sq_safe))
```

Two normalization strategies are built on this:

- **`normalize_max_global`** divides the whole field by the single largest
  pixel magnitude (per batch element). This preserves the relative shape of
  the vector field — longer vectors stay longer — while scaling the global peak
  to 1. Used by `Pol` and `Jones_direct`.

- **`normalize_elementwise`** makes every nonzero pixel a unit vector,
  discarding magnitude entirely. This is a pure orientation field. Used by
  `Normal` and as an intermediate step when constructing the alignment target.

### The Jones construction

The `Jones` and `Jones_direct` methods output a *complex* 2-vector per pixel.
The motivation is that a real unit vector field has a sign ambiguity — flipping
all vectors along a smooth path by 180° is a valid tangent field, but it
creates an artificial discontinuity in the field itself. A Jones vector encodes
orientation as a phase, so the field can vary smoothly even when the underlying
real direction flips sign.

`normalize_jones(field)` follows Antos 2009. Given a (real or complex) field
that has already been globally normalized to peak magnitude 1, it builds:

```
θ = angle(tₓ + i tᵧ)          # in-plane orientation angle
φ = (π/8)(1 + cos(π |t|))     # magnitude-dependent phase mix
jₓ = e^{iθ} (tₓ cos φ − i tᵧ sin φ)
jᵧ = e^{iθ} (tᵧ cos φ + i tₓ sin φ)
```

In code:

```python
phi   = torch.pi / 8.0 * (1.0 + torch.cos(torch.pi * magnitude))
theta = torch.angle(tx_norm + 1j * ty_norm)
exp_i_theta = torch.exp(1j * theta)
jx = exp_i_theta * (tx_norm * torch.cos(phi) - ty_norm * 1j * torch.sin(phi))
jy = exp_i_theta * (ty_norm * torch.cos(phi) + tx_norm * 1j * torch.sin(phi))
```

The phase `φ` smoothly maps magnitude into `[0, π/4]`: at a strong boundary
(`|t| = 1`) it is zero and the Jones vector collapses to a purely real unit
vector; near a field-free region (`|t| ≈ 0`) it reaches `π/4` and the two
components mix into a near-circular polarization state. This intermediate
behaviour is what allows the field to remain holomorphic across sign flips in
the underlying real field. Pixels with magnitude numerically indistinguishable
from zero (detected via `torch.isclose` with defaults `rtol=1e-5, atol=1e-8`)
receive the isotropic fallback orientation `(1/√2, 1/√2)`.

---

## The Optimization Objective

Rather than using the raw band-limited gradient as the TVF directly, the
implementation optimizes a smooth version of it. The optimization variable is
the **in-band Fourier coefficients** — specifically the real and imaginary parts
of the `(2M+1)(2N+1)` active harmonics of the 2-component field, gathered into
a real leaf tensor of shape `[B, num_terms, 2, 2]`. The band limit is a **hard
structural constraint**: the optimizer never sees or touches out-of-band
harmonics (they are kept at zero by the gather/scatter wrapper).

The total loss is:

```
L = α · L_align  +  β · L_fourier  +  γ · L_smooth
```

All three weights can be overridden per-call; pass `None` to use the defaults
(`alpha = 1.0`, `beta = gamma = 0.05`).  Because both `L_fourier` and
`L_smooth` measure the same physical Dirichlet energy `∫‖∇f‖²dA`, **β and γ sit
on the same scale** (O(0.01–0.1)); they are independent of the lattice period
and the Fourier truncation order (M, N).

**Alignment loss** pulls the field toward the target tangent direction,
weighted by boundary strength (gradient magnitude):

```
L_align = mean_xy( w(r) · ‖T(r) − T_target(r)‖² )
```

where `w(r) = |∇ε(r)|` is large at sharp boundaries and zero in flat regions.
Computed in real space after an inverse FFT of the current in-band params.

**Fourier regularization loss** penalizes high-frequency content by weighting
each physical Fourier expansion coefficient by the squared magnitude of the
corresponding reciprocal vector:

```
L_fourier = area · sum_{m,n}( |G_{mn}|² · |c_{mn}|² ),   G = m·b₁ + n·b₂
```

where `c_{mn} = F̂_{mn} / (D0·D1)` is the **physical** Fourier expansion
coefficient (O(1) for an O(1) spatial field). The raw `torch.fft.fft2` output
`F̂_{mn}` is O(D²) because `torch.fft.fft2` is unnormalized; `fourier_regularization_loss`
divides internally by `D0·D1` before squaring to recover the physical
coefficients. 

> **Why this matters**: without the `/D0·D1` normalization, the Fourier penalty
> is `(D0·D1)²` ≈ 2.7×10¹⁰ times too large for a 128×128 grid, completely
> overwhelming the alignment term and causing the Newton solver to collapse all
> spatial structure to a DC-only (constant) field.

The unit-cell area `|det[a₁|a₂]|` makes the weight dimensionally consistent
across different lattice sizes. Note the **sum** (not mean) over harmonics —
both `L_fourier` and `L_smooth` are integrals over the cell, so they remain on
the same absolute scale regardless of the number of harmonics retained.

**Smoothness loss** is the Dirichlet energy of the field, computed in real
space via `_grad_forward_periodic` (forward differences) applied to each
component:

```
L_smooth = area · mean_xy( ‖∇Tₓ‖² + ‖∇T_y‖² )  =  ∫‖∇T‖² dA
```

where `area = |det[a₁|a₂]|`.  Multiplying the per-pixel mean squared gradient
by the cell area converts it to the physical Dirichlet integral ∫‖∇T‖²dA,
making the result dimensionless and scale-invariant (independent of the lattice
size `P`).

This is the real-space twin of `L_fourier`: both measure the same Dirichlet
energy ∫‖∇f‖²dA — one via spectral `|G|²` weighting, one via forward
differences on the grid.  The forward-difference form additionally penalises
grid-scale (Nyquist) noise that the spectral form cannot reach, which is why
both regularizers coexist.  With the area factor in place, γ and β sit on the
**same scale** (both O(0.01–0.1)) regardless of the lattice period or the
truncation order (M, N).

Forward differences are required — see the "Forward-difference gradient"
section above for why central differences would be incorrect.

These three terms work together: alignment keeps the field pointing the right
way, smoothness pushes it to vary slowly (including at Nyquist scale), and
Fourier regularization provides an additional high-frequency damper in spectral
space.

In `total_loss`, the reconstruction from params to spatial field and back is:

```python
field_fft     = params[..., 0] + 1j * params[..., 1]
field_spatial = torch.fft.ifft2(torch.fft.ifftshift(field_fft, ...), ...)
loss = alpha * alignment_loss(field_spatial, target, weights)
     + beta  * fourier_regularization_loss(field_fft, a1, a2)
     + gamma * smoothness_loss(field_spatial, a1, a2)
```

---

## The Four Methods

| Method | Literature | Default β, γ | Final normalization | Output dtype |
|--------|------------|--------------|---------------------|--------------|
| `Jones` | Antos 2009 | 0.05, 0.05 | `normalize_jones` on `.real` of optimized field | complex |
| `Pol` | S4 / Liu & Fan 2012 | 0.05, 0.05 | `normalize_max_global` on `.real` | real |
| `Normal` | Schuster 2007 | 0.05, 0.05 | `normalize_elementwise` on `.real` | real |
| `Jones_direct` | FMMax / Schubert 2023 | 0.05, 0.05 | `normalize_max_global` (complex, no `.real`) | complex |

`Jones_direct` is unique in its target construction: in `_prepare_field`
the target itself is immediately converted to a Jones field via `normalize_jones`,
and the initial Fourier coefficients are set equal to that target. Because the
loss is exactly quadratic and the Newton solve is exact, the optimizer finds
the unique in-band-limited Jones field closest (in the weighted sense) to this
target in a single step.

For all other methods the initial guess is the raw (non-elementwise-normalized)
perpendicular-gradient field — a geometrically meaningful but potentially rough
starting point that the optimizer then smooths.

---

## Optimizer

The default optimizer is `NewtonExact`. The TVF total loss is a **real
quadratic function** of the in-band Fourier coefficients (all three loss terms
are sums/means of squared linear functions of the coefficients), which means a
**single Newton step gives the exact global minimum**:

```
x ← x − H⁻¹ g,   where g = ∇L,  H = ∇²L  (constant for a quadratic)
```

Two structural properties of the loss are exploited to make this fully
vectorized:

1. **Quadratic in params** — IFFT, gather/scatter, and all three loss terms are
   composed of linear and squared-linear operations in `params`. The Hessian is
   therefore constant everywhere; one step lands exactly at the minimum.

2. **Decoupled across the batch** — `L[b]` depends only on `params[b]`, so the
   full batch gradient `∇ L.sum()` equals the stack of per-sample gradients, and
   the Hessian is block-diagonal with one `[flat, flat]` block per batch element.

The implementation uses `torch.func` functional transforms:

```python
grad_fn = torch.func.grad(lambda p: loss_fn(p).sum())

# 1. All per-sample gradients in one backward pass
g = grad_fn(x)                                  # [B, *shape_per]

# 2. Hessian columns: vmap over flat basis vectors, one JVP each
def hvp_col(v):
    v_batch = v.unsqueeze(0).expand(B, *shape_per)
    return torch.func.jvp(grad_fn, (x,), (v_batch,))[1]

basis = torch.eye(flat).reshape(flat, *shape_per)
cols  = torch.func.vmap(hvp_col)(basis)         # [flat, B, *shape_per]
H     = cols.reshape(flat, B, flat).permute(1, 2, 0)   # [B, flat, flat]

# 3. One batched solve for all batch elements
delta = torch.linalg.solve(H + ε·I, g.reshape(B, flat))
```

A small diagonal shift `ε = 1e-12` regularizes the Hessian for numerical
safety; in practice H is well-conditioned for typical permittivity patterns.

With the default `steps=1` the solver runs a single Newton update, which is the
exact solution. Running `steps=2` produces negligibly different output.

The `TorchLBFGS` optimizer (wrapping `torch.optim.LBFGS`) is also available via
`optimizer="lbfgs"`. The Newton solve is preferred because it is both exact and
faster for this loss structure.

---

## Numerical Choices at a Glance

| Parameter | Value | Location |
|-----------|-------|----------|
| Default optimizer | `NewtonExact` | `TVF.__init__` |
| Optimization variable | in-band coeffs only, shape `[B, num_terms, 2, 2]` | `TVF._optimize` |
| Band limit enforcement | hard (gather/scatter via `low_pass_mask`) | `TVF._optimize` |
| Newton diagonal regularization | `1e-12` | `NewtonExact.__init__` |
| Optimization steps | `1` (exact for quadratic loss) | `TVF.compute` |
| Loss weight α (alignment) | `1.0` | `TVF.compute` |
| Loss weight β (Fourier reg) | `0.05` (all methods) | `TVF.compute` |
| Loss weight γ (smoothness) | `0.05` (all methods) | `TVF.compute` |
| β, γ scale-invariance | both weight `∫‖∇f‖²dA`; independent of P, M, N | `fourier_regularization_loss`, `smoothness_loss` |
| Fourier loss reduction | `sum` over harmonics | `fourier_regularization_loss` |
| FFT coefficient normalization | ÷ `D0·D1` inside `fourier_regularization_loss` (converts unnormalized `fft2` output to physical Fourier expansion coefficients) | `fourier_regularization_loss` |
| Smoothness gradient | **forward** difference (`_grad_forward_periodic`) | `smoothness_loss` |
| Array / target gradient | central difference (`_grad_periodic`) | `_prepare_field` |
| Central-difference factor | `0.5 · Dᵢ` | `_grad_periodic` |
| Forward-difference factor | `Dᵢ` | `_grad_forward_periodic` |
| Reciprocal-vector normalization | `2π` | `_grad_periodic`, `_grad_forward_periodic`, `fourier_regularization_loss` |
| DC centering (fftshift) | integer `// 2` | `low_pass_mask`, `fourier_regularization_loss` |
| Jones zero fallback | `(1/√2, 1/√2)` | `normalize_jones` |
| Jones phase parameter | `φ = (π/8)(1 + cos(π·‖t‖))` | `normalize_jones` |
| Jones near-zero threshold | `torch.isclose` rtol=1e-5, atol=1e-8 | `normalize_jones` |
| Zero-magnitude guard | exact `== 0` (magnitudes); `isclose` (Jones) | `_field_magnitude`, `normalize_*` |

All device and dtype choices are inherited from the input tensors — there are
no explicit `.to()` or `.cuda()` calls inside the module.

---

## Usage

```python
import torch
from metarcwa.solver.tvf import TVF

# Lattice vectors (Cartesian, shape [2])
a1 = torch.tensor([1.0, 0.0])
a2 = torch.tensor([0.0, 1.0])

# Fourier truncation orders: keep |m| <= M (a1), |n| <= N (a2)
M, N = 5, 5

tvf = TVF(a1, a2, M, N, method="Jones_direct")  # or "Jones", "Pol", "Normal"

# eps: real or complex permittivity, shape [B, D0, D1]
eps = torch.rand(2, 64, 64)

Tx, Ty = tvf.compute(eps)
# Tx, Ty: shape [B, D0, D1]
# dtype: complex for "Jones" / "Jones_direct", real for "Pol" / "Normal"
# detached from the autograd graph
```

The default optimizer is `"newton"` (exact Newton solve). To use L-BFGS instead,
pass `optimizer="lbfgs"` at construction time:

```python
tvf_lbfgs = TVF(a1, a2, M, N, method="Jones_direct", optimizer="lbfgs")
```

Loss weights default to `alpha=1.0`, `beta=gamma=0.05`.  Both `beta` and
`gamma` weight the Dirichlet energy `∫‖∇f‖²dA` and sit on the same scale
(O(0.01–0.1)); they do not depend on the period P, grid size D, or truncation
(M, N).  Pass explicit values to override; `None` restores the defaults:

```python
# Use defaults (recommended):
Tx, Ty = tvf.compute(eps)

# Override specific weights or number of steps:
Tx, Ty = tvf.compute(eps, beta=0.02, gamma=0.1, steps=2)

# Override all:
Tx, Ty = tvf.compute(eps, alpha=1.0, beta=0.05, gamma=0.05, steps=1)
```
