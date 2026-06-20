# Tangential Vector Field (TVF) — Concept and Implementation

## Motivation

Rigorous Coupled-Wave Analysis (RCWA) expresses electromagnetic fields and
material properties as truncated Fourier series. When two such series are
multiplied — as happens constantly when enforcing Maxwell's equations inside a
patterned layer — the result is not simply the convolution of the two spectra.
For *discontinuous* functions (step-like permittivity distributions), naive
multiplication in Fourier space converges slowly and can give systematically
wrong results. Li (1996) showed that the correct rule depends on whether the
two functions being multiplied are simultaneously discontinuous at the same
point: if they are, you must factor them together using the *inverse rule*; if
they are not, the standard *direct rule* applies.

The practical consequence is that the field components parallel to a material
boundary must be factorized differently from those normal to it. To apply this
selectively, you need to know the orientation of every boundary at every point
on the grid — which is exactly what the **Tangential Vector Field** encodes.
A TVF is a 2-D vector field **T(r) = (Tₓ, T_y)** that is smooth, periodic,
and everywhere tangent (parallel) to the nearest material boundary. Given T,
the permittivity tensor can be split into a parallel part (factorized with the
inverse rule) and a normal part (factorized with the direct rule), and the
resulting Fourier convolution matrices converge at the expected exponential
rate.

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

This convention is shared across the whole solver (`harmonics.py`,
`convolution.py`) so that TVF output can feed directly into the Fourier
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
       ├─ optimize params = [Re F̂, Im F̂] under total_loss (L-BFGS)
       └─ ifft2(ifftshift(optimized params))   → spatial field
  └─ method-specific final normalization → Tx [B,D0,D1], Ty [B,D0,D1]
```

The geometric idea is: compute the gradient of the *real part* of the
permittivity (which points perpendicular to material boundaries), band-limit it
to the Fourier harmonics actually retained in the simulation, then rotate it
90° to obtain a first guess for the tangential direction. An optimization step
then smooths this field — finding the nearest smooth, periodic, band-limited
vector field that still aligns with the boundaries. The result is detached from
the autograd graph; TVF is treated as a precomputed geometry quantity, not
differentiated through during training.

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
the **centred Fourier representation** of the vector field — specifically the
real and imaginary parts of the 2-component Fourier coefficient array, stacked
into a real leaf tensor of shape `[B, D0, D1, 2, 2]`. Working in Fourier space
means the band limit is enforced implicitly: any optimizer step stays within
the `(2M+1)(2N+1)` active harmonics.

The total loss is:

```
L = α · L_align  +  β · L_fourier  +  γ · L_smooth
```

with default weights `α = 1.0`, `β = 1e-8`, `γ = 1.0`.

**Alignment loss** pulls the field toward the target tangent direction,
weighted by boundary strength (gradient magnitude):

```
L_align = mean_xy( w(r) · ‖T(r) − T_target(r)‖² )
```

where `w(r) = |∇ε(r)|` is large at sharp boundaries and zero in flat regions.
Computed in real space after an inverse FFT of the current params.

**Fourier regularization loss** penalizes high-frequency content by weighting
each Fourier coefficient by the squared magnitude of the corresponding
reciprocal vector:

```
L_fourier = (area) · mean_{m,n}( |G_{mn}|² · ‖F̂_{mn}‖² ),   G = m·b₁ + n·b₂
```

Computed directly on the Fourier coefficients (no IFFT needed). The unit-cell
area `|det[a₁|a₂]|` makes the weight dimensionally consistent across different
lattice sizes.

**Smoothness loss** is the mean squared spatial gradient of the field,
computed in real space via `_grad_periodic` applied to each component:

```
L_smooth = mean_xy( ‖∇Tₓ‖² + ‖∇T_y‖² )
```

These three terms work together: alignment keeps the field pointing the right
way, smoothness pushes it to vary slowly, and Fourier regularization provides
an additional high-frequency damper directly in spectral space.

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

| Method | Literature | Final normalization | Output dtype |
|--------|------------|---------------------|--------------|
| `Jones` | Antos 2009 | `normalize_jones` on `.real` of optimized field | complex |
| `Pol` | S4 / Liu & Fan 2012 | `normalize_max_global` on `.real` | real |
| `Normal` | Schuster 2007 | `normalize_elementwise` on `.real` | real |
| `Jones_direct` | FMMax / Schubert 2023 | `normalize_max_global` (complex, no `.real`) | complex |

`Jones_direct` is different in one more way: in `_prepare_field` the target
itself is immediately converted to a Jones field via `normalize_jones`, and the
initial Fourier coefficients are set equal to that target. This means the
optimization (if run) starts at the answer, and running with `steps=1` is
effectively just refining the FMMax-style field through the band-limited
objective.

For all other methods the initial guess is the raw (non-elementwise-normalized)
perpendicular-gradient field — a geometrically meaningful but potentially rough
starting point that the optimizer then smooths.

---

## Optimizer

The optimizer is `TorchLBFGS`, a thin wrapper around `torch.optim.LBFGS`.
L-BFGS is the natural choice for this problem: the loss is smooth and the
number of optimization variables (`(2M+1)(2N+1) × 2 × 2` real numbers) is
moderate, so quasi-Newton convergence is fast and reliable without the
per-iteration cost of a full Hessian.

The closure sums the per-batch loss to a scalar so all elements in a batch
optimize simultaneously in a single run:

```python
def closure():
    opt.zero_grad()
    loss = loss_fn(params).sum()  # sum over batch
    loss.backward()
    return loss

for _ in range(steps):
    opt.step(closure)
```

One optimizer instance is created and reused across all `steps`, so the
curvature history (the L-BFGS Hessian approximation) accumulates across
steps. With the default `steps=1` and `max_iter=20`, a single `opt.step` runs
up to 20 internal L-BFGS iterations before returning.

---

## Numerical Choices at a Glance

| Parameter | Value | Location |
|-----------|-------|----------|
| Loss weight α (alignment) | `1.0` | `TVF._optimize`, `TVF.compute` |
| Loss weight β (Fourier reg) | `1e-8` | `TVF._optimize`, `TVF.compute` |
| Loss weight γ (smoothness) | `1.0` | `TVF._optimize`, `TVF.compute` |
| Optimization steps | `1` | `TVF._optimize`, `TVF.compute` |
| L-BFGS learning rate | `1.0` | `TorchLBFGS.__init__` |
| L-BFGS max iterations | `20` | `TorchLBFGS.__init__` |
| L-BFGS tolerance (grad) | `1e-8` | `TorchLBFGS.__init__` |
| L-BFGS tolerance (change) | `1e-8` | `TorchLBFGS.__init__` |
| L-BFGS line search | `None` | `TorchLBFGS.__init__` |
| Central-difference factor | `0.5` | `_grad_periodic` |
| Reciprocal-vector normalization | `2π` | `_grad_periodic`, `fourier_regularization_loss` |
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

tvf = TVF(a1, a2, M, N, method="Jones")  # or "Pol", "Normal", "Jones_direct"

# eps: real or complex permittivity, shape [B, D0, D1]
eps = torch.rand(2, 64, 64)

Tx, Ty = tvf.compute(eps)
# Tx, Ty: shape [B, D0, D1]
# dtype: complex for "Jones" / "Jones_direct", real for "Pol" / "Normal"
# detached from the autograd graph
```

The optimizer can be chosen at construction time (currently only `"LBFGS"` is
supported). Loss weights and the number of optimizer outer steps can be
overridden per-call:

```python
Tx, Ty = tvf.compute(eps, alpha=1.0, beta=1e-8, gamma=1.0, steps=3)
```
