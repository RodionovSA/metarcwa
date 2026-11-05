# Perfectly matched layers (PML)

> **Status:** design draft. Nothing described here is implemented. Sections marked **TODO** are not worked out yet.

## Overview

RCWA is periodic by construction: the permittivity of every layer is a Fourier series over the lattice (Eq. (5) in [RCWA core](rcwa_core.md)), so the method natively describes infinite arrays. Along $z$ nothing special is needed: the half-spaces are exact semi-infinite media, and outgoing waves leave the stack through the S-matrix boundaries (see [S-matrix algebra](smatrix.md)).

The PML discussed here is **lateral** (in the $xy$-plane). It sits at the edges of a large rectangular *supercell* and absorbs the light that would otherwise reach the neighbouring copies of the structure. With it, the periodic solver can treat **aperiodic** problems: isolated scatterers, finite arrays, waveguides. The main use case is a finite structure under a beam so wide that it can be treated as a plane wave, **at any angle of incidence**.

The design has two independent parts:

1. **RCWA + PML.** The PML is a complex coordinate stretch. In Fourier space it replaces the diagonal $K_x$, $K_y$ by dense $\tilde K_x$, $\tilde K_y$. $P$, $Q$, the TVF, both mode solvers and the S-matrix algebra keep their form.
2. **Illumination.** An oblique plane wave cannot pass through a lateral PML undistorted. The field is therefore split into a known incident field and a scattered field. Only the scattered field is solved with RCWA + PML, driven by a source term localized in the structure. In S-matrix terms, each layer gains a *source vector* next to its S-matrix.

The approach follows [Silberstein et al. (2001)](https://opg.optica.org/josaa/abstract.cfm?uri=josaa-18-11-2865) and [Hugonin & Lalanne (2005)](https://opg.optica.org/josaa/abstract.cfm?uri=josaa-22-9-1844) for the PML, and Pisarenco et al. (2010) for the scattered-field ("contrast-field") formulation.

---

## Complex coordinate stretching

A PML is an analytic continuation of the spatial coordinates into the complex plane. We replace the in-plane derivatives by stretched ones:

$$
\begin{align}
\frac{\partial}{\partial x} \;\to\; \frac{1}{s_x(x)}\frac{\partial}{\partial x}, \qquad
\frac{\partial}{\partial y} \;\to\; \frac{1}{s_y(y)}\frac{\partial}{\partial y}.
\end{align}
$$

This is equivalent to working in the complex coordinate $\tilde x = \int^x s_x(x')\,dx'$. In the physical region $s_x = 1$ and Maxwell's equations are unchanged. Inside the PML $s_x$ is complex, so an outgoing wave becomes

$$
\begin{align}
\exp(jk_x\tilde x) = \exp\!\Big(jk_x\!\int^x \mathrm{Re}\,s_x\,dx'\Big)\,\exp\!\Big(-k_x\!\int^x \mathrm{Im}\,s_x\,dx'\Big),
\end{align}
$$

and decays without reflection at the PML entrance (in the continuum limit). With the `exp(−jωt)` convention, absorption requires $\mathrm{Im}\,s_x > 0$.

**Profile.** The stretch is a function $s(d)$ of the normalized distance $d \in [0,1]$ into the PML ($0$ at the inner edge, $1$ at the cell edge). No single profile is best for every problem, so $s(d)$ is pluggable. Any profile must satisfy $s(0) = 1$ (a continuous entrance, needed for the factorization below), $\mathrm{Re}\,s > 0$ and $\mathrm{Im}\,s \ge 0$. Two are built in:

$$
\begin{align}
s_\text{stretch-absorb}(d) &= \big(1 + a_\text{max}\,d^{\,p}\big)\big(1 + j\,\sigma_\text{max}\sin^2(\tfrac{\pi}{2}d)\big),\\
s_\text{polynomial}(d) &= 1 + j\,\sigma_\text{max}\,d^{\,p}.
\end{align}
$$

The first is the profile of [fmmax](https://github.com/facebookresearch/fmmax) and the default ($a_\text{max} = 4$, $p = 4$, $\sigma_\text{max} = 1$). Its real factor compresses space inside the PML and damps evanescent waves, which a purely imaginary stretch, such as the second profile, does not attenuate. Both are continuous, including across the cell edge where the two PMLs of neighbouring cells meet.

Without a real stretch, an evanescent tail that reaches the cell edge couples to the neighbouring cell through the complex coordinate. For a weakly guided slab mode this produced a spurious gain $\sim 50\times$ larger than the PML loss with the default profile, and the mode was labelled with the opposite sign of $k_z$. S-matrices are not affected (the mode labelling does not change a finite layer's S-matrix), but guided-mode analysis is. Use a profile with real stretch for guided or evanescent-dominated problems.

**Choosing parameters.** Two test problems were swept at wavelength $1$: a weakly guided slab (index $1.6/1.45$, evanescent decay length $0.45$, cell $3$, PML width $0.75$), sensitive to evanescent tails, and an isolated dielectric ridge on a substrate, sensitive to radiated waves. The defaults lie in the region that is good for both:

| Parameter | Slab (evanescent) | Ridge (radiation) |
|---|---|---|
| $a_\text{max}$ | $\ge 4$ needed; $1$–$2$ give errors $\sim 5\times10^{-5}$ and a spurious gain | insensitive ($0$–$8$) |
| $p$ | insensitive ($1$–$4$) | $p = 1$ worse; $2$–$4$ equivalent |
| $\sigma_\text{max}$ | insensitive ($0.5$–$2$) | $0.5$–$2$ equivalent; $\ge 4$ worse |

Practical guidance: a PML width of about one wavelength, and a margin of a few evanescent decay lengths between the structure and the PML.

**Geometry inside the PML.** A PML is reflectionless only if the structure does not vary along its normal. The pattern is therefore **edge-padded** into the PML: inside the $x$-PML, the geometry at the inner PML boundary is continued unchanged along $x$. A substrate or film that reaches the cell edge continues straight through the PML.

---

## Rectangular lattices only

The PML requires a rectangular lattice ($\mathbf a_1 \parallel \hat x$, $\mathbf a_2 \parallel \hat y$, e.g. `Lattice.rectangular(px, py)`). Any other lattice raises an error.

For aperiodic problems the supercell is artificial, so a rectangular one can always be chosen. Even a hexagonal crystal has a rectangular supercell ($a \times a\sqrt3$, two atoms). An oblique cell would make the PML worse: its edges have normals $\hat{\mathbf b}_1$, $\hat{\mathbf b}_2$, which are not grid directions. Stretching along $\mathbf a_1$ instead fails for near-grazing outgoing waves: a wave decays only if $\mathbf k\cdot\mathbf a_1 > 0$, but it is outgoing when $\mathbf k\cdot\mathbf b_1 > 0$.

---

## Fourier-space form

In the RCWA derivation, every $\partial_x$ acting on a field becomes multiplication by the diagonal matrix $jK_x$ (see "RCWA equations" in [RCWA core](rcwa_core.md)). With stretching, $\partial_x f$ is multiplied by the periodic function $1/s_x(x)$, which in Fourier space is a convolution. The stretched wavevector operators are

$$
\begin{align}
\tilde K_x = [[1/s_x]]\,K_x, \qquad \tilde K_y = [[1/s_y]]\,K_y,
\end{align}
$$

where $[[1/s_x]]$ is the Toeplitz matrix of the Fourier coefficients of $1/s_x$ (same construction as $[[\varepsilon]]$). Since $s_x$ depends only on $x$, $[[1/s_x]]$ couples only the index $m$:

$$
\begin{align}
\big([[1/s_x]]\big)_{mn,\,m'n'} = (1/s_x)_{m-m'}\,\delta_{nn'}, \qquad
\big([[1/s_y]]\big)_{mn,\,m'n'} = \delta_{mm'}\,(1/s_y)_{n-n'}.
\end{align}
$$

Properties worth keeping in mind:

- $\tilde K_x$, $\tilde K_y$ are **dense** in the $N_h$ harmonic basis ($\tilde K_x$ is block-diagonal over $n$).
- With a rectangular truncation, $\tilde K_x$ and $\tilde K_y$ act on different indices and **commute**. A circular truncation breaks this, so the code must keep the product order exactly as derived.
- Neither commutes with $[[\varepsilon]]$.
- $\tilde K_x^2 = [[1/s_x]]K_x[[1/s_x]]K_x$, **not** $[[1/s_x^2]]K_x^2$.

### Factorization

Each stretched derivative is the product $\frac{1}{s_x}\cdot\partial_x f$, so Li's rules (see [Factorization rules](factorization.md)) apply:

- **Continuous profile** (the one above). $1/s_x$ is continuous everywhere, so at least one factor is continuous at every point and the **Laurent rule** $[[1/s_x]]$ is exact in the limit, whatever $\partial_x f$ does.
- **Profile with a jump.** $s_x$ and $\partial_x f$ jump together while their product (the physical derivative) is continuous, which calls for the **inverse rule** $[[s_x]]^{-1}$.

Only continuous profiles are accepted, with the Laurent form. $[[s_x]]^{-1}$ is kept as a debug switch.

**TVF.** Stretching changes the derivative along the PML normal $\hat n$. An interface whose normal has a component along $\hat n$ would have its normal transformed, and the TVF would have to change with it. Edge padding prevents this: inside an $x$-PML the geometry does not depend on $x$, so every interface there has its normal along $\hat y$, which the $x$-stretch leaves unchanged. With padding, **the TVF is computed from the physical $\varepsilon$ exactly as today.**

---

## Modified P and Q

The derivation of $P$ and $Q$ in [RCWA core](rcwa_core.md) never moves $K_x$ or $K_y$ past $[[\varepsilon]]$, so it carries over verbatim with $K \to \tilde K$:

$$
\begin{align}
P =
\begin{pmatrix}
-\tilde K_x[[\varepsilon]]^{-1}\tilde K_y & -I + \tilde K_x[[\varepsilon]]^{-1}\tilde K_x\\
I - \tilde K_y[[\varepsilon]]^{-1}\tilde K_y & \tilde K_y[[\varepsilon]]^{-1}\tilde K_x
\end{pmatrix},\\
Q =
\begin{pmatrix}
-\tilde K_x\tilde K_y - \Delta[[A_{yx}]] & \tilde K_x^2 - [[\varepsilon]] + \Delta[[A_{yy}]]\\
[[\varepsilon]] - \tilde K_y^2 - \Delta[[A_{xx}]] & \tilde K_y\tilde K_x + \Delta[[A_{xy}]]
\end{pmatrix}.
\end{align}
$$

These are the expressions computed by `compute_P` and `compute_Q0` / `compute_Q` in `layersolver/isotropic.py`, with dense $\tilde K$ in place of the diagonal $K$. The TVF correction ($\Delta$, $[[A_{ij}]]$) describes the $\mathbf D$–$\mathbf E$ relation, not the derivatives, so it is untouched.

**Product order in `compute_Q0`.** The code sets the bottom-right entry as `d = -a`, i.e. $K_xK_y$, which equals $K_yK_x$ only because diagonal matrices commute. This must become `d = Ky @ Kx`.

**Homogeneous layer.** For uniform $\varepsilon$, $P = Q/\varepsilon$ and

$$
\begin{align}
\lambda^2 \in \operatorname{eig}\!\big(\tilde K_x^2 + \tilde K_y^2\big) - \varepsilon .
\end{align}
$$

This generalizes $\lambda^2 = k_x^2 + k_y^2 - \varepsilon$ from [Homogeneous layer](homogeneous.md), but the closed form is lost. The eigenvectors do not depend on $\varepsilon$, so a single basis $W$ serves all homogeneous layers and both half-spaces.

> **TODO:** with a rectangular truncation, $\tilde K_x^2 + \tilde K_y^2$ is a Kronecker sum and diagonalizes from two **1D** eigenproblems of size $2M+1$ and $2N+1$. Work out $W$, $V$.

---

## Illumination: scattered-field formulation

### Why the total field does not work

The PML acts on the **total** field. An oblique plane wave varies along $x$, so the stretch distorts it: it decays on one side of the cell and grows on the other, and the field no longer matches across the cell edge. The single exception is a wave with zero tangential wavevector along the PML axis: since $K_xe_0 = 0$ for $k_{x0} = 0$, also $\tilde K_xe_0 = 0$ and the stretch does nothing. So the total field works only at normal incidence (or for tilt along a non-PML axis). It remains useful as a validation case.

### Splitting the field

Write $\mathbf E = \mathbf E_\text{inc} + \mathbf E_s$:

- **Incident field $\mathbf E_\text{inc}$.** The solution of the **background stack**: the same layers with every patterned layer filled with its void medium. This is a planar stack, so the oblique plane wave is solved exactly with one harmonic (a transfer-matrix calculation), with no PML and at any angle.
- **Scattered field $\mathbf E_s$.** Subtracting the background equations from the full ones leaves Maxwell's equations for $\mathbf E_s$ with a source:

$$
\begin{align}
\nabla\times\mathbf H_s = -j\omega\epsilon_0\,\varepsilon\,\mathbf E_s - j\omega\epsilon_0\,\Delta\varepsilon\,\mathbf E_\text{inc},
\qquad \Delta\varepsilon = \varepsilon - \varepsilon_\text{void}.
\end{align}
$$

$\Delta\varepsilon = (\varepsilon_\text{solid} - \varepsilon_\text{void})\cdot\text{pattern}$ is nonzero only inside the structure, which must lie in the physical region. $\mathbf E_s$ is purely outgoing, which is exactly what the PML absorbs. It is solved with RCWA + PML, unchanged except for the source term:

$$
\begin{align}
\frac{\partial\psi}{\partial z} = k_0\big(A\,\psi + f(z)\big), \qquad A = \begin{pmatrix}0 & P\\ Q & 0\end{pmatrix},
\end{align}
$$

where $f$ is the Fourier transform of $\Delta\varepsilon\,\mathbf E_\text{inc}$. It enters the $u$-equations directly and the $s$-equations through the $[S_z]$ elimination. $\mathbf E_\text{inc}$ is continuous, so the Laurent rule for $[[\Delta\varepsilon\,E_\text{inc}]]$ is exact.

### Where the angle enters

The angle of incidence enters through the source, which carries the phase $e^{j(k_{x0}x + k_{y0}y)}$. The RCWA Bloch phase is a different thing: it only fixes the phase between neighbouring cells. With the PML the cells are decoupled, so the Bloch phase is set to **zero along PML axes**. Along a periodic (non-PML) axis it equals the incident tangential wavevector, because the source is Bloch-periodic there. The phase $e^{jk_{x0}x}$ is not periodic over the cell, but $\Delta\varepsilon$ vanishes before the cell edge, so this does not matter.

---

## S-matrices with sources

### Affine S-matrix

In the convention of [S-matrix algebra](smatrix.md), a layer with an internal source becomes

$$
\begin{align}
\begin{pmatrix}c_L^-\\ c_R^+\end{pmatrix} = S\begin{pmatrix}c_L^+\\ c_R^-\end{pmatrix} + \begin{pmatrix}e_L\\ e_R\end{pmatrix},
\end{align}
$$

where $(e_L, e_R)$ are the waves emitted through the left and right faces when nothing comes in. $S$ is the same matrix as without the source. Layers without the structure (homogeneous layers, half-spaces, boundaries) have $e = 0$.

### Composition

For $S = S_A \star S_B$, the S-matrix part is the ordinary Redheffer star product. Solving for the waves bouncing between $A$ and $B$ gives the source part:

$$
\begin{align}
e_L &= e_{A,L} + S_{A,12}\,(I - S_{B,11}S_{A,22})^{-1}\big(e_{B,L} + S_{B,11}\,e_{A,R}\big),\\
e_R &= e_{B,R} + S_{B,21}\,(I - S_{A,22}S_{B,11})^{-1}\big(e_{A,R} + S_{A,22}\,e_{B,L}\big).
\end{align}
$$

The inverses are the ones `Block2x2.star` already forms, so the extra cost is a few matrix–vector products.

### Source vector of one layer

A layer's S-matrix is built as `S_boundary(gap, layer) ⋆ S_prop ⋆ S_boundary(layer, gap)`, and the source lives only in the middle factor. In the layer's eigenbasis $\Phi = \bigl(\begin{smallmatrix}W & W\\ V & -V\end{smallmatrix}\bigr)$, write $\Phi^{-1}f = (g^+, g^-)$. The modes decouple:

$$
\begin{align}
\partial_z c^+ = k_0(\lambda c^+ + g^+), \qquad \partial_z c^- = k_0(-\lambda c^- + g^-).
\end{align}
$$

Inside a layer, $\mathbf E_\text{inc}$ is one up-going and one down-going wave, so $g^\pm(z)$ is a sum of terms $g\,e^{\gamma k_0 z}$ with $\gamma = \pm jk_{z,\text{bg}}$. Integrating across a layer of thickness $d$ with no incoming amplitude gives, per term,

$$
\begin{align}
e_R = g^+\,\frac{e^{\gamma k_0 d} - e^{\lambda k_0 d}}{\gamma - \lambda}, \qquad
e_L = -\,g^-\,\frac{e^{(\gamma+\lambda)k_0 d} - 1}{\gamma + \lambda}.
\end{align}
$$

$\gamma = \lambda$ is a removable singularity, not a resonance: evaluate with the stable $\varphi_1(x) = (e^x - 1)/x$. Each background wave is referenced to the face where it enters the layer, so all exponents stay bounded. $(e_L, e_R)$ is then carried through the two boundary factors by the composition rule above.

**Polarization.** $S$ does not depend on polarization, but $e$ does, through $\mathbf E_\text{inc}$. Each $e$ is computed for s and p and stored as two columns, so `Observables` can still apply $(a_s, a_p)$ downstream, as today.

---

## How it fits the pipeline

| Stage | Module | Effect of PML |
|-------|--------|---------------|
| Lattice | `model/lattice.py` | Must be rectangular; raise otherwise. |
| Geometry | `model/layer.py` | Edge-pad the pattern into the PML; check that the structure does not reach the PML. |
| Harmonics | `solver/harmonics.py` | Bloch phase 0 along PML axes. New: build $\tilde K_x$, $\tilde K_y$. |
| `P`, `Q` assembly | `layersolver/isotropic.py` | Pass dense $\tilde K$ instead of diagonal $K$. Fix `d = Ky @ Kx` in `compute_Q0`. |
| TVF | `tvf/` | Unchanged (given edge padding). |
| Patterned modes | `layersolver/eigsolver.py`, `matexpsolver.py` | Unchanged, but `matexp` is the default under PML (`Config.modesolver=None`). The stretched operator is strongly non-normal, so the eigenvector matrix of a patterned layer becomes numerically singular in large cells ($\mathrm{cond} \sim 10^{12}$–$10^{17}$ for a 16-wavelength cell), and `eig` results become erratic; `matexp` needs no eigenvectors. The matexp slice-count bound uses the $\infty$-norm of $\tilde K$. |
| Homogeneous modes | `layersolver/base.py` | Closed form lost: $P = Q_0/\varepsilon$ solved by `eigsolver`. |
| Background `W0`, `V0` | `layersolver/base.py` | Unchanged (unstretched vacuum): the zero-thickness gap is exact for any basis; checked to agree with a stretched gap to $\sim 10^{-12}$. |
| Background solve | new | 1-harmonic transfer matrix of the background stack, giving $\mathbf E_\text{inc}$ per layer. |
| Source vectors | new, in `smatrix()` / `run()` | $f$ from the FFT of $\Delta\varepsilon\,e^{j\mathbf k_\parallel\cdot\mathbf r}$ (a vector, not a matrix), then $(e_L, e_R)$ per patterned layer. |
| Star product | `solver/blockmatrix.py` | Extended to carry source vectors. |
| Observables | `observables/base.py` | **TODO:** total = background (analytic) + scattered. Scattered waves are modes of the PML'd half-space, not diffraction orders. |

The expensive/cheap split is preserved: eigendecompositions stay in `prepare()`, while everything that depends on the incident field (background solve, source vectors) is cheap and happens in `run()`.

---

## Proposed API

The PML is a property of the supercell, so it is attached to the `Stack` next to the `Lattice`. A single `PML` object selects the implementation, so that other routes (e.g. a uniaxial PML material) can be added later without changing user code.

```python
from metarcwa import Lattice, Stack, Layer, Source, Model, Solver, Config
from metarcwa import PML, StretchAbsorbProfile, PolynomialProfile          # new

pml = PML(
    width=(0.5, 0.5),      # PML width per axis (x, y), physical units; 0 disables an axis
    profile=StretchAbsorbProfile(a_max=4.0, p=4.0, sigma_max=1.0),   # default profile
    method="stretch",      # only "stretch" (coordinate stretching) for now
)

# other profiles: a built-in one, or any s(d) wrapped as an nn.Module
pml = PML(width=0.5, profile=PolynomialProfile(sigma_max=2.0, p=3.0))
pml = PML(width=0.5, profile=CallableModule(lambda d: 1 + (1 + 1j) * a * d, a))

stack = Stack(
    incidence=air,
    layers=[Layer(medium_solid=si, medium_void=air, thickness=0.2, shape_fn=disk), ...],
    transmission=glass,
    lattice=Lattice.rectangular(px, py),   # rectangular required when pml is given
    pml=pml,                               # new, optional (default None: periodic as today)
)

model = Model(stack, Source(wavelength=0.6, theta=0.3, phi=0.0))   # Source unchanged
solver = Solver(model, Config(truncation="rectangular"))
solution = solver.run()
```

- **`PML`** (`model/pml.py`): an `nn.Module` like `Lattice`, so its parameters (and the profile's) move with `Model.to()`. `method` selects the implementation, `profile` the stretch $s(d)$. Its `.spec()` produces a frozen `PMLSpec` holding $s_x$, $s_y$ sampled on the grid.
- **`Stack(pml=...)`**: validates that the lattice is rectangular and the widths fit, and edge-pads patterned layers in `spec()`.
- **`Source`**: unchanged. With a PML, its angles define $\mathbf E_\text{inc}$; the Bloch phase along PML axes is set to zero internally.
- **`Config`**: two new switches, both with defaults that need no attention:
  - `pml_factorization: "laurent" | "inverse"` (default `"laurent"`, debug only);
  - `pml_formulation: "scattered" | "total"` (default `"scattered"`; `"total"` only valid at normal incidence, for validation).
- **`ModalSolution`**: gains the source vectors $(e_L, e_R)$ next to `S11`, `S21`.
- **`Observables`**: new PML-aware outputs. **TODO**, see open questions.

Internally, a new `solver/pml.py` builds $\tilde K_x$, $\tilde K_y$ as `Block`s from `PMLSpec`, and `solver/background.py` solves the background stack.

---

## Open questions

1. **Observables.** Which outputs to provide: total reflection and transmission, scattered power, extinction, near or far field?
2. **Profile.** Add the Hugonin–Lalanne nonlinear map (better convergence) as a built-in profile?
3. **Gradients.** Should the PML parameters be differentiable?
4. **Validation.**
   - Laurent vs inverse rule convergence for $[[1/s]]$.
   - Normal incidence: scattered-field vs total-field formulation.
   - Oblique incidence: convergence towards a large periodic supercell, and Mie theory for a single cylinder or sphere.
   - Slab waveguide mode vs the analytic dispersion relation.

---

## References

- E. Silberstein, P. Lalanne, J.-P. Hugonin, Q. Cao, "Use of grating theories in integrated optics," *JOSA A* **18**, 2865 (2001).
- J.-P. Hugonin, P. Lalanne, "Perfectly matched layers as nonlinear coordinate transforms: a generalized formalization," *JOSA A* **22**, 1844 (2005).
- M. Pisarenco, J. Maubach, I. Setija, R. Mattheij, "Aperiodic Fourier modal method in contrast-field formulation for simulation of scattering from finite structures," *JOSA A* **27**, 2423 (2010).
- L. Li, "Use of Fourier series in the analysis of discontinuous periodic structures," *JOSA A* **13**, 1870 (1996).
- fmmax: [github.com/facebookresearch/fmmax](https://github.com/facebookresearch/fmmax) (PML profile, `src/fmmax/pml.py`).
