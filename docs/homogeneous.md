# Eigenproblem solution for a homogeneous layer

## Motivation

RCWA solves Maxwell's equations by expanding all fields and material tensors in
a 2-D Fourier basis over the unit cell, then propagating the solution
layer-by-layer along the $z$-axis. Inside each layer this reduces to an
eigenvalue problem: the $z$-dependence of the coupled Fourier amplitudes is
governed by a matrix operator $\Omega^2$, and the electromagnetic modes of the
layer are its eigenvectors.

For a **patterned** layer — one whose permittivity $\epsilon(\mathbf{r})$ varies
within the unit cell — the Fourier harmonics are mixed by $\Omega^2$, so a full
numerical eigendecomposition is required. A **homogeneous** layer, however, has a
spatially uniform permittivity: every harmonic propagates independently, and each
Fourier mode is *already* an eigenvector of $\Omega^2$. The eigenproblem therefore
admits a **closed-form solution** — no numerical `eig` call is needed. This path
covers all incidence and transmission half-spaces as well as any unpatterned film
in the stack, making it both faster and numerically exact relative to the general
patterned-layer treatment.

---

## Problem formulation

Substituting the Fourier expansions into Maxwell's curl equations and projecting
onto each harmonic yields two decoupled eigenvalue problems of the same form,
one for the electric-field amplitudes and one for the magnetic-field amplitudes:

$$
\begin{align}
\frac{1}{k_0^2}\frac{\partial^2 s}{\partial z^2} &= PQ\, s, \\[6pt]
\frac{1}{k_0^2}\frac{\partial^2 u}{\partial z^2} &= QP\, u,
\end{align}
$$

where $k_0 = \omega/c$ is the free-space wavenumber and $P$, $Q$ are
layer-dependent matrix operators (defined in the following section). The compact
notation $(s;\, u)$ used elsewhere is a shorthand for these two parallel
equations: $\Omega^2 = PQ$ when acting on $s$ and $\Omega^2 = QP$ when acting
on $u$. Both problems share the same eigenvalues. Here $s$ and $u$ are stacked
column vectors of Fourier amplitudes for the transverse electric and magnetic
fields, respectively:

$$
\begin{align}
s = \begin{pmatrix} [S_x] \\ [S_y] \end{pmatrix}, \qquad
u = \begin{pmatrix} [U_x] \\ [U_y] \end{pmatrix}.
\end{align}
$$

Each sub-vector $[F_i]$ collects the Fourier coefficients of field component
$F_i$ ordered lexicographically over the harmonic indices $(m, n)$ from
$(-M, -N)$ to $(M, N)$:

$$
\begin{align}
[F_i] = \begin{pmatrix}
F_{i,\,-M,\,-N} \\
F_{i,\,-M,\,-N+1} \\
\vdots \\
F_{i,\,M,\,N}
\end{pmatrix}.
\end{align}
$$

The coefficients $S_{i;\,m,n}$ and $U_{i;\,m,n}$ arise from the Bloch–Fourier
expansion of the transverse fields. Writing each component as a sum of plane
waves modulated by the incident Bloch momentum $(k_{x,0},\, k_{y,0})$:

$$
\begin{align}
E_i &= e^{j(k_{x,0}\,x\,+\,k_{y,0}\,y)}
       \sum_{m,n} S_{i;\,m,n}(z,\omega)\,
       e^{j(G_{x,m}\,x\,+\,G_{y,n}\,y)}, \\[4pt]
H_i &= j\sqrt{\frac{\epsilon_0}{\mu_0}}\,
       e^{j(k_{x,0}\,x\,+\,k_{y,0}\,y)}
       \sum_{m,n} U_{i;\,m,n}(z,\omega)\,
       e^{j(G_{x,m}\,x\,+\,G_{y,n}\,y)},
\end{align}
$$

where $G_{x,m}$ and $G_{y,n}$ are the Cartesian projections of the
reciprocal-lattice vector $\mathbf{G}_{m,n} = m\,\mathbf{b}_1 + n\,\mathbf{b}_2$,
and the prefactor $j\sqrt{\epsilon_0/\mu_0}$ normalizes $U$ so that $S$ and $U$
carry the same physical dimensions throughout the solver.

Since $s$ and $u$ share the same eigenvalues, it is sufficient to solve only
the first problem (for $s$). The magnetic-field amplitudes $u$ are then
recovered from the first-order relation

$$
\begin{align}
\frac{1}{k_0}\frac{\partial u}{\partial z} = Q\, s.
\end{align}
$$

## P and Q for a homogeneous layer

For isotropic, non-magnetic media the general forms of $P$ and $Q$ are

$$
\begin{align}
P =
\begin{pmatrix}
-K_x [[\varepsilon]]^{-1} K_y &
-I + K_x [[\varepsilon]]^{-1} K_x \\
I - K_y [[\varepsilon]]^{-1} K_y &
K_y [[\varepsilon]]^{-1} K_x
\end{pmatrix},
\end{align}
$$

$$
\begin{align}
Q=
\begin{pmatrix}
-K_{x}K_{y}-\Delta[[T_xT_y^*]]&
K_xK_x-[[\varepsilon]]-\Delta[[|T_x|^2]]\\
[[\varepsilon]]-\Delta[[|T_y|^2]] - K_yK_y&
K_yK_x+\Delta[[T_x^*T_y]]
\end{pmatrix},
\end{align}
$$

where $T_x, T_y$ are the tangential-vector-field components used in Li's
factorization (see the TVF documentation), and
$\Delta = [[\varepsilon]] - [[\frac{1}{\varepsilon}]]^{-1}$
is the difference between the direct and inverse permittivity convolution
matrices. For a homogeneous layer $[[\varepsilon]] = \varepsilon I$ and
$[[\frac{1}{\varepsilon}]]^{-1} = \varepsilon I$, so $\Delta = 0$ — every
$\Delta$-weighted Li correction vanishes. Combined with
$[[\varepsilon]]^{-1} = \frac{1}{\varepsilon} I$, every block reduces to a
diagonal matrix (since $K_x$ and $K_y$ are diagonal), giving

$$
\begin{align}
P =
\dfrac{1}{\varepsilon}\begin{pmatrix}
- K_x K_y &
K_x^2 -\varepsilon I \\[8pt]
\varepsilon I - K_y^2 & K_y K_x
\end{pmatrix},
\qquad
Q =
\begin{pmatrix}
-K_x K_y & K_x^2 - \varepsilon I \\[6pt]
\varepsilon I - K_y^2 & K_y K_x
\end{pmatrix}.
\end{align}
$$

Because every block is diagonal, $P$ and $Q$ couple only within each harmonic
$(m, n)$ and not across harmonics — which is precisely what makes the
homogeneous eigenproblem solvable in closed form.

## Solving the problem

In the normalized coordinate $\tilde{z} = k_0 z$ the $s$-eigenproblem reads
$\partial^2 s / \partial \tilde{z}^2 = PQ\, s$. The exponential ansatz
$s \propto e^{\pm\lambda\tilde{z}}$ turns it into the matrix eigenvalue problem

$$
\begin{align}
PQ\, W = W\, \Lambda,
\end{align}
$$

where $\Lambda = \mathrm{diag}(\lambda^2)$ is the diagonal matrix of squared
modal exponents and the columns of $W$ are the eigenvectors (layer modes).

To evaluate $PQ$ explicitly, recall that for the homogeneous layer
$P = \frac{1}{\varepsilon}M$ and $Q = M$ with the same matrix $M$ (derived
above). Since $K_x$ and $K_y$ are diagonal they commute, and $M^2$ reduces to a
scalar multiple of the identity:

$$
\begin{align}
PQ = \frac{1}{\varepsilon}\,M^2 =
\begin{pmatrix}
K_x^2 + K_y^2 - \varepsilon I & 0 \\[6pt]
0 & K_x^2 + K_y^2 - \varepsilon I
\end{pmatrix}.
\end{align}
$$

Because $PQ$ is already diagonal the eigenvalues are immediate —

$$
\begin{align}
\Lambda = \mathrm{diag}\!\left(k_{x,mn}^2 + k_{y,mn}^2 - \varepsilon\right),
\end{align}
$$

one entry per harmonic $(m,n)$ — and we may take $W = I$. Every Fourier harmonic
is independently an eigenmode, confirming the closed-form argument from the
Motivation.

The magnetic-mode matrix $V$ follows from the first-order relation
$(1/k_0)\,\partial u/\partial z = Q\,s$. Substituting the modal forms
$s = W e^{\lambda\tilde{z}}$ and $u = V e^{\lambda\tilde{z}}$ column-wise yields
$\lambda V = QW$, so

$$
\begin{align}
V = QW\,\lambda^{-1} = Q\,\lambda^{-1},
\end{align}
$$

where $\lambda = \mathrm{diag}(\sqrt{\lambda^2})$ is the diagonal matrix of
(signed) modal exponents and $\lambda^{-1} = \mathrm{diag}(1/\lambda_{mn})$
scales each column of $Q$ by the reciprocal of its exponent. The matrices $W$
and $V$ together fully characterize the electromagnetic modes of the homogeneous
layer — the closed-form counterpart to the numerical eigendecomposition required
for patterned layers.

---

## Implementation details

The closed-form solution derived above is implemented in
`src/metarcwa/solver/homogeneous.py` through three public functions.

**Tensor layout.** All wavevectors are $k_0$-normalised. `epsilon` has shape
`[N_wl, ...]` and `kx`, `ky` have shape `[..., Nh]`, where `Nh` is the number
of Fourier harmonics. Because each harmonic contributes both a forward and a
backward mode, every output carries `2Nh` modes, organised as two duplicated
blocks along the last axis.

**`homogeneous_kz` — modal exponents.** Evaluates $\lambda^2 = k_x^2 + k_y^2 - \varepsilon$ per harmonic (equivalently $k_z^2 = \varepsilon - k_x^2 -
k_y^2$), then sets $\lambda = 1j \cdot k_z$. Under the $exp(−j\omega t)$
convention used throughout this module, the branch sign is chosen so that

- **propagating modes** ($|\mathrm{Im}(k_z)| \leq \texttt{tol}$): $\mathrm{Re}(k_z) > 0$,
- **evanescent modes** ($|\mathrm{Im}(k_z)| > \texttt{tol}$): $\mathrm{Im}(k_z) > 0$ (decaying in $+z$).

```python
lam2 = kx**2 + ky**2 - epsilon          # λ² per harmonic, duplicated → [..., 2N]
lam  = torch.sqrt(lam2)
kz   = -1j * lam                        # kz (sign adjusted by branch rule)
# ... branch sign correction ...
lam  = 1j * kz                          # forward field ∝ exp(+lam·z̃)
```

Pass `forward="negative"` to select the backward-propagating branch.

**`homogeneous_Q` — the $Q$ matrix.** Assembles the homogeneous block form of
$Q$ derived above. Since all four $N\times N$ blocks are diagonal, they are
built with `torch.diag_embed` of the per-harmonic scalar products:

```python
Q11 = torch.diag_embed(-kx * ky)           # block (1,1): -Kx Ky
Q12 = torch.diag_embed(kx**2 - epsilon)    # block (1,2): Kx² - εI
Q21 = torch.diag_embed(epsilon - ky**2)    # block (2,1): εI - Ky²
Q22 = torch.diag_embed(ky * kx)            # block (2,2): Ky Kx
```

Output shape `[..., 2N, 2N]`.

**`homogeneous_modes` — full modal decomposition.** Combines the two functions
above to return $(\lambda,\, W,\, V)$:

```python
kz  = homogeneous_kz(epsilon, kx, ky, forward=forward)
lam = 1j * kz                                     # [..., 2N]
W   = torch.eye(2N).expand(*batch, 2N, 2N)        # W = I
Q   = homogeneous_Q(epsilon, kx, ky)              # [..., 2N, 2N]
V   = Q * (1.0 / lam).unsqueeze(-2)               # V = Q diag(1/λ)
```

The column scaling `Q * (1/lam).unsqueeze(-2)` is the efficient, allocation-free
implementation of $V = Q\lambda^{-1}$ without forming the full diagonal matrix.
The caller is responsible for assembling the gap (boundary-condition) matrix

$$
\begin{align}
\Phi_0 = \begin{pmatrix} W & W \\ V & -V \end{pmatrix}.
\end{align}
$$

**Grazing incidence.** When $k_{x,mn}^2 + k_{y,mn}^2 = \varepsilon$ exactly,
$\lambda_{mn} = 0$ and the corresponding column of $V$ is undefined (NaN). A
`RuntimeWarning` is emitted; avoid exact grazing or handle it upstream.

**Practical properties.** No call to `torch.linalg.eig` is made — $W$, $V$, and
$\Lambda$ follow in closed form from diagonal arithmetic. All three functions are
fully autograd-differentiable through `epsilon`, `kx`, and `ky`, and run on both
CPU and CUDA. The test suite `tests/test_homogeneous.py` pins the dispersion
relation $k_z^2 = \varepsilon - k_x^2 - k_y^2$, the identity $W = I$, the
column-scaling relation $V = Q\,\mathrm{diag}(1/\lambda)$, the correct block
signs in $Q$, and the grazing-mode warning.

---

## Conclusion

For a homogeneous layer the Fourier harmonic basis diagonalizes the layer
operator $PQ$, and the entire modal decomposition follows in closed form:
eigenvalues $\lambda^2 = k_x^2 + k_y^2 - \varepsilon$ per harmonic,
electric-field mode matrix $W = I$, and magnetic-field mode matrix
$V = Q\lambda^{-1}$. No numerical eigensolver is required.

The practical gains are significant: the solution is exact (no rounding in the
eigensolver), computationally cheap (diagonal arithmetic instead of a dense
$2N \times 2N$ `eig`), and covers all incidence and transmission half-spaces as
well as any uniform film in the stack. For patterned layers — where the
permittivity varies within the unit cell and $\Omega^2$ is no longer diagonal in
the harmonic basis — the full numerical eigenproblem must be solved. The
homogeneous result derived here serves as the fundamental building block at those
interfaces and as the reference case for validating the general solver.