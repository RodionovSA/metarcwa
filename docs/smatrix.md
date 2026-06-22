# S-matrix algebra

## Overview

The S-matrix (scattering matrix) of a region maps the **incoming** modal amplitudes on both sides to the **outgoing** modal amplitudes on both sides. Concretely, using the $\psi(z)$ decomposition from [RCWA Core](rcwa_core.md) and [Eigenvalue problem](eigenproblem.md), we label amplitudes by propagation direction: $c^+$ travels in the $+z$ direction and $c^-$ travels in the $-z$ direction. For a region with a left face (L) and a right face (R), the S-matrix relation is:

$$
\begin{align}
\begin{pmatrix} c_L^- \\ c_R^+ \end{pmatrix} = S \begin{pmatrix} c_L^+ \\ c_R^- \end{pmatrix},
\end{align}
$$

so $S_{11}$ is the reflection from the left, $S_{22}$ from the right, and $S_{12}$, $S_{21}$ are the two transmission coefficients.

The S-matrix is preferred over the transfer matrix (T-matrix) for numerical stability. The T-matrix propagates all amplitudes from one face to the other, which requires dividing by exponentially decaying evanescent factors for thick layers; this causes catastrophic numerical cancellation. The S-matrix avoids this: its blocks remain $O(1)$ regardless of layer thickness, making it unconditionally stable.

This document derives the S-matrix for (1) an interface between two layers, (2) a homogeneous propagation layer of finite thickness, and (3) the Redheffer star product used to cascade multiple S-matrices into a single device-level matrix.

---

## S-matrix for a boundary

Let us consider a boundary between two adjacent layers. In each layer we have found $\psi(z)$ and now we need to apply boundary conditions. Since $\psi$ is built from lateral components only and represents a complete field state, we have a single continuity equation for it:

$$
\begin{align}
\psi_L(z_0) = \psi_R(z_0),
\end{align}
$$

where $\psi_L$ is the field vector on the left side of the boundary and $\psi_R$ on the right side. Since we can choose any $z_0$, let us set $z_0 = 0$. Substituting the modal decomposition $\psi = \Phi \begin{pmatrix} c^+ \\ c^- \end{pmatrix}$ with gap matrix $\Phi = \bigl(\begin{smallmatrix} W & W \\ V & -V \end{smallmatrix}\bigr)$, Eq. (2) becomes:

$$
\begin{align}
\begin{pmatrix}
W_L & W_L\\
V_L & -V_L
\end{pmatrix}
\begin{pmatrix}
c_L^+\\
c_L^-
\end{pmatrix}=
\begin{pmatrix}
W_R & W_R\\
V_R & -V_R
\end{pmatrix}
\begin{pmatrix}
c_R^+\\
c_R^-
\end{pmatrix}.
\end{align}
$$

By definition, the S-matrix relates outgoing to incoming amplitudes:

$$
\begin{align}
\begin{pmatrix}
c_L^-\\
c_R^+
\end{pmatrix}=
\begin{pmatrix}
S_{11} & S_{12}\\
S_{21} & S_{22}
\end{pmatrix}
\begin{pmatrix}
c_L^+\\
c_R^-
\end{pmatrix},
\end{align}
$$

where $S_{ij}$ are the S-matrix components. Rearranging Eq. (3) to group outgoing amplitudes on the left and incoming on the right gives:

$$
\begin{align}
\begin{pmatrix}
W_L & -W_R\\
V_L & V_R
\end{pmatrix}
\begin{pmatrix}
c_L^-\\
c_R^+
\end{pmatrix}=
\begin{pmatrix}
-W_L & W_R\\
V_L & V_R
\end{pmatrix}
\begin{pmatrix}
c_L^+\\
c_R^-
\end{pmatrix}.
\end{align}
$$

To invert the left-hand matrix $M = \begin{pmatrix} W_L & -W_R \\ V_L & V_R \end{pmatrix}$, we form its two Schur complements:

$$
\begin{align}
S_D &= W_L + W_R V_R^{-1} V_L \quad \text{(Schur complement of } V_R \text{)}, \\
S_A &= V_R + V_L W_L^{-1} W_R \quad \text{(Schur complement of } W_L \text{)}.
\end{align}
$$

Applying the block matrix inversion formula — row 1 via $S_D$, row 2 via $S_A$ — yields:

$$
\begin{align}
M^{-1} =
\begin{pmatrix}
S_D^{-1} & S_D^{-1} W_R V_R^{-1}\\
-S_A^{-1} V_L W_L^{-1} & S_A^{-1}
\end{pmatrix}.
\end{align}
$$

Right-multiplying by $N = \begin{pmatrix} -W_L & W_R \\ V_L & V_R \end{pmatrix}$ gives the boundary S-matrix:

$$
\begin{align}
S_b = M^{-1} N =
\begin{pmatrix}
I - 2S_D^{-1} W_L & 2S_D^{-1} W_R\\
2S_A^{-1} V_L & 2S_A^{-1} V_R - I
\end{pmatrix}.
\end{align}
$$

---

## S-matrix for a layer

A homogeneous propagation layer of thickness $d$ contains no interfaces, so there is no partial reflection — the S-matrix is a pure propagator. Placing the origin at the left boundary ($z = 0$) and the right boundary at $z = d$, the modal field is:

$$
\psi(z) = \Phi \begin{pmatrix} e^{\Lambda^{1/2} k_0 z} & 0\\ 0 & e^{-\Lambda^{1/2} k_0 z} \end{pmatrix} \begin{pmatrix} c^+ \\ c^- \end{pmatrix},
$$

where $\Lambda^{1/2} = \mathrm{diag}(\lambda_{mn})$ is the diagonal matrix of modal exponents from [Eigenvalue problem](eigenproblem.md). Define the propagation factor:

$$
\begin{align}
X_d = \exp\!\bigl(\Lambda^{1/2}\,k_0\,d\bigr).
\end{align}
$$

A forward mode ($c^+$) arriving at the left face exits the right face scaled by $X_d$; a backward mode ($c^-$) arriving at the right face exits the left face scaled by $X_d$. There is no reflection, so:

$$
\begin{align}
S_l = \begin{pmatrix} 0 & X_d \\ X_d & 0 \end{pmatrix}.
\end{align}
$$

The matrix $X_d$ is diagonal with entries $\exp(\lambda_{mn} k_0 d)$. Following the branch convention of [Eigenvalue problem](eigenproblem.md) and [Homogeneous layer](homogeneous.md) — where the forward field is $\propto e^{+\lambda_{mn} k_0 z}$ — propagating modes have $\mathrm{Re}(\lambda_{mn}) = 0$, $\mathrm{Im}(\lambda_{mn}) > 0$, while evanescent modes have $\mathrm{Re}(\lambda_{mn}) < 0$ (decaying in $+z$). Hence every entry of $X_d$ satisfies $|X_d| \leq 1$ — the S-matrix remains bounded regardless of layer thickness.

### Layer embedded in vacuum

The full S-matrix of a layer (mode matrices $W$, $V$; propagation factor $X_d$) embedded between two vacuum half-spaces (mode matrices $W_0$, $V_0$) is obtained by cascading two boundary S-matrices with the propagation S-matrix:

$$
S = S_b^{(L)} \star S_l \star S_b^{(R)},
$$

where $S_b^{(L)}$ is the boundary S-matrix at the left interface (vacuum $\to$ layer) and $S_b^{(R)}$ is the boundary S-matrix at the right interface (layer $\to$ vacuum). Applying the boundary formula from above with the respective left/right labels, define the four Schur complements:

$$
\begin{align}
S_D^{(L)} &= W_0 + W V^{-1}V_0, \qquad S_A^{(L)} = V + V_0 W_0^{-1} W, \\
S_D^{(R)} &= W + W_0 V_0^{-1} V, \qquad S_A^{(R)} = V_0 + V W^{-1} W_0,
\end{align}
$$

and the eight boundary blocks:

$$
\begin{align}
r_L &= I - 2\bigl(S_D^{(L)}\bigr)^{-1}W_0, & t_L &= 2\bigl(S_D^{(L)}\bigr)^{-1}W, \\
t_L' &= 2\bigl(S_A^{(L)}\bigr)^{-1}V_0, & r_L' &= 2\bigl(S_A^{(L)}\bigr)^{-1}V - I, \\
r_R &= I - 2\bigl(S_D^{(R)}\bigr)^{-1}W, & t_R &= 2\bigl(S_D^{(R)}\bigr)^{-1}W_0, \\
t_R' &= 2\bigl(S_A^{(R)}\bigr)^{-1}V, & r_R' &= 2\bigl(S_A^{(R)}\bigr)^{-1}V_0 - I.
\end{align}
$$

**Step 1 — $S_b^{(L)} \star S_l$.** Because $S_l$ has zero diagonal blocks ($S_{l,11} = S_{l,22} = 0$), every Redheffer denominator $(I - S_{l,11}\,\mathcal{A}_{22})$ and $(I - \mathcal{A}_{22}\,S_{l,11})$ collapses to $I$. The four Redheffer formulas reduce to:

$$
\begin{align}
S_b^{(L)} \star S_l =
\begin{pmatrix}
r_L & t_L\,X_d \\[4pt]
X_d\,t_L' & X_d\,r_L'\,X_d
\end{pmatrix}.
\end{align}
$$

**Step 2 — $(S_b^{(L)} \star S_l) \star S_b^{(R)}$.** Let $\mathcal{A} = S_b^{(L)} \star S_l$ with blocks $\mathcal{A}_{11} = r_L$, $\mathcal{A}_{12} = t_L X_d$, $\mathcal{A}_{21} = X_d t_L'$, $\mathcal{A}_{22} = X_d r_L' X_d$. Applying the Redheffer formula with the denominators

$$
\begin{align}
F = \bigl(I - r_R\,X_d\,r_L'\,X_d\bigr)^{-1}, \qquad
\tilde{F} = \bigl(I - X_d\,r_L'\,X_d\,r_R\bigr)^{-1},
\end{align}
$$

gives the full layer S-matrix:

$$
\begin{align}
S_{11} &= r_L + t_L\,X_d\,F\,r_R\,X_d\,t_L', \\
S_{12} &= t_L\,X_d\,F\,t_R, \\
S_{21} &= t_R'\,\tilde{F}\,X_d\,t_L', \\
S_{22} &= r_R' + t_R'\,\tilde{F}\,X_d\,r_L'\,X_d\,t_R.
\end{align}
$$

The factors $F$ and $\tilde{F}$ are the matrix analogue of the Fabry-Pérot denominator $(1 - r_1 r_2 e^{2j\phi})^{-1}$: they sum the geometric series of multiple reflections between the two interfaces of the layer.

**Symmetry of the result.** Because the layer is bounded by the *same* medium (vacuum) on both sides, the structure is mirror-symmetric about its midplane $z = d/2$: the reflection $z \to d - z$ maps it onto itself while exchanging the left and right ports. Invariance under this port exchange requires the S-matrix to be unchanged when ports $1$ and $2$ are swapped, which forces both

$$
\begin{align}
S_{11} = S_{22}, \qquad S_{12} = S_{21}.
\end{align}
$$

The transmission equality $S_{12} = S_{21}$ is the statement that a symmetric slab transmits identically in both directions. The two expressions $S_{12} = t_L\,X_d\,F\,t_R$ and $S_{21} = t_R'\,\tilde{F}\,X_d\,t_L'$ are connected by the push-through identity $A(I - BA)^{-1} = (I - AB)^{-1}A$, which moves $X_d$ and the Fabry-Pérot denominator through one another. In the scalar 1D limit ($W_0 = V_0 = 1$, $W = n$, $V = 1/n$) the blocks collapse to the familiar closed forms

$$
\begin{align}
S_{11} = S_{22} = \frac{\rho\,(1 - X_d^2)}{1 - \rho^2 X_d^2}, \qquad
S_{12} = S_{21} = \frac{\tau^2\,X_d}{1 - \rho^2 X_d^2},
\end{align}
$$

with $\rho = (n^2 - 1)/(n^2 + 1)$ and $\tau = 2n/(n^2 + 1)$ (satisfying $\rho^2 + \tau^2 = 1$) — the standard Fabry-Pérot reflection and transmission coefficients of a symmetric slab, which confirm the symmetry explicitly.

---

## Redheffer star product

The Redheffer star product $S_A \star S_B$ combines two consecutive S-matrices — $S_A$ for the left sub-stack and $S_B$ for the right sub-stack — into a single composite S-matrix. With the convention of Eq. (1):

$$
\begin{align}
S = S_A \star S_B,
\end{align}
$$

$$
\begin{align}
S_{11} &= S_{A,11} + S_{A,12}(I - S_{B,11}\,S_{A,22})^{-1}S_{B,11}\,S_{A,21}, \\
S_{12} &= S_{A,12}(I - S_{B,11}\,S_{A,22})^{-1}S_{B,12}, \\
S_{21} &= S_{B,21}(I - S_{A,22}\,S_{B,11})^{-1}S_{A,21}, \\
S_{22} &= S_{B,22} + S_{B,21}(I - S_{A,22}\,S_{B,11})^{-1}S_{A,22}\,S_{B,12}.
\end{align}
$$

The factors $(I - S_{B,11}S_{A,22})^{-1}$ and $(I - S_{A,22}S_{B,11})^{-1}$ represent the geometric series of inter-region multiple reflections. Because all S-matrix blocks are $O(1)$, these inverses are well-conditioned even when the individual layers are optically thick.

The star product is **associative** but not commutative. The S-matrix for a complete multilayer stack is assembled by sweeping from left to right:

$$
S_\text{stack} = S_{b_0} \star S_{l_1} \star S_{b_1} \star S_{l_2} \star \cdots \star S_{b_{N-1}} \star S_{l_N} \star S_{b_N},
$$

alternating boundary S-matrices $S_{b_i}$ (the boundary matrix $S_b$ derived above) with layer propagation S-matrices $S_{l_i}$ (the propagation matrix $S_l$).
