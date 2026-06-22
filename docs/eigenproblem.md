# Eigenvalue problem

## Overview

One standard approach to solving the RCWA equations within a single layer is to eliminate either $s$ or $u$ from the coupled first-order system (Eqs. (37)–(38) in [RCWA Core](rcwa_core.md)), yielding a single second-order equation:

$$
\begin{align}
\frac{1}{k_0^2}\frac{\partial^2 s}{\partial z^2} &= PQ\,s, \\[6pt]
\frac{1}{k_0^2}\frac{\partial^2 u}{\partial z^2} &= QP\,u.
\end{align}
$$

The two problems share the same eigenvalues. It is therefore sufficient to solve Eq. (1) for $s$ and then recover $u$ from the first-order relation (Eq. (38) in [RCWA Core](rcwa_core.md)).

## Eigendecomposition

Since the matrix $PQ$ does not depend on $z$, the solution of Eq. (1) has a simple exponential form. Substituting the ansatz $s \propto e^{\pm\lambda k_0 z}$ reduces the problem to the algebraic eigenvalue problem:

$$
\begin{align}
PQ\,W = W\,\Lambda,
\end{align}
$$

where the columns of $W$ are the eigenvectors (layer modes) and $\Lambda = \mathrm{diag}(\lambda^2)$ is the diagonal matrix of squared modal exponents. The general solution for $s$ is then:

$$
\begin{align}
s = W\,e^{\Lambda^{1/2} k_0 z}\,c^+ + W\,e^{-\Lambda^{1/2} k_0 z}\,c^-,
\end{align}
$$

where $c^+$ and $c^-$ are the column vectors of forward- and backward-propagating modal amplitudes, respectively. Using Eq. (38) from [RCWA Core](rcwa_core.md) to obtain the magnetic-field amplitudes gives:

$$
\begin{align}
u = V\,e^{\Lambda^{1/2} k_0 z}\,c^+ - V\,e^{-\Lambda^{1/2} k_0 z}\,c^-,
\end{align}
$$

where $V = QW\Lambda^{-1/2}$ follows directly from the first-order relation $\partial u/\partial(k_0 z) = Qs$.

## $\psi(z)$ assembly

As the final step, the solutions for $s$ and $u$ from Eqs. (4) and (5) are combined into the full transverse field vector $\psi(z)$, which is the input to the subsequent S-matrix assembly (see [S-matrix algebra](smatrix.md)):

$$
\begin{align}
\psi(z) = 
\begin{pmatrix}
s(z)\\ u(z)
\end{pmatrix}=
\underbrace{\begin{pmatrix}
W & W\\
V & -V
\end{pmatrix}}_{\Phi}
\begin{pmatrix}
e^{\Lambda^{1/2} k_0 z} & 0\\
0 & e^{-\Lambda^{1/2} k_0 z}
\end{pmatrix}
\begin{pmatrix}
c^+\\ c^-
\end{pmatrix}.
\end{align}
$$

The first factor $\Phi = \bigl(\begin{smallmatrix} W & W \\ V & {-V} \end{smallmatrix}\bigr)$ is the **gap matrix** (or mode matrix) of the layer. The second factor is the diagonal propagator that advances each mode by its complex phase over a distance $z$. Together they give $\psi$ as a function of $z$ for given modal amplitudes $(c^+, c^-)$.

## Practical notes

- For a **patterned layer**, $PQ$ is a dense $2N_h \times 2N_h$ matrix (where $N_h = (2M+1)(2N+1)$ is the number of retained harmonics) and a numerical eigendecomposition is required.
- For a **homogeneous layer**, $PQ$ is already diagonal in the harmonic basis and the eigendecomposition admits a closed-form solution — no numerical `eig` call is needed. This case is treated in detail in [Homogeneous layer](homogeneous.md).
