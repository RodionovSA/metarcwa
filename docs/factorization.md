# Factorization rules

When transitioning from the theoretical formulation to numerical simulation, the question of fast and stable convergence becomes critical. In any method operating in the Fourier domain, it is essential that the underlying real-space functions are continuous and sufficiently smooth. Because we work with truncated Fourier series and retain only a finite number of harmonics, real-space discontinuities produce broad spectral content that slows convergence and may lead to numerical instabilities. This makes the choice of an appropriate factorization scheme crucial for maintaining both accuracy and stability.

From Maxwell's equations in the frequency domain we obtain four field quantities: $\mathbf{H}$, $\mathbf{B}$, $\mathbf{E}$, and $\mathbf{D}$. For isotropic, non-magnetic media, $\mathbf{B} = \mu_0\mathbf{H}$, so both $\mathbf{B}$ and $\mathbf{H}$ are continuous across material interfaces. In contrast, $\mathbf{D} = \varepsilon_0\varepsilon\mathbf{E}$ (where $\varepsilon$ is a complex-valued function), and the components of $\mathbf{E}$ and $\mathbf{D}$ can be either continuous or discontinuous. Specifically, the normal component of $\mathbf{D}$ is continuous across a permittivity interface, whereas its tangential components are generally discontinuous. Conversely, the tangential components of $\mathbf{E}$ remain continuous across the interface, while the normal component jumps by a factor proportional to the permittivity contrast. These field continuity conditions dictate how the components should be represented in the Fourier domain and, consequently, determine the appropriate factorization scheme. The factorization rules used in RCWA were first rigorously formulated by Li in [JOSA A, 13, 1870 (1996)](https://opg.optica.org/josaa/fulltext.cfm?uri=josaa-13-9-1870). The rules state that if $\mathbf{D}(x,y) = \varepsilon_0\varepsilon(x,y)\mathbf{E}(x,y)$ holds in real space, the corresponding Fourier-domain representation must satisfy the following conditions:

1. If either $\varepsilon(x,y)$ or $E_i(x,y)$ is continuous at $(x_0,y_0)$ (and the other may be discontinuous), the **Laurent rule** applies:

$$
\begin{align}
[D_i]=[[\varepsilon]][S_i],
\end{align}
$$

where $[S_i]$ is the column vector of Fourier coefficients of $E_i$, $[D_i]$ is the column vector of Fourier coefficients of $D_i$, and $[[\varepsilon]]$ is the Toeplitz convolution matrix of $\varepsilon$ (see [RCWA Core](rcwa_core.md) for full definitions).

2. If both $\varepsilon(x,y)$ and $E_i(x,y)$ are discontinuous at $(x_0,y_0)$ but their product $\varepsilon(x,y)E_i(x,y)$ is continuous, the **inverse rule** applies:

$$
\begin{align}
[D_i]=[[\varepsilon^{-1}]]^{-1}[S_i].
\end{align}
$$

3. If all three quantities are simultaneously discontinuous, neither rule applies and the product cannot be accurately formed using a finite Fourier convolution.

These rules motivate the introduction of a basis that is continuous everywhere. Let us consider a general periodic surface $f(x,y)=0$ separating regions with different $\varepsilon$. On this surface we define two vector fields: $\mathbf{N}(x,y)$, normal to the surface, and $\mathbf{T}(x,y)$, tangential to it. Together they define a local coordinate transformation:

$$
\begin{align}
\begin{pmatrix}
E_t \\
D_N
\end{pmatrix}=
\begin{pmatrix}
T_x & T_y \\
\varepsilon N_x & \varepsilon N_y
\end{pmatrix}
\begin{pmatrix}
E_x \\
E_y
\end{pmatrix},
\end{align}
$$

where $E_t$ is the tangential component of the electric field and $D_N$ is the normal component of the electric displacement field (normalized to $\varepsilon_0$). The pair $(E_t, D_N)$ is continuous everywhere in the cross-section. The fields $\mathbf{N}(x,y)$ and $\mathbf{T}(x,y)$ are defined on the surface $f(x,y)=0$ and extended to the full plane under the rule that they must be continuous on the interface but may be discontinuous where $\varepsilon$ is continuous.

Henceforth we work exclusively with $\mathbf{T}(x,y)$, from which the normal field follows as:

$$
\begin{align}
\mathbf{N}(x,y)=(-T_y^*, T_x^*).
\end{align}
$$

Substituting Eq. (4) into Eq. (3) gives:

$$
\begin{align}
\begin{pmatrix}
E_t \\
D_N
\end{pmatrix}=
\begin{pmatrix}
T_x & T_y \\
-\varepsilon T_y^* & \varepsilon T_x^*
\end{pmatrix}
\begin{pmatrix}
E_x \\
E_y
\end{pmatrix}.
\end{align}
$$

To derive the expression for $(D_x, D_y)$, we invert Eq. (5):

$$
\begin{align}
\begin{pmatrix}
E_x \\
E_y
\end{pmatrix}=
\begin{pmatrix}
T_x^* & -\dfrac{T_y}{\varepsilon} \\[6pt]
T_y^* & \dfrac{T_x}{\varepsilon}
\end{pmatrix}
\begin{pmatrix}
E_t \\
D_N
\end{pmatrix}.
\end{align}
$$

Multiplying Eq. (6) by $\varepsilon$:

$$
\begin{align}
\begin{pmatrix}
\varepsilon E_x \\
\varepsilon E_y
\end{pmatrix}=
\begin{pmatrix}
\varepsilon T_x^* & -T_y \\
\varepsilon T_y^* & T_x
\end{pmatrix}
\begin{pmatrix}
E_t \\
D_N
\end{pmatrix}.
\end{align}
$$

Since $(E_t, D_N)$ is continuous everywhere, the Laurent rule applies to both Eqs. (6) and (7):

$$
\begin{align}
\begin{pmatrix}
[\varepsilon E_x] \\
[\varepsilon E_y]
\end{pmatrix}=
\begin{pmatrix}
[[\varepsilon T_x^*]] & -[[T_y]] \\
[[\varepsilon T_y^*]] & [[T_x]]
\end{pmatrix}
\begin{pmatrix}
[E_t] \\
[D_N]
\end{pmatrix},
\end{align}
$$

$$
\begin{align}
\begin{pmatrix}
[S_x] \\
[S_y]
\end{pmatrix}=
\begin{pmatrix}
[[T_x^*]] & -\left[[\dfrac{T_y}{\varepsilon}]\right] \\[6pt]
[[T_y^*]] & \left[[\dfrac{T_x}{\varepsilon}]\right]
\end{pmatrix}
\begin{pmatrix}
[E_t] \\
[D_N]
\end{pmatrix}.
\end{align}
$$

Using the key property of $\mathbf{T}(x,y)$ — that it is continuous on the material boundary and smooth elsewhere — the mixed Toeplitz products in Eqs. (8) and (9) factorize as $[[\varepsilon T_i^*]] \approx [[\varepsilon]][[T_i^*]]$ and $[[T_i/\varepsilon]] \approx [[\varepsilon^{-1}]][[T_i]]$:

$$
\begin{align}
\begin{pmatrix}
[\varepsilon E_x] \\
[\varepsilon E_y]
\end{pmatrix}=
\begin{pmatrix}
[[\varepsilon]][[T_x^*]] & -[[T_y]] \\
[[\varepsilon]][[T_y^*]] & [[T_x]]
\end{pmatrix}
\begin{pmatrix}
[E_t] \\
[D_N]
\end{pmatrix},
\end{align}
$$

$$
\begin{align}
\begin{pmatrix}
[S_x] \\
[S_y]
\end{pmatrix}=
\begin{pmatrix}
[[T_x^*]] & -[[\varepsilon^{-1}]][[T_y]] \\
[[T_y^*]] & [[\varepsilon^{-1}]][[T_x]]
\end{pmatrix}
\begin{pmatrix}
[E_t] \\
[D_N]
\end{pmatrix}.
\end{align}
$$

Inverting Eq. (11) and substituting into Eq. (10) yields an explicit $[\mathbf{D}]$–$[\mathbf{S}]$ relation. After eliminating the intermediate basis $(E_t, D_N)$ we obtain:

$$
\begin{align}
\begin{pmatrix}
[\varepsilon E_x] \\
[\varepsilon E_y]
\end{pmatrix}=
\begin{pmatrix}
[[\varepsilon]][[T_x^*]] & -[[T_y]] \\
[[\varepsilon]][[T_y^*]] & [[T_x]]
\end{pmatrix}
\begin{pmatrix}
[[T_x]] & [[T_y]] \\
-[[\varepsilon^{-1}]]^{-1}[[T_y^*]] & [[\varepsilon^{-1}]]^{-1}[[T_x^*]]
\end{pmatrix}
\begin{pmatrix}
[S_x] \\
[S_y]
\end{pmatrix}.
\end{align}
$$

Combining the two matrices gives the final result:

$$
\begin{align}
\begin{pmatrix}
[D_x] \\
[D_y]
\end{pmatrix}=
\begin{pmatrix}
[[\varepsilon]]-\Delta[[|T_y|^2]] & \Delta[[T_x^*T_y]] \\
\Delta[[T_xT_y^*]] & [[\varepsilon]]-\Delta[[|T_x|^2]]
\end{pmatrix}
\begin{pmatrix}
[S_x] \\
[S_y]
\end{pmatrix},
\end{align}
$$

where $\Delta = [[\varepsilon]] - [[\varepsilon^{-1}]]^{-1}$. Comparing Eq. (13) with the abstract tensor notation used in [RCWA Core](rcwa_core.md), the correction matrices $[[A_{ij}]]$ are identified explicitly as:

$$
\begin{align}
[[A_{xx}]] = [[|T_y|^2]], \quad [[A_{yy}]] = [[|T_x|^2]], \quad
[[A_{xy}]] = [[T_x^*T_y]], \quad [[A_{yx}]] = [[T_xT_y^*]].
\end{align}
$$

Each $[[A_{ij}]]$ is a Toeplitz matrix whose entries are the Fourier coefficients of the corresponding spatially-varying TVF product (e.g. $|T_y(x,y)|^2$), computed from the FFT of that product in exactly the same way as $[[\varepsilon]]$. Because $\mathbf{T}$ is unit-normalized ($|T_x|^2+|T_y|^2=1$), the diagonal blocks satisfy the constraint:

$$
\begin{align}
[[A_{xx}]] + [[A_{yy}]] = I,
\end{align}
$$

which expresses the fact that the normal and tangential directions span the full transverse plane. In the limit $\mathbf{T} \to \hat{x}$ (i.e., boundaries aligned with the $y$-axis), one has $[[A_{xx}]] \to 0$, $[[A_{yy}]] \to I$, and the correction recovers the standard 1D Li factorization.

The factorization in Eqs. (10)–(11) relies on three approximations: (1) $[[\varepsilon T_i]] \approx [[\varepsilon]][[T_i]]$ and $[[\varepsilon^{-1} T_i]] \approx [[\varepsilon^{-1}]][[T_i]]$; (2) $[[T_i T_j]] \approx [[T_i]][[T_j]]$; (3) $[[|T_x|^2]] + [[|T_y|^2]] = I$. The derivation also uses the unit normalization $T_i \to T_i/\sqrt{|T_x|^2+|T_y|^2}$.

These approximations are not exact for truncated Fourier series in general. However, if $\mathbf{T}(x,y)$ is continuous on the material interface and smooth elsewhere, the resulting error is small and the factorization converges at the expected exponential rate. This approach was first introduced in [JOSA A, 18, 2886 (2001)](https://opg.optica.org/josaa/fulltext.cfm?uri=josaa-18-11-2886) and is known as the **fast-Fourier factorization (FFF)**. In this formulation, the factorization challenge shifts from selecting an appropriate Fourier representation of the permittivity to constructing a smooth, periodic tangential vector field $\mathbf{T}(x,y)$ that aligns with the material boundaries. The construction of this field and the resulting correction tensor $[[A_{ij}]]$ used in [RCWA Core](rcwa_core.md) are described in [TVF](tvf.md).
