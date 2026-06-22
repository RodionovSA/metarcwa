# RCWA core

In this document we provide a derivation of the main equations of 3D RCWA together with the principal assumptions of the method.

## Maxwell equations

We derive the RCWA equations from Maxwell's equations in a linear, isotropic, and non-magnetic medium. While RCWA can be formulated for anisotropic and magnetic media, we restrict ourselves to the most practical case, which allows a significant simplification of the derivation. We assume that the medium is characterized by an inhomogeneous complex permittivity $\varepsilon(\mathbf{r}) = \varepsilon_\mathrm{real}(\mathbf{r}) + j\varepsilon_\mathrm{imag}(\mathbf{r})$ and consider Maxwell's equations in the frequency domain, where the time dependence takes the form $\exp(-j\omega t)$. Under this convention, Maxwell's equations read:

$$
\begin{align}
\nabla \cdot (\varepsilon\mathbf{E}) &= 0, \\
\nabla \cdot \mathbf{H} &= 0, \\
\nabla \times \mathbf{E} &= j\omega \mu_0\mathbf{H}, \\
\nabla \times \mathbf{H} &= -j\omega \epsilon_0 \varepsilon\mathbf{E}.
\end{align}
$$

We now introduce the two main assumptions of RCWA. (1) The medium is stratified and can be decomposed into layers along the $z$-axis. Within each layer the medium is homogeneous along $z$, so $\varepsilon(\mathbf{r}) = \varepsilon(x, y)$. (2) The medium is periodic in the $xy$-plane, so $\varepsilon(x,y)$ can be expressed as a discrete Fourier series:

$$
\begin{align}
\varepsilon(x,y,\omega)=\sum_{m,n}\varepsilon_{m,n}(\omega)\exp{\!\left(j G_{x,mn}x+jG_{y,mn}y\right)},
\end{align}
$$
where $G_{x,mn}$ and $G_{y,mn}$ are the $x$- and $y$-components of the reciprocal lattice vector. Note that we consider the medium to be dispersive, so $\varepsilon$ is a function of frequency $\omega$; however, we suppress this dependence in the subsequent formulas for compactness. Following Bloch's theorem, the field components $\mathbf{E}$ and $\mathbf{H}$ admit a Fourier–Bloch expansion of the form:

$$
\begin{align}
E_i=\exp{\!\left(jk_{x,0}x+jk_{y,0}y\right)}\sum_{m,n}S_{i,mn}(z)\exp{\!\left(jG_{x,mn}x+jG_{y,mn}y\right)}, \\
H_i=j\sqrt{\frac{\epsilon_0}{\mu_0}}\exp{\!\left(jk_{x,0}x+jk_{y,0}y\right)}\sum_{m,n}U_{i,mn}(z)\exp{\!\left(jG_{x,mn}x+jG_{y,mn}y\right)}.
\end{align}
$$

Here, $i$ indicates the $x$, $y$, or $z$ component; $S_{i,mn}(z)$ and $U_{i,mn}(z)$ are the Fourier amplitudes of the $\mathbf{E}$ and $\mathbf{H}$ fields; and $k_{x,0}$, $k_{y,0}$ are the transverse components of the incident wavevector. Substituting Eqs. (5)–(7) into (3) and (4) and projecting onto each harmonic, we obtain:

$$
\begin{align}
\frac{\partial S_{y,mn}}{\partial z} -jk_{y,mn}S_{z,mn} &= k_0 U_{x,mn}, \\
jk_{x,mn}S_{z,mn} - \frac{\partial S_{x,mn}}{\partial z} &= k_0 U_{y,mn}, \\
jk_{y,mn}S_{x,mn} - jk_{x,mn}S_{y,mn} &= k_0 U_{z,mn}, \\
\end{align}
$$

$$
\begin{align}
\frac{\partial U_{y,mn}}{\partial z} - jk_{y,mn}U_{z,mn} &= k_0 D_{x, mn}, \\
jk_{x,mn}U_{z,mn} - \frac{\partial U_{x,mn}}{\partial z} &= k_0 D_{y, mn}, \\
jk_{y,mn}U_{x,mn} - jk_{x,mn}U_{y,mn} &= k_0 D_{z, mn}. \\
\end{align}
$$

where $(k_{x,mn}, k_{y,mn})=(k_{x,0}+G_{x,mn},\;k_{y,0}+G_{y,mn})$ and $D_{i,mn} = (\varepsilon E_i)_{mn}$ denotes the $mn$-th Fourier coefficient of the product $\varepsilon E_i$.

At this stage we deliberately refrain from expanding $D_{i,mn}$ as a convolution of the Fourier coefficients of $\varepsilon$ and $E_i$. In the theoretical limit of infinitely many harmonics such an expansion is exact by the convolution theorem. In practical computations, however, the truncated Fourier series of products of discontinuous functions can converge extremely slowly due to the Gibbs phenomenon at material interfaces. For this reason, we defer the construction of the matrix representation of $D_{i,mn}$ until after the factorization rules have been introduced (see the following section).

## Factorization rules

Factorization rules play a critical role in RCWA because they govern the convergence of the method. Ignoring them leads to extremely slow convergence and can render the approach impractical. A detailed derivation is given in the companion document [Factorization rules](factorization.md); here we state the essential results due to [Li (1996)](https://opg.optica.org/josaa/fulltext.cfm?uri=josaa-13-9-1870).

Because we work with truncated Fourier series, the behavior of the fields at material interfaces determines the overall convergence. Under our assumptions, $\mathbf{B}$ and $\mathbf{H}$ are continuous everywhere since the medium is non-magnetic. Our primary concern is therefore the behavior of $\mathbf{D}$ and $\mathbf{E}$. From Maxwell's equations it is well known that the normal component of $\mathbf{D}$ and the tangential component of $\mathbf{E}$ are continuous across an interface, while the complementary components are discontinuous with a jump proportional to the permittivity contrast. Representing such discontinuous products as truncated Fourier convolutions leads to Gibbs-type oscillations and poor convergence.

Li established the following rules for computing the Fourier coefficients of the product $\varepsilon(x,y)\,E_i(x,y)$:

1. If either $\varepsilon(x,y)$ or $E_i(x,y)$ is continuous at $(x_0,y_0)$ (and the other may be discontinuous), the **Laurent rule** applies:

$$
\begin{align}
D_{i,mn} = \sum_{m',n'}\varepsilon_{m-m',\,n-n'}\,S_{i,m'n'}.
\end{align}
$$

2. If both $\varepsilon(x,y)$ and $E_i(x,y)$ are discontinuous at $(x_0,y_0)$ but their product $\varepsilon(x,y)E_i(x,y)$ is continuous, the **inverse rule** applies:

$$
\begin{align}
\sum_{m',n'}(\varepsilon^{-1})_{m-m',\,n-n'}\,D_{i,m'n'} = S_{i,mn}.
\end{align}
$$

3. If all three quantities are simultaneously discontinuous, neither rule applies and the product cannot be accurately represented by a finite Fourier convolution.

The Laurent rule follows from the continuity of $\mathbf{E}$, while the inverse rule exploits the continuity of $\mathbf{D}$.

## RCWA equations

To cast the system into a form amenable to numerical computation, we introduce a compact matrix notation. We stack the 2D array of harmonic amplitudes into a single column vector ordered lexicographically over the index pair $(m,n)$ from $(-M,-N)$ to $(M,N)$:

$$
[F_i] = \begin{pmatrix}
F_{i,-M,-N}\\
F_{i,-M,-N+1}\\
\vdots\\
F_{i,M,N}
\end{pmatrix}.
$$

With this notation, Eqs. (8)–(13) take the form:

$$
\begin{align}
\frac{\partial [S_y]}{\partial z} -jK_y[S_z] &= k_0I[U_x], \\
jK_{x}[S_z] -\frac{\partial [S_x]}{\partial z} &= k_0I[U_y], \\
jK_{y}[S_x] -jK_x[S_y] &= k_0I[U_z], \\
\frac{\partial [U_y]}{\partial z} -jK_y[U_z] &= k_0[D_x], \\
jK_{x}[U_z] -\frac{\partial [U_x]}{\partial z}&= k_0[D_y], \\
jK_{y}[U_x] -jK_x[U_y] &= k_0[D_z], \\

\end{align}
$$

where 

$$
\begin{align}
K_i=
\begin{pmatrix}
[k_i]_{-M} & 0 & \cdots & 0\\
0 & [k_i]_{-M+1} & \cdots & 0\\
\vdots\\
0 & 0 & \cdots & [k_i]_{M}\\
\end{pmatrix},

\end{align}
$$

$$
\begin{align}
[k_i]_{j}=
\begin{pmatrix}
[k_i]_{j,-N} & 0 & \cdots & 0\\
0 & [k_i]_{j,-N+1} & \cdots & 0\\
\vdots\\
0 & 0 & \cdots & [k_i]_{j,N}\\
\end{pmatrix}.
\end{align}
$$

Applying the factorization rules, the Laurent and inverse rules take the following Toeplitz matrix form:

$$
\begin{align}
[D_{i}] = [[\varepsilon]] [S_{i}],
\end{align}
$$

$$
\begin{align}
[[\varepsilon^{-1}]][D_{i}] = [S_{i}],
\end{align}
$$

where 

$$
\begin{align}
[[\varepsilon]] = 
\begin{pmatrix}
\overline\varepsilon_0 & \overline\varepsilon_{-1} & \cdots & \overline\varepsilon_{-2M}\\
\overline\varepsilon_1 & \overline\varepsilon_{0} & \cdots & \overline\varepsilon_{-2M+1}\\
\vdots\\
\overline\varepsilon_{2M} & \overline\varepsilon_{2M-1} & \cdots & \overline\varepsilon_{0}
\end{pmatrix},

\\

\end{align}
$$

$$
\begin{align}
\overline\varepsilon_k=
\begin{pmatrix}
\varepsilon_{k,0} & \varepsilon_{k,-1} & \cdots & \varepsilon_{k,-2N}\\
\varepsilon_{k,1} & \varepsilon_{k,0} & \cdots & \varepsilon_{k,-2N+1}\\
\vdots\\
\varepsilon_{k,2N} & \varepsilon_{k,2N-1} & \cdots & \varepsilon_{k,0}
\end{pmatrix}.

\end{align}
$$

The choice between the Laurent and inverse rules depends on the orientation of each field component relative to the material interface. The full derivation, including the construction of the correction tensor via the Tangential Vector Field formalism, is given in [Factorization rules](factorization.md). The general result is:
$$
\begin{align}
\begin{pmatrix}
[D_x]\\ [D_y]\\ [D_z]
\end{pmatrix} = 

\begin{pmatrix}
[[\varepsilon]] - \Delta [[A_{xx}]] & \Delta [[A_{xy}]] & 0\\
\Delta [[A_{yx}]] & [[\varepsilon]] - \Delta [[A_{yy}]] & 0\\
0 & 0 & [[\varepsilon]]
\end{pmatrix} 

\begin{pmatrix}
[S_x]\\ [S_y]\\ [S_z]
\end{pmatrix},
\end{align}
$$

where $\Delta = [[\varepsilon]] - [[\varepsilon^{-1}]]^{-1}$ and $[[A_{ij}]]$ is the geometry-dependent correction tensor constructed from the Tangential Vector Field components (see [Factorization rules](factorization.md) and [TVF](tvf.md)). The structure of this relation can be understood as follows: the dependence on tangential versus normal field components makes the $\mathbf{D}$–$\mathbf{E}$ relationship effectively anisotropic in Fourier space. In the limit of a homogeneous medium or infinitely many harmonics, the Laurent and inverse rules become equivalent and the tensor relation reduces to a scalar multiplication by $[[\varepsilon]]$. The term $\Delta$ quantifies the discrepancy between the direct and inverse convolution operators, while $[[A_{ij}]]$ encodes the geometrical correction. The $z$-component is not subject to anisotropic correction because $\varepsilon$ is uniform along $z$ within each layer, and the Laurent rule can be applied directly.

Substituting Eq. (28) into Eqs. (16)–(21), we obtain:

$$
\begin{align}
\frac{1}{k_0}\frac{\partial [S_x]}{\partial z} &=-K_{x}[[\varepsilon]]^{-1}K_{y}[U_x] -(I-K_x[[\varepsilon]]^{-1}K_x)[U_y], \\

\frac{1}{k_0}\frac{\partial [S_y]}{\partial z}&= (I - K_y[[\varepsilon]]^{-1}K_y)[U_x]+K_y[[\varepsilon]]^{-1}K_x[U_y], \\

\frac{1}{k_0}\frac{\partial [U_x]}{\partial z}&=-(K_xK_y + \Delta [[A_{yx}]])[S_x]+(K_x^2 - [[\varepsilon]] + \Delta [[A_{yy}]])[S_y], \\

\frac{1}{k_0}\frac{\partial [U_y]}{\partial z} &= ([[\varepsilon]] - K_y^2 - \Delta [[A_{xx}]])[S_x] + (K_yK_x + \Delta [[A_{xy}]])[S_y], \\

\end{align}
$$

and for the $z$-components:

$$
\begin{align}
[S_z]=j[[\varepsilon]]^{-1}(K_{y}[U_x] -K_{x}[U_y]),\\ 

[U_z]=jI(K_{y}[S_x] -K_{x}[S_y]).
\end{align}
$$

Here $K_x$ and $K_y$ are normalized by $k_0$. From the $z$-component equations above it is evident that $[S_z]$ and $[U_z]$ are fully determined by the transverse amplitudes $[S_x]$, $[S_y]$, $[U_x]$, $[U_y]$; therefore only the four transverse evolution equations are independent. We introduce the block vectors of transverse field amplitudes:
$$
\begin{align}
s = 
\begin{pmatrix}
[S_x] \\ [S_y]
\end{pmatrix},\\ 

u = 
\begin{pmatrix}
[U_x] \\ [U_y]
\end{pmatrix}. 
\end{align}
$$

Equations (29)–(32) then take the compact first-order form:
$$
\begin{align}
\frac{1}{k_0}\frac{\partial s}{\partial z} = Pu,\\ 
\frac{1}{k_0}\frac{\partial u}{\partial z} = Qs, 
\end{align}
$$

where 
$$
\begin{align}
P = 
\begin{pmatrix}
-K_{x}[[\varepsilon]]^{-1}K_{y}& -I+K_x[[\varepsilon]]^{-1}K_x\\
I - K_y[[\varepsilon]]^{-1}K_y&K_y[[\varepsilon]]^{-1}K_x
\end{pmatrix},\\ 
Q = 
\begin{pmatrix}
-K_xK_y - \Delta [[A_{yx}]]&K_x^2 - [[\varepsilon]] + \Delta [[A_{yy}]]\\
[[\varepsilon]] - K_y^2 - \Delta [[A_{xx}]]&K_yK_x + \Delta [[A_{xy}]]
\end{pmatrix}. 
\end{align}
$$

Equations (37) and (38) are the fundamental coupled first-order equations of RCWA. They govern the $z$-evolution of the transverse field amplitudes within each layer.

Substituting the explicit TVF identities $[[A_{xx}]]=[[|T_y|^2]]$, $[[A_{yy}]]=[[|T_x|^2]]$, $[[A_{xy}]]=[[T_x^*T_y]]$, $[[A_{yx}]]=[[T_xT_y^*]]$ (derived in [Factorization rules](factorization.md)) into $Q$ gives the fully explicit form:

$$
\begin{align}
P = 
\begin{pmatrix}
-K_{x}[[\varepsilon]]^{-1}K_{y}& -I+K_x[[\varepsilon]]^{-1}K_x\\
I - K_y[[\varepsilon]]^{-1}K_y&K_y[[\varepsilon]]^{-1}K_x
\end{pmatrix},\\
Q = 
\begin{pmatrix}
-K_xK_y - \Delta[[T_xT_y^*]]&K_x^2 - [[\varepsilon]] + \Delta[[|T_x|^2]]\\
[[\varepsilon]] - K_y^2 - \Delta[[|T_y|^2]]&K_yK_x + \Delta[[T_x^*T_y]]
\end{pmatrix}.
\end{align}
$$

The matrix $P$ is independent of the TVF correction because it originates solely from the curl-$\mathbf{E}$ equations, where $\varepsilon$ enters only through $[[\varepsilon]]^{-1}$ (the inverse Toeplitz operator) and not through the boundary-orientation tensor. The TVF correction appears exclusively in $Q$, which is built from the curl-$\mathbf{H}$ equations and carries the anisotropic $\mathbf{D}$–$\mathbf{E}$ factorization. The construction of $T_x(x,y)$ and $T_y(x,y)$ and the resulting Toeplitz matrices are described in [TVF](tvf.md).

There are two main approaches to solving Eqs. (37) and (38) — via the eigenvalue decomposition of $PQ$ (see [Eigenvalue problem](eigenproblem.md)) or via the matrix exponential of the full $2N_h \times 2N_h$ system (see [Matrix exponential](matrixexp.md)). Regardless of the approach, the combined transverse field vector used for S-matrix assembly (see [S-matrix algebra](smatrix.md)) is $\psi(z)$, defined as:

$$
\begin{align}
\psi(z) = 
\begin{pmatrix}
s(z)\\ u(z)
\end{pmatrix}=
\begin{pmatrix}
[S_x](z)\\ [S_y](z)\\ [U_x](z)\\ [U_y](z)
\end{pmatrix}.
\end{align}
$$

