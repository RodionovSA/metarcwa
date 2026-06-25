# Block matrices

## Motivation

RCWA computations involve many matrix operators that, despite being formally
$N_h \times N_h$, carry very little information: the vacuum mode matrix $W_0$
is the identity, and for homogeneous layers the $P$ and $Q$ operators are
diagonal in the Fourier basis.  Storing and manipulating all of these as full
dense matrices wastes memory by factors of $N_h$ to $N_h^2$, and makes the
code harder to follow because every special structure must be tracked manually.

The `Block` and `Block2x2` classes solve both problems at once.  A `Block` is
a single $n \times n$ operator stored in the cheapest representation that is
*exact*; a `Block2x2` is a $2 \times 2$ arrangement of four such operators
that carries the full RCWA mode matrix

$$
\begin{align}
\Phi = \begin{pmatrix} W & W \\ V & -V \end{pmatrix}.
\end{align}
$$

Arithmetic between blocks automatically dispatches to the cheapest valid
implementation, so the calling code reads like textbook algebra without
thinking about representation details.

## Block representations

A `Block` stores a batch of $n \times n$ operators in one of three
representations:

| Kind | Meaning | Data shape | Memory (per batch item, float32) |
|------|---------|------------|----------------------------------|
| `SCALAR` | $c \cdot I$ | `[*B]` | $1$ element |
| `DIAG` | $\mathrm{diag}(\mathbf{v})$ | `[*B, n]` | $n$ elements |
| `DENSE` | arbitrary | `[*B, n, n]` | $n^2$ elements |

The batch dimensions `[*B]` broadcast freely across operations; only the
harmonic axis $n$ must match between operands.  A `SCALAR` block carries no
$n$ and is compatible with any size.

Representations are promoted *upward* (SCALAR → DIAG → DENSE) only when
unavoidable:

$$
\begin{align}
\text{SCALAR} + \text{DIAG}   &\;\to\; \text{DIAG}, \\[4pt]
\text{DIAG}   + \text{DENSE}  &\;\to\; \text{DENSE}, \\[4pt]
\text{DIAG}   \times \text{DIAG}  &\;\to\; \text{DIAG}, \\[4pt]
\text{DIAG}   \times \text{DENSE} &\;\to\; \text{DENSE}.
\end{align}
$$

There is no demotion.  The representation is a property of the data, not a
user choice: passing a diagonal tensor to `Block(Block.DIAG, data)` is the
caller's responsibility.

### Memory savings

For a realistic RCWA run with batch size $B = 500$ and $n = 256$ harmonics:

| Representation | Storage | Savings vs DENSE |
|----------------|---------|-----------------|
| DENSE          | ≈ 125 MB | — |
| DIAG           | ≈ 500 KB | 256× |
| SCALAR         | ≈ 2 KB   | 65 536× |

In practice most operators at the start of the RCWA pipeline are `SCALAR` or
`DIAG`, so the solver never allocates large dense matrices until they are
genuinely needed (e.g. after an eigensolver step on a patterned layer).

## Arithmetic dispatch

The `@` operator dispatches to the cheapest exact computation:

```python
W0 = Block(Block.SCALAR, torch.ones(batch))   # identity
V  = Block(Block.DIAG,   diag_data)           # diagonal H-mode matrix

result = W0 @ V   # SCALAR @ DIAG → DIAG (scales V's data by 1)
```

Addition promotes both operands to the higher kind before adding:

```python
a = Block(Block.SCALAR, c)
b = Block(Block.DIAG,   v)
a + b   # → Block(DIAG, c[..., None] + v)
```

Inversion and solving are elementwise for `SCALAR`/`DIAG`, and use
`torch.linalg.inv` / `torch.linalg.solve` for `DENSE`:

```python
V.inv()       # DIAG → 1.0 / V.data
A.solve(B)    # DENSE → torch.linalg.solve(A.data, B.data)
```

Prefer `.solve(rhs)` over `.inv() @ rhs` for `DENSE` blocks: it avoids
forming the explicit inverse and is more numerically stable.

## Block2x2

A `Block2x2` is a $2 \times 2$ arrangement of blocks

$$
\begin{align}
M = \begin{pmatrix} a & b \\ c & d \end{pmatrix},
\end{align}
$$

where each entry is a `Block` (or a nested `Block2x2`).  Standard linear
algebra rules apply component-wise:

$$
\begin{align}
(M_1 + M_2)_{ij} &= (M_1)_{ij} + (M_2)_{ij}, \\[4pt]
(M_1 \cdot M_2)_{ij} &= \sum_k (M_1)_{ik} \cdot (M_2)_{kj}.
\end{align}
$$

### Inverse

The inverse uses the Schur complement of the lower-right block $d$:

$$
\begin{align}
S   &= a - b \, d^{-1} c, \\[6pt]
M^{-1} &= \begin{pmatrix}
  S^{-1}             & -S^{-1} b\, d^{-1} \\[4pt]
  -d^{-1} c\, S^{-1} & d^{-1} + d^{-1} c\, S^{-1} b\, d^{-1}
\end{pmatrix}.
\end{align}
$$

This is cheap when $d$ is `SCALAR` or `DIAG`, because $d^{-1}$ is then
elementwise rather than a matrix inversion.  For the RCWA mode matrices,
$d = -V$ for a homogeneous layer — a `DIAG` block — so the Schur complement
step costs only elementwise operations.

### Redheffer star product

S-matrices compose via the Redheffer star product $\star$ rather than ordinary
multiplication (see [S-matrix algebra](smatrix.md)).  `Block2x2` implements
this directly:

```python
S_total = S_left.star(S_right)
```

The star product inverts only the Fabry–Pérot denominator
$(I - S_{11}^R S_{22}^L)^{-1}$, never an S-matrix itself.  When the diagonal
reflection blocks are `DIAG` or `SCALAR` (as they are for homogeneous layers),
this inversion remains cheap.

### Dense conversion

`.to_dense(n)` flattens a `Block2x2` to a plain `(..., 2n, 2n)` tensor,
promoting all leaf blocks to `DENSE`.  Nested `Block2x2` entries are
flattened recursively.  Batch dimensions broadcast automatically, so a
`SCALAR`-derived `[n, n]` slice expands to match any batched sibling:

```python
M_dense = M.to_dense(Nh)   # → torch.Tensor of shape [..., 2*Nh, 2*Nh]
```

This conversion is used in `S_boundary` to form a single $4N_h \times 4N_h$
linear system that avoids singular sub-block issues that can arise in the
nested Schur complement when `DENSE` eigenvector matrices are present.

## Neutral elements

Two class-level identities are provided:

```python
Block2x2.identity()       # matmul identity [[I, 0], [0, I]]
Block2x2.star_identity()  # star-product identity [[0, I], [I, 0]]
```

Instance methods `.eye_like()` and `.zeros_like()` return neutrals that match
the device and dtype of the calling object without allocating new tensors.
