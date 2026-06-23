import pytest
import torch
from torch.testing import assert_close

from metarcwa.solver.blockmatrix import Block, Block2x2

N = 4

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


# ---- helpers -----------------------------------------------------------------

def scalar(val: float = 2.0, device="cpu") -> Block:
    return Block(Block.SCALAR, torch.tensor(val, device=device))

def diag(vals=None, device="cpu") -> Block:
    if vals is None:
        vals = [1.0, 2.0, 3.0, 4.0]
    return Block(Block.DIAG, torch.tensor(vals, dtype=torch.float, device=device))

def dense(mat=None, device="cpu") -> Block:
    if mat is None:
        # diagonal-dominant so it is invertible
        mat = torch.diag(torch.tensor([1.0, 2.0, 3.0, 4.0])) + 0.1 * torch.eye(N)
    return Block(Block.DENSE, mat.to(device))

def eye_dense(n=N, device="cpu") -> Block:
    return Block(Block.DENSE, torch.eye(n, device=device))


# ---- construction ------------------------------------------------------------

class TestConstruction:
    def test_diag_infers_n(self):
        b = Block(Block.DIAG, torch.ones(N))
        assert b.n == N
        assert b.kind == Block.DIAG

    def test_dense_infers_n(self):
        b = Block(Block.DENSE, torch.eye(N))
        assert b.n == N
        assert b.kind == Block.DENSE

    def test_scalar_n_is_none(self):
        b = Block(Block.SCALAR, torch.tensor(1.0))
        assert b.n is None
        assert b.kind == Block.SCALAR

    def test_diag_requires_1d(self):
        with pytest.raises(ValueError, match="1D"):
            Block(Block.DIAG, torch.tensor(1.0))

    def test_dense_requires_2d(self):
        with pytest.raises(ValueError, match="2D"):
            Block(Block.DENSE, torch.ones(N))

    def test_dense_requires_square(self):
        with pytest.raises(ValueError, match="square"):
            Block(Block.DENSE, torch.ones(3, N))

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="unknown kind"):
            Block(99, torch.ones(N))

    def test_eye_is_scalar(self):
        b = Block.eye()
        assert b.kind == Block.SCALAR
        assert_close(b.data, torch.ones(()))

    def test_zeros_is_scalar(self):
        b = Block.zeros()
        assert b.kind == Block.SCALAR
        assert_close(b.data, torch.zeros(()))

    def test_repr(self):
        r = repr(Block(Block.DIAG, torch.ones(N)))
        assert "DIAG" in r


# ---- promotion ---------------------------------------------------------------

class TestPromotion:
    def test_scalar_to_diag(self):
        b = scalar(2.0).to(Block.DIAG, N)
        assert b.kind == Block.DIAG
        assert b.n == N
        assert_close(b.data, torch.full((N,), 2.0))

    def test_scalar_to_dense(self):
        b = scalar(3.0).to(Block.DENSE, N)
        assert b.kind == Block.DENSE
        assert_close(b.data, 3.0 * torch.eye(N))

    def test_diag_to_dense(self):
        vals = torch.tensor([1.0, 2.0, 3.0, 4.0])
        b = Block(Block.DIAG, vals).to(Block.DENSE)
        assert b.kind == Block.DENSE
        assert_close(b.data, torch.diag(vals))

    def test_idempotent_same_kind(self):
        d = diag()
        assert d.to(Block.DIAG) is d

    def test_idempotent_lower_kind(self):
        d = diag()
        assert d.to(Block.SCALAR) is d

    def test_scalar_promotion_without_n_raises(self):
        with pytest.raises(ValueError, match="requires a target n"):
            scalar().to(Block.DIAG)


# ---- neg / sub ---------------------------------------------------------------

class TestNeg:
    def test_neg_preserves_kind(self):
        assert (-scalar()).kind == Block.SCALAR
        assert (-diag()).kind   == Block.DIAG
        assert (-dense()).kind  == Block.DENSE

    def test_neg_negates_scalar(self):
        assert_close((-scalar(3.0)).data, torch.tensor(-3.0))

    def test_neg_negates_diag(self):
        assert_close((-diag([1., 2.])).data, torch.tensor([-1., -2.]))

    def test_double_neg_roundtrip_scalar(self):
        s = scalar(5.0)
        assert_close((-(-s)).data, s.data)

    def test_double_neg_roundtrip_diag(self):
        d = diag()
        assert_close((-(-d)).data, d.data)

    def test_sub(self):
        b = scalar(5.0) - scalar(3.0)
        assert b.kind == Block.SCALAR
        assert_close(b.data, torch.tensor(2.0))


# ---- add ---------------------------------------------------------------------

class TestAdd:
    def test_scalar_scalar(self):
        b = scalar(2.0) + scalar(3.0)
        assert b.kind == Block.SCALAR
        assert_close(b.data, torch.tensor(5.0))

    def test_scalar_diag(self):
        b = scalar(1.0) + diag([1., 2., 3., 4.])
        assert b.kind == Block.DIAG
        assert_close(b.data, torch.tensor([2., 3., 4., 5.]))

    def test_diag_scalar(self):                     # commutativity
        b = diag([1., 2., 3., 4.]) + scalar(1.0)
        assert b.kind == Block.DIAG
        assert_close(b.data, torch.tensor([2., 3., 4., 5.]))

    def test_scalar_dense(self):
        b = scalar(1.0) + eye_dense()
        assert b.kind == Block.DENSE
        assert_close(b.data, 2.0 * torch.eye(N))

    def test_diag_diag(self):
        b = diag([1., 2., 3., 4.]) + diag([4., 3., 2., 1.])
        assert b.kind == Block.DIAG
        assert_close(b.data, torch.full((N,), 5.0))

    def test_diag_dense(self):
        b = diag([1., 1., 1., 1.]) + eye_dense()
        assert b.kind == Block.DENSE
        assert_close(b.data, 2.0 * torch.eye(N))

    def test_dense_dense(self):
        b = eye_dense() + eye_dense()
        assert b.kind == Block.DENSE
        assert_close(b.data, 2.0 * torch.eye(N))

    def test_n_mismatch_raises(self):
        with pytest.raises(ValueError, match="size mismatch"):
            Block(Block.DIAG, torch.ones(3)) + Block(Block.DIAG, torch.ones(4))


# ---- matmul ------------------------------------------------------------------

class TestMatmul:
    def test_scalar_scalar(self):
        b = scalar(2.0) @ scalar(3.0)
        assert b.kind == Block.SCALAR
        assert_close(b.data, torch.tensor(6.0))

    def test_scalar_diag(self):
        b = scalar(2.0) @ diag([1., 2., 3., 4.])
        assert b.kind == Block.DIAG
        assert_close(b.data, torch.tensor([2., 4., 6., 8.]))

    def test_diag_scalar(self):
        b = diag([1., 2., 3., 4.]) @ scalar(2.0)
        assert b.kind == Block.DIAG
        assert_close(b.data, torch.tensor([2., 4., 6., 8.]))

    def test_scalar_dense(self):
        b = scalar(2.0) @ eye_dense()
        assert b.kind == Block.DENSE
        assert_close(b.data, 2.0 * torch.eye(N))

    def test_diag_diag(self):
        b = diag([1., 2., 3., 4.]) @ diag([4., 3., 2., 1.])
        assert b.kind == Block.DIAG
        assert_close(b.data, torch.tensor([4., 6., 6., 4.]))

    def test_diag_dense_scales_rows(self):
        d_vals = torch.tensor([2., 1., 1., 1.])
        d = Block(Block.DIAG, d_vals)
        m = eye_dense()
        b = d @ m
        assert b.kind == Block.DENSE
        expected = torch.eye(N)
        expected[0] *= 2.0
        assert_close(b.data, expected)

    def test_dense_diag_scales_cols(self):
        d_vals = torch.tensor([1., 1., 1., 3.])
        d = Block(Block.DIAG, d_vals)
        m = eye_dense()
        b = m @ d
        assert b.kind == Block.DENSE
        expected = torch.eye(N)
        expected[:, -1] *= 3.0
        assert_close(b.data, expected)

    def test_dense_dense(self):
        b = eye_dense() @ eye_dense()
        assert b.kind == Block.DENSE
        assert_close(b.data, torch.eye(N))

    def test_n_mismatch_raises(self):
        with pytest.raises(ValueError, match="size mismatch"):
            Block(Block.DIAG, torch.ones(3)) @ Block(Block.DIAG, torch.ones(4))


# ---- inv ---------------------------------------------------------------------

class TestInv:
    def test_scalar_inv(self):
        b = scalar(2.0).inv()
        assert b.kind == Block.SCALAR
        assert_close(b.data, torch.tensor(0.5))

    def test_diag_inv(self):
        b = diag([2., 4., 1., 0.5]).inv()
        assert b.kind == Block.DIAG
        assert_close(b.data, torch.tensor([0.5, 0.25, 1.0, 2.0]))

    def test_dense_inv(self):
        m = dense()
        mi = m.inv()
        assert mi.kind == Block.DENSE
        identity = m @ mi
        assert_close(identity.data, torch.eye(N), atol=1e-5, rtol=1e-5)

    def test_inv_matmul_identity_scalar(self):
        s = scalar(5.0)
        assert_close((s.inv() @ s).data, torch.tensor(1.0))

    def test_inv_matmul_identity_diag(self):
        d = diag([1., 2., 3., 4.])
        result = d.inv() @ d
        assert result.kind == Block.DIAG
        assert_close(result.data, torch.ones(N))

    def test_inv_matmul_identity_dense(self):
        m = dense()
        result = m.inv() @ m
        assert_close(result.data, torch.eye(N), atol=1e-5, rtol=1e-5)


# ---- solve -------------------------------------------------------------------

class TestSolve:
    def test_solve_scalar_rhs_scalar(self):
        lhs = scalar(4.0)
        rhs = scalar(2.0)
        b = lhs.solve(rhs)
        assert_close((lhs @ b).data, rhs.data)

    def test_solve_diag_rhs_diag(self):
        lhs = diag([2., 4., 1., 2.])
        rhs = diag([4., 8., 3., 6.])
        b = lhs.solve(rhs)
        assert b.kind == Block.DIAG
        result = lhs @ b
        assert_close(result.data, rhs.data, atol=1e-5, rtol=1e-5)

    def test_solve_dense_rhs_dense(self):
        lhs = dense()
        rhs = eye_dense()
        b = lhs.solve(rhs)
        assert b.kind == Block.DENSE
        assert_close((lhs @ b).data, rhs.data, atol=1e-5, rtol=1e-5)

    def test_solve_dense_rhs_scalar(self):
        lhs = dense()
        rhs = Block.eye()               # SCALAR — promoted to DENSE inside solve
        b = lhs.solve(rhs)
        assert b.kind == Block.DENSE
        # lhs @ b should be identity
        assert_close((lhs @ b).data, torch.eye(N), atol=1e-5, rtol=1e-5)

    def test_solve_dense_rhs_diag(self):
        lhs = dense()
        rhs = diag([1., 2., 3., 4.])
        b = lhs.solve(rhs)
        assert b.kind == Block.DENSE
        result = lhs @ b
        assert_close(result.data, rhs.to(Block.DENSE).data, atol=1e-5, rtol=1e-5)

    def test_n_mismatch_raises(self):
        lhs = dense()
        rhs = Block(Block.DENSE, torch.eye(3))
        with pytest.raises(ValueError, match="size mismatch"):
            lhs.solve(rhs)


# ---- batched -----------------------------------------------------------------

class TestBatch:
    B = 5

    def test_batched_diag_add(self):
        d1 = Block(Block.DIAG, torch.ones(self.B, N))
        d2 = Block(Block.DIAG, torch.ones(self.B, N) * 2)
        b = d1 + d2
        assert b.kind == Block.DIAG
        assert b.data.shape == (self.B, N)
        assert_close(b.data, torch.full((self.B, N), 3.0))

    def test_batched_dense_matmul(self):
        mat = torch.eye(N).unsqueeze(0).expand(self.B, N, N).contiguous()
        m1 = Block(Block.DENSE, mat * 2)
        m2 = Block(Block.DENSE, mat * 3)
        b = m1 @ m2
        assert b.kind == Block.DENSE
        assert b.data.shape == (self.B, N, N)
        assert_close(b.data, mat * 6)

    def test_scalar_broadcast_into_batched_diag(self):
        s = scalar(2.0)
        d = Block(Block.DIAG, torch.ones(self.B, N))
        b = s @ d
        assert b.kind == Block.DIAG
        assert b.data.shape == (self.B, N)
        assert_close(b.data, torch.full((self.B, N), 2.0))

    def test_batched_inv(self):
        d = Block(Block.DIAG, torch.arange(1, N + 1, dtype=torch.float).unsqueeze(0).expand(self.B, N))
        di = d.inv()
        assert di.kind == Block.DIAG
        assert di.data.shape == (self.B, N)
        result = d @ di
        assert_close(result.data, torch.ones(self.B, N))


# ---- memory ------------------------------------------------------------------

class TestMemory:
    """Verify and display memory savings of structured Block representations."""

    @pytest.mark.parametrize("n", [16, 64, 256, 1024])
    def test_diag_smaller_than_dense(self, n):
        d = Block(Block.DIAG,  torch.ones(n))
        m = Block(Block.DENSE, torch.eye(n))
        assert d.data.nbytes * n == m.data.nbytes   # exactly n× smaller

    @pytest.mark.parametrize("n", [16, 64, 256, 1024])
    def test_scalar_smaller_than_diag(self, n):
        s = Block.eye()
        d = Block(Block.DIAG, torch.ones(n))
        assert s.data.nbytes < d.data.nbytes        # 1 element vs n

    def test_memory_table_unbatched(self, capsys):
        """Print per-element memory footprint across harmonic sizes."""
        elem = torch.ones(()).element_size()
        header = f"{'n':>6}  {'SCALAR':>8}  {'DIAG':>10}  {'DENSE':>12}  {'DIAG/DENSE':>12}  {'SCALAR/DENSE':>14}"
        sep    = "-" * len(header)
        rows   = []
        for n in [16, 64, 256, 1024]:
            sb = elem
            db = n * elem
            mb = n * n * elem
            rows.append(
                f"{n:>6}  {sb:>7}B  {db:>9}B  {mb:>11}B  {mb / db:>11.0f}x  {mb / sb:>13.0f}x"
            )
            assert sb < db < mb

        with capsys.disabled():
            print(f"\n\nMemory — unbatched (float32, {elem} B/elem):")
            print(header)
            print(sep)
            for row in rows:
                print(row)

    def test_memory_table_batched(self, capsys):
        """Print memory footprint with a realistic RCWA batch dimension."""
        B, n = 500, 256
        elem = torch.ones(()).element_size()

        sb = B * elem            # [B]      — SCALAR
        db = B * n * elem        # [B, n]   — DIAG
        mb = B * n * n * elem    # [B, n, n]— DENSE

        def fmt(nbytes):
            if nbytes >= 1 << 20:
                return f"{nbytes / (1 << 20):.1f} MB"
            return f"{nbytes / (1 << 10):.1f} KB"

        with capsys.disabled():
            print(f"\n\nMemory — batched B={B}, n={n} (float32):")
            print(f"  SCALAR  {fmt(sb):>10}   (1 scalar per batch item)")
            print(f"  DIAG    {fmt(db):>10}   ({n} elements per batch item)")
            print(f"  DENSE   {fmt(mb):>10}   ({n}×{n} elements per batch item)")
            print(f"  DIAG saves   {mb // db}× vs DENSE")
            print(f"  SCALAR saves {mb // sb}× vs DENSE")

        assert sb < db < mb
        assert mb // db == n
        assert mb // sb == n * n


# ---- Block2x2 helpers --------------------------------------------------------

def to_mat(m: Block2x2, n: int) -> torch.Tensor:
    """Flatten a Block2x2 to a (2n)×(2n) dense tensor for numerical checks."""
    a = m.a.to(Block.DENSE, n).data
    b = m.b.to(Block.DENSE, n).data
    c = m.c.to(Block.DENSE, n).data
    d = m.d.to(Block.DENSE, n).data
    return torch.cat([torch.cat([a, b], dim=-1),
                      torch.cat([c, d], dim=-1)], dim=-2)


def s2(val: float) -> Block:
    return Block(Block.SCALAR, torch.tensor(val, dtype=torch.float64))


def make_block2x2_scalar(a, b, c, d) -> Block2x2:
    return Block2x2(s2(a), s2(b), s2(c), s2(d))


# ---- Block2x2 tests ----------------------------------------------------------

class TestBlock2x2:

    # ---- construction & identity ----

    def test_construction_stores_entries(self):
        a, b, c, d = s2(1), s2(2), s2(3), s2(4)
        m = Block2x2(a, b, c, d)
        assert m.a is a
        assert m.b is b
        assert m.c is c
        assert m.d is d

    def test_identity_structure(self):
        eye = Block2x2.identity()
        assert eye.a.kind == Block.SCALAR
        assert eye.b.kind == Block.SCALAR
        assert eye.c.kind == Block.SCALAR
        assert eye.d.kind == Block.SCALAR
        assert_close(eye.a.data, torch.ones(()))
        assert_close(eye.b.data, torch.zeros(()))
        assert_close(eye.c.data, torch.zeros(()))
        assert_close(eye.d.data, torch.ones(()))

    def test_identity_matmul_left(self):
        M   = make_block2x2_scalar(2, 3, 4, 5)
        res = Block2x2.identity() @ M
        assert_close(res.a.data, M.a.data)
        assert_close(res.b.data, M.b.data)
        assert_close(res.c.data, M.c.data)
        assert_close(res.d.data, M.d.data)

    def test_identity_matmul_right(self):
        M   = make_block2x2_scalar(2, 3, 4, 5)
        res = M @ Block2x2.identity()
        assert_close(res.a.data, M.a.data)
        assert_close(res.b.data, M.b.data)
        assert_close(res.c.data, M.c.data)
        assert_close(res.d.data, M.d.data)

    # ---- add / sub ----

    def test_add_scalar_entries(self):
        M  = make_block2x2_scalar(1, 2, 3, 4)
        M2 = make_block2x2_scalar(4, 3, 2, 1)
        res = M + M2
        assert_close(res.a.data, torch.tensor(5.0, dtype=torch.float64))
        assert_close(res.b.data, torch.tensor(5.0, dtype=torch.float64))
        assert_close(res.c.data, torch.tensor(5.0, dtype=torch.float64))
        assert_close(res.d.data, torch.tensor(5.0, dtype=torch.float64))

    def test_sub_scalar_entries(self):
        M  = make_block2x2_scalar(3, 5, 7, 9)
        M2 = make_block2x2_scalar(1, 2, 3, 4)
        res = M - M2
        assert_close(res.a.data, torch.tensor(2.0, dtype=torch.float64))
        assert_close(res.b.data, torch.tensor(3.0, dtype=torch.float64))
        assert_close(res.c.data, torch.tensor(4.0, dtype=torch.float64))
        assert_close(res.d.data, torch.tensor(5.0, dtype=torch.float64))

    # ---- matmul ----

    def test_matmul_scalar_entries(self):
        # [[1,2],[3,4]] @ [[5,6],[7,8]] = [[1*5+2*7, 1*6+2*8],[3*5+4*7, 3*6+4*8]]
        #                               = [[19,22],[43,50]]
        M1  = make_block2x2_scalar(1, 2, 3, 4)
        M2  = make_block2x2_scalar(5, 6, 7, 8)
        res = M1 @ M2
        assert_close(res.a.data, torch.tensor(19.0, dtype=torch.float64))
        assert_close(res.b.data, torch.tensor(22.0, dtype=torch.float64))
        assert_close(res.c.data, torch.tensor(43.0, dtype=torch.float64))
        assert_close(res.d.data, torch.tensor(50.0, dtype=torch.float64))

    def test_matmul_diag_entries(self):
        # Block-diagonal: [[D,0],[0,D]] @ [[D,0],[0,D]] = [[D²,0],[0,D²]]
        n   = N
        dv  = torch.arange(1, n + 1, dtype=torch.float)
        D   = Block(Block.DIAG, dv)
        Z   = Block.zeros()
        M   = Block2x2(D, Z, Z, D)
        res = M @ M
        assert res.a.kind == Block.DIAG
        assert res.d.kind == Block.DIAG
        assert_close(res.a.data, dv ** 2)
        assert_close(res.d.data, dv ** 2)

    def test_matmul_via_dense(self):
        # Compare flattened result with torch.linalg.matmul on the 2n×2n matrix
        n   = N
        dv  = torch.arange(1, n + 1, dtype=torch.float)
        A   = Block(Block.DENSE, torch.diag(dv) + 0.1 * torch.eye(n))
        B_  = Block(Block.DENSE, 0.5 * torch.eye(n))
        C   = Block(Block.DENSE, 0.3 * torch.eye(n))
        D_  = Block(Block.DENSE, torch.diag(dv.flip(0)) + 0.1 * torch.eye(n))
        M1  = Block2x2(A, B_, C, D_)
        M2  = Block2x2(D_, C, B_, A)

        res_flat     = to_mat(M1 @ M2, n)
        expected     = to_mat(M1, n) @ to_mat(M2, n)
        assert_close(res_flat, expected, atol=1e-5, rtol=1e-5)

    # ---- inv ----

    def test_inv_identity(self):
        eye = Block2x2.identity()
        ei  = eye.inv()
        assert_close(ei.a.data, torch.ones(()))
        assert_close(ei.b.data, torch.zeros(()))
        assert_close(ei.c.data, torch.zeros(()))
        assert_close(ei.d.data, torch.ones(()))

    def test_inv_block_diagonal(self):
        # [[D, 0], [0, D']]^{-1} = [[D^{-1}, 0], [0, D'^{-1}]]
        n   = N
        dv  = torch.arange(1, n + 1, dtype=torch.float)
        dv2 = torch.arange(n + 1, 2 * n + 1, dtype=torch.float)
        D   = Block(Block.DIAG, dv)
        D2  = Block(Block.DIAG, dv2)
        Z   = Block.zeros()
        M   = Block2x2(D, Z, Z, D2)
        Mi  = M.inv()

        assert_close(Mi.a.data, 1.0 / dv)
        assert_close(Mi.d.data, 1.0 / dv2)

    def test_inv_scalar_known(self):
        # [[2,1],[1,2]]^{-1} = (1/3) * [[2,-1],[-1,2]]
        M  = make_block2x2_scalar(2.0, 1.0, 1.0, 2.0)
        Mi = M.inv()
        assert_close(Mi.a.data, torch.tensor( 2.0 / 3.0, dtype=torch.float64))
        assert_close(Mi.b.data, torch.tensor(-1.0 / 3.0, dtype=torch.float64))
        assert_close(Mi.c.data, torch.tensor(-1.0 / 3.0, dtype=torch.float64))
        assert_close(Mi.d.data, torch.tensor( 2.0 / 3.0, dtype=torch.float64))

    def test_inv_roundtrip_scalar_entries(self):
        M   = make_block2x2_scalar(3.0, 1.0, 1.0, 3.0)
        res = M @ M.inv()
        assert_close(res.a.data, torch.tensor(1.0, dtype=torch.float64), atol=1e-10, rtol=1e-10)
        assert_close(res.b.data, torch.tensor(0.0, dtype=torch.float64), atol=1e-10, rtol=1e-10)
        assert_close(res.c.data, torch.tensor(0.0, dtype=torch.float64), atol=1e-10, rtol=1e-10)
        assert_close(res.d.data, torch.tensor(1.0, dtype=torch.float64), atol=1e-10, rtol=1e-10)

    def test_inv_roundtrip_dense_entries(self):
        n   = N
        dv  = torch.arange(1, n + 1, dtype=torch.float)
        A   = Block(Block.DENSE, torch.diag(dv)          + 0.1 * torch.eye(n))
        B_  = Block(Block.DENSE, 0.3 * torch.eye(n))
        C   = Block(Block.DENSE, 0.2 * torch.eye(n))
        D_  = Block(Block.DENSE, torch.diag(dv.flip(0))  + 0.1 * torch.eye(n))
        M   = Block2x2(A, B_, C, D_)

        M_flat  = to_mat(M, n)
        Mi_flat = to_mat(M.inv(), n)
        assert_close(M_flat @ Mi_flat, torch.eye(2 * n), atol=1e-5, rtol=1e-5)
