"""Vectors and matrices."""
from __future__ import annotations

import numpy as np

from ..engine.values import EvalError, RangeValue, Table, is_number, normalize_number
from ._util import mat, to_float, to_int, vec


def _arr(x):
    if isinstance(x, RangeValue):
        return x.to_array()
    if is_number(x):
        return np.array([x])
    if not isinstance(x, np.ndarray):
        raise EvalError("a vector or matrix is expected")
    return x


def _square(m, name):
    m = _arr(m)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise EvalError(f"{name} requires a square matrix")
    return m


def register(r):
    cat = "Vectors & matrices"

    @r.function(category=cat, signature="transpose(M)", doc="transpose of a matrix (Mᵀ)")
    def transpose(m):
        m = _arr(m)
        return m.reshape(1, -1) if m.ndim == 1 else m.T

    @r.function(category=cat, signature="inverse(M)", doc="matrix inverse M⁻¹", aliases=("inv",))
    def inverse(m):
        m = _square(m, "inverse")
        try:
            return np.linalg.inv(m.astype(float) if m.dtype != complex else m)
        except np.linalg.LinAlgError:
            raise EvalError("matrix is singular") from None

    @r.function(category=cat, signature="det(M)", doc="determinant")
    def det(m):
        return normalize_number(np.linalg.det(_square(m, "det").astype(float)))

    @r.function(category=cat, signature="rank(M)", doc="matrix rank")
    def rank(m):
        return int(np.linalg.matrix_rank(_arr(m)))

    @r.function(category=cat, signature="tr(M)", doc="trace", aliases=("trace",))
    def tr(m):
        return normalize_number(np.trace(_square(m, "tr")))

    @r.function(category=cat, signature="identity(n)", doc="n×n identity matrix", aliases=("eye",))
    def identity(n):
        return np.eye(to_int(n, "n"), dtype=np.int64)

    @r.function(category=cat, signature="zeros(m, n=0)", doc="vector (n = 0) or m×n matrix of zeros")
    def zeros(m, n=0):
        m, n = to_int(m), to_int(n)
        return np.zeros(m, dtype=np.int64) if n == 0 else np.zeros((m, n), dtype=np.int64)

    @r.function(category=cat, signature="ones(m, n=0)", doc="vector or matrix of ones")
    def ones(m, n=0):
        m, n = to_int(m), to_int(n)
        return np.ones(m, dtype=np.int64) if n == 0 else np.ones((m, n), dtype=np.int64)

    @r.function(category=cat, signature="diag(v)", doc="diagonal matrix from a vector, or diagonal of a matrix")
    def diag(v):
        return np.diag(_arr(v))

    @r.function(category=cat, signature="rows(M)", doc="number of rows")
    def rows(m):
        m = _arr(m)
        return int(m.shape[0])

    @r.function(category=cat, signature="cols(M)", doc="number of columns")
    def cols(m):
        m = _arr(m)
        return 1 if m.ndim == 1 else int(m.shape[1])

    @r.function(category=cat, signature="length(v)", doc="number of elements", aliases=("len",))
    def length(v):
        if isinstance(v, str):
            return len(v)
        return int(_arr(v).size)

    @r.function(category=cat, signature="last(v)", doc="index of the last element (respects ORIGIN)", lazy=True)
    def last(ctx, args, kwargs):
        v = _arr(ctx.eval(args[0]))
        return int(v.size - 1 + ctx.origin)

    @r.function(category=cat, signature="col(M, j)", doc="column j of a matrix", lazy=True)
    def col(ctx, args, kwargs):
        m = _arr(ctx.eval(args[0]))
        j = to_int(ctx.eval(args[1]), "column") - ctx.origin
        if m.ndim == 1:
            if j != 0:
                raise EvalError("a vector has only one column")
            return m
        return m[:, j]

    @r.function(category=cat, signature="row(M, i)", doc="row i of a matrix", lazy=True)
    def row(ctx, args, kwargs):
        m = _arr(ctx.eval(args[0]))
        i = to_int(ctx.eval(args[1]), "row") - ctx.origin
        return m[i] if m.ndim == 2 else np.array([m[i]])

    @r.function(category=cat, signature="submatrix(M, ir, jr, ic, jc)", doc="rows ir..jr and columns ic..jc", lazy=True)
    def submatrix(ctx, args, kwargs):
        m = _arr(ctx.eval(args[0]))
        o = ctx.origin
        ir, jr, ic, jc = (to_int(ctx.eval(a)) - o for a in args[1:5])
        if m.ndim == 1:
            m = m.reshape(-1, 1)
        return m[ir:jr + 1, ic:jc + 1]

    @r.function(category=cat, signature="augment(A, B, …)", doc="join matrices side by side")
    def augment(*items):
        return np.hstack([_arr(i).reshape(-1, 1) if _arr(i).ndim == 1 else _arr(i) for i in items])

    @r.function(category=cat, signature="stack(A, B, …)", doc="join matrices vertically")
    def stack(*items):
        arrays = [_arr(i) for i in items]
        if all(a.ndim == 1 for a in arrays):
            return np.concatenate(arrays)
        return np.vstack([a.reshape(1, -1) if a.ndim == 1 else a for a in arrays])

    @r.function(category=cat, signature="reshape(v, m, n)", doc="reshape to m×n (row order)")
    def reshape(v, m, n):
        return _arr(v).reshape(to_int(m), to_int(n))

    @r.function(category=cat, signature="flatten(M)", doc="all elements as a vector (row order)")
    def flatten(m):
        return _arr(m).reshape(-1)

    @r.function(category=cat, signature="reverse(v)", doc="reverse the order of elements")
    def reverse(v):
        return _arr(v)[::-1].copy()

    @r.function(category=cat, signature="sort(v)", doc="sort ascending")
    def sort(v):
        return np.sort(_arr(v).reshape(-1))

    @r.function(category=cat, signature="csort(M, j)", doc="sort rows of M by column j", lazy=True)
    def csort(ctx, args, kwargs):
        m = _arr(ctx.eval(args[0]))
        j = to_int(ctx.eval(args[1])) - ctx.origin
        return m[np.argsort(m[:, j], kind="stable")]

    @r.function(category=cat, signature="rsort(M, i)", doc="sort columns of M by row i", lazy=True)
    def rsort(ctx, args, kwargs):
        m = _arr(ctx.eval(args[0]))
        i = to_int(ctx.eval(args[1])) - ctx.origin
        return m[:, np.argsort(m[i, :], kind="stable")]

    @r.function(category=cat, signature="cross(a, b)", doc="cross product")
    def cross(a, b):
        return np.cross(vec(a), vec(b))

    @r.function(category=cat, signature="dot(a, b)", doc="dot product")
    def dot(a, b):
        return normalize_number(np.dot(vec(a), vec(b)))

    @r.function(category=cat, signature="norm(v)", doc="Euclidean norm", aliases=("norme",))
    def norm(v):
        return normalize_number(np.linalg.norm(_arr(v)))

    @r.function(category=cat, signature="norm1(M)", doc="L1 norm")
    def norm1(v):
        return normalize_number(np.linalg.norm(_arr(v), 1))

    @r.function(category=cat, signature="normi(M)", doc="infinity norm")
    def normi(v):
        return normalize_number(np.linalg.norm(_arr(v), np.inf))

    @r.function(category=cat, signature="lsolve(A, b)", doc="solve the linear system A·x = b")
    def lsolve(a, b):
        a = mat(a, "A")
        b = _arr(b).astype(float)
        try:
            if a.shape[0] == a.shape[1]:
                return np.linalg.solve(a, b)
            return np.linalg.lstsq(a, b, rcond=None)[0]
        except np.linalg.LinAlgError:
            raise EvalError("the system matrix is singular") from None

    @r.function(category=cat, signature="eigenvals(M)", doc="eigenvalues")
    def eigenvals(m):
        vals = np.linalg.eigvals(_square(m, "eigenvals"))
        return vals.real if np.allclose(vals.imag, 0) else vals

    @r.function(category=cat, signature="eigenvecs(M)", doc="eigenvectors (as columns)")
    def eigenvecs(m):
        vals, vecs = np.linalg.eig(_square(m, "eigenvecs"))
        return vecs.real if np.allclose(vecs.imag, 0) else vecs

    @r.function(category=cat, signature="svds(M)", doc="singular values")
    def svds(m):
        return np.linalg.svd(_arr(m), compute_uv=False)

    @r.function(category=cat, signature="cholesky(M)", doc="Cholesky factor L (M = L·Lᵀ)")
    def cholesky(m):
        return np.linalg.cholesky(_square(m, "cholesky").astype(float))

    @r.function(category=cat, signature="linspace(a, b, n)", doc="n evenly spaced numbers from a to b")
    def linspace(a, b, n):
        return np.linspace(to_float(a), to_float(b), to_int(n))

    @r.function(category=cat, signature="seq(a, b, step=1)", doc="vector a, a+step, …, b")
    def seq(a, b, step=1):
        return RangeValue(a, b, step).to_array()

    @r.function(category=cat, signature="matrix(m, n, f)", doc="m×n matrix with elements f(i, j)", lazy=True)
    def matrix(ctx, args, kwargs):
        m, n = to_int(ctx.eval(args[0])), to_int(ctx.eval(args[1]))
        f = ctx.eval(args[2])
        o = ctx.origin
        out = np.empty((m, n), dtype=object)
        for i in range(m):
            for j in range(n):
                out[i, j] = ctx.evaluator.call_value(f, [i + o, j + o])
        try:
            return out.astype(float)
        except (TypeError, ValueError):
            return out

    @r.function(category=cat, signature="vector(n, f)", doc="vector with elements f(i)", lazy=True, aliases=("vec",))
    def vector(ctx, args, kwargs):
        n = to_int(ctx.eval(args[0]))
        f = ctx.eval(args[1])
        o = ctx.origin
        return np.array([ctx.evaluator.call_value(f, [i + o]) for i in range(n)])

    @r.function(category=cat, signature="cumsum(v)", doc="cumulative sums")
    def cumsum(v):
        return np.cumsum(_arr(v))

    @r.function(category=cat, signature="argmax(v)", doc="index of the largest element (respects ORIGIN)", lazy=True)
    def argmax(ctx, args, kwargs):
        v = _arr(ctx.eval(args[0]))
        idx = np.unravel_index(np.argmax(v), v.shape)
        return int(idx[0]) + ctx.origin if v.ndim == 1 else np.array(idx) + ctx.origin

    @r.function(category=cat, signature="argmin(v)", doc="index of the smallest element (respects ORIGIN)", lazy=True)
    def argmin(ctx, args, kwargs):
        v = _arr(ctx.eval(args[0]))
        idx = np.unravel_index(np.argmin(v), v.shape)
        return int(idx[0]) + ctx.origin if v.ndim == 1 else np.array(idx) + ctx.origin

    @r.function(category=cat, signature="kron(A, B)", doc="Kronecker product")
    def kron(a, b):
        return np.kron(_arr(a), _arr(b))

    @r.function(category=cat, signature="table(M, headers)", doc="show a matrix as a table with column headers")
    def table(m, headers=None, title=""):
        m = _arr(m)
        if m.ndim == 1:
            m = m.reshape(-1, 1)
        hdr = [str(normalize_number(h)) for h in (np.asarray(headers).reshape(-1) if headers is not None else range(m.shape[1]))]
        return Table(hdr, [[normalize_number(v) for v in row] for row in m], title)
