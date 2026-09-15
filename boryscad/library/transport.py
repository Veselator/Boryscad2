"""Transportation problem: initial plans by nine methods and the potentials (MODI) method."""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import optimize

from ..engine.values import DText, EvalError, Record, Table
from ._util import as_int_if_whole, mat, strings, vec

METHODS: dict[str, str] = {
    "nw": "North-west corner",
    "ne": "North-east corner",
    "sw": "South-west corner",
    "se": "South-east corner",
    "mincol": "Minimum cost by column",
    "minrow": "Minimum cost by row",
    "mintable": "Minimum cost in table",
    "double": "Double preference",
    "vogel": "Vogel approximation",
}


def _balance(C: np.ndarray, a: np.ndarray, b: np.ndarray):
    sa, sb = a.sum(), b.sum()
    info = "balanced (closed) plan: ΣQs = ΣQc" if np.isclose(sa, sb) else (
        f"unbalanced: supply exceeds demand by {sa - sb:g} — a fictitious consumer is added" if sa > sb else
        f"unbalanced: demand exceeds supply by {sb - sa:g} — a fictitious supplier is added")
    if sa > sb:
        C = np.hstack([C, np.zeros((C.shape[0], 1))])
        b = np.append(b, sa - sb)
    elif sb > sa:
        C = np.vstack([C, np.zeros((1, C.shape[1]))])
        a = np.append(a, sb - sa)
    return C, a.astype(float), b.astype(float), info


def northwest(C, a, b):
    a, b = a.copy(), b.copy()
    X = np.zeros(C.shape)
    i = j = 0
    m, n = C.shape
    while i < m and j < n:
        q = min(a[i], b[j])
        X[i, j] = q
        a[i] -= q
        b[j] -= q
        if a[i] <= 1e-12 and b[j] <= 1e-12:
            i += 1
            j += 1
        elif a[i] <= 1e-12:
            i += 1
        else:
            j += 1
    return X


def _flipped(method: Callable, flip_rows: bool, flip_cols: bool):
    def run(C, a, b):
        Cf, af, bf = C, a, b
        if flip_rows:
            Cf, af = Cf[::-1, :], af[::-1]
        if flip_cols:
            Cf, bf = Cf[:, ::-1], bf[::-1]
        X = method(Cf, af.copy(), bf.copy())
        if flip_rows:
            X = X[::-1, :]
        if flip_cols:
            X = X[:, ::-1]
        return X
    return run


def min_by_column(C, a, b):
    a, b = a.copy(), b.copy()
    X = np.zeros(C.shape)
    for j in range(C.shape[1]):
        while b[j] > 1e-12:
            rows = [i for i in range(C.shape[0]) if a[i] > 1e-12]
            if not rows:
                break
            i = min(rows, key=lambda r: (C[r, j], r))
            q = min(a[i], b[j])
            X[i, j] += q
            a[i] -= q
            b[j] -= q
    return X


def min_by_row(C, a, b):
    return min_by_column(C.T, b, a).T


def min_table(C, a, b):
    a, b = a.copy(), b.copy()
    X = np.zeros(C.shape)
    order = sorted(((C[i, j], i, j) for i in range(C.shape[0]) for j in range(C.shape[1])))
    for cost, i, j in order:
        if a[i] > 1e-12 and b[j] > 1e-12:
            q = min(a[i], b[j])
            X[i, j] = q
            a[i] -= q
            b[j] -= q
    return X


def double_preference(C, a, b):
    a, b = a.copy(), b.copy()
    X = np.zeros(C.shape)
    m, n = C.shape
    marks = np.zeros(C.shape, dtype=int)
    for i in range(m):
        marks[i, C[i, :] == C[i, :].min()] += 1
    for j in range(n):
        marks[C[:, j] == C[:, j].min(), j] += 1
    order = sorted(((-marks[i, j], C[i, j], i, j) for i in range(m) for j in range(n)))
    for _, cost, i, j in order:
        if a[i] > 1e-12 and b[j] > 1e-12:
            q = min(a[i], b[j])
            X[i, j] = q
            a[i] -= q
            b[j] -= q
    return X


def vogel(C, a, b):
    a, b = a.copy(), b.copy()
    X = np.zeros(C.shape)
    m, n = C.shape
    rows, cols = set(range(m)), set(range(n))
    while rows and cols:
        best = None
        for i in rows:
            costs = sorted(C[i, j] for j in cols)
            pen = costs[1] - costs[0] if len(costs) > 1 else costs[0]
            j = min(cols, key=lambda c: (C[i, c], c))
            key = (pen, -C[i, j])
            if best is None or key > best[0]:
                best = (key, i, j)
        for j in cols:
            costs = sorted(C[i, j] for i in rows)
            pen = costs[1] - costs[0] if len(costs) > 1 else costs[0]
            i = min(rows, key=lambda r: (C[r, j], r))
            key = (pen, -C[i, j])
            if key > best[0]:
                best = (key, i, j)
        _, i, j = best
        q = min(a[i], b[j])
        X[i, j] = q
        a[i] -= q
        b[j] -= q
        if a[i] <= 1e-12:
            rows.discard(i)
        if b[j] <= 1e-12:
            cols.discard(j)
    return X


METHOD_IMPL: dict[str, Callable] = {
    "nw": northwest,
    "ne": _flipped(northwest, False, True),
    "sw": _flipped(northwest, True, False),
    "se": _flipped(northwest, True, True),
    "mincol": min_by_column,
    "minrow": min_by_row,
    "mintable": min_table,
    "double": double_preference,
    "vogel": vogel,
}


def potentials(C: np.ndarray, X: np.ndarray, max_iter: int = 500):
    """MODI method starting from plan X. Returns (optimal X, iterations)."""
    m, n = C.shape
    X = X.astype(float).copy()
    basis = {(i, j) for i in range(m) for j in range(n) if X[i, j] > 1e-12}
    _complete_basis(C, basis, m, n)
    iterations = 0
    for iterations in range(1, max_iter + 1):
        u = [None] * m
        v = [None] * n
        u[0] = 0.0
        changed = True
        while changed:
            changed = False
            for i, j in basis:
                if u[i] is not None and v[j] is None:
                    v[j] = C[i, j] - u[i]; changed = True
                elif v[j] is not None and u[i] is None:
                    u[i] = C[i, j] - v[j]; changed = True
        if any(x is None for x in u) or any(x is None for x in v):
            raise EvalError("potentials: degenerate basis could not be completed")
        delta = np.array([[C[i, j] - u[i] - v[j] for j in range(n)] for i in range(m)])
        i0, j0 = np.unravel_index(np.argmin(delta), delta.shape)
        if delta[i0, j0] >= -1e-9:
            return X, iterations - 1
        cycle = _find_cycle(basis, (i0, j0))
        if cycle is None:
            raise EvalError("potentials: no cycle found")
        minus = cycle[1::2]
        theta = min(X[c] for c in minus)
        leaving = min((c for c in minus if abs(X[c] - theta) < 1e-12), key=lambda c: C[c])
        for k, cell in enumerate(cycle):
            X[cell] += theta if k % 2 == 0 else -theta
        basis.add((i0, j0))
        basis.discard(leaving)
    raise EvalError("potentials: iteration limit reached")


def _complete_basis(C, basis, m, n):
    parent = list(range(m + n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in basis:
        parent[find(i)] = find(m + j)
    for cost, i, j in sorted((C[i, j], i, j) for i in range(m) for j in range(n)):
        if len(basis) >= m + n - 1:
            break
        if (i, j) not in basis and find(i) != find(m + j):
            basis.add((i, j))
            parent[find(i)] = find(m + j)


def _find_cycle(basis, start):
    cells = set(basis) | {start}

    def search(path, horizontal):
        i, j = path[-1]
        if len(path) > 3 and (i == start[0] if horizontal else j == start[1]):
            pass
        candidates = [c for c in cells if (c[0] == i if horizontal else c[1] == j) and c != (i, j)]
        for c in candidates:
            if c == start and len(path) >= 4 and len(path) % 2 == 0:
                return path
            if c in path:
                continue
            res = search(path + [c], not horizontal)
            if res:
                return res
        return None

    return search([start], True) or search([start], False)


def plan_cost(C, X) -> float:
    return float(np.sum(C * X))


def _labels(prefix, count, extra_label):
    return [f"{prefix}{k + 1}" for k in range(count)] + ([extra_label] if extra_label else [])


def _plan_table(C, X, a, b, title, orig_shape) -> Table:
    m, n = X.shape
    col_names = _labels("C", orig_shape[1], "C*" if n > orig_shape[1] else "")
    row_names = _labels("S", orig_shape[0], "S*" if m > orig_shape[0] else "")
    rows = []
    for i in range(m):
        rows.append([row_names[i]] + [(_num(X[i, j]) if X[i, j] > 1e-12 else "") for j in range(n)] + [_num(a[i])])
    rows.append(["Qc"] + [_num(v) for v in b] + [_num(a.sum())])
    return Table([""] + col_names + ["Qs"], rows, title, {(i, j + 1) for i in range(m) for j in range(n) if X[i, j] > 1e-12})


def _num(v):
    v = float(v)
    return int(v) if v.is_integer() else round(v, 6)


def _inputs(C, a, b):
    C = mat(C, "C")
    a = vec(a, "supply")
    b = vec(b, "demand")
    if C.shape != (a.size, b.size):
        raise EvalError(f"cost matrix must be {a.size}×{b.size} (suppliers × consumers), got {C.shape[0]}×{C.shape[1]}")
    return C, a, b


def register(r):
    cat = "Operations research · Transport"

    @r.function(category=cat, signature='transport(C, supply, demand, method="nw")',
                doc="initial plan: nw, ne, sw, se, mincol, minrow, mintable, double, vogel")
    def transport(C, supply, demand, method="nw"):
        C0, a0, b0 = _inputs(C, supply, demand)
        key = str(method).lower()
        if key not in METHOD_IMPL:
            raise EvalError(f"unknown method {method!r}; use one of {', '.join(METHODS)}")
        Cb, a, b, info = _balance(C0, a0, b0)
        X = METHOD_IMPL[key](Cb, a, b)
        Z = plan_cost(Cb, X)
        filled = int(np.sum(X > 1e-12))
        degenerate = filled < Cb.shape[0] + Cb.shape[1] - 1
        details = [DText(info, "note"), _plan_table(Cb, X, a, b, METHODS[key], C0.shape).display()]
        if degenerate:
            details.append(DText(f"degenerate plan: {filled} occupied cells < m + n − 1 = {Cb.shape[0] + Cb.shape[1] - 1}", "note"))
        return Record(METHODS[key], {"X": as_int_if_whole(X), "Z": _num(Z), "method": METHODS[key], "cells": filled},
                      details, ["Z", "cells"])

    @r.function(category=cat, signature="transport_all(C, supply, demand, plans=0)",
                doc="all nine methods sorted by total cost; the cheapest plan is the best potential plan")
    def transport_all(C, supply, demand, plans=0):
        C0, a0, b0 = _inputs(C, supply, demand)
        Cb, a, b, info = _balance(C0, a0, b0)
        results = []
        for key, name in METHODS.items():
            X = METHOD_IMPL[key](Cb, a, b)
            results.append((plan_cost(Cb, X), name, key, X))
        results.sort(key=lambda t: t[0])
        rows = [[k + 1, name, _num(z)] for k, (z, name, key, X) in enumerate(results)]
        optimal_X, iters = potentials(Cb, results[0][3])
        z_opt = plan_cost(Cb, optimal_X)
        details = [DText(f"ΣQs = {_num(a0.sum())}, ΣQc = {_num(b0.sum())}: {info}", "note"),
                   Table(["rank", "method", "Z"], rows, "Potential plans sorted by cost", {(0, 2)}).display()]
        if plans:
            for z, name, key, X in results:
                details.append(_plan_table(Cb, X, a, b, f"{name}: Z = {_num(z)}", C0.shape).display())
        details.append(DText(f"Potentials (MODI) method from '{results[0][1]}': {iters} improvement(s), "
                             f"optimal Z = {_num(z_opt)}", "note"))
        details.append(_plan_table(Cb, optimal_X, a, b, "Optimal plan", C0.shape).display())
        return Record("Transportation problem", {
            "best": results[0][1], "Zbest": _num(results[0][0]), "X": as_int_if_whole(results[0][3]),
            "Z": np.array([_num(t[0]) for t in results]), "methods": np.array([t[1] for t in results], dtype=object),
            "Xopt": as_int_if_whole(optimal_X), "Zopt": _num(z_opt)}, details, ["best", "Zbest", "Zopt"])

    @r.function(category=cat, signature='transport_optimal(C, supply, demand, start="vogel")',
                doc="optimal plan by the potentials (MODI) method")
    def transport_optimal(C, supply, demand, start="vogel"):
        C0, a0, b0 = _inputs(C, supply, demand)
        Cb, a, b, info = _balance(C0, a0, b0)
        X0 = METHOD_IMPL[str(start)](Cb, a, b)
        X, iters = potentials(Cb, X0)
        z = plan_cost(Cb, X)
        # independent check with an LP solver
        m, n = Cb.shape
        A_eq = np.zeros((m + n, m * n))
        for i in range(m):
            A_eq[i, i * n:(i + 1) * n] = 1
        for j in range(n):
            A_eq[m + j, j::n] = 1
        lp = optimize.linprog(Cb.reshape(-1), A_eq=A_eq, b_eq=np.concatenate([a, b]), bounds=(0, None), method="highs")
        note = f"start: {METHODS[str(start)]} (Z = {_num(plan_cost(Cb, X0))}), {iters} improvement(s)"
        if lp.success and lp.fun < z - 1e-6:
            X = lp.x.reshape(m, n)
            z = lp.fun
            note += "; refined by LP"
        return Record("Optimal transportation plan", {"X": as_int_if_whole(np.round(X, 9)), "Z": _num(z)},
                      [DText(info, "note"), DText(note, "note"), _plan_table(Cb, X, a, b, "Optimal plan", C0.shape).display()],
                      ["Z"])

    @r.function(category=cat, signature="transport_cost(C, X)", doc="total cost Σ c·Q of a plan")
    def transport_cost(C, X):
        return _num(np.sum(mat(C) * mat(X)))
