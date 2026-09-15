"""Dynamic programming: allocation of limited resources (Bellman)."""
from __future__ import annotations

import numpy as np

from ..engine.values import DText, EvalError, Record, Table
from ._util import clean, mat, to_float, to_int


def _profits(F, with_zero: bool) -> np.ndarray:
    F = mat(F, "response functions")
    if not with_zero:
        F = np.hstack([np.zeros((F.shape[0], 1)), F])
    return F


def _fmt(v):
    v = float(v)
    return int(v) if v.is_integer() else round(v, 6)


def register(r):
    cat = "Operations research · Dynamic"

    @r.function(category=cat, signature="allocate(F, unit=1, with_zero=0)",
                doc="optimal allocation of K units between enterprises (rows of F = profit for 1..K units) — Bellman")
    def allocate(F, unit=1, with_zero=0):
        P = _profits(F, bool(with_zero))
        unit = to_float(unit, "unit")
        n, K1 = P.shape
        K = K1 - 1
        phi = np.zeros((n, K + 1))
        choice = np.zeros((n, K + 1), dtype=int)
        details = [DText("Sequential method (Bellman recurrence): φ₁(X) = f₁(X), "
                         "φₖ(X) = max₀≤ₓ≤X { fₖ(x) + φₖ₋₁(X − x) }", "note")]
        phi[0] = P[0]
        choice[0] = np.arange(K + 1)
        rows = [[_fmt(X * unit), _fmt(P[0, X]), _fmt(X * unit)] for X in range(K + 1)]
        details.append(Table(["X", "φ1(X)", "x1*"], rows, "Stage 1").display())
        for k in range(1, n):
            headers = ["X"] + [f"x{k + 1}={_fmt(x * unit)}" for x in range(K + 1)] + [f"φ{k + 1}(X)", f"x{k + 1}*"]
            rows, hl = [], set()
            for X in range(K + 1):
                values = [P[k, x] + phi[k - 1, X - x] if x <= X else None for x in range(K + 1)]
                best = max(range(X + 1), key=lambda x: (values[x], -x))
                phi[k, X] = values[best]
                choice[k, X] = best
                rows.append([_fmt(X * unit)] + ["" if v is None else _fmt(v) for v in values] +
                            [_fmt(phi[k, X]), _fmt(best * unit)])
                hl.add((X, 1 + best))
            details.append(Table(headers, rows, f"Stage {k + 1}", hl).display())
        alloc = np.zeros(n)
        X = K
        for k in range(n - 1, -1, -1):
            alloc[k] = choice[k, X]
            X -= choice[k, X]
        total = phi[n - 1, K]
        details.append(DText("Optimal allocation: " + ", ".join(f"enterprise {k + 1}: {_fmt(alloc[k] * unit)}"
                                                                for k in range(n)) + f";  max F = {_fmt(total)}", "note"))
        return Record("Resource allocation (dynamic programming)", {
            "x": (alloc * unit).astype(float) if unit != int(unit) else (alloc * unit).astype(np.int64),
            "units": alloc.astype(np.int64), "F": _fmt(total), "phi": phi}, details, ["x", "F"])

    @r.function(category=cat, signature="allocate_graph(F, unit=1, with_zero=0)",
                doc="the same allocation solved as a longest path in the network of states (transition cost method)")
    def allocate_graph(F, unit=1, with_zero=0):
        P = _profits(F, bool(with_zero))
        unit = to_float(unit, "unit")
        n, K1 = P.shape
        K = K1 - 1
        # label[k][s]: best profit after k enterprises with s units spent
        label = np.full((n + 1, K + 1), -np.inf)
        prev = np.full((n + 1, K + 1), -1, dtype=int)
        label[0, 0] = 0.0
        rows = []
        for k in range(1, n + 1):
            for s in range(K + 1):
                if label[k - 1, s] == -np.inf:
                    continue
                for x in range(0, K - s + 1):
                    t = s + x
                    value = label[k - 1, s] + P[k - 1, x]
                    rows.append([k, f"({k - 1}, {_fmt(s * unit)})", f"({k}, {_fmt(t * unit)})", _fmt(x * unit),
                                 _fmt(P[k - 1, x]), _fmt(value)])
                    if value > label[k, t] + 1e-12:
                        label[k, t] = value
                        prev[k, t] = s
        if label[n, K] == -np.inf:
            raise EvalError("no allocation found")
        alloc = np.zeros(n)
        t = K
        path = [(n, K)]
        for k in range(n, 0, -1):
            s = prev[k, t]
            alloc[k - 1] = t - s
            t = s
            path.append((k - 1, t))
        path.reverse()
        label_rows = [[f"stage {k}"] + ["" if label[k, s] == -np.inf else _fmt(label[k, s]) for s in range(K + 1)]
                      for k in range(n + 1)]
        on_path = {(k, s + 1) for k, s in path}
        details = [
            DText("States (k, spent): arc weight = profit of enterprise k for the allocated amount; the optimal "
                  "allocation is the maximum-weight path from (0, 0) to (n, S).", "note"),
            Table(["state"] + [_fmt(s * unit) for s in range(K + 1)], label_rows, "Best cumulative profit of each state",
                  on_path).display(),
            DText("Optimal path: " + " → ".join(f"({k}, {_fmt(s * unit)})" for k, s in path), "note"),
        ]
        if len(rows) <= 200:
            details.append(Table(["stage", "from", "to", "allocated", "profit", "cumulative"], rows,
                                 "All transitions").display())
        return Record("Resource allocation (transition cost method)", {
            "x": (alloc * unit).astype(np.int64) if float(unit).is_integer() else alloc * unit,
            "F": _fmt(label[n, K]), "labels": label[1:]}, details, ["x", "F"])
