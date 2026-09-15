"""Linear programming: numeric LP, exact simplex with tableaux and the graphical method."""
from __future__ import annotations

import itertools
import math

import numpy as np
from scipy import optimize

from ..engine.values import DText, EvalError, Record, SeriesGroup, Table
from ._tableau import Tableau, fmt_frac, frac, parse_signs
from ._util import clean, clean_array, mat, strings, to_float, to_int, vec


def _lp_input(c, A, b, signs):
    c = vec(c, "c")
    A = np.asarray(A, dtype=float)
    if A.ndim == 1:
        A = A.reshape(1, -1) if A.size == c.size else A.reshape(-1, 1)
    b = vec(b, "b")
    if A.shape != (b.size, c.size):
        raise EvalError(f"A must be {b.size}×{c.size} (constraints × variables), got {A.shape[0]}×{A.shape[1]}")
    return c, A, b, parse_signs(signs, b.size)


def _sense(sense) -> str:
    s = str(sense).lower()
    if s in ("max", "maximize", "1"):
        return "max"
    if s in ("min", "minimize", "-1", "0"):
        return "min"
    raise EvalError('sense must be "max" or "min"')


def _names(n, names=None, prefix="x"):
    if names is not None:
        lst = strings(names)
        if len(lst) != n:
            raise EvalError(f"expected {n} variable names")
        return lst
    return [f"{prefix}{j + 1}" for j in range(n)]


def solve_lp(c, A, b, signs, sense="max", free=False, integer=False):
    A_ub, b_ub, A_eq, b_eq = [], [], [], []
    for row, rhs, s in zip(A, b, signs):
        if s == "<=":
            A_ub.append(row); b_ub.append(rhs)
        elif s == ">=":
            A_ub.append(-row); b_ub.append(-rhs)
        else:
            A_eq.append(row); b_eq.append(rhs)
    res = optimize.linprog(-c if sense == "max" else c,
                           A_ub=np.array(A_ub) if A_ub else None, b_ub=np.array(b_ub) if b_ub else None,
                           A_eq=np.array(A_eq) if A_eq else None, b_eq=np.array(b_eq) if b_eq else None,
                           bounds=[(None, None) if free else (0, None)] * c.size, method="highs",
                           integrality=np.ones(c.size) if integer else None)
    if res.status == 2:
        raise EvalError("the problem is infeasible")
    if res.status == 3:
        raise EvalError("the objective is unbounded")
    if not res.success:
        raise EvalError(f"LP solver failed: {res.message}")
    return clean_array(res.x, 1e-9)


def register(r):
    cat = "Operations research · LP"

    @r.function(category=cat, signature='linprog(c, A, b, sense="max", signs="<=", free=0, integer=0, names=)',
                doc="solve a linear program; returns x, F, used resources and remainders")
    def linprog(c, A, b, sense="max", signs=None, free=0, integer=0, names=None):
        c, A, b, signs = _lp_input(c, A, b, signs)
        sense = _sense(sense)
        x = solve_lp(c, A, b, signs, sense, bool(free), bool(integer))
        used = A @ x
        remainder = clean_array(b - used, 1e-9)
        F = clean(float(c @ x), 1e-9)
        var_names = _names(c.size, names)
        rows = [[f"({i + 1})", " + ".join(f"{A[i, j]:g}·{var_names[j]}" for j in range(c.size)) + f" {signs[i]} {b[i]:g}",
                 clean(float(used[i]), 1e-9), float(b[i]), float(remainder[i])] for i in range(b.size)]
        details = [Table(["#", "constraint", "used", "limit", "remainder"], rows, "Constraints").display()]
        return Record("Linear programming", {"x": x, "F": F, "used": clean_array(used), "remainder": remainder},
                      details, ["x", "F"])

    @r.function(category=cat, signature='simplex(c, A, b, sense="max", signs="<=", names=, fractions=1)',
                doc="exact simplex method with all intermediate tableaux")
    def simplex(c, A, b, sense="max", signs=None, names=None, fractions=1):
        c, A, b, signs = _lp_input(c, A, b, signs)
        sense = _sense(sense)
        cost = c if sense == "max" else -c
        tab = Tableau.standard(cost, A, b, signs, _names(c.size, names))
        tab.as_fraction = bool(fractions)
        status = tab.solve()
        details = []
        for step in tab.steps:
            details.append(step.table.display())
        if status == "infeasible":
            raise EvalError("the problem is infeasible")
        if status == "unbounded":
            raise EvalError("the objective is unbounded")
        x = np.array([float(v) for v in tab.values(c.size)])
        F = float(tab.objective()) * (1 if sense == "max" else -1)
        x = clean_array(x, 1e-12)
        remainder = clean_array(b - A @ x, 1e-9) if all(s != "=" for s in signs) else clean_array(b - A @ x, 1e-9)
        return Record(f"Simplex method ({sense})", {"x": x, "F": clean(F, 1e-12), "remainder": remainder,
                                                     "iterations": len(tab.steps)},
                      details, ["x", "F", "remainder"])

    @r.function(category=cat, signature='lp_graphical(c, A, b, sense="max", signs="<=", nonneg=1)',
                doc="graphical method for 2 variables: vertices of the feasible polygon and the optimum")
    def lp_graphical(c, A, b, sense="max", signs=None, nonneg=1):
        c, A, b, signs = _lp_input(c, A, b, signs)
        if c.size != 2:
            raise EvalError("the graphical method needs exactly 2 variables")
        sense = _sense(sense)
        vertices = feasible_vertices(A, b, signs, bool(nonneg))
        if not vertices:
            raise EvalError("the feasible region is empty")
        values = [float(c @ v) for v in vertices]
        best = int(np.argmax(values) if sense == "max" else np.argmin(values))
        rows = [[f"P{i + 1}", clean(v[0]), clean(v[1]), clean(val, 1e-9)] for i, (v, val) in enumerate(zip(vertices, values))]
        table = Table(["vertex", "x1", "x2", "W"], rows, "Vertices of the feasible region",
                      highlight={(best, 3)})
        x = np.array([clean(vertices[best][0]), clean(vertices[best][1])])
        bounded = _check_bounded(c, A, b, signs, bool(nonneg), sense)
        details = [table.display()]
        if not bounded:
            details.append(DText("⚠ the objective is unbounded in the optimisation direction", "error"))
        grad = c / (np.linalg.norm(c) or 1)
        return Record("Graphical method", {"x": x, "F": clean(values[best], 1e-9), "vertices": np.array(vertices),
                                           "gradient": c, "direction": grad}, details, ["x", "F"])

    @r.function(category=cat, signature='lp_graph(c, A, b, sense="max", signs="<=", nonneg=1)',
                doc="plot series for the graphical method: constraints, feasible region, gradient and level line")
    def lp_graph(c, A, b, sense="max", signs=None, nonneg=1, names=None):
        from .plotting import ArrowSeries, LineRefSeries, LineSeries, PolygonSeries, ScatterSeries
        c, A, b, signs = _lp_input(c, A, b, signs)
        if c.size != 2:
            raise EvalError("lp_graph needs exactly 2 variables")
        sense = _sense(sense)
        vertices = feasible_vertices(A, b, signs, bool(nonneg))
        var_names = _names(2, names)
        if vertices:
            pts = np.array(vertices)
            span = max(float(np.max(np.abs(pts))), 1.0)
        else:
            span = max(float(np.max(np.abs(b))), 1.0)
        xmax = span * 1.25
        group = SeriesGroup()
        palette = ["#e4572e", "#29335c", "#f3a712", "#669bbc", "#a8c686", "#8e6c8a", "#2a9d8f", "#e76f51"]
        for i in range(b.size):
            a1, a2 = A[i]
            color = palette[i % len(palette)]
            label = f"({i + 1}) {a1:g}·{var_names[0]} + {a2:g}·{var_names[1]} = {b[i]:g}"
            if abs(a2) > 1e-12:
                xs = np.linspace(-0.05 * xmax, xmax, 2)
                ys = (b[i] - a1 * xs) / a2
                group.append(LineSeries(label=label, color=color, width=1.8, x=xs, y=ys))
            elif abs(a1) > 1e-12:
                group.append(LineRefSeries(label=label, color=color, width=1.8, value=b[i] / a1, orientation="v"))
        if vertices:
            pts = np.array(vertices)
            center = pts.mean(axis=0)
            order = np.argsort(np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0]))
            poly = pts[order]
            group.append(PolygonSeries(label="feasible region", color="#57cc99", alpha=0.3, x=poly[:, 0], y=poly[:, 1]))
            values = pts @ c
            best = int(np.argmax(values) if sense == "max" else np.argmin(values))
            opt = pts[best]
            # level line through the optimum (perpendicular to the gradient)
            if abs(c[1]) > 1e-12:
                xs = np.linspace(-0.05 * xmax, xmax, 2)
                ys = (c @ opt - c[0] * xs) / c[1]
                group.append(LineSeries(label=f"level line W = {clean(float(c @ opt), 1e-9):g}", color="#7b2cbf",
                                        style="--", width=1.6, x=xs, y=ys))
            else:
                group.append(LineRefSeries(label="level line", color="#7b2cbf", style="--", value=opt[0], orientation="v"))
            g = c / (np.linalg.norm(c) or 1) * span * 0.25
            group.append(ArrowSeries(label="gradient ∇W", color="#d62828", width=2.2, x0=0.0, y0=0.0, dx=float(g[0]),
                                     dy=float(g[1])))
            group.append(ScatterSeries(label=f"optimum ({clean(opt[0]):g}; {clean(opt[1]):g})", color="#d62828",
                                       size=70, marker="o", x=np.array([opt[0]]), y=np.array([opt[1]])))
            group.append(ScatterSeries(label="", color="#333333", size=18, marker="o", x=pts[:, 0], y=pts[:, 1],
                                       annotations=[f"P{i + 1}" for i in range(len(pts))]))
        return group


def feasible_vertices(A: np.ndarray, b: np.ndarray, signs: list[str], nonneg: bool = True) -> list[np.ndarray]:
    lines = [(A[i], b[i]) for i in range(b.size)]
    if nonneg:
        lines += [(np.array([1.0, 0.0]), 0.0), (np.array([0.0, 1.0]), 0.0)]
    pts: list[np.ndarray] = []
    for (a1, b1), (a2, b2) in itertools.combinations(lines, 2):
        M = np.array([a1, a2], dtype=float)
        if abs(np.linalg.det(M)) < 1e-12:
            continue
        p = np.linalg.solve(M, np.array([b1, b2], dtype=float))
        if _feasible(p, A, b, signs, nonneg) and not any(np.allclose(p, q, atol=1e-9) for q in pts):
            pts.append(p)
    if pts:
        center = np.mean(pts, axis=0)
        pts.sort(key=lambda p: math.atan2(p[1] - center[1], p[0] - center[0]))
    return pts


def _feasible(p, A, b, signs, nonneg, tol=1e-9):
    if nonneg and (p[0] < -tol or p[1] < -tol):
        return False
    for row, rhs, s in zip(A, b, signs):
        v = row @ p
        scale = tol * max(1.0, abs(rhs))
        if s == "<=" and v > rhs + scale or s == ">=" and v < rhs - scale or s == "=" and abs(v - rhs) > scale:
            return False
    return True


def _check_bounded(c, A, b, signs, nonneg, sense) -> bool:
    try:
        solve_lp(c, A, b, signs, sense, free=not nonneg)
        return True
    except EvalError:
        return False
