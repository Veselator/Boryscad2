"""Facility location: sum of distances (Weber point) and centre of gravity."""
from __future__ import annotations

import numpy as np
from scipy import optimize

from ..engine.values import DText, EvalError, Record, SeriesGroup, Table
from ._util import clean, to_float, vec


def _xy(X, Y):
    X, Y = vec(X, "X"), vec(Y, "Y")
    if X.size != Y.size:
        raise EvalError("X and Y must have the same length")
    return X, Y


def weber_point(X, Y, W=None, tol=1e-12, max_iter=10000):
    W = np.ones(X.size) if W is None else W
    res = optimize.minimize(lambda p: float(np.sum(W * np.hypot(p[0] - X, p[1] - Y))),
                            np.array([np.average(X, weights=W), np.average(Y, weights=W)]),
                            method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": max_iter})
    p = res.x
    for _ in range(max_iter):  # Weiszfeld refinement
        d = np.hypot(p[0] - X, p[1] - Y)
        if np.any(d < 1e-12):
            break
        w = W / d
        new = np.array([np.sum(w * X) / np.sum(w), np.sum(w * Y) / np.sum(w)])
        if np.hypot(*(new - p)) < tol:
            p = new
            break
        p = new
    return p


def register(r):
    cat = "Operations research · Location"

    @r.function(category=cat, signature="sumdist(x, y, X, Y)", doc="R(x, y) = Σ √((x − Xᵢ)² + (y − Yᵢ)²)")
    def sumdist(x, y, X, Y):
        X, Y = _xy(X, Y)
        return float(np.sum(np.hypot(to_float(x) - X, to_float(y) - Y)))

    @r.function(category=cat, signature="distances(x, y, X, Y)", doc="vector of distances from (x, y) to every point")
    def distances(x, y, X, Y):
        X, Y = _xy(X, Y)
        return np.hypot(to_float(x) - X, to_float(y) - Y)

    @r.function(category=cat, signature="weber(X, Y, W=1)", doc="point minimizing the (weighted) sum of distances")
    def weber(X, Y, W=None):
        X, Y = _xy(X, Y)
        w = None if W is None else vec(W, "W")
        p = weber_point(X, Y, w)
        return np.array([clean(p[0], 1e-9), clean(p[1], 1e-9)])

    @r.function(category=cat, signature="centroid(X, Y, Q)", doc="conditional centre of mass Σ Xᵢ·Qᵢ / Σ Qᵢ")
    def centroid(X, Y, Q=None):
        X, Y = _xy(X, Y)
        Q = np.ones(X.size) if Q is None else vec(Q, "Q")
        if Q.size != X.size:
            raise EvalError("Q must have the same length as X")
        return np.array([float(np.sum(X * Q) / np.sum(Q)), float(np.sum(Y * Q) / np.sum(Q))])

    @r.function(category=cat, signature="warehouse(X, Y, Q)",
                doc="full report: initial location (min Σ distances), refined centre of mass, distances and plot")
    def warehouse(X, Y, Q):
        from .plotting import ScatterSeries
        X, Y = _xy(X, Y)
        Q = vec(Q, "Q")
        p0 = weber_point(X, Y)
        pc = np.array([np.sum(X * Q) / np.sum(Q), np.sum(Y * Q) / np.sum(Q)])
        d0 = np.hypot(p0[0] - X, p0[1] - Y)
        dc = np.hypot(pc[0] - X, pc[1] - Y)
        rows = [[i + 1, X[i], Y[i], Q[i], round(float(d0[i]), 4), round(float(dc[i]), 4)] for i in range(X.size)]
        rows.append(["Σ", "", "", float(np.sum(Q)), round(float(d0.sum()), 4), round(float(dc.sum()), 4)])
        plot = SeriesGroup([
            ScatterSeries(label="consumers", color="#1d3557", size=50, x=X, y=Y,
                          annotations=[f"{i + 1} ({Q[i]:g})" for i in range(X.size)]),
            ScatterSeries(label=f"initial ({p0[0]:.3f}; {p0[1]:.3f})", color="#e76f51", size=110, marker="s",
                          x=np.array([p0[0]]), y=np.array([p0[1]])),
            ScatterSeries(label=f"centre of mass ({pc[0]:.3f}; {pc[1]:.3f})", color="#2a9d8f", size=130, marker="*",
                          x=np.array([pc[0]]), y=np.array([pc[1]])),
        ])
        details = [
            DText(f"Initial location (min R): x = {p0[0]:.4f}, y = {p0[1]:.4f}, R = {d0.sum():.4f}", "note"),
            DText(f"Refined location (centre of mass): x = {pc[0]:.4f}, y = {pc[1]:.4f}", "note"),
            Table(["№", "X", "Y", "Q", "dist. to initial", "dist. to refined"], rows, "Distances").display(),
        ]
        return Record("Optimal warehouse location", {
            "initial": np.array([clean(p0[0], 1e-9), clean(p0[1], 1e-9)]), "refined": pc, "R0": float(d0.sum()),
            "d0": d0, "d": dc, "plot": plot}, details, ["initial", "refined"])
