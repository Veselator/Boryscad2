"""Automatic plot bounds.

Plot functions may omit their ranges::

    fplot(sin(x)/x, x)                     x-range chosen from roots/extrema/domain
    contour(f(x, y), x, y)                 area around the minimum / maximum
    implicit(x^2 + y^2 == 25, x, y)        bounding box of the curve
    region(x + y <= 5 and x >= 0 and y >= 0, x, y)

Such calls produce :class:`DeferredSeries`; after all series of a plot region are
evaluated, :func:`resolve_plot` picks common bounds (explicit plot limits → extents
of concrete data → features of the deferred series → default) and turns every
deferred series into a real one.  When the y-limit field is empty, curves with
poles (tan, 1/x, …) are clipped to a robust y-range.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Optional

import numpy as np

from ..engine.values import EvalError
from ..language import ast as A
from . import plotting as P

Bounds = Optional[tuple[float, float]]
WINDOWS = (10.0, 100.0, 1000.0)


@dataclass
class DeferredSeries(P.Series):
    series_kind: ClassVar[str] = "deferred"
    role: str = "curve"                                   # curve | set | field
    build: Optional[Callable[[tuple, tuple], Any]] = None  # (xlim, ylim) -> series
    features: Optional[Callable[[], Any]] = None           # curve: (lo, hi); set/field: (x0, x1, y0, y1)


def _num(v: float) -> A.Expr:
    return A.Number(repr(float(v)))


# ------------------------------------------------------------------ features
def _safe_curve(ctx, node, name, xs) -> np.ndarray:
    try:
        with np.errstate(all="ignore"):
            return P._eval_curve(ctx, node, name, xs)
    except Exception:
        return np.full(xs.shape, np.nan)


def _safe_grid(ctx, node, xname, yname, X, Y) -> np.ndarray:
    try:
        with np.errstate(all="ignore"):
            Z = ctx.eval_vectorized(node, {xname: X, yname: Y})
            Z = P._real_only(np.asarray(Z))
            if Z.shape == X.shape:
                return Z.astype(float)
            if Z.size == 1:
                return np.full(X.shape, float(Z.reshape(-1)[0]))
    except Exception:
        pass
    Z = np.full(X.shape, np.nan)
    for idx in np.ndindex(X.shape):
        try:
            v = ctx.eval(node, {xname: float(X[idx]), yname: float(Y[idx])})
            Z[idx] = float(np.real(v))
        except Exception:
            pass
    return Z


def curve_features(ctx, node, name) -> Bounds:
    """Interesting x-range of y = f(x): roots, extrema and domain edges."""
    domain: Bounds = None
    for W in WINDOWS:
        xs = np.linspace(-W, W, 4001)
        ys = _safe_curve(ctx, node, name, xs)
        finite = np.isfinite(ys)
        if finite.sum() < 3:
            continue
        window_domain = (float(xs[finite].min()), float(xs[finite].max()))
        if domain is None:
            domain = window_domain
        scale = float(np.median(np.abs(ys[finite]))) or 1.0
        feats: list[float] = []
        # domain edges (log, sqrt, …)
        edges = np.nonzero(np.diff(finite.astype(int)))[0]
        feats += [float(xs[i + (0 if finite[i] else 1)]) for i in edges]
        y = np.where(finite, ys, np.nan)
        big = 50 * scale
        for i in range(len(xs) - 1):
            a, b = y[i], y[i + 1]
            if not (np.isfinite(a) and np.isfinite(b)):
                continue
            if a == 0 or a * b < 0:
                if min(abs(a), abs(b)) < big:          # a real root, not a jump over a pole
                    feats.append(float(xs[i]))
        dy = np.diff(y)
        for i in range(len(dy) - 1):
            if np.isfinite(dy[i]) and np.isfinite(dy[i + 1]) and dy[i] * dy[i + 1] < 0 and abs(y[i + 1]) < big:
                feats.append(float(xs[i + 1]))
        feats = [f for f in feats if abs(f) < W * 0.999]
        if not feats:
            continue
        if len(feats) > 80:                          # periodic or noisy: show a few periods
            return (-W, W) if W > 10 else (-10.0, 10.0)
        lo, hi = min(feats), max(feats)
        span = hi - lo
        if span < 1e-9:
            pad = max(1.0, abs(lo) * 0.5)
        else:
            pad = max(0.15 * span, 0.5 if W == 10 else 0.05 * span)
        lo, hi = lo - pad, hi + pad
        if hi - lo < 3.0 and W == 10:  # keep a readable minimum width
            extra = (3.0 - (hi - lo)) / 2
            lo, hi = lo - extra, hi + extra
            if lo < window_domain[0]:
                hi += window_domain[0] - lo
            if hi > window_domain[1]:
                lo -= hi - window_domain[1]
        lo, hi = max(lo, window_domain[0]), min(hi, window_domain[1])
        if hi - lo > 1e-9:
            return (lo, hi)
    if domain is not None:
        return (max(-10.0, domain[0]), min(10.0, domain[1])) if domain[0] < 10 and domain[1] > -10 else domain
    return None


def _bbox(X, Y, mask, W) -> tuple[tuple[float, float, float, float], bool]:
    xs, ys = X[mask], Y[mask]
    x0, x1, y0, y1 = float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())
    step = 2 * W / (X.shape[1] - 1)
    touches = x0 <= -W + step or x1 >= W - step or y0 <= -W + step or y1 >= W - step
    return (x0, x1, y0, y1), touches


def _pad_box(box, rel=0.1, minimum=0.5):
    x0, x1, y0, y1 = box
    px = max((x1 - x0) * rel, minimum)
    py = max((y1 - y0) * rel, minimum)
    return (x0 - px, x1 + px, y0 - py, y1 + py)


def set_features(ctx, node, xname, yname, equation: bool) -> Optional[tuple[float, float, float, float]]:
    first = None
    for W in WINDOWS:
        n = 161
        X, Y = np.meshgrid(np.linspace(-W, W, n), np.linspace(-W, W, n))
        Z = _safe_grid(ctx, node, xname, yname, X, Y)
        if equation:
            s = np.sign(np.where(np.isfinite(Z), Z, 0.0))
            mask = np.zeros(Z.shape, dtype=bool)
            mask[:, :-1] |= (s[:, :-1] * s[:, 1:] < 0)
            mask[:-1, :] |= (s[:-1, :] * s[1:, :] < 0)
            mask |= (Z == 0)
        else:
            mask = np.where(np.isfinite(Z), Z, 0.0).astype(bool)
            if mask.all():
                return None
        if not mask.any():
            continue
        box, touches = _bbox(X, Y, mask, W)
        if not touches:
            return _pad_box(box)
        if first is None:
            first = box
    return _pad_box(first) if first is not None else None


def field_features(ctx, node, xname, yname) -> tuple[float, float, float, float]:
    W = 10.0
    X, Y = np.meshgrid(np.linspace(-W, W, 121), np.linspace(-W, W, 121))
    Z = _safe_grid(ctx, node, xname, yname, X, Y)
    finite = np.isfinite(Z)
    if finite.sum() < 10:
        return (-W, W, -W, W)
    Zf = np.where(finite, Z, np.nan)

    def interior(idx):
        i, j = np.unravel_index(idx, Z.shape)
        return 2 < i < Z.shape[0] - 3 and 2 < j < Z.shape[1] - 3

    imin, imax = int(np.nanargmin(Zf)), int(np.nanargmax(Zf))
    if interior(imin):
        mask = finite & (Zf <= np.nanquantile(Zf, 0.2))
    elif interior(imax):
        mask = finite & (Zf >= np.nanquantile(Zf, 0.8))
    else:
        return (-W, W, -W, W)
    box, _ = _bbox(X, Y, mask, W)
    x0, x1, y0, y1 = _pad_box(box, 0.35, 1.0)
    return (max(x0, -W), min(x1, W), max(y0, -W), min(y1, W))


# ------------------------------------------------------------------ resolution
def _extent(series: list[P.Series]) -> tuple[Bounds, Bounds]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for s in series:
        if isinstance(s, (P.LineSeries, P.ScatterSeries, P.PolygonSeries)):
            xs.append(np.asarray(s.x, float)); ys.append(np.asarray(s.y, float))
        elif isinstance(s, P.BarSeries):
            xs.append(np.asarray(s.x, float)); ys.append(np.append(np.asarray(s.h, float), 0.0))
        elif isinstance(s, P.ArrowSeries):
            xs.append(np.array([s.x0, s.x0 + s.dx])); ys.append(np.array([s.y0, s.y0 + s.dy]))
        elif isinstance(s, P.TextSeries):
            xs.append(np.array([s.x])); ys.append(np.array([s.y]))
        elif isinstance(s, P.LineRefSeries):
            (ys if s.orientation == "h" else xs).append(np.array([s.value]))
        elif isinstance(s, (P.ContourSeries, P.ImplicitSeries, P.RegionSeries, P.SurfaceSeries)):
            xs.append(np.asarray(s.X, float).reshape(-1)); ys.append(np.asarray(s.Y, float).reshape(-1))

    def rng(parts):
        if not parts:
            return None
        data = np.concatenate([p.reshape(-1) for p in parts])
        data = data[np.isfinite(data)]
        if data.size == 0:
            return None
        lo, hi = float(data.min()), float(data.max())
        pad = (hi - lo) * 0.08 or max(1.0, abs(lo) * 0.1)
        return (lo - pad, hi + pad)
    return rng(xs), rng(ys)


def resolve_plot(data: P.PlotData) -> None:
    deferred = [s for s in data.series if isinstance(s, DeferredSeries)]
    if deferred:
        concrete = [s for s in data.series if not isinstance(s, DeferredSeries)]
        ex, ey = _extent(concrete)
        x: Bounds = data.xlim or ex
        y: Bounds = data.ylim or ey
        for s in deferred:
            if s.role == "set" and (x is None or y is None):
                box = s.features()
                if box is not None:
                    x = x or (box[0], box[1])
                    y = y or (box[2], box[3])
        if x is None:
            ranges = [r for r in (s.features() for s in deferred if s.role == "curve") if r is not None]
            if ranges:
                x = (min(r[0] for r in ranges), max(r[1] for r in ranges))
        for s in deferred:
            if s.role == "field" and (x is None or y is None):
                box = s.features()
                x = x or (box[0], box[1])
                y = y or (box[2], box[3])
        x = x or (-10.0, 10.0)
        y = y or x
        resolved: list[P.Series] = []
        for s in data.series:
            if isinstance(s, DeferredSeries):
                try:
                    resolved.extend(P.collect_series(s.build(x, y)))
                except EvalError as exc:
                    data.errors.append(str(exc))
            else:
                resolved.append(s)
        data.series = resolved
    if data.ylim is None:
        _clip_poles(data)


def _clip_poles(data: P.PlotData) -> None:
    curves = [s for s in data.series if isinstance(s, P.LineSeries) and getattr(s, "auto_clip", False)]
    if not curves:
        return
    values = np.concatenate([np.asarray(s.y, float) for s in curves])
    values = values[np.isfinite(values)]
    if values.size < 20:
        return
    q1, q99 = np.quantile(values, [0.02, 0.98])
    robust = q99 - q1
    full = values.max() - values.min()
    if robust > 0 and full > 8 * robust:
        others = [s for s in data.series if s not in curves]
        _, ey = _extent(others)
        lo, hi = q1 - 0.25 * robust, q99 + 0.25 * robust
        if ey is not None:
            lo, hi = min(lo, ey[0]), max(hi, ey[1])
        data.ylim = (float(lo), float(hi))
        for s in curves:  # break the line at poles instead of drawing vertical jumps
            yy = np.asarray(s.y, float).copy()
            jump = np.abs(np.diff(yy)) > 3 * (hi - lo)
            yy[1:][jump] = np.nan
            s.y = yy


# ------------------------------------------------------------------ registration
def register(r):
    def wrap(name: str, role: str, short_len: int, make_full: Callable, feature_fn: Callable, sig: str):
        spec = r.get(name)
        original = spec.impl

        def impl(ctx, args, kwargs):
            if len(args) != short_len:
                result = original(ctx, args, kwargs)
                if name == "fplot" and isinstance(result, P.LineSeries):
                    result.auto_clip = True
                return result

            def build(xlim, ylim):
                result = original(ctx, make_full(args, xlim, ylim), kwargs)
                if isinstance(result, P.LineSeries) and name == "fplot":
                    result.auto_clip = True
                return result

            return DeferredSeries(role=role, build=build, features=lambda: feature_fn(ctx, args))

        spec.impl = impl
        spec.signature = sig

    wrap("fplot", "curve", 2,
         lambda a, x, y: (a[0], a[1], _num(x[0]), _num(x[1])),
         lambda ctx, a: curve_features(ctx, a[0], P._var(a[1])),
         "fplot(f(x), x [, a, b], n=400, label=, …)")

    def grid_full(a, x, y):
        return (a[0], a[1], _num(x[0]), _num(x[1]), a[2], _num(y[0]), _num(y[1]))

    def eq_node(node):
        node = A.strip_parens(node)
        if isinstance(node, A.Compare) and len(node.ops) == 1 and node.ops[0] == "==":
            return A.BinOp("-", node.operands[0], node.operands[1]), True
        return node, False

    wrap("implicit", "set", 3, grid_full,
         lambda ctx, a: set_features(ctx, eq_node(a[0])[0], P._var(a[1]), P._var(a[2]), True),
         "implicit(lhs == rhs, x [, a, b], y [, c, d])")
    wrap("region", "set", 3, grid_full,
         lambda ctx, a: set_features(ctx, a[0], P._var(a[1]), P._var(a[2]), False),
         "region(conditions, x [, a, b], y [, c, d], color=, alpha=)")
    wrap("contour", "field", 3, grid_full,
         lambda ctx, a: field_features(ctx, a[0], P._var(a[1]), P._var(a[2])),
         "contour(f(x, y), x [, a, b], y [, c, d], levels=12, filled=0)")
    wrap("surface", "field", 3, grid_full,
         lambda ctx, a: field_features(ctx, a[0], P._var(a[1]), P._var(a[2])),
         "surface(f(x, y), x [, a, b], y [, c, d], wireframe=0)")

    spec = r.get("parametric")
    original_parametric = spec.impl

    def parametric(ctx, args, kwargs):
        if len(args) == 3:
            args = (args[0], args[1], args[2], _num(0.0), A.BinOp("*", _num(2.0), A.Var("π")))
        return original_parametric(ctx, args, kwargs)

    spec.impl = parametric
    spec.signature = "parametric(x(t), y(t), t [, a, b], n=400)"
