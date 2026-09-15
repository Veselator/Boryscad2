"""Plot series constructors.

A plot region contains expressions; each expression evaluates to one or more
series objects defined here.  Rendering is done by
:mod:`boryscad.render.plot_renderer` which dispatches on ``series_kind``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Optional

import numpy as np

from ..engine.values import EvalError, RangeValue, SeriesGroup, normalize_number
from ..language import ast as A
from ._util import to_float, to_int, vec


# ------------------------------------------------------------------ series
@dataclass
class Series:
    series_kind: ClassVar[str] = "series"
    label: str = ""
    color: Optional[str] = None
    width: float = 2.0
    style: str = "-"
    marker: str = ""
    alpha: float = 1.0
    size: float = 36.0

    def display(self):
        from ..engine.values import DText
        return DText(f"‹{self.series_kind} series{': ' + self.label if self.label else ''}›", "note")


@dataclass
class LineSeries(Series):
    series_kind: ClassVar[str] = "line"
    x: np.ndarray = field(default_factory=lambda: np.zeros(0))
    y: np.ndarray = field(default_factory=lambda: np.zeros(0))
    z: Optional[np.ndarray] = None


@dataclass
class ScatterSeries(Series):
    series_kind: ClassVar[str] = "scatter"
    x: np.ndarray = field(default_factory=lambda: np.zeros(0))
    y: np.ndarray = field(default_factory=lambda: np.zeros(0))
    z: Optional[np.ndarray] = None
    annotations: list[str] = field(default_factory=list)


@dataclass
class ContourSeries(Series):
    series_kind: ClassVar[str] = "contour"
    X: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    Y: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    Z: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    levels: Any = 12
    filled: bool = False
    labels: bool = True


@dataclass
class ImplicitSeries(Series):
    series_kind: ClassVar[str] = "implicit"
    X: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    Y: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    Z: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))


@dataclass
class RegionSeries(Series):
    series_kind: ClassVar[str] = "region"
    X: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    Y: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    mask: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))


@dataclass
class ArrowSeries(Series):
    series_kind: ClassVar[str] = "arrow"
    x0: float = 0.0
    y0: float = 0.0
    dx: float = 1.0
    dy: float = 1.0


@dataclass
class LineRefSeries(Series):
    series_kind: ClassVar[str] = "refline"
    value: float = 0.0
    orientation: str = "h"


@dataclass
class TextSeries(Series):
    series_kind: ClassVar[str] = "text"
    x: float = 0.0
    y: float = 0.0
    text: str = ""


@dataclass
class BarSeries(Series):
    series_kind: ClassVar[str] = "bar"
    x: np.ndarray = field(default_factory=lambda: np.zeros(0))
    h: np.ndarray = field(default_factory=lambda: np.zeros(0))


@dataclass
class PolygonSeries(Series):
    series_kind: ClassVar[str] = "polygon"
    x: np.ndarray = field(default_factory=lambda: np.zeros(0))
    y: np.ndarray = field(default_factory=lambda: np.zeros(0))
    fill: bool = True


@dataclass
class SurfaceSeries(Series):
    series_kind: ClassVar[str] = "surface"
    X: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    Y: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    Z: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    wireframe: bool = False


@dataclass
class GraphSeries(Series):
    series_kind: ClassVar[str] = "graph"
    positions: dict = field(default_factory=dict)          # node -> (x, y)
    edges: list = field(default_factory=list)              # (u, v, w)
    names: dict = field(default_factory=dict)              # node -> label
    highlight: list = field(default_factory=list)          # [(u, v)]
    show_weights: bool = True
    directed: bool = False
    node_colors: dict = field(default_factory=dict)


@dataclass
class PlotData:
    series: list[Series] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    xlim: Optional[tuple[float, float]] = None
    ylim: Optional[tuple[float, float]] = None
    zlim: Optional[tuple[float, float]] = None


def collect_series(value: Any, source: str = "", index: int = 0) -> list[Series]:
    if isinstance(value, Series):
        return [value]
    if isinstance(value, (SeriesGroup, list, tuple)):
        out: list[Series] = []
        for v in value:
            out.extend(collect_series(v, source, index))
        return out
    if isinstance(value, RangeValue):
        value = value.to_array()
    if isinstance(value, np.ndarray) and value.dtype != object:
        if value.ndim == 1:
            return [LineSeries(label=source, x=np.arange(value.size), y=value.astype(float))]
        if value.ndim == 2 and value.shape[1] >= 2:
            return [LineSeries(label=source if value.shape[1] == 2 else f"{source}[{j}]",
                               x=value[:, 0].astype(float), y=value[:, j].astype(float)) for j in range(1, value.shape[1])]
    raise EvalError("a plot expression must produce a series, e.g. line(x, f(x)) or fplot(sin(x), x, 0, 10)")


# -------------------------------------------------------------- functions
_STYLE_KEYS = {"label", "color", "width", "style", "marker", "alpha", "size"}


def _style(kwargs: dict[str, Any]) -> dict[str, Any]:
    unknown = set(kwargs) - _STYLE_KEYS - {"levels", "filled", "n", "fill", "labels", "names", "path", "weights",
                                           "pos", "directed", "wireframe", "annotate", "colors"}
    if unknown:
        raise EvalError(f"unknown option(s): {', '.join(sorted(unknown))}")
    out = {}
    for k in _STYLE_KEYS & set(kwargs):
        v = kwargs[k]
        if k in ("width", "alpha", "size"):
            out[k] = to_float(v, k)
        else:
            out[k] = str(v)
    return out


def _var(node: A.Expr) -> str:
    node = A.strip_parens(node)
    if not isinstance(node, A.Var):
        raise EvalError("expected a variable name")
    return node.name


def _grid(ctx, args, n):
    xname, yname = _var(args[1]), _var(args[4])
    a, b = to_float(ctx.eval(args[2])), to_float(ctx.eval(args[3]))
    c, d = to_float(ctx.eval(args[5])), to_float(ctx.eval(args[6]))
    xs, ys = np.linspace(a, b, n), np.linspace(c, d, n)
    X, Y = np.meshgrid(xs, ys)
    return xname, yname, X, Y


def _eval_grid(ctx, node, xname, yname, X, Y):
    try:
        Z = ctx.eval_vectorized(node, {xname: X, yname: Y})
        Z = np.asarray(Z, dtype=float)
        if Z.shape == X.shape:
            return Z
        if Z.size == 1:
            return np.full(X.shape, float(Z.reshape(-1)[0]))
    except (EvalError, TypeError, ValueError):
        pass
    Z = np.empty(X.shape)
    for idx in np.ndindex(X.shape):
        v = ctx.eval(node, {xname: float(X[idx]), yname: float(Y[idx])})
        v = normalize_number(v)
        Z[idx] = float(v.real) if isinstance(v, complex) else float(v)
    return Z


def _eval_curve(ctx, node, name, xs):
    try:
        ys = ctx.eval_vectorized(node, {name: xs})
        ys = np.asarray(ys)
        if ys.shape == xs.shape:
            return np.real(ys).astype(float)
        if ys.size == 1:
            return np.full(xs.shape, float(np.real(ys.reshape(-1)[0])))
    except (EvalError, TypeError, ValueError):
        pass
    out = np.empty(xs.shape)
    for i, x in enumerate(xs):
        try:
            v = normalize_number(ctx.eval(node, {name: float(x)}))
            out[i] = float(v.real) if isinstance(v, complex) else float(v)
        except EvalError:
            out[i] = np.nan
    return out


def register(r):
    cat = "Plots"

    @r.function(category=cat, signature="line(x, y, label=, color=, width=, style=, marker=)",
                doc="line through points (x, y)")
    def line(x, y, z=None, **kw):
        return LineSeries(**_style(kw), x=vec(x, "x"), y=vec(y, "y"), z=None if z is None else vec(z, "z"))

    @r.function(category=cat, signature="scatter(x, y, label=, color=, size=, marker=, annotate=)",
                doc="points (x, y)", aliases=("points",))
    def scatter(x, y, z=None, annotate=None, **kw):
        style = _style(kw)
        style.setdefault("marker", "o")
        ann = []
        if annotate is not None:
            if isinstance(annotate, np.ndarray):
                ann = [str(normalize_number(a)) for a in annotate.reshape(-1)]
            elif isinstance(annotate, (int, float)) and annotate:
                ann = [str(i + 1) for i in range(vec(x).size)]
        return ScatterSeries(**style, x=vec(x, "x"), y=vec(y, "y"), z=None if z is None else vec(z, "z"), annotations=ann)

    @r.function(lazy=True, category=cat, signature="fplot(f(x), x, a, b, n=400, label=, …)",
                doc="graph of a function of one variable")
    def fplot(ctx, args, kwargs):
        if len(args) != 4:
            raise EvalError("fplot(expression, x, from, to)")
        opts = ctx.eval_kwargs(kwargs)
        n = to_int(opts.pop("n", 400))
        name = _var(args[1])
        a, b = to_float(ctx.eval(args[2])), to_float(ctx.eval(args[3]))
        xs = np.linspace(a, b, n)
        ys = _eval_curve(ctx, args[0], name, xs)
        style = _style(opts)
        style.setdefault("label", "")
        return LineSeries(**style, x=xs, y=ys)

    @r.function(lazy=True, category=cat, signature="contour(f(x, y), x, a, b, y, c, d, levels=12, filled=0)",
                doc="contour lines (level curves) of f")
    def contour(ctx, args, kwargs):
        if len(args) != 7:
            raise EvalError("contour(expression, x, x0, x1, y, y0, y1)")
        opts = ctx.eval_kwargs(kwargs)
        n = to_int(opts.pop("n", 160))
        levels = opts.pop("levels", 12)
        levels = to_int(levels) if not isinstance(levels, np.ndarray) else np.sort(levels.astype(float))
        filled = bool(opts.pop("filled", 0))
        labels = bool(opts.pop("labels", 1))
        xname, yname, X, Y = _grid(ctx, args, n)
        Z = _eval_grid(ctx, args[0], xname, yname, X, Y)
        return ContourSeries(**_style(opts), X=X, Y=Y, Z=Z, levels=levels, filled=filled, labels=labels)

    @r.function(lazy=True, category=cat, signature="implicit(lhs == rhs, x, a, b, y, c, d)",
                doc="curve given by an equation F(x, y) = 0")
    def implicit(ctx, args, kwargs):
        if len(args) != 7:
            raise EvalError("implicit(equation, x, x0, x1, y, y0, y1)")
        opts = ctx.eval_kwargs(kwargs)
        n = to_int(opts.pop("n", 300))
        expr = A.strip_parens(args[0])
        if isinstance(expr, A.Compare) and len(expr.ops) == 1:
            expr = A.BinOp("-", expr.operands[0], expr.operands[1])
        xname, yname, X, Y = _grid(ctx, args, n)
        Z = _eval_grid(ctx, expr, xname, yname, X, Y)
        return ImplicitSeries(**_style(opts), X=X, Y=Y, Z=Z)

    @r.function(lazy=True, category=cat, signature="region(conditions, x, a, b, y, c, d, color=, alpha=)",
                doc="shade the set of points where the condition holds (e.g. x + y <= 5 and x >= 0)")
    def region(ctx, args, kwargs):
        if len(args) != 7:
            raise EvalError("region(condition, x, x0, x1, y, y0, y1)")
        opts = ctx.eval_kwargs(kwargs)
        n = to_int(opts.pop("n", 400))
        style = _style(opts)
        style.setdefault("alpha", 0.25)
        xname, yname, X, Y = _grid(ctx, args, n)
        M = _eval_grid(ctx, args[0], xname, yname, X, Y)
        return RegionSeries(**style, X=X, Y=Y, mask=M.astype(bool))

    @r.function(category=cat, signature="arrow(x0, y0, dx, dy, label=, color=)", doc="arrow (vector) from a point")
    def arrow(x0, y0, dx, dy, **kw):
        return ArrowSeries(**_style(kw), x0=to_float(x0), y0=to_float(y0), dx=to_float(dx), dy=to_float(dy))

    @r.function(category=cat, signature="hline(y, …)", doc="horizontal reference line")
    def hline(y, **kw):
        return LineRefSeries(**_style(kw), value=to_float(y), orientation="h")

    @r.function(category=cat, signature="vline(x, …)", doc="vertical reference line")
    def vline(x, **kw):
        return LineRefSeries(**_style(kw), value=to_float(x), orientation="v")

    @r.function(category=cat, signature="label(x, y, text)", doc="text annotation on a plot", aliases=("annotate",))
    def label(x, y, text, **kw):
        return TextSeries(**_style(kw), x=to_float(x), y=to_float(y), text=str(normalize_number(text)))

    @r.function(category=cat, signature="bar(x, h, label=, color=)", doc="bar chart")
    def bar(x, h, **kw):
        return BarSeries(**_style(kw), x=vec(x, "x"), h=vec(h, "h"))

    @r.function(category=cat, signature="polygon(x, y, fill=1, …)", doc="closed polygon")
    def polygon(x, y, fill=1, **kw):
        style = _style(kw)
        return PolygonSeries(**style, x=vec(x, "x"), y=vec(y, "y"), fill=bool(fill))

    @r.function(lazy=True, category=cat, signature="surface(f(x, y), x, a, b, y, c, d, wireframe=0)",
                doc="3D surface z = f(x, y) (use in a 3D plot)")
    def surface(ctx, args, kwargs):
        if len(args) != 7:
            raise EvalError("surface(expression, x, x0, x1, y, y0, y1)")
        opts = ctx.eval_kwargs(kwargs)
        n = to_int(opts.pop("n", 60))
        wire = bool(opts.pop("wireframe", 0))
        xname, yname, X, Y = _grid(ctx, args, n)
        Z = _eval_grid(ctx, args[0], xname, yname, X, Y)
        return SurfaceSeries(**_style(opts), X=X, Y=Y, Z=Z, wireframe=wire)

    @r.function(category=cat, signature="surfm(M)", doc="3D surface from a matrix of heights")
    def surfm(m, **kw):
        m = np.asarray(m, dtype=float)
        X, Y = np.meshgrid(np.arange(m.shape[1]), np.arange(m.shape[0]))
        return SurfaceSeries(**_style(kw), X=X, Y=Y, Z=m)

    @r.function(lazy=True, category=cat, signature="graph(E, pos=P, names=, path=, weights=1, directed=0)",
                doc="draw a network: E rows are [from, to, weight]; path is a vector of nodes to highlight")
    def graph(ctx, args, kwargs):
        from .graphs import edges_from_value
        opts = ctx.eval_kwargs(kwargs)
        origin = ctx.origin
        edges = edges_from_value(ctx.eval(args[0]))
        nodes = sorted({u for u, v, w in edges} | {v for u, v, w in edges}, key=lambda n: (isinstance(n, str), n))
        names = {}
        if "names" in opts:
            labels = [str(normalize_number(s)) for s in np.asarray(opts.pop("names")).reshape(-1)]
            for k, node in enumerate(nodes):
                if k < len(labels):
                    names[node] = labels[k]
        positions = {}
        if "pos" in opts:
            P = np.asarray(opts.pop("pos"), dtype=float)
            for k, node in enumerate(nodes):
                if k < P.shape[0]:
                    positions[node] = (P[k, 0], P[k, 1])
        if len(positions) < len(nodes):
            import networkx as nx
            G = nx.Graph()
            for u, v, w in edges:
                G.add_edge(u, v, weight=max(float(w), 1e-6))
            try:
                layout = nx.kamada_kawai_layout(G, weight="weight")
            except Exception:
                layout = nx.spring_layout(G, seed=3)
            for node in nodes:
                positions.setdefault(node, tuple(layout[node]))
        highlight = []
        if "path" in opts:
            p = opts.pop("path")
            if hasattr(p, "fields") and "path" in getattr(p, "fields", {}):
                p = p.fields["path"]
            seq = [normalize_number(v) for v in np.asarray(p).reshape(-1)]
            highlight = list(zip(seq, seq[1:]))
        show_weights = bool(opts.pop("weights", 1))
        directed = bool(opts.pop("directed", 0))
        return GraphSeries(**_style(opts), positions=positions, edges=edges, names=names, highlight=highlight,
                           show_weights=show_weights, directed=directed)

    @r.function(lazy=True, category=cat, signature="parametric(x(t), y(t), t, a, b, n=400)",
                doc="parametric curve")
    def parametric(ctx, args, kwargs):
        if len(args) != 5:
            raise EvalError("parametric(x(t), y(t), t, from, to)")
        opts = ctx.eval_kwargs(kwargs)
        n = to_int(opts.pop("n", 400))
        name = _var(args[2])
        ts = np.linspace(to_float(ctx.eval(args[3])), to_float(ctx.eval(args[4])), n)
        return LineSeries(**_style(opts), x=_eval_curve(ctx, args[0], name, ts), y=_eval_curve(ctx, args[1], name, ts))
