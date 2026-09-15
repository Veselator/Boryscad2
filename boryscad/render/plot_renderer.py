"""Matplotlib rendering of plot regions (thread-safe, no pyplot)."""
from __future__ import annotations

import io
from typing import Callable

import matplotlib

matplotlib.use("Agg", force=False)

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from ..core.document import PlotRegion
from ..core.theme import Theme
from ..library import plotting as P

SeriesPainter = Callable[["PlotContext", P.Series], None]
_PAINTERS: dict[str, SeriesPainter] = {}


def series_painter(kind: str):
    def deco(fn: SeriesPainter) -> SeriesPainter:
        _PAINTERS[kind] = fn
        return fn
    return deco


class PlotContext:
    def __init__(self, ax, theme: Theme, is3d: bool):
        self.ax = ax
        self.theme = theme
        self.is3d = is3d
        self.cycle = list(theme.plot_cycle) or ["#1f77b4"]
        self._next = 0
        self.has_labels = False

    def color(self, series: P.Series) -> str:
        if series.color:
            return _mpl_color(series.color)
        c = self.cycle[self._next % len(self.cycle)]
        self._next += 1
        return c

    def label(self, series: P.Series):
        if series.label:
            self.has_labels = True
            return series.label
        return None


_NAMED = {"red": "#e63946", "blue": "#1d4ed8", "green": "#2a9d8f", "orange": "#f4a261", "purple": "#8338ec",
          "black": "#111111", "gray": "#888888", "grey": "#888888", "yellow": "#e9c46a", "pink": "#ff5c8a"}


def _mpl_color(value: str) -> str:
    return _NAMED.get(value.lower(), value)


@series_painter("line")
def _line(ctx: PlotContext, s: P.LineSeries):
    kw = dict(color=ctx.color(s), lw=s.width, ls=s.style or "-", alpha=s.alpha, label=ctx.label(s))
    if s.marker:
        kw["marker"] = s.marker
    if ctx.is3d and s.z is not None:
        ctx.ax.plot(s.x, s.y, s.z, **kw)
    else:
        ctx.ax.plot(s.x, s.y, **kw)


@series_painter("scatter")
def _scatter(ctx: PlotContext, s: P.ScatterSeries):
    color = ctx.color(s)
    if ctx.is3d and s.z is not None:
        ctx.ax.scatter(s.x, s.y, s.z, s=s.size, c=color, marker=s.marker or "o", alpha=s.alpha, label=ctx.label(s))
    else:
        ctx.ax.scatter(s.x, s.y, s=s.size, c=color, marker=s.marker or "o", alpha=s.alpha, label=ctx.label(s),
                       zorder=5, edgecolors="none")
        for x, y, text in zip(s.x, s.y, s.annotations):
            ctx.ax.annotate(text, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8,
                            color=ctx.theme.color("plot_fg"))


@series_painter("contour")
def _contour(ctx: PlotContext, s: P.ContourSeries):
    if ctx.is3d:
        ctx.ax.contour(s.X, s.Y, s.Z, levels=s.levels, cmap="viridis")
        return
    if s.filled:
        cs = ctx.ax.contourf(s.X, s.Y, s.Z, levels=s.levels, cmap="viridis", alpha=0.75 * s.alpha)
        ctx.ax.figure.colorbar(cs, ax=ctx.ax, shrink=0.85)
    else:
        kw = {"colors": [_mpl_color(s.color)]} if s.color else {"cmap": "viridis"}
        cs = ctx.ax.contour(s.X, s.Y, s.Z, levels=s.levels, linewidths=max(0.6, s.width * 0.6), alpha=s.alpha, **kw)
        if s.labels:
            ctx.ax.clabel(cs, inline=True, fontsize=7, fmt="%g")
    if s.label:
        ctx.ax.plot([], [], color=_mpl_color(s.color) if s.color else "#440154", label=s.label)
        ctx.has_labels = True


@series_painter("implicit")
def _implicit(ctx: PlotContext, s: P.ImplicitSeries):
    color = ctx.color(s)
    if np.nanmin(s.Z) > 0 or np.nanmax(s.Z) < 0:
        return
    ctx.ax.contour(s.X, s.Y, s.Z, levels=[0], colors=[color], linewidths=s.width, linestyles=s.style or "-")
    if s.label:
        ctx.ax.plot([], [], color=color, lw=s.width, ls=s.style or "-", label=s.label)
        ctx.has_labels = True


@series_painter("region")
def _region(ctx: PlotContext, s: P.RegionSeries):
    color = ctx.color(s)
    from matplotlib.colors import ListedColormap
    masked = np.ma.masked_where(~s.mask, np.ones(s.mask.shape))
    ctx.ax.pcolormesh(s.X, s.Y, masked, cmap=ListedColormap([color]), alpha=s.alpha, shading="auto", zorder=0)
    if s.label:
        import matplotlib.patches as mpatches
        ctx.ax.add_patch(mpatches.Rectangle((np.nan, np.nan), 0, 0, color=color, alpha=s.alpha, label=s.label))
        ctx.has_labels = True


@series_painter("arrow")
def _arrow(ctx: PlotContext, s: P.ArrowSeries):
    color = ctx.color(s)
    ctx.ax.annotate("", xy=(s.x0 + s.dx, s.y0 + s.dy), xytext=(s.x0, s.y0),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=s.width, mutation_scale=14))
    ctx.ax.plot([s.x0, s.x0 + s.dx], [s.y0, s.y0 + s.dy], alpha=0, label=None)
    if s.label:
        ctx.ax.plot([], [], color=color, lw=s.width, label=s.label)
        ctx.has_labels = True


@series_painter("refline")
def _refline(ctx: PlotContext, s: P.LineRefSeries):
    kw = dict(color=ctx.color(s), lw=s.width, ls=s.style or "-", alpha=s.alpha, label=ctx.label(s))
    if s.orientation == "h":
        ctx.ax.axhline(s.value, **kw)
    else:
        ctx.ax.axvline(s.value, **kw)


@series_painter("text")
def _text(ctx: PlotContext, s: P.TextSeries):
    ctx.ax.text(s.x, s.y, s.text, color=_mpl_color(s.color) if s.color else ctx.theme.color("plot_fg"), fontsize=9)


@series_painter("bar")
def _bar(ctx: PlotContext, s: P.BarSeries):
    width = 0.8 * (np.min(np.diff(np.sort(s.x))) if s.x.size > 1 else 1.0)
    ctx.ax.bar(s.x, s.h, width=width, color=ctx.color(s), alpha=s.alpha, label=ctx.label(s))


@series_painter("polygon")
def _polygon(ctx: PlotContext, s: P.PolygonSeries):
    color = ctx.color(s)
    if s.fill:
        ctx.ax.fill(s.x, s.y, color=color, alpha=s.alpha if s.alpha < 1 else 0.3, label=ctx.label(s), zorder=1)
        ctx.ax.plot(np.append(s.x, s.x[:1]), np.append(s.y, s.y[:1]), color=color, lw=1.0)
    else:
        ctx.ax.plot(np.append(s.x, s.x[:1]), np.append(s.y, s.y[:1]), color=color, lw=s.width, label=ctx.label(s))


@series_painter("surface")
def _surface(ctx: PlotContext, s: P.SurfaceSeries):
    if not ctx.is3d:
        cs = ctx.ax.contourf(s.X, s.Y, s.Z, levels=20, cmap="viridis")
        ctx.ax.figure.colorbar(cs, ax=ctx.ax, shrink=0.85)
        return
    if s.wireframe:
        ctx.ax.plot_wireframe(s.X, s.Y, s.Z, color=ctx.color(s), linewidth=0.6)
    else:
        ctx.ax.plot_surface(s.X, s.Y, s.Z, cmap="viridis", alpha=s.alpha if s.alpha < 1 else 0.92, linewidth=0,
                            antialiased=True)


@series_painter("graph")
def _graph(ctx: PlotContext, s: P.GraphSeries):
    ax = ctx.ax
    fg = ctx.theme.color("plot_fg")
    hl = {(u, v) for u, v in s.highlight} | ({(v, u) for u, v in s.highlight} if not s.directed else set())
    accent = _mpl_color(s.color) if s.color else ctx.theme.color("accent", "#e63946")
    for u, v, w in s.edges:
        (x1, y1), (x2, y2) = s.positions[u], s.positions[v]
        on = (u, v) in hl
        ax.plot([x1, x2], [y1, y2], color=accent if on else ctx.theme.color("plot_grid"), lw=3.2 if on else 1.3,
                zorder=2 if on else 1, solid_capstyle="round")
        if s.show_weights:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2, f"{w:g}", fontsize=7.5, color=fg, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc=ctx.theme.color("plot_bg"), ec="none", alpha=0.85), zorder=3)
    on_nodes = {n for pair in s.highlight for n in pair}
    for node, (x, y) in s.positions.items():
        ax.scatter([x], [y], s=260, c=accent if node in on_nodes else ctx.theme.color("surface", "#ffffff"),
                   edgecolors=fg, linewidths=1.2, zorder=4)
        ax.text(x, y, s.names.get(node, str(node)), fontsize=8, ha="center", va="center",
                color="#ffffff" if node in on_nodes else fg, zorder=5, fontweight="bold")
    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")


def render_figure(region: PlotRegion, data: P.PlotData, theme: Theme, width: float, height: float, dpi: float = 100.0) -> Figure:
    fig = Figure(figsize=(max(width, 50) / 100.0, max(height, 40) / 100.0), dpi=dpi)
    FigureCanvasAgg(fig)
    bg, fg, grid = theme.color("plot_bg"), theme.color("plot_fg"), theme.color("plot_grid")
    fig.patch.set_facecolor(bg)
    is3d = region.mode == "3d" or any(s.series_kind == "surface" for s in data.series) and region.mode != "2d"
    if is3d:
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor(bg)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color((0, 0, 0, 0))
            axis.label.set_color(fg)
        ax.tick_params(colors=fg, labelsize=8)
    elif region.mode == "polar":
        ax = fig.add_subplot(111, projection="polar")
        ax.set_facecolor(bg)
        ax.tick_params(colors=fg, labelsize=8)
    else:
        ax = fig.add_subplot(111)
        ax.set_facecolor(bg)
        for spine in ax.spines.values():
            spine.set_color(grid)
        ax.tick_params(colors=fg, labelsize=8)
    ctx = PlotContext(ax, theme, is3d)
    for series in data.series:
        painter = _PAINTERS.get(series.series_kind)
        if painter is not None:
            try:
                painter(ctx, series)
            except Exception as exc:  # a broken series must not break the whole plot
                data.errors.append(f"{series.series_kind}: {exc}")
    if region.title:
        ax.set_title(region.title, color=fg, fontsize=10)
    if region.xlabel:
        ax.set_xlabel(region.xlabel, color=fg, fontsize=9)
    if region.ylabel:
        ax.set_ylabel(region.ylabel, color=fg, fontsize=9)
    if is3d and region.zlabel:
        ax.set_zlabel(region.zlabel, color=fg, fontsize=9)
    if data.xlim:
        ax.set_xlim(*data.xlim)
    if data.ylim:
        ax.set_ylim(*data.ylim)
    if is3d and data.zlim:
        ax.set_zlim(*data.zlim)
    if not is3d and region.mode != "polar":
        if region.xscale in ("log", "symlog"):
            ax.set_xscale(region.xscale)
        if region.yscale in ("log", "symlog"):
            ax.set_yscale(region.yscale)
        if region.equal:
            ax.set_aspect("equal", adjustable="datalim" if not (data.xlim and data.ylim) else "box")
        if not any(s.series_kind == "graph" for s in data.series):
            ax.axhline(0, color=grid, lw=0.9, zorder=0)
            ax.axvline(0, color=grid, lw=0.9, zorder=0)
    if region.grid and not any(s.series_kind == "graph" for s in data.series):
        ax.grid(True, color=grid, lw=0.6, alpha=0.9)
    if region.legend and ctx.has_labels:
        leg = ax.legend(fontsize=8, framealpha=0.85, facecolor=bg, edgecolor=grid, labelcolor=fg)
        leg.set_zorder(10)
    if not data.series:
        ax.text(0.5, 0.5, "empty plot — add series", ha="center", va="center", transform=ax.transAxes, color=fg,
                fontsize=10, alpha=0.6)
    try:
        fig.tight_layout(pad=0.6)
    except Exception:
        pass
    return fig


def render_rgba(region: PlotRegion, data: P.PlotData, theme: Theme, width: float, height: float,
                scale: float = 1.0) -> tuple[bytes, int, int]:
    fig = render_figure(region, data, theme, width, height, dpi=100.0 * scale)
    canvas = fig.canvas
    canvas.draw()
    buf = canvas.buffer_rgba()
    w, h = canvas.get_width_height()
    return bytes(buf), w, h


def render_png(region: PlotRegion, data: P.PlotData, theme: Theme, width: float, height: float, scale: float = 2.0) -> bytes:
    fig = render_figure(region, data, theme, width, height, dpi=100.0 * scale)
    out = io.BytesIO()
    fig.savefig(out, format="png", facecolor=fig.get_facecolor())
    return out.getvalue()


def render_svg(region: PlotRegion, data: P.PlotData, theme: Theme, width: float, height: float) -> str:
    fig = render_figure(region, data, theme, width, height, dpi=100.0)
    out = io.StringIO()
    fig.savefig(out, format="svg", facecolor=fig.get_facecolor())
    return out.getvalue()
