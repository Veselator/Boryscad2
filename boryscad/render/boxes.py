"""Backend independent box layout for mathematical notation.

Boxes are measured once when constructed (using a :class:`Measurer`) and can
be drawn any number of times on any :class:`Canvas`.  Coordinates: ``x`` grows
to the right, ``y`` grows downwards, and ``draw(canvas, x, y)`` receives the
*baseline* position.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence


@dataclass(frozen=True)
class FontSpec:
    family: str
    size: float
    italic: bool = False
    bold: bool = False

    def scaled(self, factor: float) -> "FontSpec":
        return FontSpec(self.family, max(5.0, self.size * factor), self.italic, self.bold)

    def with_(self, italic: Optional[bool] = None, bold: Optional[bool] = None, family: Optional[str] = None) -> "FontSpec":
        return FontSpec(family or self.family, self.size, self.italic if italic is None else italic,
                        self.bold if bold is None else bold)


class Measurer(Protocol):
    def text(self, text: str, font: FontSpec) -> tuple[float, float, float]:
        """Return (advance width, ascent, descent)."""


class Canvas(Protocol):
    def text(self, x: float, y: float, text: str, font: FontSpec, color: str) -> None: ...
    def line(self, x1: float, y1: float, x2: float, y2: float, color: str, width: float = 1.0) -> None: ...
    def rect(self, x: float, y: float, w: float, h: float, fill: Optional[str], stroke: Optional[str] = None,
             radius: float = 0.0, width: float = 1.0) -> None: ...
    def polyline(self, points: Sequence[tuple[float, float]], color: str, width: float = 1.0,
                 closed: bool = False, fill: Optional[str] = None) -> None: ...
    def wave(self, x1: float, x2: float, y: float, color: str) -> None: ...


class Box:
    w: float = 0.0
    a: float = 0.0   # ascent above baseline
    d: float = 0.0   # descent below baseline

    @property
    def h(self) -> float:
        return self.a + self.d

    def draw(self, c: Canvas, x: float, y: float) -> None:  # pragma: no cover - abstract
        raise NotImplementedError


class Space(Box):
    def __init__(self, w: float, a: float = 0.0, d: float = 0.0):
        self.w, self.a, self.d = w, a, d

    def draw(self, c, x, y):
        pass


class Text(Box):
    def __init__(self, m: Measurer, text: str, font: FontSpec, color: str, pad_right: float = 0.0):
        self.text, self.font, self.color = text, font, color
        w, a, d = m.text(text, font) if text else (0.0, font.size * 0.75, font.size * 0.25)
        self.w = w + (font.size * 0.08 if font.italic and text else 0.0) + pad_right
        self.a, self.d = a, d

    def draw(self, c, x, y):
        if self.text:
            c.text(x, y, self.text, self.font, self.color)


class HBox(Box):
    def __init__(self, children: Sequence[Box], gap: float = 0.0):
        self.children = [ch for ch in children if ch is not None]
        self.gap = gap
        self.w = sum(ch.w for ch in self.children) + gap * max(0, len(self.children) - 1)
        self.a = max((ch.a for ch in self.children), default=0.0)
        self.d = max((ch.d for ch in self.children), default=0.0)

    def draw(self, c, x, y):
        for ch in self.children:
            ch.draw(c, x, y)
            x += ch.w + self.gap


class VStack(Box):
    """Vertical list; the baseline is the baseline of the first child."""

    def __init__(self, children: Sequence[Box], gap: float = 4.0, align: str = "left"):
        self.children = [ch for ch in children if ch is not None]
        self.gap, self.align = gap, align
        self.w = max((ch.w for ch in self.children), default=0.0)
        if not self.children:
            return
        self.a = self.children[0].a
        total = sum(ch.h for ch in self.children) + gap * (len(self.children) - 1)
        self.d = total - self.a

    def draw(self, c, x, y):
        top = y - self.a
        for ch in self.children:
            dx = 0.0
            if self.align == "center":
                dx = (self.w - ch.w) / 2
            elif self.align == "right":
                dx = self.w - ch.w
            ch.draw(c, x + dx, top + ch.a)
            top += ch.h + self.gap


class Raise(Box):
    """Shift a box vertically (positive ``dy`` moves it up)."""

    def __init__(self, child: Box, dy: float):
        self.child, self.dy = child, dy
        self.w = child.w
        self.a = max(0.0, child.a + dy)
        self.d = max(0.0, child.d - dy)

    def draw(self, c, x, y):
        self.child.draw(c, x, y - self.dy)


class Fraction(Box):
    def __init__(self, num: Box, den: Box, size: float, color: str, axis: float):
        self.num, self.den, self.color = num, den, color
        self.thick = max(1.0, size * 0.055)
        self.gap = size * 0.14
        self.pad = size * 0.12
        self.axis = axis
        self.w = max(num.w, den.w) + 2 * self.pad
        self.a = axis + self.thick / 2 + self.gap + num.h
        self.d = den.h + self.gap + self.thick / 2 - axis

    def draw(self, c, x, y):
        bar_y = y - self.axis
        self.num.draw(c, x + (self.w - self.num.w) / 2, bar_y - self.thick / 2 - self.gap - self.num.d)
        self.den.draw(c, x + (self.w - self.den.w) / 2, bar_y + self.thick / 2 + self.gap + self.den.a)
        c.line(x + self.pad * 0.4, bar_y, x + self.w - self.pad * 0.4, bar_y, self.color, self.thick)


class Scripts(Box):
    def __init__(self, base: Box, sup: Optional[Box], sub: Optional[Box], size: float):
        self.base, self.sup, self.sub = base, sup, sub
        self.size = size
        gap = size * 0.04
        self.sup_rise = 0.0
        self.sub_drop = 0.0
        a, d = base.a, base.d
        if sup is not None:
            self.sup_rise = max(size * 0.42, base.a - sup.a * 0.55)
            a = max(a, self.sup_rise + sup.a)
        if sub is not None:
            self.sub_drop = max(size * 0.2, base.d + sub.a * 0.35 - size * 0.12)
            d = max(d, self.sub_drop + sub.d)
            if sup is not None:
                min_sep = size * 0.15
                overlap = (self.sup_rise - sup.d) - (-self.sub_drop + sub.a)
                if overlap < min_sep:
                    self.sub_drop += (min_sep - overlap)
                    d = max(d, self.sub_drop + sub.d)
        self.gap = gap
        self.w = base.w + gap + max(sup.w if sup else 0.0, sub.w if sub else 0.0)
        self.a, self.d = a, d

    def draw(self, c, x, y):
        self.base.draw(c, x, y)
        sx = x + self.base.w + self.gap
        if self.sup is not None:
            self.sup.draw(c, sx, y - self.sup_rise)
        if self.sub is not None:
            self.sub.draw(c, sx, y + self.sub_drop)


class Radical(Box):
    def __init__(self, body: Box, size: float, color: str, index: Optional[Box] = None):
        self.body, self.color, self.index = body, color, index
        self.thick = max(1.0, size * 0.055)
        self.gap = size * 0.12
        self.sign_w = size * 0.55 + body.h * 0.08
        self.index_w = max(0.0, index.w - self.sign_w * 0.45) if index is not None else 0.0
        self.w = self.index_w + self.sign_w + body.w + size * 0.12
        self.a = body.a + self.gap + self.thick
        self.d = body.d + size * 0.02
        if index is not None:
            self.a = max(self.a, (self.a + self.d) * 0.55 - self.d + index.h)

    def draw(self, c, x, y):
        x0 = x + self.index_w
        top = y - self.body.a - self.gap - self.thick / 2
        bottom = y + self.body.d
        mid = y - (self.body.a - self.body.d) * 0.25
        sw = self.sign_w
        pts = [(x0, mid + sw * 0.08), (x0 + sw * 0.22, mid - sw * 0.06), (x0 + sw * 0.5, bottom),
               (x0 + sw * 0.95, top), (x0 + sw + self.body.w + sw * 0.1, top)]
        c.polyline(pts, self.color, self.thick)
        if self.index is not None:
            self.index.draw(c, x, mid - sw * 0.2 - self.index.d)
        self.body.draw(c, x0 + sw, y)


class Delimited(Box):
    """Body surrounded by stretchable delimiters: ( [ { | ‖ ⌊ ⌈ or empty string."""

    def __init__(self, m: Measurer, body: Box, left: str, right: str, font: FontSpec, color: str, axis: float):
        self.body, self.left, self.right, self.font, self.color = body, left, right, font, color
        size = font.size
        normal_h = size * 1.2
        self.stretch = body.h > normal_h * 1.08
        if self.stretch:
            half = max(body.a - axis, body.d + axis) + size * 0.1
            self.a = axis + half
            self.d = half - axis
            self.dw = size * 0.28 + body.h * 0.05
            self.lw = self.dw if left else 0.0
            self.rw = self.dw if right else 0.0
            self.thick = max(1.0, size * 0.06)
        else:
            self.lt = Text(m, left, font.with_(italic=False), color) if left else None
            self.rt = Text(m, right, font.with_(italic=False), color) if right else None
            self.lw = self.lt.w if self.lt else 0.0
            self.rw = self.rt.w if self.rt else 0.0
            self.a = max(body.a, self.lt.a if self.lt else 0, self.rt.a if self.rt else 0)
            self.d = max(body.d, self.lt.d if self.lt else 0, self.rt.d if self.rt else 0)
        self.inner = size * 0.04
        self.w = self.lw + self.inner + body.w + self.inner + self.rw

    def draw(self, c, x, y):
        if not self.stretch:
            if self.lt:
                self.lt.draw(c, x, y)
            self.body.draw(c, x + self.lw + self.inner, y)
            if self.rt:
                self.rt.draw(c, x + self.lw + self.inner * 2 + self.body.w, y)
            return
        top, bottom = y - self.a, y + self.d
        if self.left:
            _draw_delim(c, self.left, x, top, bottom, self.dw, self.color, self.thick, left=True)
        self.body.draw(c, x + self.lw + self.inner, y)
        if self.right:
            _draw_delim(c, self.right, x + self.lw + 2 * self.inner + self.body.w, top, bottom, self.dw, self.color,
                        self.thick, left=False)


def _draw_delim(c: Canvas, kind: str, x: float, top: float, bottom: float, w: float, color: str, thick: float, left: bool):
    h = bottom - top
    inner, outer = (x + w * 0.85, x + w * 0.2) if left else (x + w * 0.15, x + w * 0.8)
    if kind in "()":
        pts = []
        for i in range(25):
            t = i / 24
            yy = top + h * t
            bulge = math.sin(math.pi * t)
            xx = inner + (outer - inner) * bulge
            pts.append((xx, yy))
        c.polyline(pts, color, thick)
    elif kind in "[]":
        c.polyline([(inner, top), (outer, top), (outer, bottom), (inner, bottom)], color, thick)
    elif kind in "⌊⌋":
        c.polyline([(outer, top), (outer, bottom), (inner, bottom)], color, thick)
    elif kind in "⌈⌉":
        c.polyline([(inner, top), (outer, top), (outer, bottom)], color, thick)
    elif kind in "{}":
        mid = (top + bottom) / 2
        arm = (inner + outer) / 2
        hook = min(h * 0.08, w * 0.9)
        pts = [(inner, top)]
        pts += _quad((inner, top), (arm, top), (arm, top + hook))
        pts += [(arm, mid - hook)]
        pts += _quad((arm, mid - hook), (arm, mid), (outer, mid))
        pts += _quad((outer, mid), (arm, mid), (arm, mid + hook))
        pts += [(arm, bottom - hook)]
        pts += _quad((arm, bottom - hook), (arm, bottom), (inner, bottom))
        c.polyline(pts, color, thick)
    elif kind == "|":
        mx = (inner + outer) / 2
        c.line(mx, top, mx, bottom, color, thick)
    elif kind == "‖":
        mx = (inner + outer) / 2
        c.line(mx - w * 0.15, top, mx - w * 0.15, bottom, color, thick)
        c.line(mx + w * 0.15, top, mx + w * 0.15, bottom, color, thick)


def _quad(p0, p1, p2, n: int = 8):
    out = []
    for i in range(1, n + 1):
        t = i / n
        out.append(((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t * t * p2[0],
                    (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t * t * p2[1]))
    return out


class Matrix(Box):
    def __init__(self, m: Measurer, cells: Sequence[Sequence[Box]], font: FontSpec, color: str, axis: float,
                 left: str = "[", right: str = "]", align: str = "center"):
        self.cells = [list(r) for r in cells]
        size = font.size
        self.col_gap = size * 0.9
        self.row_gap = size * 0.28
        ncols = max((len(r) for r in self.cells), default=0)
        self.col_w = [max((r[j].w for r in self.cells if j < len(r)), default=0.0) for j in range(ncols)]
        self.row_a = [max((b.a for b in r), default=size * 0.7) for r in self.cells]
        self.row_d = [max((b.d for b in r), default=size * 0.25) for r in self.cells]
        self.align = align
        inner_w = sum(self.col_w) + self.col_gap * max(0, ncols - 1)
        inner_h = sum(a + d for a, d in zip(self.row_a, self.row_d)) + self.row_gap * max(0, len(self.cells) - 1)
        grid = Space(inner_w, inner_h / 2 + axis, inner_h / 2 - axis)
        self.grid = grid
        self.delim = Delimited(m, grid, left, right, font, color, axis)
        # force stretched delimiters for matrices with several rows
        if len(self.cells) > 1 and not self.delim.stretch:
            self.delim.stretch = True
            half = max(grid.a - axis, grid.d + axis) + size * 0.1
            self.delim.a = axis + half
            self.delim.d = half - axis
            self.delim.dw = size * 0.3
            self.delim.lw = self.delim.dw if left else 0.0
            self.delim.rw = self.delim.dw if right else 0.0
            self.delim.thick = max(1.0, size * 0.06)
            self.delim.w = self.delim.lw + 2 * self.delim.inner + grid.w + self.delim.rw
        self.w, self.a, self.d = self.delim.w, self.delim.a, self.delim.d

    def draw(self, c, x, y):
        self.delim.draw(c, x, y)
        gx = x + self.delim.lw + self.delim.inner
        top = y - self.grid.a
        for i, row in enumerate(self.cells):
            base = top + self.row_a[i]
            cx = gx
            for j, cell in enumerate(row):
                if self.align == "right":
                    dx = self.col_w[j] - cell.w
                elif self.align == "left":
                    dx = 0.0
                else:
                    dx = (self.col_w[j] - cell.w) / 2
                cell.draw(c, cx + dx, base)
                cx += self.col_w[j] + self.col_gap
            top += self.row_a[i] + self.row_d[i] + self.row_gap


class BigOperator(Box):
    """Σ / Π with limits above and below, followed by the body."""

    def __init__(self, symbol: Box, lower: Optional[Box], upper: Optional[Box], body: Box, size: float, axis: float):
        self.symbol, self.lower, self.upper, self.body = symbol, lower, upper, body
        gap = size * 0.08
        self.op_w = max(symbol.w, lower.w if lower else 0, upper.w if upper else 0)
        # centre the symbol on the math axis
        self.sym_shift = (symbol.a - symbol.d) / 2 - axis
        sym_a = symbol.a - self.sym_shift
        sym_d = symbol.d + self.sym_shift
        self.sym_a, self.sym_d = sym_a, sym_d
        self.gap = gap
        self.a = max(body.a, sym_a + (gap + upper.h if upper else 0))
        self.d = max(body.d, sym_d + (gap + lower.h if lower else 0))
        self.space = size * 0.18
        self.w = self.op_w + self.space + body.w

    def draw(self, c, x, y):
        self.symbol.draw(c, x + (self.op_w - self.symbol.w) / 2, y + self.sym_shift)
        if self.upper:
            self.upper.draw(c, x + (self.op_w - self.upper.w) / 2, y - self.sym_a - self.gap - self.upper.d)
        if self.lower:
            self.lower.draw(c, x + (self.op_w - self.lower.w) / 2, y + self.sym_d + self.gap + self.lower.a)
        self.body.draw(c, x + self.op_w + self.space, y)


class Brace(Box):
    """Left curly brace spanning a stack of lines (systems, piecewise, solve blocks)."""

    def __init__(self, content: Box, size: float, color: str, axis: float):
        self.content, self.color = content, color
        self.bw = size * 0.45
        self.thick = max(1.0, size * 0.06)
        self.pad = size * 0.2
        self.w = self.bw + self.pad + content.w
        self.a, self.d = content.a + size * 0.05, content.d + size * 0.05

    def draw(self, c, x, y):
        _draw_delim(c, "{", x, y - self.a, y + self.d, self.bw, self.color, self.thick, left=True)
        self.content.draw(c, x + self.bw + self.pad, y)


class ProgramBar(Box):
    """Mathcad style program: vertical bar with the statements to its right."""

    def __init__(self, lines: Sequence[Box], size: float, color: str):
        self.stack = VStack(lines, gap=size * 0.25)
        self.color = color
        self.bar_w = size * 0.5
        self.thick = max(1.2, size * 0.07)
        self.w = self.bar_w + self.stack.w
        self.a, self.d = self.stack.a + size * 0.05, self.stack.d + size * 0.05

    def draw(self, c, x, y):
        bx = x + self.bar_w * 0.35
        c.line(bx, y - self.a, bx, y + self.d, self.color, self.thick)
        self.stack.draw(c, x + self.bar_w, y)


class Framed(Box):
    def __init__(self, child: Box, fill: Optional[str], stroke: Optional[str] = None, pad: float = 3.0,
                 radius: float = 3.0, pad_y: Optional[float] = None):
        self.child, self.fill, self.stroke, self.pad, self.radius = child, fill, stroke, pad, radius
        self.pad_y = pad if pad_y is None else pad_y
        self.w = child.w + 2 * pad
        self.a = child.a + self.pad_y
        self.d = child.d + self.pad_y

    def draw(self, c, x, y):
        if self.fill or self.stroke:
            c.rect(x, y - self.a, self.w, self.a + self.d, self.fill, self.stroke, self.radius)
        self.child.draw(c, x + self.pad, y)


class Underlined(Box):
    def __init__(self, child: Box, color: str, wavy: bool = True):
        self.child, self.color, self.wavy = child, color, wavy
        self.w, self.a = child.w, child.a
        self.d = child.d + 4.0

    def draw(self, c, x, y):
        self.child.draw(c, x, y)
        yy = y + self.child.d + 2.0
        if self.wavy:
            c.wave(x, x + self.w, yy, self.color)
        else:
            c.line(x, yy, x + self.w, yy, self.color, 1.2)


class TableBox(Box):
    def __init__(self, m: Measurer, headers: Sequence[Box], rows: Sequence[Sequence[Box]], font: FontSpec,
                 colors: dict[str, str], title: Optional[Box] = None, highlight: set[tuple[int, int]] | None = None):
        self.headers, self.rows, self.title = list(headers), [list(r) for r in rows], title
        self.colors = colors
        self.highlight = highlight or set()
        size = font.size
        self.px, self.py = size * 0.45, size * 0.22
        ncols = max([len(self.headers)] + [len(r) for r in self.rows]) if (self.rows or self.headers) else 0
        all_rows = ([self.headers] if self.headers else []) + self.rows
        self.col_w = [max((r[j].w for r in all_rows if j < len(r)), default=0.0) + 2 * self.px for j in range(ncols)]
        self.row_h = [max((b.h for b in r), default=size) + 2 * self.py for r in all_rows]
        self.row_a = [max((b.a for b in r), default=size * 0.7) + self.py for r in all_rows]
        self.title_h = (title.h + size * 0.3) if title else 0.0
        grid_w = sum(self.col_w)
        self.w = max(grid_w, title.w if title else 0.0) + 1
        total_h = self.title_h + sum(self.row_h) + 1
        first = self.title.a if title else (self.row_a[0] if self.row_a else size * 0.7)
        self.a = first
        self.d = total_h - first

    def draw(self, c, x, y):
        top = y - self.a
        if self.title:
            self.title.draw(c, x, top + self.title.a)
            top += self.title_h
        all_rows = ([self.headers] if self.headers else []) + self.rows
        grid_w = sum(self.col_w)
        border = self.colors.get("table_border", "#999999")
        yy = top
        for i, row in enumerate(all_rows):
            is_header = bool(self.headers) and i == 0
            data_i = i - (1 if self.headers else 0)
            if is_header:
                c.rect(x, yy, grid_w, self.row_h[i], self.colors.get("table_header"), None, 0)
            cx = x
            for j in range(len(self.col_w)):
                if not is_header and (data_i, j) in self.highlight:
                    c.rect(cx, yy, self.col_w[j], self.row_h[i], self.colors.get("table_highlight"), None, 0)
                if j < len(row):
                    cell = row[j]
                    dx = (self.col_w[j] - cell.w) / 2 if is_header else self.col_w[j] - self.px - cell.w
                    cell.draw(c, cx + dx, yy + self.row_a[i])
                cx += self.col_w[j]
            yy += self.row_h[i]
        # grid lines
        yy = top
        for h in [0.0] + self.row_h:
            yy += h
            c.line(x, yy, x + grid_w, yy, border, 1.0)
        cx = x
        total = sum(self.row_h)
        for w in [0.0] + self.col_w:
            cx += w
            c.line(cx, top, cx, top + total, border, 1.0)
