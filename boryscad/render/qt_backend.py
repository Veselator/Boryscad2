"""Qt implementation of the box layout Measurer and Canvas."""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Optional, Sequence

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QFontMetricsF, QPainter, QPainterPath, QPen

from .boxes import FontSpec


def qcolor(value: Optional[str]) -> QColor:
    """Parse ``#RGB``, ``#RRGGBB`` and CSS style ``#RRGGBBAA`` colors."""
    if not value:
        return QColor(0, 0, 0, 0)
    v = value.strip()
    if v.startswith("#") and len(v) == 9:
        r, g, b, a = (int(v[i:i + 2], 16) for i in (1, 3, 5, 7))
        return QColor(r, g, b, a)
    return QColor(v)


def css_rgba(value: str) -> str:
    c = qcolor(value)
    return f"rgba({c.red()}, {c.green()}, {c.blue()}, {c.alpha()})"


@lru_cache(maxsize=512)
def qfont(spec: FontSpec) -> QFont:
    f = QFont(spec.family)
    f.setPixelSize(max(1, int(round(spec.size))))
    f.setItalic(spec.italic)
    f.setBold(spec.bold)
    f.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
    f.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    return f


class QtMeasurer:
    def __init__(self):
        self._metrics: dict[FontSpec, QFontMetricsF] = {}

    def _fm(self, font: FontSpec) -> QFontMetricsF:
        fm = self._metrics.get(font)
        if fm is None:
            fm = QFontMetricsF(qfont(font))
            self._metrics[font] = fm
        return fm

    def text(self, text: str, font: FontSpec) -> tuple[float, float, float]:
        fm = self._fm(font)
        width = fm.horizontalAdvance(text)
        # Cambria Math has very tall ascent/descent; use tighter values based on the glyph box
        rect = fm.tightBoundingRect(text) if text.strip() else QRectF(0, -fm.ascent() * 0.7, 0, 0)
        ascent = max(-rect.top(), font.size * 0.72)
        descent = max(rect.bottom(), font.size * 0.22)
        return width, ascent, descent


class QtCanvas:
    def __init__(self, painter: QPainter):
        self.p = painter
        self.p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.p.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

    def text(self, x, y, text, font, color):
        self.p.setFont(qfont(font))
        self.p.setPen(qcolor(color))
        self.p.drawText(QPointF(x, y), text)

    def line(self, x1, y1, x2, y2, color, width=1.0):
        pen = QPen(qcolor(color), width)
        pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        self.p.setPen(pen)
        self.p.drawLine(QPointF(x1, y1), QPointF(x2, y2))

    def rect(self, x, y, w, h, fill, stroke=None, radius=0.0, width=1.0):
        self.p.setPen(QPen(qcolor(stroke), width) if stroke else Qt.PenStyle.NoPen)
        self.p.setBrush(QBrush(qcolor(fill)) if fill else Qt.BrushStyle.NoBrush)
        if radius:
            self.p.drawRoundedRect(QRectF(x, y, w, h), radius, radius)
        else:
            self.p.drawRect(QRectF(x, y, w, h))
        self.p.setBrush(Qt.BrushStyle.NoBrush)

    def polyline(self, points: Sequence[tuple[float, float]], color, width=1.0, closed=False, fill=None):
        if not points:
            return
        path = QPainterPath(QPointF(*points[0]))
        for pt in points[1:]:
            path.lineTo(QPointF(*pt))
        if closed:
            path.closeSubpath()
        pen = QPen(qcolor(color), width)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        self.p.setPen(pen)
        self.p.setBrush(QBrush(qcolor(fill)) if fill else Qt.BrushStyle.NoBrush)
        self.p.drawPath(path)
        self.p.setBrush(Qt.BrushStyle.NoBrush)

    def wave(self, x1, x2, y, color):
        pts = []
        step = 3.0
        n = max(2, int((x2 - x1) / step))
        for i in range(n + 1):
            xx = x1 + (x2 - x1) * i / n
            pts.append((xx, y + (1.2 if i % 2 else -1.2)))
        self.polyline(pts, color, 1.0)
