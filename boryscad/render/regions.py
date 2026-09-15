"""Region views: measure and paint each region kind with QPainter.

Used both by the GUI worksheet items and by exporters, so what you see is
exactly what gets exported.  New region kinds register a view in
:data:`REGION_VIEWS` (Open/Closed).
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

from PySide6.QtCore import QByteArray, QPointF, QRectF, QSizeF, Qt
from PySide6.QtGui import (QAbstractTextDocumentLayout, QColor, QFont, QImage, QPainter, QPalette, QPen,
                           QTextDocument)

from ..core.document import ImageRegion, MathRegion, PageBreakRegion, PlotRegion, Region, TextRegion
from ..core.markdown import to_html
from ..core.theme import Theme
from ..engine.formatting import NumberFormat
from ..engine.worksheet import RegionResult
from ..language import ParseError, parse_cached
from .boxes import Box
from .builder import FormulaBuilder, MathStyle
from .qt_backend import QtCanvas, QtMeasurer, css_rgba, qcolor

PAD = 6.0


@dataclass
class RenderContext:
    theme: Theme
    fmt: NumberFormat = field(default_factory=NumberFormat)
    registry: Any = None
    page_width: float = 860.0
    margin: float = 48.0
    scale: float = 1.0           # device pixel ratio for raster content (plots, images)
    measurer: QtMeasurer = field(default_factory=QtMeasurer)
    _builder: Optional[FormulaBuilder] = None
    plot_cache: dict = field(default_factory=dict)

    @property
    def builder(self) -> FormulaBuilder:
        if self._builder is None:
            self._builder = FormulaBuilder(self.measurer, MathStyle.from_theme(self.theme), self.fmt, self.registry)
        return self._builder


class RegionLayout:
    width: float = 0.0
    height: float = 0.0

    def paint(self, painter: QPainter, x: float, y: float) -> None:  # pragma: no cover
        raise NotImplementedError


class RegionView(Protocol):
    def layout(self, region: Region, result: Optional[RegionResult], ctx: RenderContext) -> RegionLayout: ...


# --------------------------------------------------------------------- math
class BoxLayout(RegionLayout):
    def __init__(self, box: Box, pad: float = PAD):
        self.box, self.pad = box, pad
        self.width = box.w + 2 * pad
        self.height = box.h + 2 * pad

    def paint(self, painter, x, y):
        self.box.draw(QtCanvas(painter), x + self.pad, y + self.pad + self.box.a)


class MathView:
    def layout(self, region: MathRegion, result, ctx):  # type: ignore[override]
        b = ctx.builder
        if not region.source.strip():
            return BoxLayout(b.text("▮", 0, "text_muted"))
        if result is not None and result.parse_error is not None:
            return BoxLayout(b.source_with_error(region.source, str(result.parse_error)))
        try:
            statements = parse_cached(region.source)
        except ParseError as exc:
            return BoxLayout(b.source_with_error(region.source, str(exc)))
        results = result.statements if result is not None else None
        return BoxLayout(b.statements(statements, results))


# --------------------------------------------------------------------- text
TEXT_STYLES = {
    "h1": ("heading", 1.9, QFont.Weight.Bold),
    "h2": ("heading", 1.45, QFont.Weight.DemiBold),
    "h3": ("heading", 1.2, QFont.Weight.DemiBold),
    "body": ("text_body", 1.0, QFont.Weight.Normal),
    "note": ("text_muted", 0.9, QFont.Weight.Normal),
}


def make_text_document(region: TextRegion, text: str, ctx: RenderContext, width: float) -> QTextDocument:
    color_key, factor, weight = TEXT_STYLES.get(region.style, TEXT_STYLES["body"])
    doc = QTextDocument()
    font = QFont(ctx.theme.font("text", "Segoe UI"))
    font.setPixelSize(int(round(float(ctx.theme.font("text_size", 11)) * 1.33 * factor)))
    font.setWeight(weight)
    doc.setDefaultFont(font)
    doc.setDocumentMargin(2)
    color = ctx.theme.color(color_key)
    accent = ctx.theme.color("accent")
    doc.setDefaultStyleSheet(
        f"body, p, li {{ color: {css_rgba(color)}; }} h1, h2, h3 {{ color: {css_rgba(ctx.theme.color('heading'))}; }}"
        f"a {{ color: {css_rgba(accent)}; }} code {{ font-family: '{ctx.theme.font('mono', 'Consolas')}'; "
        f"background-color: {css_rgba(ctx.theme.color('surface_alt'))}; }}")
    if region.format == "html":
        doc.setHtml(text)
    elif region.style in ("h1", "h2", "h3"):
        from ..core.markdown import inline
        doc.setHtml(f"<p>{inline(text)}</p>")
    else:
        doc.setHtml(to_html(text) or "<p></p>")
    if width > 0:
        doc.setTextWidth(width)
    return doc


class TextLayout(RegionLayout):
    def __init__(self, doc: QTextDocument, width: float, color: str, underline: Optional[str] = None):
        self.doc = doc
        size = doc.size()
        self.width = max(width if width > 0 else size.width(), 20.0)
        self.height = size.height() + (6.0 if underline else 0.0)
        self.color = color
        self.underline = underline

    def paint(self, painter, x, y):
        painter.save()
        painter.translate(x, y)
        ctx = QAbstractTextDocumentLayout.PaintContext()
        palette = ctx.palette
        palette.setColor(QPalette.ColorRole.Text, qcolor(self.color))
        ctx.palette = palette
        self.doc.documentLayout().draw(painter, ctx)
        if self.underline:
            pen = QPen(qcolor(self.underline), 2.0)
            painter.setPen(pen)
            h = self.doc.size().height() + 3
            painter.drawLine(QPointF(2, h), QPointF(min(self.width, self.doc.idealWidth() + 40), h))
        painter.restore()


class TextView:
    def layout(self, region: TextRegion, result, ctx):  # type: ignore[override]
        text = result.text if (result is not None and result.text is not None) else region.text
        width = region.width if region.width > 0 else max(120.0, ctx.page_width - region.x - ctx.margin)
        doc = make_text_document(region, text or " ", ctx, width)
        if region.style in ("h1", "h2", "h3") or region.width <= 0:
            doc.setTextWidth(width)
        color_key = TEXT_STYLES.get(region.style, TEXT_STYLES["body"])[0]
        underline = ctx.theme.color("accent") if region.style == "h1" else None
        layout = TextLayout(doc, width if region.style != "body" or region.width > 0 else min(width, doc.idealWidth() + 8),
                            ctx.theme.color(color_key), underline)
        return layout


# -------------------------------------------------------------------- image
class ImageLayout(RegionLayout):
    def __init__(self, image: QImage, width: float, height: float, caption: Optional[QTextDocument]):
        self.image, self.caption = image, caption
        self.img_w, self.img_h = width, height
        self.width = width
        self.height = height + (caption.size().height() + 4 if caption else 0)

    def paint(self, painter, x, y):
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        if self.image.isNull():
            painter.setPen(QPen(QColor("#d6336c"), 1.5, Qt.PenStyle.DashLine))
            painter.drawRect(QRectF(x, y, self.img_w, self.img_h))
            painter.drawText(QRectF(x, y, self.img_w, self.img_h), Qt.AlignmentFlag.AlignCenter, "image not available")
        else:
            painter.drawImage(QRectF(x, y, self.img_w, self.img_h), self.image)
        if self.caption:
            painter.translate(x + (self.img_w - self.caption.size().width()) / 2, y + self.img_h + 4)
            self.caption.drawContents(painter)
        painter.restore()


_image_cache: dict[int, QImage] = {}


def decode_image(region: ImageRegion) -> QImage:
    key = hash(region.data)
    img = _image_cache.get(key)
    if img is None:
        img = QImage()
        if region.data:
            img.loadFromData(QByteArray(base64.b64decode(region.data)))
        if len(_image_cache) > 64:
            _image_cache.clear()
        _image_cache[key] = img
    return img


class ImageView:
    def layout(self, region: ImageRegion, result, ctx):  # type: ignore[override]
        caption = None
        if region.caption:
            caption = make_text_document(TextRegion(text=region.caption, style="note"), region.caption, ctx, region.width)
            caption.setTextWidth(-1)
        return ImageLayout(decode_image(region), region.width, region.height, caption)


# --------------------------------------------------------------------- plot
class PlotLayout(RegionLayout):
    def __init__(self, image: QImage, width: float, height: float, errors: list[str], ctx: RenderContext):
        self.image = image
        self.plot_w, self.plot_h = width, height
        self.errors = errors
        self.ctx = ctx
        self.err_h = 16.0 * min(len(errors), 4)
        self.width, self.height = width, height + self.err_h

    def paint(self, painter, x, y):
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        radius = float(self.ctx.theme.radius)
        painter.setPen(QPen(qcolor(self.ctx.theme.color("border")), 1.0))
        painter.setBrush(qcolor(self.ctx.theme.color("plot_bg")))
        painter.drawRoundedRect(QRectF(x - 0.5, y - 0.5, self.plot_w + 1, self.plot_h + 1), radius, radius)
        if not self.image.isNull():
            painter.drawImage(QRectF(x, y, self.plot_w, self.plot_h), self.image)
        if self.errors:
            font = QFont(self.ctx.theme.font("text", "Segoe UI"))
            font.setPixelSize(12)
            painter.setFont(font)
            painter.setPen(qcolor(self.ctx.theme.color("math_error")))
            for k, err in enumerate(self.errors[:4]):
                painter.drawText(QPointF(x + 2, y + self.plot_h + 13 + 16 * k), f"⚠ {err}"[:160])
        painter.restore()


def _plot_key(region: PlotRegion, theme: Theme, scale: float) -> tuple:
    return (region.width, region.height, theme.name, tuple(sorted(theme.colors.items())), scale, region.title,
            region.xlabel, region.ylabel, region.zlabel, region.grid, region.legend, region.equal, region.mode,
            region.xscale, region.yscale)


def prerender_plot(region: PlotRegion, data, theme: Theme, scale: float) -> QImage:
    """Render (or fetch from cache) the plot image; safe to call from a worker thread."""
    from .plot_renderer import render_rgba
    key = _plot_key(region, theme, scale)
    cache = data.__dict__.setdefault("_qimage_cache", {})
    image = cache.get(key)
    if image is None:
        rgba, w, h = render_rgba(region, data, theme, region.width, region.height, scale)
        image = QImage(rgba, w, h, QImage.Format.Format_RGBA8888).copy()
        image.setDevicePixelRatio(scale)
        cache.clear()
        cache[key] = image
    return image


class PlotView:
    def layout(self, region: PlotRegion, result, ctx):  # type: ignore[override]
        from ..library.plotting import PlotData
        data = result.plot if (result is not None and result.plot is not None) else PlotData()
        image = prerender_plot(region, data, ctx.theme, ctx.scale)
        return PlotLayout(image, region.width, region.height, list(data.errors), ctx)


# ---------------------------------------------------------------- pagebreak
class PageBreakLayout(RegionLayout):
    def __init__(self, ctx: RenderContext, width: float):
        self.ctx = ctx
        self.width, self.height = width, 18.0

    def paint(self, painter, x, y):
        painter.save()
        pen = QPen(qcolor(self.ctx.theme.color("text_muted")), 1.0, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawLine(QPointF(x, y + 9), QPointF(x + self.width, y + 9))
        painter.restore()


class PageBreakView:
    def layout(self, region, result, ctx):
        return PageBreakLayout(ctx, ctx.page_width - region.x - ctx.margin)


REGION_VIEWS: dict[str, RegionView] = {
    "math": MathView(),
    "text": TextView(),
    "image": ImageView(),
    "plot": PlotView(),
    "pagebreak": PageBreakView(),
}


def layout_region(region: Region, result: Optional[RegionResult], ctx: RenderContext) -> RegionLayout:
    view = REGION_VIEWS.get(region.kind)
    if view is None:
        return BoxLayout(ctx.builder.text(f"unsupported region '{region.kind}'", 0, "math_error"))
    return view.layout(region, result, ctx)
