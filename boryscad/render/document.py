"""Whole-document layout and export (PNG, SVG, PDF, HTML)."""
from __future__ import annotations

import base64
import html
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QMarginsF, QPointF, QRectF, QSize, QSizeF, Qt
from PySide6.QtGui import QColor, QImage, QPageLayout, QPageSize, QPainter, QPdfWriter
from PySide6.QtSvg import QSvgGenerator

from ..core.document import Document, ImageRegion, MathRegion, PageBreakRegion, PlotRegion, Region, TextRegion
from ..core.theme import Theme
from ..engine.formatting import NumberFormat
from ..engine.worksheet import DocumentResult, WorksheetEngine
from .qt_backend import qcolor
from .regions import RegionLayout, RenderContext, layout_region

FLOW_GAP = 14.0
ROW_GAP = 28.0


@dataclass
class Placed:
    region: Region
    layout: RegionLayout
    x: float
    y: float


class DocumentRenderer:
    def __init__(self, theme: Theme, engine: WorksheetEngine | None = None, scale: float = 2.0):
        self.theme = theme
        self.engine = engine or WorksheetEngine()
        self.scale = scale

    def context(self, doc: Document) -> RenderContext:
        fmt = NumberFormat(precision=doc.settings.precision, exp_threshold=doc.settings.exp_threshold)
        return RenderContext(self.theme, fmt, self.engine.registry, doc.settings.page_width, doc.settings.margin,
                             self.scale)

    def calculate(self, doc: Document) -> DocumentResult:
        return self.engine.run(doc)

    # ------------------------------------------------------------ layout
    def layout(self, doc: Document, results: DocumentResult, ctx: RenderContext | None = None) -> tuple[list[Placed], float, float]:
        ctx = ctx or self.context(doc)
        placed: list[Placed] = []
        margin = doc.settings.margin
        if doc.flow_layout:
            row_of = {rid: tuple(row) for row in doc.rows for rid in row}
            y = margin
            i = 0
            regions = doc.regions
            while i < len(regions):
                region = regions[i]
                row = row_of.get(region.id)
                group = [region]
                if row:
                    group = [r for r in regions[i:i + len(row)] if r.id in row]
                x = margin
                row_h = 0.0
                for r in group:
                    r.x, r.y = x, y
                    lay = layout_region(r, results.regions.get(r.id), ctx)
                    placed.append(Placed(r, lay, x, y))
                    x += lay.width + ROW_GAP
                    row_h = max(row_h, lay.height)
                gap = FLOW_GAP * (1.6 if isinstance(region, TextRegion) and region.style in ("h1", "h2") else 1.0)
                y += row_h + gap
                i += len(group)
            width = max([doc.settings.page_width] + [p.x + p.layout.width + margin for p in placed])
            return placed, width, y + margin
        for region in doc.regions:
            lay = layout_region(region, results.regions.get(region.id), ctx)
            placed.append(Placed(region, lay, region.x, region.y))
        width = max([doc.settings.page_width] + [p.x + p.layout.width + margin for p in placed])
        height = max([400.0] + [p.y + p.layout.height + margin for p in placed])
        return placed, width, height

    def paint(self, painter: QPainter, placed: list[Placed], width: float, height: float, background: bool = True,
              offset_y: float = 0.0) -> None:
        if background:
            painter.fillRect(QRectF(0, 0, width, height), qcolor(self.theme.color("page")))
        for p in placed:
            p.layout.paint(painter, p.x, p.y - offset_y)

    # ------------------------------------------------------------ export
    def export(self, doc: Document, path: str | Path, results: DocumentResult | None = None) -> Path:
        path = Path(path)
        results = results or self.calculate(doc)
        ext = path.suffix.lower()
        exporter: dict[str, Callable[[Document, Path, DocumentResult], None]] = {
            ".png": self.export_png, ".jpg": self.export_png, ".svg": self.export_svg, ".pdf": self.export_pdf,
            ".html": self.export_html, ".htm": self.export_html,
        }
        fn = exporter.get(ext)
        if fn is None:
            raise ValueError(f"unsupported export format: {ext} (use .png, .svg, .pdf or .html)")
        fn(doc, path, results)
        return path

    def export_png(self, doc: Document, path: Path, results: DocumentResult) -> None:
        placed, width, height = self.layout(doc, results)
        image = QImage(int(width * self.scale), int(height * self.scale), QImage.Format.Format_ARGB32_Premultiplied)
        image.setDevicePixelRatio(self.scale)
        image.fill(qcolor(self.theme.color("page")))
        painter = QPainter(image)
        self.paint(painter, placed, width, height)
        painter.end()
        if not image.save(str(path)):
            raise OSError(f"cannot write {path}")

    def export_svg(self, doc: Document, path: Path, results: DocumentResult) -> None:
        placed, width, height = self.layout(doc, results)
        gen = QSvgGenerator()
        gen.setFileName(str(path))
        gen.setSize(QSize(int(width), int(height)))
        gen.setViewBox(QRectF(0, 0, width, height))
        gen.setTitle(doc.title)
        gen.setDescription("Generated by Boryscad")
        painter = QPainter(gen)
        self.paint(painter, placed, width, height)
        painter.end()

    def export_pdf(self, doc: Document, path: Path, results: DocumentResult) -> None:
        placed, width, height = self.layout(doc, results)
        writer = QPdfWriter(str(path))
        writer.setTitle(doc.title)
        writer.setCreator("Boryscad")
        writer.setResolution(96)
        page_size = QPageSize(QPageSize.PageSizeId.A4)
        writer.setPageLayout(QPageLayout(page_size, QPageLayout.Orientation.Portrait, QMarginsF(0, 0, 0, 0)))
        page_px = page_size.sizePixels(96)
        scale = page_px.width() / width
        page_h = page_px.height() / scale
        painter = QPainter(writer)
        painter.scale(scale, scale)
        # paginate: never cut a region that fits on a page; taller regions are sliced
        breaks = sorted(p.y for p in placed if isinstance(p.region, PageBreakRegion))
        pages: list[tuple[float, float]] = []
        top = 0.0
        while top < height - 1:
            bottom = top + page_h
            forced = next((b for b in breaks if top < b < bottom), None)
            if forced is not None:
                bottom = forced
            else:
                for p in placed:
                    h = p.layout.height
                    if p.y > top + 1 and p.y < bottom < p.y + h and h < page_h * 0.9:
                        bottom = min(bottom, p.y - 8)
            if bottom <= top + 40:
                bottom = top + page_h
            pages.append((top, bottom))
            top = bottom + (18.0 if forced is not None else 0.0)
        for k, (top, bottom) in enumerate(pages):
            if k:
                writer.newPage()
            painter.save()
            painter.fillRect(QRectF(0, 0, width, page_h), qcolor(self.theme.color("page")))
            painter.setClipRect(QRectF(0, 0, width, bottom - top))
            for p in placed:
                if isinstance(p.region, PageBreakRegion):
                    continue
                if p.y < bottom and p.y + p.layout.height > top:
                    p.layout.paint(painter, p.x, p.y - top)
            painter.restore()
        painter.end()

    def export_html(self, doc: Document, path: Path, results: DocumentResult) -> None:
        ctx = self.context(doc)
        placed, width, height = self.layout(doc, results, ctx)
        t = self.theme
        parts = []
        for p in sorted(placed, key=lambda q: (round(q.y / 4), q.x)):
            region = p.region
            if isinstance(region, TextRegion):
                from ..core.markdown import inline, to_html
                result = results.regions.get(region.id)
                text = result.text if result is not None and result.text is not None else region.text
                if region.style in ("h1", "h2", "h3"):
                    body = f"<{region.style}>{inline(text)}</{region.style}>"
                elif region.format == "html":
                    body = text
                else:
                    body = to_html(text)
                parts.append(_positioned(p, f'<div class="text {region.style}">{body}</div>', doc.flow_layout))
            elif isinstance(region, PageBreakRegion):
                parts.append('<div class="pagebreak"></div>')
            else:
                svg = self._region_svg(p)
                parts.append(_positioned(p, f'<div class="region {region.kind}">{svg}</div>', doc.flow_layout))
        container = "flow" if doc.flow_layout else "absolute"
        page_style = "" if doc.flow_layout else f"height:{height:.0f}px;"
        document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{html.escape(doc.title)}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root {{ color-scheme: {'dark' if t.dark else 'light'}; }}
body {{ margin:0; background:{t.color('desk')}; font-family:'{t.font('text')}', system-ui, sans-serif;
       color:{t.color('text_body')}; }}
.page {{ position:relative; box-sizing:border-box; width:min({width:.0f}px, 100%); margin:32px auto; padding:{doc.settings.margin:.0f}px;
        background:{t.color('page')}; border-radius:14px; box-shadow:0 10px 40px {t.color('page_shadow')}; {page_style} }}
.page.flow .region, .page.flow .text {{ margin: 0 0 {FLOW_GAP:.0f}px 0; overflow-x:auto; }}
.page.absolute > div {{ position:absolute; }}
.text h1 {{ color:{t.color('heading')}; font-size:1.9em; margin:.2em 0; border-bottom:2px solid {t.color('accent')}; }}
.text h2 {{ color:{t.color('heading')}; font-size:1.45em; margin:.2em 0; }}
.text h3 {{ color:{t.color('heading')}; font-size:1.2em; margin:.2em 0; }}
.text p {{ margin:.3em 0; line-height:1.5; }}
.text code {{ background:{t.color('surface_alt')}; padding:1px 4px; border-radius:4px; }}
.pagebreak {{ border-top:1px dashed {t.color('text_muted')}; margin:18px 0; }}
svg {{ max-width:100%; height:auto; display:block; }}
.footer {{ text-align:center; color:{t.color('text_muted')}; font-size:12px; margin-bottom:24px; }}
</style></head>
<body><div class="page {container}">
{''.join(parts)}
</div><div class="footer">Boryscad · {html.escape(doc.title)}</div></body></html>"""
        path.write_text(document, encoding="utf-8")

    def _region_svg(self, p: Placed) -> str:
        w, h = max(1.0, p.layout.width), max(1.0, p.layout.height)
        buffer = QBuffer()
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        gen = QSvgGenerator()
        gen.setOutputDevice(buffer)
        gen.setSize(QSize(int(w + 1), int(h + 1)))
        gen.setViewBox(QRectF(0, 0, w + 1, h + 1))
        painter = QPainter(gen)
        p.layout.paint(painter, 0, 0)
        painter.end()
        svg = bytes(buffer.data()).decode("utf-8")
        start = svg.find("<svg")
        return svg[start:] if start >= 0 else svg


def _positioned(p: Placed, inner: str, flow: bool) -> str:
    if flow:
        return inner
    return f'<div style="left:{p.x:.0f}px;top:{p.y:.0f}px">{inner}</div>'


def render_formula(source: str, theme: Theme, path: str | Path, evaluate: bool = True, scale: float = 3.0) -> Path:
    """Render a single formula (optionally evaluated) to an image/SVG/PDF/HTML file."""
    doc = Document(flow_layout=True, regions=[MathRegion(source=source)])
    doc.settings.margin = 12
    doc.settings.page_width = 10
    renderer = DocumentRenderer(theme, scale=scale)
    results = renderer.calculate(doc) if evaluate else DocumentResult({}, None, 0.0)  # type: ignore[arg-type]
    return renderer.export(doc, path, results)
