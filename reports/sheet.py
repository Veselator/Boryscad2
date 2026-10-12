"""Bridge between a Boryscad worksheet and a Word report.

Runs an ``.bmath`` example once and exposes its values, result tables, plots and
formula listings (rendered as pictures with the same engine the editor uses).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from boryscad.core.document import Document, MathRegion, PlotRegion, TextRegion
from boryscad.core.io import load_document
from boryscad.core.theme import load_theme
from boryscad.engine import WorksheetEngine
from boryscad.engine.formatting import NumberFormat, format_number, format_value
from boryscad.engine.values import DStack, DTable, DText, Record, Table
from boryscad.render import ensure_qt_app
from boryscad.render.document import DocumentRenderer
from boryscad.render.plot_renderer import render_png

from i18n import uk

EXAMPLES = Path(__file__).resolve().parents[1] / "boryscad" / "examples"


def number_format(precision: int = 4) -> NumberFormat:
    """Ukrainian reports use a decimal comma."""
    return NumberFormat(precision=precision, decimal_sep=",")


def fmt(value: Any, precision: int = 4) -> str:
    """Number or vector as plain text for the report body."""
    f = number_format(precision)
    if isinstance(value, (list, tuple, np.ndarray)):
        items = np.asarray(value).reshape(-1)
        return "(" + "; ".join(format_number(v, f) for v in items) + ")"
    return format_value(value, f)


class Sheet:
    def __init__(self, name: str, theme: str = "Paper", scale: float = 2.5):
        ensure_qt_app()
        self.path = EXAMPLES / name
        self.document = load_document(self.path)
        self.engine = WorksheetEngine()
        self.results = self.engine.run(self.document)
        self.env = self.results.env
        self.theme = load_theme(theme)
        self.scale = scale
        errors = self.results.errors()
        if errors:
            raise RuntimeError("{}: {}".format(name, errors))

    # ------------------------------------------------------------- values
    def value(self, name: str) -> Any:
        return self.env.lookup(name)

    def num(self, name: str, precision: int = 4) -> str:
        return fmt(self.value(name), precision)

    def record(self, name: str) -> Record:
        value = self.value(name)
        if not isinstance(value, Record):
            raise TypeError("{} is not a record".format(name))
        return value

    def field(self, record: str, field: str) -> Any:
        return self.record(record).get(field)

    def call(self, name: str, *args: Any) -> Any:
        """Call a function defined in the worksheet, e.g. sheet.call("f", 1, 2)."""
        from boryscad.engine.evaluator import Evaluator
        evaluator = Evaluator(self.engine.registry)
        return evaluator.call_value(self.value(name), list(args))

    # ------------------------------------------------------------- tables
    @staticmethod
    def _tables_of(node) -> list[DTable]:
        if isinstance(node, DTable):
            return [node]
        if isinstance(node, DStack):
            out: list[DTable] = []
            for item in node.items:
                out.extend(Sheet._tables_of(item))
            return out
        return []

    def tables(self, record_name: str) -> list[DTable]:
        return self._tables_of(self.record(record_name).display())

    def title(self, record_name: str, index: int) -> str:
        return uk(self.tables(record_name)[index].title)

    def notes(self, record_name: str) -> list[str]:
        node = self.record(record_name).display()
        items = node.items if isinstance(node, DStack) else [node]
        return [i.text for i in items if isinstance(i, DText)]

    @staticmethod
    def rows(table: DTable, precision: int = 4) -> tuple[list[str], list[list[str]]]:
        f = number_format(precision)
        def cell(v):
            if isinstance(v, str):
                return uk(v)
            if v is None:
                return ""
            return format_value(v, f)
        return [uk(h) for h in table.headers], [[cell(v) for v in row] for row in table.rows]

    @staticmethod
    def matrix_rows(matrix, headers: Sequence[str] | None = None, row_names: Sequence[str] | None = None,
                    precision: int = 4) -> tuple[list[str], list[list[str]]]:
        data = np.asarray(matrix)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        f = number_format(precision)
        head = list(headers) if headers is not None else [str(j + 1) for j in range(data.shape[1])]
        body = []
        for i, row in enumerate(data):
            cells = [format_value(v, f) for v in row]
            if row_names is not None:
                cells = [str(row_names[i])] + cells
            body.append(cells)
        if row_names is not None:
            head = [""] + head
        return head, body

    # -------------------------------------------------------------- images
    def plot(self, out: Path, index: int = 0, width: float | None = None, height: float | None = None) -> Path:
        plots = [r for r in self.document.regions if isinstance(r, PlotRegion)]
        region = plots[index]
        data = self.results.regions[region.id].plot
        if width:
            region.width = width
        if height:
            region.height = height
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(render_png(region, data, self.theme, region.width, region.height, self.scale))
        return out

    def listing(self, out: Path, contains: Iterable[str] | None = None, skip: Iterable[str] = (),
                page_width: float = 520.0) -> Path:
        """Picture of the calculation listing: the math regions of the worksheet."""
        regions = [r for r in self.document.ordered() if isinstance(r, MathRegion)]
        if contains is not None:
            keys = list(contains)
            regions = [r for r in regions if any(k in r.source for k in keys)]
        if skip:
            regions = [r for r in regions if not any(k in r.source for k in skip)]
        sub = Document(flow_layout=True, regions=regions)
        sub.settings.page_width = page_width
        sub.settings.margin = 10
        sub.settings.precision = self.document.settings.precision
        renderer = DocumentRenderer(self.theme, self.engine, scale=self.scale)
        out.parent.mkdir(parents=True, exist_ok=True)
        renderer.export(sub, out, self.results)
        return out
