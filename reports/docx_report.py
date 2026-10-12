"""Report builder following Додаток Б/Г/Д of the lab manual.

The report is collected as a list of blocks (title page, headings, paragraphs,
numbered formulas, figures, tables, conclusions) and then written by two
back ends: ``.docx`` (python-docx) and ``.pdf`` (Qt, no Word required).

A4, margins 20 mm, Times New Roman 14 pt, line spacing 1.5, first line indent
1.25 cm, page numbers in the top right corner (not printed on the title page),
continuous numbering of formulas, figures and tables inside one report.
"""
from __future__ import annotations

import html as html_mod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from docx import Document as DocxDocument
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

FONT = "Times New Roman"
SIZE = Pt(14)
SMALL = Pt(11)
TINY = Pt(9)

PAGE_W_CM, PAGE_H_CM, MARGIN_CM = 21.0, 29.7, 2.0
TEXT_W_CM = PAGE_W_CM - 2 * MARGIN_CM


@dataclass
class Block:
    kind: str
    data: dict[str, Any] = field(default_factory=dict)


def _field(paragraph, instruction: str) -> None:
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = instruction
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.append(begin)
    run._r.append(instr)
    run._r.append(end)


class Report:
    """Collects the content of one lab report; :meth:`save` writes .docx and .pdf."""

    pdf_default = True

    def __init__(self, number: int, topic: str, goal: str, meta: dict):
        self.number, self.topic, self.goal, self.meta = number, topic, goal, meta
        self.formula_no = 0
        self.figure_no = 0
        self.table_no = 0
        self.blocks: list[Block] = []

    # ----------------------------------------------------------- content
    def para(self, text: str, bold_prefix: str | None = None) -> None:
        self.blocks.append(Block("para", {"text": text, "bold_prefix": bold_prefix}))

    def heading(self, text: str) -> None:
        self.blocks.append(Block("heading", {"text": text}))

    def note(self, text: str) -> None:
        """Explanation of the symbols right under a formula (starts with «де»)."""
        self.blocks.append(Block("note", {"text": text}))

    def formula(self, text: str, numbered: bool = True) -> int:
        number = 0
        if numbered:
            self.formula_no += 1
            number = self.formula_no
        self.blocks.append(Block("formula", {"text": text, "number": number}))
        return number

    def figure(self, image: str | Path, caption: str, width_cm: float = 15.0) -> int:
        self.figure_no += 1
        self.blocks.append(Block("figure", {"image": Path(image), "caption": caption,
                                            "width_cm": width_cm, "number": self.figure_no}))
        return self.figure_no

    def table(self, headers: Sequence[str], rows: Iterable[Sequence], caption: str,
              size=SMALL, widths: Sequence[float] | None = None) -> int:
        self.table_no += 1
        self.blocks.append(Block("table", {"headers": [str(h) for h in headers],
                                           "rows": [[("" if v is None else str(v)) for v in r] for r in rows],
                                           "caption": caption, "size": size, "widths": widths,
                                           "number": self.table_no}))
        return self.table_no

    def conclusions(self, lines: Sequence[str]) -> None:
        self.blocks.append(Block("conclusions", {"lines": list(lines)}))

    # -------------------------------------------------------------- save
    def save(self, path: str | Path, pdf: bool | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        DocxWriter(self).write(path)
        if self.pdf_default if pdf is None else pdf:
            PdfWriter(self).write(path.with_suffix(".pdf"))
        return path


# --------------------------------------------------------------------- docx
class DocxWriter:
    def __init__(self, report: Report):
        self.r = report
        self.doc = DocxDocument()

    def write(self, path: Path) -> Path:
        self._setup()
        self._title_page()
        for block in self.r.blocks:
            getattr(self, "_" + block.kind)(block.data)
        self.doc.save(str(path))
        return path

    # ---- layout helpers
    def _setup(self) -> None:
        style = self.doc.styles["Normal"]
        style.font.name = FONT
        style.font.size = SIZE
        style.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
        pf = style.paragraph_format
        pf.line_spacing = 1.5
        pf.first_line_indent = Cm(1.25)
        pf.space_after = Pt(0)
        section = self.doc.sections[0]
        section.page_width, section.page_height = Cm(PAGE_W_CM), Cm(PAGE_H_CM)
        for side in ("left", "right", "top", "bottom"):
            setattr(section, side + "_margin", Cm(MARGIN_CM))
        section.different_first_page_header_footer = True
        header = section.header.paragraphs[0]
        header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        header.paragraph_format.first_line_indent = Cm(0)
        _field(header, "PAGE")
        for run in header.runs:
            run.font.name, run.font.size = FONT, SIZE

    def _center(self, text: str = "", bold: bool = False, size=SIZE, spacing: float = 1.0):
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.first_line_indent = Cm(0)
        p.paragraph_format.line_spacing = spacing
        p.paragraph_format.space_after = Pt(0)
        run = p.add_run(text)
        run.bold = bold
        run.font.name, run.font.size = FONT, size
        return p

    def _title_page(self) -> None:
        r = self.r
        for text, align, gap, spacing in title_lines(r.number, r.meta):
            p = self.doc.add_paragraph()
            pf = p.paragraph_format
            pf.first_line_indent = Cm(0)
            pf.space_before = Pt(gap)
            pf.space_after = Pt(0)
            pf.line_spacing = spacing
            if align == "c":
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                pf.left_indent = Cm(SIGN_INDENT_CM)
            run = p.add_run(text)
            run.font.name, run.font.size = FONT, SIZE
        self.doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
        self._center("Лабораторна робота № {}".format(r.number), bold=True)
        self._center()
        self._para_text("Тема: " + r.topic, "Тема:")
        self._para_text("Мета: " + r.goal, "Мета:")
        self._center()
        self._center("Хід роботи", bold=True)

    def _para_text(self, text: str, bold_prefix: str | None = None):
        p = self.doc.add_paragraph()
        if bold_prefix and text.startswith(bold_prefix):
            run = p.add_run(bold_prefix)
            run.bold = True
            run.font.name, run.font.size = FONT, SIZE
            text = text[len(bold_prefix):]
        run = p.add_run(text)
        run.font.name, run.font.size = FONT, SIZE
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        return p

    # ---- block writers
    def _para(self, d: dict) -> None:
        self._para_text(d["text"], d["bold_prefix"])

    def _heading(self, d: dict) -> None:
        p = self.doc.add_paragraph()
        run = p.add_run(d["text"])
        run.bold = True
        run.font.name, run.font.size = FONT, SIZE
        p.paragraph_format.space_before = Pt(8)
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    def _note(self, d: dict) -> None:
        p = self.doc.add_paragraph()
        p.paragraph_format.first_line_indent = Cm(0)
        run = p.add_run(d["text"])
        run.font.name, run.font.size = FONT, SIZE
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    def _formula(self, d: dict) -> None:
        p = self.doc.add_paragraph()
        p.paragraph_format.first_line_indent = Cm(0)
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.line_spacing = 1.0
        tabs = p.paragraph_format.tab_stops
        tabs.add_tab_stop(Cm(8.5), WD_TAB_ALIGNMENT.CENTER)
        tabs.add_tab_stop(Cm(17), WD_TAB_ALIGNMENT.RIGHT)
        text = "\t{}\t({})".format(d["text"], d["number"]) if d["number"] else "\t" + d["text"]
        run = p.add_run(text)
        run.font.name, run.font.size = FONT, SIZE

    def _figure(self, d: dict) -> None:
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.first_line_indent = Cm(0)
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.line_spacing = 1.0
        p.add_run().add_picture(str(d["image"]), width=Cm(d["width_cm"]))
        self._center("Рисунок {} – {}".format(d["number"], d["caption"]), spacing=1.0)
        self._center(spacing=1.0)

    def _table(self, d: dict) -> None:
        p = self.doc.add_paragraph()
        p.paragraph_format.first_line_indent = Cm(0)
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.line_spacing = 1.0
        run = p.add_run("Таблиця {} – {}".format(d["number"], d["caption"]))
        run.font.name, run.font.size = FONT, SIZE
        headers, rows, size = d["headers"], d["rows"], d["size"]
        ncols = max(len(headers), max((len(r) for r in rows), default=1))
        table = self.doc.add_table(rows=1 + len(rows), cols=ncols)
        table.style = "Table Grid"
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        for j in range(ncols):
            self._cell(table.cell(0, j), headers[j] if j < len(headers) else "", bold=True, size=size)
        for i, row in enumerate(rows, start=1):
            for j in range(ncols):
                self._cell(table.cell(i, j), row[j] if j < len(row) else "", size=size)
        if d["widths"]:
            for j, w in enumerate(d["widths"]):
                for row in table.rows:
                    if j < len(row.cells):
                        row.cells[j].width = Cm(w)
        tail = self.doc.add_paragraph()
        tail.paragraph_format.space_after = Pt(0)
        tail.paragraph_format.line_spacing = 1.0

    @staticmethod
    def _cell(cell, text: str, bold: bool = False, size=SMALL) -> None:
        cell.text = ""
        p = cell.paragraphs[0]
        p.paragraph_format.first_line_indent = Cm(0)
        p.paragraph_format.line_spacing = 1.0
        p.paragraph_format.space_after = Pt(0)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(text)
        run.bold = bold
        run.font.name, run.font.size = FONT, size

    def _conclusions(self, d: dict) -> None:
        self._center()
        lines = d["lines"]
        p = self.doc.add_paragraph()
        run = p.add_run("Висновки: ")
        run.bold = True
        run.font.name, run.font.size = FONT, SIZE
        run = p.add_run(lines[0])
        run.font.name, run.font.size = FONT, SIZE
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        for line in lines[1:]:
            self._para_text(line)


# ---------------------------------------------------------------------- pdf
# Title page copied from the student's Example.docx: plain Times New Roman 14 pt,
# single spacing, blocks separated by "space before" (pt), signature block indented 11.9 cm.
#            text key               align  gap pt  line spacing
TITLE_SPEC = [("МІНІСТЕРСТВО ОСВІТИ І НАУКИ УКРАЇНИ", "c", 0.0, 1.0),
              ("КРЕМЕНЧУЦЬКИЙ НАЦІОНАЛЬНИЙ УНІВЕРСИТЕТ ІМЕНІ МИХАЙЛА ОСТРОГРАДСЬКОГО", "c", 13.9, 1.43),
              ("НАВЧАЛЬНО-НАУКОВИЙ ІНСТИТУТ ЕЛЕКТРИЧНОЇ ІНЖЕНЕРІЇ ТА ІНФОРМАЦІЙНИХ ТЕХНОЛОГІЙ", "c", 7.7, 1.43),
              ("КАФЕДРА АВТОМАТИЗАЦІЇ ТА ІНФОРМАЦІЙНИХ СИСТЕМ", "c", 31.9, 1.0),
              ("ОСВІТНІЙ КОМПОНЕНТ", "c", 31.9, 1.0),
              ("«ДОСЛІДЖЕННЯ ОПЕРАЦІЙ»", "c", 21.8, 1.0),
              ("ЗВІТ", "c", 62.1, 1.0),
              ("З ЛАБОРАТОРНОЇ РОБОТИ №{number}", "c", 7.7, 1.0),
              ("Виконав:", "s", 80.2, 1.0),
              ("здобувач освіти", "s", 7.7, 1.0),
              ("{group}", "s", 7.7, 1.0),
              ("{student}", "s", 7.7, 1.0),
              ("Перевірив:", "s", 32.0, 1.0),
              ("доцент кафедри АІС", "s", 7.7, 1.0),
              ("{teacher}", "s", 7.7, 1.0),
              ("{city} {year}", "c", 56.1, 1.0)]

SIGN_INDENT_CM = 11.9


def title_lines(number: int, meta: dict):
    """(text, align, gap in pt, line spacing) for every line of the title page."""
    values = dict(meta, number=number)
    return [(text.format(**values), align, gap, spacing) for text, align, gap, spacing in TITLE_SPEC]


def _esc(text: str) -> str:
    return html_mod.escape(str(text), quote=False)


class PdfWriter:
    """Prints the same blocks to PDF with Qt (selectable text, no Word needed)."""

    DPI = 96

    def __init__(self, report: Report):
        self.r = report

    def html(self) -> str:
        r, m = self.r, self.r.meta
        px = lambda cm: int(round(cm / 2.54 * self.DPI))
        parts = ["<html><head><meta charset='utf-8'></head><body>"]
        for text, align, gap, spacing in title_lines(r.number, m):
            style = "margin-top:{}px; line-height:{}%;".format(round(gap * 96 / 72), round(spacing * 100))
            if align == "c":
                style += " text-align:center; text-indent:0px;"
            else:
                style += " text-align:left; text-indent:0px; margin-left:{}px;".format(
                    round(SIGN_INDENT_CM / 2.54 * self.DPI))
            parts.append("<p style='{}'>{}</p>".format(style, _esc(text) or "&nbsp;"))
        parts.append("<p class='break'>&nbsp;</p>")
        parts.append("<p class='c'><b>Лабораторна робота № {}</b></p><p class='c'>&nbsp;</p>".format(r.number))
        parts.append("<p class='j'><b>Тема:</b> {}</p>".format(_esc(r.topic)))
        parts.append("<p class='j'><b>Мета:</b> {}</p>".format(_esc(r.goal)))
        parts.append("<p class='c'>&nbsp;</p><p class='c'><b>Хід роботи</b></p>")
        # --- body
        for block in r.blocks:
            d = block.data
            if block.kind == "para":
                text = _esc(d["text"])
                if d["bold_prefix"] and d["text"].startswith(d["bold_prefix"]):
                    prefix = _esc(d["bold_prefix"])
                    text = "<b>{}</b>{}".format(prefix, text[len(prefix):])
                parts.append("<p class='j'>{}</p>".format(text))
            elif block.kind == "heading":
                parts.append("<p class='h'><b>{}</b></p>".format(_esc(d["text"])))
            elif block.kind == "note":
                parts.append("<p class='n'>{}</p>".format(_esc(d["text"])))
            elif block.kind == "formula":
                number = "({})".format(d["number"]) if d["number"] else "&nbsp;"
                parts.append("<table class='f' width='100%'><tr>"
                             "<td width='88%' align='center'>{}</td>"
                             "<td width='12%' align='right'>{}</td></tr></table>".format(_esc(d["text"]), number))
            elif block.kind == "figure":
                src = Path(d["image"]).resolve().as_posix()
                parts.append("<p class='c'><img src='file:///{}' width='{}'></p>".format(src, px(d["width_cm"])))
                parts.append("<p class='c'>Рисунок {} – {}</p><p class='c'>&nbsp;</p>".format(
                    d["number"], _esc(d["caption"])))
            elif block.kind == "table":
                parts.append(self._table_html(d))
            elif block.kind == "conclusions":
                lines = d["lines"]
                parts.append("<p class='c'>&nbsp;</p>")
                parts.append("<p class='j'><b>Висновки: </b>{}</p>".format(_esc(lines[0])))
                for line in lines[1:]:
                    parts.append("<p class='j'>{}</p>".format(_esc(line)))
        parts.append("</body></html>")
        return "".join(parts)

    @staticmethod
    def _table_html(d: dict) -> str:
        pt = int(d["size"].pt) if hasattr(d["size"], "pt") else 11
        out = ["<p class='tc'>Таблиця {} – {}</p>".format(d["number"], _esc(d["caption"]))]
        out.append("<table class='g' border='1' cellspacing='0' cellpadding='2' width='100%' "
                   "style='font-size:{}pt'>".format(pt))
        ncols = max(len(d["headers"]), max((len(r) for r in d["rows"]), default=1))
        out.append("<tr>" + "".join(
            "<th align='center'>{}</th>".format(_esc(d["headers"][j]) if j < len(d["headers"]) else "")
            for j in range(ncols)) + "</tr>")
        for row in d["rows"]:
            out.append("<tr>" + "".join(
                "<td align='center'>{}</td>".format(_esc(row[j]) if j < len(row) else "")
                for j in range(ncols)) + "</tr>")
        out.append("</table><p class='tc'>&nbsp;</p>")
        return "".join(out)

    def stylesheet(self) -> str:
        return """
        body {{ font-family: '{font}'; font-size: {size}pt; }}
        p {{ margin-top: 0px; margin-bottom: 0px; line-height: 150%; text-indent: 47px; }}
        p.c {{ text-align: center; text-indent: 0px; line-height: 100%; }}
        p.j {{ text-align: justify; }}
        p.h {{ text-align: justify; margin-top: 8px; }}
        p.n {{ text-align: justify; text-indent: 0px; }}
        p.s {{ text-indent: 0px; line-height: 100%; }}
        p.r {{ text-indent: 0px; line-height: 100%; text-align: right; margin-right: 38px; }}
        p.tc {{ text-indent: 0px; line-height: 100%; margin-top: 6px; }}
        p.break {{ page-break-after: always; }}
        table.f td {{ font-size: {size}pt; }}
        table.g {{ border-collapse: collapse; }}
        table.g th, table.g td {{ border: 1px solid #000000; }}
        """.format(font=FONT, size=int(SIZE.pt))

    def write(self, path: Path) -> Path:
        from PySide6.QtCore import QRectF, QSizeF
        from PySide6.QtGui import QFont, QPageSize, QPainter, QPdfWriter, QTextDocument
        from boryscad.render import ensure_qt_app
        ensure_qt_app()

        writer = QPdfWriter(str(path))
        writer.setPageSize(QPageSize(QPageSize.PageSizeId.A4))
        writer.setResolution(self.DPI)
        writer.setTitle("Звіт з лабораторної роботи № {}".format(self.r.number))
        writer.setCreator("Boryscad")
        from PySide6.QtCore import QMarginsF
        from PySide6.QtGui import QPageLayout
        writer.setPageMargins(QMarginsF(MARGIN_CM * 10, MARGIN_CM * 10, MARGIN_CM * 10, MARGIN_CM * 10),
                              QPageLayout.Unit.Millimeter)

        width = (PAGE_W_CM - 2 * MARGIN_CM) / 2.54 * self.DPI
        height = (PAGE_H_CM - 2 * MARGIN_CM) / 2.54 * self.DPI
        doc = QTextDocument()
        doc.setDefaultFont(QFont(FONT, int(SIZE.pt)))
        doc.setDefaultStyleSheet(self.stylesheet())
        doc.setHtml(self.html())
        doc.setPageSize(QSizeF(width, height))

        painter = QPainter(writer)
        pages = doc.pageCount()
        for page in range(pages):
            if page:
                writer.newPage()
            painter.save()
            painter.translate(0, -page * height)
            painter.setClipRect(QRectF(0, page * height, width, height))
            doc.drawContents(painter, QRectF(0, page * height, width, height))
            painter.restore()
            if page:  # the title page is counted but not numbered
                painter.setFont(QFont(FONT, int(SIZE.pt)))
                painter.drawText(QRectF(0, -0.55 / 2.54 * self.DPI, width, 20), 0x0002, str(page + 1))
        painter.end()
        return path
