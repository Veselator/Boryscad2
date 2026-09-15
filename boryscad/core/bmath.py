"""The ``.bmath`` plain-text worksheet format (easy to write by hand and to render from the terminal).

Syntax overview::

    @title Lab 3 — nonlinear programming      document title
    # Heading / ## Subheading / ### Minor heading
    > Paragraph text with **bold**, *italic* and {{x}} value interpolation.
    N := 5                                    math lines; a blank line starts a new region
    f(x) := x^2 - N
    f(2) =
    @plot title="f(x)" x=-5..5 y=-10..30 width=520 height=320
    fplot(f(x), x, -5, 5, label="f")
    @end
    @image picture.png width=300 caption="Figure 1"
    @row            regions until @end are placed side by side
    @pagebreak
"""
from __future__ import annotations

import base64
import mimetypes
import shlex
from pathlib import Path
from typing import Optional

from .document import Document, ImageRegion, MathRegion, PageBreakRegion, PlotRegion, Region, TextRegion

_BOOL = {"1": True, "true": True, "yes": True, "on": True, "0": False, "false": False, "no": False, "off": False}


def _options(text: str) -> dict[str, str]:
    try:
        tokens = shlex.split(text, posix=True)
    except ValueError:
        tokens = text.split()
    opts: dict[str, str] = {}
    for tok in tokens:
        if "=" in tok:
            k, v = tok.split("=", 1)
            opts[k.strip().lower()] = v
        else:
            opts[tok.strip().lower()] = "1"
    return opts


def read_bmath(text: str, base_dir: Path | None = None) -> Document:
    doc = Document(flow_layout=True)
    base_dir = base_dir or Path.cwd()
    lines = text.replace("\r\n", "\n").split("\n")
    math_lines: list[str] = []
    para_lines: list[str] = []
    row_stack: Optional[list[str]] = None
    brace_depth = 0

    def add(region: Region):
        doc.add(region)
        if row_stack is not None:
            row_stack.append(region.id)

    def flush_math():
        nonlocal brace_depth
        if math_lines and any(l.strip() for l in math_lines):
            add(MathRegion(source="\n".join(math_lines).strip("\n")))
        math_lines.clear()
        brace_depth = 0

    def flush_para():
        if para_lines:
            add(TextRegion(text="\n".join(para_lines).strip("\n"), style="body"))
        para_lines.clear()

    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.rstrip()
        stripped = line.strip()
        i += 1
        if brace_depth > 0 and not stripped.startswith("@"):
            math_lines.append(line)
            brace_depth += line.count("{") - line.count("}")
            continue
        if not stripped:
            flush_math()
            flush_para()
            continue
        if stripped.startswith("//") and not math_lines:
            continue
        if stripped.startswith(">"):
            flush_math()
            para_lines.append(stripped[1:].lstrip() if len(stripped) > 1 else "")
            continue
        heading = next((lvl for lvl in (3, 2, 1) if stripped.startswith("#" * lvl + " ")), None)
        if heading:
            flush_math(); flush_para()
            add(TextRegion(text=stripped[heading + 1:].strip(), style=f"h{heading}"))
            continue
        if stripped.startswith("@"):
            flush_math(); flush_para()
            directive, _, rest = stripped[1:].partition(" ")
            directive = directive.lower()
            if directive == "title":
                doc.title = rest.strip()
            elif directive == "author":
                doc.author = rest.strip()
            elif directive == "precision":
                doc.settings.precision = int(rest.strip() or 4)
            elif directive == "theme":
                doc.settings.theme = rest.strip()
            elif directive == "pagebreak":
                add(PageBreakRegion())
            elif directive == "row":
                row_stack = []
            elif directive == "end":
                if row_stack is not None:
                    if len(row_stack) > 1:
                        doc.rows.append(row_stack)
                    row_stack = None
            elif directive == "plot":
                opts = _options(rest)
                series: list[str] = []
                while i < len(lines) and lines[i].strip().lower() != "@end":
                    s = lines[i].strip()
                    if s and not s.startswith("//"):
                        series.append(s)
                    i += 1
                i += 1  # skip @end
                add(_plot_from_options(opts, series))
            elif directive == "image":
                try:
                    tokens = shlex.split(rest)
                except ValueError:
                    tokens = rest.split()
                path_text = next((t for t in tokens if "=" not in t), "")
                opts = {k.lower(): v for k, _, v in (t.partition("=") for t in tokens if "=" in t)}
                add(image_region(base_dir / path_text, opts))
            elif directive == "text":
                opts = _options(rest)
                block: list[str] = []
                while i < len(lines) and lines[i].strip().lower() != "@end":
                    block.append(lines[i])
                    i += 1
                i += 1
                add(TextRegion(text="\n".join(block), style=opts.get("style", "body")))
            else:
                add(TextRegion(text=f"Unknown directive @{directive}", style="note"))
            continue
        flush_para()
        math_lines.append(line)
        brace_depth += line.count("{") - line.count("}")
    flush_math()
    flush_para()
    return doc


def _plot_from_options(opts: dict[str, str], series: list[str]) -> PlotRegion:
    region = PlotRegion(series=series)
    region.title = opts.get("title", "")
    region.xlabel = opts.get("xlabel", "")
    region.ylabel = opts.get("ylabel", "")
    region.zlabel = opts.get("zlabel", "")
    region.xlim = opts.get("x", opts.get("xlim", ""))
    region.ylim = opts.get("y", opts.get("ylim", ""))
    region.zlim = opts.get("z", opts.get("zlim", ""))
    region.mode = opts.get("mode", "3d" if opts.get("3d") == "1" else "2d")
    region.xscale = opts.get("xscale", "linear")
    region.yscale = opts.get("yscale", "linear")
    for key in ("grid", "legend", "equal"):
        if key in opts:
            setattr(region, key, _BOOL.get(opts[key].lower(), True))
    for key in ("width", "height"):
        if key in opts:
            setattr(region, key, float(opts[key]))
    return region


def image_region(path: Path, opts: dict[str, str] | None = None) -> Region:
    opts = opts or {}
    try:
        data = path.read_bytes()
    except OSError as exc:
        return TextRegion(text=f"⚠ cannot load image {path}: {exc}", style="note")
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    width = float(opts.get("width", 0) or 0)
    height = float(opts.get("height", 0) or 0)
    if not width or not height:
        w, h = image_size(data)
        if w and h:
            if width and not height:
                height = width * h / w
            elif height and not width:
                width = height * w / h
            else:
                scale = min(1.0, 640 / w)
                width, height = w * scale, h * scale
        width = width or 320
        height = height or 240
    return ImageRegion(data=base64.b64encode(data).decode("ascii"), mime=mime, width=width, height=height,
                       caption=opts.get("caption", ""), source_path=str(path))


def image_size(data: bytes) -> tuple[int, int]:
    try:
        from PIL import Image
        import io
        with Image.open(io.BytesIO(data)) as img:
            return img.size
    except Exception:
        return 0, 0


def write_bmath(doc: Document) -> str:
    out: list[str] = []
    if doc.title and doc.title != "Untitled":
        out.append(f"@title {doc.title}")
    if doc.author:
        out.append(f"@author {doc.author}")
    if doc.settings.precision != 4:
        out.append(f"@precision {doc.settings.precision}")
    if out:
        out.append("")
    for region in doc.ordered():
        if isinstance(region, MathRegion):
            out.append(region.source)
        elif isinstance(region, TextRegion):
            if region.style in ("h1", "h2", "h3"):
                out.append("#" * int(region.style[1]) + " " + region.text.replace("\n", " "))
            elif region.format == "html":
                out.append("@text style=html")
                out.append(region.text)
                out.append("@end")
            else:
                out.extend("> " + l if l else ">" for l in region.text.split("\n"))
        elif isinstance(region, PlotRegion):
            opts = []
            for key, attr in (("title", "title"), ("xlabel", "xlabel"), ("ylabel", "ylabel"), ("x", "xlim"), ("y", "ylim")):
                value = getattr(region, attr)
                if value:
                    opts.append(f'{key}="{value}"')
            if region.mode != "2d":
                opts.append(f"mode={region.mode}")
            if region.equal:
                opts.append("equal=1")
            opts.append(f"width={int(region.width)} height={int(region.height)}")
            out.append("@plot " + " ".join(opts))
            out.extend(region.series)
            out.append("@end")
        elif isinstance(region, ImageRegion):
            if region.source_path:
                out.append(f'@image "{region.source_path}" width={int(region.width)} height={int(region.height)}')
            else:
                out.append("// (embedded image omitted — save as .bcad to keep it)")
        elif isinstance(region, PageBreakRegion):
            out.append("@pagebreak")
        out.append("")
    return "\n".join(out).rstrip() + "\n"
