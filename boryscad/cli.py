"""Command line interface.

    boryscad                          open the GUI
    boryscad gui [file]               open the GUI with a file
    boryscad render lab3.bmath        render to lab3.html (or -o out.pdf / .png / .svg)
    boryscad formula "x := √(a^2+b^2)" -o formula.png
    boryscad run lab3.bmath           calculate and print results in the terminal
    boryscad convert lab3.bmath lab3.bcad
    boryscad themes | functions
"""
from __future__ import annotations

import argparse
import glob
import io
import os
import sys
import time
from pathlib import Path

from . import APP_NAME, __version__


def _utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass


def _expand(inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    for item in inputs:
        matches = glob.glob(item)
        files.extend(Path(m) for m in (matches or [item]))
    return files


def cmd_render(args) -> int:
    from .core.io import load_document
    from .core.theme import load_theme
    from .render import ensure_qt_app
    from .render.document import DocumentRenderer
    ensure_qt_app(headless=True)
    files = _expand(args.inputs)
    if args.output and len(files) > 1 and not Path(args.output).is_dir():
        print("error: -o must be a directory when rendering several files", file=sys.stderr)
        return 2
    status = 0
    for src in files:
        if not src.exists():
            print(f"error: {src} not found", file=sys.stderr)
            status = 1
            continue
        doc = load_document(src)
        theme = load_theme(args.theme or doc.settings.theme or ("Paper" if args.format in ("pdf", "png") else None))
        renderer = DocumentRenderer(theme, scale=args.scale)
        started = time.perf_counter()
        results = renderer.calculate(doc)
        if args.output:
            out = Path(args.output)
            if out.is_dir():
                out = out / (src.stem + "." + (args.format or "html"))
        else:
            out = src.with_suffix("." + (args.format or "html"))
        renderer.export(doc, out, results)
        errors = results.errors()
        elapsed = time.perf_counter() - started
        print(f"✓ {src} → {out}  ({elapsed:.2f} s, {len(doc.regions)} regions, {len(errors)} error(s))")
        for rid, msg in errors[:20]:
            print(f"   ⚠ {msg}")
        if errors:
            status = max(status, 1 if args.strict else 0)
        if args.open:
            os.startfile(out) if hasattr(os, "startfile") else None  # type: ignore[attr-defined]
    return status


def cmd_formula(args) -> int:
    from .core.theme import load_theme
    from .render import ensure_qt_app
    from .render.document import render_formula
    ensure_qt_app(headless=True)
    source = args.source.replace("\\n", "\n")
    out = Path(args.output or "formula.png")
    render_formula(source, load_theme(args.theme or "Paper"), out, evaluate=not args.no_eval, scale=args.scale)
    print(f"✓ formula → {out}")
    if args.open and hasattr(os, "startfile"):
        os.startfile(out)  # type: ignore[attr-defined]
    return 0


def cmd_run(args) -> int:
    from .core.document import MathRegion, PlotRegion, TextRegion
    from .core.io import load_document
    from .core.markdown import to_plain
    from .engine import NumberFormat, WorksheetEngine, format_value
    from .language import ast as A
    doc = load_document(args.input)
    engine = WorksheetEngine()
    results = engine.run(doc)
    fmt = NumberFormat(precision=doc.settings.precision)
    bold, dim, red, green, reset = ("\033[1m", "\033[2m", "\033[31m", "\033[32m", "\033[0m") if sys.stdout.isatty() else ("",) * 5
    for region in doc.ordered():
        res = results.regions.get(region.id)
        if isinstance(region, TextRegion):
            text = res.text if res is not None and res.text is not None else region.text
            if region.style in ("h1", "h2", "h3"):
                print(f"\n{bold}{to_plain(text)}{reset}")
            elif not args.quiet:
                print(f"{dim}{to_plain(text)}{reset}")
        elif isinstance(region, MathRegion):
            if res is None:
                continue
            if res.parse_error:
                print(f"{red}syntax error in:\n{region.source}\n  {res.parse_error}{reset}")
                continue
            lines = region.source.split("\n")
            for st in res.statements:
                src = _stmt_source(region.source, st.stmt)
                if st.error:
                    print(f"{src}\n  {red}⚠ {st.error}{reset}")
                elif st.kind == "evaluate":
                    value = format_value(st.value, fmt)
                    if "\n" in value:
                        print(f"{src}\n" + "\n".join("    " + l for l in value.split("\n")))
                    else:
                        print(f"{src} {green}{value}{reset}")
                elif st.kind == "symbolic":
                    print(f"{src}  {green}{st.symbolic_text}{reset}")
                elif not args.quiet:
                    print(f"{dim}{src}{reset}")
        elif isinstance(region, PlotRegion):
            n = len(res.plot.series) if res is not None and res.plot is not None else 0
            print(f"{dim}[plot: {region.title or 'untitled'} — {n} series]{reset}")
            if res is not None and res.error:
                print(f"  {red}⚠ {res.error}{reset}")
    print(f"\n{dim}calculated in {results.elapsed * 1000:.0f} ms{reset}")
    return 1 if results.errors() and args.strict else 0


def _stmt_source(source: str, stmt) -> str:
    if stmt.span:
        return source[stmt.span[0]:stmt.span[1]].strip()
    return type(stmt).__name__


def cmd_convert(args) -> int:
    from .core.io import load_document, save_document
    doc = load_document(args.input)
    if args.output.lower().endswith(".bcad") and doc.flow_layout:
        from .render import ensure_qt_app
        from .core.theme import load_theme
        from .render.document import DocumentRenderer
        ensure_qt_app(headless=True)
        renderer = DocumentRenderer(load_theme(None))
        renderer.layout(doc, renderer.calculate(doc))  # assigns absolute positions
        doc.flow_layout = False
        doc.rows = []
    save_document(doc, args.output)
    print(f"✓ {args.input} → {args.output}")
    return 0


def cmd_themes(args) -> int:
    from .core.theme import available_themes, themes_dir
    for name, theme in available_themes().items():
        print(f"  {name:<18} {'dark ' if theme.dark else 'light'}  {theme.path}")
    print(f"\nCustom themes: {themes_dir()}")
    return 0


def cmd_functions(args) -> int:
    from .engine.registry import default_registry
    reg = default_registry()
    for category, specs in reg.categories().items():
        if args.category and args.category.lower() not in category.lower():
            continue
        print(f"\n{category}")
        for spec in specs:
            print(f"  {spec.signature:<60} {spec.doc}")
    if not args.category:
        print("\nSymbolic keywords (expr -> keyword):", ", ".join(reg.symbolic_names()))
    return 0


def cmd_gui(args) -> int:
    from .ui.app import run_gui
    return run_gui(getattr(args, "file", None))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="boryscad", description=f"{APP_NAME} {__version__} — math worksheets")
    parser.add_argument("--version", action="version", version=f"{APP_NAME} {__version__}")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("gui", help="open the graphical editor")
    p.add_argument("file", nargs="?")
    p.set_defaults(func=cmd_gui)

    p = sub.add_parser("render", help="calculate and render .bmath/.bcad files to html, pdf, png or svg")
    p.add_argument("inputs", nargs="+")
    p.add_argument("-o", "--output")
    p.add_argument("-f", "--format", choices=["html", "pdf", "png", "svg"])
    p.add_argument("-t", "--theme")
    p.add_argument("--scale", type=float, default=2.0, help="raster scale for plots and PNG")
    p.add_argument("--open", action="store_true", help="open the result")
    p.add_argument("--strict", action="store_true", help="exit with 1 when the sheet has errors")
    p.set_defaults(func=cmd_render)

    p = sub.add_parser("formula", help="render a single formula")
    p.add_argument("source")
    p.add_argument("-o", "--output")
    p.add_argument("-t", "--theme")
    p.add_argument("--scale", type=float, default=3.0)
    p.add_argument("--no-eval", action="store_true", help="only typeset, do not calculate")
    p.add_argument("--open", action="store_true")
    p.set_defaults(func=cmd_formula)

    p = sub.add_parser("run", help="calculate a worksheet and print the results")
    p.add_argument("input")
    p.add_argument("-q", "--quiet", action="store_true", help="only print evaluated results")
    p.add_argument("--strict", action="store_true")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("convert", help="convert between .bmath and .bcad")
    p.add_argument("input")
    p.add_argument("output")
    p.set_defaults(func=cmd_convert)

    p = sub.add_parser("themes", help="list themes")
    p.set_defaults(func=cmd_themes)

    p = sub.add_parser("functions", help="list builtin functions")
    p.add_argument("category", nargs="?")
    p.set_defaults(func=cmd_functions)
    return parser


def main(argv: list[str] | None = None) -> int:
    _utf8_stdout()
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    if argv and not argv[0].startswith("-") and argv[0] not in parser._subparsers._group_actions[0].choices:  # type: ignore[union-attr]
        if Path(argv[0]).suffix.lower() in (".bmath", ".bcad"):
            argv = ["gui"] + argv
    if not argv:
        argv = ["gui"]
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        return 130


def gui_main() -> int:
    return main(["gui"] + sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
