"""Number and value formatting shared by renderers and the terminal output."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .values import (BuiltinRef, DisplayNode, DPair, DStack, DTable, DText, DValue, Displayable, RangeValue, Record,
                     SeriesGroup, Table, UserFunction, normalize_number)


@dataclass
class NumberFormat:
    precision: int = 4          # digits after the decimal point
    exp_threshold: int = 6      # use exponential notation when |exponent| >= threshold
    trailing_zeros: bool = False
    decimal_sep: str = "."


DEFAULT_FORMAT = NumberFormat()


def split_number(x: float, fmt: NumberFormat = DEFAULT_FORMAT) -> tuple[str, int | None]:
    """Return ``(mantissa, exponent)``; exponent is None for plain notation."""
    if isinstance(x, bool):
        x = int(x)
    if isinstance(x, (int, np.integer)) and abs(int(x)) < 10 ** 15:
        return str(int(x)), None
    x = float(x)
    if math.isnan(x):
        return "NaN", None
    if math.isinf(x):
        return ("∞" if x > 0 else "-∞"), None
    if x == 0:
        return "0", None
    exponent = int(math.floor(math.log10(abs(x))))
    if abs(exponent) >= fmt.exp_threshold or (abs(x) < 10 ** (-fmt.precision) and x != 0):
        mant = x / 10 ** exponent
        s = _fixed(mant, fmt)
        if s in ("10", "-10"):
            s = "1" if not s.startswith("-") else "-1"
            exponent += 1
        return s, exponent
    return _fixed(x, fmt), None


def _fixed(x: float, fmt: NumberFormat) -> str:
    s = f"{x:.{fmt.precision}f}"
    if not fmt.trailing_zeros and "." in s:
        s = s.rstrip("0").rstrip(".")
    if s in ("-0", ""):
        s = "0"
    if fmt.decimal_sep != ".":
        s = s.replace(".", fmt.decimal_sep)
    return s


def format_number(x: Any, fmt: NumberFormat = DEFAULT_FORMAT) -> str:
    x = normalize_number(x)
    if isinstance(x, complex):
        re, im = x.real, x.imag
        im_s = format_number(abs(im), fmt)
        if abs(re) < 10 ** (-fmt.precision - 2):
            return f"{'-' if im < 0 else ''}{im_s}i"
        return f"{format_number(re, fmt)} {'-' if im < 0 else '+'} {im_s}i"
    mant, exp = split_number(x, fmt)
    return mant if exp is None else f"{mant}·10^{exp}"


def format_value(value: Any, fmt: NumberFormat = DEFAULT_FORMAT, max_items: int = 40) -> str:
    value = normalize_number(value)
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, (int, float, complex)):
        return format_number(value, fmt)
    if isinstance(value, RangeValue):
        return format_value(value.to_array(), fmt, max_items)
    if isinstance(value, np.ndarray):
        return format_array(value, fmt, max_items)
    if isinstance(value, (UserFunction, BuiltinRef)):
        return repr(value)
    if isinstance(value, SeriesGroup):
        return f"<{len(value)} plot series>"
    if isinstance(value, Displayable) or isinstance(value, (Record, Table)):
        return format_display(value.display(), fmt)
    if hasattr(value, "series_kind"):
        return f"<plot series: {value.series_kind}>"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(format_value(v, fmt) for v in value) + "]"
    if value is None:
        return "—"
    return str(value)


def format_array(arr: np.ndarray, fmt: NumberFormat = DEFAULT_FORMAT, max_items: int = 40) -> str:
    if arr.ndim == 1:
        cells = [[format_value(v, fmt)] for v in arr[:max_items]]
        if arr.size > max_items:
            cells.append(["⋮"])
    elif arr.ndim == 2:
        cells = [[format_value(v, fmt) for v in row[:max_items]] for row in arr[:max_items]]
    else:
        return str(arr)
    if not cells:
        return "[ ]"
    widths = [max(len(r[c]) for r in cells if c < len(r)) for c in range(max(len(r) for r in cells))]
    lines = []
    for i, row in enumerate(cells):
        left, right = ("⎡", "⎤") if i == 0 else ("⎣", "⎦") if i == len(cells) - 1 else ("⎢", "⎥")
        if len(cells) == 1:
            left, right = "[", "]"
        lines.append(left + " " + "  ".join(c.rjust(widths[j]) for j, c in enumerate(row)) + " " + right)
    return "\n".join(lines)


def format_display(node: DisplayNode, fmt: NumberFormat = DEFAULT_FORMAT) -> str:
    if isinstance(node, DText):
        if node.role == "title":
            return f"── {node.text} ──"
        return node.text
    if isinstance(node, DValue):
        return format_value(node.value, fmt)
    if isinstance(node, DPair):
        text = format_value(node.value, fmt)
        if "\n" in text:
            pad = " " * (len(node.label) + 3)
            lines = text.split("\n")
            return f"{node.label} = " + lines[0] + "\n" + "\n".join(pad + l for l in lines[1:])
        return f"{node.label} = {text}"
    if isinstance(node, DTable):
        return format_table(node, fmt)
    if isinstance(node, DStack):
        return "\n".join(format_display(i, fmt) for i in node.items)
    return str(node)


def format_table(table: DTable, fmt: NumberFormat = DEFAULT_FORMAT) -> str:
    rows = [[format_value(c, fmt) if not isinstance(c, str) else c for c in r] for r in table.rows]
    rows = [["*" + c if (ri, ci) in table.highlight else c for ci, c in enumerate(r)] for ri, r in enumerate(rows)]
    table = DTable(table.headers, table.rows, table.title, set())
    headers = list(table.headers)
    ncols = max([len(headers)] + [len(r) for r in rows]) if (rows or headers) else 0
    headers += [""] * (ncols - len(headers))
    rows = [r + [""] * (ncols - len(r)) for r in rows]
    widths = [max([len(headers[c])] + [len(r[c]) for r in rows]) for c in range(ncols)]
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    out = []
    if table.title:
        out.append(table.title)
    out.append(sep)
    if any(headers):
        out.append("| " + " | ".join(h.center(widths[i]) for i, h in enumerate(headers)) + " |")
        out.append(sep.replace("-", "="))
    for ri, r in enumerate(rows):
        cells = []
        for ci, c in enumerate(r):
            text = c.rjust(widths[ci])
            if (ri, ci) in table.highlight:
                text = ("*" + c).rjust(widths[ci])
            cells.append(text)
        out.append("| " + " | ".join(cells) + " |")
    out.append(sep)
    return "\n".join(out)
