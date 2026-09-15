"""Builds box layouts for statements, expressions and values."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from ..core.theme import Theme
from ..engine.formatting import NumberFormat, split_number
from ..engine.values import (BuiltinRef, DisplayNode, Displayable, DPair, DStack, DTable, DText, DValue, RangeValue,
                             Record, SeriesGroup, Table, UserFunction, normalize_number)
from ..language import ast as A
from . import boxes as B

GREEK = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "zeta": "ζ", "eta": "η",
    "theta": "θ", "iota": "ι", "kappa": "κ", "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ", "omicron": "ο",
    "rho": "ρ", "sigma": "σ", "tau": "τ", "upsilon": "υ", "phi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω",
    "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Xi": "Ξ", "Pi": "Π", "Sigma": "Σ", "Phi": "Φ",
    "Psi": "Ψ", "Omega": "Ω", "pi": "π",
}
COMPARE_SYMBOLS = {"==": "=", "!=": "≠", "<=": "≤", ">=": "≥", "<": "<", ">": ">"}
_TRAILING_DIGITS = re.compile(r"^([^\W\d_]+)(\d+)$")


@dataclass
class MathStyle:
    family: str = "Cambria Math"
    size: float = 17.0
    text_family: str = "Segoe UI"
    mono_family: str = "Cascadia Mono"
    colors: dict[str, str] = field(default_factory=dict)
    digits_subscript: bool = True
    max_rows: int = 18
    max_cols: int = 12

    @classmethod
    def from_theme(cls, theme: Theme, scale: float = 1.0) -> "MathStyle":
        return cls(theme.font("math", "Cambria Math"), float(theme.font("math_size", 17)) * scale,
                   theme.font("text", "Segoe UI"), theme.font("mono", "Cascadia Mono"), dict(theme.colors))

    def c(self, key: str) -> str:
        return self.colors.get(key, "#000000")


class FormulaBuilder:
    SCALES = (1.0, 0.72, 0.58)

    def __init__(self, measurer: B.Measurer, style: MathStyle, fmt: NumberFormat | None = None, registry=None):
        self.m = measurer
        self.s = style
        self.fmt = fmt or NumberFormat()
        self.registry = registry
        self.special_calls: dict[str, Callable[[A.Call, int], Optional[B.Box]]] = {
            "sqrt": self._call_sqrt, "nthroot": self._call_nthroot, "abs": self._call_abs,
            "sum": self._call_bigop, "Σ": self._call_bigop, "prod": self._call_bigop, "Π": self._call_bigop,
            "integral": self._call_integral, "integrate": self._call_integral, "deriv": self._call_deriv,
            "diff": self._call_deriv, "limit": self._call_limit, "floor": self._call_floor, "ceil": self._call_ceil,
            "transpose": self._call_transpose, "inverse": self._call_inverse, "exp": self._call_exp,
            "det": self._call_det, "norm": self._call_norm, "log": self._call_log,
        }

    # ------------------------------------------------------------- basics
    def font(self, level: int = 0, italic: bool = False, bold: bool = False, family: str | None = None) -> B.FontSpec:
        size = self.s.size * self.SCALES[min(level, 2)]
        return B.FontSpec(family or self.s.family, size, italic, bold)

    def size(self, level: int) -> float:
        return self.s.size * self.SCALES[min(level, 2)]

    def axis(self, level: int) -> float:
        return self.size(level) * 0.25

    def text(self, text: str, level: int = 0, color: str = "math_text", italic: bool = False, bold: bool = False,
             family: str | None = None) -> B.Text:
        return B.Text(self.m, text, self.font(level, italic, bold, family), self.s.c(color) if not color.startswith("#") else color)

    def op(self, symbol: str, level: int, space: float = 0.22, color: str = "math_operator", bold: bool = False) -> B.Box:
        gap = self.size(level) * space
        return B.HBox([B.Space(gap), self.text(symbol, level, color, bold=bold), B.Space(gap)])

    def row(self, *items: B.Box, gap: float = 0.0) -> B.HBox:
        return B.HBox(list(items), gap)

    # -------------------------------------------------------------- names
    def name(self, name: str, level: int = 0, color: str = "math_variable", italic: bool = True) -> B.Box:
        base, sub = name, None
        if "_" in name.strip("_"):
            base, sub = name.split("_", 1)
        elif self.s.digits_subscript:
            match = _TRAILING_DIGITS.match(name)
            if match and len(match.group(1)) <= 3:
                base, sub = match.group(1), match.group(2)
        base_text = GREEK.get(base, base)
        is_italic = italic and not (len(base_text) == 1 and not base_text.isascii() and base_text.isupper())
        base_box = self.text(base_text, level, color, italic=is_italic)
        if sub is None:
            return base_box
        sub_text = " ".join(GREEK.get(p, p) for p in sub.split("_"))
        sub_box = self.text(sub_text, level + 1, color, italic=not sub_text.isdigit() and italic)
        return B.Scripts(base_box, None, sub_box, self.size(level))

    # -------------------------------------------------------- expressions
    def expr(self, node: A.Node, level: int = 0) -> B.Box:
        method = getattr(self, "_" + type(node).__name__, None)
        if method is None:
            return self.text(f"‹{type(node).__name__}›", level, "math_error")
        return method(node, level)

    def _Number(self, node: A.Number, level):
        text = node.text
        if "e" in text.lower():
            mant, exp = re.split("[eE]", text)
            return self._sci(mant, int(exp), level, "math_number")
        return self.text(text, level, "math_number")

    def _sci(self, mant: str, exp: int, level: int, color: str) -> B.Box:
        base = self.text(f"{mant}·10", level, color)
        return B.Scripts(base, self.text(str(exp).replace("-", "−"), level + 1, color), None, self.size(level))

    def _Imag(self, node: A.Imag, level):
        return self.row(self.text(node.text, level, "math_number"), self.text("i", level, "math_number", italic=True))

    def _Infinity(self, node, level):
        return self.text("∞", level, "math_number")

    def _String(self, node: A.String, level):
        return self.text(f"“{node.value}”", level, "math_string", family=self.s.text_family)

    def _Var(self, node: A.Var, level):
        if self.registry is not None and node.name in getattr(self.registry, "_constants", {}) and len(node.name) > 1 \
                and node.name not in GREEK:
            return self.text(node.name, level, "math_variable")
        return self.name(node.name, level)

    def _Paren(self, node: A.Paren, level):
        return self.delim(self.expr(node.body, level), "(", ")", level)

    def delim(self, body: B.Box, left: str, right: str, level: int, color: str = "math_operator") -> B.Box:
        return B.Delimited(self.m, body, left, right, self.font(level), self.s.c(color), self.axis(level))

    def _BinOp(self, node: A.BinOp, level):
        op = node.op
        if op in ("/", "./"):
            num = self.expr(A.strip_parens(node.left), level if level > 0 else 0)
            den = self.expr(A.strip_parens(node.right), level if level > 0 else 0)
            if level >= 1:
                return self.row(self.expr(node.left, level), self.text("/", level, "math_operator"), self.expr(node.right, level))
            return B.Fraction(num, den, self.size(level), self.s.c("math_operator"), self.axis(level))
        if op in ("^", ".^"):
            base = self.expr(node.left, level)
            if isinstance(node.right, A.Var) and node.right.name == "T":
                sup = self.text("T", level + 1, "math_operator")
            else:
                sup = self.expr(A.strip_parens(node.right), level + 1)
            if isinstance(A.strip_parens(node.left), (A.BinOp, A.UnOp, A.Compare)) and not isinstance(node.left, A.Paren):
                base = self.delim(base, "(", ")", level)
            return B.Scripts(base, sup, None, self.size(level))
        symbol = {"*": "·", ".*": "·", "+": "+", "-": "−", "and": "∧", "or": "∨"}.get(op, op)
        space = 0.1 if op in ("*", ".*") else 0.22
        return self.row(self.expr(node.left, level), self.op(symbol, level, space), self.expr(node.right, level))

    def _UnOp(self, node: A.UnOp, level):
        symbol = {"-": "−", "not": "¬"}.get(node.op, node.op)
        return self.row(self.text(symbol, level, "math_operator"), B.Space(self.size(level) * 0.04),
                        self.expr(node.operand, level))

    def _Compare(self, node: A.Compare, level, constraint: bool = False):
        items: list[B.Box] = [self.expr(node.operands[0], level)]
        for op, operand in zip(node.ops, node.operands[1:]):
            items.append(self.op(COMPARE_SYMBOLS.get(op, op), level, 0.25,
                                 "math_constraint" if constraint else "math_operator", bold=constraint and op == "=="))
            items.append(self.expr(operand, level))
        return B.HBox(items)

    def _RangeExpr(self, node: A.RangeExpr, level):
        items = [self.expr(node.start, level)]
        if node.second is not None:
            items += [self.text(",", level, "math_operator"), B.Space(self.size(level) * 0.2), self.expr(node.second, level)]
        items += [self.op("..", level, 0.08), self.expr(node.stop, level)]
        if node.step is not None:
            items += [self.op("step", level, 0.25, "math_keyword"), self.expr(node.step, level)]
        return B.HBox(items)

    def _Keyword(self, node: A.Keyword, level):
        return self.row(self.text(node.name, level, "math_keyword"), self.op("=", level, 0.1), self.expr(node.value, level))

    def args_box(self, args: tuple, kwargs: tuple, level: int) -> B.Box:
        items: list[B.Box] = []
        for i, a in enumerate(list(args) + list(kwargs)):
            if i:
                items.append(self.text(",", level, "math_operator"))
                items.append(B.Space(self.size(level) * 0.2))
            items.append(self.expr(a, level))
        return B.HBox(items)

    def _Call(self, node: A.Call, level):
        special = self.special_calls.get(node.name)
        if special is not None:
            box = special(node, level)
            if box is not None:
                return box
        is_builtin = self.registry is not None and self.registry.get(node.name) is not None
        name = self.text(node.name, level, "math_function") if is_builtin else self.name(node.name, level)
        return self.row(name, self.delim(self.args_box(node.args, node.kwargs, level), "(", ")", level))

    def _Index(self, node: A.Index, level):
        base = self.expr(node.target, level)
        sub = self.args_box(node.indices, (), level + 1)
        return B.Scripts(base, None, sub, self.size(level))

    def _Member(self, node: A.Member, level):
        return self.row(self.expr(node.target, level), self.text(".", level, "math_operator"),
                        self.name(node.name, level, "math_variable", italic=True))

    def _Factorial(self, node: A.Factorial, level):
        return self.row(self.expr(node.operand, level), self.text("!", level, "math_operator"))

    def _Abs(self, node: A.Abs, level):
        return self.delim(self.expr(node.operand, level), "|", "|", level)

    def _Sqrt(self, node: A.Sqrt, level):
        return B.Radical(self.expr(A.strip_parens(node.operand), level), self.size(level), self.s.c("math_operator"))

    def _MatrixLit(self, node: A.MatrixLit, level):
        if not node.rows:
            return self.text("[ ]", level, "math_operator")
        cells = [[self.expr(c, level) for c in row] for row in node.rows]
        return B.Matrix(self.m, cells, self.font(level), self.s.c("math_operator"), self.axis(level))

    def _IfExpr(self, node: A.IfExpr, level):
        lines = [
            self.row(self.expr(node.then, level), self.op("if", level, 0.35, "math_keyword"), self.expr(node.cond, level)),
            self.row(self.expr(node.otherwise, level), self.op("otherwise", level, 0.35, "math_keyword")),
        ]
        return B.Brace(B.VStack(lines, self.size(level) * 0.2), self.size(level), self.s.c("math_operator"), self.axis(level))

    # --------------------------------------------------------- programs
    def _Program(self, node: A.Program, level):
        return B.ProgramBar([self.pstmt(p, level) for p in node.body] or [self.text(" ", level)], self.size(level),
                            self.s.c("math_operator"))

    def pstmt(self, st: A.PStmt, level: int) -> B.Box:
        if isinstance(st, A.PExpr):
            return self.expr(st.expr, level)
        if isinstance(st, A.LocalAssign):
            return self.row(self.expr(st.target, level), self.op("←", level, 0.25), self.expr(st.value, level))
        if isinstance(st, A.ForStmt):
            head = self.row(self.text("for", level, "math_keyword", bold=True), B.Space(self.size(level) * 0.3),
                            self.name(st.var, level), self.op("∈", level, 0.25), self.expr(st.iterable, level))
            return B.VStack([head, self.row(B.Space(self.size(level) * 0.8), self._Program(st.body, level))],
                            self.size(level) * 0.15)
        if isinstance(st, A.WhileStmt):
            head = self.row(self.text("while", level, "math_keyword", bold=True), B.Space(self.size(level) * 0.3),
                            self.expr(st.cond, level))
            return B.VStack([head, self.row(B.Space(self.size(level) * 0.8), self._Program(st.body, level))],
                            self.size(level) * 0.15)
        if isinstance(st, A.IfStmt):
            head = self.row(self.text("if", level, "math_keyword", bold=True), B.Space(self.size(level) * 0.3),
                            self.expr(st.cond, level))
            parts = [head, self.row(B.Space(self.size(level) * 0.8), self._Program(st.body, level))]
            if isinstance(st.orelse, A.Program):
                parts += [self.text("else", level, "math_keyword", bold=True),
                          self.row(B.Space(self.size(level) * 0.8), self._Program(st.orelse, level))]
            elif isinstance(st.orelse, A.IfStmt):
                parts += [self.row(self.text("else", level, "math_keyword", bold=True), B.Space(self.size(level) * 0.3),
                                   self.pstmt(st.orelse, level))]
            return B.VStack(parts, self.size(level) * 0.15)
        if isinstance(st, A.ReturnStmt):
            return self.row(self.text("return", level, "math_keyword", bold=True), B.Space(self.size(level) * 0.3),
                            self.expr(st.value, level))
        if isinstance(st, A.BreakStmt):
            return self.text("break", level, "math_keyword", bold=True)
        if isinstance(st, A.ContinueStmt):
            return self.text("continue", level, "math_keyword", bold=True)
        return self.text("?", level, "math_error")

    # ------------------------------------------------- special function layouts
    def _call_sqrt(self, node, level):
        if len(node.args) != 1 or node.kwargs:
            return None
        return B.Radical(self.expr(A.strip_parens(node.args[0]), level), self.size(level), self.s.c("math_operator"))

    def _call_nthroot(self, node, level):
        if len(node.args) != 2:
            return None
        return B.Radical(self.expr(A.strip_parens(node.args[0]), level), self.size(level), self.s.c("math_operator"),
                         self.expr(node.args[1], level + 1))

    def _call_abs(self, node, level):
        if len(node.args) != 1:
            return None
        return self.delim(self.expr(A.strip_parens(node.args[0]), level), "|", "|", level)

    def _call_norm(self, node, level):
        if len(node.args) != 1:
            return None
        return self.delim(self.expr(A.strip_parens(node.args[0]), level), "‖", "‖", level)

    def _call_det(self, node, level):
        return self._call_abs(node, level)

    def _call_floor(self, node, level):
        if len(node.args) != 1:
            return None
        return self.delim(self.expr(A.strip_parens(node.args[0]), level), "⌊", "⌋", level)

    def _call_ceil(self, node, level):
        if len(node.args) != 1:
            return None
        return self.delim(self.expr(A.strip_parens(node.args[0]), level), "⌈", "⌉", level)

    def _call_exp(self, node, level):
        if len(node.args) != 1 or level >= 1:
            return None
        return B.Scripts(self.text("e", level, "math_variable", italic=True), self.expr(A.strip_parens(node.args[0]), level + 1),
                         None, self.size(level))

    def _call_transpose(self, node, level):
        if len(node.args) != 1:
            return None
        return B.Scripts(self._maybe_paren(node.args[0], level), self.text("T", level + 1, "math_operator"), None, self.size(level))

    def _call_inverse(self, node, level):
        if len(node.args) != 1:
            return None
        return B.Scripts(self._maybe_paren(node.args[0], level), self.text("−1", level + 1, "math_number"), None, self.size(level))

    def _call_log(self, node, level):
        if len(node.args) != 2:
            return None
        base = B.Scripts(self.text("log", level, "math_function"), None, self.expr(node.args[1], level + 1), self.size(level))
        return self.row(base, self.delim(self.expr(A.strip_parens(node.args[0]), level), "(", ")", level))

    def _maybe_paren(self, node, level):
        inner = A.strip_parens(node)
        box = self.expr(inner, level)
        if isinstance(inner, (A.BinOp, A.UnOp)):
            return self.delim(box, "(", ")", level)
        return box

    def _call_bigop(self, node, level):
        symbol = "∑" if node.name in ("sum", "Σ") else "∏"
        big = self.text(symbol, level, "math_operator", family=self.s.family)
        big = B.Text(self.m, symbol, self.font(level).scaled(1.55).with_(italic=False), self.s.c("math_operator"))
        if len(node.args) == 1:
            return B.BigOperator(big, None, None, self._maybe_paren(node.args[0], level), self.size(level), self.axis(level))
        if len(node.args) == 2:
            lower = self.expr(node.args[1], level + 1)
            return B.BigOperator(big, lower, None, self._maybe_paren(node.args[0], level), self.size(level), self.axis(level))
        if len(node.args) == 4:
            lower = self.row(self.expr(node.args[1], level + 1), self.text("=", level + 1, "math_operator"),
                             self.expr(node.args[2], level + 1))
            upper = self.expr(node.args[3], level + 1)
            body = self._maybe_paren(node.args[0], level)
            return B.BigOperator(big, lower, upper, body, self.size(level), self.axis(level))
        return None

    def _call_integral(self, node, level):
        if len(node.args) not in (2, 4):
            return None
        sign = B.Text(self.m, "∫", self.font(level).scaled(1.6).with_(italic=False), self.s.c("math_operator"))
        body = self._maybe_paren(node.args[0], level)
        dx = self.row(B.Space(self.size(level) * 0.2), self.text("d", level, "math_operator"), self.expr(node.args[1], level))
        if len(node.args) == 4:
            shift = (sign.a - sign.d) / 2 - self.axis(level)
            sign_box = B.Raise(sign, -shift)
            scripts = B.Scripts(sign_box, self.expr(node.args[3], level + 1), self.expr(node.args[2], level + 1),
                                self.size(level) * 1.5)
            return self.row(scripts, B.Space(self.size(level) * 0.1), body, dx)
        shift = (sign.a - sign.d) / 2 - self.axis(level)
        return self.row(B.Raise(sign, -shift), B.Space(self.size(level) * 0.1), body, dx)

    def _call_deriv(self, node, level):
        if len(node.args) not in (2, 3):
            return None
        var = self.expr(node.args[1], level)
        d = self.text("d", level, "math_operator")
        if len(node.args) == 3:
            n = self.expr(node.args[2], level + 1)
            num = B.Scripts(d, n, None, self.size(level))
            den = self.row(d, B.Scripts(var, self.expr(node.args[2], level + 1), None, self.size(level)))
        else:
            num, den = d, self.row(d, var)
        frac = B.Fraction(num, den, self.size(level), self.s.c("math_operator"), self.axis(level))
        return self.row(frac, B.Space(self.size(level) * 0.1), self._maybe_paren(node.args[0], level))

    def _call_limit(self, node, level):
        if len(node.args) != 3:
            return None
        lim = self.text("lim", level, "math_function")
        lower = self.row(self.expr(node.args[1], level + 1), self.text("→", level + 1, "math_operator"),
                         self.expr(node.args[2], level + 1))
        return B.BigOperator(lim, lower, None, self._maybe_paren(node.args[0], level), self.size(level),
                             (lim.a - lim.d) / 2)

    # -------------------------------------------------------------- values
    def value(self, value: Any, level: int = 0, color: str = "math_result") -> B.Box:
        value = normalize_number(value)
        if isinstance(value, bool):
            value = int(value)
        if isinstance(value, (int, float)):
            return self.number(value, level, color)
        if isinstance(value, complex):
            re_b = self.number(value.real, level, color)
            im_b = self.number(abs(value.imag), level, color)
            i_b = self.text("i", level, color, italic=True)
            if abs(value.real) < 10 ** (-self.fmt.precision - 2):
                return self.row(self.text("−", level, color) if value.imag < 0 else B.Space(0), im_b, i_b)
            return self.row(re_b, self.op("−" if value.imag < 0 else "+", level, 0.18, color), im_b, i_b)
        if isinstance(value, str):
            return self.text(f"“{value}”", level, "math_string", family=self.s.text_family)
        if isinstance(value, RangeValue):
            return self.value(value.to_array(), level, color)
        if isinstance(value, np.ndarray):
            return self.array(value, level, color)
        if isinstance(value, UserFunction):
            return self.text(f"{value.name}({', '.join(value.params)})", level, "math_function", italic=True)
        if isinstance(value, BuiltinRef):
            return self.text(value.name, level, "math_function")
        if isinstance(value, SeriesGroup):
            return self.text(f"‹{len(value)} plot series›", level, "text_muted", family=self.s.text_family)
        if hasattr(value, "series_kind"):
            return self.text(f"‹{value.series_kind} series›", level, "text_muted", family=self.s.text_family)
        if isinstance(value, (Record, Table)) or isinstance(value, Displayable):
            return self.display(value.display(), level)
        if isinstance(value, DisplayNode):
            return self.display(value, level)
        if isinstance(value, (list, tuple)):
            return self.array(np.array(value, dtype=object), level, color)
        if value is None:
            return self.text("—", level, "text_muted")
        return self.text(str(value), level, color, family=self.s.text_family)

    def number(self, x: float, level: int, color: str) -> B.Box:
        mant, exp = split_number(x, self.fmt)
        mant = mant.replace("-", "−")
        if exp is None:
            return self.text(mant, level, color)
        base = self.text(f"{mant}·10", level, color)
        return B.Scripts(base, self.text(str(exp).replace("-", "−"), level + 1, color), None, self.size(level))

    def array(self, arr: np.ndarray, level: int, color: str) -> B.Box:
        if arr.ndim == 0:
            return self.value(arr.item(), level, color)
        if arr.size == 0:
            return self.text("[ ]", level, color)
        mat = arr.reshape(-1, 1) if arr.ndim == 1 else arr
        if mat.ndim > 2:
            return self.text(f"array {arr.shape}", level, color)
        rows_idx = self._truncate(mat.shape[0], self.s.max_rows)
        cols_idx = self._truncate(mat.shape[1], self.s.max_cols)
        cells = []
        for i in rows_idx:
            row = []
            for j in cols_idx:
                if i is None:
                    row.append(self.text("⋮", level, color))
                elif j is None:
                    row.append(self.text("⋯", level, color))
                else:
                    row.append(self.value(mat[i, j], level, color))
            cells.append(row)
        return B.Matrix(self.m, cells, self.font(level), self.s.c(color), self.axis(level), align="right")

    @staticmethod
    def _truncate(n: int, limit: int) -> list[Optional[int]]:
        if n <= limit:
            return list(range(n))
        head = limit - 2
        return list(range(head)) + [None, n - 1]

    def display(self, node: DisplayNode, level: int = 0) -> B.Box:
        if isinstance(node, DText):
            role = node.role
            if role == "title":
                return self.text(node.text, level, "heading", bold=True, family=self.s.text_family)
            if role == "error":
                return self.text(node.text, level + 1, "math_error", family=self.s.text_family)
            return self.text(node.text, level + 1, "text_body", family=self.s.text_family)
        if isinstance(node, DValue):
            return self.value(node.value, level)
        if isinstance(node, DPair):
            return self.row(self.name(node.label, level), self.op("=", level, 0.25), self.value(node.value, level))
        if isinstance(node, DTable):
            cell_font = self.font(1, family=self.s.text_family).scaled(1.12)
            def cell(v):
                if isinstance(v, str):
                    return B.Text(self.m, v, cell_font, self.s.c("text_body"))
                return B.Text(self.m, _plain_number(v, self.fmt), cell_font, self.s.c("math_text"))
            headers = [B.Text(self.m, h, cell_font.with_(bold=True), self.s.c("heading")) for h in node.headers]
            rows = [[cell(v) for v in r] for r in node.rows]
            title = B.Text(self.m, node.title, cell_font.with_(bold=True), self.s.c("heading")) if node.title else None
            return B.TableBox(self.m, headers, rows, cell_font, self.s.colors, title, node.highlight)
        if isinstance(node, DStack):
            return B.VStack([self.display(i, level) for i in node.items], self.size(level) * 0.45)
        return self.text(str(node), level)

    # ---------------------------------------------------------- statements
    def statement(self, stmt: A.Stmt, result=None) -> B.Box:
        lvl = 0
        box: B.Box
        if isinstance(stmt, A.Assign):
            box = self.row(self.name(stmt.name, lvl), self.op("≔", lvl, 0.25), self.expr(stmt.value, lvl))
        elif isinstance(stmt, A.FuncDef):
            params = B.HBox(sum(([self.text(",", lvl, "math_operator"), B.Space(self.size(lvl) * 0.2), self.name(p, lvl)]
                                 for p in stmt.params), [])[2:] if stmt.params else [])
            head = self.row(self.name(stmt.name, lvl), self.delim(params, "(", ")", lvl))
            box = self.row(head, self.op("≔", lvl, 0.25), self.expr(stmt.body, lvl))
        elif isinstance(stmt, A.IndexAssign):
            head = B.Scripts(self.name(stmt.name, lvl), None, self.args_box(stmt.indices, (), lvl + 1), self.size(lvl))
            box = self.row(head, self.op("≔", lvl, 0.25), self.expr(stmt.value, lvl))
        elif isinstance(stmt, A.Destructure):
            cells = [[self.name(n, lvl)] for n in stmt.names]
            head = B.Matrix(self.m, cells, self.font(lvl), self.s.c("math_operator"), self.axis(lvl))
            box = self.row(head, self.op("≔", lvl, 0.25), self.expr(stmt.value, lvl))
        elif isinstance(stmt, A.Evaluate):
            left = self.expr(stmt.expr, lvl)
            if result is not None and result.error is None:
                res = B.Framed(self.value(result.value, lvl), self.s.c("math_result_bg"), None, 3.0, 4.0, 1.0)
                box = self.row(left, self.op("=", lvl, 0.25), res)
            else:
                box = self.row(left, self.op("=", lvl, 0.25), self.text("?", lvl, "math_error"))
        elif isinstance(stmt, A.Symbolic):
            left = self.expr(stmt.expr, lvl)
            cmds = None
            if stmt.commands:
                parts = []
                for cmd in stmt.commands:
                    label = cmd.name
                    if cmd.args or cmd.kwargs:
                        parts.append(self.row(self.text(label, lvl + 1, "math_keyword"), self.text(",", lvl + 1, "math_operator"),
                                              B.Space(2), self.args_box(cmd.args, cmd.kwargs, lvl + 1)))
                    else:
                        parts.append(self.text(label, lvl + 1, "math_keyword"))
                cmds = B.VStack(parts, 1.0, "center")
            arrow = self.text("→", lvl, "math_operator")
            if result is not None and result.error is None and result.symbolic_ast is not None:
                right = self.expr(result.symbolic_ast, lvl)
                right = B.Framed(right, self.s.c("math_result_bg"), None, 3.0, 4.0, 1.0)
            else:
                right = self.text("?", lvl, "math_error")
            sym = B.BigOperator(arrow, None, cmds, right, self.size(lvl) * 1.2, (arrow.a - arrow.d) / 2) if cmds else \
                self.row(self.op("→", lvl, 0.3), right)
            box = self.row(left, B.Space(self.size(lvl) * 0.3), sym) if cmds else self.row(left, sym)
        elif isinstance(stmt, A.Given):
            box = self.text("Given", lvl, "math_keyword", bold=True, family=self.s.text_family)
        elif isinstance(stmt, A.ExprStmt):
            if isinstance(stmt.expr, A.Compare) and result is not None and result.kind == "constraint":
                box = self._Compare(stmt.expr, lvl, constraint=True)
            else:
                box = self.expr(stmt.expr, lvl)
        else:
            box = self.text(type(stmt).__name__, lvl, "math_error")
        if result is not None and result.error is not None:
            message = self.text(f"⚠ {result.error}", 1, "math_error", family=self.s.text_family)
            box = B.VStack([B.Underlined(box, self.s.c("math_error")), message], 2.0)
        return box

    def statements(self, stmts, results=None) -> B.Box:
        lines = []
        for i, st in enumerate(stmts):
            res = results[i] if results is not None and i < len(results) else None
            lines.append(self.statement(st, res))
        return B.VStack(lines, self.size(0) * 0.55)

    def source_with_error(self, source: str, message: str) -> B.Box:
        mono = B.FontSpec(self.s.mono_family, self.size(0) * 0.8)
        lines = [B.Text(self.m, line or " ", mono, self.s.c("math_text")) for line in source.split("\n")]
        code = B.Underlined(B.VStack(lines, 2.0), self.s.c("math_error"))
        return B.VStack([code, self.text(f"⚠ {message}", 1, "math_error", family=self.s.text_family)], 3.0)


def _plain_number(v: Any, fmt: NumberFormat) -> str:
    from ..engine.formatting import format_value
    v = normalize_number(v)
    if isinstance(v, (int, float, complex)):
        return format_value(v, fmt).replace("-", "−")
    if v is None or v == "":
        return ""
    return format_value(v, fmt)
