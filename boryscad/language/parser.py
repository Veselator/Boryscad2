"""Lark based parser producing :mod:`boryscad.language.ast` nodes."""
from __future__ import annotations

from functools import lru_cache
from importlib import resources
from typing import Any

from lark import Lark, Token, Transformer, v_args
from lark.exceptions import UnexpectedCharacters, UnexpectedEOF, UnexpectedInput, UnexpectedToken, VisitError

from . import ast as A

_OP_NORMAL = {
    "−": "-", "·": "*", "×": "*", "÷": "/", "**": "^",
    "≠": "!=", "≤": "<=", "≥": ">=", "∨": "or", "∧": "and", "¬": "not",
}


class ParseError(Exception):
    """Syntax error with position information (0-based offset into the source)."""

    def __init__(self, message: str, pos: int | None = None, line: int | None = None, column: int | None = None):
        super().__init__(message)
        self.message = message
        self.pos = pos
        self.line = line
        self.column = column

    def __str__(self) -> str:
        if self.line is not None:
            return f"{self.message} (line {self.line}, col {self.column})"
        return self.message


def _span(meta) -> A.Span:
    if meta is None or getattr(meta, "empty", True):
        return None
    return (meta.start_pos, meta.end_pos)


def _norm(op: Token | str) -> str:
    s = str(op)
    return _OP_NORMAL.get(s, s)


@v_args(meta=True)
class _ToAst(Transformer):
    # ------------------------------------------------------------ statements
    def start(self, meta, items):
        return [i for i in items if isinstance(i, A.Node)]

    def assign(self, meta, items):
        lhs, _tok, rhs = items
        return self._make_assign(lhs, rhs, meta)

    def step_range_assign(self, meta, items):
        lhs, _tok, first, rest = items
        rest = A.strip_parens(rest)
        if not isinstance(rest, A.RangeExpr) or rest.step is not None:
            raise ParseError("expected a range 'first, second .. last'", _span(meta)[0] if _span(meta) else None)
        rng = A.RangeExpr(first, rest.stop, None, rest.start, span=_span(meta))
        return self._make_assign(lhs, rng, meta)

    def _make_assign(self, lhs, rhs, meta):
        span = _span(meta)
        lhs = A.strip_parens(lhs)
        if isinstance(lhs, A.Var):
            return A.Assign(lhs.name, rhs, span=span)
        if isinstance(lhs, A.Call):
            params = []
            for arg in lhs.args:
                if not isinstance(arg, A.Var):
                    raise ParseError("function parameters must be plain names", span[0] if span else None)
                params.append(arg.name)
            if lhs.kwargs:
                raise ParseError("keyword arguments are not allowed in a definition", span[0] if span else None)
            return A.FuncDef(lhs.name, tuple(params), rhs, span=span)
        if isinstance(lhs, A.Index) and isinstance(A.strip_parens(lhs.target), A.Var):
            return A.IndexAssign(A.strip_parens(lhs.target).name, lhs.indices, rhs, span=span)
        if isinstance(lhs, A.MatrixLit):
            names = []
            for row in lhs.rows:
                for cell in row:
                    if not isinstance(cell, A.Var):
                        raise ParseError("destructuring target must contain names only", span[0] if span else None)
                    names.append(cell.name)
            return A.Destructure(tuple(names), rhs, span=span)
        raise ParseError("invalid left-hand side of ':='", span[0] if span else None)

    def evaluate(self, meta, items):
        return A.Evaluate(items[0], span=_span(meta))

    def equation(self, meta, items):
        lhs, _eq, rhs = items
        return A.ExprStmt(A.Compare((lhs, rhs), ("==",), span=_span(meta)), span=_span(meta))

    def symbolic(self, meta, items):
        expr = items[0]
        cmds = tuple(i for i in items[1:] if isinstance(i, A.SymCommand))
        return A.Symbolic(expr, cmds, span=_span(meta))

    def symcmd(self, meta, items):
        name = str(items[0])
        args, kwargs = self._split_args(items[1] if len(items) > 1 else None)
        return A.SymCommand(name, args, kwargs, span=_span(meta))

    def given(self, meta, items):
        return A.Given(span=_span(meta))

    def expr_stmt(self, meta, items):
        return A.ExprStmt(items[0], span=_span(meta))

    # -------------------------------------------------------------- programs
    def program(self, meta, items):
        body = []
        for item in items:
            if isinstance(item, A.PStmt):
                body.append(item)
            elif isinstance(item, A.Expr):
                body.append(A.PExpr(item, span=item.span))
        return A.Program(tuple(body), span=_span(meta))

    def local_assign(self, meta, items):
        target, _arrow, value = items
        target = A.strip_parens(target)
        if not isinstance(target, (A.Var, A.Index)):
            raise ParseError("invalid target of '<-'", _span(meta)[0] if _span(meta) else None)
        return A.LocalAssign(target, value, span=_span(meta))

    def for_stmt(self, meta, items):
        name, iterable, body = items
        return A.ForStmt(str(name), iterable, body, span=_span(meta))

    def while_stmt(self, meta, items):
        cond, body = items
        return A.WhileStmt(cond, body, span=_span(meta))

    def if_stmt(self, meta, items):
        cond, body = items[0], items[1]
        orelse = items[2] if len(items) > 2 else None
        return A.IfStmt(cond, body, orelse, span=_span(meta))

    def return_stmt(self, meta, items):
        return A.ReturnStmt(items[0], span=_span(meta))

    def break_stmt(self, meta, items):
        return A.BreakStmt(span=_span(meta))

    def continue_stmt(self, meta, items):
        return A.ContinueStmt(span=_span(meta))

    def pexpr(self, meta, items):
        return A.PExpr(items[0], span=_span(meta))

    # ----------------------------------------------------------- expressions
    def binop(self, meta, items):
        left, op, right = items
        return A.BinOp(_norm(op), left, right, span=_span(meta))

    def unop(self, meta, items):
        op, operand = items
        op = _norm(op)
        if op == "+":
            return operand
        return A.UnOp(op, operand, span=_span(meta))

    def sqrt_prefix(self, meta, items):
        return A.Sqrt(items[1], span=_span(meta))

    def compare(self, meta, items):
        operands = tuple(items[0::2])
        ops = tuple(_norm(o) for o in items[1::2])
        return A.Compare(operands, ops, span=_span(meta))

    def range(self, meta, items):
        start, _r, stop = items[0], items[1], items[2]
        step = items[4] if len(items) > 4 and items[4] is not None else None
        return A.RangeExpr(start, stop, step, span=_span(meta))

    def factorial(self, meta, items):
        return A.Factorial(items[0], span=_span(meta))

    def index(self, meta, items):
        target, args = items
        pos, kw = self._split_args(args)
        return A.Index(target, pos, span=_span(meta))

    def member(self, meta, items):
        target, name = items
        return A.Member(target, str(name), span=_span(meta))

    def number(self, meta, items):
        return A.Number(str(items[0]), span=_span(meta))

    def imag(self, meta, items):
        return A.Imag(str(items[0])[:-1], span=_span(meta))

    def var(self, meta, items):
        return A.Var(str(items[0]), span=_span(meta))

    def inf(self, meta, items):
        return A.Infinity(span=_span(meta))

    def string(self, meta, items):
        return A.String(str(items[0])[1:-1], span=_span(meta))

    def call(self, meta, items):
        name = str(items[0])
        args, kwargs = self._split_args(items[1] if len(items) > 1 else None)
        return A.Call(name, args, kwargs, span=_span(meta))

    def paren(self, meta, items):
        return A.Paren(items[0], span=_span(meta))

    def matrix(self, meta, items):
        rows = items[0] if items and items[0] is not None else []
        return A.MatrixLit(tuple(tuple(r) for r in rows), span=_span(meta))

    def matrix_rows(self, meta, items):
        rows = list(items)
        widths = {len(r) for r in rows}
        if len(widths) > 1:
            raise ParseError("all matrix rows must have the same number of elements", _span(meta)[0] if _span(meta) else None)
        return rows

    def mrow(self, meta, items):
        return list(items)

    def abs(self, meta, items):
        return A.Abs(items[0], span=_span(meta))

    def if_expr(self, meta, items):
        return A.IfExpr(items[0], items[1], items[2], span=_span(meta))

    def args(self, meta, items):
        return list(items)

    def kwarg(self, meta, items):
        return A.Keyword(str(items[0]), items[-1], span=_span(meta))

    @staticmethod
    def _split_args(args: Any):
        if args is None:
            return (), ()
        pos = tuple(a for a in args if not isinstance(a, A.Keyword))
        kw = tuple(a for a in args if isinstance(a, A.Keyword))
        return pos, kw


@lru_cache(maxsize=1)
def _lark() -> Lark:
    grammar = resources.files("boryscad.language").joinpath("grammar.lark").read_text(encoding="utf-8")
    return Lark(
        grammar,
        parser="lalr",
        start=["start", "expr"],
        propagate_positions=True,
        maybe_placeholders=True,
    )


_TOKEN_NAMES = {
    "ASSIGN": "':='", "EQ": "'='", "RANGE": "'..'", "ADDOP": "'+' or '-'", "MULOP": "'*' or '/'",
    "POWOP": "'^'", "NAME": "a name", "NUMBER": "a number", "RPAR": "')'", "LPAR": "'('",
    "RSQB": "']'", "LSQB": "'['", "COMMA": "','", "_NL": "end of line", "$END": "end of input",
    "COMPOP": "a comparison", "ARROW": "'->'", "RBRACE": "'}'", "LBRACE": "'{'", "SEMICOLON": "';'",
    "VBAR": "'|'", "STRING": "a string", "LARROW": "'<-'",
}


def _friendly(err: UnexpectedInput, text: str) -> ParseError:
    pos = getattr(err, "pos_in_stream", None)
    line = getattr(err, "line", None)
    col = getattr(err, "column", None)
    if isinstance(err, UnexpectedToken):
        tok = err.token
        got = "end of input" if tok.type == "$END" else repr(str(tok))
        expected = sorted({_TOKEN_NAMES.get(t, t) for t in err.expected if not t.startswith("__")})
        exp = ", ".join(expected[:6])
        msg = f"unexpected {got}" + (f"; expected {exp}" if exp else "")
        pos = tok.start_pos if tok.start_pos is not None else len(text)
    elif isinstance(err, UnexpectedCharacters):
        msg = f"unexpected character {text[err.pos_in_stream]!r}"
    elif isinstance(err, UnexpectedEOF):
        msg = "unexpected end of input"
        pos = len(text)
    else:  # pragma: no cover
        msg = str(err)
    return ParseError(msg, pos, line, col)


def parse(text: str) -> list[A.Stmt]:
    """Parse a region (one or more statements)."""
    try:
        return _ToAst().transform(_lark().parse(text, start="start"))
    except VisitError as exc:
        if isinstance(exc.orig_exc, ParseError):
            raise exc.orig_exc from None
        raise
    except UnexpectedInput as exc:
        raise _friendly(exc, text) from None


def parse_expr(text: str) -> A.Expr:
    """Parse a single expression."""
    try:
        return _ToAst().transform(_lark().parse(text, start="expr"))
    except VisitError as exc:
        if isinstance(exc.orig_exc, ParseError):
            raise exc.orig_exc from None
        raise
    except UnexpectedInput as exc:
        raise _friendly(exc, text) from None


@lru_cache(maxsize=4096)
def parse_cached(text: str) -> tuple[A.Stmt, ...]:
    return tuple(parse(text))
