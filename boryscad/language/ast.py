"""Abstract syntax tree of the Boryscad math language.

Nodes are immutable dataclasses.  Every node keeps an optional source span
``(start, end)`` so that errors can be highlighted in the editor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Optional, Sequence

Span = Optional[tuple[int, int]]


@dataclass(frozen=True)
class Node:
    span: Span = field(default=None, compare=False, repr=False, kw_only=True)

    def children(self) -> Iterator["Node"]:
        for value in self.__dict__.values():
            if isinstance(value, Node):
                yield value
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Node):
                        yield item
                    elif isinstance(item, (list, tuple)):
                        yield from (x for x in item if isinstance(x, Node))

    def walk(self) -> Iterator["Node"]:
        yield self
        for child in self.children():
            yield from child.walk()


# ----------------------------------------------------------------- expressions
class Expr(Node):
    pass


@dataclass(frozen=True)
class Number(Expr):
    text: str

    @property
    def value(self) -> int | float:
        if any(c in self.text for c in ".eE"):
            return float(self.text)
        return int(self.text)


@dataclass(frozen=True)
class Imag(Expr):
    text: str  # without the trailing i/j

    @property
    def value(self) -> complex:
        return complex(0, float(self.text))


@dataclass(frozen=True)
class Infinity(Expr):
    pass


@dataclass(frozen=True)
class String(Expr):
    value: str


@dataclass(frozen=True)
class Var(Expr):
    name: str


@dataclass(frozen=True)
class Paren(Expr):
    body: Expr


@dataclass(frozen=True)
class BinOp(Expr):
    op: str
    left: Expr
    right: Expr


@dataclass(frozen=True)
class UnOp(Expr):
    op: str
    operand: Expr


@dataclass(frozen=True)
class Compare(Expr):
    operands: tuple[Expr, ...]
    ops: tuple[str, ...]


@dataclass(frozen=True)
class RangeExpr(Expr):
    start: Expr
    stop: Expr
    step: Optional[Expr] = None
    # Mathcad style "a, a2 .. b": second element instead of a step
    second: Optional[Expr] = None


@dataclass(frozen=True)
class Keyword(Node):
    name: str
    value: Expr


@dataclass(frozen=True)
class Call(Expr):
    name: str
    args: tuple[Expr, ...] = ()
    kwargs: tuple[Keyword, ...] = ()


@dataclass(frozen=True)
class Index(Expr):
    target: Expr
    indices: tuple[Expr, ...]


@dataclass(frozen=True)
class Member(Expr):
    target: Expr
    name: str


@dataclass(frozen=True)
class Factorial(Expr):
    operand: Expr


@dataclass(frozen=True)
class Abs(Expr):
    operand: Expr


@dataclass(frozen=True)
class Sqrt(Expr):
    operand: Expr


@dataclass(frozen=True)
class MatrixLit(Expr):
    rows: tuple[tuple[Expr, ...], ...]


@dataclass(frozen=True)
class IfExpr(Expr):
    cond: Expr
    then: Expr
    otherwise: Expr


# -------------------------------------------------------------------- programs
class PStmt(Node):
    pass


@dataclass(frozen=True)
class LocalAssign(PStmt):
    target: Expr  # Var or Index
    value: Expr


@dataclass(frozen=True)
class ForStmt(PStmt):
    var: str
    iterable: Expr
    body: "Program"


@dataclass(frozen=True)
class WhileStmt(PStmt):
    cond: Expr
    body: "Program"


@dataclass(frozen=True)
class IfStmt(PStmt):
    cond: Expr
    body: "Program"
    orelse: Optional[Node] = None  # Program | IfStmt


@dataclass(frozen=True)
class ReturnStmt(PStmt):
    value: Expr


@dataclass(frozen=True)
class BreakStmt(PStmt):
    pass


@dataclass(frozen=True)
class ContinueStmt(PStmt):
    pass


@dataclass(frozen=True)
class PExpr(PStmt):
    expr: Expr


@dataclass(frozen=True)
class Program(Expr):
    body: tuple[PStmt, ...]


# ------------------------------------------------------------------ statements
class Stmt(Node):
    pass


@dataclass(frozen=True)
class Assign(Stmt):
    name: str
    value: Expr


@dataclass(frozen=True)
class FuncDef(Stmt):
    name: str
    params: tuple[str, ...]
    body: Expr


@dataclass(frozen=True)
class IndexAssign(Stmt):
    name: str
    indices: tuple[Expr, ...]
    value: Expr


@dataclass(frozen=True)
class Destructure(Stmt):
    names: tuple[str, ...]
    value: Expr


@dataclass(frozen=True)
class Evaluate(Stmt):
    expr: Expr


@dataclass(frozen=True)
class SymCommand(Node):
    name: str
    args: tuple[Expr, ...] = ()
    kwargs: tuple[Keyword, ...] = ()


@dataclass(frozen=True)
class Symbolic(Stmt):
    expr: Expr
    commands: tuple[SymCommand, ...] = ()


@dataclass(frozen=True)
class Given(Stmt):
    pass


@dataclass(frozen=True)
class ExprStmt(Stmt):
    expr: Expr


def free_names(node: Node) -> set[str]:
    """Names of variables referenced anywhere inside ``node``."""
    return {n.name for n in node.walk() if isinstance(n, Var)}


def called_names(node: Node) -> set[str]:
    return {n.name for n in node.walk() if isinstance(n, Call)}


def strip_parens(expr: Expr) -> Expr:
    while isinstance(expr, Paren):
        expr = expr.body
    return expr


__all__: Sequence[str] = [name for name in dir() if not name.startswith("_")]
