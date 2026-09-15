"""Runtime value types of the worksheet engine.

Plain Python numbers, ``numpy.ndarray`` and ``str`` are used directly; the
classes below cover everything else.  Rich values implement the
:class:`Displayable` protocol so renderers can show them without knowing
their concrete type (dependency inversion).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Protocol, Sequence, runtime_checkable

import numpy as np


class EvalError(Exception):
    """Error raised while evaluating an expression."""

    def __init__(self, message: str, span: tuple[int, int] | None = None):
        super().__init__(message)
        self.message = message
        self.span = span

    def __str__(self) -> str:
        return self.message


# ------------------------------------------------------------------ display
@runtime_checkable
class Displayable(Protocol):
    def display(self) -> "DisplayNode": ...


class DisplayNode:
    """Backend-independent description of a rich result."""


@dataclass
class DText(DisplayNode):
    text: str
    role: str = "text"  # text | title | note | error


@dataclass
class DValue(DisplayNode):
    value: Any


@dataclass
class DTable(DisplayNode):
    headers: list[str]
    rows: list[list[Any]]
    title: str = ""
    highlight: set[tuple[int, int]] = field(default_factory=set)


@dataclass
class DStack(DisplayNode):
    items: list[DisplayNode]


@dataclass
class DPair(DisplayNode):
    label: str
    value: Any


# ------------------------------------------------------------------- values
@dataclass(frozen=True)
class RangeValue:
    """Mathcad style range variable ``a..b`` or ``a, a2 .. b``."""

    start: float
    stop: float
    step: float

    def __post_init__(self):
        if self.step == 0:
            raise EvalError("range step must not be zero")
        if (self.stop - self.start) * self.step < 0:
            raise EvalError("range step points away from the end value")
        if self.count > 5_000_000:
            raise EvalError("range is too large")

    @property
    def count(self) -> int:
        return int(math.floor((self.stop - self.start) / self.step + 1e-9)) + 1

    def __iter__(self) -> Iterator[float]:
        for i in range(self.count):
            v = self.start + i * self.step
            yield int(v) if float(v).is_integer() and isinstance(self.start, int) and isinstance(self.step, int) else v

    def to_array(self) -> np.ndarray:
        arr = self.start + np.arange(self.count) * self.step
        if all(isinstance(v, (int, np.integer)) for v in (self.start, self.step)):
            return arr.astype(np.int64)
        return arr.astype(float)

    def __len__(self) -> int:
        return self.count


@dataclass
class UserFunction:
    name: str
    params: tuple[str, ...]
    body: Any  # ast.Expr
    closure: Any  # Environment

    def __repr__(self) -> str:
        return f"{self.name}({', '.join(self.params)})"


@dataclass
class BuiltinRef:
    """A reference to a registered builtin used as a value (e.g. passed to a function)."""

    name: str

    def __repr__(self) -> str:
        return self.name


class Record:
    """A structured result with named fields, accessible via ``r.field``."""

    def __init__(self, title: str = "", fields: dict[str, Any] | None = None,
                 details: Sequence[DisplayNode] = (), summary: Sequence[str] | None = None):
        self.title = title
        self.fields: dict[str, Any] = dict(fields or {})
        self.details = list(details)
        self.summary = list(summary) if summary is not None else list(self.fields)

    def get(self, name: str) -> Any:
        if name not in self.fields:
            raise EvalError(f"'{self.title or 'record'}' has no field '{name}'. Fields: {', '.join(self.fields)}")
        return self.fields[name]

    def display(self) -> DisplayNode:
        items: list[DisplayNode] = []
        if self.title:
            items.append(DText(self.title, "title"))
        for key in self.summary:
            if key in self.fields:
                items.append(DPair(key, self.fields[key]))
        items.extend(self.details)
        return DStack(items)

    def __repr__(self) -> str:
        return f"Record({self.title!r}, {list(self.fields)})"


class Table:
    def __init__(self, headers: Sequence[str], rows: Sequence[Sequence[Any]], title: str = "",
                 highlight: Iterable[tuple[int, int]] = ()):
        self.headers = [str(h) for h in headers]
        self.rows = [list(r) for r in rows]
        self.title = title
        self.highlight = set(highlight)

    def display(self) -> DisplayNode:
        return DTable(self.headers, self.rows, self.title, self.highlight)

    def column(self, name: str) -> np.ndarray:
        idx = self.headers.index(name)
        return np.array([r[idx] for r in self.rows], dtype=object)


class SeriesGroup(list):
    """A list of plot series returned by helper functions such as ``lp_graph``."""


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float, complex, np.number, bool, np.bool_)) and not isinstance(value, str)


def is_array(value: Any) -> bool:
    return isinstance(value, np.ndarray)


def to_float(value: Any, what: str = "value") -> float:
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, (complex, np.complexfloating)):
        if abs(value.imag) > 1e-12:
            raise EvalError(f"{what} must be real, got {value}")
        return float(value.real)
    if isinstance(value, np.ndarray) and value.size == 1:
        return to_float(value.reshape(-1)[0], what)
    raise EvalError(f"{what} must be a number")


def to_int(value: Any, what: str = "value") -> int:
    f = to_float(value, what)
    if abs(f - round(f)) > 1e-9:
        raise EvalError(f"{what} must be an integer, got {f:g}")
    return int(round(f))


def to_array(value: Any, what: str = "value") -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, RangeValue):
        return value.to_array()
    if isinstance(value, (list, tuple)):
        return np.array(value)
    if is_number(value):
        return np.array([value])
    raise EvalError(f"{what} must be a vector or matrix")


def normalize_number(value: Any) -> Any:
    """Convert numpy scalars to Python scalars and drop negligible imaginary parts."""
    if isinstance(value, np.bool_):
        return int(value)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, (complex, np.complexfloating)):
        c = complex(value)
        if c.imag == 0:
            return c.real
        return c
    return value


def type_name(value: Any) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "boolean"
    if isinstance(value, (int, np.integer)):
        return "integer"
    if isinstance(value, (float, np.floating)):
        return "real"
    if isinstance(value, (complex, np.complexfloating)):
        return "complex"
    if isinstance(value, np.ndarray):
        return "vector" if value.ndim == 1 else f"matrix {value.shape[0]}×{value.shape[1]}" if value.ndim == 2 else "array"
    if isinstance(value, str):
        return "string"
    if isinstance(value, RangeValue):
        return "range"
    if isinstance(value, (UserFunction, BuiltinRef)):
        return "function"
    if isinstance(value, Record):
        return "record"
    if isinstance(value, Table):
        return "table"
    return type(value).__name__


Callback = Callable[..., Any]
