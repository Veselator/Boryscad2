"""String functions."""
from __future__ import annotations

import numpy as np

from ..engine.formatting import format_value
from ..engine.values import EvalError, normalize_number
from ._util import to_int


def register(r):
    cat = "Strings"

    @r.function(category=cat, signature="concat(s1, s2, …)", doc="join strings")
    def concat(*parts):
        return "".join(p if isinstance(p, str) else format_value(p) for p in parts)

    @r.function(category=cat, signature="str(x)", doc="convert a value to text", aliases=("num2str",))
    def str_(x):
        return x if isinstance(x, str) else format_value(x)

    @r.function(category=cat, signature="num(s)", doc="convert text to a number", aliases=("str2num",))
    def num(s):
        try:
            v = float(str(s).replace(",", "."))
        except ValueError:
            raise EvalError(f"cannot convert {s!r} to a number") from None
        return int(v) if v.is_integer() else v

    @r.function(category=cat, signature="strlen(s)", doc="length of a string")
    def strlen(s):
        return len(str(s))

    @r.function(category=cat, signature="substr(s, start, n)", doc="substring (0-based)")
    def substr(s, start, n):
        start, n = to_int(start), to_int(n)
        return str(s)[start:start + n]

    @r.function(category=cat, signature="upper(s)", doc="upper case")
    def upper(s):
        return str(s).upper()

    @r.function(category=cat, signature="lower(s)", doc="lower case")
    def lower(s):
        return str(s).lower()

    @r.function(category=cat, signature="labels(prefix, n, start=1)", doc="vector of names like S1, S2, …")
    def labels(prefix, n, start=1):
        return np.array([f"{prefix}{i}" for i in range(to_int(start), to_int(start) + to_int(n))], dtype=object)
