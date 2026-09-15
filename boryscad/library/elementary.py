"""Elementary functions and constants."""
from __future__ import annotations

import math

import numpy as np
from scipy import special

from ..engine.values import EvalError, is_number, normalize_number
from ._util import to_float, to_int, vec


def _num(x):
    return np.asarray(x) if isinstance(x, np.ndarray) else x


def _real_or_complex(fn, cfn=None):
    """Use the real numpy function, switching to complex when the domain requires it."""

    def wrapper(x):
        arr = np.asarray(x)
        with np.errstate(invalid="ignore", divide="ignore"):
            if np.iscomplexobj(arr):
                res = (cfn or fn)(arr)
            else:
                res = fn(arr.astype(float))
                if np.any(np.isnan(res)) and not np.any(np.isnan(arr.astype(float))) and cfn is not None:
                    res = cfn(arr.astype(complex))
        return res.item() if np.ndim(res) == 0 else res

    return wrapper


def register(r):
    cat = "Elementary"
    simple = {
        "sin": (np.sin, "sine (radians)"), "cos": (np.cos, "cosine"), "tan": (np.tan, "tangent"),
        "sinh": (np.sinh, "hyperbolic sine"), "cosh": (np.cosh, "hyperbolic cosine"), "tanh": (np.tanh, "hyperbolic tangent"),
        "atan": (np.arctan, "inverse tangent"), "asinh": (np.arcsinh, "inverse hyperbolic sine"),
        "exp": (np.exp, "exponential eˣ"), "floor": (np.floor, "largest integer ≤ x"), "ceil": (np.ceil, "smallest integer ≥ x"),
        "trunc": (np.trunc, "integer part"), "sign": (np.sign, "sign of x"), "erf": (special.erf, "error function"),
        "gamma": (special.gamma, "Gamma function"), "re": (np.real, "real part"), "im": (np.imag, "imaginary part"),
        "conj": (np.conj, "complex conjugate"), "arg": (np.angle, "argument (angle) of a complex number"),
        "deg": (np.degrees, "radians → degrees"), "rad": (np.radians, "degrees → radians"),
    }
    for name, (fn, doc) in simple.items():
        def make(f, fname=name):
            def impl(x):
                res = f(_num(x))
                if fname in ("floor", "ceil", "trunc") and np.ndim(res) == 0 and np.isfinite(res):
                    return int(res)
                return res
            return impl
        r.function(name, category=cat, signature=f"{name}(x)", doc=doc)(make(fn))

    r.function("asin", category=cat, signature="asin(x)", doc="inverse sine")(_real_or_complex(np.arcsin, np.emath.arcsin))
    r.function("acos", category=cat, signature="acos(x)", doc="inverse cosine")(_real_or_complex(np.arccos, np.emath.arccos))
    r.function("acosh", category=cat, signature="acosh(x)", doc="inverse hyperbolic cosine")(_real_or_complex(np.arccosh, np.emath.arccos))
    r.function("atanh", category=cat, signature="atanh(x)", doc="inverse hyperbolic tangent")(_real_or_complex(np.arctanh, np.emath.arctanh))
    r.function("ln", category=cat, signature="ln(x)", doc="natural logarithm")(_real_or_complex(np.log, np.emath.log))
    r.function("sqrt", category=cat, signature="sqrt(x)", doc="square root")(_real_or_complex(np.sqrt, np.emath.sqrt))
    r.function("lg", category=cat, signature="lg(x)", doc="decimal logarithm")(_real_or_complex(np.log10, np.emath.log10))
    r.function("log2", category=cat, signature="log2(x)", doc="binary logarithm")(_real_or_complex(np.log2, np.emath.log2))

    @r.function(category=cat, signature="log(x, base=10)", doc="logarithm (decimal by default)")
    def log(x, base=10):
        return np.emath.log(_num(x)) / np.log(base) if not (is_number(x) and x > 0) else math.log(x, base)

    @r.function(category=cat, signature="cot(x)", doc="cotangent")
    def cot(x):
        return 1 / np.tan(_num(x))

    @r.function(category=cat, signature="sec(x)", doc="secant")
    def sec(x):
        return 1 / np.cos(_num(x))

    @r.function(category=cat, signature="csc(x)", doc="cosecant")
    def csc(x):
        return 1 / np.sin(_num(x))

    @r.function(category=cat, signature="acot(x)", doc="inverse cotangent")
    def acot(x):
        return np.pi / 2 - np.arctan(_num(x))

    @r.function(category=cat, signature="atan2(y, x)", doc="angle of the point (x, y)")
    def atan2(y, x):
        return np.arctan2(_num(y), _num(x))

    @r.function(category=cat, signature="nthroot(x, n)", doc="real n-th root")
    def nthroot(x, n):
        x = _num(x)
        n = to_float(n, "n")
        if n.is_integer() and int(n) % 2 == 1:
            return np.sign(x) * np.abs(x) ** (1.0 / n)
        return np.emath.power(x, 1.0 / n)

    @r.function(category=cat, signature="cbrt(x)", doc="cube root")
    def cbrt(x):
        return np.cbrt(_num(x))

    @r.function("abs", category=cat, signature="abs(x)", doc="absolute value")
    def abs_(x):
        return np.abs(_num(x))

    @r.function(category=cat, signature="round(x, n=0)", doc="round to n decimals", aliases=("ROUND",))
    def round_(x, n=0):
        n = to_int(n, "n")
        res = np.round(_num(x), n)
        if n <= 0:
            if np.ndim(res) == 0:
                return int(res)
            return res.astype(np.int64)
        return res

    @r.function(category=cat, signature="mod(a, b)", doc="remainder of a / b")
    def mod(a, b):
        if is_number(b) and b == 0:
            raise EvalError("mod by zero")
        return np.mod(_num(a), _num(b))

    @r.function(category=cat, signature="factorial(n)", doc="n!")
    def factorial(n):
        if is_number(n):
            n = to_float(n)
            if n.is_integer() and 0 <= n <= 170:
                return math.factorial(int(n))
        return special.factorial(_num(n), exact=False)

    @r.function(category=cat, signature="comb(n, k)", doc="binomial coefficient C(n, k)", aliases=("binomial",))
    def comb(n, k):
        if is_number(n) and is_number(k):
            return math.comb(to_int(n), to_int(k))
        return special.comb(_num(n), _num(k))

    @r.function(category=cat, signature="perm(n, k)", doc="number of permutations P(n, k)")
    def perm(n, k):
        return math.perm(to_int(n), to_int(k))

    @r.function(category=cat, signature="gcd(a, b, …)", doc="greatest common divisor")
    def gcd(*values):
        items = [int(v) for x in values for v in (np.asarray(x).reshape(-1) if isinstance(x, np.ndarray) else [x])]
        return math.gcd(*items)

    @r.function(category=cat, signature="lcm(a, b, …)", doc="least common multiple")
    def lcm(*values):
        items = [int(v) for x in values for v in (np.asarray(x).reshape(-1) if isinstance(x, np.ndarray) else [x])]
        return math.lcm(*items)

    @r.function("max", category=cat, signature="max(a, b, …)", doc="largest value of numbers, vectors or matrices")
    def max_(*values):
        items = _flatten(values)
        return normalize_number(np.max(items))

    @r.function("min", category=cat, signature="min(a, b, …)", doc="smallest value")
    def min_(*values):
        items = _flatten(values)
        return normalize_number(np.min(items))

    @r.function(category=cat, signature="hypot(a, b)", doc="√(a² + b²)")
    def hypot(a, b):
        return np.hypot(_num(a), _num(b))

    @r.function(category=cat, signature="heaviside(x)", doc="unit step Φ(x)", aliases=("Φ",))
    def heaviside(x):
        return np.heaviside(_num(x), 1.0)

    @r.function(category=cat, signature="complex(re, im)", doc="build a complex number")
    def complex_(re, im):
        return complex(to_float(re), to_float(im))

    @r.function(category=cat, signature="isprime(n)", doc="1 when n is prime")
    def isprime(n):
        n = to_int(n)
        if n < 2:
            return 0
        return int(all(n % p for p in range(2, int(math.isqrt(n)) + 1)))

    r.constant("π", math.pi, "Archimedes' constant")
    r.constant("pi", math.pi, "Archimedes' constant")
    r.constant("e", math.e, "Euler's number")
    r.constant("inf", math.inf, "infinity")
    r.constant("NaN", math.nan, "not a number")
    r.constant("g", 9.80665, "standard gravity, m/s²")
    r.constant("TRUE", 1, "logical true")
    r.constant("FALSE", 0, "logical false")


def _flatten(values):
    items = []
    for v in values:
        if isinstance(v, np.ndarray):
            items.extend(v.reshape(-1).tolist())
        else:
            items.append(v)
    if not items:
        raise EvalError("no values given")
    return np.array(items)


__all__ = ["register", "vec"]
