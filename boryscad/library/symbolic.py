"""Keywords of the symbolic evaluation operator ``->``."""
from __future__ import annotations

import sympy as sp

from ..engine.values import EvalError


def _symbols(expr, args):
    syms = [a for a in args if isinstance(a, sp.Symbol)]
    if not syms:
        free = sorted(getattr(expr, "free_symbols", set()), key=lambda s: s.name)
        if len(free) == 1:
            return free
        if not free:
            return []
        raise EvalError("specify the variable, e.g. -> solve(x)")
    return syms


def register(r):
    cmd = r.symbolic_command

    @cmd("simplify", "simplify the expression")
    def simplify(expr, *a, **k):
        return sp.simplify(expr)

    @cmd("expand", "expand products and powers")
    def expand(expr, *a, **k):
        return sp.expand(expr)

    @cmd("factor", "factor polynomials")
    def factor(expr, *a, **k):
        return sp.factor(expr)

    @cmd("collect", "collect terms in powers of a variable")
    def collect(expr, *args, **k):
        return sp.collect(sp.expand(expr), args[0]) if args else expr

    @cmd("cancel", "cancel common factors of a fraction")
    def cancel(expr, *a, **k):
        return sp.cancel(expr)

    @cmd("together", "combine into a single fraction")
    def together(expr, *a, **k):
        return sp.together(expr)

    @cmd("partfrac", "partial fraction decomposition")
    def partfrac(expr, *args, **k):
        return sp.apart(expr, *[x for x in args if isinstance(x, sp.Symbol)])

    @cmd("trigsimp", "simplify trigonometric expressions")
    def trigsimp(expr, *a, **k):
        return sp.trigsimp(expr)

    @cmd("float", "evaluate numerically with n significant digits")
    def float_(expr, *args, **k):
        n = int(args[0]) if args else 10
        return expr.evalf(n) if hasattr(expr, "evalf") else expr

    @cmd("solve", "solve an equation (or expression = 0) for a variable")
    def solve(expr, *args, **k):
        syms = _symbols(expr, args)
        if isinstance(expr, sp.And):
            solutions = sp.solve(list(expr.args), syms, dict=True)
        elif isinstance(expr, sp.MatrixBase):
            solutions = sp.solve(list(expr), syms, dict=True)
        else:
            solutions = sp.solve(expr, syms, dict=True)
        if not solutions:
            return sp.EmptySet
        if len(syms) == 1:
            vals = [s[syms[0]] for s in solutions if syms[0] in s]
            return vals[0] if len(vals) == 1 else sp.Matrix(vals)
        rows = [[s.get(v, v) for v in syms] for s in solutions]
        return sp.Matrix(rows[0]) if len(rows) == 1 else sp.Matrix(rows)

    @cmd("substitute", "substitute values: -> substitute(x = 2)")
    def substitute(expr, *args, substitutions=None, **k):
        subs = dict(substitutions or {})
        for a in args:
            if isinstance(a, sp.Equality):
                subs[a.lhs] = a.rhs
        return expr.subs(subs)

    @cmd("series", "Taylor series: -> series(x, x0, n)")
    def series(expr, *args, **k):
        syms = _symbols(expr, args[:1])
        x = syms[0]
        x0 = args[1] if len(args) > 1 else 0
        n = int(args[2]) if len(args) > 2 else 6
        return sp.series(expr, x, x0, n).removeO()

    @cmd("diff", "derivative: -> diff(x)")
    def diff(expr, *args, **k):
        syms = _symbols(expr, args[:1])
        n = int(args[1]) if len(args) > 1 else 1
        return sp.diff(expr, syms[0], n)

    @cmd("integrate", "antiderivative: -> integrate(x)")
    def integrate(expr, *args, **k):
        syms = _symbols(expr, args[:1])
        if len(args) >= 3:
            return sp.integrate(expr, (syms[0], args[1], args[2]))
        return sp.integrate(expr, syms[0])

    @cmd("limit", "limit: -> limit(x, a)")
    def limit(expr, *args, **k):
        syms = _symbols(expr, args[:1])
        return sp.limit(expr, syms[0], args[1] if len(args) > 1 else 0)

    @cmd("coeffs", "polynomial coefficients in ascending order")
    def coeffs(expr, *args, **k):
        syms = _symbols(expr, args[:1])
        poly = sp.Poly(sp.expand(expr), syms[0])
        return sp.Matrix(list(reversed(poly.all_coeffs())))

    @cmd("numer", "numerator")
    def numer(expr, *a, **k):
        return sp.fraction(sp.together(expr))[0]

    @cmd("denom", "denominator")
    def denom(expr, *a, **k):
        return sp.fraction(sp.together(expr))[1]

    @cmd("laplace", "Laplace transform: -> laplace(t, s)")
    def laplace(expr, *args, **k):
        t, s = args[0], args[1] if len(args) > 1 else sp.Symbol("s")
        return sp.laplace_transform(expr, t, s, noconds=True)

    @cmd("invlaplace", "inverse Laplace transform: -> invlaplace(s, t)")
    def invlaplace(expr, *args, **k):
        s, t = args[0], args[1] if len(args) > 1 else sp.Symbol("t")
        return sp.inverse_laplace_transform(expr, s, t)

    @cmd("rewrite", "rewrite in terms of a function, e.g. -> rewrite(\"exp\")")
    def rewrite(expr, *args, **k):
        target = args[0] if args else "exp"
        fn = getattr(sp, str(target), None)
        return expr.rewrite(fn if fn is not None else str(target))
