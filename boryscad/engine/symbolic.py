"""Bridge between the language AST and SymPy (symbolic ``->`` evaluation)."""
from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
import sympy as sp
from sympy.printing.str import StrPrinter

from ..language import ast as A
from .environment import Environment
from .values import EvalError, RangeValue, UserFunction, is_number

GREEK_CONSTANTS = {"π": sp.pi, "pi": sp.pi, "e": sp.E, "∞": sp.oo, "inf": sp.oo}

SYMPY_FUNCTIONS: dict[str, Any] = {
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "cot": sp.cot, "sec": sp.sec, "csc": sp.csc,
    "asin": sp.asin, "acos": sp.acos, "atan": sp.atan, "acot": sp.acot, "atan2": sp.atan2,
    "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh, "asinh": sp.asinh, "acosh": sp.acosh, "atanh": sp.atanh,
    "exp": sp.exp, "ln": sp.log, "log": lambda x, b=10: sp.log(x, b), "lg": lambda x: sp.log(x, 10),
    "log2": lambda x: sp.log(x, 2), "sqrt": sp.sqrt, "cbrt": sp.cbrt, "nthroot": lambda x, n: sp.root(x, n),
    "abs": sp.Abs, "sign": sp.sign, "floor": sp.floor, "ceil": sp.ceiling, "gamma": sp.gamma,
    "factorial": sp.factorial, "re": sp.re, "im": sp.im, "conj": sp.conjugate, "arg": sp.arg,
    "max": sp.Max, "min": sp.Min, "binomial": sp.binomial, "comb": sp.binomial, "erf": sp.erf,
    "mod": sp.Mod, "heaviside": sp.Heaviside,
}


class SymbolicTranslator:
    """Translate AST into SymPy.

    Variables that have a numeric value in the environment are substituted,
    unless they are listed in ``keep`` (these become symbols).
    """

    def __init__(self, env: Environment, keep: Iterable[str] = (), substitute: bool = True, evaluator=None):
        self.env = env
        self.keep = set(keep)
        self.substitute = substitute
        self.evaluator = evaluator
        self._symbols: dict[str, sp.Symbol] = {}
        self._depth = 0

    def symbol(self, name: str) -> sp.Symbol:
        if name not in self._symbols:
            self._symbols[name] = sp.Symbol(name)
        return self._symbols[name]

    def to_sympy(self, node: A.Node, bindings: dict[str, Any] | None = None) -> Any:
        bindings = bindings or {}
        method = getattr(self, "_" + type(node).__name__, None)
        if method is None:
            raise EvalError(f"symbolic evaluation does not support {type(node).__name__}", node.span)
        return method(node, bindings)

    # ---------------------------------------------------------------- nodes
    def _Number(self, node: A.Number, b):
        return sp.Rational(node.text) if "." in node.text or "e" in node.text.lower() else sp.Integer(node.text)

    def _Imag(self, node: A.Imag, b):
        return sp.Rational(node.text) * sp.I

    def _Infinity(self, node, b):
        return sp.oo

    def _String(self, node, b):
        raise EvalError("strings are not supported in symbolic expressions", node.span)

    def _Var(self, node: A.Var, b):
        name = node.name
        if name in b:
            return b[name]
        if name in self.keep:
            return self.symbol(name)
        if self.substitute and self.env.has(name):
            value = self.env.lookup(name)
            converted = value_to_sympy(value)
            if converted is not None:
                return converted
        if name in GREEK_CONSTANTS and not self.env.has(name):
            return GREEK_CONSTANTS[name]
        return self.symbol(name)

    def _Paren(self, node: A.Paren, b):
        return self.to_sympy(node.body, b)

    def _BinOp(self, node: A.BinOp, b):
        l = self.to_sympy(node.left, b)
        if node.op == "^" and isinstance(node.right, A.Var) and node.right.name == "T" and isinstance(l, sp.MatrixBase):
            return l.T
        r = self.to_sympy(node.right, b)
        op = node.op
        if op == "+":
            return l + r
        if op == "-":
            return l - r
        if op in ("*", ".*"):
            if isinstance(l, sp.MatrixBase) and isinstance(r, sp.MatrixBase) and op == ".*":
                return l.multiply_elementwise(r)
            if isinstance(l, sp.MatrixBase) and isinstance(r, sp.MatrixBase) and l.shape == r.shape and l.shape[1] == 1:
                return (l.T * r)[0]
            return l * r
        if op in ("/", "./"):
            if isinstance(r, sp.MatrixBase):
                return l * r.inv()
            return l / r
        if op in ("^", ".^"):
            return l ** r
        if op == "and":
            return sp.And(l, r)
        if op == "or":
            return sp.Or(l, r)
        raise EvalError(f"operator {op} is not supported symbolically", node.span)

    def _UnOp(self, node: A.UnOp, b):
        v = self.to_sympy(node.operand, b)
        if node.op == "-":
            return -v
        if node.op == "not":
            return sp.Not(v)
        raise EvalError(f"operator {node.op} is not supported symbolically", node.span)

    def _Compare(self, node: A.Compare, b):
        vals = [self.to_sympy(o, b) for o in node.operands]
        rels = []
        for op, l, r in zip(node.ops, vals, vals[1:]):
            rels.append({"==": sp.Eq, "!=": sp.Ne, "<": sp.Lt, "<=": sp.Le, ">": sp.Gt, ">=": sp.Ge}[op](l, r, evaluate=False))
        return rels[0] if len(rels) == 1 else sp.And(*rels)

    def _Sqrt(self, node, b):
        return sp.sqrt(self.to_sympy(node.operand, b))

    def _Abs(self, node, b):
        v = self.to_sympy(node.operand, b)
        if isinstance(v, sp.MatrixBase):
            return v.det() if v.shape[0] == v.shape[1] and v.shape[1] > 1 else v.norm()
        return sp.Abs(v)

    def _Factorial(self, node, b):
        return sp.factorial(self.to_sympy(node.operand, b))

    def _MatrixLit(self, node: A.MatrixLit, b):
        rows = [[self.to_sympy(c, b) for c in row] for row in node.rows]
        if len(rows) == 1:
            return sp.Matrix(rows[0])
        return sp.Matrix(rows)

    def _IfExpr(self, node: A.IfExpr, b):
        cond = self.to_sympy(node.cond, b)
        return sp.Piecewise((self.to_sympy(node.then, b), cond), (self.to_sympy(node.otherwise, b), True))

    def _Index(self, node: A.Index, b):
        target = self.to_sympy(node.target, b)
        idx = [self.to_sympy(i, b) for i in node.indices]
        origin = int(self.env.get("ORIGIN", 0) or 0)
        if isinstance(target, sp.MatrixBase) and all(i.is_Integer for i in idx):
            ii = [int(i) - origin for i in idx]
            return target[ii[0]] if len(ii) == 1 else target[ii[0], ii[1]]
        if isinstance(target, sp.Symbol):
            return sp.Indexed(sp.IndexedBase(target.name), *idx)
        raise EvalError("unsupported symbolic indexing", node.span)

    def _Member(self, node, b):
        if self.evaluator is not None:
            return value_to_sympy(self.evaluator.eval(node, self.env))
        raise EvalError("fields are not supported symbolically", node.span)

    def _RangeExpr(self, node, b):
        raise EvalError("ranges are not supported symbolically", node.span)

    def _Program(self, node, b):
        raise EvalError("programs are not supported symbolically", node.span)

    def _Call(self, node: A.Call, b):
        name = node.name
        self._depth += 1
        try:
            if self._depth > 60:
                raise EvalError("symbolic expansion is too deep (recursive function?)", node.span)
            fn = self.env.get(name) if self.env.has(name) else None
            if isinstance(fn, UserFunction) and not isinstance(fn.body, A.Program):
                args = [self.to_sympy(a, b) for a in node.args]
                if len(args) != len(fn.params):
                    raise EvalError(f"{name} expects {len(fn.params)} argument(s)", node.span)
                inner = dict(zip(fn.params, args))
                return self.to_sympy(fn.body, inner)
            special = getattr(self, "_call_" + name, None)
            if special is not None:
                return special(node, b)
            if name in SYMPY_FUNCTIONS:
                args = [self.to_sympy(a, b) for a in node.args]
                return SYMPY_FUNCTIONS[name](*args)
            if self.evaluator is not None and not (fn is None and not self.evaluator.registry.get(name)):
                # fall back to numeric evaluation when all arguments are numeric
                value = self.evaluator.eval(node, self.env.child(
                    {k: sympy_to_value(v) for k, v in b.items() if _is_numeric_sym(v)}))
                converted = value_to_sympy(value)
                if converted is not None:
                    return converted
            args = [self.to_sympy(a, b) for a in node.args]
            return sp.Function(name)(*args)
        finally:
            self._depth -= 1

    # --- calculus operators with symbolic meaning
    def _var_name(self, node: A.Expr) -> str:
        node = A.strip_parens(node)
        if not isinstance(node, A.Var):
            raise EvalError("a variable name is expected", node.span)
        return node.name

    def _inner(self, b, names):
        inner = dict(b)
        for n in names:
            inner[n] = self.symbol(n)
        return inner

    def _call_deriv(self, node: A.Call, b):
        var = self._var_name(node.args[1])
        order = int(self.to_sympy(node.args[2], b)) if len(node.args) > 2 else 1
        expr = self.to_sympy(node.args[0], self._inner(b, [var]))
        return sp.diff(expr, self.symbol(var), order)

    _call_diff = _call_deriv

    def _call_integral(self, node: A.Call, b):
        var = self._var_name(node.args[1])
        expr = self.to_sympy(node.args[0], self._inner(b, [var]))
        if len(node.args) >= 4:
            lo, hi = self.to_sympy(node.args[2], b), self.to_sympy(node.args[3], b)
            return sp.integrate(expr, (self.symbol(var), lo, hi))
        return sp.integrate(expr, self.symbol(var))

    _call_integrate = _call_integral

    def _call_sum(self, node: A.Call, b):
        if len(node.args) == 1:
            v = self.to_sympy(node.args[0], b)
            return sum(v) if isinstance(v, sp.MatrixBase) else v
        var = self._var_name(node.args[1])
        expr = self.to_sympy(node.args[0], self._inner(b, [var]))
        return sp.summation(expr, (self.symbol(var), self.to_sympy(node.args[2], b), self.to_sympy(node.args[3], b)))

    def _call_prod(self, node: A.Call, b):
        if len(node.args) == 1:
            v = self.to_sympy(node.args[0], b)
            return sp.prod(list(v)) if isinstance(v, sp.MatrixBase) else v
        var = self._var_name(node.args[1])
        expr = self.to_sympy(node.args[0], self._inner(b, [var]))
        return sp.product(expr, (self.symbol(var), self.to_sympy(node.args[2], b), self.to_sympy(node.args[3], b)))

    def _call_limit(self, node: A.Call, b):
        var = self._var_name(node.args[1])
        expr = self.to_sympy(node.args[0], self._inner(b, [var]))
        point = self.to_sympy(node.args[2], b)
        direction = "+-"
        for kw in node.kwargs:
            if kw.name == "dir":
                direction = self.evaluator.eval(kw.value, self.env) if self.evaluator else "+-"
        return sp.limit(expr, self.symbol(var), point, dir=direction)

    def _call_transpose(self, node, b):
        return self.to_sympy(node.args[0], b).T

    def _call_inverse(self, node, b):
        return self.to_sympy(node.args[0], b).inv()

    def _call_det(self, node, b):
        return self.to_sympy(node.args[0], b).det()


def _is_numeric_sym(v: Any) -> bool:
    return isinstance(v, sp.Basic) and v.is_number


def value_to_sympy(value: Any) -> Any:
    if isinstance(value, bool):
        return sp.Integer(int(value))
    if isinstance(value, (int, np.integer)):
        return sp.Integer(int(value))
    if isinstance(value, (float, np.floating)):
        f = float(value)
        if math.isinf(f):
            return sp.oo if f > 0 else -sp.oo
        if f.is_integer() and abs(f) < 1e15:
            return sp.Integer(int(f))
        return sp.nsimplify(f, rational=False, tolerance=1e-15) if len(repr(f)) <= 6 else sp.Float(f)
    if isinstance(value, (complex, np.complexfloating)):
        return value_to_sympy(value.real) + sp.I * value_to_sympy(value.imag)
    if isinstance(value, np.ndarray) and value.dtype != object and value.ndim in (1, 2):
        return sp.Matrix([[value_to_sympy(v) for v in row] for row in (value.reshape(-1, 1) if value.ndim == 1 else value)])
    if isinstance(value, RangeValue):
        return None
    return None


def sympy_to_value(expr: Any) -> Any:
    """Convert a numeric SymPy result into a Python/numpy value."""
    if isinstance(expr, sp.MatrixBase):
        rows = [[sympy_to_value(expr[i, j]) for j in range(expr.shape[1])] for i in range(expr.shape[0])]
        arr = np.array(rows)
        return arr.reshape(-1) if expr.shape[1] == 1 else arr
    if isinstance(expr, (list, tuple)):
        return np.array([sympy_to_value(e) for e in expr])
    if isinstance(expr, sp.Basic):
        if expr.is_Integer:
            return int(expr)
        if expr.is_number:
            c = complex(expr.evalf())
            return c.real if abs(c.imag) < 1e-14 else c
        raise EvalError(f"result is not numeric: {expr}")
    return expr


# ------------------------------------------------------------------ printing
class BoryscadPrinter(StrPrinter):
    """Prints SymPy expressions using the Boryscad language syntax."""

    def _print_Pow(self, expr, rational=False):
        base, exp = expr.as_base_exp()
        if exp == sp.Rational(1, 2):
            return f"sqrt({self._print(base)})"
        if exp == -sp.Rational(1, 2):
            return f"1/sqrt({self._print(base)})"
        if exp.is_Rational and exp.q != 1 and exp.p == 1:
            return f"nthroot({self._print(base)}, {exp.q})"
        if exp == -1:
            return f"1/{self.parenthesize(base, sp.printing.precedence.PRECEDENCE['Mul'] + 1)}"
        prec = sp.printing.precedence.precedence(expr)
        b = self.parenthesize(base, prec, strict=True)
        e = self.parenthesize(exp, prec, strict=False)
        if exp.is_Rational and not exp.is_Integer or (exp.is_Integer and exp < 0) or exp.is_Add or exp.is_Mul:
            e = f"({self._print(exp)})"
        return f"{b}^{e}"

    def _print_Exp1(self, expr):
        return "e"

    def _print_Pi(self, expr):
        return "π"

    def _print_ImaginaryUnit(self, expr):
        return "1i"

    def _print_Infinity(self, expr):
        return "∞"

    def _print_NegativeInfinity(self, expr):
        return "-∞"

    def _print_ComplexInfinity(self, expr):
        return "∞"

    def _print_log(self, expr):
        if len(expr.args) == 1:
            return f"ln({self._print(expr.args[0])})"
        return f"log({self._print(expr.args[0])}, {self._print(expr.args[1])})"

    def _print_exp(self, expr):
        return f"exp({self._print(expr.args[0])})"

    def _print_Abs(self, expr):
        return f"|{self._print(expr.args[0])}|"

    def _print_MatrixBase(self, expr):
        if expr.shape[1] == 1:
            return "[" + "; ".join(self._print(expr[i, 0]) for i in range(expr.shape[0])) + "]"
        rows = ["; ".join([]) for _ in range(0)]
        rows = [", ".join(self._print(expr[i, j]) for j in range(expr.shape[1])) for i in range(expr.shape[0])]
        return "[" + "; ".join(rows) + "]"

    _print_ImmutableDenseMatrix = _print_MatrixBase
    _print_MutableDenseMatrix = _print_MatrixBase
    _print_ImmutableMatrix = _print_MatrixBase
    _print_Matrix = _print_MatrixBase

    def _print_Relational(self, expr):
        op = {"==": "==", "!=": "!=", "<": "<", "<=": "<=", ">": ">", ">=": ">="}.get(expr.rel_op, expr.rel_op)
        return f"{self._print(expr.lhs)} {op} {self._print(expr.rhs)}"

    def _print_Equality(self, expr):
        return f"{self._print(expr.lhs)} == {self._print(expr.rhs)}"

    def _print_And(self, expr):
        return " and ".join(self.parenthesize(a, 1) for a in expr.args)

    def _print_Or(self, expr):
        return " or ".join(self.parenthesize(a, 1) for a in expr.args)

    def _print_Piecewise(self, expr):
        parts = list(expr.args)
        if len(parts) == 2 and parts[1][1] == True:  # noqa: E712
            return f"if({self._print(parts[0][1])}, {self._print(parts[0][0])}, {self._print(parts[1][0])})"
        return f"piecewise({', '.join(self._print(e) + ', ' + self._print(c) for e, c in parts)})"

    def _print_Float(self, expr):
        f = float(expr)
        if f.is_integer() and abs(f) < 1e15:
            return str(int(f))
        s = f"{f:.10g}"
        if "e" in s:
            mant, exp = s.split("e")
            return f"{mant}*10^({int(exp)})"
        return s

    def _print_Rational(self, expr):
        return f"{expr.p}/{expr.q}"

    def _print_Integer(self, expr):
        return str(expr.p)

    def _print_Symbol(self, expr):
        return expr.name

    def _print_Indexed(self, expr):
        return f"{self._print(expr.base.label)}[{', '.join(self._print(i) for i in expr.indices)}]"

    def _print_Derivative(self, expr):
        inner = self._print(expr.expr)
        for var, count in expr.variable_count:
            inner = f"deriv({inner}, {self._print(var)}{', ' + str(count) if count != 1 else ''})"
        return inner

    def _print_Integral(self, expr):
        inner = self._print(expr.function)
        for lim in expr.limits:
            if len(lim) == 3:
                inner = f"integral({inner}, {self._print(lim[0])}, {self._print(lim[1])}, {self._print(lim[2])})"
            else:
                inner = f"integral({inner}, {self._print(lim[0])})"
        return inner

    def _print_Sum(self, expr):
        inner = self._print(expr.function)
        for v, a, b in expr.limits:
            inner = f"sum({inner}, {self._print(v)}, {self._print(a)}, {self._print(b)})"
        return inner

    def _print_ceiling(self, expr):
        return f"ceil({self._print(expr.args[0])})"

    def _print_conjugate(self, expr):
        return f"conj({self._print(expr.args[0])})"

    def _print_Mod(self, expr):
        return f"mod({self._print(expr.args[0])}, {self._print(expr.args[1])})"

    def _print_Tuple(self, expr):
        return "[" + "; ".join(self._print(a) for a in expr.args) + "]"

    def _print_list(self, expr):
        return "[" + "; ".join(self._print(a) for a in expr) + "]"

    _print_tuple = _print_list

    def _print_dict(self, expr):
        return "[" + "; ".join(f"{self._print(k)} == {self._print(v)}" for k, v in expr.items()) + "]"

    def _print_EmptySet(self, expr):
        return '"no solution"'

    def _print_FiniteSet(self, expr):
        return "[" + "; ".join(self._print(a) for a in expr.args) + "]"

    def _print_BooleanTrue(self, expr):
        return "1"

    def _print_BooleanFalse(self, expr):
        return "0"

    def _print_Order(self, expr):
        return f"O({self._print(expr.expr)})"


_printer = BoryscadPrinter({"order": None})


def sympy_to_source(expr: Any) -> str:
    return _printer.doprint(expr)
