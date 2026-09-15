"""Exact simplex tableau (fractions) with two-phase, dual simplex and step recording."""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Optional, Sequence

import numpy as np

from ..engine.values import EvalError, Table


def frac(value) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, (int, np.integer)):
        return Fraction(int(value))
    f = float(value)
    return Fraction(f).limit_denominator(10 ** 9) if not f.is_integer() else Fraction(int(f))


def fmt_frac(q: Fraction, as_fraction: bool = True) -> str:
    if q.denominator == 1:
        return str(q.numerator)
    if as_fraction and q.denominator <= 999:
        return f"{q.numerator}/{q.denominator}"
    return f"{float(q):.4g}"


def parse_signs(signs, m: int) -> list[str]:
    if signs is None:
        return ["<="] * m
    if isinstance(signs, str):
        items = [s.strip() for s in signs.replace(";", ",").split(",") if s.strip()]
    else:
        items = [str(s).strip() for s in np.asarray(signs, dtype=object).reshape(-1)]
    norm = {"<=": "<=", "≤": "<=", "<": "<=", ">=": ">=", "≥": ">=", ">": ">=", "=": "=", "==": "="}
    out = []
    for s in items:
        if s not in norm:
            raise EvalError(f"unknown constraint sign {s!r}; use \"<=\", \">=\" or \"=\"")
        out.append(norm[s])
    if len(out) == 1 and m > 1:
        out = out * m
    if len(out) != m:
        raise EvalError(f"{len(out)} sign(s) given for {m} constraint(s)")
    return out


@dataclass
class Step:
    title: str
    table: Table
    note: str = ""


@dataclass
class Tableau:
    names: list[str]
    A: list[list[Fraction]]
    b: list[Fraction]
    basis: list[int]
    cost: list[Fraction]
    artificial: set[int] = field(default_factory=set)
    steps: list[Step] = field(default_factory=list)
    record: bool = True
    as_fraction: bool = True

    # ------------------------------------------------------------ building
    @classmethod
    def standard(cls, c: Sequence, A: Sequence[Sequence], b: Sequence, signs: Sequence[str],
                 var_names: Sequence[str] | None = None) -> "Tableau":
        m, n = len(A), len(c)
        rows = [[frac(v) for v in row] for row in A]
        rhs = [frac(v) for v in b]
        signs = list(signs)
        for i in range(m):
            if rhs[i] < 0:
                rows[i] = [-v for v in rows[i]]
                rhs[i] = -rhs[i]
                signs[i] = {"<=": ">=", ">=": "<=", "=": "="}[signs[i]]
        names = list(var_names) if var_names else [f"x{j + 1}" for j in range(n)]
        cost = [frac(v) for v in c]
        extra_cols: list[list[Fraction]] = []
        basis = [-1] * m
        artificial: set[int] = set()
        slack_no = 0
        for i, s in enumerate(signs):
            if s in ("<=", ">="):
                slack_no += 1
                col = [Fraction(0)] * m
                col[i] = Fraction(1 if s == "<=" else -1)
                extra_cols.append(col)
                names.append(f"s{slack_no}")
                cost.append(Fraction(0))
                if s == "<=":
                    basis[i] = len(names) - 1
        art_no = 0
        for i, s in enumerate(signs):
            if s in (">=", "="):
                art_no += 1
                col = [Fraction(0)] * m
                col[i] = Fraction(1)
                extra_cols.append(col)
                names.append(f"a{art_no}")
                cost.append(Fraction(0))
                basis[i] = len(names) - 1
                artificial.add(len(names) - 1)
        full = [rows[i] + [col[i] for col in extra_cols] for i in range(m)]
        return cls(names, full, rhs, basis, cost, artificial)

    # ------------------------------------------------------------ algebra
    @property
    def m(self) -> int:
        return len(self.A)

    @property
    def n(self) -> int:
        return len(self.names)

    def deltas(self, cost: Sequence[Fraction]) -> tuple[list[Fraction], Fraction]:
        d = []
        for j in range(self.n):
            z = sum((cost[self.basis[i]] * self.A[i][j] for i in range(self.m)), Fraction(0))
            d.append(z - cost[j])
        value = sum((cost[self.basis[i]] * self.b[i] for i in range(self.m)), Fraction(0))
        return d, value

    def pivot(self, r: int, c: int) -> None:
        p = self.A[r][c]
        self.A[r] = [v / p for v in self.A[r]]
        self.b[r] = self.b[r] / p
        for i in range(self.m):
            if i != r and self.A[i][c] != 0:
                factor = self.A[i][c]
                self.A[i] = [a - factor * pr for a, pr in zip(self.A[i], self.A[r])]
                self.b[i] = self.b[i] - factor * self.b[r]
        self.basis[r] = c

    def snapshot(self, title: str, cost: Sequence[Fraction], pivot: Optional[tuple[int, int]] = None,
                 theta: Optional[list[Optional[Fraction]]] = None, note: str = "") -> None:
        if not self.record:
            return
        d, value = self.deltas(cost)
        headers = ["Basis", "c_b", "b"] + self.names + (["θ"] if theta is not None else [])
        rows = []
        for i in range(self.m):
            row = [self.names[self.basis[i]], fmt_frac(cost[self.basis[i]], self.as_fraction),
                   fmt_frac(self.b[i], self.as_fraction)]
            row += [fmt_frac(v, self.as_fraction) for v in self.A[i]]
            if theta is not None:
                row.append("—" if theta[i] is None else fmt_frac(theta[i], self.as_fraction))
            rows.append(row)
        rows.append(["Δ", "", fmt_frac(value, self.as_fraction)] + [fmt_frac(v, self.as_fraction) for v in d]
                    + ([""] if theta is not None else []))
        highlight = set()
        if pivot is not None:
            highlight.add((pivot[0], 3 + pivot[1]))
        self.steps.append(Step(title, Table(headers, rows, title, highlight), note))

    # ------------------------------------------------------------ methods
    def primal(self, cost: Sequence[Fraction], phase: str, allowed: Optional[set[int]] = None, max_iter: int = 500) -> str:
        for it in range(max_iter):
            d, _ = self.deltas(cost)
            candidates = [j for j in range(self.n) if d[j] < 0 and (allowed is None or j in allowed)]
            if not candidates:
                self.snapshot(f"{phase}: optimal tableau", cost)
                return "optimal"
            most = min(d[j] for j in candidates)
            c = min(j for j in candidates if d[j] == most)
            theta: list[Optional[Fraction]] = []
            for i in range(self.m):
                theta.append(self.b[i] / self.A[i][c] if self.A[i][c] > 0 else None)
            valid = [(t, self.basis[i], i) for i, t in enumerate(theta) if t is not None]
            if not valid:
                self.snapshot(f"{phase}: iteration {it + 1} (unbounded)", cost, theta=theta)
                return "unbounded"
            r = min(valid)[2]
            self.snapshot(f"{phase}: iteration {it + 1} — {self.names[c]} enters, {self.names[self.basis[r]]} leaves",
                          cost, (r, c), theta)
            self.pivot(r, c)
        raise EvalError("simplex iteration limit reached")

    def dual(self, cost: Sequence[Fraction], phase: str, max_iter: int = 500) -> str:
        for it in range(max_iter):
            negative = [i for i in range(self.m) if self.b[i] < 0]
            if not negative:
                return "optimal"
            most = min(self.b[i] for i in negative)
            r = min(i for i in negative if self.b[i] == most)
            d, _ = self.deltas(cost)
            ratios = [(abs(d[j] / self.A[r][j]), j) for j in range(self.n) if self.A[r][j] < 0]
            if not ratios:
                self.snapshot(f"{phase}: row {self.names[self.basis[r]]} has no negative element — infeasible", cost)
                return "infeasible"
            c = min(ratios)[1]
            self.snapshot(f"{phase}: dual simplex step {it + 1} — {self.names[self.basis[r]]} leaves, {self.names[c]} enters",
                          cost, (r, c))
            self.pivot(r, c)
        raise EvalError("dual simplex iteration limit reached")

    def solve(self) -> str:
        """Two-phase simplex for maximisation of ``cost``."""
        if self.artificial:
            phase1 = [Fraction(-1) if j in self.artificial else Fraction(0) for j in range(self.n)]
            status = self.primal(phase1, "Phase I")
            _, value = self.deltas(phase1)
            if value < 0:
                return "infeasible"
            for i in range(self.m):
                if self.basis[i] in self.artificial:
                    col = next((j for j in range(self.n) if j not in self.artificial and self.A[i][j] != 0), None)
                    if col is not None:
                        self.pivot(i, col)
            keep_rows = [i for i in range(self.m) if self.basis[i] not in self.artificial]
            self.A = [self.A[i] for i in keep_rows]
            self.b = [self.b[i] for i in keep_rows]
            self.basis = [self.basis[i] for i in keep_rows]
            self.remove_columns(sorted(self.artificial))
        return self.primal(self.cost, "Simplex")

    def remove_columns(self, cols: list[int]) -> None:
        keep = [j for j in range(self.n) if j not in set(cols)]
        remap = {old: new for new, old in enumerate(keep)}
        self.names = [self.names[j] for j in keep]
        self.cost = [self.cost[j] for j in keep]
        self.A = [[row[j] for j in keep] for row in self.A]
        self.basis = [remap[b] for b in self.basis]
        self.artificial = set()

    def add_column(self, name: str, cost: Fraction = Fraction(0)) -> int:
        self.names.append(name)
        self.cost.append(cost)
        for row in self.A:
            row.append(Fraction(0))
        return self.n - 1

    def add_row(self, coeffs: list[Fraction], rhs: Fraction, basic_col: int) -> None:
        self.A.append(coeffs)
        self.b.append(rhs)
        self.basis.append(basic_col)

    def values(self, count: int) -> list[Fraction]:
        x = [Fraction(0)] * count
        for i, bi in enumerate(self.basis):
            if bi < count:
                x[bi] = self.b[i]
        return x

    def objective(self) -> Fraction:
        return self.deltas(self.cost)[1]
