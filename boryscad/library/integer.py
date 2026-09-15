"""Integer programming: Gomory's cutting-plane method with all tableaux."""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
from scipy import optimize

from ..engine.values import DText, EvalError, Record
from ._tableau import Tableau, fmt_frac, parse_signs
from ._util import clean, clean_array, strings, to_int, vec


def _frac_part(q: Fraction) -> Fraction:
    return q - math.floor(q)


def register(r):
    cat = "Operations research · Integer"

    @r.function(category=cat, signature='gomory(c, A, b, signs="<=", sense="max", names=, max_cuts=30)',
                doc="integer linear programming by Gomory's cutting-plane method (x ≥ 0, integer)")
    def gomory(c, A, b, signs=None, sense="max", names=None, max_cuts=30, fractions=1):
        from .optimization import _lp_input, _names, _sense
        c, A, b, signs = _lp_input(c, A, b, signs)
        sense = _sense(sense)
        n = c.size
        var_names = _names(n, names)
        tab = Tableau.standard(c if sense == "max" else -c, A, b, signs, var_names)
        tab.as_fraction = bool(fractions)
        status = tab.solve()
        if status == "infeasible":
            raise EvalError("the linear relaxation is infeasible")
        if status == "unbounded":
            raise EvalError("the linear relaxation is unbounded")
        sign = 1 if sense == "max" else -1
        x_lp = [tab.values(n)[j] for j in range(n)]
        F_lp = tab.objective() * sign
        details = [DText("Step 1. Linear relaxation (simplex method without integrality)", "title")]
        details += [s.table.display() for s in tab.steps]
        details.append(DText("LP optimum: " + ", ".join(f"{var_names[j]} = {fmt_frac(x_lp[j])}" for j in range(n)) +
                             f";  F = {fmt_frac(F_lp)}", "note"))
        recorded = len(tab.steps)
        cuts = []
        limit = to_int(max_cuts)
        for k in range(limit + 1):
            fractional = [(_frac_part(tab.b[i]), -i, i) for i, bi in enumerate(tab.basis)
                          if bi < n and tab.b[i].denominator != 1]
            if not fractional:
                break
            if k == limit:
                raise EvalError(f"no integer solution after {limit} cuts")
            f0, _, row_i = max(fractional)
            row = list(tab.A[row_i])
            coeffs = [_frac_part(a) for a in row]
            terms = " + ".join(f"{fmt_frac(f)}·{tab.names[j]}" for j, f in enumerate(coeffs) if f != 0)
            basic_name = tab.names[tab.basis[row_i]]
            text = (f"Cut {k + 1}: variable {basic_name} = {fmt_frac(tab.b[row_i])} has the largest fractional part "
                    f"{fmt_frac(f0)} → {terms} ≥ {fmt_frac(f0)}")
            cuts.append(f"{terms} ≥ {fmt_frac(f0)}")
            col = tab.add_column(f"g{k + 1}")
            tab.add_row([-f for f in coeffs] + [Fraction(1)], -f0, col)
            tab.snapshot(f"Cut {k + 1} added (row {tab.names[col]})", tab.cost)
            status = tab.dual(tab.cost, f"Cut {k + 1}")
            if status == "infeasible":
                details.append(DText(text, "title"))
                details += [s.table.display() for s in tab.steps[recorded:]]
                raise EvalError("the integer problem has no solution")
            tab.snapshot(f"Tableau after cut {k + 1}", tab.cost)
            details.append(DText(text, "title"))
            details += [s.table.display() for s in tab.steps[recorded:]]
            recorded = len(tab.steps)
        x = np.array([float(v) for v in tab.values(n)])
        F = float(tab.objective()) * sign
        # independent verification
        try:
            A_ub, b_ub, A_eq, b_eq = [], [], [], []
            for row, rhs, s in zip(A, b, signs):
                if s == "<=":
                    A_ub.append(row); b_ub.append(rhs)
                elif s == ">=":
                    A_ub.append(-row); b_ub.append(-rhs)
                else:
                    A_eq.append(row); b_eq.append(rhs)
            check = optimize.linprog(-c if sense == "max" else c, A_ub=A_ub or None, b_ub=b_ub or None,
                                     A_eq=A_eq or None, b_eq=b_eq or None, bounds=[(0, None)] * n,
                                     integrality=np.ones(n), method="highs")
            if check.success and abs(float(c @ check.x) - F) > 1e-6:
                details.append(DText(f"⚠ branch-and-bound check found F = {float(c @ check.x):g}", "error"))
            elif check.success:
                details.append(DText("✓ verified by branch-and-bound (HiGHS)", "note"))
        except Exception:
            pass
        return Record("Gomory's method", {
            "x": np.round(x).astype(np.int64), "F": clean(F, 1e-9), "x_lp": clean_array(np.array([float(v) for v in x_lp])),
            "F_lp": clean(float(F_lp), 1e-12), "cuts": len(cuts)}, details, ["x", "F", "x_lp", "F_lp", "cuts"])
