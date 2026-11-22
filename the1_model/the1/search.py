"""Generic, data-driven equation discovery strategies.

The search never contains a named physical law.  Instead it combines several
general inductive biases:

* sparse additive regression over a broad library of elementary features;
* sparse log-linear regression (products, powers, exponentials, trig products);
* free-exponent power-law fitting with sign handling;
* rational-function regression with target transformations;
* an optional evolutionary search over the full grammar.

All strategies return SymPy expressions that are later scored and validated by
the core engine.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import sympy as sp

from .constants import snap_constant, snap_exponent
from .grammar import (
    ALGEBRA_OPS, ALL_UNARY_OPS, EXP_LOG_OPS, HYPERBOLIC_OPS, SPECIAL_OPS,
    TRIG_OPS, safe_exp, safe_gamma, safe_log, safe_sqrt,
)

# ---------------------------------------------------------------------------
# Numeric evaluation
# ---------------------------------------------------------------------------
try:
    from scipy import special as _scipy_special
    _erf = _scipy_special.erf
    _gamma = safe_gamma
except Exception:  # pragma: no cover
    _erf = np.vectorize(math.erf)
    _gamma = safe_gamma


_SAFE_MODULE = {
    'exp': safe_exp,
    'log': safe_log,
    'sqrt': safe_sqrt,
    'Abs': np.abs,
    'abs': np.abs,
    'sin': np.sin,
    'cos': np.cos,
    'tan': np.tan,
    'asin': np.arcsin,
    'acos': np.arccos,
    'atan': np.arctan,
    'sinh': np.sinh,
    'cosh': np.cosh,
    'tanh': np.tanh,
    'asinh': np.arcsinh,
    'acosh': np.arccosh,
    'atanh': np.arctanh,
    'erf': _erf,
    'gamma': _gamma,
    'pi': np.pi,
    'E': np.e,
    'cbrt': np.cbrt,
    'sign': np.sign,
}


def evaluate_expr(expr: sp.Expr,
                  X: np.ndarray,
                  var_names: Sequence[str]) -> Optional[np.ndarray]:
    """Evaluate a SymPy expression with domain-safe numeric fallbacks."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    syms = [sp.Symbol(v) for v in var_names]
    try:
        fn = sp.lambdify(syms, expr, modules=[_SAFE_MODULE, 'numpy'])
        args = [X[:, i] for i in range(len(var_names))]
        with np.errstate(all='ignore'):
            out = fn(*args)
    except Exception:
        return None
    try:
        arr = np.asarray(out, dtype=float)
        if arr.ndim == 0:
            arr = np.full(X.shape[0], float(arr))
        arr = np.broadcast_to(arr, (X.shape[0],)).copy()
    except Exception:
        return None
    finite = np.isfinite(arr)
    if finite.mean() < 0.99:
        return None
    if not np.all(finite):
        arr = np.nan_to_num(arr, nan=0.0, posinf=1e12, neginf=-1e12)
    return arr


def _snap_simple(value: float,
                 tol: float = 0.06,
                 quantum: float = 0.25,
                 small: float = 1e-6) -> float:
    """Round a fitted coefficient to a legible value when it is nearly there.

    This is a *candidate generator*, not a decision rule: the raw fit is kept
    alongside the cleaned one and validation decides which generalises.
    """
    value = float(value)
    if not math.isfinite(value):
        return value
    if abs(value) <= tol and abs(value) > small:
        return 0.0
    if abs(value) >= 10.0:
        return value
    nearest = round(value / quantum) * quantum
    if abs(nearest - value) <= tol:
        return float(nearest)
    return value


def r2_rmse(y: np.ndarray, pred: np.ndarray) -> Tuple[float, float]:
    y = np.asarray(y, dtype=float).ravel()
    pred = np.asarray(pred, dtype=float).ravel()
    if y.shape != pred.shape or y.size == 0:
        return -1.0, float('inf')
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    rmse = float(np.sqrt(ss_res / y.size))
    if not np.isfinite(r2):
        r2 = -1.0
    return r2, rmse


# ---------------------------------------------------------------------------
# Candidate containers
# ---------------------------------------------------------------------------
@dataclass
class Feature:
    name: str
    values: np.ndarray
    expr: sp.Expr
    mode: str = 'linear'      # 'linear' -> exp(coef*expr), 'log' -> expr**coef
    complexity: int = 1


@dataclass
class Candidate:
    expr: sp.Expr
    source: str
    train_r2: float = -1.0
    train_rmse: float = float('inf')
    complexity: int = 1
    details: Dict[str, object] = field(default_factory=dict)
    train_pred: Optional[np.ndarray] = field(default=None, repr=False)

    def score(self, occam_weight: float = 0.002) -> float:
        return self.train_r2 - occam_weight * self.complexity


# ---------------------------------------------------------------------------
# Search engine
# ---------------------------------------------------------------------------
class GenericSearch:
    def __init__(self,
                 max_terms: int = 6,
                 max_library: int = 360,
                 max_monomials: int = 180,
                 min_train_r2: float = 0.85,
                 rational_tol: float = 1e-5,
                 random_state: int = 42,
                 enable_rational: bool = True,
                 enable_transforms: bool = True):
        self.max_terms = max_terms
        self.max_library = max_library
        self.max_monomials = max_monomials
        self.min_train_r2 = min_train_r2
        self.rational_tol = rational_tol
        self.random_state = random_state
        self.enable_rational = enable_rational
        self.enable_transforms = enable_transforms
        self.rng = np.random.RandomState(random_state)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def generate(self,
                 X: np.ndarray,
                 y: np.ndarray,
                 var_names: Sequence[str]) -> List[Candidate]:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        n, d = X.shape
        if n < 2 or d < 1 or y.size != n:
            return []

        methods = [
            ('power_law', self._power_law),
            ('log_linear', self._log_linear),
            ('sparse_additive', self._sparse_additive),
        ]
        candidates: List[Candidate] = []
        for name, fn in methods:
            try:
                candidates.extend(fn(X, y, var_names))
            except Exception:
                continue

        # If a compact algebraic/log-linear law already explains the data to
        # numerical precision, rational search cannot improve it.
        exact = any(c.train_r2 > 1.0 - 1e-9 and c.complexity <= 12 for c in candidates)
        if self.enable_rational and not exact:
            try:
                candidates.extend(self._rational_with_transforms(X, y, var_names))
            except Exception:
                pass

        # Deduplicate numerically, then keep the best representative.
        unique: Dict[tuple, Candidate] = {}
        for c in candidates:
            if not self._valid(c):
                continue
            key = self._fingerprint(c.expr, X, var_names, c.train_pred)
            old = unique.get(key)
            if old is None or (c.train_r2, -c.complexity) > (old.train_r2, -old.complexity):
                unique[key] = c

        ordered = sorted(unique.values(), key=lambda c: c.score(), reverse=True)

        # Keep a diverse portfolio.  Ranking purely by training fit lets a
        # high-complexity expression that has absorbed measurement noise push
        # the true compact law out of the candidate list before validation can
        # judge it.
        keep: List[Candidate] = list(ordered[:48])
        kept = {id(c) for c in keep}
        best_by_source: Dict[str, Candidate] = {}
        for c in ordered:
            if c.source not in best_by_source:
                best_by_source[c.source] = c
        for c in best_by_source.values():
            if id(c) not in kept:
                keep.append(c)
                kept.add(id(c))
        for c in sorted(ordered, key=lambda c: (c.complexity, -c.train_r2)):
            if len(keep) >= 96:
                break
            if id(c) not in kept and c.train_r2 >= max(self.min_train_r2, 0.9):
                keep.append(c)
                kept.add(id(c))
        return keep[:96]

    # ------------------------------------------------------------------
    # Validation / fingerprints
    # ------------------------------------------------------------------
    def _valid(self, c: Candidate) -> bool:
        return (c.expr is not None
                and np.isfinite(c.train_r2)
                and c.train_r2 >= self.min_train_r2)

    def _finalize_pred(self,
                       expr: sp.Expr,
                       source: str,
                       y: np.ndarray,
                       pred: np.ndarray,
                       details: Optional[Dict[str, object]] = None) -> Optional[Candidate]:
        """Finalize a candidate whose training predictions are already known.

        Avoids a SymPy -> NumPy round trip for every support variant, which
        dominates few-shot search time.
        """
        if expr is None or pred is None:
            return None
        pred = np.asarray(pred, dtype=float).ravel()
        if pred.shape != y.shape or not np.all(np.isfinite(pred)):
            return None
        r2, rmse = r2_rmse(y, pred)
        if not np.isfinite(r2) or r2 < self.min_train_r2:
            return None
        complexity = int(sp.count_ops(expr) + len(expr.free_symbols))
        return Candidate(expr=expr, source=source, train_r2=float(r2),
                         train_rmse=float(rmse), complexity=complexity,
                         details=details or {}, train_pred=pred)

    def _finalize(self,
                  expr: sp.Expr,
                  source: str,
                  X: np.ndarray,
                  y: np.ndarray,
                  var_names: Sequence[str],
                  details: Optional[Dict[str, object]] = None) -> Optional[Candidate]:
        if expr is None:
            return None
        pred = evaluate_expr(expr, X, var_names)
        if pred is None:
            return None
        return self._finalize_pred(expr, source, y, pred, details)

    @staticmethod
    def _simplify(expr: sp.Expr) -> sp.Expr:
        try:
            if sp.count_ops(expr) <= 80:
                expr = sp.simplify(expr)
        except Exception:
            pass
        return expr

    def _fingerprint(self, expr: sp.Expr, X: np.ndarray, var_names: Sequence[str],
                     pred: Optional[np.ndarray] = None) -> tuple:
        if pred is None:
            pred = evaluate_expr(expr, X, var_names)
        if pred is None:
            return (sp.srepr(expr),)
        if pred.size > 16:
            idx = np.linspace(0, pred.size - 1, 16).astype(int)
            pred = pred[idx]
        return tuple(np.round(pred, 5).tolist())

    # ------------------------------------------------------------------
    # Feature libraries
    # ------------------------------------------------------------------
    def _additive_features(self,
                           X: np.ndarray,
                           y: np.ndarray,
                           var_names: Sequence[str]) -> List[Feature]:
        n, d = X.shape
        syms = [sp.Symbol(v) for v in var_names]
        feats: List[Feature] = []

        def add(name: str, values: np.ndarray, expr: sp.Expr, complexity: int = 1):
            values = np.asarray(values, dtype=float)
            if values.shape != (n,) or not np.all(np.isfinite(values)):
                return
            if np.std(values) < 1e-12:
                return
            feats.append(Feature(name, values, expr, 'linear', complexity))

        for j in range(d):
            x = X[:, j]
            s = syms[j]
            ax = np.abs(x)
            add(f"{var_names[j]}", x, s)
            add(f"{var_names[j]}^2", x ** 2, s ** 2, 2)
            add(f"{var_names[j]}^3", x ** 3, s ** 3, 2)
            add(f"|{var_names[j]}|", ax, sp.Abs(s), 2)
            add(f"sqrt|{var_names[j]}|", np.sqrt(ax), sp.sqrt(sp.Abs(s)), 2)
            add(f"cbrt({var_names[j]})", np.cbrt(x),
                sp.sign(s) * sp.Abs(s) ** sp.Rational(1, 3), 2)
            nz = np.abs(x) > 1e-12
            if nz.all():
                add(f"1/{var_names[j]}", 1.0 / x, 1 / s, 2)
                add(f"1/{var_names[j]}^2", 1.0 / (x ** 2), 1 / (s ** 2), 2)
            if (x > 0).all():
                add(f"log({var_names[j]})", np.log(x), sp.log(s), 2)
                add(f"{var_names[j]}^1.5", np.power(x, 1.5), s ** sp.Rational(3, 2), 2)
                add(f"{var_names[j]}^-0.5", np.power(x, -0.5), s ** sp.Rational(-1, 2), 2)
            elif (x < 0).all():
                add(f"log(-{var_names[j]})", np.log(-x), sp.log(-s), 2)
            elif nz.all():
                add(f"log|{var_names[j]}|", np.log(ax), sp.log(sp.Abs(s)), 2)
            add(f"log(1+|{var_names[j]}|)", np.log1p(ax), sp.log(1 + sp.Abs(s)), 3)
            add(f"exp(-|{var_names[j]}|)", np.exp(-ax), sp.exp(-sp.Abs(s)), 3)
            if ax.max() <= 20.0:
                add(f"exp({var_names[j]})", np.exp(np.clip(x, -20, 20)), sp.exp(s), 2)
                add(f"exp(-{var_names[j]})", np.exp(np.clip(-x, -20, 20)), sp.exp(-s), 2)
                add(f"sinh({var_names[j]})", np.sinh(np.clip(x, -20, 20)), sp.sinh(s), 3)
                add(f"cosh({var_names[j]})", np.cosh(np.clip(x, -20, 20)), sp.cosh(s), 3)
            add(f"sin({var_names[j]})", np.sin(x), sp.sin(s), 2)
            add(f"cos({var_names[j]})", np.cos(x), sp.cos(s), 2)
            add(f"sin({var_names[j]})^2", np.sin(x) ** 2, sp.sin(s) ** 2, 3)
            add(f"cos({var_names[j]})^2", np.cos(x) ** 2, sp.cos(s) ** 2, 3)
            add(f"tanh({var_names[j]})", np.tanh(x), sp.tanh(s), 3)
            add(f"tanh({var_names[j]})^2", np.tanh(x) ** 2, sp.tanh(s) ** 2, 4)
            add(f"logistic({var_names[j]})", 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40))),
                1 / (1 + sp.exp(-s)), 4)
            add(f"erf({var_names[j]})", _erf(x), sp.erf(s), 4)
            add(f"atan({var_names[j]})", np.arctan(x), sp.atan(s), 3)
            add(f"tan({var_names[j]})", np.tan(x), sp.tan(s), 3)
            add(f"asinh({var_names[j]})", np.arcsinh(x), sp.asinh(s), 3)
            if (ax <= 1.0).all():
                add(f"asin({var_names[j]})", np.arcsin(x), sp.asin(s), 3)
                add(f"acos({var_names[j]})", np.arccos(x), sp.acos(s), 3)
            if (x >= 1.0).all():
                add(f"acosh({var_names[j]})", np.arccosh(x), sp.acosh(s), 4)
            if (ax < 1.0).all():
                add(f"atanh({var_names[j]})", np.arctanh(x), sp.atanh(s), 4)

        # Pairwise interactions for a manageable number of variables.
        if d <= 12:
            for i in range(d):
                for j in range(i + 1, d):
                    xi, xj = X[:, i], X[:, j]
                    si, sj = syms[i], syms[j]
                    add(f"{var_names[i]}*{var_names[j]}", xi * xj, si * sj, 2)
                    if (np.abs(xj) > 1e-12).all():
                        add(f"{var_names[i]}/{var_names[j]}", xi / xj, si / sj, 3)
                    if (np.abs(xi) > 1e-12).all():
                        add(f"{var_names[j]}/{var_names[i]}", xj / xi, sj / si, 3)
                    add(f"sin({var_names[i]})*sin({var_names[j]})", np.sin(xi) * np.sin(xj),
                        sp.sin(si) * sp.sin(sj), 4)
                    add(f"cos({var_names[i]})*cos({var_names[j]})", np.cos(xi) * np.cos(xj),
                        sp.cos(si) * sp.cos(sj), 4)

        return self._cap_library(feats, y)

    def _log_features(self, X: np.ndarray, var_names: Sequence[str],
                      target: Optional[np.ndarray] = None) -> List[Feature]:
        n, d = X.shape
        syms = [sp.Symbol(v) for v in var_names]
        feats: List[Feature] = []

        def add(name: str, values: np.ndarray, expr: sp.Expr,
                mode: str = 'linear', complexity: int = 1):
            values = np.asarray(values, dtype=float)
            if values.shape != (n,) or not np.all(np.isfinite(values)):
                return
            if np.std(values) < 1e-12:
                return
            feats.append(Feature(name, values, expr, mode, complexity))

        for j in range(d):
            x = X[:, j]
            s = syms[j]
            ax = np.abs(x)
            nz = ax > 1e-12
            if (x > 0).all():
                add(f"log({var_names[j]})", np.log(x), s, 'log', 2)
            elif (x < 0).all():
                add(f"log(-{var_names[j]})", np.log(-x), sp.Abs(s), 'log', 2)
            elif nz.all():
                add(f"log|{var_names[j]}|", np.log(ax), sp.Abs(s), 'log', 2)

            add(f"log(1+|{var_names[j]}|)", np.log1p(ax), 1 + sp.Abs(s), 'log', 3)
            add(f"{var_names[j]}", x, s, 'linear', 1)
            add(f"{var_names[j]}^2", x ** 2, s ** 2, 'linear', 2)
            add(f"sqrt|{var_names[j]}|", np.sqrt(ax), sp.sqrt(sp.Abs(s)), 'linear', 2)
            if nz.all():
                add(f"1/{var_names[j]}", 1.0 / x, 1 / s, 'linear', 2)

            # Products/quotients involving trigonometric features are common in
            # wave, optics and oscillation laws; the search does not know which
            # variable is an angle.
            for fn_name, fn_np, fn_sp in [
                ('sin', np.sin, sp.sin), ('cos', np.cos, sp.cos), ('tan', np.tan, sp.tan),
            ]:
                vals = fn_np(x)
                av = np.abs(vals)
                if np.all(av > 1e-4) and np.all(np.isfinite(vals)):
                    base = fn_sp(s) if (vals > 0).all() else sp.Abs(fn_sp(s))
                    add(f"log|{fn_name}({var_names[j]})|", np.log(av), base, 'log', 3)

        if d <= 12:
            for i in range(d):
                for j in range(i + 1, d):
                    xi, xj = X[:, i], X[:, j]
                    si, sj = syms[i], syms[j]
                    add(f"{var_names[i]}*{var_names[j]}", xi * xj, si * sj, 'linear', 2)
                    if (np.abs(xj) > 1e-12).all():
                        add(f"{var_names[i]}/{var_names[j]}", xi / xj, si / sj, 'linear', 3)
                    if (np.abs(xi) > 1e-12).all():
                        add(f"{var_names[j]}/{var_names[i]}", xj / xi, sj / si, 'linear', 3)

        return self._cap_library(feats, target if target is not None else np.zeros(n))

    def _cap_library(self, feats: List[Feature], target: np.ndarray) -> List[Feature]:
        if len(feats) <= self.max_library:
            return feats
        # Keep the most correlated features and all raw variables.
        scores = []
        for f in feats:
            v = f.values
            if np.std(v) < 1e-12 or np.std(target) < 1e-12:
                scores.append(0.0)
            else:
                scores.append(abs(float(np.corrcoef(v, target)[0, 1])))
        order = np.argsort(scores)[::-1][: self.max_library]
        return [feats[i] for i in sorted(order)]

    # ------------------------------------------------------------------
    # Sparse linear regression
    # ------------------------------------------------------------------
    def _beam_search(self,
                     features: Sequence[Feature],
                     target: np.ndarray,
                     max_terms: int,
                     beam_width: int = 16,
                     ) -> List[Tuple[Tuple[int, ...], np.ndarray, float, float]]:
        """Beam search over sparse term subsets.

        Greedy forward selection is easily trapped when the first selected term
        is not part of the true (small) support.  A modest beam keeps several
        partial supports alive, which recovers exact laws without an exponential
        best-subset search.
        """
        n = target.size
        if not features:
            return []
        max_terms = max(1, min(max_terms, max(2, n // 2)))

        def evaluate_support(sel: Tuple[int, ...]) -> Optional[Tuple[Tuple[int, ...], np.ndarray, float, float]]:
            A = np.column_stack([np.ones(n)] + [features[k].values for k in sel])
            try:
                coef, *_ = np.linalg.lstsq(A, target, rcond=None)
            except Exception:
                return None
            rss = float(np.sum((target - A @ coef) ** 2))
            if not np.isfinite(rss):
                return None
            k = A.shape[1]
            # The tiny feature-complexity term breaks exact numerical ties in
            # favour of simpler atoms (x over |x| on an all-positive domain).
            atom_cost = sum(features[j].complexity for j in sel)
            bic = n * math.log(max(rss / n, 1e-15)) + k * math.log(max(n, 2))
            bic += 1e-6 * atom_cost
            return (sel, coef, rss, bic)

        empty_coef = np.array([float(np.mean(target))])
        empty_rss = float(np.sum((target - np.full(n, empty_coef[0])) ** 2))
        empty_bic = n * math.log(max(empty_rss / n, 1e-15)) + math.log(max(n, 2))
        seen_global: List[Tuple[Tuple[int, ...], np.ndarray, float, float]] = [
            (tuple(), empty_coef, empty_rss, empty_bic)
        ]
        seen_supports = {tuple()}

        # Level 1: all singletons.  Keeping every singleton (not just the beam)
        # matters because individually weak terms can be jointly exact, e.g.
        # det = a*d - b*c where neither product correlates strongly alone.
        singles: List[Tuple[Tuple[int, ...], np.ndarray, float, float]] = []
        for j in range(len(features)):
            fit = evaluate_support((j,))
            if fit is not None:
                singles.append(fit)
                seen_supports.add((j,))
        singles.sort(key=lambda item: (item[3], len(item[0])))
        seen_global.extend(singles)

        # Bounded pair enumeration over a moderate pool of singletons.
        pool = [item[0][0] for item in singles[: min(len(singles), max(64, 2 * beam_width))]]
        pair_fits: List[Tuple[Tuple[int, ...], np.ndarray, float, float]] = []
        for a_idx in range(len(pool)):
            for b_idx in range(a_idx + 1, len(pool)):
                key = tuple(sorted((pool[a_idx], pool[b_idx])))
                if key in seen_supports:
                    continue
                seen_supports.add(key)
                fit = evaluate_support(key)
                if fit is not None:
                    pair_fits.append(fit)
                    seen_global.append(fit)
        pair_fits.sort(key=lambda item: (item[3], len(item[0])))

        states: List[Tuple[Tuple[int, ...], np.ndarray, float, float]] = (
            pair_fits[:beam_width] if pair_fits else singles[:beam_width]
        )
        start_terms = 3 if pair_fits else 2

        for _ in range(start_terms, max_terms + 1):
            candidates: List[Tuple[Tuple[int, ...], np.ndarray, float, float]] = []
            for sel, _, _, _ in states:
                for j in range(len(features)):
                    if j in sel:
                        continue
                    key = tuple(sorted(sel + (j,)))
                    if key in seen_supports:
                        continue
                    fit = evaluate_support(key)
                    if fit is not None:
                        candidates.append(fit)

            if not candidates:
                break
            candidates.sort(key=lambda item: (item[3], len(item[0])))
            new_states = []
            for cand in candidates:
                key = tuple(sorted(cand[0]))
                if key in seen_supports:
                    continue
                seen_supports.add(key)
                new_states.append(cand)
                seen_global.append(cand)
                if len(new_states) >= beam_width:
                    break
            if not new_states:
                break
            states = new_states
            if min(c[2] for c in states) < 1e-18:
                break

        # Prefer the lowest BIC; among numerical ties choose the smallest support.
        def rank(item):
            sel, _, rss, bic = item
            return (round(bic, 9), len(sel), rss)

        seen_global.sort(key=rank)
        return seen_global

    def _beam_select(self,
                     features: Sequence[Feature],
                     target: np.ndarray,
                     max_terms: int,
                     beam_width: int = 16) -> Tuple[np.ndarray, List[Feature]]:
        n = target.size
        seen = self._beam_search(features, target, max_terms, beam_width)
        if not seen:
            return np.array([float(np.mean(target))]), []
        best_sel, _, _, _ = seen[0]
        if not best_sel:
            return np.array([float(np.mean(target))]), []
        A = np.column_stack([np.ones(n)] + [features[j].values for j in best_sel])
        try:
            coef, *_ = np.linalg.lstsq(A, target, rcond=None)
        except Exception:
            coef = np.zeros(A.shape[1])
            coef[0] = float(np.mean(target))
        return coef, [features[j] for j in best_sel]

    def _beam_portfolio(self,
                        features: Sequence[Feature],
                        target: np.ndarray,
                        max_terms: int,
                        beam_width: int = 16,
                        top: int = 6,
                        per_size: int = 4) -> List[Tuple[np.ndarray, List[Feature]]]:
        """Several high-quality supports, not just the single best one.

        Measurement noise inflates BIC's preference toward extra terms, so the
        top-scoring supports may all be overfits.  Keeping the best few
        supports of *every* cardinality lets validation recover the compact
        true law that the training-set criterion discarded.
        """
        seen = self._beam_search(features, target, max_terms, beam_width)
        chosen: List[Tuple[Tuple[int, ...], np.ndarray]] = []
        used = set()

        def take(item):
            sel, coef, _, _ = item
            key = tuple(sorted(sel))
            if not sel or key in used:
                return
            used.add(key)
            chosen.append((sel, coef))

        for item in seen[:top]:
            take(item)
        for size in range(1, max_terms + 1):
            count = 0
            for item in seen:
                if len(item[0]) != size:
                    continue
                before = len(used)
                take(item)
                if len(used) > before:
                    count += 1
                if count >= per_size:
                    break
        return [(coef, [features[j] for j in sel]) for sel, coef in chosen]

    def _sparse_additive(self, X, y, var_names) -> List[Candidate]:
        feats = self._additive_features(X, y, var_names)
        portfolio = self._beam_portfolio(feats, y, self.max_terms, beam_width=12, top=6)
        out: List[Candidate] = []
        for coef, selected in portfolio:
            coef_snapped = np.array([snap_constant(c) for c in coef])
            coef_clean = np.array([_snap_simple(c, tol=0.06) for c in coef])

            def predict(cvec: np.ndarray) -> np.ndarray:
                pred = np.full(len(y), float(cvec[0]))
                for c, f in zip(cvec[1:], selected):
                    pred = pred + float(c) * f.values
                return pred

            for cvec, snapped in ((coef, False), (coef_snapped, True), (coef_clean, True)):
                expr = self._linear_expression(cvec, selected)
                cand = self._finalize_pred(
                    expr, 'sparse_additive', y, predict(cvec),
                    {'terms': [f.name for f in selected], 'snapped': snapped},
                )
                if cand is not None:
                    out.append(cand)
        return out

    @staticmethod
    def _linear_expression(coef: np.ndarray, selected: Sequence[Feature]) -> sp.Expr:
        terms = []
        if abs(float(coef[0])) > 1e-12:
            terms.append(sp.Float(float(coef[0])))
        for c, f in zip(coef[1:], selected):
            if abs(float(c)) < 1e-12:
                continue
            terms.append(sp.Float(float(c)) * f.expr)
        return sp.Add(*terms) if terms else sp.Integer(0)

    # ------------------------------------------------------------------
    # Power laws
    # ------------------------------------------------------------------
    def _power_law(self, X, y, var_names) -> List[Candidate]:
        n, d = X.shape
        mask = (np.abs(X) > 1e-12).all(axis=1) & (np.abs(y) > 1e-12)
        if mask.sum() < max(3, d + 1):
            return []
        Xm, ym = X[mask], y[mask]
        A = np.column_stack([np.ones(Xm.shape[0]), np.log(np.abs(Xm))])
        try:
            sol, *_ = np.linalg.lstsq(A, np.log(np.abs(ym)), rcond=None)
        except Exception:
            return []
        intercept = float(sol[0])
        powers = np.array([snap_exponent(p) for p in sol[1:]])
        if abs(intercept) > 60:
            return []

        raw_powers = np.asarray(sol[1:], dtype=float)
        log_abs_X = np.log(np.abs(Xm))
        log_abs_y = np.log(np.abs(ym))

        def round_set(quantum: float, tol: float) -> np.ndarray:
            out = []
            for p in raw_powers:
                nearest = round(p / quantum) * quantum
                out.append(float(nearest) if abs(nearest - p) <= tol else float(p))
            return np.asarray(out)

        syms = [sp.Symbol(v) for v in var_names]
        sign_y = 1.0 if np.all(y > 0) else (-1.0 if np.all(y < 0) else None)

        exponent_sets = [(powers, intercept, '')]
        candidate_powers = [round_set(1.0, 0.25), round_set(0.5, 0.18)]
        for ps in candidate_powers:
            if any(np.allclose(ps, existing[0], atol=1e-12) for existing in exponent_sets):
                continue
            icpt = float(np.mean(log_abs_y - log_abs_X @ ps))
            exponent_sets.append((ps, icpt, '_clean'))

        candidates: List[Candidate] = []
        for ps, icpt, tag in exponent_sets:
            int_powers = np.all(np.abs(ps - np.round(ps)) < 1e-7)
            prod_abs = np.ones(n)
            prod_signed = np.ones(n)
            expr_abs = sp.Integer(1)
            expr_signed = sp.Integer(1)
            for j, p in enumerate(ps):
                if abs(p) < 1e-9:
                    continue
                prod_abs = prod_abs * np.power(np.abs(X[:, j]), p)
                if p == int(p):
                    prod_signed = prod_signed * np.power(X[:, j], int(p))
                    expr_signed = expr_signed * syms[j] ** int(p)
                else:
                    prod_signed = prod_signed * np.power(np.abs(X[:, j]), p)
                    expr_signed = expr_signed * sp.Abs(syms[j]) ** p
                expr_abs = expr_abs * sp.Abs(syms[j]) ** p

            source = 'power_law' + tag
            # Signed product handles mixed-sign targets when powers are integers.
            if int_powers and np.std(prod_signed) > 1e-12:
                c = float(np.sum(y * prod_signed) / (np.sum(prod_signed ** 2) + 1e-12))
                c = snap_constant(c)
                cand = self._finalize_pred(c * expr_signed, source, y, c * prod_signed,
                                           {'exponents': ps.tolist(), 'signed': True})
                if cand is not None:
                    candidates.append(cand)

            if sign_y is not None and np.std(prod_abs) > 1e-12:
                c = snap_constant(sign_y * float(np.exp(icpt)))
                cand = self._finalize_pred(c * expr_abs, source, y, c * prod_abs,
                                           {'exponents': ps.tolist()})
                if cand is not None:
                    candidates.append(cand)
                if sign_y > 0:
                    # Amplitude fitted in the original space avoids the
                    # log-transform bias of additive measurement noise.
                    denom = float(np.sum(prod_abs * prod_abs))
                    if denom > 1e-12:
                        a = snap_constant(float(np.sum(y * prod_abs) / denom))
                        cand = self._finalize_pred(a * expr_abs, source + '_lsq',
                                                   y, a * prod_abs,
                                                   {'exponents': ps.tolist(),
                                                    'amplitude': 'lsq'})
                        if cand is not None:
                            candidates.append(cand)

            # If all variables are positive, the absolute expression is exact.
            if (X > 0).all() and sign_y is not None:
                expr_pos = sp.Integer(1)
                for j, p in enumerate(ps):
                    if abs(p) > 1e-9:
                        expr_pos *= syms[j] ** p
                c = snap_constant(sign_y * float(np.exp(icpt)))
                cand = self._finalize_pred(c * expr_pos, source + '_positive',
                                           y, c * prod_abs,
                                           {'exponents': ps.tolist()})
                if cand is not None:
                    candidates.append(cand)
        return candidates

    # ------------------------------------------------------------------
    # Log-linear products / exponentials
    # ------------------------------------------------------------------
    def _log_linear(self, X, y, var_names) -> List[Candidate]:
        if np.any(np.abs(y) < 1e-12):
            return []
        sign_y = 1.0 if np.all(y > 0) else (-1.0 if np.all(y < 0) else None)
        if sign_y is None:
            return []

        z = np.log(np.abs(y))
        feats = self._log_features(X, var_names, target=z)
        portfolio = self._beam_portfolio(
            feats, z, max_terms=min(self.max_terms, 6), beam_width=20, top=5,
        )
        def make_expr(amplitude: float, slopes: Sequence[float],
                      features: Sequence, tag: str) -> Optional[Candidate]:
            amplitude = snap_constant(sign_y * amplitude)
            factors = [sp.Float(amplitude)]
            used = []
            g = np.ones(len(y))
            for cj, f in zip(slopes, features):
                g = g * np.exp(np.clip(float(cj) * f.values, -60, 60))
                if abs(cj) < 1e-9:
                    continue
                used.append((f.name, float(cj)))
                if f.mode == 'log':
                    factors.append(f.expr ** float(cj))
                else:
                    factors.append(sp.exp(float(cj) * f.expr))
            expr = sp.Mul(*factors)
            return self._finalize_pred(
                expr, 'log_linear' + tag, y, amplitude * g,
                {'terms': used, 'sign': sign_y},
            )

        def build(intercept: float, slopes: Sequence[float], features: Sequence,
                  tag: str) -> Optional[Candidate]:
            if not np.isfinite(intercept) or abs(intercept) > 60:
                return None
            return make_expr(math.exp(intercept), slopes, features, tag)

        def build_lsq(slopes: Sequence[float], features: Sequence,
                      tag: str) -> Optional[Candidate]:
            """Refit the multiplicative amplitude in the original target space.

            Fitting ``log y`` gives the wrong amplitude under additive noise
            (the log of a noisy small value is strongly biased).  Once the
            slopes are fixed, a single linear least-squares solve recovers the
            amplitude without that bias.
            """
            g = np.exp(np.clip(B @ np.asarray(slopes, dtype=float), -60, 60))
            denom = float(np.sum(g * g))
            if denom < 1e-12:
                return None
            amplitude = float(np.sum(y * sign_y * g) / denom)
            if amplitude <= 0 or not np.isfinite(amplitude):
                return None
            return make_expr(amplitude, slopes, features, tag)

        out: List[Candidate] = []
        for coef, selected in portfolio:
            if not selected:
                continue
            B = np.column_stack([f.values for f in selected])
            slopes = np.array([snap_exponent(float(c)) for c in coef[1:]])
            for cand in (
                build(float(coef[0]), slopes, selected, ''),
                build_lsq(slopes, selected, '_lsq'),
            ):
                if cand is not None:
                    out.append(cand)

            clean_slopes = np.array([_snap_simple(float(c), tol=0.13) for c in coef[1:]])
            if not np.allclose(clean_slopes, slopes, atol=1e-12):
                c0_clean = float(np.mean(z - B @ clean_slopes))
                for cand in (
                    build(c0_clean, clean_slopes, selected, '_clean'),
                    build_lsq(clean_slopes, selected, '_clean_lsq'),
                ):
                    if cand is not None:
                        out.append(cand)
        return out

    # ------------------------------------------------------------------
    # Rational functions and target transforms
    # ------------------------------------------------------------------
    @staticmethod
    def _monomials(var_names: Sequence[str], degree: int,
                   max_count: int) -> List[Tuple[Tuple[int, ...], sp.Expr]]:
        d = len(var_names)
        syms = [sp.Symbol(v) for v in var_names]
        exps: List[Tuple[int, ...]] = []

        def rec(i: int, remaining: int, current: List[int]):
            if len(exps) >= max_count:
                return
            if i == d:
                exps.append(tuple(current))
                return
            for e in range(remaining + 1):
                rec(i + 1, remaining - e, current + [e])

        rec(0, degree, [])
        exps = sorted(set(exps), key=lambda e: (sum(e), e))
        out = []
        for e in exps:
            expr = sp.Integer(1)
            for s, p in zip(syms, e):
                if p:
                    expr *= s ** p
            out.append((e, expr))
        return out

    @staticmethod
    def _monomial_values(X: np.ndarray, exponent_tuples: Sequence[Tuple[int, ...]]) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        M = np.ones((X.shape[0], len(exponent_tuples)), dtype=float)
        for k, exps in enumerate(exponent_tuples):
            for j, p in enumerate(exps):
                if p:
                    M[:, k] *= X[:, j] ** p
        return M

    def _fit_rational(self,
                      X: np.ndarray,
                      target: np.ndarray,
                      var_names: Sequence[str],
                      pdeg: int,
                      qdeg: int) -> Optional[Tuple[sp.Expr, float, int, float]]:
        n = target.size
        monos_p = self._monomials(var_names, pdeg, self.max_monomials)
        monos_q_all = self._monomials(var_names, qdeg, self.max_monomials)
        monos_q = [(e, expr) for e, expr in monos_q_all if sum(e) > 0]
        if not monos_p:
            return None
        # A dense rational fit with as many free coefficients as observations
        # can interpolate the training set while generalising arbitrarily badly.
        # Require residual degrees of freedom.
        if len(monos_p) + len(monos_q) > n - 2:
            return None

        Mp = self._monomial_values(X, [e for e, _ in monos_p])
        Mq = self._monomial_values(X, [e for e, _ in monos_q]) if monos_q else np.zeros((n, 0))
        if not np.all(np.isfinite(Mp)) or (Mq.size and not np.all(np.isfinite(Mq))):
            return None
        if Mq.size:
            A = np.column_stack([Mp, -target[:, None] * Mq])
        else:
            A = Mp
        b = target

        # Column scaling improves conditioning when variables have different units.
        scales = np.linalg.norm(A, axis=0)
        scales[scales < 1e-12] = 1.0
        try:
            coef_s, *_ = np.linalg.lstsq(A / scales, b, rcond=None)
        except Exception:
            return None
        coef = coef_s / scales
        pcoef = coef[: len(monos_p)]
        qcoef = coef[len(monos_p):]

        P = Mp @ pcoef
        Q = np.ones(n)
        if Mq.size:
            Q = Q + Mq @ qcoef
        if np.min(np.abs(Q)) < 1e-7:
            return None
        with np.errstate(all='ignore'):
            pred = P / Q
        r2, _ = r2_rmse(target, pred)
        if not np.isfinite(r2):
            return None

        P_expr = sp.Add(*[sp.Float(float(c)) * m for c, (_, m) in zip(pcoef, monos_p)
                          if abs(float(c)) > 1e-12])
        if P_expr == 0:
            P_expr = sp.Integer(0)
        Q_expr = sp.Integer(1)
        if Mq.size:
            Q_expr = Q_expr + sp.Add(*[sp.Float(float(c)) * m for c, (_, m) in zip(qcoef, monos_q)
                                       if abs(float(c)) > 1e-12])
        # Do not call sp.simplify here: it dominates runtime and the expression
        # is already in a compact P/Q form.  The core engine can simplify the
        # final winner once.
        expr = P_expr / Q_expr
        k = len(monos_p) + len(monos_q)
        return expr, float(r2), k, self._bic(target, pred, k)

    @staticmethod
    def _bic(target: np.ndarray, pred: np.ndarray, k: int) -> float:
        n = max(2, target.size)
        rss = float(np.sum((target - pred) ** 2))
        return n * math.log(max(rss / n, 1e-30)) + k * math.log(n)

    def _fit_rational_support(self,
                              X: np.ndarray,
                              target: np.ndarray,
                              p_terms: Sequence[Tuple[Tuple[int, ...], sp.Expr]],
                              q_terms: Sequence[Tuple[Tuple[int, ...], sp.Expr]],
                              q_has_constant: bool = True,
                              ) -> Optional[Tuple[sp.Expr, float, int, float]]:
        """Fit ``P/Q`` for one explicit (sparse) monomial support.

        Dense rational regression spreads an exact law over every available
        monomial, which overfits few-shot data.  Enumerating small supports
        recovers compact forms such as ``r1*r2/(r1+r2)`` with very few
        coefficients.

        A denominator with a fixed constant term represents laws such as
        ``1/(1 + x^2)``.  Homogeneous denominators (``q_has_constant=False``,
        e.g. ``r1*r2/(r1 + r2)``) require solving a homogeneous system; the
        smallest singular vector of the design matrix is the normalised
        coefficient vector.
        """
        n = target.size
        Mp = self._monomial_values(X, [e for e, _ in p_terms])
        Mq = (self._monomial_values(X, [e for e, _ in q_terms])
              if q_terms else np.zeros((n, 0)))
        A = np.column_stack([Mp, -target[:, None] * Mq]) if Mq.size else Mp
        scales = np.linalg.norm(A, axis=0)
        scales[scales < 1e-12] = 1.0
        try:
            if q_has_constant or not Mq.size:
                coef_s, *_ = np.linalg.lstsq(A / scales, target, rcond=None)
            else:
                _, _, vt = np.linalg.svd(A / scales, full_matrices=False)
                coef_s = vt[-1]
        except Exception:
            return None
        coef = coef_s / scales
        pcoef = coef[: len(p_terms)]
        qcoef = coef[len(p_terms):]

        # Homogeneous denominators are scale invariant; normalise by one
        # coefficient so the printed law has no arbitrary common factor.
        if Mq.size and not q_has_constant and qcoef.size:
            ref = int(np.argmax(np.abs(qcoef)))
            ref_val = float(qcoef[ref])
            if abs(ref_val) > 1e-12:
                pcoef = pcoef / ref_val
                qcoef = qcoef / ref_val

        Q = np.ones(n) if (q_has_constant or not Mq.size) else np.zeros(n)
        if Mq.size:
            Q = Q + Mq @ qcoef
        if np.min(np.abs(Q)) < 1e-7:
            return None
        with np.errstate(all='ignore'):
            pred = (Mp @ pcoef) / Q
        r2, _ = r2_rmse(target, pred)
        if not np.isfinite(r2):
            return None

        sig_p = [abs(float(c)) > 1e-9 for c in pcoef]
        sig_q = [abs(float(c)) > 1e-9 for c in qcoef]
        if not any(sig_p):
            return None
        k = int(sum(sig_p) + sum(sig_q))

        P_expr = sp.Add(*[sp.Float(float(c)) * m
                          for c, s, (_, m) in zip(pcoef, sig_p, p_terms) if s])
        if Mq.size:
            q_sum = sp.Add(*[sp.Float(float(c)) * m
                             for c, s, (_, m) in zip(qcoef, sig_q, q_terms) if s])
            Q_expr = q_sum if not q_has_constant else sp.Integer(1) + q_sum
        else:
            Q_expr = sp.Integer(1)
        if Q_expr == 0:
            return None
        return P_expr / Q_expr, float(r2), k, self._bic(target, pred, k)

    def _sparse_rational(self,
                         X: np.ndarray,
                         target: np.ndarray,
                         var_names: Sequence[str],
                         top_k: int = 4) -> List[Tuple[sp.Expr, float, int, float]]:
        n, d = X.shape
        if not (2 <= d <= 3) or n < 5:
            return []
        p_terms = self._monomials(var_names, 2, self.max_monomials)
        q_terms_all = self._monomials(var_names, 1, self.max_monomials)
        q_terms = [(e, ex) for e, ex in q_terms_all if sum(e) > 0]
        if not p_terms or not q_terms:
            return []

        p_supports: List[Tuple[int, ...]] = []
        for size in range(1, min(4, len(p_terms) + 1)):
            p_supports.extend(itertools.combinations(range(len(p_terms)), size))
        q_supports: List[Tuple[int, ...]] = [tuple()]
        for size in range(1, min(3, len(q_terms) + 1)):
            q_supports.extend(itertools.combinations(range(len(q_terms)), size))

        fits: List[Tuple[sp.Expr, float, int, float]] = []
        for ps in p_supports:
            for qs in q_supports:
                if len(ps) + len(qs) > n - 2:
                    continue
                modes = (True, False) if qs else (True,)
                for q_const in modes:
                    fit = self._fit_rational_support(
                        X, target,
                        [p_terms[i] for i in ps],
                        [q_terms[i] for i in qs],
                        q_has_constant=q_const,
                    )
                    if fit is not None and fit[1] >= max(self.min_train_r2, 0.95):
                        fits.append(fit)
        fits.sort(key=lambda item: (item[3], item[2]))
        return fits[:top_k]

    @staticmethod
    def _target_transforms(y: np.ndarray) -> List[Tuple[str, np.ndarray, Callable[[sp.Expr], sp.Expr]]]:
        out: List[Tuple[str, np.ndarray, Callable[[sp.Expr], sp.Expr]]] = [
            ('identity', y, lambda e: e),
        ]
        ypos = np.all(y > 0)
        yneg = np.all(y < 0)
        nz = np.all(np.abs(y) > 1e-12)
        if ypos:
            out += [
                ('log', np.log(y), lambda e: sp.exp(e)),
                ('sqrt', np.sqrt(y), lambda e: e ** 2),
                ('cbrt', np.cbrt(y), lambda e: e ** 3),
                ('square', y ** 2, lambda e: sp.sqrt(e)),
                ('reciprocal', 1.0 / y, lambda e: 1 / e),
                ('inv_square', 1.0 / (y ** 2), lambda e: 1 / sp.sqrt(e)),
            ]
        if yneg:
            out += [
                ('cbrt', np.cbrt(y), lambda e: e ** 3),
                ('cube', y ** 3, lambda e: sp.cbrt(e)),
                ('square', y ** 2, lambda e: -sp.sqrt(e)),
                ('reciprocal', 1.0 / y, lambda e: 1 / e),
                ('inv_square', 1.0 / (y ** 2), lambda e: -1 / sp.sqrt(e)),
            ]
        if nz and not (ypos or yneg):
            out += [('reciprocal', 1.0 / y, lambda e: 1 / e)]
        # Deduplicate by transform name + values fingerprint.
        seen = set()
        unique = []
        for name, z, inv in out:
            key = (name, tuple(np.round(z[: min(8, z.size)], 8).tolist()))
            if key in seen:
                continue
            seen.add(key)
            unique.append((name, z, inv))
        return unique

    def _rational_with_transforms(self, X, y, var_names) -> List[Candidate]:
        candidates: List[Candidate] = []
        n, d = X.shape
        max_p = 3 if d <= 4 else 2
        max_q = 2 if d <= 4 else 1
        for tname, z, inverse in self._target_transforms(y):
            if not np.all(np.isfinite(z)):
                continue
            fits: List[Tuple[sp.Expr, float, int, float]] = []
            for pdeg in range(0, max_p + 1):
                for qdeg in range(0, max_q + 1):
                    if pdeg == 0 and qdeg == 0:
                        continue
                    fit = self._fit_rational(X, z, var_names, pdeg, qdeg)
                    if fit is not None:
                        fits.append(fit)
            # Compact rational laws are sparse in the monomial basis; search
            # small supports directly instead of trusting a dense fit.
            if tname in ('identity', 'reciprocal'):
                fits.extend(self._sparse_rational(X, z, var_names))
            if not fits:
                continue
            fits.sort(key=lambda item: (item[3], item[2]))
            for expr_z, r2z, k, _ in fits[:3]:
                if r2z < max(self.min_train_r2, 0.95):
                    continue
                try:
                    expr = inverse(expr_z)
                except Exception:
                    continue
                cand = self._finalize(expr, f'rational_{tname}', X, y, var_names,
                                      {'coefficients': k, 'transform': tname})
                if cand is not None:
                    candidates.append(cand)
        return candidates


__all__ = ['GenericSearch', 'Candidate', 'Feature', 'evaluate_expr', 'r2_rmse']
