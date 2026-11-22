"""Lean 4 hypothesis verification for discovered symbolic laws.

A discovered equation is only an empirical fit until it is checked against
formal reasoning.  This module turns a ``SymbolicLaw`` into a series of
conjectures (symmetry, permutation invariance, homogeneity, positivity, axis
values and derivative rules), translates both the law and the conjectures to
Lean 4 + Mathlib, and asks the Lean kernel whether they hold.

If no Lean toolchain is installed the module still produces the complete Lean
source and reports every hypothesis as ``unverified``; nothing fails silently.
"""

from __future__ import annotations

import math
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import sympy as sp
from sympy.core.numbers import ImaginaryUnit

from .constants import snap_exponent


class LeanTranslationError(ValueError):
    """The SymPy expression uses an operation without a Lean 4 counterpart."""


_LEAN_FUNCTIONS = {
    'sin': 'Real.sin',
    'cos': 'Real.cos',
    'tan': 'Real.tan',
    'asin': 'Real.asin',
    'acos': 'Real.acos',
    'atan': 'Real.atan',
    'sinh': 'Real.sinh',
    'cosh': 'Real.cosh',
    'tanh': 'Real.tanh',
    'asinh': 'Real.asinh',
    'acosh': 'Real.acosh',
    'atanh': 'Real.atanh',
    'exp': 'Real.exp',
    'log': 'Real.log',
    'sqrt': 'Real.sqrt',
    'erf': 'Real.erf',
}

_LEAN_KEYWORDS = {
    'theorem', 'lemma', 'example', 'def', 'variable', 'variables', 'section',
    'namespace', 'end', 'open', 'import', 'by', 'fun', 'let', 'in', 'have',
    'show', 'from', 'if', 'then', 'else', 'match', 'with', 'do', 'return',
    'forall', 'exists', 'Prop', 'Type', 'Sort', 'axiom', 'structure', 'class',
    'instance', 'where', 'deriving', 'attribute', 'set_option', 'noncomputable',
}


def lean_identifier(name: str) -> str:
    """Turn a data column name into a legal Lean binder name."""
    ident = re.sub(r'[^0-9A-Za-z_]', '_', str(name)).strip('_')
    if not ident:
        ident = 'v'
    if ident[0].isdigit():
        ident = 'v_' + ident
    if ident in _LEAN_KEYWORDS:
        ident += '_'
    return ident


def lean_symbol_map(var_names: Sequence[str]) -> Dict[str, str]:
    """Legal, collision-free Lean names for every data column."""
    mapping: Dict[str, str] = {}
    used = set()
    for name in var_names:
        base = lean_identifier(name)
        ident = base
        suffix = 2
        while ident in used:
            ident = f"{base}_{suffix}"
            suffix += 1
        used.add(ident)
        mapping[str(name)] = ident
    return mapping


def _real_literal(value: float) -> str:
    value = float(value)
    if not math.isfinite(value):
        raise LeanTranslationError(f"non-finite constant {value!r}")
    if value == 0.0:
        return "(0 : ℝ)"
    if value.is_integer() and abs(value) < 10 ** 15:
        return f"({int(value)} : ℝ)"
    if abs(value) >= 10 ** 15 or abs(value) < 1e-6:
        # Lean's decimal parser is reliable; exponent notation for ℝ is not
        # available in every Mathlib version, so print positional digits.
        text = np.format_float_positional(value, unique=True, trim='-')
    else:
        text = repr(value)
    return f"({text} : ℝ)"


def _pow_text(base: str, exponent: sp.Expr) -> str:
    if exponent.is_Float:
        exact = sp.Rational(float(exponent)).limit_denominator(24)
        if abs(float(exact) - float(exponent)) <= 1e-12:
            exponent = exact
    if exponent.is_Integer:
        n = int(exponent)
        if n == 0:
            return "(1 : ℝ)"
        if n == 1:
            return f"({base})"
        if n == -1:
            return f"(1 / {base})"
        if n > 0:
            return f"({base} ^ {n})"
        return f"({base} ^ ({n} : ℤ))"
    if exponent.is_Rational:
        p, q = int(exponent.p), int(exponent.q)
        if p == 1 and q == 2:
            return f"(Real.sqrt {base})"
        if p == -1 and q == 2:
            return f"(1 / Real.sqrt {base})"
        sign = "-" if p < 0 else ""
        p = abs(p)
        if q == 1:
            return f"({base} ^ ({sign}{p} : ℤ))"
        return f"({base} ^ ({sign}{p} / {q} : ℝ))"
    if exponent.is_Float:
        return f"({base} ^ ({_real_literal(float(exponent))}))"
    raise LeanTranslationError(f"unsupported exponent {exponent}")


def sympy_to_lean(expr: sp.Expr,
                  var_names: Sequence[str],
                  symbols: Optional[Dict[str, str]] = None) -> str:
    """Translate a SymPy expression to a Lean 4 term over ``ℝ``."""
    symbols = symbols or lean_symbol_map(var_names)

    def rec(e: sp.Expr) -> str:
        if isinstance(e, sp.Symbol):
            if e.name not in symbols:
                raise LeanTranslationError(f"unknown symbol {e.name}")
            return symbols[e.name]
        if isinstance(e, sp.Integer):
            return _real_literal(int(e))
        if isinstance(e, sp.Rational):
            p, q = int(e.p), int(e.q)
            sign = "-" if p < 0 else ""
            p = abs(p)
            if q == 1:
                return f"({sign}{p} : ℝ)"
            return f"({sign}{p} / {q} : ℝ)"
        if isinstance(e, sp.Float):
            return _real_literal(float(e))
        if e == sp.pi:
            return "Real.pi"
        if e == sp.E:
            return "Real.exp 1"
        if e == sp.GoldenRatio:
            return "((1 + Real.sqrt 5) / 2)"
        if e == sp.EulerGamma:
            raise LeanTranslationError("EulerGamma is not translated")
        if e == sp.I or isinstance(e, ImaginaryUnit):
            raise LeanTranslationError("complex numbers are not translated")
        if isinstance(e, sp.Abs):
            return f"(|{rec(e.args[0])}|)"
        if isinstance(e, sp.Pow):
            return _pow_text(rec(e.base), e.exp)
        if isinstance(e, sp.Add):
            return "(" + " + ".join(rec(a) for a in e.args) + ")"
        if isinstance(e, sp.Mul):
            return "(" + " * ".join(rec(a) for a in e.args) + ")"
        if isinstance(e, sp.Function):
            name = str(e.func)
            if name in _LEAN_FUNCTIONS and len(e.args) == 1:
                return f"{_LEAN_FUNCTIONS[name]} {rec(e.args[0])}"
        raise LeanTranslationError(f"unsupported operation {e.func}")

    return rec(expr)


def _is_clean_rational(value: float) -> bool:
    if not math.isfinite(value) or abs(value) > 1e4:
        return False
    frac = sp.Rational(value).limit_denominator(100)
    return abs(float(frac) - value) <= 1e-9


def _has_transcendental(expr: sp.Expr) -> bool:
    if any(True for _ in expr.atoms(sp.Function)):
        return True
    return any(atom in (sp.pi, sp.E) for atom in expr.atoms(sp.NumberSymbol))


def _positivity_friendly(expr: sp.Expr) -> bool:
    """Whether Mathlib's ``positivity`` has a realistic chance on the term."""
    if expr.is_number:
        return bool(expr.is_positive)
    if isinstance(expr, sp.Symbol) or isinstance(expr, sp.Abs):
        return True
    if isinstance(expr, sp.Pow):
        return _positivity_friendly(expr.base) and not expr.exp.free_symbols
    if isinstance(expr, sp.Mul):
        return all(a.is_positive for a in expr.args if a.is_number) and all(
            _positivity_friendly(a) for a in expr.args if not a.is_number)
    if isinstance(expr, sp.Add):
        return all(_positivity_friendly(a) for a in expr.args)
    if isinstance(expr, sp.Function):
        return str(expr.func) == 'exp'
    return False


@dataclass
class LeanHypothesis:
    """A single conjecture about a law, with its Lean formalisation."""

    name: str
    kind: str
    statement: str
    tactics: str
    binders: str = ""
    support: float = float('nan')
    status: str = "pending"
    detail: str = ""
    line_start: int = 0
    line_end: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'kind': self.kind,
            'statement': self.statement,
            'binders': self.binders,
            'status': self.status,
            'detail': self.detail,
            'numerical_support': None if not math.isfinite(self.support) else float(self.support),
        }


def _vectorised(fn):
    def wrapped(x):
        arr = np.asarray(x, dtype=float)
        try:
            return np.vectorize(fn, otypes=[float])(arr)
        except Exception:
            return np.full_like(arr, np.nan)
    return wrapped


_STRICT_MODULE = {
    'exp': np.exp,
    'log': np.log,
    'sqrt': np.sqrt,
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
    'erf': _vectorised(math.erf),
    'gamma': _vectorised(math.gamma),
    'pi': np.pi,
    'E': np.e,
}


def _strict_evaluate(expr: sp.Expr, X: np.ndarray,
                     var_names: Sequence[str]) -> Optional[np.ndarray]:
    """Evaluate with true mathematical domains (no abs/log fallbacks).

    Finite samples are the only ones a conjecture may rely on: if negating a
    variable leaves the law undefined, no symmetry claim is generated.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    try:
        fn = sp.lambdify([sp.Symbol(v) for v in var_names], expr,
                         modules=[_STRICT_MODULE])
        with np.errstate(all='ignore'):
            out = fn(*[X[:, i] for i in range(len(var_names))])
        arr = np.asarray(out, dtype=float)
        if arr.ndim == 0:
            arr = np.full(X.shape[0], float(arr))
        return np.broadcast_to(arr, (X.shape[0],)).copy()
    except Exception:
        return None


def _normalised_error(pred: np.ndarray, alt: np.ndarray, tol: float = 1e-6) -> Optional[float]:
    pred = np.asarray(pred, dtype=float)
    alt = np.asarray(alt, dtype=float)
    if pred.shape != alt.shape:
        return None
    finite = np.isfinite(pred) & np.isfinite(alt)
    if finite.sum() < 3:
        return None
    scale = max(float(np.max(np.abs(alt[finite]))), 1e-12)
    return float(np.max(np.abs(pred[finite] - alt[finite])) / scale)


class LeanVerifier:
    """Generates Lean 4 conjectures for a law and checks them with the kernel.

    Parameters
    ----------
    project_dir:
        A Lake project that depends on Mathlib.  When given (or when the
        current directory is a Lake project) Lean is invoked through
        ``lake env lean`` so Mathlib is on ``LEAN_PATH``.
    lean_cmd:
        Explicit command prefix, e.g. ``["lake", "env", "lean"]``.
    timeout:
        Seconds before a verification run is abandoned.
    """

    def __init__(self,
                 project_dir: Optional[str] = None,
                 imports: Sequence[str] = ("Mathlib",),
                 timeout: float = 180.0,
                 max_hypotheses: int = 6,
                 tol: float = 1e-6,
                 lean_cmd: Optional[Sequence[str]] = None):
        self.project_dir = project_dir
        self.imports = list(imports)
        self.timeout = float(timeout)
        self.max_hypotheses = max_hypotheses
        self.tol = tol
        self.lean_cmd = list(lean_cmd) if lean_cmd else None
        self._version: Optional[str] = None

    # ------------------------------------------------------------------
    # Toolchain discovery
    # ------------------------------------------------------------------
    def _command(self, file: Path) -> Optional[Tuple[List[str], Optional[str]]]:
        if self.lean_cmd is not None:
            return list(self.lean_cmd) + [str(file)], self.project_dir
        lean = shutil.which("lean")
        lake = shutil.which("lake")
        candidates = []
        if self.project_dir:
            candidates.append(Path(self.project_dir))
        candidates.append(Path.cwd())
        if lake:
            for cwd in candidates:
                if (cwd / 'lakefile.toml').exists() or (cwd / 'lakefile.lean').exists():
                    return [lake, 'env', 'lean', str(file)], str(cwd)
        if lean:
            return [lean, str(file)], self.project_dir
        return None

    def toolchain_version(self) -> Optional[str]:
        if self._version is not None:
            return self._version or None
        lean = shutil.which("lean")
        if not lean:
            self._version = ""
            return None
        try:
            proc = subprocess.run([lean, '--version'], capture_output=True,
                                  text=True, timeout=30)
            self._version = (proc.stdout or proc.stderr).strip()
        except Exception:
            self._version = ""
        return self._version or None

    def available(self) -> bool:
        return self._command(Path("probe.lean")) is not None

    # ------------------------------------------------------------------
    # Hypothesis generation
    # ------------------------------------------------------------------
    def generate_hypotheses(self, law, X: Optional[np.ndarray] = None) -> List[LeanHypothesis]:
        expr = law.sympy_expr
        if expr is None:
            return []
        vars_ = list(law.var_names)
        symbol_map = lean_symbol_map(vars_)
        # Translation of the law body is a prerequisite for every conjecture.
        sympy_to_lean(expr, vars_, symbol_map)
        if sp.count_ops(expr) > 90:
            return []

        lean_vars = [symbol_map[v] for v in vars_]
        hyps: List[LeanHypothesis] = []

        def law_call(args: Sequence[str]) -> str:
            return "law " + " ".join(args)

        if X is not None:
            X = np.atleast_2d(np.asarray(X, dtype=float))
            pred = _strict_evaluate(expr, X, vars_)

        def add(name, kind, statement, tactics, binders="", support=float('nan')):
            hyps.append(LeanHypothesis(
                name=name, kind=kind, statement=statement,
                tactics=tactics, binders=binders, support=support,
            ))

        # 1. Axis values: a concrete numeric identity that also checks the
        #    translation itself.
        axis = self._axis_hypothesis(expr, vars_, lean_vars, law_call)
        if axis is not None:
            add('axis_value', 'axis_value', axis[0], axis[1], support=axis[2])

        # 2. Symmetries and permutations.
        if X is not None and pred is not None:
            for j, var in enumerate(vars_):
                X_neg = X.copy()
                X_neg[:, j] *= -1.0
                alt = _strict_evaluate(expr, X_neg, vars_)
                if alt is None:
                    continue
                err_even = _normalised_error(pred, alt, self.tol)
                err_odd = _normalised_error(pred, -alt, self.tol)
                args_neg = list(lean_vars)
                args_neg[j] = f"(-{lean_vars[j]})"
                binders = " (" + " ".join(lean_vars) + " : ℝ)"
                if err_even is not None and err_even <= self.tol:
                    add(f'even_{lean_vars[j]}', 'symmetry',
                        f"{law_call(args_neg)} = {law_call(lean_vars)}",
                        "unfold law\n  first | ring_nf | (simp; ring)",
                        binders, err_even)
                elif err_odd is not None and err_odd <= self.tol:
                    add(f'odd_{lean_vars[j]}', 'symmetry',
                        f"{law_call(args_neg)} = - {law_call(lean_vars)}",
                        "unfold law\n  first | ring_nf | (simp; ring)",
                        binders, err_odd)

            pairs = [(i, j) for i in range(len(vars_)) for j in range(i + 1, len(vars_))][:2]
            for i, j in pairs:
                X_swapped = X.copy()
                X_swapped[:, [i, j]] = X_swapped[:, [j, i]]
                alt = _strict_evaluate(expr, X_swapped, vars_)
                if alt is None:
                    continue
                err = _normalised_error(pred, alt, self.tol)
                if err is None or err > self.tol:
                    continue
                args = list(lean_vars)
                args[i], args[j] = args[j], args[i]
                binders = " (" + " ".join(lean_vars) + " : ℝ)"
                add(f'symmetric_{lean_vars[i]}_{lean_vars[j]}', 'permutation',
                    f"{law_call(args)} = {law_call(lean_vars)}",
                    "unfold law\n  first | ring_nf | (simp; ring; ring_nf)",
                    binders, err)

        # 3. Homogeneity.
        if X is not None and pred is not None:
            hom = self._homogeneity_hypothesis(expr, X, pred, vars_, lean_vars, law_call)
            if hom is not None:
                add('homogeneous', 'homogeneity', hom[0], hom[1], hom[2], hom[3])

        # 4. Positivity on the positive orthant.
        if X is not None and pred is not None and _positivity_friendly(expr):
            finite = np.isfinite(pred) & np.all(np.isfinite(X), axis=1)
            if finite.sum() >= 3 and np.all(X[finite] > 0) and np.all(pred[finite] > 0):
                binders = " (" + " ".join(lean_vars) + " : ℝ)" + "".join(
                    f" (h{lean_vars[k]} : 0 < {lean_vars[k]})" for k in range(len(vars_)))
                add('positive', 'positivity',
                    f"0 < {law_call(lean_vars)}",
                    "unfold law\n  positivity",
                    binders, float(np.min(pred[finite])))

        # 5. Derivative rule (univariate laws only).
        if len(vars_) == 1:
            deriv = self._derivative_hypothesis(expr, vars_[0])
            if deriv is not None:
                add('derivative_rule', 'derivative', deriv[0], deriv[1],
                    " (x : ℝ)", float('nan'))

        return hyps[: self.max_hypotheses]

    def _axis_hypothesis(self, expr, vars_, lean_vars, law_call):
        # Prefer a unit point with a clean rational value on a law without
        # transcendental atoms; otherwise use the origin when the law vanishes
        # there.  Anything else is not reliably provable by norm_num/simp.
        unit = self._evaluate_point(expr, vars_, 1.0)
        if unit is not None and _is_clean_rational(unit) and (
                abs(unit) == 0.0 or not _has_transcendental(expr)):
            args = [f"({_real_literal(1.0)})" for _ in vars_]
            return (f"{law_call(args)} = {_real_literal(unit)}",
                    "first | (unfold law; norm_num) | (simp [law]) | norm_num",
                    abs(unit))
        origin = self._evaluate_point(expr, vars_, 0.0)
        if origin is not None and abs(origin) == 0.0:
            args = [f"({_real_literal(0.0)})" for _ in vars_]
            return (f"{law_call(args)} = (0 : ℝ)",
                    "first | (unfold law; norm_num) | (simp [law]) | (unfold law; simp)",
                    0.0)
        return None

    @staticmethod
    def _evaluate_point(expr, vars_, point: float) -> Optional[float]:
        subs = {sp.Symbol(v): sp.Float(point) for v in vars_}
        try:
            value = complex(sp.N(expr.subs(subs), 15))
        except Exception:
            return None
        if abs(value.imag) > 1e-9 or not math.isfinite(value.real):
            return None
        if abs(value.real) > 1e12:
            return None
        return float(value.real)

    def _homogeneity_hypothesis(self, expr, X, pred, vars_, lean_vars, law_call):
        scale = 1.75
        mask = np.isfinite(pred) & (np.abs(pred) > 1e-9)
        if mask.sum() < 3:
            return None
        X_scaled = X * scale
        alt = _strict_evaluate(expr, X_scaled, vars_)
        if alt is None:
            return None
        mask &= np.isfinite(alt) & (np.abs(alt) > 1e-9)
        if mask.sum() < 3:
            return None
        same_sign = np.all(np.sign(pred[mask]) == np.sign(alt[mask]))
        if not same_sign:
            return None
        ratios = np.abs(alt[mask] / pred[mask])
        k_values = np.log(ratios) / math.log(scale)
        mean_k = float(np.mean(k_values))
        if not np.all(np.abs(k_values - mean_k) <= 1e-4):
            return None
        snapped = float(snap_exponent(mean_k))
        if abs(snapped - mean_k) > 0.02 or abs(snapped) < 1e-9:
            return None
        frac = sp.Rational(snapped).limit_denominator(24)
        scale = next(n for n in ('s', 's1', 's2', 's3', 's4') if n not in lean_vars)
        exponent = _pow_text(scale, frac)
        expr_rhs = exponent + f" * {law_call(lean_vars)}"
        scaled_args = [f"({scale} * {v})" for v in lean_vars]
        binders = (f" ({scale} " + " ".join(lean_vars) + " : ℝ)"
                   + f" (h{scale} : 0 < {scale})")
        tactics = (
            "unfold law\n"
            "  first | ring_nf | (field_simp; ring)"
            f" | (rw [abs_mul, abs_of_pos h{scale}, Real.sqrt_mul (by positivity)]; ring_nf)"
            f" | (simp [abs_mul, abs_of_pos h{scale}]; ring_nf)"
        )
        statement = f"{law_call(scaled_args)} = {expr_rhs}"
        return (statement, tactics, binders,
                float(np.max(np.abs(k_values - mean_k))))

    def _derivative_hypothesis(self, expr, var):
        x = sp.Symbol(var)
        try:
            d = sp.diff(expr, x)
            if sp.count_ops(d) > 90:
                return None
            body = sympy_to_lean(expr, [var], {var: 't'})
            dbody = sympy_to_lean(d, [var], {var: 'x'})
        except Exception:
            return None
        statement = f"deriv (fun t => {body}) x = {dbody}"
        tactics = "first | compute_deriv | (simp [deriv]; ring_nf) | (simp [deriv])"
        return statement, tactics

    # ------------------------------------------------------------------
    # Lean source + verification
    # ------------------------------------------------------------------
    def build_source(self, law, hypotheses: Sequence[LeanHypothesis]) -> Tuple[str, List[LeanHypothesis]]:
        symbol_map = lean_symbol_map(law.var_names)
        body = sympy_to_lean(law.sympy_expr, law.var_names, symbol_map)
        vars_ = [symbol_map[v] for v in law.var_names]
        binders = " (" + " ".join(vars_) + " : ℝ)"
        lines = [
            "-- Generated by The1: formal hypothesis checks for a discovered law.",
            f"-- law: {law.str_repr}",
        ]
        lines.extend(f"import {m}" for m in self.imports)
        lines.extend([
            "",
            "noncomputable section",
            "",
            f"def law{binders} : ℝ := {body}",
            "",
        ])
        placed: List[LeanHypothesis] = []
        for h in hypotheses:
            start = len(lines) + 1
            lines.append(f"theorem {h.name}{h.binders} : {h.statement} := by")
            lines.append("  " + h.tactics)
            lines.append("")
            h.line_start, h.line_end = start, len(lines)
            placed.append(h)
        lines.append("end")
        lines.append("")
        return "\n".join(lines), placed

    def verify_law(self,
                   law,
                   X: Optional[np.ndarray] = None,
                   output_dir: Optional[str] = None,
                   keep_file: bool = False) -> Dict[str, Any]:
        try:
            hypotheses = self.generate_hypotheses(law, X)
        except LeanTranslationError as exc:
            return {
                'status': 'unsupported',
                'engine': 'lean4',
                'error': str(exc),
                'hypotheses': [],
                'proven': 0,
                'total': 0,
            }
        return self.verify(law, hypotheses, output_dir=output_dir, keep_file=keep_file)

    def verify(self,
               law,
               hypotheses: Sequence[LeanHypothesis],
               output_dir: Optional[str] = None,
               keep_file: bool = False) -> Dict[str, Any]:
        if not hypotheses:
            return {
                'status': 'skipped',
                'engine': 'lean4',
                'error': 'no checkable hypothesis was generated',
                'hypotheses': [],
                'proven': 0,
                'total': 0,
            }
        source, placed = self.build_source(law, hypotheses)
        report: Dict[str, Any] = {
            'status': 'pending',
            'engine': 'lean4',
            'toolchain': self.toolchain_version(),
            'source': source,
            'hypotheses': placed,
            'proven': 0,
            'total': len(placed),
            'file': None,
            'error': None,
        }
        cmd = self._command(Path("LawVerification.lean"))
        if cmd is None:
            for h in placed:
                h.status = 'unverified'
                h.detail = 'Lean toolchain not found (install elan + Mathlib)'
            report['status'] = 'unavailable'
            report['hypotheses'] = [h.to_dict() for h in placed]
            return report

        workdir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix='the1_lean_'))
        workdir.mkdir(parents=True, exist_ok=True)
        path = workdir / 'LawVerification.lean'
        path.write_text(source, encoding='utf-8')
        command, cwd = cmd
        command = [c if c != 'LawVerification.lean' else str(path) for c in command]
        report['file'] = str(path)

        try:
            proc = subprocess.run(command, cwd=cwd, capture_output=True,
                                  text=True, timeout=self.timeout)
        except subprocess.TimeoutExpired:
            for h in placed:
                h.status = 'timeout'
                h.detail = f'Lean timed out after {self.timeout:.0f}s'
            report['status'] = 'timeout'
            report['hypotheses'] = [h.to_dict() for h in placed]
            return report
        except OSError as exc:
            report['status'] = 'error'
            report['error'] = str(exc)
            report['hypotheses'] = [h.to_dict() for h in placed]
            return report

        output = (proc.stdout or '') + (proc.stderr or '')
        errors, warnings = _parse_diagnostics(output)
        unmapped: List[str] = []
        for line_no, message in errors:
            target = next((h for h in placed if h.line_start <= line_no <= h.line_end), None)
            if target is None:
                unmapped.append(message)
            elif target.status == 'pending':
                target.status = 'unproven'
                target.detail = message
        for h in placed:
            if h.status == 'pending':
                h.status = 'proven'
                h.detail = ''
        if warnings and not errors:
            report['warnings'] = warnings

        proven = sum(1 for h in placed if h.status == 'proven')
        report['proven'] = proven
        report['hypotheses'] = [h.to_dict() for h in placed]
        if proven == len(placed):
            report['status'] = 'verified'
        elif proven > 0:
            report['status'] = 'partial'
        else:
            report['status'] = 'failed'
        if unmapped:
            # An error outside every theorem (bad import, syntax error before the
            # theorems) invalidates the whole run, whatever the per-line mapping
            # suggested.
            report['status'] = 'error'
            report['error'] = unmapped[0]
            for h in placed:
                if h.status == 'proven':
                    h.status = 'unverified'
                    h.detail = 'Lean reported a file-level error'
            report['hypotheses'] = [h.to_dict() for h in placed]
            report['proven'] = sum(1 for h in placed if h.status == 'proven')
        elif proc.returncode != 0 and not errors:
            report['status'] = 'error'
            report['error'] = (output.strip() or f'lean exited {proc.returncode}')[-500:]
        if not keep_file and output_dir is None:
            try:
                path.unlink()
                workdir.rmdir()
                report['file'] = None
            except OSError:
                pass
        return report


def _parse_diagnostics(output: str) -> Tuple[List[Tuple[int, str]], List[Tuple[int, str]]]:
    errors: List[Tuple[int, str]] = []
    warnings: List[Tuple[int, str]] = []
    pattern = re.compile(r'^(?P<file>.*?):(?P<line>\d+):(?P<col>\d+): '
                         r'(?P<kind>error|warning): (?P<msg>.*)$')
    for raw in output.splitlines():
        match = pattern.match(raw.strip())
        if not match:
            continue
        entry = (int(match.group('line')), match.group('msg').strip())
        if match.group('kind') == 'error':
            errors.append(entry)
        else:
            warnings.append(entry)
    return errors, warnings


__all__ = [
    'LeanVerifier', 'LeanHypothesis', 'LeanTranslationError',
    'sympy_to_lean', 'lean_identifier',
]
