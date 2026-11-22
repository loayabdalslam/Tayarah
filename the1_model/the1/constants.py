"""Generic numerical constant recognition.

No law-specific constants live here.  The recogniser only asks whether a fitted
number is close to a *low-complexity* rational (continued-fraction reconstruction)
so that coefficients such as 0.5, 1/3, 4/3 or 120 print cleanly.
"""

from fractions import Fraction
import math
import numpy as np

try:
    import sympy as sp
except Exception:  # pragma: no cover
    sp = None


def rationalize(value: float,
                max_denominator: int = 1000,
                rel_tol: float = 1e-6,
                abs_tol: float = 1e-9) -> float:
    """Return a nearby low-denominator rational as float, or the input."""
    if value is None:
        return value
    try:
        value = float(value)
    except Exception:
        return value
    if not math.isfinite(value) or abs(value) < 1e-15:
        return 0.0
    if abs(value) < 1e-6:
        # Very small physical constants (e.g. G) would otherwise collapse to 0.
        return value
    frac = Fraction(value).limit_denominator(max_denominator)
    if abs(float(frac) - value) <= max(abs_tol, rel_tol * abs(value)):
        return float(frac)
    return value


def rationalize_exponent(value: float,
                         max_denominator: int = 24,
                         rel_tol: float = 2e-2) -> float:
    """Exponents get a looser tolerance: 0.4998 should become 1/2."""
    if value is None:
        return value
    try:
        value = float(value)
    except Exception:
        return value
    if not math.isfinite(value) or abs(value) < 1e-9:
        return 0.0
    frac = Fraction(value).limit_denominator(max_denominator)
    if abs(float(frac) - value) <= max(1e-7, rel_tol * max(1.0, abs(value))):
        return float(frac)
    return value


def snap_constant(value: float) -> float:
    """Snap a coefficient to a clean rational when this is numerically safe."""
    return rationalize(value, max_denominator=1000, rel_tol=1e-5)


def snap_exponent(value: float) -> float:
    return rationalize_exponent(value, max_denominator=24, rel_tol=2e-2)


def nsimplify_symbolic(value: float, tolerance: float = 1e-8):
    """Optionally identify pi / e / simple radicals for display.

    This is only used for pretty-printing; the evaluable AST keeps the numeric
    value, so a display constant can never silently change model predictions.
    """
    if sp is None or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return None
    try:
        candidate = sp.nsimplify(
            float(value),
            [sp.pi, sp.E, sp.sqrt(2), sp.sqrt(3), sp.sqrt(5), sp.GoldenRatio],
            rational=True,
            tolerance=tolerance,
        )
        if candidate.free_symbols:
            return candidate
    except Exception:
        pass
    return None


def is_close(a: float, b: float, rel_tol: float = 1e-6, abs_tol: float = 1e-12) -> bool:
    return abs(a - b) <= max(abs_tol, rel_tol * max(abs(a), abs(b)))


def clean_vector(values, max_denominator: int = 1000) -> np.ndarray:
    return np.asarray([rationalize(v, max_denominator=max_denominator) for v in np.ravel(values)])


__all__ = [
    'rationalize', 'rationalize_exponent', 'snap_constant', 'snap_exponent',
    'nsimplify_symbolic', 'is_close', 'clean_vector',
]
