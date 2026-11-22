"""MathBenchmarkSuite: a broad, cross-domain benchmark of closed-form mathematics.

The corpus spans algebra, trigonometry, exponential/logarithmic functions,
hyperbolic functions, special functions, geometry, calculus, linear algebra,
probability and sequences.  Every law is generated numerically from a reference
implementation; no benchmark expression is ever injected into the discovery
engine.

Each law carries a ``compositional`` flag.  Compositional laws require nested
function composition (e.g. ``sqrt(a**2 + b**2)`` or ``x * sin(x)``) that the
current search grammar does not synthesise directly.  They are reported
separately so that the headline accuracy is not inflated by easy cases.
"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

from .benchmark_suite import NaturalLawBenchmark


class MathLawBenchmark(NaturalLawBenchmark):
    """A benchmark law with an explicit composability annotation."""

    def __init__(self, *args, compositional: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.compositional = compositional


def _law(name, category, target_name, var_names, formula_str, eval_fn,
         domain, ood_domain, description="", compositional=False,
         target_positive=False):
    law = MathLawBenchmark(
        name=name,
        category=category,
        target_name=target_name,
        var_names=var_names,
        formula_str=formula_str,
        eval_fn=eval_fn,
        domain=domain,
        ood_domain=ood_domain,
        description=description,
        compositional=compositional,
    )
    law.target_positive = target_positive
    return law


class MathBenchmarkSuite:
    """Cross-domain corpus of mathematical identities and closed forms."""

    @staticmethod
    def get_all_laws() -> Dict[str, MathLawBenchmark]:
        laws: Dict[str, MathLawBenchmark] = {}

        def add(key, *args, **kwargs):
            laws[key] = _law(*args, **kwargs)

        # ------------------------------------------------------------------
        # Algebra
        # ------------------------------------------------------------------
        add("algebra_quadratic", "Quadratic Polynomial", "Algebra", "y", ["x"],
            "3*x^2 + 2*x + 1",
            lambda X: 3.0 * X[:, 0] ** 2 + 2.0 * X[:, 0] + 1.0,
            [(0.5, 4.0)], [(4.0, 8.0)],
            "General quadratic with non-zero constant term.")
        add("algebra_cubic", "Cubic Polynomial", "Algebra", "y", ["x"],
            "2*x^3 - x",
            lambda X: 2.0 * X[:, 0] ** 3 - X[:, 0],
            [(0.2, 2.0)], [(2.0, 4.0)],
            "Odd cubic with mixed-sign outputs.")
        add("algebra_fractional_power", "Fractional Power", "Algebra", "y", ["x"],
            "3*x^(3/2)",
            lambda X: 3.0 * np.power(X[:, 0], 1.5),
            [(0.5, 3.0)], [(3.0, 6.0)],
            "Non-integer power law.")
        add("algebra_rational_simple", "Shifted Reciprocal", "Algebra", "y", ["x"],
            "1/(x + 2)",
            lambda X: 1.0 / (X[:, 0] + 2.0),
            [(0.5, 3.0)], [(3.0, 6.0)],
            "Rational function with a shifted pole.")
        add("algebra_parallel", "Parallel Resistance", "Algebra", "R", ["r1", "r2"],
            "r1*r2/(r1 + r2)",
            lambda X: X[:, 0] * X[:, 1] / (X[:, 0] + X[:, 1]),
            [(1.0, 5.0), (1.0, 5.0)], [(5.0, 10.0), (5.0, 10.0)],
            "Harmonic combination of two positive quantities.")
        add("algebra_sum_squares", "Sum of Squares", "Algebra", "s", ["x", "y"],
            "x^2 + y^2",
            lambda X: X[:, 0] ** 2 + X[:, 1] ** 2,
            [(0.5, 4.0), (0.5, 4.0)], [(4.0, 8.0), (4.0, 8.0)],
            "Squared Euclidean norm in two dimensions.")
        add("algebra_product_sums", "Product of Affine Factors", "Algebra", "y", ["x", "y"],
            "(x + 1)*(y + 2)",
            lambda X: (X[:, 0] + 1.0) * (X[:, 1] + 2.0),
            [(0.5, 4.0), (0.5, 4.0)], [(4.0, 8.0), (4.0, 8.0)],
            "Expanded bilinear polynomial with offsets.")
        add("algebra_sqrt_sum", "Sum of Square Roots", "Algebra", "y", ["x", "y"],
            "sqrt(x) + sqrt(y)",
            lambda X: np.sqrt(X[:, 0]) + np.sqrt(X[:, 1]),
            [(0.5, 4.0), (0.5, 4.0)], [(4.0, 8.0), (4.0, 8.0)],
            "Additive radical expression.")
        add("algebra_cbrt", "Cube Root", "Algebra", "y", ["x"],
            "cbrt(x)",
            lambda X: np.cbrt(X[:, 0]),
            [(0.5, 5.0)], [(5.0, 10.0)],
            "Real cube root on a positive domain.")
        add("algebra_mixed_monomial", "Mixed Monomial", "Algebra", "y", ["x", "y"],
            "2*x^2*y",
            lambda X: 2.0 * X[:, 0] ** 2 * X[:, 1],
            [(0.5, 3.0), (0.5, 3.0)], [(3.0, 6.0), (3.0, 6.0)],
            "Multi-variable monomial with integer exponents.")

        # ------------------------------------------------------------------
        # Trigonometry
        # ------------------------------------------------------------------
        add("trig_sine", "Sine Wave", "Trigonometry", "y", ["x"],
            "2.5*sin(x)",
            lambda X: 2.5 * np.sin(X[:, 0]),
            [(-3.0, 3.0)], [(-6.0, 6.0)],
            "Scaled sine over a mixed-sign domain.")
        add("trig_cosine", "Cosine Wave", "Trigonometry", "y", ["x"],
            "-4*cos(x)",
            lambda X: -4.0 * np.cos(X[:, 0]),
            [(-3.0, 3.0)], [(-6.0, 6.0)],
            "Negatively scaled cosine.")
        add("trig_tangent", "Tangent", "Trigonometry", "y", ["x"],
            "tan(x)",
            lambda X: np.tan(X[:, 0]),
            [(-1.3, 1.3)], [(1.3, 1.45)],
            "Tangent away from its poles.")
        add("trig_sin_squared", "Sine Squared", "Trigonometry", "y", ["x"],
            "sin(x)^2",
            lambda X: np.sin(X[:, 0]) ** 2,
            [(-3.0, 3.0)], [(-6.0, 6.0)],
            "Squared sine identity.")
        add("trig_cos_squared", "Cosine Squared", "Trigonometry", "y", ["x"],
            "cos(x)^2",
            lambda X: np.cos(X[:, 0]) ** 2,
            [(-3.0, 3.0)], [(-6.0, 6.0)],
            "Squared cosine identity.")
        add("trig_sin_plus_cos", "Sine Plus Cosine", "Trigonometry", "y", ["x"],
            "sin(x) + cos(x)",
            lambda X: np.sin(X[:, 0]) + np.cos(X[:, 0]),
            [(-3.0, 3.0)], [(-6.0, 6.0)],
            "First Fourier mode with equal amplitudes.")
        add("trig_product", "Product of Sines", "Trigonometry", "y", ["x", "y"],
            "sin(x)*sin(y)",
            lambda X: np.sin(X[:, 0]) * np.sin(X[:, 1]),
            [(-1.5, 1.5), (-1.5, 1.5)], [(-3.0, 3.0), (-3.0, 3.0)],
            "Separable trigonometric product of two angles.")
        add("trig_arctan", "Arctangent", "Trigonometry", "y", ["x"],
            "2*atan(x)",
            lambda X: 2.0 * np.arctan(X[:, 0]),
            [(-4.0, 4.0)], [(-8.0, 8.0)],
            "Scaled inverse tangent.")
        add("trig_arcsin", "Arcsine", "Trigonometry", "y", ["x"],
            "asin(x)",
            lambda X: np.arcsin(X[:, 0]),
            [(-0.9, 0.9)], [(0.9, 0.99)],
            "Inverse sine on its principal domain.")
        add("trig_arccos", "Arccosine", "Trigonometry", "y", ["x"],
            "2*acos(x)",
            lambda X: 2.0 * np.arccos(X[:, 0]),
            [(-0.9, 0.9)], [(0.9, 0.99)],
            "Scaled inverse cosine.")

        # ------------------------------------------------------------------
        # Exponential and logarithmic
        # ------------------------------------------------------------------
        add("exp_growth", "Exponential Growth", "Exponential & Log", "y", ["x"],
            "2*exp(0.5*x)",
            lambda X: 2.0 * np.exp(0.5 * X[:, 0]),
            [(0.0, 2.0)], [(2.0, 4.0)],
            "Exponential growth with a fractional rate.", target_positive=True)
        add("exp_decay", "Exponential Decay", "Exponential & Log", "y", ["x"],
            "5*exp(-0.3*x)",
            lambda X: 5.0 * np.exp(-0.3 * X[:, 0]),
            [(0.0, 5.0)], [(5.0, 10.0)],
            "Decaying exponential.", target_positive=True)
        add("exp_sum", "Exponential of a Sum", "Exponential & Log", "y", ["x", "y"],
            "exp(x + y)",
            lambda X: np.exp(X[:, 0] + X[:, 1]),
            [(0.0, 1.5), (0.0, 1.5)], [(1.5, 3.0), (1.5, 3.0)],
            "Exponential of a linear combination.", target_positive=True)
        add("log_natural", "Natural Logarithm", "Exponential & Log", "y", ["x"],
            "3*log(x)",
            lambda X: 3.0 * np.log(X[:, 0]),
            [(0.5, 5.0)], [(5.0, 10.0)],
            "Scaled natural logarithm.")
        add("log_sum", "Weighted Log Sum", "Exponential & Log", "y", ["x", "y"],
            "log(x) + 2*log(y)",
            lambda X: np.log(X[:, 0]) + 2.0 * np.log(X[:, 1]),
            [(0.5, 5.0), (0.5, 5.0)], [(5.0, 10.0), (5.0, 10.0)],
            "Linear combination of logarithms.")
        add("log_base10", "Base-10 Logarithm", "Exponential & Log", "y", ["x"],
            "log10(x)",
            lambda X: np.log10(X[:, 0]),
            [(0.5, 5.0)], [(5.0, 10.0)],
            "Decadic logarithm, a rescaled natural log.")
        add("exp_plus_one", "Shifted Exponential", "Exponential & Log", "y", ["x"],
            "exp(x) + 1",
            lambda X: np.exp(X[:, 0]) + 1.0,
            [(0.0, 2.0)], [(2.0, 4.0)],
            "Exponential with an additive offset.", target_positive=True)
        add("log_one_plus", "Logarithm of One Plus", "Exponential & Log", "y", ["x"],
            "log(1 + x)",
            lambda X: np.log1p(X[:, 0]),
            [(0.5, 5.0)], [(5.0, 10.0)],
            "Shifted logarithm.")

        # ------------------------------------------------------------------
        # Hyperbolic
        # ------------------------------------------------------------------
        add("hyper_sinh", "Hyperbolic Sine", "Hyperbolic", "y", ["x"],
            "2*sinh(x)",
            lambda X: 2.0 * np.sinh(X[:, 0]),
            [(-2.0, 2.0)], [(-3.0, 3.0)],
            "Scaled hyperbolic sine.")
        add("hyper_cosh", "Hyperbolic Cosine", "Hyperbolic", "y", ["x"],
            "3*cosh(x)",
            lambda X: 3.0 * np.cosh(X[:, 0]),
            [(-2.0, 2.0)], [(-3.0, 3.0)],
            "Scaled hyperbolic cosine.", target_positive=True)
        add("hyper_tanh", "Hyperbolic Tangent", "Hyperbolic", "y", ["x"],
            "4*tanh(x)",
            lambda X: 4.0 * np.tanh(X[:, 0]),
            [(-3.0, 3.0)], [(-6.0, 6.0)],
            "Scaled sigmoidal hyperbolic tangent.")
        add("hyper_tanh_squared", "Squared Hyperbolic Tangent", "Hyperbolic", "y", ["x"],
            "tanh(x)^2",
            lambda X: np.tanh(X[:, 0]) ** 2,
            [(-3.0, 3.0)], [(-6.0, 6.0)],
            "Squared hyperbolic tangent.", target_positive=True)
        add("hyper_asinh", "Inverse Hyperbolic Sine", "Hyperbolic", "y", ["x"],
            "asinh(x)",
            lambda X: np.arcsinh(X[:, 0]),
            [(-3.0, 3.0)], [(-6.0, 6.0)],
            "Inverse hyperbolic sine on the real line.")
        add("hyper_atanh", "Inverse Hyperbolic Tangent", "Hyperbolic", "y", ["x"],
            "atanh(x)",
            lambda X: np.arctanh(X[:, 0]),
            [(-0.9, 0.9)], [(0.9, 0.99)],
            "Inverse hyperbolic tangent inside its domain.")

        # ------------------------------------------------------------------
        # Special functions
        # ------------------------------------------------------------------
        add("special_erf", "Error Function", "Special Functions", "y", ["x"],
            "2*erf(x)",
            lambda X: 2.0 * _erf(X[:, 0]),
            [(-2.0, 2.0)], [(-4.0, 4.0)],
            "Scaled Gauss error function.")
        add("special_logistic", "Logistic Sigmoid", "Special Functions", "y", ["x"],
            "1/(1 + exp(-x))",
            lambda X: 1.0 / (1.0 + np.exp(-X[:, 0])),
            [(-4.0, 4.0)], [(-8.0, 8.0)],
            "Standard logistic function.", target_positive=True)
        add("special_log1p_abs", "Log One Plus Absolute", "Special Functions", "y", ["x"],
            "log(1 + |x|)",
            lambda X: np.log1p(np.abs(X[:, 0])),
            [(-4.0, 4.0)], [(-8.0, 8.0)],
            "Soft absolute-value logarithm.")
        add("special_softplus", "Softplus", "Special Functions", "y", ["x"],
            "log(1 + exp(x))",
            lambda X: np.log1p(np.exp(X[:, 0])),
            [(-3.0, 3.0)], [(-6.0, 6.0)],
            "Smooth rectifier requiring nested composition.",
            compositional=True, target_positive=True)
        add("special_gamma", "Gamma Function", "Special Functions", "y", ["x"],
            "gamma(x)",
            lambda X: _gamma(X[:, 0]),
            [(0.5, 4.0)], [(4.0, 6.0)],
            "Euler gamma function, outside the current feature library.",
            compositional=True, target_positive=True)

        # ------------------------------------------------------------------
        # Geometry
        # ------------------------------------------------------------------
        add("geo_circle_area", "Circle Area", "Geometry", "A", ["r"],
            "pi*r^2",
            lambda X: math.pi * X[:, 0] ** 2,
            [(0.5, 4.0)], [(4.0, 8.0)],
            "Area of a circle.", target_positive=True)
        add("geo_sphere_volume", "Sphere Volume", "Geometry", "V", ["r"],
            "(4/3)*pi*r^3",
            lambda X: (4.0 / 3.0) * math.pi * X[:, 0] ** 3,
            [(0.5, 3.0)], [(3.0, 6.0)],
            "Volume of a sphere.", target_positive=True)
        add("geo_sphere_surface", "Sphere Surface Area", "Geometry", "S", ["r"],
            "4*pi*r^2",
            lambda X: 4.0 * math.pi * X[:, 0] ** 2,
            [(0.5, 3.0)], [(3.0, 6.0)],
            "Surface area of a sphere.", target_positive=True)
        add("geo_cylinder_volume", "Cylinder Volume", "Geometry", "V", ["r", "h"],
            "pi*r^2*h",
            lambda X: math.pi * X[:, 0] ** 2 * X[:, 1],
            [(0.5, 3.0), (1.0, 5.0)], [(3.0, 6.0), (5.0, 10.0)],
            "Volume of a right circular cylinder.", target_positive=True)
        add("geo_cone_volume", "Cone Volume", "Geometry", "V", ["r", "h"],
            "(1/3)*pi*r^2*h",
            lambda X: (math.pi / 3.0) * X[:, 0] ** 2 * X[:, 1],
            [(0.5, 3.0), (1.0, 5.0)], [(3.0, 6.0), (5.0, 10.0)],
            "Volume of a right circular cone.", target_positive=True)
        add("geo_triangle_area", "Triangle Area", "Geometry", "A", ["b", "h"],
            "0.5*b*h",
            lambda X: 0.5 * X[:, 0] * X[:, 1],
            [(0.5, 5.0), (0.5, 5.0)], [(5.0, 10.0), (5.0, 10.0)],
            "Area of a triangle given base and height.", target_positive=True)
        add("geo_hypotenuse", "Pythagorean Hypotenuse", "Geometry", "c", ["a", "b"],
            "sqrt(a^2 + b^2)",
            lambda X: np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2),
            [(0.5, 5.0), (0.5, 5.0)], [(5.0, 10.0), (5.0, 10.0)],
            "Radical of a sum of squares, requiring composition.",
            compositional=True, target_positive=True)

        # ------------------------------------------------------------------
        # Calculus
        # ------------------------------------------------------------------
        add("calc_derivative_power", "Derivative of a Cubic", "Calculus", "f'", ["x"],
            "3*x^2 + 4",
            lambda X: 3.0 * X[:, 0] ** 2 + 4.0,
            [(0.5, 4.0)], [(4.0, 8.0)],
            "Derivative of x^3 + 4x.")
        add("calc_derivative_log", "Derivative of Log", "Calculus", "f'", ["x"],
            "1/x",
            lambda X: 1.0 / X[:, 0],
            [(0.5, 5.0)], [(5.0, 10.0)],
            "Derivative of the natural logarithm.", target_positive=True)
        add("calc_derivative_sqrt", "Derivative of Sqrt", "Calculus", "f'", ["x"],
            "1/(2*sqrt(x))",
            lambda X: 1.0 / (2.0 * np.sqrt(X[:, 0])),
            [(0.5, 5.0)], [(5.0, 10.0)],
            "Derivative of the square root.", target_positive=True)
        add("calc_product_rule", "Product Rule", "Calculus", "f'", ["x"],
            "2*x*sin(x) + x^2*cos(x)",
            lambda X: 2.0 * X[:, 0] * np.sin(X[:, 0]) + X[:, 0] ** 2 * np.cos(X[:, 0]),
            [(0.5, 3.0)], [(3.0, 5.0)],
            "Derivative of x^2 sin(x), needing a composed product.",
            compositional=True)
        add("calc_chain_rule", "Chain Rule", "Calculus", "f'", ["x"],
            "2*x*exp(x^2)",
            lambda X: 2.0 * X[:, 0] * np.exp(X[:, 0] ** 2),
            [(0.2, 1.5)], [(1.5, 2.5)],
            "Derivative of exp(x^2), needing nested composition.",
            compositional=True, target_positive=True)

        # ------------------------------------------------------------------
        # Linear algebra
        # ------------------------------------------------------------------
        add("linalg_dot_product", "Dot Product", "Linear Algebra", "s", ["a", "b", "c", "d"],
            "a*c + b*d",
            lambda X: X[:, 0] * X[:, 2] + X[:, 1] * X[:, 3],
            [(0.5, 3.0)] * 4, [(3.0, 6.0)] * 4,
            "Inner product of two two-dimensional vectors.")
        add("linalg_norm_squared", "Squared Norm", "Linear Algebra", "s",
            ["a", "b", "c"],
            "a^2 + b^2 + c^2",
            lambda X: X[:, 0] ** 2 + X[:, 1] ** 2 + X[:, 2] ** 2,
            [(0.5, 3.0)] * 3, [(3.0, 6.0)] * 3,
            "Squared Euclidean norm in three dimensions.", target_positive=True)
        add("linalg_determinant2", "2x2 Determinant", "Linear Algebra", "det",
            ["a", "b", "c", "d"],
            "a*d - b*c",
            lambda X: X[:, 0] * X[:, 3] - X[:, 1] * X[:, 2],
            [(0.5, 3.0)] * 4, [(3.0, 6.0)] * 4,
            "Determinant of a two-by-two matrix.")
        add("linalg_quadratic_form", "Quadratic Form", "Linear Algebra", "q", ["x", "y"],
            "x^2 + 2*x*y + y^2",
            lambda X: X[:, 0] ** 2 + 2.0 * X[:, 0] * X[:, 1] + X[:, 1] ** 2,
            [(0.5, 3.0), (0.5, 3.0)], [(3.0, 6.0), (3.0, 6.0)],
            "Positive-definite quadratic form.")
        add("linalg_frobenius", "Frobenius Norm", "Linear Algebra", "n",
            ["a", "b", "c"],
            "sqrt(a^2 + b^2 + c^2)",
            lambda X: np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2 + X[:, 2] ** 2),
            [(0.5, 3.0)] * 3, [(3.0, 6.0)] * 3,
            "Euclidean norm requiring a composed radical.",
            compositional=True, target_positive=True)

        # ------------------------------------------------------------------
        # Probability and statistics
        # ------------------------------------------------------------------
        add("prob_gaussian_pdf", "Gaussian Density", "Probability", "p", ["x"],
            "exp(-x^2/2)/sqrt(2*pi)",
            lambda X: np.exp(-X[:, 0] ** 2 / 2.0) / math.sqrt(2.0 * math.pi),
            [(-3.0, 3.0)], [(-6.0, 6.0)],
            "Standard normal probability density.", target_positive=True)
        add("prob_uniform_mean", "Uniform Mean", "Probability", "mu", ["a", "b"],
            "(a + b)/2",
            lambda X: 0.5 * (X[:, 0] + X[:, 1]),
            [(0.0, 5.0), (0.0, 5.0)], [(5.0, 10.0), (5.0, 10.0)],
            "Mean of a uniform distribution.")
        add("prob_variance_two", "Variance of Two Values", "Probability", "v",
            ["x", "y"],
            "((x - y)/2)^2",
            lambda X: ((X[:, 0] - X[:, 1]) / 2.0) ** 2,
            [(0.0, 5.0), (0.0, 5.0)], [(5.0, 10.0), (5.0, 10.0)],
            "Population variance of two observations, requiring a composed square.",
            compositional=True, target_positive=True)

        # ------------------------------------------------------------------
        # Sequences and series
        # ------------------------------------------------------------------
        add("series_triangular", "Triangular Numbers", "Sequences & Series", "S", ["n"],
            "n*(n + 1)/2",
            lambda X: X[:, 0] * (X[:, 0] + 1.0) / 2.0,
            [(1.0, 10.0)], [(10.0, 20.0)],
            "Sum of the first n positive integers.")
        add("series_arithmetic_sum", "Arithmetic Series Sum", "Sequences & Series", "S",
            ["n", "a1", "an"],
            "n*(a1 + an)/2",
            lambda X: X[:, 0] * (X[:, 1] + X[:, 2]) / 2.0,
            [(1.0, 10.0), (0.5, 3.0), (3.0, 10.0)],
            [(10.0, 20.0), (3.0, 6.0), (10.0, 20.0)],
            "Sum of an arithmetic progression.")
        add("series_geometric_sum", "Geometric Series Sum", "Sequences & Series", "S",
            ["a", "r", "n"],
            "a*(r^n - 1)/(r - 1)",
            lambda X: X[:, 0] * (X[:, 1] ** X[:, 2] - 1.0) / (X[:, 1] - 1.0),
            [(0.5, 3.0), (1.2, 2.0), (1.0, 6.0)],
            [(3.0, 6.0), (2.0, 3.0), (6.0, 10.0)],
            "Finite geometric series, requiring composed exponentiation.",
            compositional=True, target_positive=True)

        return laws

    # ------------------------------------------------------------------
    # Convenience views
    # ------------------------------------------------------------------
    @staticmethod
    def get_discoverable() -> Dict[str, MathLawBenchmark]:
        return {k: v for k, v in MathBenchmarkSuite.get_all_laws().items()
                if not v.compositional}

    @staticmethod
    def get_compositional() -> Dict[str, MathLawBenchmark]:
        return {k: v for k, v in MathBenchmarkSuite.get_all_laws().items()
                if v.compositional}

    @staticmethod
    def categories() -> List[str]:
        seen: List[str] = []
        for law in MathBenchmarkSuite.get_all_laws().values():
            if law.category not in seen:
                seen.append(law.category)
        return seen


def _erf(x):
    try:
        from scipy.special import erf
        return erf(x)
    except Exception:  # pragma: no cover
        return np.vectorize(math.erf)(x)


def _gamma(x):
    try:
        from scipy.special import gamma
        return gamma(x)
    except Exception:  # pragma: no cover
        return np.vectorize(math.gamma)(x)


__all__ = ['MathLawBenchmark', 'MathBenchmarkSuite']
