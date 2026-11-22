"""
Mathematical grammar and expression trees for The1.

The grammar is deliberately broad: algebra, trigonometry, hyperbolic functions,
exponentials, logarithms in several bases, roots, absolute value, and a small
set of special functions (erf, gamma, logistic, softplus).  Search algorithms
choose operators from these families, so new mathematics can be covered by
extending the operator table rather than by writing equation templates.
"""

import math
import numpy as np
import sympy as sp
from scipy import special as _special
from typing import List, Dict, Any, Optional, Union

# ---------------------------------------------------------------------------
# Safe numerical operations
# ---------------------------------------------------------------------------
def safe_div(a, b):
    with np.errstate(divide='ignore', invalid='ignore'):
        denom = np.where(np.abs(b) < 1e-9, np.sign(b) * 1e-9 + (b == 0) * 1e-9, b)
        res = a / denom
        return np.nan_to_num(res, nan=1e9, posinf=1e9, neginf=-1e9)


def safe_exp(a):
    with np.errstate(over='ignore', under='ignore'):
        return np.exp(np.clip(a, -60.0, 60.0))


def safe_log(a):
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.log(np.maximum(np.abs(a), 1e-12))


def safe_sqrt(a):
    return np.sqrt(np.maximum(a, 0.0))


def safe_pow(a, b):
    """Power that preserves sign for integer exponents and stays finite otherwise."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    with np.errstate(invalid='ignore', over='ignore', under='ignore', divide='ignore'):
        rounded = np.round(b)
        if np.all(np.isfinite(b)) and np.all(np.abs(b - rounded) < 1e-9) and np.all(np.abs(rounded) <= 50):
            res = np.power(a, rounded.astype(np.int64))
        else:
            res = np.power(np.abs(a) + 1e-12, np.clip(b, -20.0, 20.0))
    return np.nan_to_num(res, nan=0.0, posinf=1e9, neginf=-1e9)


def safe_inv_sqr(a):
    return 1.0 / np.maximum(a ** 2, 1e-12)


def safe_lorentz(v):
    v_norm = np.clip(np.abs(v), 0.0, 0.9999)
    return 1.0 / np.sqrt(np.maximum(1.0 - v_norm ** 2, 1e-8))


def safe_tan(a):
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        res = np.tan(a)
    return np.clip(np.nan_to_num(res, nan=0.0, posinf=1e9, neginf=-1e9), -1e9, 1e9)


def safe_asin(a):
    return np.arcsin(np.clip(a, -1.0, 1.0))


def safe_acos(a):
    return np.arccos(np.clip(a, -1.0, 1.0))


def safe_acosh(a):
    return np.arccosh(np.maximum(a, 1.0))


def safe_atanh(a):
    eps = 1e-9
    return np.arctanh(np.clip(a, -1.0 + eps, 1.0 - eps))


def safe_sinh(a):
    with np.errstate(over='ignore'):
        return np.sinh(np.clip(a, -30.0, 30.0))


def safe_cosh(a):
    with np.errstate(over='ignore'):
        return np.cosh(np.clip(a, -30.0, 30.0))


def safe_gamma(a):
    with np.errstate(all='ignore'):
        res = _special.gamma(np.clip(a, -170.0, 170.0))
    return np.clip(np.nan_to_num(res, nan=0.0, posinf=1e9, neginf=-1e9), -1e9, 1e9)


def safe_erf(a):
    return _special.erf(a)


def safe_logistic(a):
    with np.errstate(over='ignore'):
        return 1.0 / (1.0 + np.exp(-np.clip(a, -60.0, 60.0)))


def safe_softplus(a):
    return np.log1p(np.exp(-np.abs(a))) + np.maximum(a, 0.0)


def safe_sech(a):
    return 1.0 / safe_cosh(a)


def safe_csch(a):
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        res = 1.0 / np.sinh(np.clip(a, -30.0, 30.0))
    return np.clip(np.nan_to_num(res, nan=0.0, posinf=1e9, neginf=-1e9), -1e9, 1e9)


def safe_cbrt(a):
    return np.cbrt(a)


# ---------------------------------------------------------------------------
# Operator families
# ---------------------------------------------------------------------------
# name -> (numpy evaluator, sympy builder, complexity)
UNARY_OPS: Dict[str, Any] = {
    # elementary algebra
    'neg': (np.negative, lambda x: -x, 1),
    'abs': (np.abs, sp.Abs, 1),
    'sqr': (lambda x: x ** 2, lambda x: x ** 2, 1),
    'cube': (lambda x: x ** 3, lambda x: x ** 3, 1),
    'inv': (lambda x: safe_div(1.0, x), lambda x: 1 / x, 2),
    'inv_sqr': (safe_inv_sqr, lambda x: 1 / (x ** 2), 2),
    'sqrt': (safe_sqrt, sp.sqrt, 2),
    'cbrt': (safe_cbrt, sp.cbrt, 2),

    # exponential / logarithmic
    'exp': (safe_exp, sp.exp, 2),
    'log': (safe_log, sp.log, 2),
    'log2': (lambda x: safe_log(x) / math.log(2.0), lambda x: sp.log(x, 2), 3),
    'log10': (lambda x: safe_log(x) / math.log(10.0), lambda x: sp.log(x, 10), 3),
    'logistic': (safe_logistic, lambda x: 1 / (1 + sp.exp(-x)), 4),
    'softplus': (safe_softplus, lambda x: sp.log(1 + sp.exp(x)), 4),

    # trigonometry
    'sin': (np.sin, sp.sin, 2),
    'cos': (np.cos, sp.cos, 2),
    'tan': (safe_tan, sp.tan, 3),
    'asin': (safe_asin, sp.asin, 3),
    'acos': (safe_acos, sp.acos, 3),
    'atan': (np.arctan, sp.atan, 3),

    # hyperbolic
    'sinh': (safe_sinh, sp.sinh, 3),
    'cosh': (safe_cosh, sp.cosh, 3),
    'tanh': (np.tanh, sp.tanh, 3),
    'asinh': (np.arcsinh, sp.asinh, 4),
    'acosh': (safe_acosh, sp.acosh, 4),
    'atanh': (safe_atanh, sp.atanh, 4),
    'sech': (safe_sech, sp.sech, 4),
    'csch': (safe_csch, sp.csch, 4),

    # special functions
    'erf': (safe_erf, sp.erf, 4),
    'gamma': (safe_gamma, sp.gamma, 4),

    # physics-inductive unary form (kept as an operator, not an equation template)
    'lorentz': (safe_lorentz, lambda x: 1 / sp.sqrt(1 - x ** 2), 4),
}

# Convenient families used by the search engine.  None of these encode a law;
# they are just candidate operators.
ALGEBRA_OPS = ['neg', 'abs', 'sqr', 'cube', 'inv', 'inv_sqr', 'sqrt', 'cbrt']
EXP_LOG_OPS = ['exp', 'log', 'log2', 'log10', 'logistic', 'softplus']
TRIG_OPS = ['sin', 'cos', 'tan', 'asin', 'acos', 'atan']
HYPERBOLIC_OPS = ['sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh', 'sech', 'csch']
SPECIAL_OPS = ['erf', 'gamma']
ALL_UNARY_OPS = list(UNARY_OPS.keys())

# Binary operators are handled explicitly in BinaryOpNode.
ADDITIVE_OPS = ['+', '-', '*', '/', '^']


# ---------------------------------------------------------------------------
# AST nodes
# ---------------------------------------------------------------------------
class Node:
    """Base class for AST nodes."""
    def evaluate(self, X: np.ndarray, params: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def to_sympy(self, var_names: List[str], params: np.ndarray) -> sp.Expr:
        raise NotImplementedError

    def complexity(self) -> int:
        raise NotImplementedError

    def get_num_params(self) -> int:
        raise NotImplementedError

    def clone(self) -> 'Node':
        raise NotImplementedError


class VariableNode(Node):
    def __init__(self, var_idx: int):
        self.var_idx = var_idx

    def evaluate(self, X: np.ndarray, params: np.ndarray) -> np.ndarray:
        return X[:, self.var_idx]

    def to_sympy(self, var_names: List[str], params: np.ndarray) -> sp.Expr:
        name = var_names[self.var_idx] if self.var_idx < len(var_names) else f"x{self.var_idx}"
        return sp.Symbol(name)

    def complexity(self) -> int:
        return 1

    def get_num_params(self) -> int:
        return 0

    def clone(self) -> 'VariableNode':
        return VariableNode(self.var_idx)

    def __repr__(self):
        return f"x_{self.var_idx}"


class ParamNode(Node):
    def __init__(self, param_idx: int):
        self.param_idx = param_idx

    def evaluate(self, X: np.ndarray, params: np.ndarray) -> np.ndarray:
        val = params[self.param_idx] if self.param_idx < len(params) else 1.0
        return np.full(X.shape[0], val)

    def to_sympy(self, var_names: List[str], params: np.ndarray) -> sp.Expr:
        val = float(params[self.param_idx]) if self.param_idx < len(params) else 1.0
        return _number_to_sympy(val)

    def complexity(self) -> int:
        return 1

    def get_num_params(self) -> int:
        return 1

    def clone(self) -> 'ParamNode':
        return ParamNode(self.param_idx)

    def __repr__(self):
        return f"c_{self.param_idx}"


class ConstantNode(Node):
    def __init__(self, value: float):
        self.value = value

    def evaluate(self, X: np.ndarray, params: np.ndarray) -> np.ndarray:
        return np.full(X.shape[0], self.value)

    def to_sympy(self, var_names: List[str], params: np.ndarray) -> sp.Expr:
        return _number_to_sympy(self.value)

    def complexity(self) -> int:
        return 1

    def get_num_params(self) -> int:
        return 0

    def clone(self) -> 'ConstantNode':
        return ConstantNode(self.value)

    def __repr__(self):
        return str(self.value)


class UnaryOpNode(Node):
    OP_MAP = UNARY_OPS

    def __init__(self, op_name: str, child: Node):
        if op_name not in self.OP_MAP:
            raise ValueError(f"Unknown unary operator: {op_name}")
        self.op_name = op_name
        self.child = child

    def evaluate(self, X: np.ndarray, params: np.ndarray) -> np.ndarray:
        val = self.child.evaluate(X, params)
        return self.OP_MAP[self.op_name][0](val)

    def to_sympy(self, var_names: List[str], params: np.ndarray) -> sp.Expr:
        child_expr = self.child.to_sympy(var_names, params)
        return self.OP_MAP[self.op_name][1](child_expr)

    def complexity(self) -> int:
        return self.OP_MAP[self.op_name][2] + self.child.complexity()

    def get_num_params(self) -> int:
        return self.child.get_num_params()

    def clone(self) -> 'UnaryOpNode':
        return UnaryOpNode(self.op_name, self.child.clone())

    def __repr__(self):
        return f"{self.op_name}({self.child})"


class BinaryOpNode(Node):
    def __init__(self, op_name: str, left: Node, right: Node):
        if op_name not in ADDITIVE_OPS:
            raise ValueError(f"Unknown binary operator: {op_name}")
        self.op_name = op_name
        self.left = left
        self.right = right

    def evaluate(self, X: np.ndarray, params: np.ndarray) -> np.ndarray:
        l = self.left.evaluate(X, params)
        r = self.right.evaluate(X, params)
        if self.op_name == '+':
            return l + r
        if self.op_name == '-':
            return l - r
        if self.op_name == '*':
            return l * r
        if self.op_name == '/':
            return safe_div(l, r)
        return safe_pow(l, r)

    def to_sympy(self, var_names: List[str], params: np.ndarray) -> sp.Expr:
        l_expr = self.left.to_sympy(var_names, params)
        r_expr = self.right.to_sympy(var_names, params)
        if self.op_name == '+':
            return l_expr + r_expr
        if self.op_name == '-':
            return l_expr - r_expr
        if self.op_name == '*':
            return l_expr * r_expr
        if self.op_name == '/':
            return l_expr / r_expr
        return l_expr ** r_expr

    def complexity(self) -> int:
        op_cost = 2 if self.op_name in ['/', '^'] else 1
        return op_cost + self.left.complexity() + self.right.complexity()

    def get_num_params(self) -> int:
        return self.left.get_num_params() + self.right.get_num_params()

    def clone(self) -> 'BinaryOpNode':
        return BinaryOpNode(self.op_name, self.left.clone(), self.right.clone())

    def __repr__(self):
        return f"({self.left} {self.op_name} {self.right})"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _number_to_sympy(val: float) -> sp.Expr:
    if not np.isfinite(val):
        return sp.Integer(0)
    if abs(val) < 1e-15:
        return sp.Integer(0)
    if abs(val) > 1e-4:
        rat = sp.Rational(val).limit_denominator(24)
        if abs(float(rat) - val) / (abs(val) + 1e-12) < 1e-4:
            return rat
    return sp.Float(val, 8)


def _combine(op: str, parts: List[Node]) -> Node:
    if not parts:
        raise ValueError("Cannot combine an empty node list")
    node = parts[0]
    for p in parts[1:]:
        node = BinaryOpNode(op, node, p)
    return node


def sympy_to_node(expr: sp.Expr, var_names: List[str]) -> Node:
    """Convert a supported SymPy expression into an evaluable AST."""
    if isinstance(expr, sp.Symbol):
        name = str(expr)
        if name in var_names:
            return VariableNode(var_names.index(name))
        raise ValueError(f"Unbound symbol {name}")
    if expr == sp.pi:
        return ConstantNode(float(sp.pi))
    if expr == sp.E:
        return ConstantNode(float(sp.E))
    if isinstance(expr, sp.Number):
        return ConstantNode(float(expr))
    if isinstance(expr, sp.Add):
        return _combine('+', [sympy_to_node(a, var_names) for a in expr.args])
    if isinstance(expr, sp.Mul):
        return _combine('*', [sympy_to_node(a, var_names) for a in expr.args])
    if isinstance(expr, sp.Pow):
        base, exponent = expr.args
        b = sympy_to_node(base, var_names)
        if isinstance(exponent, sp.Rational) and exponent.q in (2, 3):
            if exponent == sp.Rational(1, 2):
                return UnaryOpNode('sqrt', b)
            if exponent == sp.Rational(1, 3):
                return UnaryOpNode('cbrt', b)
            if exponent == sp.Rational(-1, 2):
                return UnaryOpNode('inv_sqr', b)
        if isinstance(exponent, sp.Integer):
            n = int(exponent)
            if n == 2:
                return UnaryOpNode('sqr', b)
            if n == 3:
                return UnaryOpNode('cube', b)
            if n == -1:
                return UnaryOpNode('inv', b)
        return BinaryOpNode('^', b, sympy_to_node(exponent, var_names))
    if isinstance(expr, sp.Abs):
        return UnaryOpNode('abs', sympy_to_node(expr.args[0], var_names))

    function_map = {
        sp.sin: 'sin', sp.cos: 'cos', sp.tan: 'tan',
        sp.asin: 'asin', sp.acos: 'acos', sp.atan: 'atan',
        sp.sinh: 'sinh', sp.cosh: 'cosh', sp.tanh: 'tanh',
        sp.asinh: 'asinh', sp.acosh: 'acosh', sp.atanh: 'atanh',
        sp.sech: 'sech', sp.csch: 'csch',
        sp.exp: 'exp', sp.log: 'log', sp.sqrt: 'sqrt', sp.cbrt: 'cbrt',
        sp.erf: 'erf', sp.gamma: 'gamma',
    }
    for fn, op_name in function_map.items():
        if expr.func == fn:
            return UnaryOpNode(op_name, sympy_to_node(expr.args[0], var_names))
    raise ValueError(f"Unsupported SymPy expression: {expr}")


__all__ = [
    'Node', 'VariableNode', 'ParamNode', 'ConstantNode', 'UnaryOpNode', 'BinaryOpNode',
    'safe_div', 'safe_exp', 'safe_log', 'safe_sqrt', 'safe_pow',
    'safe_tan', 'safe_asin', 'safe_acos', 'safe_acosh', 'safe_atanh',
    'safe_sinh', 'safe_cosh', 'safe_gamma', 'safe_erf', 'safe_logistic',
    'safe_softplus', 'safe_sech', 'safe_csch', 'safe_cbrt',
    'UNARY_OPS', 'ALGEBRA_OPS', 'EXP_LOG_OPS', 'TRIG_OPS', 'HYPERBOLIC_OPS',
    'SPECIAL_OPS', 'ALL_UNARY_OPS', 'ADDITIVE_OPS', 'sympy_to_node',
]
