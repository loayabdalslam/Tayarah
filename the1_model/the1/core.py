"""
The1 core: inductive neuro-symbolic world model engine.

Discovery is performed by a portfolio of generic, data-driven searches
(sparse additive regression, sparse log-linear regression, free-exponent power
laws, rational functions with target transforms, and an evolutionary search).
No named physical equation is hard-coded.
"""

from __future__ import annotations

import copy
import math
import random
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import sympy as sp

from .constants import snap_constant
from .grammar import (
    ALL_UNARY_OPS, BinaryOpNode, ConstantNode, Node, UnaryOpNode, VariableNode,
    sympy_to_node, safe_div, safe_exp, safe_log, safe_sqrt, safe_pow,
)
from .search import Candidate, GenericSearch, evaluate_expr, r2_rmse


class SymbolicLaw:
    """A discovered symbolic law with metrics, optional AST, and SymPy form."""

    def __init__(self,
                 sympy_expr: sp.Expr,
                 latex_str: str,
                 str_repr: str,
                 ast_tree: Optional[Node],
                 params: np.ndarray,
                 var_names: List[str],
                 train_r2: float,
                 val_r2: Optional[float] = None,
                 train_rmse: float = 0.0,
                 val_rmse: Optional[float] = None,
                 complexity: int = 1,
                 occam_score: float = 0.0,
                 source: str = '',
                 metadata: Optional[Dict[str, Any]] = None,
                 verification: Optional[Dict[str, Any]] = None):
        self.sympy_expr = sympy_expr
        self.latex_str = latex_str
        self.str_repr = str_repr
        self.ast_tree = ast_tree
        self.params = params
        self.var_names = var_names
        self.train_r2 = train_r2
        self.val_r2 = val_r2
        self.train_rmse = train_rmse
        self.val_rmse = val_rmse
        self.complexity = complexity
        self.occam_score = occam_score
        self.source = source
        self.metadata = metadata or {}
        self.verification = verification or {}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Evaluate the discovered law on new inputs X of shape (N, D)."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if self.ast_tree is not None:
            try:
                preds = self.ast_tree.evaluate(X, self.params)
                return np.nan_to_num(preds, nan=0.0, posinf=1e9, neginf=-1e9)
            except Exception:
                pass
        preds = evaluate_expr(self.sympy_expr, X, self.var_names)
        if preds is None:
            return np.zeros(X.shape[0])
        return preds

    def __repr__(self):
        val_str = f", Val R²={self.val_r2:.4f}" if self.val_r2 is not None else ""
        return (f"<SymbolicLaw: {self.str_repr} (Train R²={self.train_r2:.4f}"
                f"{val_str}, Complexity={self.complexity}, Source={self.source})>")


class The1WorldModel:
    """
    The1: inductive world model for extreme few-shot symbolic law discovery.

    Parameters
    ----------
    max_complexity:
        Hard ceiling on AST/SymPy complexity for accepted laws.
    max_terms:
        Maximum number of additive/log-linear terms in sparse searches.
    use_evolutionary:
        Enable the grammar-wide evolutionary fallback.
    use_rational / use_transforms:
        Enable rational-function search and target transformations.
    law_library:
        Optional cumulative ``LawLibrary`` whose laws become candidate priors.
    remember:
        If true and a library is attached, append the discovered law to it.
    """

    def __init__(self,
                 max_complexity: int = 25,
                 population_size: int = 60,
                 generations: int = 20,
                 tournament_size: int = 3,
                 mutation_rate: float = 0.35,
                 crossover_rate: float = 0.65,
                 occam_weight: float = 0.01,
                 rational_snap_tol: float = 0.04,
                 random_state: int = 42,
                 min_train_r2: float = 0.85,
                 max_terms: int = 6,
                 max_library: int = 360,
                 use_evolutionary: bool = True,
                 use_rational: bool = True,
                 use_transforms: bool = True,
                 law_library: Optional[Any] = None,
                 remember: bool = False,
                 max_search_samples: int = 4000,
                 verify_with_lean: bool = False,
                 lean_project: Optional[str] = None,
                 lean_timeout: float = 180.0):
        self.max_complexity = max_complexity
        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.occam_weight = occam_weight
        self.rational_snap_tol = rational_snap_tol
        self.random_state = random_state
        self.min_train_r2 = min_train_r2
        self.use_evolutionary = use_evolutionary
        self.law_library = law_library
        self.remember = remember
        self.max_search_samples = max_search_samples
        self.verify_with_lean = verify_with_lean
        self.lean_project = lean_project
        self.lean_timeout = lean_timeout

        self.rng = np.random.RandomState(random_state)
        random.seed(random_state)
        self.search = GenericSearch(
            max_terms=max_terms,
            max_library=max_library,
            min_train_r2=min_train_r2,
            random_state=random_state,
            enable_rational=use_rational,
            enable_transforms=use_transforms,
        )
        self.discovered_law_: Optional[SymbolicLaw] = None
        self.pareto_frontier_: List[SymbolicLaw] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            var_names: Optional[List[str]] = None,
            verbose: bool = False) -> SymbolicLaw:
        start_time = time.time()
        X_train = np.atleast_2d(np.asarray(X_train, dtype=float))
        y_train = np.asarray(y_train, dtype=float).ravel()
        n_samples, n_features = X_train.shape
        if var_names is None:
            var_names = [f"x{i+1}" for i in range(n_features)]
        if len(var_names) != n_features:
            raise ValueError("var_names length must match the number of columns in X_train")

        # Drop non-finite rows rather than silently poisoning the search.
        finite = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train)
        if not finite.all():
            X_train, y_train = X_train[finite], y_train[finite]
            n_samples = X_train.shape[0]
        if n_samples < 2:
            raise ValueError("At least two finite training observations are required")

        X_search, y_search = self._subsample(X_train, y_train)
        candidates: List[Candidate] = self.search.generate(X_search, y_search, var_names)
        candidates.sort(key=lambda c: c.score(), reverse=True)

        if self.law_library is not None:
            try:
                candidates.extend(self.law_library.candidates(X_search, y_search, var_names))
            except Exception:
                pass

        has_exact = any(c.train_r2 > 0.9995 for c in candidates)
        if self.use_evolutionary and not has_exact:
            for expr, details in self._evolutionary_candidates(X_search, y_search, var_names):
                cand = self.search._finalize(expr, 'evolutionary', X_search, y_search, var_names, details)
                if cand is not None:
                    candidates.append(cand)

        laws: List[SymbolicLaw] = []
        for cand in candidates:
            if cand.complexity > self.max_complexity:
                continue
            law = self._build_symbolic_law(
                cand.expr, cand.source, var_names,
                X_train, y_train, X_val, y_val,
                complexity=cand.complexity, metadata=cand.details,
            )
            if law is not None:
                laws.append(law)

        if not laws:
            mean_y = float(np.mean(y_train))
            law = self._build_symbolic_law(
                sp.Float(mean_y), 'constant_fallback', var_names,
                X_train, y_train, X_val, y_val, complexity=1,
            )
            if law is None:
                raise RuntimeError("Failed to construct even a constant fallback law")
            laws.append(law)

        # When a validation set exists, a law that cannot be evaluated on it is
        # not comparable with laws that can.  Drop such candidates unless every
        # candidate failed validation.
        if X_val is not None and y_val is not None:
            validated = [law for law in laws if law.val_r2 is not None]
            if validated:
                laws = validated

        laws.sort(key=self._law_selection_score, reverse=True)
        self.discovered_law_ = laws[0]
        self._simplify_winner(self.discovered_law_)
        try:
            self.discovered_law_.latex_str = sp.latex(self.discovered_law_.sympy_expr)
        except Exception:
            self.discovered_law_.latex_str = str(self.discovered_law_.sympy_expr)
        self.pareto_frontier_ = self._extract_pareto_frontier(laws)

        if self.verify_with_lean:
            self.discovered_law_.verification = self._verify_with_lean(
                self.discovered_law_, X_train)

        if self.law_library is not None and self.remember:
            try:
                self.law_library.add(self.discovered_law_, X_train, y_train)
            except Exception:
                pass

        if verbose:
            elapsed = time.time() - start_time
            print(f"[The1] Discovery finished in {elapsed:.3f}s")
            print(f"Law: {self.discovered_law_.str_repr}  (source={self.discovered_law_.source})")
            print(f"Train R²: {self.discovered_law_.train_r2:.6f}")
            if self.discovered_law_.val_r2 is not None:
                print(f"Unseen Val R²: {self.discovered_law_.val_r2:.6f}")

        return self.discovered_law_

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------
    def _verify_with_lean(self, law: SymbolicLaw, X: np.ndarray) -> Dict[str, Any]:
        try:
            from .lean4 import LeanVerifier
            verifier = LeanVerifier(project_dir=self.lean_project,
                                    timeout=self.lean_timeout)
            sample = X if len(X) <= 500 else X[:500]
            report = verifier.verify_law(law, sample)
        except Exception as exc:
            return {'status': 'error', 'engine': 'lean4', 'error': str(exc),
                    'hypotheses': [], 'proven': 0, 'total': 0}
        if report.get('status') == 'unavailable':
            report['hint'] = (
                'Install Lean 4 + Mathlib (elan, lake new the1_lean math, then '
                'lake exe cache get) and pass lean_project=<path> to enable proofs.'
            )
        return report

    def _subsample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = X.shape[0]
        if n <= self.max_search_samples:
            return X, y
        idx = np.linspace(0, n - 1, self.max_search_samples).astype(int)
        # A uniform grid can alias periodic data; mix in a random sample.
        rng = np.random.RandomState(self.random_state)
        extra = rng.choice(n, size=min(n, self.max_search_samples // 2), replace=False)
        idx = np.unique(np.concatenate([idx, extra]))
        return X[idx], y[idx]

    # ------------------------------------------------------------------
    # Law construction / selection
    # ------------------------------------------------------------------
    def _build_symbolic_law(self,
                            expr: sp.Expr,
                            source: str,
                            var_names: List[str],
                            X_train: np.ndarray,
                            y_train: np.ndarray,
                            X_val: Optional[np.ndarray],
                            y_val: Optional[np.ndarray],
                            complexity: Optional[int] = None,
                            metadata: Optional[Dict[str, Any]] = None) -> Optional[SymbolicLaw]:
        try:
            train_preds = evaluate_expr(expr, X_train, var_names)
            if train_preds is None:
                return None
            train_r2, train_rmse = r2_rmse(y_train, train_preds)

            val_r2 = None
            val_rmse = None
            if X_val is not None and y_val is not None and len(y_val) > 0:
                X_val_arr = np.atleast_2d(np.asarray(X_val, dtype=float))
                y_val_arr = np.asarray(y_val, dtype=float).ravel()
                val_preds = evaluate_expr(expr, X_val_arr, var_names)
                if val_preds is not None:
                    val_r2, val_rmse = r2_rmse(y_val_arr, val_preds)

            comp = int(complexity if complexity is not None
                       else (sp.count_ops(expr) + len(expr.free_symbols)))
            ast_tree = None
            try:
                ast_tree = sympy_to_node(expr, var_names)
            except Exception:
                ast_tree = None
            occam = (val_r2 if val_r2 is not None and np.isfinite(val_r2) else train_r2) \
                - self.occam_weight * comp
            meta = dict(metadata or {})
            meta.setdefault('n_train', int(len(y_train)))
            meta.setdefault('y_var', float(np.var(y_train)) + 1e-12)

            return SymbolicLaw(
                sympy_expr=expr,
                latex_str='',
                str_repr=str(expr),
                ast_tree=ast_tree,
                params=np.array([]),
                var_names=list(var_names),
                train_r2=float(train_r2),
                val_r2=(float(val_r2) if val_r2 is not None and np.isfinite(val_r2) else None),
                train_rmse=float(train_rmse),
                val_rmse=(float(val_rmse) if val_rmse is not None and np.isfinite(val_rmse) else None),
                complexity=comp,
                occam_score=float(occam),
                source=source,
                metadata=meta,
            )
        except Exception:
            return None

    def _law_selection_score(self, law: SymbolicLaw) -> Tuple[float, int]:
        if law.val_r2 is not None and np.isfinite(law.val_r2):
            # Rank on generalisation first (rounded so numerically equivalent
            # fits tie), then prefer the simplest expression among near-ties.
            # A raw ``val_r2 - w*complexity`` score could rank a 98.9% law above
            # an exact 100% law purely because the former is shorter.
            return (round(float(law.val_r2), 5), -law.complexity)

        # No validation set: use a BIC-like normalized-error criterion.  This
        # strongly prefers an exact compact law over an approximate one.
        n_eff = max(2, int(law.metadata.get('n_train', 2)))
        y_var = max(float(law.metadata.get('y_var', 1.0)), 1e-15)
        mse_norm = (law.train_rmse ** 2) / y_var
        bic = n_eff * math.log(max(mse_norm, 1e-15)) + law.complexity * math.log(max(n_eff, 2))
        return (-bic, -law.complexity)

    @staticmethod
    def _clean_floats(expr: sp.Expr) -> sp.Expr:
        """Drop float noise such as ``1.0*x**1.0`` without touching real constants."""
        def repl(e):
            if isinstance(e, sp.Pow) and e.exp.is_Float:
                exact = sp.Rational(float(e.exp)).limit_denominator(24)
                if abs(float(exact) - float(e.exp)) <= 1e-9:
                    return sp.Pow(e.base, exact)
                return e
            if e.is_Float:
                value = float(e)
                if abs(value - round(value)) <= 1e-12 * max(1.0, abs(value)):
                    return sp.Integer(int(round(value)))
            return e

        return expr.replace(
            lambda e: (isinstance(e, sp.Pow) and e.exp.is_Float) or e.is_Float,
            repl,
        )

    @staticmethod
    def _simplify_winner(law: SymbolicLaw) -> None:
        """Canonicalise only the winning expression; search stays fast."""
        expr = law.sympy_expr
        try:
            if sp.count_ops(expr) > 80:
                return
            expr = The1WorldModel._clean_floats(expr)
            simple = sp.cancel(sp.simplify(expr))
            if simple is None or simple.free_symbols != expr.free_symbols:
                return
            if sp.count_ops(simple) > sp.count_ops(expr):
                return
            law.sympy_expr = simple
            law.str_repr = str(simple)
            law.complexity = int(sp.count_ops(simple) + len(simple.free_symbols))
            try:
                law.ast_tree = sympy_to_node(simple, law.var_names)
            except Exception:
                pass
        except Exception:
            return

    def _extract_pareto_frontier(self, laws: List[SymbolicLaw]) -> List[SymbolicLaw]:
        frontier = []
        for l1 in laws:
            dominated = False
            for l2 in laws:
                if l2 is not l1 and l2.train_r2 >= l1.train_r2 and l2.complexity < l1.complexity:
                    dominated = True
                    break
            if not dominated and l1 not in frontier:
                frontier.append(l1)
        frontier.sort(key=lambda x: x.complexity)
        return frontier

    # ------------------------------------------------------------------
    # Evolutionary fallback (grammar-wide)
    # ------------------------------------------------------------------
    def _evolutionary_candidates(self,
                                 X: np.ndarray,
                                 y: np.ndarray,
                                 var_names: Sequence[str]) -> List[Tuple[sp.Expr, Dict[str, Any]]]:
        n_features = X.shape[1]
        population: List[Node] = [self._random_tree(n_features, max_depth=2)
                                  for _ in range(self.population_size)]
        best: List[Tuple[Node, float, float]] = []

        for _ in range(self.generations):
            scored = []
            for ind in population:
                tree, params, nmse, r2 = self._quick_evaluate(ind, X, y)
                scored.append((tree, params, nmse, r2, tree.complexity()))
            scored.sort(key=lambda item: item[2] + self.occam_weight * item[4])
            best.append((scored[0][0], scored[0][2], scored[0][3]))
            if scored[0][3] > 0.9995:
                break

            elite_count = max(2, int(self.population_size * 0.1))
            new_pop = [scored[i][0].clone() for i in range(elite_count)]
            while len(new_pop) < self.population_size:
                p1 = self._tournament_select(scored)
                if self.rng.rand() < self.crossover_rate:
                    p2 = self._tournament_select(scored)
                    child = self._crossover(p1, p2)
                else:
                    child = p1.clone()
                if self.rng.rand() < self.mutation_rate:
                    child = self._mutate(child, n_features)
                if child.complexity() <= self.max_complexity:
                    new_pop.append(child)
                else:
                    new_pop.append(p1.clone())
            population = new_pop

        out: List[Tuple[sp.Expr, Dict[str, Any]]] = []
        for tree, nmse, r2 in best[:6]:
            try:
                expr = tree.to_sympy(list(var_names), np.ones(tree.get_num_params()))
                expr = self._refine_linear_scale(expr, X, y, var_names)
                out.append((expr, {'gp_r2': float(r2), 'gp_nmse': float(nmse)}))
            except Exception:
                continue
        return out

    @staticmethod
    def _refine_linear_scale(expr: sp.Expr,
                             X: np.ndarray,
                             y: np.ndarray,
                             var_names: Sequence[str]) -> sp.Expr:
        pred = evaluate_expr(expr, X, var_names)
        if pred is None:
            return expr
        A = np.column_stack([np.ones(len(pred)), pred])
        try:
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        except Exception:
            return expr
        a, b = float(coef[0]), float(coef[1])
        if not np.isfinite(a) or not np.isfinite(b) or abs(b) < 1e-12:
            return expr
        a = snap_constant(a)
        b = snap_constant(b)
        refined = b * expr
        if abs(a) > 1e-10:
            refined = a + refined
        return sp.simplify(refined)

    def _quick_evaluate(self, tree: Node, X: np.ndarray, y: np.ndarray) -> Tuple[Node, np.ndarray, float, float]:
        n_params = tree.get_num_params()
        params = np.ones(n_params)
        try:
            pred = tree.evaluate(X, params)
            # Fit an affine correction; this lets a structure such as sqrt(L)
            # become 2.006*sqrt(L) without hard-coding any constant.
            A = np.column_stack([np.ones(len(pred)), pred])
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            pred = A @ coef
            ss_res = np.sum((y - pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2) + 1e-12
            r2 = 1.0 - (ss_res / ss_tot)
            nmse = ss_res / ss_tot
            if np.isnan(nmse) or np.isinf(nmse):
                nmse, r2 = 1e6, -1.0
        except Exception:
            nmse, r2 = 1e6, -1.0
        return tree, params, float(nmse), float(r2)

    def _tournament_select(self, scored_pop):
        sample_indices = self.rng.choice(len(scored_pop), size=min(self.tournament_size, len(scored_pop)), replace=False)
        best = min([scored_pop[i] for i in sample_indices],
                   key=lambda x: x[2] + self.occam_weight * x[4])
        return best[0]

    def _crossover(self, parent1: Node, parent2: Node) -> Node:
        p1 = parent1.clone()
        p2 = parent2.clone()
        nodes1 = self._collect_nodes(p1)
        nodes2 = self._collect_nodes(p2)
        if not nodes1 or not nodes2:
            return p1
        target = self.rng.choice(nodes1)
        source = self.rng.choice(nodes2)
        return self._replace_node(p1, target, source.clone())

    def _mutate(self, tree: Node, n_features: int) -> Node:
        t = tree.clone()
        nodes = self._collect_nodes(t)
        if not nodes:
            return t
        target = self.rng.choice(nodes)
        if isinstance(target, ConstantNode) and self.rng.rand() < 0.5:
            target.value = float(target.value * self.rng.choice([0.5, 2.0, -1.0, 1.0 / 3.0, 3.0]))
            return t
        sub = self._random_tree(n_features, max_depth=1)
        return self._replace_node(t, target, sub)

    def _collect_nodes(self, root: Node) -> List[Node]:
        nodes = [root]
        if isinstance(root, UnaryOpNode):
            nodes.extend(self._collect_nodes(root.child))
        elif isinstance(root, BinaryOpNode):
            nodes.extend(self._collect_nodes(root.left))
            nodes.extend(self._collect_nodes(root.right))
        return nodes

    def _replace_node(self, root: Node, target: Node, replacement: Node) -> Node:
        if root is target:
            return replacement
        if isinstance(root, UnaryOpNode):
            root.child = self._replace_node(root.child, target, replacement)
        elif isinstance(root, BinaryOpNode):
            root.left = self._replace_node(root.left, target, replacement)
            root.right = self._replace_node(root.right, target, replacement)
        return root

    def _random_tree(self, n_features: int, max_depth: int) -> Node:
        if max_depth <= 1:
            if self.rng.rand() < 0.7:
                return VariableNode(self.rng.randint(0, n_features))
            return ConstantNode(float(self.rng.choice([0.5, 1.0, 2.0, -1.0, 3.0, 10.0])))

        if self.rng.rand() < 0.45:
            op = self.rng.choice(ALL_UNARY_OPS)
            return UnaryOpNode(op, self._random_tree(n_features, max_depth - 1))
        op = self.rng.choice(['+', '-', '*', '/', '^'])
        left = self._random_tree(n_features, max_depth - 1)
        right = self._random_tree(n_features, max_depth - 1)
        return BinaryOpNode(op, left, right)


# Canonical aliases
The1Model = The1WorldModel
