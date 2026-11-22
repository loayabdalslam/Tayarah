"""Cumulative law library.

Every discovered law can be remembered and re-used as a candidate prior on
future datasets.  This is how the model grows from exploration: the library is
learned from data, not populated with hand-written equations.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import sympy as sp

from .search import Candidate, evaluate_expr, r2_rmse


class LawLibrary:
    def __init__(self, path: Optional[str] = None, min_r2: float = 0.90,
                 max_entries: int = 2000):
        self.path = Path(path) if path else None
        self.min_r2 = min_r2
        self.max_entries = max_entries
        self.entries: List[Dict[str, Any]] = []
        if self.path is not None and self.path.exists():
            self.load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                payload = payload.get("laws", [])
            if isinstance(payload, list):
                self.entries = [e for e in payload if isinstance(e, dict) and e.get("srepr")]
        except Exception:
            self.entries = []

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"laws": self.entries}, indent=2, ensure_ascii=False),
                             encoding="utf-8")

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------
    def add(self,
            law,
            X: Optional[np.ndarray] = None,
            y: Optional[np.ndarray] = None,
            metadata: Optional[Dict[str, Any]] = None) -> None:
        if law is None or law.sympy_expr is None:
            return
        if law.train_r2 < self.min_r2 and (law.val_r2 is None or law.val_r2 < self.min_r2):
            return
        try:
            srepr = sp.srepr(law.sympy_expr)
        except Exception:
            return
        key = srepr
        entry = {
            "id": f"law_{len(self.entries) + 1}",
            "srepr": srepr,
            "expr": str(law.sympy_expr),
            "latex": getattr(law, "latex_str", ""),
            "var_names": list(law.var_names),
            "source": getattr(law, "source", ""),
            "train_r2": float(law.train_r2),
            "val_r2": float(law.val_r2) if law.val_r2 is not None else None,
            "complexity": int(law.complexity),
            "domain": self._domain(X, law.var_names),
            "target_range": self._range(y),
            "metadata": metadata or {},
            "updated_at": time.time(),
        }
        for i, old in enumerate(self.entries):
            if old.get("srepr") == key:
                entry["id"] = old.get("id", entry["id"])
                entry["first_seen"] = old.get("first_seen", old.get("updated_at"))
                self.entries[i] = entry
                self.save()
                return
        entry["first_seen"] = entry["updated_at"]
        self.entries.append(entry)
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]
        self.save()

    def add_sympy(self, expr: sp.Expr, var_names: Sequence[str],
                  X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None,
                  source: str = "manual", metadata: Optional[Dict[str, Any]] = None) -> None:
        pred = evaluate_expr(expr, X, var_names) if X is not None and y is not None else None
        train_r2 = r2_rmse(y, pred)[0] if pred is not None else 1.0
        law = _SimpleLaw(expr, list(var_names), train_r2, source)
        self.add(law, X, y, metadata)

    @staticmethod
    def _domain(X: Optional[np.ndarray], var_names: Sequence[str]) -> Dict[str, List[float]]:
        if X is None:
            return {}
        X = np.atleast_2d(np.asarray(X, dtype=float))
        out = {}
        for j, name in enumerate(var_names):
            if j < X.shape[1]:
                out[name] = [float(np.nanmin(X[:, j])), float(np.nanmax(X[:, j]))]
        return out

    @staticmethod
    def _range(y: Optional[np.ndarray]) -> Optional[List[float]]:
        if y is None or len(y) == 0:
            return None
        y = np.asarray(y, dtype=float)
        return [float(np.nanmin(y)), float(np.nanmax(y))]

    # ------------------------------------------------------------------
    # Reuse
    # ------------------------------------------------------------------
    def candidates(self,
                   X: np.ndarray,
                   y: np.ndarray,
                   var_names: Sequence[str],
                   min_r2: Optional[float] = None) -> List[Candidate]:
        threshold = self.min_r2 if min_r2 is None else min_r2
        out: List[Candidate] = []
        seen = set()
        for entry in self.entries:
            try:
                expr = sp.sympify(entry["srepr"])
                stored_vars = list(entry.get("var_names", []))
                if len(stored_vars) != len(var_names):
                    # Same arity is not enough to safely rename variables.
                    if len(expr.free_symbols) != len(var_names):
                        continue
                    stored_syms = sorted(expr.free_symbols, key=str)
                    mapping = {s: sp.Symbol(n) for s, n in zip(stored_syms, var_names)}
                else:
                    mapping = {sp.Symbol(old): sp.Symbol(new)
                               for old, new in zip(stored_vars, var_names)}
                expr = expr.subs(mapping)
                key = sp.srepr(expr)
                if key in seen:
                    continue
                pred = evaluate_expr(expr, X, var_names)
                if pred is None:
                    continue
                r2, rmse = r2_rmse(y, pred)
                if not np.isfinite(r2) or r2 < threshold:
                    continue
                seen.add(key)
                comp = int(sp.count_ops(expr) + len(expr.free_symbols))
                out.append(Candidate(
                    expr=expr,
                    source=f"library:{entry.get('id', '?')}",
                    train_r2=float(r2),
                    train_rmse=float(rmse),
                    complexity=comp,
                    details={"library_entry": entry.get("id"), "library_source": entry.get("source")},
                ))
            except Exception:
                continue
        out.sort(key=lambda c: c.train_r2 - 0.002 * c.complexity, reverse=True)
        return out

    def __len__(self) -> int:
        return len(self.entries)

    def __repr__(self) -> str:
        return f"<LawLibrary: {len(self.entries)} learned laws path={self.path}>"


class _SimpleLaw:
    def __init__(self, expr, var_names, train_r2, source):
        self.sympy_expr = expr
        self.var_names = var_names
        self.train_r2 = train_r2
        self.val_r2 = None
        self.complexity = int(sp.count_ops(expr) + len(expr.free_symbols))
        self.latex_str = sp.latex(expr)
        self.source = source


__all__ = ["LawLibrary"]
