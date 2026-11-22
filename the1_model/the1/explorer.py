"""Dataset exploration, profiling, batch discovery, and reporting.

This module turns The1 from a single-dataset equation finder into a data
explorer: it profiles columns, proposes targets/features, handles large tables,
discovers laws for many targets, and accumulates learned laws in a library.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .core import SymbolicLaw, The1WorldModel
from .library import LawLibrary


@dataclass
class DiscoveryReport:
    target: str
    features: List[str]
    law: SymbolicLaw
    n_train: int
    n_val: int
    split: str
    profile: Dict[str, Any] = field(default_factory=dict)
    verification: Dict[str, Any] = field(default_factory=dict)

    @property
    def equation(self) -> str:
        return f"{self.target} = {self.law.str_repr}"

    @property
    def latex(self) -> str:
        latex = self.law.latex_str or str(self.law.sympy_expr)
        return f"{self.target} = {latex}"

    @property
    def metrics(self) -> Dict[str, Any]:
        return {
            "train_r2": float(self.law.train_r2),
            "val_r2": float(self.law.val_r2) if self.law.val_r2 is not None else None,
            "train_rmse": float(self.law.train_rmse),
            "val_rmse": float(self.law.val_rmse) if self.law.val_rmse is not None else None,
            "complexity": int(self.law.complexity),
            "source": self.law.source,
            "n_train": self.n_train,
            "n_val": self.n_val,
            "split": self.split,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "features": self.features,
            "equation": self.equation,
            "latex": self.latex,
            "metrics": self.metrics,
            "profile": self.profile,
            "verification": self.verification,
        }

    def summary(self) -> str:
        val = self.metrics["val_r2"]
        val_s = f"{val:.6f}" if val is not None else "N/A"
        lines = [
            "=" * 68,
            f"  {self.equation}",
            "=" * 68,
            f"Target / features : {self.target} ~ {', '.join(self.features)}",
            f"Train / validation: {self.n_train} / {self.n_val} ({self.split})",
            f"Train R²          : {self.metrics['train_r2']:.6f}",
            f"Validation R²     : {val_s}",
            f"Complexity        : {self.metrics['complexity']} (source: {self.law.source})",
            "=" * 68,
        ]
        if self.verification:
            lines.append(f"Lean 4 verification: {self.verification.get('status', 'n/a')} "
                         f"({self.verification.get('proven', 0)}/"
                         f"{self.verification.get('total', 0)} hypotheses proven)")
            for hyp in self.verification.get('hypotheses', []):
                lines.append(f"  - {hyp.get('name', '?')} [{hyp.get('kind', '?')}]: "
                             f"{hyp.get('status', '?')}")
        return "\n".join(lines)


class DatasetExplorer:
    def __init__(self,
                 model: Optional[The1WorldModel] = None,
                 max_features: int = 8,
                 max_rows: int = 20000,
                 random_state: int = 42,
                 library: Optional[LawLibrary] = None,
                 **model_kwargs):
        self.max_features = max_features
        self.max_rows = max_rows
        self.random_state = random_state
        self.library = library
        if model is None:
            kwargs = dict(model_kwargs)
            if library is not None:
                kwargs.setdefault("law_library", library)
            model = The1WorldModel(random_state=random_state, **kwargs)
        self.model = model

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    @staticmethod
    def load(path: Union[str, Path]) -> pd.DataFrame:
        path = Path(path)
        if path.suffix.lower() in (".xlsx", ".xls"):
            return pd.read_excel(path)
        for sep, engine in [(None, "python"), (",", None), (r"\s+", "python"), ("\t", None)]:
            try:
                return pd.read_csv(path, sep=sep, engine=engine) if engine else pd.read_csv(path, sep=sep)
            except Exception:
                continue
        raise ValueError(f"Could not parse data file: {path}")

    # ------------------------------------------------------------------
    # Profiling
    # ------------------------------------------------------------------
    def profile(self, data: Union[pd.DataFrame, str, Path],
                target: Optional[str] = None) -> Dict[str, Any]:
        df = self.load(data) if not isinstance(data, pd.DataFrame) else data
        numeric = df.select_dtypes(include=[np.number])
        out: Dict[str, Any] = {
            "n_rows": int(len(df)),
            "n_columns": int(len(df.columns)),
            "numeric_columns": list(numeric.columns),
            "columns": {},
        }
        target_series = None
        if target and target in df.columns:
            try:
                target_series = pd.to_numeric(df[target], errors="coerce")
            except Exception:
                target_series = None

        for col in df.columns:
            s = df[col]
            info: Dict[str, Any] = {
                "dtype": str(s.dtype),
                "missing": int(s.isna().sum()),
                "unique": int(s.nunique(dropna=True)),
            }
            if col in numeric.columns:
                v = pd.to_numeric(s, errors="coerce").dropna().to_numpy(dtype=float)
                if v.size:
                    info.update({
                        "min": float(np.min(v)),
                        "max": float(np.max(v)),
                        "mean": float(np.mean(v)),
                        "std": float(np.std(v)),
                        "zeros": int(np.sum(np.abs(v) < 1e-12)),
                        "negative": bool(np.any(v < 0)),
                        "monotone": ("increasing" if np.all(np.diff(v) >= 0)
                                     else "decreasing" if np.all(np.diff(v) <= 0) else "noisy"),
                    })
                    if target_series is not None and col != target:
                        tv = target_series.to_numpy(dtype=float)
                        mask = np.isfinite(tv) & np.isfinite(v)
                        if mask.sum() > 2 and np.std(v[mask]) > 1e-12 and np.std(tv[mask]) > 1e-12:
                            info["corr_with_target"] = float(np.corrcoef(v[mask], tv[mask])[0, 1])
                        info["suggested_transforms"] = self._suggest_transforms(v)
            out["columns"][col] = info
        return out

    @staticmethod
    def _suggest_transforms(v: np.ndarray) -> List[str]:
        suggestions = []
        if np.all(v > 0):
            suggestions.append("log")
        if np.all(v >= 0):
            suggestions.append("sqrt")
        if np.all(np.abs(v) > 1e-12):
            suggestions.append("reciprocal")
        if np.all(np.abs(v) <= 1.0):
            suggestions.append("asin")
        return suggestions

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------
    def discover(self,
                 data: Union[pd.DataFrame, str, Path],
                 target: Optional[str] = None,
                 features: Optional[Sequence[str]] = None,
                 split: str = "ood",
                 val_fraction: float = 0.2,
                 max_rows: Optional[int] = None,
                 verbose: bool = False) -> DiscoveryReport:
        df = self.load(data) if not isinstance(data, pd.DataFrame) else data.copy()
        numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
        if target is None:
            if not numeric_cols:
                raise ValueError("No numeric columns found")
            target = numeric_cols[-1]
        if target not in df.columns:
            raise ValueError(f"Target column not found: {target}")

        if features is None:
            features = [c for c in numeric_cols if c != target]
        features = list(dict.fromkeys(features))
        if not features:
            raise ValueError("At least one feature column is required")

        # Keep the most informative features when a table has many columns.
        if len(features) > self.max_features:
            features = self._rank_features(df, target, features)[: self.max_features]

        clean = df[features + [target]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(clean) < 3:
            raise ValueError("At least three complete numeric rows are required")
        n = len(clean)
        cap = max_rows or self.max_rows
        if n > cap:
            # Evenly spaced rows preserve the observed range; a random supplement
            # avoids aliasing periodic signals.
            rng = np.random.RandomState(self.random_state)
            idx = np.unique(np.concatenate([
                np.linspace(0, n - 1, cap // 2).astype(int),
                rng.choice(n, size=cap // 2, replace=False),
            ]))
            clean = clean.iloc[idx]
        n = len(clean)

        X = clean[features].to_numpy(dtype=float)
        y = clean[target].to_numpy(dtype=float)

        if split == "ood":
            order = np.argsort(y)
            n_val = max(1, int(round(n * val_fraction)))
            train_idx, val_idx = order[:-n_val], order[-n_val:]
        elif split == "random":
            rng = np.random.RandomState(self.random_state)
            idx = rng.permutation(n)
            n_val = max(1, int(round(n * val_fraction)))
            train_idx, val_idx = idx[:-n_val], idx[-n_val:]
        elif split == "none":
            train_idx, val_idx = np.arange(n), np.array([], dtype=int)
        else:
            raise ValueError("split must be 'ood', 'random', or 'none'")

        X_train, y_train = X[train_idx], y[train_idx]
        X_val = X[val_idx] if len(val_idx) else None
        y_val = y[val_idx] if len(val_idx) else None

        law = self.model.fit(
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            var_names=features, verbose=verbose,
        )

        return DiscoveryReport(
            target=target,
            features=features,
            law=law,
            n_train=len(train_idx),
            n_val=len(val_idx),
            split=split,
            profile=self.profile(df, target=target) if n <= 200000 else {},
            verification=dict(law.verification),
        )

    def _rank_features(self, df: pd.DataFrame, target: str,
                       features: Sequence[str]) -> List[str]:
        y = pd.to_numeric(df[target], errors="coerce")
        scores = []
        for c in features:
            x = pd.to_numeric(df[c], errors="coerce")
            mask = x.notna() & y.notna()
            if mask.sum() > 2 and x[mask].std() > 1e-12 and y[mask].std() > 1e-12:
                scores.append((abs(float(np.corrcoef(x[mask], y[mask])[0, 1])), c))
            else:
                scores.append((0.0, c))
        scores.sort(reverse=True)
        return [c for _, c in scores]

    def discover_all(self,
                     data: Union[pd.DataFrame, str, Path],
                     targets: Optional[Sequence[str]] = None,
                     max_targets: int = 6,
                     **kwargs) -> List[DiscoveryReport]:
        df = self.load(data) if not isinstance(data, pd.DataFrame) else data
        numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
        if targets is None:
            targets = numeric_cols[:max_targets]
        reports: List[DiscoveryReport] = []
        for target in targets:
            try:
                reports.append(self.discover(df, target=target, **kwargs))
            except Exception as exc:
                reports.append(DiscoveryReport(
                    target=target, features=[], law=_failed_law(str(exc)),
                    n_train=0, n_val=0, split=kwargs.get("split", "ood"),
                    profile={}, verification={"status": "error", "error": str(exc)},
                ))
        reports.sort(key=lambda r: (r.metrics.get("val_r2") if r.metrics.get("val_r2") is not None
                                    else r.metrics.get("train_r2", -1.0)), reverse=True)
        return reports

    def batch(self,
              directory: Union[str, Path],
              pattern: str = "*.csv",
              output: Optional[Union[str, Path]] = None,
              **kwargs) -> List[DiscoveryReport]:
        directory = Path(directory)
        reports: List[DiscoveryReport] = []
        for path in sorted(directory.glob(pattern)):
            try:
                report = self.discover(path, **kwargs)
                report.profile.setdefault("file", str(path))
                reports.append(report)
            except Exception as exc:
                reports.append(DiscoveryReport(
                    target=path.stem, features=[], law=_failed_law(str(exc)),
                    n_train=0, n_val=0, split=kwargs.get("split", "ood"),
                    profile={"file": str(path)},
                    verification={"status": "error", "error": str(exc)},
                ))
        if output is not None:
            save_reports(reports, output)
        return reports


def _failed_law(message: str) -> SymbolicLaw:
    import sympy as sp
    return SymbolicLaw(
        sympy_expr=sp.Integer(0), latex_str="0", str_repr=f"FAILED({message})",
        ast_tree=None, params=np.array([]), var_names=[], train_r2=float("nan"),
        complexity=0, source="failed",
    )


def save_reports(reports: Sequence[DiscoveryReport], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [r.to_dict() for r in reports]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def reports_to_frame(reports: Sequence[DiscoveryReport]) -> pd.DataFrame:
    rows = []
    for r in reports:
        row = {
            "target": r.target,
            "features": ", ".join(r.features),
            "equation": r.equation,
            "train_r2": r.metrics.get("train_r2"),
            "val_r2": r.metrics.get("val_r2"),
            "complexity": r.metrics.get("complexity"),
            "source": r.metrics.get("source"),
            "n_train": r.n_train,
            "n_val": r.n_val,
            "file": r.profile.get("file", ""),
        }
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = ["DatasetExplorer", "DiscoveryReport", "save_reports", "reports_to_frame"]
