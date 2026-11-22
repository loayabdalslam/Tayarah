"""
Hugging Face style Pipeline API for The1 Inductive Symbolic World Model.
Enables one-line natural law discovery from extreme few-shot observations.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Union
from .core import The1WorldModel, SymbolicLaw

class PipelineDiscoveryResult:
    """Convenient, interpretable wrapper for discovery results."""
    def __init__(self,
                 target_name: str,
                 var_names: List[str],
                 law: SymbolicLaw,
                 n_samples: int):
        self.target_name = target_name
        self.var_names = var_names
        self.law = law
        self.n_samples = n_samples
        self.equation = f"{target_name} = {law.str_repr}"
        self.latex = f"{target_name} = {law.latex_str}"
        self.sympy = law.sympy_expr
        self.verification = dict(law.verification)
        self.metrics = {
            "train_r2": float(law.train_r2),
            "val_r2": float(law.val_r2) if law.val_r2 is not None else None,
            "train_rmse": float(law.train_rmse),
            "val_rmse": float(law.val_rmse) if law.val_rmse is not None else None,
            "complexity": int(law.complexity),
            "occam_score": float(law.occam_score),
            "n_train_points": int(n_samples)
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.law.predict(X)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            f"  THE1 DISCOVERED LAW: {self.equation}",
            "=" * 60,
            f"LaTeX Format   : {self.latex}",
            f"Training Data  : {self.n_samples} points",
            f"Train R² Fit   : {self.metrics['train_r2']:.6f} (RMSE: {self.metrics['train_rmse']:.6e})",
        ]
        if self.metrics['val_r2'] is not None:
            lines.append(f"Unseen Val R²  : {self.metrics['val_r2']:.6f} (OOD RMSE: {self.metrics['val_rmse']:.6e})")
        lines.append(f"Complexity     : {self.metrics['complexity']} AST nodes")
        if self.verification:
            proven = self.verification.get('proven', 0)
            total = self.verification.get('total', 0)
            lines.append(f"Lean 4 checks  : {self.verification.get('status', 'n/a')} "
                         f"({proven}/{total} proven)")
        lines.append("=" * 60)
        return "\n".join(lines)

    def __repr__(self):
        return f"<The1DiscoveryResult: '{self.equation}' (Train R²={self.metrics['train_r2']:.4f})>"


class The1Pipeline:
    """
    High-level Hugging Face style Pipeline for The1 World Model.
    Supports local execution and downloading pretrained models from Hugging Face Hub.
    """
    def __init__(self, model: Optional[Union[The1WorldModel, str, Any]] = None, **kwargs):
        if isinstance(model, str):
            # Load from Hugging Face Hub model ID or local directory
            try:
                from transformers import AutoModel
                self.hf_model = AutoModel.from_pretrained(model, trust_remote_code=True)
                self.model = self.hf_model.engine
            except Exception:
                self.model = The1WorldModel(**kwargs)
        elif hasattr(model, "engine"):
            self.hf_model = model
            self.model = model.engine
        elif model is not None:
            self.model = model
        else:
            self.model = The1WorldModel(**kwargs)

    def discover(self,
                 X_train: Union[np.ndarray, List],
                 y_train: Union[np.ndarray, List],
                 X_val: Optional[Union[np.ndarray, List]] = None,
                 y_val: Optional[Union[np.ndarray, List]] = None,
                 var_names: Optional[List[str]] = None,
                 target_name: str = "y",
                 verbose: bool = False) -> PipelineDiscoveryResult:
        """
        Discovers the symbolic mathematical equation from data points (pairs).
        """
        X_train_arr = np.asarray(X_train)
        y_train_arr = np.asarray(y_train)
        if X_train_arr.ndim == 1:
            X_train_arr = X_train_arr.reshape(-1, 1)

        X_val_arr = np.asarray(X_val) if X_val is not None else None
        y_val_arr = np.asarray(y_val) if y_val is not None else None
        if X_val_arr is not None and X_val_arr.ndim == 1:
            X_val_arr = X_val_arr.reshape(-1, 1)

        if var_names is None:
            var_names = [f"x{i+1}" for i in range(X_train_arr.shape[1])]

        law = self.model.fit(
            X_train=X_train_arr,
            y_train=y_train_arr,
            X_val=X_val_arr,
            y_val=y_val_arr,
            var_names=var_names,
            verbose=verbose
        )

        return PipelineDiscoveryResult(
            target_name=target_name,
            var_names=var_names,
            law=law,
            n_samples=len(y_train_arr)
        )

    def __call__(self, *args, **kwargs) -> PipelineDiscoveryResult:
        return self.discover(*args, **kwargs)


SymbolicWorldPipeline = The1Pipeline


def pipeline(task: str = "the1", model: Optional[Union[str, Any]] = None, **kwargs) -> The1Pipeline:
    """
    Factory function mirroring Hugging Face's pipeline API.
    Usage:
        pipe = pipeline("the1")
        # Or load model directly from Hugging Face Hub (increments download counter!):
        pipe = pipeline("the1", model="your-username/The1")
        res = pipe.discover(X_train, y_train, var_names=['m', 'a'], target_name='F')
    """
    valid_tasks = ["the1", "the-one", "symbolic-world-model", "symbolic-regression", "law-discovery"]
    if task not in valid_tasks:
        raise ValueError(f"Unknown task: {task}. Choose from {valid_tasks}.")
    return The1Pipeline(model=model, **kwargs)
