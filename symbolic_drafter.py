import os
import sys
import time
import numpy as np
import sympy as sp
from concurrent.futures import ThreadPoolExecutor

# Ensure OpenMP compatibility for Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add the1_model directory to sys.path
the1_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "the1_model")
if the1_dir not in sys.path:
    sys.path.insert(0, the1_dir)

from the1.core import The1WorldModel

class TayarahSymbolicDrafter:
    """
    Tayarah Symbolic Drafter (طيارة) - Powered by racore/The1
    Discovers closed-form mathematical equations governing token sequences in extreme few-shot
    data pairs (4-8 observations), synthesizing speculative tokens in microseconds directly
    within CPU registers to break the memory bandwidth bottleneck.
    """
    def __init__(self, vocab_size: int = 256000):
        self.vocab_size = vocab_size
        self.world_model = The1WorldModel(
            population_size=20,
            generations=6,
            max_complexity=8,
            verify_with_lean=False
        )
        self.equation_cache = {}
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.active_tasks = set()
        self.stats = {
            "discoveries": 0,
            "fast_signatures": 0,
            "cache_hits": 0,
            "drafts_generated": 0,
            "total_eval_time_us": 0.0
        }

    def _sanitize_token(self, val) -> int:
        try:
            tok = int(round(float(val)))
            return max(0, min(tok, self.vocab_size - 1))
        except (ValueError, OverflowError):
            return 0

    def _async_learn_law(self, obs_x: np.ndarray, obs_y: np.ndarray, diff_key: tuple):
        try:
            law = self.world_model.fit(
                X_train=obs_x.reshape(-1, 1),
                y_train=obs_y,
                var_names=["t"],
                verbose=False
            )
            if law.train_r2 >= 0.85 and law.sympy_expr is not None:
                t_sym = sp.Symbol("t")
                fast_func = sp.lambdify(t_sym, law.sympy_expr, modules=["numpy", "math"])
                self.equation_cache[diff_key] = fast_func
                self.stats["discoveries"] += 1
        except Exception:
            pass

    def fit_and_extrapolate(self, sequence: list, k_tokens: int = 4) -> list:
        if len(sequence) < 4:
            return []

        obs_y = np.array(sequence[-8:], dtype=float)
        obs_x = np.arange(len(obs_y), dtype=float)

        # 1. Instant Arithmetic Signature (Linear Recurrence)
        diff1 = np.diff(obs_y)
        if len(diff1) >= 2 and np.allclose(diff1, diff1[-1], atol=1e-1):
            self.stats["fast_signatures"] += 1
            step = diff1[-1]
            start_val = obs_y[-1]
            t0 = time.perf_counter()
            next_vals = [self._sanitize_token(start_val + (i + 1) * step) for i in range(k_tokens)]
            self.stats["total_eval_time_us"] += (time.perf_counter() - t0) * 1e6
            self.stats["drafts_generated"] += k_tokens
            return next_vals

        # 2. Instant Geometric Signature (Exponential Recurrence)
        if np.all(obs_y > 0):
            ratios = obs_y[1:] / obs_y[:-1]
            if len(ratios) >= 2 and np.allclose(ratios, ratios[-1], atol=1e-1):
                self.stats["fast_signatures"] += 1
                r = ratios[-1]
                t0 = time.perf_counter()
                next_vals = [self._sanitize_token(obs_y[-1] * (r ** (i + 1))) for i in range(k_tokens)]
                self.stats["total_eval_time_us"] += (time.perf_counter() - t0) * 1e6
                self.stats["drafts_generated"] += k_tokens
                return next_vals

        # 3. Cache Check
        diff_key = tuple(np.round(diff1[-4:], 1))
        if diff_key in self.equation_cache:
            self.stats["cache_hits"] += 1
            func = self.equation_cache[diff_key]
            t0 = time.perf_counter()
            next_vals = [self._sanitize_token(func(len(obs_y) + i)) for i in range(k_tokens)]
            self.stats["total_eval_time_us"] += (time.perf_counter() - t0) * 1e6
            self.stats["drafts_generated"] += k_tokens
            return next_vals

        # 4. Asynchronous Inductive Discovery (racore/The1)
        if diff_key not in self.active_tasks and len(self.active_tasks) < 2:
            self.active_tasks.add(diff_key)
            self.executor.submit(self._async_learn_law, obs_x, obs_y, diff_key)

        return []

    def get_performance_report(self):
        total_evals = max(1, self.stats["drafts_generated"])
        avg_eval_us = self.stats["total_eval_time_us"] / total_evals
        return {
            "the1_discoveries": self.stats["discoveries"],
            "fast_signatures": self.stats["fast_signatures"],
            "cache_hits": self.stats["cache_hits"],
            "drafts_generated": self.stats["drafts_generated"],
            "avg_draft_eval_latency_us": round(avg_eval_us, 3)
        }

# Backwards compatibility alias
AsyncThe1SymbolicDrafter = TayarahSymbolicDrafter
