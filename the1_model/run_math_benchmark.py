"""Cross-domain mathematics benchmark for The1.

Runs the 62-law MathBenchmarkSuite under three regimes:

* 8 training pairs, noise-free, 60 unseen out-of-distribution validation pairs;
* 16 training pairs with 1% target noise;
* 16 training pairs with 3% target noise.

Results are written to ``paper/math_benchmark_results.{csv,json}``, a Markdown
summary table, and figures.  Discoverable and compositional laws are reported
separately so that the headline number is not inflated by easy cases.
"""

from __future__ import annotations

import json
import time
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from the1 import MathBenchmarkSuite, pipeline

OUT_DIR = "paper"
FIG_DIR = "paper/figures"


def _run_regime(pipe, laws, n_train: int, noise: float, seed: int = 42) -> List[Dict]:
    records = []
    for key, meta in laws.items():
        t0 = time.time()
        X_tr, y_tr = meta.sample_data(n_train, noise_std=noise, seed=seed)
        X_val, y_val = meta.sample_data(60, ood=True, seed=100)
        result = pipe.discover(
            X_tr, y_tr, X_val, y_val,
            var_names=meta.var_names,
            target_name=meta.target_name,
        )
        elapsed = (time.time() - t0) * 1000.0
        val_r2 = result.metrics["val_r2"]
        records.append({
            "law_id": key,
            "law_name": meta.name,
            "category": meta.category,
            "group": "compositional" if meta.compositional else "discoverable",
            "ground_truth": f"{meta.target_name} = {meta.formula_str}",
            "discovered": result.equation,
            "n_train": n_train,
            "noise": noise,
            "train_r2": result.metrics["train_r2"],
            "val_r2": val_r2,
            "val_rmse": result.metrics["val_rmse"],
            "complexity": result.metrics["complexity"],
            "latency_ms": elapsed,
            "solved": bool(val_r2 is not None and val_r2 >= 0.99),
            "exact": bool(val_r2 is not None and val_r2 >= 0.999),
        })
    return records


def _summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (regime, group), part in df.groupby(["regime", "group"]):
        rows.append({
            "regime": regime,
            "group": group,
            "n_laws": len(part),
            "solve_rate": float(part["solved"].mean()),
            "exact_rate": float(part["exact"].mean()),
            "mean_val_r2": float(part["val_r2"].dropna().mean()),
            "median_val_r2": float(part["val_r2"].dropna().median()),
            "median_latency_ms": float(part["latency_ms"].median()),
        })
    return pd.DataFrame(rows)


def _category_summary(df: pd.DataFrame, regime: str) -> pd.DataFrame:
    part = df[(df["regime"] == regime) & (df["group"] == "discoverable")]
    rows = []
    for cat, chunk in part.groupby("category"):
        rows.append({
            "category": cat,
            "n_laws": len(chunk),
            "solve_rate": float(chunk["solved"].mean()),
            "mean_val_r2": float(chunk["val_r2"].dropna().mean()),
        })
    return pd.DataFrame(rows).sort_values("category")


def _figures(df: pd.DataFrame, laws) -> None:
    main = df[df["regime"] == "8 pairs, 0% noise"]
    if main.empty:
        return

    # Per-category solve rate for discoverable laws.
    cats = _category_summary(main, "8 pairs, 0% noise")
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.bar(cats["category"], cats["solve_rate"], color="#2a7fb8", alpha=0.9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Solved fraction (unseen OOD R² ≥ 0.99)")
    ax.set_title("The1 cross-domain discovery (8 pairs, noise-free)")
    ax.tick_params(axis="x", rotation=30)
    for i, (rate, n) in enumerate(zip(cats["solve_rate"], cats["n_laws"])):
        ax.text(i, min(rate + 0.02, 1.04), f"{int(rate * n)}/{n}",
                ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/math_category_accuracy.png")
    plt.close()

    # Train vs OOD validation R² per law.
    fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
    order = main.sort_values(["category", "law_name"])
    indices = np.arange(len(order))
    ax.bar(indices - 0.2, order["train_r2"], 0.4, label="Train R² (8 points)",
           color="#1f77b4")
    ax.bar(indices + 0.2, order["val_r2"].fillna(0.0), 0.4,
           label="Unseen OOD R² (60 points)", color="#2ca02c")
    ax.set_ylim(0.9, 1.005)
    ax.set_xticks(indices)
    ax.set_xticklabels([n[:18] for n in order["law_name"]], rotation=90, fontsize=6)
    ax.axhline(1.0, color="red", linestyle="--", alpha=0.4)
    ax.set_ylabel("R²")
    ax.set_title("Cross-domain mathematical law discovery, 62 laws")
    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/math_benchmark_summary.png")
    plt.close()

    # Extrapolation examples: rational and trigonometric behaviour.
    examples = [
        ("algebra_parallel", "R = r1*r2/(r1+r2)", "r1", 1),
        ("prob_gaussian_pdf", "p = exp(-x^2/2)/sqrt(2*pi)", "x", 0),
    ]
    fig, axes = plt.subplots(1, len(examples), figsize=(12, 4.2), dpi=300)
    pipe = pipeline("the1")
    for ax, (key, _label, xvar, col) in zip(axes, examples):
        meta = laws[key]
        X_tr, y_tr = meta.sample_data(8, seed=42)
        X_val, y_val = meta.sample_data(80, ood=True, seed=100)
        result = pipe.discover(X_tr, y_tr, X_val, y_val,
                               var_names=meta.var_names,
                               target_name=meta.target_name)
        grid = np.linspace(meta.domain[col][0], meta.ood_domain[col][1], 400)
        if len(meta.var_names) == 1:
            Xg = grid.reshape(-1, 1)
        else:
            Xg = np.tile(np.array([[np.mean(meta.domain[0])]] * len(grid)),
                         (1, len(meta.var_names)))
            Xg[:, col] = grid
        pred = result.predict(Xg)
        ax.plot(grid, pred, color="#d62728", lw=2.2,
                label=f"Discovered: {result.sympy}")
        ax.scatter(X_tr[:, col], y_tr, s=55, color="#1f77b4", zorder=5,
                   label="Train (8 pairs)")
        ax.scatter(X_val[:, col], y_val, s=18, color="#2ca02c", alpha=0.55,
                   zorder=4, label="Unseen OOD")
        ood_start = meta.domain[col][1]
        ax.axvline(ood_start, color="gray", ls=":", alpha=0.7)
        ax.set_title(meta.name)
        ax.set_xlabel(meta.var_names[col])
        ax.set_ylabel(meta.target_name)
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/math_extrapolation_demo.png")
    plt.close()


def main() -> None:
    import os
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    laws = MathBenchmarkSuite.get_all_laws()
    pipe = pipeline("the1")
    regimes = [
        ("8 pairs, 0% noise", 8, 0.00),
        ("16 pairs, 1% noise", 16, 0.01),
        ("16 pairs, 3% noise", 16, 0.03),
    ]

    records: List[Dict] = []
    print("=" * 86)
    print("THE1 CROSS-DOMAIN MATHEMATICS BENCHMARK")
    print("=" * 86)
    for label, n_train, noise in regimes:
        rows = _run_regime(pipe, laws, n_train, noise)
        for row in rows:
            row["regime"] = label
        records.extend(rows)
        disc = [r for r in rows if r["group"] == "discoverable"]
        comp = [r for r in rows if r["group"] == "compositional"]
        print(f"{label:22s} | discoverable {sum(r['solved'] for r in disc)}/{len(disc)}"
              f" | compositional {sum(r['solved'] for r in comp)}/{len(comp)}"
              f" | mean OOD R² {np.mean([r['val_r2'] for r in disc if r['val_r2'] is not None]):.5f}")

    df = pd.DataFrame(records)
    df.to_csv(f"{OUT_DIR}/math_benchmark_results.csv", index=False)
    with open(f"{OUT_DIR}/math_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    summary = _summary(df)
    summary.to_csv(f"{OUT_DIR}/math_benchmark_summary.csv", index=False)
    categories = _category_summary(df, "8 pairs, 0% noise")

    with open(f"{OUT_DIR}/math_benchmark_table.md", "w", encoding="utf-8") as f:
        f.write("# The1 Cross-Domain Mathematics Benchmark\n\n")
        f.write("Tasks are generated from numerical reference implementations. "
                "No benchmark expression is injected into the search engine.\n\n")
        f.write("## Regime summary\n\n")
        f.write(summary.to_markdown(index=False))
        f.write("\n\n## Discoverable laws by category (8 pairs, noise-free)\n\n")
        f.write(categories.to_markdown(index=False))
        f.write("\n\n## Per-law results (8 pairs, noise-free)\n\n")
        main = df[df["regime"] == "8 pairs, 0% noise"]
        table = main[["category", "law_name", "ground_truth", "discovered",
                      "train_r2", "val_r2", "complexity", "latency_ms"]]
        f.write(table.to_markdown(index=False))
        f.write("\n")

    _figures(df, laws)

    print("\nSaved:")
    print(f"  {OUT_DIR}/math_benchmark_results.csv")
    print(f"  {OUT_DIR}/math_benchmark_results.json")
    print(f"  {OUT_DIR}/math_benchmark_table.md")
    print(f"  {FIG_DIR}/math_category_accuracy.png")
    print(f"  {FIG_DIR}/math_benchmark_summary.png")
    print(f"  {FIG_DIR}/math_extrapolation_demo.png")


if __name__ == "__main__":
    main()
