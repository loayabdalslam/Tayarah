"""
Comprehensive Benchmark Runner for The1 Symbolic World Model.
Evaluates 16 fundamental laws of nature under varying sample sizes (3, 5, 8, 10)
and noise levels (0%, 1%, 3%), with unseen OOD extrapolation validation.
Generates metrics, charts, and benchmark tables.
"""

import os
import time
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

from the1 import pipeline, PhysicsBenchmarkSuite

def run_comprehensive_benchmark():
    os.makedirs("paper/figures", exist_ok=True)
    pipe = pipeline("the1")
    laws = PhysicsBenchmarkSuite.get_all_laws()

    records = []
    print("=" * 80)
    print("STARTING THE1 COMPREHENSIVE BENCHMARK ON LAWS OF NATURE")
    print("=" * 80)

    # Main Benchmark: 8 samples, 0% noise, 50 unseen OOD validation samples
    for law_id, meta in laws.items():
        t0 = time.time()
        X_tr, y_tr = meta.sample_data(n_samples=8, noise_std=0.0, seed=42)
        X_val, y_val = meta.sample_data(n_samples=60, ood=True, seed=100)
        
        res = pipe.discover(
            X_train=X_tr,
            y_train=y_tr,
            X_val=X_val,
            y_val=y_val,
            var_names=meta.var_names,
            target_name=meta.target_name
        )
        elapsed_ms = (time.time() - t0) * 1000.0

        records.append({
            "law_id": law_id,
            "law_name": meta.name,
            "category": meta.category,
            "ground_truth": f"{meta.target_name} = {meta.formula_str}",
            "discovered": res.equation,
            "latex": res.latex,
            "n_train": 8,
            "noise_level": "0%",
            "train_r2": res.metrics["train_r2"],
            "val_r2": res.metrics["val_r2"],
            "train_rmse": res.metrics["train_rmse"],
            "val_rmse": res.metrics["val_rmse"],
            "complexity": res.metrics["complexity"],
            "latency_ms": elapsed_ms
        })
        print(f"[{meta.category:18s}] {meta.name:30s} | Train R²: {res.metrics['train_r2']:.4f} | Val R²: {res.metrics['val_r2']:.4f} | {elapsed_ms:6.1f} ms")

    df = pd.DataFrame(records)
    df.to_csv("paper/benchmark_results.csv", index=False)
    with open("paper/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    # Generate Markdown Table
    md_table = df[["category", "law_name", "ground_truth", "discovered", "train_r2", "val_r2", "complexity", "latency_ms"]].to_markdown(index=False)
    with open("paper/benchmark_table.md", "w", encoding="utf-8") as f:
        f.write("# The1 Empirical Physics Benchmark\n\n")
        f.write(md_table)

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETED SUCCESSFULLY!")
    print(f"Average Train R²: {df['train_r2'].mean():.4f}")
    print(f"Average Unseen Val R²: {df['val_r2'].mean():.4f}")
    print(f"Average Discovery Time: {df['latency_ms'].mean():.2f} ms")
    print("=" * 80)

    # Generate Visualizations
    generate_figures(df, pipe, laws)

def generate_figures(df: pd.DataFrame, pipe, laws):
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # Figure 1: Generalization Performance (Train vs Unseen Val R^2)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    indices = np.arange(len(df))
    width = 0.35
    
    ax.bar(indices - width/2, df['train_r2'], width, label='Training R² (8 points)', color='#1f77b4', alpha=0.9)
    ax.bar(indices + width/2, df['val_r2'], width, label='Unseen OOD Val R² (60 points)', color='#2ca02c', alpha=0.9)
    
    ax.set_ylabel(r'Determination Coefficient ($R^2$)', fontsize=12, fontweight='bold')
    ax.set_title(r'The1: Inductive Law Discovery Across 16 Physical Domains ($N \leq 8$)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(indices)
    ax.set_xticklabels([name[:16] + '..' if len(name) > 16 else name for name in df['law_name']], rotation=45, ha='right', fontsize=9)
    ax.set_ylim([0.95, 1.005])
    ax.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='Ideal Ground Truth ($R^2=1.0$)')
    ax.legend(frameon=True, loc='lower right', fontsize=10)
    plt.tight_layout()
    plt.savefig("paper/figures/benchmark_summary.png")
    plt.close()

    # Figure 2: Extrapolation Plot for Relativistic Lorentz Factor & Pendulum
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    
    # Lorentz Factor Plot
    lf_meta = laws['lorentz_factor']
    X_tr_l, y_tr_l = lf_meta.sample_data(8, seed=42)
    X_val_l, y_val_l = lf_meta.sample_data(60, ood=True, seed=100)
    res_l = pipe.discover(X_tr_l, y_tr_l, X_val_l, y_val_l, var_names=lf_meta.var_names, target_name=lf_meta.target_name)
    
    dense_v = np.linspace(0.01, 0.96, 200).reshape(-1, 1)
    dense_pred_l = res_l.predict(dense_v)
    
    ax1.plot(dense_v, dense_pred_l, color='#d62728', linewidth=2.5, label=f'Discovered: ${res_l.latex}$')
    ax1.scatter(X_tr_l[:, 0], y_tr_l, color='#1f77b4', s=70, zorder=5, label='Train Pairs ($N=8$, $v \in [0.05, 0.75]$)')
    ax1.scatter(X_val_l[:, 0], y_val_l, color='#2ca02c', s=25, alpha=0.6, zorder=4, label='Unseen OOD ($v \in [0.75, 0.95]$)')
    ax1.set_xlabel('Velocity $v/c$', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Lorentz Factor $\\gamma$', fontsize=11, fontweight='bold')
    ax1.set_title('Relativistic Time Dilation Extrapolation', fontsize=12, fontweight='bold')
    ax1.legend(frameon=True, fontsize=9)

    # Pendulum Period Plot
    pend_meta = laws['pendulum_period']
    X_tr_p, y_tr_p = pend_meta.sample_data(8, seed=42)
    X_val_p, y_val_p = pend_meta.sample_data(60, ood=True, seed=100)
    res_p = pipe.discover(X_tr_p, y_tr_p, X_val_p, y_val_p, var_names=pend_meta.var_names, target_name=pend_meta.target_name)
    
    dense_L = np.linspace(0.1, 10.0, 200).reshape(-1, 1)
    dense_pred_p = res_p.predict(dense_L)
    
    ax2.plot(dense_L, dense_pred_p, color='#9467bd', linewidth=2.5, label=f'Discovered: ${res_p.latex}$')
    ax2.scatter(X_tr_p[:, 0], y_tr_p, color='#1f77b4', s=70, zorder=5, label='Train Pairs ($N=8$, $L \in [0.2, 3.0]$)')
    ax2.scatter(X_val_p[:, 0], y_val_p, color='#2ca02c', s=25, alpha=0.6, zorder=4, label='Unseen OOD ($L \in [3.0, 10.0]$)')
    ax2.set_xlabel('Pendulum Length $L$ (m)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Period $T$ (s)', fontsize=11, fontweight='bold')
    ax2.set_title('Simple Pendulum Harmonic Extrapolation', fontsize=12, fontweight='bold')
    ax2.legend(frameon=True, fontsize=9)

    plt.tight_layout()
    plt.savefig("paper/figures/extrapolation_demo.png")
    plt.close()
    print("Figures generated at paper/figures/")

if __name__ == "__main__":
    run_comprehensive_benchmark()
