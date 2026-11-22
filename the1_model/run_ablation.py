"""
Noise and Sample-Size Ablation Study for The1.
Evaluates performance across sample counts N in [3, 5, 8, 10] and noise std in [0%, 1%, 3%].
"""

import numpy as np
import pandas as pd
from the1 import pipeline, PhysicsBenchmarkSuite

def run_ablation():
    pipe = pipeline("the1")
    laws = PhysicsBenchmarkSuite.get_all_laws()
    test_laws = ['newton_second', 'kinetic_energy', 'coulomb', 'kepler_third', 'pendulum_period', 'lorentz_factor']
    
    sample_sizes = [4, 6, 8, 10]
    noise_levels = [0.0, 0.01, 0.03]
    
    records = []
    
    for N in sample_sizes:
        for noise in noise_levels:
            r2_scores = []
            val_r2_scores = []
            for law_id in test_laws:
                meta = laws[law_id]
                X_tr, y_tr = meta.sample_data(n_samples=N, noise_std=noise, seed=42)
                X_val, y_val = meta.sample_data(n_samples=50, ood=True, seed=100)
                res = pipe.discover(X_tr, y_tr, X_val, y_val, var_names=meta.var_names, target_name=meta.target_name)
                r2_scores.append(res.metrics['train_r2'])
                val_r2_scores.append(res.metrics['val_r2'])
                
            records.append({
                "N_samples": N,
                "Noise_std": f"{int(noise*100)}%",
                "Mean_Train_R2": np.mean(r2_scores),
                "Mean_Unseen_Val_R2": np.mean(val_r2_scores)
            })
            print(f"N={N:2d} | Noise={int(noise*100)}% | Train R²: {np.mean(r2_scores):.4f} | Unseen Val R²: {np.mean(val_r2_scores):.4f}")

    df = pd.DataFrame(records)
    df.to_csv("paper/ablation_results.csv", index=False)
    print("\nAblation study complete. Saved to paper/ablation_results.csv")

if __name__ == "__main__":
    run_ablation()
