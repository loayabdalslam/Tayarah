from axiom_world import pipeline, PhysicsBenchmarkSuite

pipe = pipeline()
laws = PhysicsBenchmarkSuite.get_all_laws()

for key, law_meta in laws.items():
    X_tr, y_tr = law_meta.sample_data(n_samples=8, seed=42)
    X_val, y_val = law_meta.sample_data(n_samples=50, ood=True, seed=100)
    res = pipe.discover(X_tr, y_tr, X_val, y_val, var_names=law_meta.var_names, target_name=law_meta.target_name)
    print(f"[{law_meta.category}] {law_meta.name}")
    print(f"  Ground Truth: {law_meta.target_name} = {law_meta.formula_str}")
    print(f"  Discovered  : {res.equation}")
    print(f"  Train R²: {res.metrics['train_r2']:.4f} | Unseen Val R²: {res.metrics['val_r2']:.4f}")
    print()
