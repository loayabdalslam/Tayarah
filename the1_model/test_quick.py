from axiom_world import pipeline, PhysicsBenchmarkSuite

pipe = pipeline()
laws = PhysicsBenchmarkSuite.get_all_laws()

keys = ['newton_second', 'kinetic_energy', 'kepler_third', 'pendulum_period', 'lorentz_factor', 'gravitation_normalized', 'ideal_gas', 'coulomb', 'stefan_boltzmann', 'radioactive_decay']

for key in keys:
    law_meta = laws[key]
    X_tr, y_tr = law_meta.sample_data(n_samples=8, seed=42)
    X_val, y_val = law_meta.sample_data(n_samples=50, ood=True, seed=100)
    res = pipe.discover(X_tr, y_tr, X_val, y_val, var_names=law_meta.var_names, target_name=law_meta.target_name)
    print(f"=== {law_meta.name} ===")
    print(f"Ground Truth: {law_meta.target_name} = {law_meta.formula_str}")
    print(f"Discovered  : {res.equation}")
    print(f"Train R2: {res.metrics['train_r2']:.4f} | Unseen Val R2: {res.metrics['val_r2']:.4f}")
    print()
