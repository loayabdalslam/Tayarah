# Advanced equation discovery, version 2

Run `scripts/discover_equations.py` using Python 3.9+, NumPy and SciPy in the host calculation environment. Dependency versions are in `scripts/requirements.txt`. No API key, external server or trained model is bundled. This is an implemented finite-library sparse identification engine guided by the host AI, not a promise to discover every mathematical equation.

## Inputs

Multivariable regression (all columns equal length, numeric and finite):

```json
{"variables":{"mass":[1,2,3],"a":[2,3,4]},"targets":{"force":[2,6,12]}}
```

This short example illustrates the schema; actual runs require at least eight training rows and three validation rows. Identical predictor vectors are grouped in the same split. Multiple target columns are accepted. With measured derivatives, regression mode can also fit supplied derivative targets directly.

```text
python scripts/discover_equations.py --data observations.json --mode regression --degree 3
python scripts/discover_equations.py --data gas.json --functions none --degree 1 --term "n*T/V"
python scripts/discover_equations.py --data nonlinear.json --term "exp(-x*x)" --term "r**(-2)" --term "x**1.5"
python scripts/discover_equations.py --data periodic.json --functions sin,cos --frequencies 0.5,1,2,3
```

CSV input is supported through the host: read the selected columns, validate them, and convert them to this JSON format. Ask which columns are targets if unclear. No arbitrary cell expression is executable. The original one-input helper still accepts CSV directly.

ODE input:

```json
{"t":[0,0.1,0.2],"states":{"position":[1,0.99,0.96]}}
```

Use at least 30 points, and more for smoothing or high derivatives. A system has several state vectors under `states`. To provide distinct experiments, use `{"trajectories":[{"t":[...],"states":{...}}, ...]}`; each trajectory has its own increasing time vector, and all share the same ordered state names. The last trajectory is held out. With only one trajectory, the last approximately 20% is validation, with a guard band for differentiation/smoothing.

```text
python scripts/discover_equations.py --data motion.json --mode ode --order 2 --functions none --degree 2
python scripts/discover_equations.py --data system.json --mode ode --order 1 --include-time
python scripts/discover_equations.py --data noisy.json --mode ode --smooth-window 11
```

Predictors include states and lower derivatives, named `d1_position_dt1`, `d2_position_dt2`, etc. A second-order target is `d2_position_dt2`. `--include-time` adds `time` for nonautonomous laws. Uniform grids use fourth-order centered first differences, composed for higher derivatives; nonuniform grids use second-order NumPy gradients. Boundaries are trimmed to remove lower-accuracy stencil contamination. Optional Savitzky-Golay smoothing requires a uniform grid and is explicit; it alters the data and must be disclosed. Derivative estimation can fail under noise even when training residuals are small.

PDE input:

```text
{"t": [time coordinates], "axes": [[x0 coordinates], [x1 coordinates]], "u": nested numeric field}
```

The field shape must be `[number_of_times, len(x0), len(x1), ...]`. One to three strictly increasing spatial axes are accepted. Grid samples must be complete; no missing values are interpolated. One scalar field is fitted at a time.

```text
python scripts/discover_equations.py --data field.json --mode pde --order 2 --degree 2 --functions none
```

`--order` limits total spatial derivative order, including mixed derivatives. Terms are named `u_x0`, `u_x0_x0`, `u_x0_x1`, etc. The target is `du_dt`. Degree 2 includes nonlinear terms such as `u*u_x0`, enabling candidate advection/diffusion/reaction laws. Validation holds out late time slabs, with a guard band. This engine does not fit coupled fields, higher-order time PDEs, irregular meshes or boundary conditions. A derivative residual is not a PDE simulation test.

Implicit polynomial relations:

```text
python scripts/discover_equations.py --data geometry.json --mode implicit --degree 2 --functions none
```

Input is `{"variables":{"x":[...],"y":[...]}}` with no targets. Normalized homogeneous least squares searches a near-null direction. There must be more than twice as many observations as library terms and more training rows than terms. Only generated polynomials are accepted here, avoiding custom function identities. Several near-null directions imply nonidentifiability. The returned homogeneous residual is scale-normalized and is not an R-squared value or a probability. No support sparsification is claimed for implicit mode.

## Search and validation

The explicit engine combines normalized orthogonal matching pursuit paths with thresholded least-squares supports when the training system is overdetermined. Coefficients are fitted on training data only. It selects the fewest terms within 5% of the best validation RMSE, with a numerical floor `max(selection_atol, selection_rtol * max(abs(training_target)))`. Defaults are 1e-10 and 1e-7 respectively; both can be changed. This tolerance can hide weak contributions: report it, and check sensitivity when weak physical effects matter. These heuristics are not confidence intervals or a global optimality guarantee.

Training and validation errors, selected terms, active-design conditioning, whole-library rank, feature names and exclusions are recorded in the JSON report. A rank-deficient library, insufficiently excited trajectory or correlated inputs can admit many laws. Features whose domains fail at any observation are excluded as whole candidates; observations are never silently removed. A domain-invalid user-requested custom term is an error.

`--degree` is 0..8; `--order` is 1..4; `--max-terms` is 1..50 (default 12); `--max-features` is 1..1000 (default 400). Combinatorial libraries that exceed the budget are rejected instead of silently truncated. Functions can be `none` or a comma-separated subset of `sin,cos,exp,log,sqrt,inverse,tanh,sinh,cosh`. Trial frequencies are explicit, bounded and not continuously optimized. Custom terms accept arithmetic, constant powers with magnitude <=8 and these unary functions through an allowlisted AST. Attributes, indexing, imports and arbitrary Python are rejected. Compositions and rational forms are supported only when present in the candidate library.

The default regression split assumes independent rows; use `--ordered` for chronological splitting. ODEs and PDEs use chronological/trajectory splits automatically. Reported selection-validation scores are not independent test scores. The engine does not automatically integrate selected models, run a PDE solver, infer units, estimate confidence intervals, discover fractional/delay/stochastic equations, or search unlimited symbolic structures.

## Method background

Sparse candidate-library discovery is motivated by [Brunton, Proctor and Kutz (2016)](https://doi.org/10.1073/pnas.1517384113) and [Rudy et al. (2017)](https://doi.org/10.1126/sciadv.1602614). This implementation is a small independent engine; it does not claim to reproduce those complete research systems.

## Included verification

Run `python validation/test_discovery.py` from the plugin root. The included synthetic tests cover physical and mathematical examples, numerical input checks, and the expression boundary. They do not constitute testing against all physics laws or arbitrary real experiments.
