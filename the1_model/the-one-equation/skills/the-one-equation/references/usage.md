# Numerical helper

Requires Python 3.9+ and NumPy. No network, credentials, model download, or external API is used by the helper. It reads only the specified dataset and writes JSON to standard output. Use available host calculation tools if this runtime is absent. The plugin itself is a reusable AI workflow, not a trained model.

Run relative to the skill directory (resolve the actual installed path):

```text
python scripts/fit_equation.py --pairs "[[1,3],[2,5],[3,7],[4,9]]"
python scripts/fit_equation.py --csv observations.csv
python scripts/fit_equation.py --csv observations.csv --x time --y distance
python scripts/fit_equation.py --csv observations.csv --no-header
```

Default CSV handling expects a header and exactly two columns unless --x and --y explicitly select names. UTF-8 BOM is supported. Invalid rows produce an error with a row number. The host can convert approved cleaned data to JSON pairs; the helper never silently deletes observations.

Families: constant, linear, quadratic, cubic, logarithmic (x>0), reciprocal (x!=0), square root (x>=0), exponential and power (same-sign nonzero y; power also requires x>0). Exponential and power coefficients are estimated in log space; all reported residual scores use original y values. Their log-space estimates do not minimize original-scale squared error.

Polynomials use centered/scaled x for stability, and the printed formula defines z explicitly. Full coefficients and transform parameters accompany the display formula; use full precision for predictions. Ill-conditioned fits are skipped.

With at least six distinct x values, the helper uses up to five deterministic shuffled folds grouped by x; otherwise validation is unavailable. Every candidate must have enough distinct training x values in every fold. This validation assumes independent observations, not time series. For ordered/dependent data, use an appropriate split with host calculation tools instead.

Selection: first prefer the lowest-parameter fit whose maximum absolute residual is below 1e-9 times the maximum absolute y (with a 1e-12 absolute floor). This is a numerical tolerance, not proof of exactness or uniqueness. Otherwise prefer the lowest-parameter model within 5% of the best cross-validation RMSE, or training RMSE when validation is unavailable. The 5% rule is a heuristic, not a statistical confidence bound. High training R-squared alone is not a reason to recommend a complex model. Results explicitly warn when validation is unavailable.

The helper does not search arbitrary symbolic expressions, handle multivariable regression, generate charts, or guarantee a good equation. The AI can use host tools for further candidates, charts, and requested exports while following SKILL.md.
