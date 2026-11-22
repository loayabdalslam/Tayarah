"""Fit common curve families to paired numeric data; Python 3.9+, NumPy."""
import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np


def read_pairs(args):
    if args.pairs is not None:
        pairs = json.loads(args.pairs)
    else:
        with Path(args.csv).open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.reader(handle))
        if not rows:
            raise ValueError("CSV is empty")
        if args.no_header:
            if args.x or args.y:
                raise ValueError("Column names require a header")
            indices, first = (0, 1), 1
        else:
            header = rows.pop(0)
            if args.x or args.y:
                if not (args.x and args.y) or args.x == args.y:
                    raise ValueError("Specify distinct --x and --y column names")
                if header.count(args.x) != 1 or header.count(args.y) != 1:
                    raise ValueError("Selected column names must each occur exactly once")
                indices = header.index(args.x), header.index(args.y)
            elif len(header) == 2:
                indices = 0, 1
            else:
                raise ValueError("Select --x and --y for CSVs with more than two columns")
            first = 2
        pairs = []
        for number, row in enumerate(rows, first):
            try:
                if args.no_header and len(row) != 2:
                    raise ValueError("Expected two columns")
                if not args.no_header and len(row) != len(header):
                    raise ValueError("Row width differs from header")
                pairs.append([float(row[i]) for i in indices])
            except (ValueError, IndexError) as exc:
                raise ValueError(f"Invalid CSV row {number}: {exc}") from exc
    if not isinstance(pairs, list) or any(
        not isinstance(row, list) or len(row) != 2
        or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in row)
        for row in pairs
    ):
        raise ValueError("Expected a JSON array of numeric [x,y] pairs")
    data = np.asarray(pairs, dtype=float)
    if data.ndim != 2 or data.shape[1] != 2 or len(data) < 2:
        raise ValueError("Provide at least two numeric pairs")
    if not np.isfinite(data).all():
        raise ValueError("All values must be finite; no rows have been removed")
    return data[:, 0], data[:, 1]


def fit_model(kind, x, y):
    degree = {"constant": 0, "linear": 1, "quadratic": 2, "cubic": 3}.get(kind)
    if degree is not None:
        center = float(np.mean(x))
        scale = float(np.max(np.abs(x - center))) or 1.0
        terms = ["1"] + ["z" if i == 1 else f"z^{i}" for i in range(1, degree + 1)]
        transform = lambda v: np.vander((v - center) / scale, degree + 1, increasing=True)
        target, log_output, sign = y, False, 1.0
        domain = f"z = (x - ({center:.12g})) / ({scale:.12g})"
    else:
        valid = {"logarithmic": np.all(x > 0), "reciprocal": np.all(x != 0),
                 "square-root": np.all(x >= 0), "exponential": True,
                 "power": np.all(x > 0)}[kind]
        if not valid:
            raise ValueError("Outside family domain")
        feature = {"logarithmic": np.log, "reciprocal": lambda v: 1.0 / v,
                   "square-root": np.sqrt, "exponential": lambda v: v,
                   "power": np.log}[kind]
        transform = lambda v: np.column_stack([np.ones(len(v)), feature(v)])
        log_output = kind in ("exponential", "power")
        sign = -1.0 if np.all(y < 0) else 1.0
        if log_output and not (np.all(y > 0) or np.all(y < 0)):
            raise ValueError("Log fit needs same-sign nonzero y")
        target = np.log(np.abs(y)) if log_output else y
        terms = ["1", {"logarithmic": "ln(x)", "reciprocal": "1/x",
                       "square-root": "sqrt(x)", "exponential": "x", "power": "ln(x)"}[kind]]
        domain = {"logarithmic": "x > 0", "reciprocal": "x != 0",
                  "square-root": "x >= 0", "exponential": "all real x", "power": "x > 0"}[kind]
        center, scale = None, None
    design = transform(x)
    if len(np.unique(x)) < design.shape[1]:
        raise ValueError("Too few distinct x values")
    coefficients, _, rank, singular = np.linalg.lstsq(design, target, rcond=None)
    if rank != design.shape[1] or singular[0] / singular[-1] > 1e12:
        raise ValueError("Ill-conditioned fit")

    def predict(v):
        result = transform(np.asarray(v)) @ coefficients
        return sign * np.exp(result) if log_output else result

    expression = " + ".join(f"({float(c):.12g})*{term}" for c, term in zip(coefficients, terms))
    formula = f"y = ({sign:g})*exp({expression})" if log_output else f"y = {expression}"
    if degree is not None and degree > 0:
        formula += f"; {domain}"
    metadata = {"family": kind, "equation": formula, "parameters": len(coefficients),
                "coefficients": coefficients.tolist(), "domain_or_transform": domain,
                "x_center": center, "x_scale": scale, "log_output": log_output,
                "output_sign": sign}
    return predict, metadata


def analyze(x, y):
    unique = np.unique(x)
    if len(unique) < 2:
        raise ValueError("At least two distinct x values are needed to infer x dependence")
    folds = []
    if len(unique) >= 6:
        shuffled = unique.copy()
        np.random.default_rng(42).shuffle(shuffled)
        folds = [np.isin(x, values) for values in np.array_split(shuffled, min(5, len(unique)))]
    candidates, skipped = [], []
    total_variance = float(np.sum((y - np.mean(y)) ** 2))
    for kind in ["constant", "linear", "quadratic", "cubic", "logarithmic",
                 "reciprocal", "square-root", "exponential", "power"]:
        try:
            with np.errstate(all="raise"), warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                predict, model = fit_model(kind, x, y)
                residual = y - predict(x)
                sse = float(np.sum(residual ** 2))
                model.update(rmse=float(np.sqrt(np.mean(residual ** 2))),
                             mae=float(np.mean(np.abs(residual))),
                             max_absolute_error=float(np.max(np.abs(residual))),
                             r_squared=1.0 - sse / total_variance if total_variance else None,
                             validation_rmse=None)
                if folds:
                    errors = []
                    for test in folds:
                        fold_predict, _ = fit_model(kind, x[~test], y[~test])
                        errors.extend((y[test] - fold_predict(x[test])).tolist())
                    model["validation_rmse"] = float(np.sqrt(np.mean(np.square(errors))))
                candidates.append(model)
        except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError, RuntimeWarning) as exc:
            skipped.append({"family": kind, "reason": str(exc)})
    if not candidates:
        raise ValueError("No numerically stable candidate found; rescale data or use another family")
    tolerance = max(1e-12, 1e-9 * float(np.max(np.abs(y))))
    exact = [m for m in candidates if m["max_absolute_error"] <= tolerance]
    metric = "validation_rmse" if folds else "rmse"
    if exact:
        chosen = min(exact, key=lambda m: m["parameters"])
        reason = "Simplest candidate matching observations within numerical tolerance"
    else:
        best_error = min(m[metric] for m in candidates)
        close = [m for m in candidates if m[metric] <= max(best_error * 1.05, 1e-12)]
        chosen = min(close, key=lambda m: (m["parameters"], m[metric]))
        reason = f"Simplest candidate within 5% of best {metric}"
    notes = ["Finite observations do not determine a unique equation.",
             "Candidate families are limited; no unrestricted symbolic search was performed."]
    if not folds:
        notes.append("Too few distinct x values for validation; recommendation uses training data only.")
    else:
        notes.append("Validation assumes independent observations; selection scores are not an independent test.")
    if any(len(np.unique(y[x == value])) > 1 for value in unique):
        notes.append("Repeated x values have different y values; an exact single-valued fit is impossible.")
    return {"observations": len(x), "distinct_x": len(unique),
            "observed_x_range": [float(x.min()), float(x.max())],
            "validation_folds": len(folds), "recommended": chosen,
            "selection_reason": reason, "numerical_exact_tolerance": tolerance,
            "candidates": sorted(candidates, key=lambda m: m[metric]),
            "skipped": skipped, "notes": notes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pairs", help="JSON numeric pairs")
    source.add_argument("--csv", help="UTF-8 CSV path")
    parser.add_argument("--x", help="CSV input column name")
    parser.add_argument("--y", help="CSV output column name")
    parser.add_argument("--no-header", action="store_true")
    args = parser.parse_args()
    try:
        result = analyze(*read_pairs(args))
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, OSError, FloatingPointError, OverflowError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
