"""Bounded sparse equation discovery: regression, ODEs, grid PDEs, implicit laws.

Requires Python 3.9+, NumPy and SciPy. Reads JSON; prints an auditable JSON report.
Expressions are evaluated through an allowlisted AST, never Python eval.
"""
import argparse
import ast
import itertools
import json
import math
import re
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter


FUNCTIONS = {"sin": np.sin, "cos": np.cos, "exp": np.exp, "log": np.log,
             "sqrt": np.sqrt, "abs": np.abs, "tanh": np.tanh,
             "sinh": np.sinh, "cosh": np.cosh}


def vector(value, label, minimum=1):
    a = np.asarray(value)
    if a.dtype.kind not in "iuf" or a.ndim != 1 or len(a) < minimum:
        raise ValueError(f"{label} must be a numeric vector with at least {minimum} entries")
    a = a.astype(float)
    if not np.isfinite(a).all():
        raise ValueError(f"{label} contains missing or nonfinite values")
    return a


def variables(raw):
    if not isinstance(raw, dict) or not raw:
        raise ValueError("Provide a nonempty variables/states object")
    for name in raw:
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) or name in FUNCTIONS:
            raise ValueError(f"Use a nonreserved simple identifier for variable: {name}")
    result = {name: vector(values, name) for name, values in raw.items()}
    if len({len(a) for a in result.values()}) != 1:
        raise ValueError("Variable columns must have equal lengths")
    return result


def expression(text, env):
    """Safe numeric grammar. No attributes, subscripts, comprehensions or imports."""
    tree = ast.parse(text, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 120:
        raise ValueError("Expression exceeds 120 AST nodes")

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Name) and node.id in env:
            return env[node.id]
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            if not math.isfinite(node.value) or abs(node.value) > 1e12:
                raise ValueError("Invalid numeric constant")
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        if isinstance(node, ast.BinOp):
            a, b = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                return a + b
            if isinstance(node.op, ast.Sub):
                return a - b
            if isinstance(node.op, ast.Mult):
                return a * b
            if isinstance(node.op, ast.Div):
                if np.any(np.abs(b) < 1e-12):
                    raise ValueError("Near-zero denominator")
                return a / b
            if isinstance(node.op, ast.Pow):
                if not np.isscalar(b) or abs(b) > 8:
                    raise ValueError("Exponents must be constants with absolute value <=8")
                return np.power(a, b)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in FUNCTIONS and len(node.args) == 1 and not node.keywords):
            return FUNCTIONS[node.func.id](visit(node.args[0]))
        raise ValueError("Unsupported expression syntax or unknown name")

    with np.errstate(all="raise"):
        out = np.broadcast_to(visit(tree), next(iter(env.values())).shape).astype(float)
    if not np.isfinite(out).all():
        raise ValueError("Expression produced nonfinite values")
    return out


def library(env, args):
    names = list(env)
    terms = ["1"]
    if len(names) > 20:
        raise ValueError("At most 20 library variables; restrict variables or PDE derivative order")
    for degree in range(1, args.degree + 1):
        for combo in itertools.combinations_with_replacement(names, degree):
            terms.append("*".join(combo))
            if len(terms) > args.max_features:
                raise ValueError("Polynomial library exceeds --max-features; reduce degree/variables")
    functions = [] if args.functions == "none" else args.functions.split(",")
    for func in functions:
        if func not in FUNCTIONS and func != "inverse":
            raise ValueError(f"Unknown function: {func}")
        for name in names:
            if func in ("sin", "cos"):
                terms.extend(f"{func}(({frequency})*{name})" for frequency in args.frequencies.split(","))
            else:
                terms.append(f"1/({name})" if func == "inverse" else f"{func}({name})")
    terms.extend(args.term)
    if len(terms) > args.max_features:
        raise ValueError("Library exceeds --max-features; reduce the search")
    columns, accepted, skipped = [], [], []
    for term in dict.fromkeys(terms):
        try:
            values = expression(term, env)
            if np.max(np.abs(values)) > 1e100:
                raise ValueError("Feature magnitude exceeds numerical limit")
            columns.append(values)
            accepted.append(term)
        except (ValueError, SyntaxError, FloatingPointError, OverflowError, TypeError) as exc:
            if term in args.term:
                raise ValueError(f"Requested term {term!r} is invalid: {exc}") from exc
            skipped.append({"term": term, "reason": str(exc)})
    return np.column_stack(columns), accepted, skipped


def sparse_path(a, b, max_terms):
    """Normalized OMP path plus thresholded least squares; fit on training only."""
    scales = np.linalg.norm(a, axis=0)
    scales[scales < 1e-14] = 1
    z = a / scales
    supports, active, residual = [()], [], b.copy()
    for _ in range(min(max_terms, a.shape[1], max(1, a.shape[0] // 3))):
        scores = np.abs(z.T @ residual)
        scores[active] = -1
        chosen = int(np.argmax(scores))
        if scores[chosen] <= 1e-14:
            break
        active.append(chosen)
        supports.append(tuple(sorted(active)))
        residual = b - z[:, active] @ np.linalg.lstsq(z[:, active], b, rcond=1e-10)[0]
    if a.shape[0] > a.shape[1]:
        full = np.linalg.lstsq(z, b, rcond=1e-10)[0]
        for threshold in [1e-10, 1e-7, 1e-4, .001, .01, .05, .1]:
            support = np.flatnonzero(np.abs(full) > threshold * max(np.max(np.abs(full)), 1e-14))
            for _ in range(8):
                if not len(support):
                    break
                coefficients = np.linalg.lstsq(z[:, support], b, rcond=1e-10)[0]
                keep = np.abs(coefficients) > threshold * max(np.max(np.abs(coefficients)), 1e-14)
                if keep.all():
                    break
                support = support[keep]
            if len(support) <= max_terms:
                supports.append(tuple(support))
    results = []
    for support in dict.fromkeys(supports):
        coefficients = np.zeros(a.shape[1])
        if support:
            coefficients[list(support)] = np.linalg.lstsq(z[:, support], b, rcond=1e-10)[0] / scales[list(support)]
        results.append(coefficients)
    return results


def rmse(y, estimate):
    return float(np.sqrt(np.mean((y - estimate) ** 2)))


def select_fit(a, b, train, valid, names, target, args):
    if len(train) < 8 or len(valid) < 3:
        raise ValueError("Need at least 8 training and 3 validation observations")
    candidates = []
    for coef in sparse_path(a[train], b[train], args.max_terms):
        error = rmse(b[valid], a[valid] @ coef)
        candidates.append((error, int(np.count_nonzero(coef)), coef))
    best = min(item[0] for item in candidates)
    tolerance = max(args.selection_atol, args.selection_rtol * float(np.max(np.abs(b[train]))))
    near = [item for item in candidates if item[0] <= max(best * 1.05, tolerance)]
    error, count, coef = min(near, key=lambda item: (item[1], item[0]))
    residual = b[valid] - a[valid] @ coef
    variance = float(np.sum((b[valid] - np.mean(b[valid])) ** 2))
    support = np.flatnonzero(coef)
    active = a[train][:, support]
    scaled = active / np.maximum(np.linalg.norm(active, axis=0), 1e-14) if count else active
    condition = float(np.linalg.cond(scaled)) if count else 1.0
    return {"target": target, "equation": target + " = " + (" + ".join(
                f"({coef[i]:.12g})*({names[i]})" for i in support) or "0"),
            "terms": [{"expression": names[i], "coefficient": float(coef[i])} for i in support],
            "training_rmse": rmse(b[train], a[train] @ coef),
            "validation_rmse": error, "validation_mae": float(np.mean(np.abs(residual))),
            "validation_r_squared": 1 - float(np.sum(residual**2)) / variance if variance else None,
            "selected_terms": count, "selected_design_condition": condition if math.isfinite(condition) else None,
            "training_rows": len(train), "validation_rows": len(valid),
            "coefficients_fitted_on": "training subset only; no refit on validation",
            "candidate_supports_evaluated": len(candidates)}


def split_rows(env, ordered=False, groups=None, guard=4):
    n = len(next(iter(env.values())))
    if groups is not None:
        unique = np.unique(groups)
        if len(unique) >= 2:
            held = groups == unique[-1]
            return np.flatnonzero(~held), np.flatnonzero(held), "last trajectory held out"
    if ordered:
        cut = int(n * .8)
        # Guard band prevents finite-difference/smoothing stencils crossing split.
        gap = max(guard, int(n * .02))
        return np.arange(max(0, cut-gap)), np.arange(min(n, cut+gap), n), "chronological 80/20 with guard band"
    rows = np.column_stack(list(env.values()))
    _, inverse = np.unique(rows, axis=0, return_inverse=True)
    groups = np.arange(inverse.max()+1)
    np.random.default_rng(42).shuffle(groups)
    cut = int(len(groups)*.8)
    held = np.isin(inverse, groups[cut:])
    return np.flatnonzero(~held), np.flatnonzero(held), "seeded 80/20 grouped by identical inputs"


def derivative(a, axis_values, axis, order):
    result = a
    for _ in range(order):
        estimated = np.gradient(result, axis_values, axis=axis, edge_order=2)
        spacing = np.diff(axis_values)
        if len(axis_values) >= 5 and np.allclose(spacing,spacing[0],rtol=1e-7,atol=1e-12):
            source = np.moveaxis(result,axis,0)
            output = np.moveaxis(estimated,axis,0)
            output[2:-2] = (source[:-4]-8*source[1:-3]+8*source[3:-1]-source[4:])/(12*spacing[0])
        result = estimated
    return result


def time_axis(raw, minimum=15):
    t = vector(raw, "time/axis", minimum)
    if not np.all(np.diff(t) > 0):
        raise ValueError("Time and grid axes must be strictly increasing, without duplicates")
    return t


def prepare_ode(data, args):
    trajectories = data.get("trajectories", [data])
    if not isinstance(trajectories, list) or not trajectories:
        raise ValueError("trajectories must be a nonempty list")
    parts, targets, groups = [], [], []
    keys = None
    trim = args.smooth_window // 2 + 2*args.order + 2
    for group, entry in enumerate(trajectories):
        t = time_axis(entry["t"], max(30, 2*trim+15))
        states = variables(entry["states"])
        if keys is None:
            keys = list(states)
        if list(states) != keys or len(next(iter(states.values()))) != len(t):
            raise ValueError("All trajectories need the same ordered state names and matching time lengths")
        if args.smooth_window:
            if args.smooth_window < 5 or args.smooth_window % 2 != 1 or args.smooth_window >= len(t):
                raise ValueError("Smoothing window must be odd, >=5 and shorter than each trajectory")
            if not np.allclose(np.diff(t), np.diff(t)[0], rtol=1e-4, atol=1e-12):
                raise ValueError("Savitzky-Golay smoothing requires a uniform time grid")
            states = {k: savgol_filter(v, args.smooth_window, min(3,args.smooth_window-1)) for k,v in states.items()}
        env, outputs = {}, {}
        for name, values in states.items():
            env[name] = values
            for order in range(1, args.order+1):
                d = derivative(values, t, 0, order)
                label = f"d{order}_{name}_dt{order}"
                if label in states:
                    raise ValueError("State name collides with generated derivative identifier")
                if order < args.order:
                    env[label] = d
                else:
                    outputs[label] = d
        if args.include_time:
            if "time" in env:
                raise ValueError("State 'time' conflicts with --include-time")
            env["time"] = t
        keep = slice(trim,-trim)
        parts.append({k: v[keep] for k,v in env.items()})
        targets.append({k: v[keep] for k,v in outputs.items()})
        groups.extend([group]*len(t[keep]))
    env = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    outputs = {k: np.concatenate([p[k] for p in targets]) for k in targets[0]}
    # Explicit training/validation derivative arrays can avoid numerical differentiation.
    return env, outputs, np.asarray(groups)


def prepare_pde(data, args):
    t = time_axis(data["t"], 15)
    axes = data["axes"]
    if not isinstance(axes, list) or not 1 <= len(axes) <= 3:
        raise ValueError("PDE needs 1 to 3 spatial axes")
    axes = [time_axis(axis, 2*(2*args.order+2)+5) for axis in axes]
    u = np.asarray(data["u"])
    if u.dtype.kind not in "iuf" or u.shape != (len(t), *(len(axis) for axis in axes)):
        raise ValueError("u shape must be [time, x0, x1, ...] matching the axes")
    u = u.astype(float)
    if not np.isfinite(u).all():
        raise ValueError("u contains missing/nonfinite values")
    trim = 2*args.order+2
    keep = (slice(2,-2),) + (slice(trim,-trim),)*len(axes)
    env = {"u": u[keep].ravel()}
    # Mixed spatial derivatives up to the requested total order.
    for total in range(1,args.order+1):
        for combo in itertools.combinations_with_replacement(range(len(axes)),total):
            d = u
            for axis in combo:
                d = derivative(d,axes[axis],axis+1,1)
            label = "u_" + "_".join(f"x{axis}" for axis in combo)
            env[label] = d[keep].ravel()
    target = derivative(u,t,0,1)[keep].ravel()
    time_indices = np.broadcast_to(np.arange(len(t))[:,None],(len(t),int(np.prod(u.shape[1:])))).reshape(u.shape)[keep].ravel()
    cut = int(len(t)*.8)
    train = np.flatnonzero(time_indices < cut-2)
    valid = np.flatnonzero(time_indices >= cut+2)
    return env, {"du_dt":target}, train, valid


def implicit_fit(a, names, train, valid):
    if a.shape[0] <= a.shape[1] * 2 or len(train) <= a.shape[1]:
        raise ValueError("Implicit discovery needs more observations than library terms; reduce the library")
    scale = np.linalg.norm(a[train],axis=0)
    if np.any(scale < 1e-12):
        raise ValueError("Implicit library contains a zero feature; remove degenerate variables/terms")
    _, s, vh = np.linalg.svd(a[train]/scale,full_matrices=False)
    vector_ = vh[-1]/scale
    active = np.abs(vector_*scale) > 1e-5*np.max(np.abs(vector_*scale))
    if active.sum() < 2:
        raise ValueError("Only a trivial implicit relation was found")
    vector_[~active] = 0
    vector_ /= np.max(np.abs(vector_))
    contributions = a[valid]*vector_
    denominator = float(np.sqrt(np.mean(np.sum(contributions**2,axis=1))))
    error = rmse(np.zeros(len(valid)),contributions.sum(axis=1))
    return {"equation":" + ".join(f"({vector_[i]:.12g})*({names[i]})" for i in np.flatnonzero(active))+" = 0",
            "terms":[{"expression":names[i],"coefficient":float(vector_[i])} for i in np.flatnonzero(active)],
            "normalized_validation_residual":error/max(denominator,1e-14),
            "training_smallest_to_largest_singular_ratio":float(s[-1]/s[0]),
            "near_null_directions":int(np.sum(s <= s[0]*1e-7)),
            "warning":"Homogeneous least squares candidate; multiple null directions mean ambiguity, not unique discovery."}


def discover(data,args):
    notes = ["Bounded candidate-library search, not discovery of every possible equation.",
             "Validation used for selection is not an independent final test.",
             "Correlated variables and insufficient excitation can make laws unidentifiable."]
    if args.mode == "ode":
        env,outputs,groups = prepare_ode(data,args)
        train,valid,strategy = split_rows(env,True,groups,
                                        args.smooth_window//2+2*args.order+2)
        notes += ["Derivatives are numerical estimates; noise and sampling resolution affect results.",
                  "Reported errors compare derivatives, not integrated trajectory predictions."]
    elif args.mode == "pde":
        if args.smooth_window:
            raise ValueError("--smooth-window is supported only in ODE mode")
        env,outputs,train,valid = prepare_pde(data,args)
        strategy = "chronological time-slab holdout with derivative guard band"
        notes += ["PDE mode fits scalar first-order time evolution on a complete rectangular grid.",
                  "Spatial derivatives include mixed terms; time derivatives are first order.",
                  "No PDE simulation validation is performed; errors are derivative residuals."]
    else:
        env = variables(data["variables"])
        if args.mode == "implicit" and (args.functions != "none" or args.term):
            raise ValueError("Implicit mode currently requires --functions none and no custom terms, to avoid trivial function identities")
        train,valid,strategy = split_rows(env,args.ordered)
        outputs = {} if args.mode == "implicit" else variables(data["targets"])
        if any(len(v) != len(next(iter(env.values()))) for v in outputs.values()):
            raise ValueError("Targets must match input lengths")
    a,names,skipped = library(env,args)
    if len(train) < 8 or len(valid) < 3:
        raise ValueError("Insufficient independent training/validation rows")
    if args.mode == "implicit":
        models = [implicit_fit(a,names,train,valid)]
        notes.append("Implicit relations are not solved into a unique explicit function.")
    else:
        models = [select_fit(a,y,train,valid,names,target,args) for target,y in outputs.items()]
    normalized = a[train]/np.maximum(np.linalg.norm(a[train],axis=0),1e-14)
    singular = np.linalg.svd(normalized,compute_uv=False)
    rank = int(np.sum(singular > singular[0]*max(normalized.shape)*np.finfo(float).eps))
    effective_rank = int(np.sum(singular > singular[0]*1e-7))
    if effective_rank < a.shape[1]:
        notes.append("Candidate library is rank deficient or nearly collinear at relative tolerance 1e-7: multiple coefficient sets may explain these data.")
    return {"mode":args.mode,"models":models,"library_terms":names,
            "library_size":len(names),"training_library_rank":rank,
            "training_library_effective_rank":effective_rank,"effective_rank_relative_tolerance":1e-7,
            "skipped_domain_terms":skipped,"validation_strategy":strategy,
            "settings":{"degree":args.degree,"derivative_order":args.order,
                        "max_terms":args.max_terms,"functions":args.functions,
                        "frequencies":args.frequencies,"custom_terms":args.term,
                        "smoothing_window":args.smooth_window,
                        "selection_atol":args.selection_atol,
                        "selection_rtol":args.selection_rtol},"notes":notes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",required=True,help="JSON input path")
    parser.add_argument("--mode",choices=["regression","ode","pde","implicit"],default="regression")
    parser.add_argument("--degree",type=int,default=2)
    parser.add_argument("--order",type=int,default=1,help="ODE order or maximum spatial PDE derivative order")
    parser.add_argument("--functions",default="sin,cos,exp,log,sqrt,inverse,tanh")
    parser.add_argument("--frequencies",default="1",help="Comma-separated numeric trial frequencies")
    parser.add_argument("--term",action="append",default=[],help="Extra allowlisted expression; repeatable")
    parser.add_argument("--max-terms",type=int,default=12)
    parser.add_argument("--max-features",type=int,default=400)
    parser.add_argument("--selection-atol",type=float,default=1e-10)
    parser.add_argument("--selection-rtol",type=float,default=1e-7)
    parser.add_argument("--smooth-window",type=int,default=0)
    parser.add_argument("--include-time",action="store_true")
    parser.add_argument("--ordered",action="store_true",help="Chronological regression/implicit split")
    args = parser.parse_args()
    try:
        if not (0 <= args.degree <= 8 and 1 <= args.order <= 4 and 1 <= args.max_terms <= 50
                and 1 <= args.max_features <= 1000 and args.smooth_window >= 0):
            raise ValueError("Limits: degree 0..8, order 1..4, max-terms 1..50, max-features 1..1000")
        if not all(math.isfinite(v) and v>=0 for v in (args.selection_atol,args.selection_rtol)):
            raise ValueError("Selection tolerances must be finite and nonnegative")
        frequencies = [float(v) for v in args.frequencies.split(",")]
        if not frequencies or len(frequencies)>30 or not all(math.isfinite(v) and abs(v)<=1e6 for v in frequencies):
            raise ValueError("Provide up to 30 finite trial frequencies with absolute value <=1e6")
        args.frequencies = ",".join(str(v) for v in frequencies)
        with np.errstate(over="raise",invalid="raise",divide="raise"), warnings.catch_warnings():
            warnings.simplefilter("error",RuntimeWarning)
            report = discover(json.loads(Path(args.data).read_text(encoding="utf-8-sig")),args)
        print(json.dumps(report,indent=2,allow_nan=False))
    except (ValueError,KeyError,TypeError,OSError,FloatingPointError,OverflowError,RuntimeWarning,np.linalg.LinAlgError) as exc:
        print(json.dumps({"error":str(exc)}),file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
