"""Command-line interface for The1.

Examples
--------
    python -m the1 discover data.csv --target force --features mass,acceleration
    python -m the1 discover data.csv --all-targets --library .the1_laws.json --remember
    python -m the1 batch ./datasets --pattern "*.csv" --out report.json
    python -m the1 profile data.csv --target y
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from .core import The1WorldModel
from .explorer import DatasetExplorer, reports_to_frame, save_reports
from .library import LawLibrary


def _build_explorer(args) -> DatasetExplorer:
    library = None
    if args.library:
        library = LawLibrary(args.library)
    model = The1WorldModel(
        random_state=args.seed,
        law_library=library,
        remember=bool(args.remember and library is not None),
        max_search_samples=args.max_search_samples,
        use_evolutionary=not args.no_evolutionary,
        use_rational=not args.no_rational,
        verify_with_lean=args.lean,
        lean_project=args.lean_project,
        lean_timeout=args.lean_timeout,
    )
    explorer = DatasetExplorer(model=model, max_features=args.max_features,
                               max_rows=args.max_rows, random_state=args.seed,
                               library=library)
    return explorer


def _print_reports(reports) -> None:
    for r in reports:
        print(r.summary())
        print()


def cmd_profile(args) -> int:
    explorer = DatasetExplorer()
    profile = explorer.profile(args.file, target=args.target)
    print(json.dumps(profile, indent=2, ensure_ascii=False))
    return 0


def cmd_discover(args) -> int:
    explorer = _build_explorer(args)
    if args.all_targets:
        reports = explorer.discover_all(
            args.file,
            targets=args.targets,
            max_targets=args.max_targets,
            split=args.split,
            val_fraction=args.val_fraction,
            verbose=args.verbose,
        )
    else:
        features = args.features.split(",") if args.features else None
        reports = [explorer.discover(
            args.file,
            target=args.target,
            features=features,
            split=args.split,
            val_fraction=args.val_fraction,
            verbose=args.verbose,
        )]
    _print_reports(reports)
    if args.json:
        save_reports(reports, args.json)
        print(f"Saved report to {args.json}")
    if args.csv:
        reports_to_frame(reports).to_csv(args.csv, index=False)
        print(f"Saved table to {args.csv}")
    return 0


def cmd_batch(args) -> int:
    explorer = _build_explorer(args)
    directory = Path(args.directory)
    if args.all_targets:
        reports = []
        for path in sorted(directory.glob(args.pattern)):
            try:
                reports.extend(explorer.discover_all(
                    path, max_targets=args.max_targets,
                    split=args.split, val_fraction=args.val_fraction,
                ))
            except Exception as exc:
                print(f"[skip] {path}: {exc}", file=sys.stderr)
    else:
        reports = explorer.batch(
            directory, pattern=args.pattern,
            split=args.split, val_fraction=args.val_fraction,
        )
    _print_reports(reports)
    if args.out:
        save_reports(reports, args.out)
        print(f"Saved {len(reports)} reports to {args.out}")
    if args.csv:
        reports_to_frame(reports).to_csv(args.csv, index=False)
        print(f"Saved table to {args.csv}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="the1", description="The1 inductive law discovery")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--split", choices=["ood", "random", "none"], default="ood")
        sp.add_argument("--val-fraction", type=float, default=0.2)
        sp.add_argument("--seed", type=int, default=42)
        sp.add_argument("--max-rows", type=int, default=20000)
        sp.add_argument("--max-search-samples", type=int, default=4000)
        sp.add_argument("--max-features", type=int, default=8)
        sp.add_argument("--library", type=str, default=None,
                        help="Path to a persistent law library (JSON)")
        sp.add_argument("--remember", action="store_true",
                        help="Append discovered laws to the library")
        sp.add_argument("--no-evolutionary", action="store_true")
        sp.add_argument("--no-rational", action="store_true")
        sp.add_argument("--lean", action="store_true",
                        help="Check the discovered law's hypotheses in Lean 4")
        sp.add_argument("--lean-project", default=None,
                        help="Path to a Lake project depending on Mathlib")
        sp.add_argument("--lean-timeout", type=float, default=180.0)
        sp.add_argument("--verbose", action="store_true")

    sp_profile = sub.add_parser("profile", help="Profile a dataset")
    sp_profile.add_argument("file")
    sp_profile.add_argument("--target", default=None)
    sp_profile.set_defaults(func=cmd_profile)

    sp_disc = sub.add_parser("discover", help="Discover a law for one or all targets")
    sp_disc.add_argument("file")
    sp_disc.add_argument("--target", default=None)
    sp_disc.add_argument("--features", default=None, help="Comma-separated feature columns")
    sp_disc.add_argument("--all-targets", action="store_true")
    sp_disc.add_argument("--targets", nargs="*", default=None)
    sp_disc.add_argument("--max-targets", type=int, default=6)
    sp_disc.add_argument("--json", default=None)
    sp_disc.add_argument("--csv", default=None)
    common(sp_disc)
    sp_disc.set_defaults(func=cmd_discover)

    sp_batch = sub.add_parser("batch", help="Discover across a directory of datasets")
    sp_batch.add_argument("directory")
    sp_batch.add_argument("--pattern", default="*.csv")
    sp_batch.add_argument("--all-targets", action="store_true")
    sp_batch.add_argument("--max-targets", type=int, default=4)
    sp_batch.add_argument("--out", default=None)
    sp_batch.add_argument("--csv", default=None)
    common(sp_batch)
    sp_batch.set_defaults(func=cmd_batch)
    return p


def main(argv: Optional[list] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
