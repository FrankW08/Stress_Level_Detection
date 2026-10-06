#!/usr/bin/env python3
"""Run reproducible audit experiment A (fixed-parameter models)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stress_detection.audit import PERSISTENT_ERROR_THRESHOLD, run_full_audit  # noqa: E402
from stress_detection.hashes import sha256_file_normalized_lf  # noqa: E402
from stress_detection.io_guard import OutputDirError  # noqa: E402
from stress_detection.run_config import (  # noqa: E402
    ConfigError,
    load_yaml_config,
    resolve_run_settings,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stress level detection audit (experiment A)",
        epilog=(
            "The output directory is required and must be new or empty. "
            "Example: python scripts/run_audit.py StressLevelDataset_original.csv "
            "results/audit_new --config configs/audit_primary.yaml"
        ),
    )
    parser.add_argument(
        "csv",
        type=Path,
        nargs="?",
        default=ROOT / "StressLevelDataset_original.csv",
        help="Path to StressLevelDataset_original.csv",
    )
    parser.add_argument(
        "out_dir",
        type=Path,
        help=(
            "Output directory (required). Must not already contain files. "
            "Example: results/audit_new"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML config (CLI flags override config values except as documented for smoke)",
    )
    parser.add_argument("--n-perm", type=int, default=None, help="Permutation count (>=1)")
    parser.add_argument(
        "--scope",
        choices=("primary", "sensitivity"),
        default=None,
        help="primary: rows 0-1099 convention; sensitivity: policy-dependent",
    )
    parser.add_argument(
        "--smoke",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Fast smoke run (3x1 CV and 5 perms unless --n-splits/--n-repeats/--n-perm given)",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=None,
        help="Outer CV folds (overrides config eval.n_splits and smoke default)",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=None,
        help="Outer CV repeats (overrides config eval.n_repeats and smoke default)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for CV / RF / perms")
    parser.add_argument(
        "--sensitivity-policy",
        choices=("raw", "quarantine_out_of_range", "grouped_duplicates"),
        default=None,
        help="Sensitivity handling (default: raw)",
    )
    parser.add_argument(
        "--near-dup-row-limit",
        type=int,
        default=None,
        help="Row id upper bound for near-duplicate scan (default 1100)",
    )
    args = parser.parse_args()

    file_cfg: dict[str, Any] = {}
    config_info: dict[str, Any] = {"path": None, "sha256_normalized_lf": None}
    try:
        if args.config is not None:
            config_path = args.config.resolve()
            file_cfg = load_yaml_config(config_path)
            config_info = {
                "path": args.config.as_posix(),
                "sha256_normalized_lf": sha256_file_normalized_lf(config_path),
            }
        settings = resolve_run_settings(
            file_cfg=file_cfg,
            persistent_error_threshold=PERSISTENT_ERROR_THRESHOLD,
            cli_scope=args.scope,
            cli_sensitivity_policy=args.sensitivity_policy,
            cli_smoke=args.smoke,
            cli_seed=args.seed,
            cli_n_perm=args.n_perm,
            cli_n_splits=args.n_splits,
            cli_n_repeats=args.n_repeats,
            cli_near_dup_row_limit=args.near_dup_row_limit,
        )
    except ConfigError as exc:
        parser.error(str(exc))

    for note in settings.notes:
        print(f"note: {note}", file=sys.stderr)

    csv_path = args.csv.resolve()
    out_dir = args.out_dir.resolve()

    try:
        results = run_full_audit(
            csv_path,
            out_dir,
            settings=settings,
            project_root=ROOT,
            settings_resolution={
                "config_file": config_info,
                "eval_sources": {
                    "n_splits": settings.sources["n_splits"],
                    "n_repeats": settings.sources["n_repeats"],
                },
                "eval_notes": settings.notes,
                "sources": settings.sources,
            },
        )
    except (ConfigError, OutputDirError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    keys = ("run_mode", "config", "models", "paired", "permutation", "sensitivity")
    print(json.dumps({k: results[k] for k in keys if k in results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
