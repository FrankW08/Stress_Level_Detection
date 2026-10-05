#!/usr/bin/env python3
"""Run reproducible audit experiment A (fixed-parameter models)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stress_detection.audit import run_full_audit  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Stress level detection audit (experiment A)")
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
        nargs="?",
        default=ROOT / "results" / "audit_full",
        help="Output directory (created if missing)",
    )
    parser.add_argument("--n-perm", type=int, default=100, help="Permutation count")
    parser.add_argument(
        "--scope",
        choices=("primary", "sensitivity"),
        default="primary",
        help="primary: rows 0-1099 convention; sensitivity: all rows",
    )
    parser.add_argument("--smoke", action="store_true", help="Fast smoke run (3x1 CV, 5 perms)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (via EvalConfig in audit)")
    args = parser.parse_args()

    csv_path = args.csv.resolve()
    out_dir = args.out_dir.resolve()
    if args.smoke:
        out_dir = out_dir.parent / (out_dir.name + "_smoke")

    results = run_full_audit(
        csv_path,
        out_dir,
        scope=args.scope,
        n_perm=args.n_perm,
        smoke=args.smoke,
        project_root=ROOT,
    )
    print(json.dumps({k: results[k] for k in ("run_mode", "models", "paired", "permutation")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
