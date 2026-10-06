#!/usr/bin/env python3
"""Experiment B: nested hyperparameter search (optional; not run by default in CI)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stress_detection.data import prepare_data
from stress_detection.evaluation import EvalConfig, run_audit_cv
from stress_detection.io_guard import OutputDirError, ensure_empty_output_dir, write_run_status
from stress_detection.models import audit_model_registry


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark B placeholder — LR/SVM/RF/XGB nested search",
        epilog=(
            "The output directory is required and must be new or empty. "
            "Example: python scripts/run_benchmark.py StressLevelDataset_original.csv "
            "results/benchmark_new"
        ),
    )
    parser.add_argument(
        "csv",
        type=Path,
        nargs="?",
        default=ROOT / "StressLevelDataset_original.csv",
    )
    parser.add_argument(
        "out_dir",
        type=Path,
        help="Output directory (required). Must not already contain files.",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    try:
        out = ensure_empty_output_dir(args.out_dir.resolve())
    except OutputDirError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    write_run_status(out, "in_progress", experiment="B")
    status = {
        "status": "not_implemented_full_nested_search",
        "note": (
            "Use run_audit.py for experiment A. Benchmark B requires shared outer folds "
            "+ inner GridSearch; this script does not perform that search."
        ),
        "smoke_ran_audit_subset": bool(args.smoke),
    }
    if args.smoke:
        prep = prepare_data(args.csv.resolve(), scope="primary")
        cfg = EvalConfig(n_splits=3, n_repeats=1)
        models = audit_model_registry(prep.feature_names)
        cv = run_audit_cv(prep.X, prep.y, prep.source_row_ids, models, cfg)
        status["smoke_summary"] = cv["summary"]
    (out / "benchmark_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    write_run_status(out, "completed", experiment="B", note="stub only")
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
