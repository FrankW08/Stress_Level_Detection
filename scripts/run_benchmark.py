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
from stress_detection.models import audit_model_registry


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark B placeholder — LR/SVM/RF/XGB nested search")
    parser.add_argument("csv", type=Path, nargs="?", default=ROOT / "StressLevelDataset_original.csv")
    parser.add_argument("out_dir", type=Path, nargs="?", default=ROOT / "results" / "benchmark")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    status = {
        "status": "not_implemented_full_nested_search",
        "note": "Use run_audit.py for experiment A. Benchmark B requires shared outer folds + inner GridSearch; roadmap item.",
        "smoke_ran_audit_subset": bool(args.smoke),
    }
    if args.smoke:
        prep = prepare_data(args.csv.resolve(), scope="primary")
        cfg = EvalConfig(n_splits=3, n_repeats=1)
        models = audit_model_registry(prep.feature_names)
        cv = run_audit_cv(prep.X, prep.y, prep.source_row_ids, models, cfg)
        status["smoke_summary"] = cv["summary"]
    (out / "benchmark_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
