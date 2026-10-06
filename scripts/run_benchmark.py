#!/usr/bin/env python3
"""Experiment B nested search — design dump only; does not run the search."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stress_detection.experiment_b import experiment_b_status
from stress_detection.io_guard import OutputDirError, ensure_empty_output_dir, write_run_status


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Experiment B: write nested-search design (search is not executed)",
        epilog=(
            "The output directory is required and must be new or empty. "
            "Example: python scripts/run_benchmark.py results/benchmark_design"
        ),
    )
    parser.add_argument(
        "out_dir",
        type=Path,
        help="Output directory (required). Must not already contain files.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Accepted for compatibility; still does not run the nested search.",
    )
    args = parser.parse_args()

    try:
        out = ensure_empty_output_dir(args.out_dir.resolve())
    except OutputDirError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    write_run_status(out, "in_progress", experiment="B")
    status = experiment_b_status()
    status["smoke_flag"] = bool(args.smoke)
    status["plan_doc"] = "docs/RESEARCH_PLAN.md"
    (out / "benchmark_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    write_run_status(out, "completed", experiment="B", note="design only; search not executed")
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
