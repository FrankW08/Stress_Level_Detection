#!/usr/bin/env python3
"""Check how much of stress_level is recoverable from single fields (descriptive, no new claims).

Example:
  python scripts/run_label_proxy_check.py results/label_proxy_check_new \
      --audit-results results/audit_full_4865bb3_clean/results.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stress_detection.data import prepare_data  # noqa: E402
from stress_detection.io_guard import OutputDirError, ensure_empty_output_dir  # noqa: E402
from stress_detection.label_proxy import (  # noqa: E402
    PROXY_FEATURE,
    crosstab,
    feature_label_purity,
    label_proxy_report,
)


def git_state() -> dict[str, object]:
    """HEAD commit and whether uncommitted changes exist (the result is not tied to HEAD if dirty)."""
    try:
        run = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout
        return {"commit": run("rev-parse", "HEAD").strip(), "dirty": bool(run("status", "--porcelain").strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out_dir", type=Path, help="New or empty output directory")
    parser.add_argument("--csv", type=Path, default=ROOT / "StressLevelDataset_original.csv")
    parser.add_argument(
        "--audit-results",
        type=Path,
        required=True,
        help="Experiment A results.json; its single_feature_exploratory scores pick the strong features",
    )
    parser.add_argument("--strong-threshold", type=float, default=0.85)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    audit = json.loads(args.audit_results.read_text(encoding="utf-8"))
    single = audit.get("single_feature_exploratory") or {}
    strong = [f for f, s in single.items() if s >= args.strong_threshold]
    if not strong:
        print("audit results have no single_feature_exploratory scores above the threshold", file=sys.stderr)
        return 2

    prep = prepare_data(args.csv, "primary")  # validate inputs before touching the output path
    try:
        out = ensure_empty_output_dir(args.out_dir)
    except OutputDirError as exc:
        print(exc, file=sys.stderr)
        return 2

    report = label_proxy_report(
        prep.X, prep.y, strong_single_features=strong,
        n_splits=args.n_splits, n_repeats=args.n_repeats, seed=args.seed,
    )
    report.update(
        {
            "rows": int(len(prep.y)),
            "scope": "primary",
            "csv_sha256_normalized_lf": prep.csv_sha256_normalized_lf,
            "git": git_state(),
            "audit_results": str(args.audit_results),
            "strong_threshold": args.strong_threshold,
            "single_feature_exploratory_from_audit": {f: single[f] for f in strong},
            "interpretation": (
                "Descriptive. A near-deterministic feature-label table means the label is reproducible "
                "from that field in this CSV; it does not establish how the label was produced."
            ),
        }
    )
    crosstab(prep.X, prep.y).to_csv(out / f"{PROXY_FEATURE}_by_label.csv", encoding="utf-8")
    feature_label_purity(prep.X, prep.y).to_csv(out / "feature_label_purity.csv", index=False, encoding="utf-8")
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, v in report["lr_variants"].items():
        print(f"{name:40s} macro-F1 {v['mean']:.3f} ± {v['std']:.3f}  ({v['n_features']} features)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
