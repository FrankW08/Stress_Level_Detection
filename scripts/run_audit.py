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
from stress_detection.run_config import (  # noqa: E402
    ConfigError,
    resolve_eval_settings,
    validate_file_config,
)


def _load_yaml_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text) or {}
        if not isinstance(data, dict):
            raise ValueError("config root must be a mapping")
        return data
    except ImportError:
        # Minimal YAML subset: key: value lines and one-level nested maps
        cfg: dict[str, Any] = {}
        current: dict[str, Any] | None = None
        current_key: str | None = None
        for line in text.splitlines():
            if not line.strip() or line.strip().startswith("#"):
                continue
            if line.startswith("  ") and current is not None and current_key:
                k, _, v = line.strip().partition(":")
                current[k.strip()] = _parse_scalar(v.strip())
            elif ":" in line and not line.startswith(" "):
                k, _, v = line.partition(":")
                k = k.strip()
                v = v.strip()
                if v == "":
                    current = {}
                    current_key = k
                    cfg[k] = current
                else:
                    current = None
                    current_key = None
                    cfg[k] = _parse_scalar(v)
        return cfg


def _parse_scalar(v: str) -> Any:
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v.strip('"').strip("'")


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
        help="Output directory (created if missing; used exactly as specified)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML config (CLI flags override config values)",
    )
    parser.add_argument("--n-perm", type=int, default=None, help="Permutation count")
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
        help="Fast smoke run (3x1 CV unless --n-splits/--n-repeats given, 5 perms)",
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
    if args.config is not None:
        config_path = args.config.resolve()
        file_cfg = _load_yaml_config(config_path)
        config_info = {
            "path": args.config.as_posix(),
            "sha256_normalized_lf": sha256_file_normalized_lf(config_path),
        }

    try:
        eval_cfg = validate_file_config(
            file_cfg, persistent_error_threshold=PERSISTENT_ERROR_THRESHOLD
        )
    except ConfigError as exc:
        parser.error(f"invalid config: {exc}")

    def pick(cli_val: Any, *keys: str, default: Any) -> Any:
        if cli_val is not None:
            return cli_val
        for k in keys:
            if k in file_cfg:
                return file_cfg[k]
            if k in eval_cfg:
                return eval_cfg[k]
        return default

    scope = pick(args.scope, "scope", default="primary")
    smoke = pick(args.smoke, "smoke", default=False)
    seed = int(pick(args.seed, "seed", default=0))
    n_perm = int(pick(args.n_perm, "n_perm", default=100))
    sensitivity_policy = pick(
        args.sensitivity_policy, "sensitivity_policy", default="raw"
    )
    near_dup = int(
        pick(args.near_dup_row_limit, "near_dup_row_limit", "tail_quarantine_from", default=1100)
    )
    tail_q = int(file_cfg.get("tail_quarantine_from", 1100))

    if not isinstance(smoke, bool):
        parser.error(f"invalid config: smoke must be true/false, got {smoke!r}")
    try:
        eval_settings = resolve_eval_settings(
            smoke=smoke,
            cli_n_splits=args.n_splits,
            cli_n_repeats=args.n_repeats,
            file_eval=eval_cfg,
        )
    except ConfigError as exc:
        parser.error(str(exc))
    for note in eval_settings["notes"]:
        print(f"note: {note}", file=sys.stderr)

    csv_path = args.csv.resolve()
    out_dir = args.out_dir.resolve()  # exact path; no automatic _smoke suffix

    results = run_full_audit(
        csv_path,
        out_dir,
        scope=scope,
        n_perm=n_perm,
        smoke=smoke,
        seed=seed,
        project_root=ROOT,
        tail_quarantine_from=tail_q,
        near_dup_row_limit=near_dup,
        sensitivity_policy=sensitivity_policy,
        n_splits=eval_settings["n_splits"],
        n_repeats=eval_settings["n_repeats"],
        settings_resolution={
            "config_file": config_info,
            "eval_sources": eval_settings["sources"],
            "eval_notes": eval_settings["notes"],
        },
    )
    keys = ("run_mode", "config", "models", "paired", "permutation", "sensitivity")
    print(json.dumps({k: results[k] for k in keys if k in results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
