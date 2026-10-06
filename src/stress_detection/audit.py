"""Full audit pipeline: duplicates, permutations, persistent errors, artifacts."""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier

from stress_detection.data import (
    SensitivityPolicy,
    apply_label_driven_zero_map,
    feature_columns,
    feature_values_differ,
    invalid_or_missing_label_table,
    label_conflict_table,
    load_raw_csv,
    parse_dataset,
    prepare_data,
)
from stress_detection.evaluation import (
    FOLD_AGGREGATION,
    REPEAT_AGGREGATION,
    EvalConfig,
    check_split_feasibility,
    paired_comparison,
    plan_outer_splits,
    run_audit_cv,
)
from stress_detection.hashes import (
    code_manifest_sha256,
    sha256_bytes,
    write_environment_lock,
)
from stress_detection.models import (
    audit_model_registry,
    lr_pipeline,
    svm_linear_pipeline,
    tree_depth3_pipeline,
)
from stress_detection.io_guard import OutputDirError, ensure_empty_output_dir, write_run_status
from stress_detection.run_config import ConfigError, RunSettings, resolve_run_settings

CORE_MODELS = ["lr_all", "rf_all", "svm_all", "nested_single"]
PERSISTENT_ERROR_THRESHOLD = 0.8


def duplicate_audit(
    raw: pd.DataFrame,
    parsed: pd.DataFrame,
    *,
    near_dup_row_limit: int = 1100,
    include_near: bool = True,
) -> dict[str, pd.DataFrame]:
    """Exact dupes, normalized dupes, same features different label, near-duplicate candidates."""
    feature_cols = feature_columns(parsed)
    exact = raw.duplicated(keep=False)
    exact_df = pd.DataFrame(
        {"source_row_id": parsed.loc[exact, "source_row_id"].astype(int).tolist()}
    )
    norm_key = parsed[feature_cols + ["stress_level"]].astype("string")
    norm_dup = norm_key.duplicated(keep=False)
    norm_df = parsed.loc[norm_dup, ["source_row_id"] + feature_cols + ["stress_level"]]

    conflict_df = label_conflict_table(parsed)
    invalid_labels = invalid_or_missing_label_table(parsed)

    if include_near:
        near_rows: list[dict] = []
        primary = parsed[parsed["source_row_id"] < near_dup_row_limit]
        arr = primary[feature_cols].to_numpy(dtype=float)
        ids = primary["source_row_id"].astype(int).to_numpy()
        n = len(primary)
        for i in range(n):
            for j in range(i + 1, n):
                n_diff = int(feature_values_differ(arr[i], arr[j]).sum())
                if n_diff == 1:
                    near_rows.append(
                        {
                            "source_row_id_a": int(ids[i]),
                            "source_row_id_b": int(ids[j]),
                            "n_feature_diffs": 1,
                        }
                    )
        near_df = pd.DataFrame(near_rows)
    else:
        near_df = pd.DataFrame(
            [{"note": "skipped in smoke mode; run full audit for near-duplicate tables"}]
        )
    return {
        "exact_duplicate_rows": exact_df,
        "normalized_duplicate_rows": norm_df,
        "same_features_different_label": conflict_df,
        "invalid_or_missing_labels": invalid_labels,
        "near_duplicate_candidates": near_df,
    }


PERMUTATION_N_SPLITS = 5


def permutation_protocol(
    *,
    seed: int,
    n_perm: int,
    grouped: bool,
    run_generalization: bool,
) -> dict[str, Any]:
    if not run_generalization:
        return {
            "status": "not_available",
            "reason": "generalization / permutation skipped because the chosen scope refused CV",
            "splitter": None,
            "n_splits": None,
            "n_repeats": None,
            "seed": seed,
            "n_perm": n_perm,
            "uses_groups": None,
            "label_permutation_unit": None,
            "difference_from_main_cv": None,
        }
    if grouped:
        return {
            "status": "not_available",
            "reason": (
                "grouped_duplicates treats identical-feature rows as groups, not confirmed "
                "subject identifiers. A valid group-aware permutation test would need an "
                "exchangeability assumption for those groups; groups with conflicting labels "
                "have no single group-level label. Row-wise permutation under "
                "StratifiedGroupKFold is not a valid grouped permutation test and is not reported."
            ),
            "splitter": None,
            "n_splits": None,
            "n_repeats": None,
            "seed": seed,
            "n_perm": n_perm,
            "uses_groups": None,
            "label_permutation_unit": None,
            "difference_from_main_cv": (
                "Main CV uses StratifiedGroupKFold on identical-feature groups. "
                "No grouped permutation protocol is implemented."
            ),
        }
    return {
        "status": "completed",
        "splitter": "StratifiedKFold",
        "n_splits": PERMUTATION_N_SPLITS,
        "n_repeats": 1,
        "seed": seed,
        "n_perm": n_perm,
        "uses_groups": False,
        "label_permutation_unit": "row",
        "difference_from_main_cv": (
            "Main CV is RepeatedStratifiedKFold (n_splits x n_repeats). "
            "Permutation uses a single StratifiedKFold(5) on the same included rows."
        ),
    }


def run_permutations(
    X: pd.DataFrame,
    labels: np.ndarray,
    *,
    n_perm: int,
    seed: int,
) -> pd.DataFrame:
    def paired_scores(lab: np.ndarray) -> tuple[float, float]:
        cv = list(
            StratifiedKFold(PERMUTATION_N_SPLITS, shuffle=True, random_state=seed).split(X, lab)
        )
        clean = float(
            cross_val_score(svm_linear_pipeline(), X, lab, cv=cv, scoring="f1_macro").mean()
        )
        leaky = float(
            cross_val_score(
                svm_linear_pipeline(),
                apply_label_driven_zero_map(X, lab),
                lab,
                cv=cv,
                scoring="f1_macro",
            ).mean()
        )
        return clean, leaky

    obs_clean, obs_leaky = paired_scores(labels)
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_perm):
        perm_labels = rng.permutation(labels)
        c, l = paired_scores(perm_labels)
        rows.append({"perm": p, "clean": c, "leaky": l, "leaky_minus_clean": l - c})
    df = pd.DataFrame(rows)
    df.attrs["observed_clean"] = obs_clean
    df.attrs["observed_leaky"] = obs_leaky
    return df


def persistent_error_table(
    oof: dict[str, np.ndarray],
    y: np.ndarray,
    source_row_ids: np.ndarray,
    X: pd.DataFrame,
    *,
    n_repeats: int,
    threshold: float = PERSISTENT_ERROR_THRESHOLD,
) -> pd.DataFrame:
    rates = {}
    for k in CORE_MODELS:
        wrong = oof[k] != y.reshape(1, -1)
        rates[k] = wrong.mean(axis=0)
    pe = pd.DataFrame(rates)
    pe.insert(0, "source_row_id", source_row_ids)
    pe["true"] = y
    pe["persistent_error_all_models"] = pe[CORE_MODELS].ge(threshold).all(axis=1)
    grp = (X["blood_pressure"].to_numpy() == 3) & (y < 2)
    pe["bp3_label_lt2_posthoc"] = grp
    pe["n_valid_repeats"] = n_repeats
    return pe


PERSISTENT_DESC_N_SPLITS = 5


def describe_persistent_flags(
    X: pd.DataFrame,
    flag: np.ndarray,
    seed: int,
    *,
    groups: np.ndarray | None = None,
) -> dict[str, Any]:
    base = {
        "role": "same-data descriptive analysis, not a generalization or causal test",
        "n_flagged": int(np.asarray(flag).sum()),
        "n_total": int(len(flag)),
        "n_splits": PERSISTENT_DESC_N_SPLITS,
        "seed": seed,
    }
    if groups is not None:
        return {
            **base,
            "status": "not_available",
            "reason": (
                "grouped_duplicates: the auxiliary persistent-error tree is not given a "
                "validated group-aware protocol in this round and is not reported as "
                "ordinary StratifiedKFold(5)."
            ),
        }
    try:
        check_split_feasibility(
            np.asarray(flag, dtype=int),
            PERSISTENT_DESC_N_SPLITS,
            name="persistent_error_desc.n_splits",
        )
    except ConfigError as exc:
        return {**base, "status": "not_available", "reason": str(exc)}
    desc_cv = StratifiedKFold(PERSISTENT_DESC_N_SPLITS, shuffle=True, random_state=seed)
    pipe = make_pipeline(
        SimpleImputer(strategy="median"),
        DecisionTreeClassifier(max_depth=3, random_state=seed),
    )
    dp = cross_val_predict(pipe, X, flag, cv=desc_cv)
    pr, rc, f1, _ = precision_recall_fscore_support(flag, dp, labels=[1], zero_division=0)
    return {
        **base,
        "status": "completed",
        "splitter": "StratifiedKFold",
        "uses_groups": False,
        "tree_macro_f1": float(f1_score(flag, dp, average="macro")),
        "majority_baseline_macro_f1": float(f1_score(flag, np.zeros_like(flag), average="macro")),
        "flagged_class_precision": float(pr[0]),
        "flagged_class_recall": float(rc[0]),
        "confusion_matrix_[[TN,FP],[FN,TP]]": confusion_matrix(flag, dp).tolist(),
    }


def git_info(project_root: Path) -> dict[str, Any]:
    """
    Snapshot of HEAD and working-tree state. `dirty` counts modified tracked files
    and untracked (non-ignored) files. If git cannot be queried, commit/dirty are
    "unknown" and available=False; this is never reported as clean.
    """
    def _git(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        ).stdout

    try:
        commit = _git("rev-parse", "HEAD").strip()
        status = _git("status", "--porcelain", "--untracked-files=normal")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        detail = getattr(exc, "stderr", None) or str(exc)
        return {
            "commit": "unknown",
            "dirty": "unknown",
            "available": False,
            "error": str(detail).strip()[:500],
        }
    lines = [ln for ln in status.splitlines() if ln.strip()]
    return {
        "commit": commit,
        "dirty": bool(lines),
        "available": True,
        "n_status_entries": len(lines),
    }


def _fold_scores_for_csv(fs: pd.DataFrame) -> pd.DataFrame:
    out = fs.copy()
    if "confusion_matrix" in out.columns:
        out["confusion_matrix"] = out["confusion_matrix"].apply(
            lambda m: json.dumps(m) if not isinstance(m, str) else m
        )
    return out


def run_full_audit(
    csv_path: Path,
    out_dir: Path,
    *,
    scope: str = "primary",
    n_perm: int | None = None,
    smoke: bool = False,
    seed: int = 0,
    project_root: Path | None = None,
    tail_quarantine_from: int = 1100,
    near_dup_row_limit: int = 1100,
    sensitivity_policy: SensitivityPolicy = "raw",
    run_generalization: bool = True,
    n_splits: int | None = None,
    n_repeats: int | None = None,
    settings: RunSettings | None = None,
    settings_resolution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # Capture git state before creating or writing any output, so artifacts written
    # under the repository do not make a clean starting tree look dirty.
    project_root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
    git_start = git_info(project_root)
    git_start["git_state_capture"] = "run_start"

    if settings is None:
        settings = resolve_run_settings(
            persistent_error_threshold=PERSISTENT_ERROR_THRESHOLD,
            cli_scope=scope,
            cli_sensitivity_policy=sensitivity_policy,
            cli_smoke=smoke,
            cli_seed=seed,
            cli_n_perm=n_perm,
            cli_n_splits=n_splits,
            cli_n_repeats=n_repeats,
            cli_tail_quarantine_from=tail_quarantine_from,
            cli_near_dup_row_limit=near_dup_row_limit,
            cli_run_generalization=run_generalization,
        )
    resolution = dict(settings_resolution or {})
    resolution.setdefault("eval_sources", {
        "n_splits": settings.sources["n_splits"],
        "n_repeats": settings.sources["n_repeats"],
    })
    resolution.setdefault("sources", settings.sources)
    resolution.setdefault("eval_notes", settings.notes)
    cfg = EvalConfig(
        n_repeats=settings.n_repeats,
        n_splits=settings.n_splits,
        seed=settings.seed,
    )
    scope = settings.scope
    sensitivity_policy = settings.sensitivity_policy  # type: ignore[assignment]
    smoke = settings.smoke
    n_perm = settings.n_perm
    tail_quarantine_from = settings.tail_quarantine_from
    near_dup_row_limit = settings.near_dup_row_limit
    run_generalization = settings.run_generalization

    csv_path = Path(csv_path).resolve()
    raw = load_raw_csv(csv_path)
    parsed, parse_issues = parse_dataset(raw)
    prep = prepare_data(
        csv_path,
        scope=scope,  # type: ignore[arg-type]
        tail_quarantine_from=tail_quarantine_from,
        sensitivity_policy=sensitivity_policy,
        raw=raw,
        parsed=parsed,
        parse_issues=parse_issues,
    )
    if len(prep.y) == 0:
        raise ConfigError("included sample is empty after exclusions")

    sensitivity_notes: dict[str, Any] = {}
    groups = None
    if scope == "sensitivity":
        sensitivity_notes = {
            "policy": sensitivity_policy,
            "n_out_of_range_cells": int(len(prep.out_of_range)),
            "n_duplicate_groups_with_size_gt1": int(
                pd.Series(prep.duplicate_group_ids).value_counts().gt(1).sum()
            ),
            "provisional_bounds_note": "Bounds are provisional (not a confirmed data dictionary).",
        }
        if sensitivity_policy == "grouped_duplicates":
            groups = prep.duplicate_group_ids
            sensitivity_notes["split"] = "StratifiedGroupKFold on duplicate_group_id"
        elif sensitivity_policy == "raw" and run_generalization:
            # Identical-feature duplicates may cross folds; refuse CV unless policy set.
            vc = pd.Series(prep.duplicate_group_ids).value_counts()
            if (vc > 1).any():
                sensitivity_notes["generalization"] = (
                    "refused: identical-feature duplicates present; "
                    "use --sensitivity-policy grouped_duplicates or quarantine_out_of_range"
                )
                run_generalization = False

    grouped = bool(groups is not None)
    X = prep.X
    y = prep.y
    models = audit_model_registry(prep.feature_names, random_state=cfg.seed)

    outer: list = []
    perm_protocol = permutation_protocol(
        seed=cfg.seed,
        n_perm=n_perm,
        grouped=grouped,
        run_generalization=run_generalization,
    )
    if run_generalization:
        outer = plan_outer_splits(y, cfg, groups=groups)
        if perm_protocol["status"] == "completed":
            check_split_feasibility(y, PERMUTATION_N_SPLITS, name="permutation.n_splits")

    out_dir = ensure_empty_output_dir(out_dir)
    write_run_status(
        out_dir,
        "in_progress",
        git=git_start,
        note="Partial files here are not a completed audit. Rerun into a new directory.",
    )

    dupes = duplicate_audit(
        raw, parsed, near_dup_row_limit=near_dup_row_limit, include_near=not smoke
    )

    cv_out: dict[str, Any] | None = None
    if run_generalization:
        cv_out = run_audit_cv(
            X, y, prep.source_row_ids, models, cfg, groups=groups, outer=outer
        )

    prep.exclusions.to_csv(out_dir / "exclusions.csv", index=False, encoding="utf-8")
    prep.parse_issues.to_csv(out_dir / "parse_issues.csv", index=False, encoding="utf-8")
    prep.field_checks.to_csv(out_dir / "field_checks.csv", index=False, encoding="utf-8")
    prep.out_of_range.to_csv(out_dir / "out_of_range.csv", index=False, encoding="utf-8")
    prep.domain_violations.to_csv(out_dir / "domain_violations.csv", index=False, encoding="utf-8")
    pd.DataFrame(
        {
            "source_row_id": prep.source_row_ids,
            "duplicate_group_id": prep.duplicate_group_ids,
            "true": y,
        }
    ).to_csv(out_dir / "duplicate_group_ids.csv", index=False, encoding="utf-8")

    for name, df in dupes.items():
        df.to_csv(out_dir / f"duplicate_audit_{name}.csv", index=False, encoding="utf-8")

    if cv_out is not None:
        cv_out["folds"].to_csv(out_dir / "folds.csv", index=False, encoding="utf-8")
        _fold_scores_for_csv(cv_out["fold_scores"]).to_csv(
            out_dir / "fold_scores.csv", index=False, encoding="utf-8"
        )
        _fold_scores_for_csv(cv_out["repeat_scores"]).to_csv(
            out_dir / "repeat_scores.csv", index=False, encoding="utf-8"
        )
        cv_out["inner_candidates"].to_csv(
            out_dir / "inner_candidates.csv", index=False, encoding="utf-8"
        )
        if len(cv_out["inner_folds"]):
            cv_out["inner_folds"].to_csv(
                out_dir / "inner_folds.csv", index=False, encoding="utf-8"
            )
        oof_rows = []
        for rep in range(cfg.n_repeats):
            for i in range(len(y)):
                row = {
                    "repeat": rep,
                    "source_row_id": int(prep.source_row_ids[i]),
                    "true": int(y[i]),
                }
                for k, arr in cv_out["oof"].items():
                    row[k] = int(arr[rep, i])
                oof_rows.append(row)
        pd.DataFrame(oof_rows).to_csv(
            out_dir / "oof_predictions.csv", index=False, encoding="utf-8"
        )

        if perm_protocol["status"] == "completed":
            perm = run_permutations(X, y, n_perm=n_perm, seed=cfg.seed)
            perm.to_csv(out_dir / "permutations.csv", index=False, encoding="utf-8")
        else:
            perm = pd.DataFrame()

        pe = persistent_error_table(
            cv_out["oof"], y, prep.source_row_ids, X, n_repeats=cfg.n_repeats
        )
        pe.to_csv(out_dir / "per_sample_errors.csv", index=False, encoding="utf-8")
        flag = pe["persistent_error_all_models"].to_numpy().astype(int)
        desc = describe_persistent_flags(X, flag, cfg.seed, groups=groups)

        paired = {
            "lr_all-nested_single": paired_comparison(
                cv_out["fold_scores"], "lr_all", "nested_single", outer
            ),
            "lr_all-rf_all": paired_comparison(cv_out["fold_scores"], "lr_all", "rf_all", outer),
            "lr_all-lr_no_psych": paired_comparison(
                cv_out["fold_scores"], "lr_all", "lr_no_psych", outer
            ),
        }

        exploratory: dict[str, Any] = {}
        if not smoke:
            for c in X.columns:
                scores = []
                for tr, te in outer:
                    m = tree_depth3_pipeline(random_state=cfg.seed).fit(
                        X.iloc[tr][[c]], y[tr]
                    )
                    scores.append(
                        f1_score(y[te], m.predict(X.iloc[te][[c]]), average="macro")
                    )
                exploratory[c] = float(np.mean(scores))
    else:
        perm = pd.DataFrame()
        desc = {
            "status": "not_available",
            "reason": "generalization skipped",
            "role": "same-data descriptive analysis, not a generalization or causal test",
        }
        paired = {}
        exploratory = {"note": "generalization skipped"}
        flag = np.array([])

    lock_text = write_environment_lock(out_dir / "requirements-lock.txt")
    lock_sha = sha256_bytes(lock_text.encode("utf-8"))
    manifest_sha, manifest_entries = code_manifest_sha256(project_root)

    def summ(s: pd.Series) -> dict[str, float]:
        return {
            "mean": float(s.mean()),
            "sd": float(s.std(ddof=1)),
            "q05": float(s.quantile(0.05)),
            "q95": float(s.quantile(0.95)),
            "min": float(s.min()),
            "max": float(s.max()),
        }

    results: dict[str, Any] = {
        "run_mode": "smoke" if smoke else "full",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_start,
        "config": {
            **settings.as_config_block(),
            "eval": cfg.__dict__,
            "run_generalization": run_generalization,
            "settings_resolution": resolution,
            "persistent_error_threshold": PERSISTENT_ERROR_THRESHOLD,
            "quarantine_out_of_range_semantics": (
                "legacy/provisional: excludes rows with any feature above the historical "
                "PROVISIONAL_MAX / default max 5; does not apply below_min, non_integer, "
                "invalid_category or non_finite rules"
            ),
        },
        "domain_violations": {
            "schema_status": "provisional",
            "note": (
                "Reporting only; does not change primary inclusion or model inputs. "
                "A violation is not a confirmed data-entry error or mislabel."
            ),
            **prep.domain_violation_summary,
        },
        "env": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "csv_sha256_raw": prep.csv_sha256_raw,
            "csv_sha256_normalized_lf": prep.csv_sha256_normalized_lf,
            "environment_lock_sha256": lock_sha,
            "code_manifest_sha256": manifest_sha,
            "code_manifest_entries": manifest_entries,
        },
        "sample": {
            "rows_used": int(len(y)),
            "source_row_id_range": [
                int(prep.source_row_ids.min()),
                int(prep.source_row_ids.max()),
            ],
            "rule": "primary: rows 0-1099 + valid labels; sensitivity: policy-dependent",
        },
        "sensitivity": sensitivity_notes,
        "models_aggregation": FOLD_AGGREGATION,
        "models": cv_out["summary"] if cv_out else {},
        "models_repeat_aggregation": REPEAT_AGGREGATION,
        "models_repeat": cv_out["repeat_summary"] if cv_out else {},
        "model_params": cv_out["model_params"] if cv_out else {},
        "nested_single_feature_picks": (
            cv_out["fold_scores"]
            .loc[cv_out["fold_scores"].model == "nested_single", "selected_feature"]
            .value_counts()
            .to_dict()
            if cv_out
            else {}
        ),
        "paired": paired,
        "permutation": (
            {
                **perm_protocol,
                "observed_clean": perm.attrs.get("observed_clean"),
                "observed_leaky": perm.attrs.get("observed_leaky"),
                "null_clean": summ(perm["clean"]),
                "null_leaky": summ(perm["leaky"]),
                "null_leaky_minus_clean": summ(perm["leaky_minus_clean"]),
                "share_perms_leaky_gt_clean": float((perm["leaky_minus_clean"] > 0).mean()),
            }
            if perm_protocol["status"] == "completed" and len(perm)
            else perm_protocol
        ),
        "persistent_errors": (
            {
                "definition": (
                    f"error rate >= {PERSISTENT_ERROR_THRESHOLD} over repeats in each of "
                    + ", ".join(CORE_MODELS)
                ),
                "feature_description_same_data": desc,
                "n_flagged": int(flag.sum()) if len(flag) else 0,
            }
            if cv_out
            else {"note": "skipped"}
        ),
        "single_feature_exploratory": (
            dict(sorted(exploratory.items(), key=lambda kv: -kv[1]))
            if isinstance(exploratory, dict) and exploratory and "note" not in exploratory
            else exploratory
        ),
    }

    artifact_warning = None
    try:
        import joblib

        full_lr = lr_pipeline(random_state=cfg.seed)
        full_lr.fit(X, y)
        artifact_name = f"lr_all_{scope}_fitted.joblib"
        joblib.dump(
            {
                "pipeline": full_lr,
                "feature_names": list(X.columns),
                "label_classes": [0, 1, 2],
                "scope": scope,
                "csv_sha256_raw": prep.csv_sha256_raw,
            },
            out_dir / artifact_name,
        )
        results["model_artifact"] = artifact_name
    except Exception as exc:
        artifact_warning = f"Could not save joblib pipeline: {exc}"
        results["artifact_warning"] = artifact_warning

    (out_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_run_status(out_dir, "completed", results_json="results.json")
    return results
