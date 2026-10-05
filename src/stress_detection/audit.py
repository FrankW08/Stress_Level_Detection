"""Full audit pipeline: duplicates, permutations, persistent errors, artifacts."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
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
    PreparedData,
    apply_label_driven_zero_map,
    load_raw_csv,
    parse_dataset,
    prepare_data,
)
from stress_detection.evaluation import EvalConfig, paired_comparison, run_audit_cv
from stress_detection.models import audit_model_registry, svm_linear_pipeline, tree_depth3_pipeline

CORE_MODELS = ["lr_all", "rf_all", "svm_all", "nested_single"]
PERSISTENT_ERROR_THRESHOLD = 0.8


def duplicate_audit(raw: pd.DataFrame, parsed: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Exact dupes, normalized dupes, same features different label, near-duplicate candidates."""
    feature_cols = [c for c in parsed.columns if c not in ("source_row_id", "stress_level")]
    exact = raw.duplicated(keep=False)
    exact_df = pd.DataFrame(
        {"source_row_id": parsed.loc[exact, "source_row_id"].astype(int).tolist()}
    )
    norm_key = parsed[feature_cols + ["stress_level"]].astype("string")
    norm_dup = norm_key.duplicated(keep=False)
    norm_df = parsed.loc[norm_dup, ["source_row_id"] + feature_cols + ["stress_level"]]

    feat_only = parsed[feature_cols]
    label_conflict = feat_only.duplicated(keep=False) & ~parsed.duplicated(
        subset=feature_cols + ["stress_level"], keep=False
    )
    conflict_df = parsed.loc[label_conflict, ["source_row_id"] + feature_cols + ["stress_level"]]

    # Near-duplicate candidates: Hamming distance 1 on feature columns (primary rows only).
    near_rows: list[dict] = []
    primary = parsed[parsed["source_row_id"] < 1100] if "source_row_id" in parsed.columns else parsed.iloc[:1100]
    arr = primary[feature_cols].to_numpy()
    ids = primary["source_row_id"].astype(int).to_numpy()
    n = len(primary)
    for i in range(n):
        for j in range(i + 1, n):
            ne = arr[i] != arr[j]
            if ne.sum() == 1:
                near_rows.append(
                    {
                        "source_row_id_a": int(ids[i]),
                        "source_row_id_b": int(ids[j]),
                        "n_feature_diffs": 1,
                    }
                )
    near_df = pd.DataFrame(near_rows)
    return {
        "exact_duplicate_rows": exact_df,
        "normalized_duplicate_rows": norm_df,
        "same_features_different_label": conflict_df,
        "near_duplicate_candidates": near_df,
    }


def run_permutations(
    X: pd.DataFrame,
    labels: np.ndarray,
    *,
    n_perm: int,
    seed: int,
) -> pd.DataFrame:
    def paired_scores(lab: np.ndarray) -> tuple[float, float]:
        cv = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(X, lab))
        clean = float(cross_val_score(svm_linear_pipeline(), X, lab, cv=cv, scoring="f1_macro").mean())
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


def describe_persistent_flags(X: pd.DataFrame, flag: np.ndarray, seed: int) -> dict[str, Any]:
    desc_cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    pipe = make_pipeline(
        SimpleImputer(strategy="median"),
        DecisionTreeClassifier(max_depth=3, random_state=seed),
    )
    dp = cross_val_predict(pipe, X, flag, cv=desc_cv)
    pr, rc, f1, _ = precision_recall_fscore_support(flag, dp, labels=[1], zero_division=0)
    return {
        "n_flagged": int(flag.sum()),
        "n_total": int(len(flag)),
        "tree_macro_f1": float(f1_score(flag, dp, average="macro")),
        "majority_baseline_macro_f1": float(f1_score(flag, np.zeros_like(flag), average="macro")),
        "flagged_class_precision": float(pr[0]),
        "flagged_class_recall": float(rc[0]),
        "confusion_matrix_[[TN,FP],[FN,TP]]": confusion_matrix(flag, dp).tolist(),
    }


def git_info(project_root: Path) -> dict[str, str]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return {"commit": commit, "dirty": bool(dirty)}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": "unknown", "dirty": "unknown"}


def write_requirements_lock(out_dir: Path) -> str:
    lock = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True,
        text=True,
    ).stdout
    (out_dir / "requirements-lock.txt").write_text(lock, encoding="utf-8")
    return lock


def run_full_audit(
    csv_path: Path,
    out_dir: Path,
    *,
    scope: str = "primary",
    n_perm: int = 100,
    smoke: bool = False,
    project_root: Path | None = None,
) -> dict[str, Any]:
    import hashlib

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    project_root = project_root or Path(__file__).resolve().parents[2]

    cfg = EvalConfig(
        n_repeats=1 if smoke else 10,
        n_splits=3 if smoke else 5,
    )
    n_perm = 5 if smoke else n_perm

    prep = prepare_data(csv_path, scope=scope)  # type: ignore[arg-type]
    raw = load_raw_csv(csv_path)
    parsed, _ = parse_dataset(raw)
    dupes = duplicate_audit(raw, parsed) if not smoke else {
        k: pd.DataFrame() for k in (
            "exact_duplicate_rows",
            "normalized_duplicate_rows",
            "same_features_different_label",
            "near_duplicate_candidates",
        )
    }
    if smoke:
        dupes["near_duplicate_candidates"] = pd.DataFrame(
            [{"note": "skipped in smoke mode; run full audit for duplicate/near-duplicate tables"}]
        )

    X = prep.X
    y = prep.y
    models = audit_model_registry(prep.feature_names)
    cv_out = run_audit_cv(X, y, prep.source_row_ids, models, cfg)
    outer = __import__("stress_detection.evaluation", fromlist=["outer_splits"]).outer_splits(
        len(y), y, cfg
    )

    prep.exclusions.to_csv(out_dir / "exclusions.csv", index=False, encoding="utf-8")
    prep.parse_issues.to_csv(out_dir / "parse_issues.csv", index=False, encoding="utf-8")
    prep.field_checks.to_csv(out_dir / "field_checks.csv", index=False, encoding="utf-8")
    for name, df in dupes.items():
        df.to_csv(out_dir / f"duplicate_audit_{name}.csv", index=False, encoding="utf-8")

    cv_out["folds"].to_csv(out_dir / "folds.csv", index=False, encoding="utf-8")
    cv_out["fold_scores"].to_csv(out_dir / "fold_scores.csv", index=False, encoding="utf-8")
    cv_out["inner_candidates"].to_csv(out_dir / "inner_candidates.csv", index=False, encoding="utf-8")

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
    pd.DataFrame(oof_rows).to_csv(out_dir / "oof_predictions.csv", index=False, encoding="utf-8")

    perm = run_permutations(X, y, n_perm=n_perm, seed=cfg.seed)
    perm.to_csv(out_dir / "permutations.csv", index=False, encoding="utf-8")

    pe = persistent_error_table(
        cv_out["oof"], y, prep.source_row_ids, X, n_repeats=cfg.n_repeats
    )
    pe.to_csv(out_dir / "per_sample_errors.csv", index=False, encoding="utf-8")
    flag = pe["persistent_error_all_models"].to_numpy().astype(int)
    desc = describe_persistent_flags(X, flag, cfg.seed)

    lock = write_requirements_lock(out_dir)
    script_path = project_root / "scripts" / "run_audit.py"
    script_sha = hashlib.sha256(script_path.read_bytes()).hexdigest() if script_path.is_file() else ""

    paired = {
        "lr_all-nested_single": paired_comparison(
            cv_out["fold_scores"], "lr_all", "nested_single", outer
        ),
        "lr_all-rf_all": paired_comparison(cv_out["fold_scores"], "lr_all", "rf_all", outer),
        "lr_all-lr_no_psych": paired_comparison(
            cv_out["fold_scores"], "lr_all", "lr_no_psych", outer
        ),
    }

    exploratory = {}
    if not smoke:
        for c in X.columns:
            scores = []
            for tr, te in outer:
                m = tree_depth3_pipeline().fit(X.iloc[tr][[c]], y[tr])
                scores.append(f1_score(y[te], m.predict(X.iloc[te][[c]]), average="macro"))
            exploratory[c] = float(np.mean(scores))

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
        "git": git_info(project_root),
        "config": {
            "scope": scope,
            "eval": cfg.__dict__,
            "n_perm": n_perm,
            "persistent_error_threshold": PERSISTENT_ERROR_THRESHOLD,
        },
        "env": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "csv_sha256": prep.csv_sha256,
            "script_sha256": script_sha,
        },
        "sample": {
            "rows_used": int(len(y)),
            "source_row_id_range": [int(prep.source_row_ids.min()), int(prep.source_row_ids.max())],
            "rule": "primary: rows 0-1099 + valid labels; sensitivity: all rows + valid labels",
        },
        "models": cv_out["summary"],
        "nested_single_feature_picks": cv_out["fold_scores"]
        .loc[cv_out["fold_scores"].model == "nested_single", "selected_feature"]
        .value_counts()
        .to_dict(),
        "paired": paired,
        "permutation": {
            "observed_clean": perm.attrs.get("observed_clean"),
            "observed_leaky": perm.attrs.get("observed_leaky"),
            "null_clean": summ(perm["clean"]),
            "null_leaky": summ(perm["leaky"]),
            "null_leaky_minus_clean": summ(perm["leaky_minus_clean"]),
            "share_perms_leaky_gt_clean": float((perm["leaky_minus_clean"] > 0).mean()),
        },
        "persistent_errors": {
            "definition": f"error rate >= {PERSISTENT_ERROR_THRESHOLD} over repeats in each of "
            + ", ".join(CORE_MODELS),
            "feature_description_same_data": desc,
            "n_flagged": int(flag.sum()),
        },
        "single_feature_exploratory": dict(sorted(exploratory.items(), key=lambda kv: -kv[1]))
        if exploratory
        else {"note": "skipped in smoke mode"},
    }
    (out_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    try:
        import joblib

        from stress_detection.models import lr_pipeline

        full_lr = lr_pipeline()
        full_lr.fit(X, y)
        joblib.dump(
            {
                "pipeline": full_lr,
                "feature_names": list(X.columns),
                "label_classes": [0, 1, 2],
                "scope": scope,
                "csv_sha256": prep.csv_sha256,
            },
            out_dir / "lr_all_primary_fitted.joblib",
        )
    except Exception as exc:  # pragma: no cover
        results["artifact_warning"] = f"Could not save joblib pipeline: {exc}"

    return results
