"""Cross-validation evaluation, metrics, and paired comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold

from stress_detection.models import NestedBestSingleFeature, ModelFactory, estimator_params
from stress_detection.run_config import ConfigError


@dataclass
class EvalConfig:
    n_splits: int = 5
    n_repeats: int = 10
    seed: int = 0
    scoring: str = "f1_macro"


def outer_splits(
    n_samples: int,
    y: np.ndarray,
    cfg: EvalConfig,
    *,
    groups: np.ndarray | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    X_dummy = np.zeros((n_samples, 1))
    if groups is not None:
        # Group-aware: StratifiedGroupKFold does not support n_repeats; emulate repeats
        splits: list[tuple[np.ndarray, np.ndarray]] = []
        for rep in range(cfg.n_repeats):
            sgkf = StratifiedGroupKFold(
                n_splits=cfg.n_splits,
                shuffle=True,
                random_state=cfg.seed + rep,
            )
            for tr, te in sgkf.split(X_dummy, y, groups):
                splits.append((tr, te))
        return splits
    rskf = RepeatedStratifiedKFold(
        n_splits=cfg.n_splits,
        n_repeats=cfg.n_repeats,
        random_state=cfg.seed,
    )
    return list(rskf.split(X_dummy, y))


def check_split_feasibility(
    y: np.ndarray,
    n_splits: int,
    *,
    groups: np.ndarray | None = None,
    name: str = "n_splits",
) -> None:
    """Raise ConfigError if a stratified (group) split cannot be built."""
    y = np.asarray(y)
    if len(y) == 0:
        raise ConfigError("included sample is empty")
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        raise ConfigError(
            f"{name}={n_splits} not feasible: included sample has {len(classes)} class(es) "
            f"(values={classes.tolist()})"
        )
    min_class = int(counts.min())
    if n_splits > min_class:
        raise ConfigError(
            f"{name}={n_splits} exceeds the smallest class count ({min_class}); "
            f"class counts={dict(zip(classes.tolist(), counts.tolist()))}"
        )
    if groups is not None:
        n_groups = int(len(np.unique(groups)))
        if n_splits > n_groups:
            raise ConfigError(
                f"{name}={n_splits} exceeds n_groups={n_groups}; "
                "refusing to fall back to ungrouped CV"
            )


def validate_generated_splits(
    splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    groups: np.ndarray | None,
    name: str,
) -> None:
    if not splits:
        raise ConfigError(f"{name}: no splits were generated")
    for i, (tr, te) in enumerate(splits):
        if len(tr) == 0 or len(te) == 0:
            raise ConfigError(f"{name} fold {i}: empty train or valid set")
        if groups is not None:
            overlap = set(groups[tr]) & set(groups[te])
            if overlap:
                raise ConfigError(
                    f"{name} fold {i}: train/valid groups overlap ({sorted(overlap)[:8]})"
                )


def plan_outer_splits(
    y: np.ndarray,
    cfg: EvalConfig,
    *,
    groups: np.ndarray | None = None,
    inner_splits: int = 3,
) -> list[tuple[np.ndarray, np.ndarray]]:
    check_split_feasibility(y, cfg.n_splits, groups=groups, name="n_splits")
    outer = outer_splits(len(y), y, cfg, groups=groups)
    validate_generated_splits(outer, groups=groups, name="outer CV")
    for i, (tr, _) in enumerate(outer):
        gtr = groups[tr] if groups is not None else None
        check_split_feasibility(y[tr], inner_splits, groups=gtr, name="nested_single.inner_splits")
    return outer


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    labels = [0, 1, 2]
    p, r, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    out: dict[str, Any] = {}
    for i, lab in enumerate(labels):
        out[str(lab)] = {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f1[i]),
            "support": int(sup[i]),
        }
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["macro_f1"] = float(f1_score(y_true, y_pred, average="macro"))
    out["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    cm = np.array(out["confusion_matrix"])
    denom = int((y_true == 0).sum() + (y_true == 2).sum())
    cross_02 = int(cm[0, 2] + cm[2, 0]) if cm.shape == (3, 3) else 0
    out["cross_level_0_2"] = {
        "numerator": cross_02,
        "denominator": denom,
        "rate": float(cross_02 / denom) if denom else float("nan"),
    }
    return out


def _fold_row_from_metrics(
    rep: int,
    fold: int,
    name: str,
    metrics: dict[str, Any],
    selected_feature: str = "",
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "repeat": rep,
        "fold": fold,
        "model": name,
        "macro_f1": metrics["macro_f1"],
        "accuracy": metrics["accuracy"],
        "selected_feature": selected_feature,
        "cross_level_0_2_numerator": metrics["cross_level_0_2"]["numerator"],
        "cross_level_0_2_denominator": metrics["cross_level_0_2"]["denominator"],
        "cross_level_0_2_rate": metrics["cross_level_0_2"]["rate"],
        "confusion_matrix": metrics["confusion_matrix"],
    }
    for lab in ("0", "1", "2"):
        for k in ("precision", "recall", "f1", "support"):
            row[f"class_{lab}_{k}"] = metrics[lab][k]
    return row


SUMMARY_METRIC_COLS = [
    "macro_f1",
    "accuracy",
    "cross_level_0_2_rate",
    "class_0_precision",
    "class_0_recall",
    "class_0_f1",
    "class_1_precision",
    "class_1_recall",
    "class_1_f1",
    "class_2_precision",
    "class_2_recall",
    "class_2_f1",
]

FOLD_AGGREGATION = {
    "unit": "outer_fold",
    "definition": (
        "Each metric is computed on one outer validation fold; *_mean / *_std are "
        "taken over all outer folds (n_splits x n_repeats), std with ddof=1. "
        "Fold std describes fold-to-fold score variation on this fixed dataset."
    ),
    "confusion_matrix_sum": (
        "Sum of per-fold confusion matrices over all folds of all repeats; each "
        "included sample is counted once per repeat (n_repeats times in total), "
        "so the total is not a count of independent samples."
    ),
}

REPEAT_AGGREGATION = {
    "unit": "repeat",
    "definition": (
        "For each repeat, predictions from all outer validation folds are merged by "
        "source_row_id into one complete out-of-fold (OOF) vector covering every "
        "included sample exactly once; metrics (including macro-F1) are recomputed "
        "from that pooled vector. *.mean / *.std are taken over repeats, std with ddof=1."
    ),
    "interpretation": (
        "Repeat std describes variation due to the random partition on the same "
        "fixed dataset. Repeats reuse the same data and are not independent samples "
        "of new datasets; neither repeat std nor fold std is a confidence interval "
        "for external generalization."
    ),
    "confusion_matrix_sum_over_repeats": (
        "Sum of per-repeat OOF confusion matrices; each included sample is counted "
        "n_repeats times, so the total is not a count of independent samples."
    ),
}


def summarize_fold_scores(fs: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name, g in fs.groupby("model"):
        entry: dict[str, Any] = {"aggregation_unit": "outer_fold", "n_folds": int(len(g))}
        for col in SUMMARY_METRIC_COLS:
            if col not in g.columns:
                continue
            s = g[col].astype(float)
            entry[f"{col}_mean"] = float(s.mean())
            entry[f"{col}_std"] = float(s.std(ddof=1)) if len(s) > 1 else 0.0
        cms = [np.array(m) for m in g["confusion_matrix"].tolist()]
        entry["confusion_matrix_sum"] = np.sum(cms, axis=0).astype(int).tolist()
        entry["confusion_matrix_sum_unit"] = (
            "sum over all outer folds; each sample counted once per repeat"
        )
        entry["confusion_matrix_per_fold"] = g["confusion_matrix"].tolist()
        summary[name] = entry
    return summary


def repeat_oof_scores(
    oof: dict[str, np.ndarray],
    y: np.ndarray,
) -> pd.DataFrame:
    """One row per model x repeat, computed from the pooled OOF predictions of that repeat."""
    rows: list[dict[str, Any]] = []
    for name, arr in oof.items():
        for rep in range(arr.shape[0]):
            pred = arr[rep]
            if (pred < 0).any():
                missing = int((pred < 0).sum())
                raise ValueError(
                    f"{name} repeat {rep}: {missing} samples have no OOF prediction"
                )
            metrics = per_class_metrics(y, pred)
            row: dict[str, Any] = {
                "model": name,
                "repeat": rep,
                "n_samples": int(len(y)),
                "macro_f1": metrics["macro_f1"],
                "accuracy": metrics["accuracy"],
                "cross_level_0_2_numerator": metrics["cross_level_0_2"]["numerator"],
                "cross_level_0_2_denominator": metrics["cross_level_0_2"]["denominator"],
                "cross_level_0_2_rate": metrics["cross_level_0_2"]["rate"],
                "confusion_matrix": metrics["confusion_matrix"],
            }
            for lab in ("0", "1", "2"):
                for k in ("precision", "recall", "f1", "support"):
                    row[f"class_{lab}_{k}"] = metrics[lab][k]
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_repeat_scores(rs: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name, g in rs.groupby("model", sort=False):
        g = g.sort_values("repeat")
        n = int(len(g))
        metrics: dict[str, Any] = {}
        for col in SUMMARY_METRIC_COLS:
            s = g[col].astype(float)
            metrics[col] = {
                "mean": float(s.mean()),
                "std": float(s.std(ddof=1)) if n > 1 else None,
            }
        cms = [np.array(m) for m in g["confusion_matrix"].tolist()]
        entry: dict[str, Any] = {
            "aggregation_unit": "repeat",
            "n_repeats": n,
            "metrics": metrics,
            "confusion_matrix_label_order": [0, 1, 2],
            "confusion_matrix_per_repeat": g["confusion_matrix"].tolist(),
            "confusion_matrix_sum_over_repeats": np.sum(cms, axis=0).astype(int).tolist(),
            "confusion_matrix_sum_unit": (
                f"sum over {n} repeats; each sample counted {n} times"
            ),
        }
        if n <= 1:
            entry["std_note"] = (
                "Only one repeat: across-repeat standard deviation cannot be estimated."
            )
        summary[name] = entry
    return summary


def _fit_estimator(est: Any, X_tr: pd.DataFrame, y_tr: np.ndarray, groups_tr: np.ndarray | None):
    """Pass groups only to NestedBestSingleFeature; never to generic Pipelines."""
    if isinstance(est, NestedBestSingleFeature) and groups_tr is not None:
        return est.fit(X_tr, y_tr, groups=groups_tr)
    return est.fit(X_tr, y_tr)


def run_audit_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    source_row_ids: np.ndarray,
    models: dict[str, tuple[ModelFactory, list[str]]],
    cfg: EvalConfig,
    *,
    groups: np.ndarray | None = None,
    outer: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict[str, Any]:
    if outer is None:
        outer = plan_outer_splits(y, cfg, groups=groups)
    fold_rows: list[dict] = []
    inner_rows: list[dict] = []
    inner_fold_rows: list[dict] = []
    n_folds = len(outer)
    oof = {k: np.full((cfg.n_repeats, len(y)), -1, dtype=int) for k in models}
    fold_membership: list[dict] = []
    model_params: dict[str, Any] = {}

    for f_idx, (tr, te) in enumerate(outer):
        rep, fold = divmod(f_idx, cfg.n_splits)
        for i in tr:
            row = {
                "repeat": rep,
                "fold": fold,
                "source_row_id": int(source_row_ids[i]),
                "role": "train",
            }
            if groups is not None:
                row["group_id"] = int(groups[i])
            fold_membership.append(row)
        for i in te:
            row = {
                "repeat": rep,
                "fold": fold,
                "source_row_id": int(source_row_ids[i]),
                "role": "valid",
            }
            if groups is not None:
                row["group_id"] = int(groups[i])
            fold_membership.append(row)

        groups_tr = groups[tr] if groups is not None else None
        for name, (factory, cols) in models.items():
            est = factory()
            if name not in model_params:
                model_params[name] = estimator_params(est)
            _fit_estimator(est, X.iloc[tr][cols], y[tr], groups_tr)
            pred = est.predict(X.iloc[te][cols])
            oof[name][rep, te] = pred
            metrics = per_class_metrics(y[te], pred)
            fold_rows.append(
                _fold_row_from_metrics(
                    rep,
                    fold,
                    name,
                    metrics,
                    selected_feature=getattr(est, "feature_", ""),
                )
            )
            if name == "nested_single" and hasattr(est, "candidates_"):
                for feat, score in est.candidates_.items():
                    inner_rows.append(
                        {
                            "repeat": rep,
                            "fold": fold,
                            "feature": feat,
                            "inner_macro_f1": score,
                        }
                    )
                if hasattr(est, "inner_splits_used_"):
                    for inner_k, (itr, ite) in enumerate(est.inner_splits_used_):
                        for local_i, role in (
                            *((int(j), "train") for j in itr),
                            *((int(j), "valid") for j in ite),
                        ):
                            global_i = int(tr[local_i])
                            inner_row = {
                                "repeat": rep,
                                "fold": fold,
                                "inner_fold": inner_k,
                                "source_row_id": int(source_row_ids[global_i]),
                                "role": role,
                            }
                            if groups is not None:
                                inner_row["group_id"] = int(groups[global_i])
                            inner_fold_rows.append(inner_row)

    fs = pd.DataFrame(fold_rows)
    rs = repeat_oof_scores(oof, y)
    return {
        "fold_scores": fs,
        "repeat_scores": rs,
        "repeat_summary": summarize_repeat_scores(rs),
        "inner_candidates": pd.DataFrame(inner_rows),
        "inner_folds": pd.DataFrame(inner_fold_rows),
        "folds": pd.DataFrame(fold_membership),
        "oof": oof,
        "summary": summarize_fold_scores(fs),
        "n_outer_folds": n_folds,
        "outer": outer,
        "model_params": model_params,
    }


def paired_comparison(
    fold_scores: pd.DataFrame,
    model_a: str,
    model_b: str,
    outer: list[tuple[np.ndarray, np.ndarray]],
    *,
    direction: str = "a_minus_b",
) -> dict[str, Any]:
    """Nadeau–Bengio style corrected SE; approximate interval (df = n_folds - 1)."""
    a = fold_scores.loc[fold_scores.model == model_a, "macro_f1"].to_numpy()
    b = fold_scores.loc[fold_scores.model == model_b, "macro_f1"].to_numpy()
    if len(a) != len(b):
        raise ValueError("Models must share the same outer folds")
    d = a - b if direction == "a_minus_b" else b - a
    n_te = np.mean([len(te) for _, te in outer])
    n_tr = np.mean([len(tr) for tr, _ in outer])
    se = np.sqrt((1 / len(d) + n_te / n_tr) * d.var(ddof=1))
    df = len(d) - 1
    t_crit = float(stats.t.ppf(0.975, df=df))
    return {
        "model_a": model_a,
        "model_b": model_b,
        "mean_diff": float(d.mean()),
        "fold_diffs": d.tolist(),
        "corrected_se": float(se),
        "df": df,
        "t_crit": t_crit,
        "approx_ci95": [float(d.mean() - t_crit * se), float(d.mean() + t_crit * se)],
        "share_of_folds_a_higher_descriptive": float((d > 0).mean()),
        "interpretation": (
            "Approximate corrected interval over repeated-CV folds (folds are not "
            "independent). An interval containing 0 does not establish equivalence."
        ),
    }


def metrics_from_oof(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Recompute accuracy / macro-F1 from a single OOF vector."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }
