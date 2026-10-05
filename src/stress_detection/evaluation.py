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
from sklearn.model_selection import RepeatedStratifiedKFold

from stress_detection.models import ModelFactory


@dataclass
class EvalConfig:
    n_splits: int = 5
    n_repeats: int = 10
    seed: int = 0
    scoring: str = "f1_macro"


def outer_splits(n_samples: int, y: np.ndarray, cfg: EvalConfig) -> list[tuple[np.ndarray, np.ndarray]]:
    rskf = RepeatedStratifiedKFold(
        n_splits=cfg.n_splits,
        n_repeats=cfg.n_repeats,
        random_state=cfg.seed,
    )
    X_dummy = np.zeros((n_samples, 1))
    return list(rskf.split(X_dummy, y))


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    labels = [0, 1, 2]
    p, r, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    out = {}
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
    out["cross_level_0_2_error_rate"] = {
        "numerator": cross_02,
        "denominator": denom,
        "rate": float(cross_02 / denom) if denom else float("nan"),
    }
    return out


def run_audit_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    source_row_ids: np.ndarray,
    models: dict[str, tuple[ModelFactory, list[str]]],
    cfg: EvalConfig,
) -> dict[str, Any]:
    outer = outer_splits(len(y), y, cfg)
    fold_rows: list[dict] = []
    inner_rows: list[dict] = []
    n_folds = len(outer)
    oof = {k: np.full((cfg.n_repeats, len(y)), -1, dtype=int) for k in models}
    fold_membership: list[dict] = []

    for f_idx, (tr, te) in enumerate(outer):
        rep, fold = divmod(f_idx, cfg.n_splits)
        for i in tr:
            fold_membership.append(
                {
                    "repeat": rep,
                    "fold": fold,
                    "source_row_id": int(source_row_ids[i]),
                    "role": "train",
                }
            )
        for i in te:
            fold_membership.append(
                {
                    "repeat": rep,
                    "fold": fold,
                    "source_row_id": int(source_row_ids[i]),
                    "role": "valid",
                }
            )

        for name, (factory, cols) in models.items():
            est = factory()
            est.fit(X.iloc[tr][cols], y[tr])
            pred = est.predict(X.iloc[te][cols])
            oof[name][rep, te] = pred
            metrics = per_class_metrics(y[te], pred)
            row = {
                "repeat": rep,
                "fold": fold,
                "model": name,
                "macro_f1": metrics["macro_f1"],
                "accuracy": metrics["accuracy"],
                "selected_feature": getattr(est, "feature_", ""),
            }
            fold_rows.append(row)
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

    fs = pd.DataFrame(fold_rows)
    summary = {}
    for name, g in fs.groupby("model"):
        summary[name] = {
            "macro_f1_mean": float(g["macro_f1"].mean()),
            "macro_f1_std": float(g["macro_f1"].std(ddof=1)),
            "accuracy_mean": float(g["accuracy"].mean()),
            "accuracy_std": float(g["accuracy"].std(ddof=1)),
        }
    return {
        "fold_scores": fs,
        "inner_candidates": pd.DataFrame(inner_rows),
        "folds": pd.DataFrame(fold_membership),
        "oof": oof,
        "summary": summary,
        "n_outer_folds": n_folds,
    }


def paired_comparison(
    fold_scores: pd.DataFrame,
    model_a: str,
    model_b: str,
    outer: list[tuple[np.ndarray, np.ndarray]],
    *,
    direction: str = "a_minus_b",
) -> dict[str, float]:
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
    }
