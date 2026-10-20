"""Label-proxy check: how much of ``stress_level`` is recoverable from single inputs.

Descriptive only. A near-deterministic feature-label table shows that the
CSV's label is reproducible from that field; it does not say *why* (derived
label, synthetic generation, or coding convention). Scores use the same
RepeatedStratifiedKFold splits and pooled-OOF-per-repeat aggregation as the
experiment A audit, so ``lr_all`` here matches the audit's ``lr_all``.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold

from stress_detection.cv_protocol import PROTOCOL_LABELS, ZERO_DIVISION
from stress_detection.models import lr_pipeline

PROXY_FEATURE = "blood_pressure"


def feature_label_purity(X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """Per feature: share of rows whose value co-occurs with exactly one label."""
    rows = []
    labels = pd.Series(y, index=X.index)
    for col in X.columns:
        n_labels = labels.groupby(X[col]).nunique()
        pure = X[col].map(n_labels).eq(1)
        rows.append({"feature": col, "n_values": int(X[col].nunique()), "pure_row_share": float(pure.mean())})
    return pd.DataFrame(rows).sort_values("pure_row_share", ascending=False, ignore_index=True)


def crosstab(X: pd.DataFrame, y: np.ndarray, feature: str = PROXY_FEATURE) -> pd.DataFrame:
    table = pd.crosstab(X[feature], pd.Series(y, index=X.index, name="stress_level"))
    table.index.name = feature
    return table.reindex(columns=list(PROTOCOL_LABELS), fill_value=0)


def pooled_repeat_macro_f1(
    X: pd.DataFrame,
    y: np.ndarray,
    columns: Sequence[str],
    *,
    n_splits: int,
    n_repeats: int,
    seed: int,
) -> dict[str, float]:
    """Logistic regression macro-F1: pooled OOF per repeat, mean and sample SD across repeats."""
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    splits = list(rskf.split(np.zeros((len(y), 1)), y))
    scores = []
    for rep in range(n_repeats):
        oof = np.full(len(y), -1)
        for tr, te in splits[rep * n_splits : (rep + 1) * n_splits]:
            model = lr_pipeline(random_state=seed).fit(X.iloc[tr][list(columns)], y[tr])
            oof[te] = model.predict(X.iloc[te][list(columns)])
        scores.append(
            f1_score(y, oof, average="macro", labels=list(PROTOCOL_LABELS), zero_division=ZERO_DIVISION)
        )
    arr = np.asarray(scores)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0}


def label_proxy_report(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    strong_single_features: Sequence[str],
    n_splits: int = 5,
    n_repeats: int = 10,
    seed: int = 0,
) -> dict[str, Any]:
    feats = list(X.columns)
    variants = {
        "lr_all": feats,
        f"lr_{PROXY_FEATURE}_only": [PROXY_FEATURE],
        f"lr_without_{PROXY_FEATURE}": [c for c in feats if c != PROXY_FEATURE],
        "lr_without_strong_single_features": [c for c in feats if c not in set(strong_single_features)],
    }
    corr = X.corr().abs().to_numpy(copy=True)
    np.fill_diagonal(corr, np.nan)
    return {
        "proxy_feature": PROXY_FEATURE,
        "strong_single_features": list(strong_single_features),
        "mean_abs_feature_correlation": float(np.nanmean(corr)),
        "lr_variants": {
            name: {
                "n_features": len(cols),
                **pooled_repeat_macro_f1(X, y, cols, n_splits=n_splits, n_repeats=n_repeats, seed=seed),
            }
            for name, cols in variants.items()
        },
        "protocol": {"n_splits": n_splits, "n_repeats": n_repeats, "seed": seed, "scoring": "f1_macro"},
    }
