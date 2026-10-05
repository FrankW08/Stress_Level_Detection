"""Training-only preprocessing helpers (used by notebooks and benchmarks)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def fit_imputer_median(X_train: pd.DataFrame) -> SimpleImputer:
    """Fit median imputer on training features only."""
    imp = SimpleImputer(strategy="median")
    imp.fit(X_train)
    return imp


def transform_imputer(imp: SimpleImputer, X: pd.DataFrame) -> pd.DataFrame:
    cols = list(X.columns)
    out = imp.transform(X)
    return pd.DataFrame(out, columns=cols, index=X.index)


def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def transform_scaler(scaler: StandardScaler, X: pd.DataFrame) -> pd.DataFrame:
    cols = list(X.columns)
    out = scaler.transform(X)
    return pd.DataFrame(out, columns=cols, index=X.index)


def fit_median_impute_scale(X_train: pd.DataFrame) -> Pipeline:
    """Fit imputer+scaler on train only; validation extremes do not affect train stats."""
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    pipe.fit(X_train)
    return pipe


def imputer_statistics_dict(imp: SimpleImputer, columns: list[str]) -> dict[str, float]:
    stats = np.asarray(imp.statistics_, dtype=float)
    return {c: float(stats[i]) for i, c in enumerate(columns)}
