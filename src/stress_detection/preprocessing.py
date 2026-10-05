"""Training-only preprocessing helpers (used by notebooks and benchmarks)."""

from __future__ import annotations

import pandas as pd
from sklearn.impute import SimpleImputer


def fit_imputer_median(X_train: pd.DataFrame) -> SimpleImputer:
    imp = SimpleImputer(strategy="median")
    imp.fit(X_train)
    return imp


def transform_imputer(imp: SimpleImputer, X: pd.DataFrame) -> pd.DataFrame:
    cols = X.columns
    out = imp.transform(X)
    return pd.DataFrame(out, columns=cols, index=X.index)
