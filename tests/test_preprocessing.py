"""Preprocessing API tests."""

import numpy as np
import pandas as pd

from stress_detection.preprocessing import (
    fit_imputer_median,
    fit_median_impute_scale,
    imputer_statistics_dict,
    transform_imputer,
)


def test_train_imputer_ignores_valid_extremes():
    """Train median must ignore validation extremes; missing train values use train median."""
    X_train = pd.DataFrame({"x": [1.0, 2.0, np.nan], "y": [10.0, 20.0, 30.0]})
    X_valid = pd.DataFrame({"x": [1000.0], "y": [np.nan]})
    imp = fit_imputer_median(X_train)
    stats = imputer_statistics_dict(imp, ["x", "y"])
    assert stats["x"] == 1.5  # median of 1,2 (nan ignored)
    assert stats["y"] == 20.0
    # Valid extreme must not be used to fit; transform keeps 1000 after impute of x
    Xt = transform_imputer(imp, X_valid)
    assert Xt.loc[0, "x"] == 1000.0
    assert Xt.loc[0, "y"] == 20.0  # imputed from train median


def test_fit_median_impute_scale_train_only():
    X_train = pd.DataFrame({"a": [0.0, 1.0, np.nan]})
    X_valid = pd.DataFrame({"a": [999.0]})
    pipe = fit_median_impute_scale(X_train)
    # scaler mean/scale from train imputed values [0,1,0.5]
    train_t = pipe.transform(X_train)
    valid_t = pipe.transform(X_valid)
    assert train_t.shape[1] == 1
    assert valid_t[0, 0] != train_t[0, 0]
