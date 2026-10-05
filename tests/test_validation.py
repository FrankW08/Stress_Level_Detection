import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from stress_detection.validation import (
    assert_label_invariance,
    assert_no_stress_level_in_features,
    predict_without_labels,
)


def test_no_stress_in_features():
    assert_no_stress_level_in_features(["a", "b"])
    with pytest.raises(AssertionError):
        assert_no_stress_level_in_features(["stress_level", "a"])


def test_label_invariance_pipeline():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(40, 3)), columns=list("abc"))
    y = np.array([0, 1, 2] * 13 + [0])
    pipe = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2000),
    )
    pipe.fit(X.iloc[:30], y[:30])
    assert_label_invariance(pipe, X.iloc[30:], y[30:])
    pred = predict_without_labels(pipe, X.iloc[30:])
    assert pred.shape == (10,)


def test_train_imputer_ignores_valid_extremes():
    X_train = pd.DataFrame({"x": [1.0, 2.0, np.nan]})
    X_valid = pd.DataFrame({"x": [100.0]})
    imp = SimpleImputer(strategy="median")
    imp.fit(X_train)
    assert imp.transform(X_valid)[0, 0] == 100.0
