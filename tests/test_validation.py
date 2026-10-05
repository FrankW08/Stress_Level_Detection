"""Label-invariance and deliberate-leakage adapter tests."""

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from stress_detection.validation import (
    assert_label_invariance,
    assert_no_stress_level_in_features,
    clean_feature_adapter,
    deterministic_label_shuffle,
    leaky_zero_map_adapter,
    predict_without_labels,
    split_pipeline_preprocessor,
)


def test_no_stress_in_features():
    assert_no_stress_level_in_features(["a", "b"])
    with pytest.raises(AssertionError):
        assert_no_stress_level_in_features(["stress_level", "a"])


def _make_pipe():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2000),
    )


def test_label_invariance_clean_passes():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(60, 3)), columns=list("abc"))
    # inject some zeros for leak path contrast
    X.iloc[30:, 0] = 0
    y = np.array([0, 1, 2] * 20)
    pipe = _make_pipe()
    pipe.fit(X.iloc[:40], y[:40])
    pre, est = split_pipeline_preprocessor(pipe)
    assert_label_invariance(
        pre,
        pipe,
        X.iloc[40:].reset_index(drop=True),
        y[40:],
        feature_adapter=clean_feature_adapter(pre),
    )
    pred = predict_without_labels(pipe, X.iloc[40:])
    assert len(pred) == 20


def test_label_invariance_leaky_adapter_raises():
    rng = np.random.default_rng(1)
    # Columns that appear in LEAK_MAP
    X = pd.DataFrame(
        {
            "headache": [0, 0, 0, 1, 2, 0, 0, 0, 1, 0] * 4,
            "sleep_quality": [1, 0, 2, 0, 1, 0, 3, 0, 1, 2] * 4,
            "noise_level": rng.integers(0, 5, size=40),
        }
    )
    y = np.array([0, 1, 2, 0, 1, 2, 1, 0, 2, 1] * 4)
    pipe = _make_pipe()
    pipe.fit(X.iloc[:30], y[:30])
    pre, _ = split_pipeline_preprocessor(pipe)
    Xv = X.iloc[30:].reset_index(drop=True)
    yv = y[30:]
    with pytest.raises(AssertionError, match="label leakage|Validation features changed"):
        assert_label_invariance(
            pre,
            None,
            Xv,
            yv,
            feature_adapter=leaky_zero_map_adapter(pre),
            shuffle_seed=0,
        )


def test_y_valid_length_mismatch():
    pipe = _make_pipe()
    X = pd.DataFrame({"a": [1.0, 2.0], "b": [0.0, 1.0], "c": [0.0, 0.0]})
    y = np.array([0, 1])
    pipe.fit(X, y)
    pre, _ = split_pipeline_preprocessor(pipe)
    with pytest.raises(ValueError, match="length"):
        assert_label_invariance(pre, pipe, X, np.array([0]))


def test_y_valid_illegal_label():
    pipe = _make_pipe()
    X = pd.DataFrame({"a": [1.0, 2.0], "b": [0.0, 1.0], "c": [0.0, 0.0]})
    y = np.array([0, 1])
    pipe.fit(X, y)
    pre, _ = split_pipeline_preprocessor(pipe)
    with pytest.raises(ValueError, match="illegal"):
        assert_label_invariance(pre, pipe, X, np.array([0, 9]))


def test_deterministic_shuffle_changes_when_possible():
    y = np.array([0, 1, 2, 0, 1, 2])
    sh = deterministic_label_shuffle(y, seed=0)
    assert len(sh) == len(y)
    assert not np.array_equal(sh, y) or len(np.unique(y)) == 1
