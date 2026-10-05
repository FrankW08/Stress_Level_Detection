"""Leakage checks: validation labels must not affect features or predictions."""

from __future__ import annotations

from typing import Callable, Protocol

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from stress_detection.data import VALID_STRESS_LABELS, apply_label_driven_zero_map


class FeatureAdapter(Protocol):
    """Build validation features from fixed X_valid, optionally using y_valid."""

    def __call__(self, X_valid: pd.DataFrame, y_valid: np.ndarray) -> np.ndarray | pd.DataFrame: ...


def clean_feature_adapter(fitted_preprocessor: BaseEstimator) -> FeatureAdapter:
    """Ignores y_valid; transforms X_valid only (production-safe)."""

    def _adapt(X_valid: pd.DataFrame, y_valid: np.ndarray) -> np.ndarray:
        _ = y_valid
        return fitted_preprocessor.transform(X_valid)

    return _adapt


def leaky_zero_map_adapter(fitted_preprocessor: BaseEstimator) -> FeatureAdapter:
    """Deliberate leakage: rewrite zeros using y_valid before transform (audit-only)."""

    def _adapt(X_valid: pd.DataFrame, y_valid: np.ndarray) -> np.ndarray:
        Xm = apply_label_driven_zero_map(X_valid, y_valid)
        return fitted_preprocessor.transform(Xm)

    return _adapt


def predict_without_labels(estimator: BaseEstimator, X_valid: pd.DataFrame) -> np.ndarray:
    """Production-style predict: no y_valid."""
    return estimator.predict(X_valid)


def _as_array(features: np.ndarray | pd.DataFrame) -> np.ndarray:
    if isinstance(features, pd.DataFrame):
        return features.to_numpy()
    return np.asarray(features)


def _validate_y_valid(y_valid: np.ndarray, n_rows: int) -> np.ndarray:
    y = np.asarray(y_valid)
    if y.ndim != 1:
        raise ValueError(f"y_valid must be 1-D, got shape {y.shape}")
    if len(y) != n_rows:
        raise ValueError(f"y_valid length {len(y)} != X_valid rows {n_rows}")
    for v in y:
        if int(v) not in VALID_STRESS_LABELS or float(v) != int(v):
            raise ValueError(f"illegal label in y_valid: {v}")
    return y.astype(int)


def deterministic_label_shuffle(y_valid: np.ndarray, seed: int = 0) -> np.ndarray:
    """Deterministic permutation of labels (same length; not identity when possible)."""
    y = np.asarray(y_valid).copy()
    if len(y) <= 1:
        return y
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    # Guarantee a change when there are at least two distinct labels
    out = y[perm]
    if np.array_equal(out, y) and len(np.unique(y)) > 1:
        # swap first two differing positions
        for i in range(len(y)):
            for j in range(i + 1, len(y)):
                if y[i] != y[j]:
                    out = y.copy()
                    out[i], out[j] = out[j], out[i]
                    return out
    return out


def assert_label_invariance(
    fitted_preprocessor: BaseEstimator,
    fitted_estimator: BaseEstimator | None,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    *,
    feature_adapter: FeatureAdapter | None = None,
    shuffle_seed: int = 0,
) -> None:
    """
    Fixed fitted preprocessor and X_valid: changing y_valid must not change
    constructed validation features (via adapter) or predictions.

    Default adapter is clean (ignores y_valid). A leaky adapter must raise.
    """
    y = _validate_y_valid(y_valid, len(X_valid))
    adapter = feature_adapter or clean_feature_adapter(fitted_preprocessor)
    y_shuffled = deterministic_label_shuffle(y, seed=shuffle_seed)

    feats_true = _as_array(adapter(X_valid, y))
    feats_shuf = _as_array(adapter(X_valid, y_shuffled))
    if feats_true.shape != feats_shuf.shape:
        raise AssertionError(
            f"Feature shape changed under label shuffle: {feats_true.shape} vs {feats_shuf.shape}"
        )
    if not np.allclose(feats_true, feats_shuf, equal_nan=True):
        raise AssertionError(
            "Validation features changed when y_valid was shuffled "
            "(label leakage in feature construction)"
        )

    if fitted_estimator is not None:
        # Prefer predicting from original X when estimator is a full pipeline;
        # otherwise predict from constructed features.
        if isinstance(fitted_estimator, Pipeline) and fitted_estimator is not fitted_preprocessor:
            pred_true = predict_without_labels(fitted_estimator, X_valid)
            pred_shuf = predict_without_labels(fitted_estimator, X_valid)
        else:
            pred_true = fitted_estimator.predict(feats_true)
            pred_shuf = fitted_estimator.predict(feats_shuf)
        if not np.array_equal(pred_true, pred_shuf):
            raise AssertionError("Predictions changed when y_valid was shuffled")


def assert_no_stress_level_in_features(feature_names: list[str]) -> None:
    forbidden = {n for n in feature_names if n == "stress_level" or n.startswith("stress_level")}
    if forbidden:
        raise AssertionError(f"Target leakage in feature list: {forbidden}")


def split_pipeline_preprocessor(pipeline: Pipeline) -> tuple[Pipeline, BaseEstimator]:
    """Split a sklearn Pipeline into preprocessor (all but last) and final estimator."""
    if len(pipeline.steps) < 2:
        raise ValueError("Pipeline needs at least preprocessor + estimator")
    pre = Pipeline(pipeline.steps[:-1])
    est = pipeline.steps[-1][1]
    return pre, est
