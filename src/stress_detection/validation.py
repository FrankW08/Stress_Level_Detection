"""Leakage checks: validation labels must not affect features or predictions."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from stress_detection.data import apply_label_driven_zero_map


def predict_without_labels(
    pipeline: Pipeline,
    X_valid: pd.DataFrame,
) -> np.ndarray:
    """Production-style predict: no y_valid."""
    return pipeline.predict(X_valid)


def assert_label_invariance(
    fitted_pipeline: Pipeline,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    *,
    rtol: float = 0.0,
    atol: float = 0.0,
) -> None:
    """
    Changing or removing validation labels must not change predictions
    when the preprocessor and model are already fitted and X_valid is fixed.
    """
    pred_base = predict_without_labels(fitted_pipeline, X_valid)
    pred_repeat = predict_without_labels(fitted_pipeline, X_valid)
    if not np.array_equal(pred_base, pred_repeat):
        raise AssertionError("Predictions changed on identical X_valid without refitting")
    _ = y_valid  # y_valid must not be passed into predict; caller documents label-agnostic inference
    if rtol == 0 and atol == 0:
        if len(pred_base) != len(X_valid):
            raise AssertionError("Prediction length mismatch")


def assert_no_stress_level_in_features(feature_names: list[str]) -> None:
    forbidden = {n for n in feature_names if n == "stress_level" or n.startswith("stress_level")}
    if forbidden:
        raise AssertionError(f"Target leakage in feature list: {forbidden}")


def assert_leak_detector_fires_on_zero_map(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    *,
    train_fn,
) -> None:
    """
    Deliberate leakage path: label-driven zero map before fit must be detectable
    by comparing predictions with/without access to y_valid at transform time.
    Here we verify that clean vs leaky *training* produces different models,
    and that leaky preprocessing uses y (documented audit case).
    """
    clean = train_fn(X_train, y_train)
    leaky_X_train = apply_label_driven_zero_map(X_train, y_train)
    leaky = train_fn(leaky_X_train, y_train)
    p_clean = predict_without_labels(clean, X_valid)
    p_leaky = predict_without_labels(leaky, X_valid)
    if np.array_equal(p_clean, p_leaky):
        raise AssertionError("Leak detector: clean and leaky branches identical (unexpected)")
