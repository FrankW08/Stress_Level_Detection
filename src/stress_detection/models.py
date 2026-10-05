"""Model factories and nested single-feature selector."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

DEFAULT_SEED = 0
PSYCH_FEATURES = ["anxiety_level", "self_esteem", "mental_health_history", "depression"]

ModelFactory = Callable[[], Pipeline | BaseEstimator]


def json_safe_params(obj: Any) -> Any:
    """Convert get_params(deep=True) to JSON-serializable structure."""
    if isinstance(obj, dict):
        return {str(k): json_safe_params(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe_params(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # sklearn estimators / callables
    return repr(obj)


def estimator_params(est: BaseEstimator) -> dict[str, Any]:
    return json_safe_params(est.get_params(deep=True))


def lr_pipeline(*, random_state: int = DEFAULT_SEED) -> Pipeline:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=5000, random_state=random_state),
    )


def svm_linear_pipeline(*, C: float = 0.2) -> Pipeline:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        SVC(kernel="linear", C=C),
    )


def rf_pipeline(*, random_state: int = DEFAULT_SEED) -> Pipeline:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(random_state=random_state),
    )


def tree_depth3_pipeline(*, random_state: int = DEFAULT_SEED) -> Pipeline:
    return make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        DecisionTreeClassifier(max_depth=3, random_state=random_state),
    )


def dummy_most_frequent() -> DummyClassifier:
    return DummyClassifier(strategy="most_frequent")


def tie_break_feature(scores: dict[str, float]) -> str:
    """Deterministic tie-break: highest score, then alphabetical feature name."""
    best = max(scores.values())
    candidates = sorted(k for k, v in scores.items() if v == best)
    return candidates[0]


class NestedBestSingleFeature(BaseEstimator, ClassifierMixin):
    """Inner 3-fold CV on training fold only; depth-3 tree on one feature."""

    def __init__(self, random_state: int = DEFAULT_SEED, inner_splits: int = 3):
        self.random_state = random_state
        self.inner_splits = inner_splits

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        X = pd.DataFrame(X)
        inner = StratifiedKFold(
            n_splits=self.inner_splits, shuffle=True, random_state=self.random_state
        )
        self.candidates_: dict[str, float] = {}
        for col in X.columns:
            scores = cross_val_score(
                tree_depth3_pipeline(random_state=self.random_state),
                X[[col]],
                y,
                cv=inner,
                scoring="f1_macro",
            )
            self.candidates_[col] = float(scores.mean())
        self.feature_ = tie_break_feature(self.candidates_)
        self.model_ = tree_depth3_pipeline(random_state=self.random_state).fit(
            X[[self.feature_]], y
        )
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = pd.DataFrame(X)
        return self.model_.predict(X[[self.feature_]])


def audit_model_registry(
    all_features: Sequence[str],
    *,
    random_state: int = DEFAULT_SEED,
) -> dict[str, tuple[ModelFactory, list[str]]]:
    feats = list(all_features)
    no_psych = [c for c in feats if c not in PSYCH_FEATURES]
    seed = int(random_state)
    return {
        "dummy": (dummy_most_frequent, feats),
        "lr_all": (lambda: lr_pipeline(random_state=seed), feats),
        "rf_all": (lambda: rf_pipeline(random_state=seed), feats),
        "svm_all": (lambda: svm_linear_pipeline(C=0.2), feats),
        "lr_no_psych": (lambda: lr_pipeline(random_state=seed), no_psych),
        "nested_single": (
            lambda: NestedBestSingleFeature(random_state=seed),
            feats,
        ),
    }
