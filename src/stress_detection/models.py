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
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from stress_detection.cv_protocol import MACRO_F1_PROTOCOL, validate_generated_splits

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


def grouped_inner_splits(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    n_splits: int,
    random_state: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """StratifiedGroupKFold splits with an explicit train/valid group-disjoint check.

    Refuses to fall back to ungrouped StratifiedKFold.
    """
    n_groups = int(len(np.unique(groups)))
    if n_splits > n_groups:
        raise ValueError(
            f"inner_splits={n_splits} exceeds n_groups={n_groups} in the outer training "
            "fold; refusing to fall back to ungrouped StratifiedKFold"
        )
    class_counts = np.bincount(np.asarray(y, dtype=int))
    class_counts = class_counts[class_counts > 0]
    if len(class_counts) < 2:
        raise ValueError(
            f"inner_splits={n_splits} not feasible: outer training fold has "
            f"{len(class_counts)} class(es); refusing to fall back to ungrouped CV"
        )
    if n_splits > int(class_counts.min()):
        raise ValueError(
            f"inner_splits={n_splits} exceeds smallest class count "
            f"({int(class_counts.min())}) in the outer training fold; "
            "refusing to fall back to ungrouped StratifiedKFold"
        )
    try:
        sgkf = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
        splits = list(sgkf.split(X, y, groups))
    except ValueError as exc:
        raise ValueError(
            f"inner_splits={n_splits} is not feasible for grouped inner CV "
            f"(n_groups={n_groups}): {exc}. Refusing to fall back to ungrouped StratifiedKFold"
        ) from exc
    validate_generated_splits(
        splits, np.asarray(y), groups=np.asarray(groups), name="grouped inner CV"
    )
    return splits


class NestedBestSingleFeature(BaseEstimator, ClassifierMixin):
    """Inner 3-fold CV on training fold only; depth-3 tree on one feature."""

    def __init__(self, random_state: int = DEFAULT_SEED, inner_splits: int = 3):
        self.random_state = random_state
        self.inner_splits = inner_splits

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        groups: np.ndarray | None = None,
        inner_cv: list[tuple[np.ndarray, np.ndarray]] | None = None,
    ):
        X = pd.DataFrame(X)
        y = np.asarray(y)
        if inner_cv is not None:
            cv_splits = [(np.asarray(tr), np.asarray(te)) for tr, te in inner_cv]
            validate_generated_splits(
                cv_splits, y, groups=groups, name="nested_single inner CV"
            )
        elif groups is None:
            inner = StratifiedKFold(
                n_splits=self.inner_splits, shuffle=True, random_state=self.random_state
            )
            cv_splits = list(inner.split(X, y))
            validate_generated_splits(cv_splits, y, groups=None, name="nested_single inner CV")
        else:
            groups = np.asarray(groups)
            if len(groups) != len(X):
                raise ValueError(
                    f"groups length {len(groups)} != n_samples {len(X)} for nested inner CV"
                )
            cv_splits = grouped_inner_splits(
                X, y, groups, n_splits=self.inner_splits, random_state=self.random_state
            )
        self.inner_splits_used_ = cv_splits
        self.used_groups_ = groups is not None
        self.candidates_: dict[str, float] = {}
        for col in X.columns:
            scores = cross_val_score(
                tree_depth3_pipeline(random_state=self.random_state),
                X[[col]],
                y,
                cv=cv_splits,
                scoring=MACRO_F1_PROTOCOL,
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
