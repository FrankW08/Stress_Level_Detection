"""Fixed 3-class evaluation protocol for stress_level in {0, 1, 2}."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.metrics import f1_score, make_scorer

from stress_detection.run_config import ConfigError

PROTOCOL_LABELS: tuple[int, int, int] = (0, 1, 2)
ZERO_DIVISION = 0

MACRO_F1_PROTOCOL = make_scorer(
    f1_score,
    average="macro",
    labels=list(PROTOCOL_LABELS),
    zero_division=ZERO_DIVISION,
)


def protocol_class_counts(
    y: np.ndarray, labels: Sequence[int] = PROTOCOL_LABELS
) -> dict[int, int]:
    y = np.asarray(y)
    return {int(c): int(np.sum(y == c)) for c in labels}


def missing_protocol_classes(
    y: np.ndarray, labels: Sequence[int] = PROTOCOL_LABELS
) -> list[int]:
    return [c for c, n in protocol_class_counts(y, labels).items() if n == 0]


def assert_split_pair(
    tr: np.ndarray,
    te: np.ndarray,
    y: np.ndarray,
    *,
    groups: np.ndarray | None = None,
    name: str = "CV",
    fold_i: int = 0,
    required_labels: Sequence[int] = PROTOCOL_LABELS,
) -> None:
    """Check emptiness, index overlap, group isolation, and required class coverage."""
    tr = np.asarray(tr)
    te = np.asarray(te)
    y = np.asarray(y)
    if len(tr) == 0 or len(te) == 0:
        raise ConfigError(f"{name} fold {fold_i}: empty train or valid set")
    n = len(y)
    if int(tr.min()) < 0 or int(te.min()) < 0 or int(tr.max()) >= n or int(te.max()) >= n:
        raise ConfigError(
            f"{name} fold {fold_i}: split indices out of range for n={n} "
            f"(train {int(tr.min())}..{int(tr.max())}, valid {int(te.min())}..{int(te.max())})"
        )
    overlap_idx = np.intersect1d(tr, te)
    if overlap_idx.size:
        raise ConfigError(
            f"{name} fold {fold_i}: train/valid sample indices overlap "
            f"(n={int(overlap_idx.size)})"
        )
    gtr = gte = None
    if groups is not None:
        groups = np.asarray(groups)
        gtr = set(groups[tr].tolist())
        gte = set(groups[te].tolist())
        overlap_g = sorted(gtr & gte)
        if overlap_g:
            raise ConfigError(
                f"{name} fold {fold_i}: train/valid groups overlap ({overlap_g[:8]}); "
                "refusing to fall back to ungrouped CV"
            )
    for role, idx in (("train", tr), ("valid", te)):
        y_role = y[idx]
        missing = missing_protocol_classes(y_role, required_labels)
        if not missing:
            continue
        counts = protocol_class_counts(y_role, required_labels)
        n_groups_role: Any = None
        if groups is not None:
            n_groups_role = int(len(np.unique(groups[idx])))
        extra = f", n_groups={n_groups_role}" if n_groups_role is not None else ""
        raise ConfigError(
            f"{name} fold {fold_i} {role}: missing required class(es) {missing}; "
            f"class counts={counts}{extra}. "
            f"This audit uses a 3-class protocol with labels {list(required_labels)}. "
            "Refusing to fall back to ungrouped CV or resample seeds."
        )


def validate_generated_splits(
    splits: list[tuple[np.ndarray, np.ndarray]],
    y: np.ndarray,
    *,
    groups: np.ndarray | None = None,
    name: str,
    required_labels: Sequence[int] = PROTOCOL_LABELS,
) -> None:
    if not splits:
        raise ConfigError(f"{name}: no splits were generated")
    for i, (tr, te) in enumerate(splits):
        assert_split_pair(
            tr,
            te,
            y,
            groups=groups,
            name=name,
            fold_i=i,
            required_labels=required_labels,
        )
