"""Load and parse the stress dataset without mutating the original CSV."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

AnalysisScope = Literal["primary", "sensitivity"]

VALID_STRESS_LABELS = {0, 1, 2}

# Provisional schema bounds (inferred from bulk data; not a confirmed data dictionary).
PROVISIONAL_MAX: dict[str, float] = {
    "anxiety_level": 21,
    "self_esteem": 30,
    "mental_health_history": 1,
    "depression": 27,
    "blood_pressure": 3,
    "social_support": 3,
    "stress_level": 2,
}
DEFAULT_FEATURE_MAX = 5.0

NULL_TOKENS = frozenset({"", "null", "Null", "NULL", "nan", "NaN", "NA", "None"})


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_cell(value: object) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).strip()
    s = s.strip('"').strip("'")
    s = s.strip("\u201c\u201d")  # curly quotes
    if s in NULL_TOKENS:
        return None
    return s


def parse_numeric_series(series: pd.Series) -> pd.Series:
    """Parse string cells to float; invalid values become NaN (original string logged separately)."""
    normalized = series.map(normalize_cell)
    out = pd.to_numeric(normalized, errors="coerce")
    return out


def load_raw_csv(csv_path: Path) -> pd.DataFrame:
    """Read CSV as strings; never overwrite the source file."""
    return pd.read_csv(csv_path, dtype=str)


def parse_dataset(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (parsed numeric frame with source_row_id, parse_issues table).
    """
    parsed = pd.DataFrame(
        {c: parse_numeric_series(raw[c]) for c in raw.columns},
        index=raw.index,
    )
    parsed.insert(0, "source_row_id", parsed.index.astype(int))

    issues: list[dict] = []
    for row_idx in raw.index:
        for col in raw.columns:
            orig = raw.at[row_idx, col]
            norm = normalize_cell(orig)
            if norm is None:
                continue
            try:
                float(norm)
            except ValueError:
                issues.append(
                    {
                        "source_row_id": int(row_idx),
                        "column": col,
                        "original_value": str(orig),
                        "reason": "non_numeric_after_normalization",
                    }
                )
    issues_df = pd.DataFrame(issues)
    return parsed, issues_df


def stress_label_exclusion_reason(value: float | np.floating | None) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "missing_stress_level"
    if not float(value).is_integer():
        return "non_integer_stress_level"
    iv = int(value)
    if iv not in VALID_STRESS_LABELS:
        return f"illegal_stress_level_{iv}"
    return None


def build_exclusions(
    parsed: pd.DataFrame,
    scope: AnalysisScope,
    *,
    tail_quarantine_from: int = 1100,
) -> pd.DataFrame:
    rows: list[dict] = []
    n = len(parsed)
    if scope == "primary":
        for i in range(tail_quarantine_from, n):
            rows.append(
                {
                    "source_row_id": int(i),
                    "reason": "tail_rows_quarantined_by_audit_convention",
                    "detail": f"rows>={tail_quarantine_from} excluded from primary scope",
                }
            )
        label_rows = parsed.index[parsed.index < tail_quarantine_from]
    else:
        label_rows = parsed.index

    for i in label_rows:
        reason = stress_label_exclusion_reason(parsed.at[i, "stress_level"])
        if reason:
            rows.append({"source_row_id": int(i), "reason": reason, "detail": ""})
    # Deduplicate exclusion rows (same row, multiple reasons unlikely but keep first)
    if not rows:
        return pd.DataFrame(columns=["source_row_id", "reason", "detail"])
    ex = pd.DataFrame(rows).drop_duplicates(subset=["source_row_id"], keep="first")
    return ex


def select_analysis_frame(
    parsed: pd.DataFrame,
    exclusions: pd.DataFrame,
) -> pd.DataFrame:
    excluded_ids = set(exclusions["source_row_id"].astype(int))
    keep = parsed[~parsed["source_row_id"].isin(excluded_ids)].copy()
    return keep.reset_index(drop=True)


def field_checks(parsed: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in parsed.columns if c not in ("source_row_id", "stress_level")]
    records = []
    for c in feature_cols:
        s = parsed[c]
        prov = PROVISIONAL_MAX.get(c, DEFAULT_FEATURE_MAX)
        records.append(
            {
                "field": c,
                "min": float(s.min(skipna=True)) if s.notna().any() else np.nan,
                "max": float(s.max(skipna=True)) if s.notna().any() else np.nan,
                "n_missing": int(s.isna().sum()),
                "provisional_max": prov,
                "n_above_provisional_max": int((s > prov).sum(skipna=True)),
            }
        )
    return pd.DataFrame(records)


@dataclass(frozen=True)
class PreparedData:
    X: pd.DataFrame
    y: np.ndarray
    source_row_ids: np.ndarray
    feature_names: list[str]
    exclusions: pd.DataFrame
    parse_issues: pd.DataFrame
    field_checks: pd.DataFrame
    csv_sha256: str
    scope: AnalysisScope


def prepare_data(
    csv_path: Path,
    scope: AnalysisScope = "primary",
    *,
    tail_quarantine_from: int = 1100,
) -> PreparedData:
    csv_path = Path(csv_path).resolve()
    raw = load_raw_csv(csv_path)
    parsed, parse_issues = parse_dataset(raw)
    exclusions = build_exclusions(parsed, scope, tail_quarantine_from=tail_quarantine_from)
    frame = select_analysis_frame(parsed, exclusions)

    label_reasons = []
    valid_mask = []
    for _, row in frame.iterrows():
        r = stress_label_exclusion_reason(row["stress_level"])
        label_reasons.append(r)
        valid_mask.append(r is None)
    # Rows with bad labels should already be in exclusions; double-check:
    if not all(valid_mask):
        bad = frame.loc[[not m for m in valid_mask], "source_row_id"].tolist()
        raise ValueError(f"Unexpected illegal labels after exclusion: {bad}")

    y = frame["stress_level"].astype(np.int64).to_numpy()
    X = frame.drop(columns=["stress_level"]).copy()
    if "source_row_id" in X.columns:
        ids = X["source_row_id"].to_numpy()
        X = X.drop(columns=["source_row_id"])
    else:
        ids = frame["source_row_id"].to_numpy()

    return PreparedData(
        X=X.reset_index(drop=True),
        y=y,
        source_row_ids=ids.astype(int),
        feature_names=list(X.columns),
        exclusions=exclusions,
        parse_issues=parse_issues,
        field_checks=field_checks(parsed),
        csv_sha256=sha256_file(csv_path),
        scope=scope,
    )


# Notebook / audit: intentional label-driven zero map (leakage demo only).
LEAK_MAP: dict[str, list[int]] = {
    "headache": [0, 2, 4],
    "sleep_quality": [0, 3, 4],
    "breathing_problem": [0, 2, 5],
    "noise_level": [0, 3, 5],
    "living_conditions": [4, 5, 0],
    "safety": [5, 3, 0],
    "basic_needs": [5, 3, 0],
    "academic_performance": [5, 3, 0],
    "study_load": [0, 3, 5],
    "teacher_student_relationship": [5, 3, 0],
    "future_career_concerns": [0, 3, 5],
    "social_support": [3, 2, 0],
    "peer_pressure": [0, 3, 5],
    "extracurricular_activities": [0, 3, 5],
    "bullying": [0, 3, 5],
}


def apply_label_driven_zero_map(X: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Audit-only: replace zeros using stress labels (data leakage)."""
    Xm = X.copy()
    for col, mapping in LEAK_MAP.items():
        if col not in Xm.columns:
            continue
        z = (Xm[col] == 0).to_numpy()
        if not z.any():
            continue
        Xm.loc[z, col] = [mapping[int(v)] for v in labels[z]]
    return Xm
