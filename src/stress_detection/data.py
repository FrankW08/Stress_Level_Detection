"""Load and parse the stress dataset without mutating the original CSV."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from stress_detection.hashes import sha256_file, sha256_file_normalized_lf
from stress_detection.schema import FieldRule, PROVISIONAL_SCHEMA, rule_for

AnalysisScope = Literal["primary", "sensitivity"]
SensitivityPolicy = Literal["raw", "quarantine_out_of_range", "grouped_duplicates"]

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


def normalize_cell(value: object) -> tuple[str | None, list[str]]:
    """Return (normalized string or None for missing, list of actions applied)."""
    actions: list[str] = []
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None, ["already_missing"]
    s = str(value)
    stripped = s.strip()
    if stripped != s:
        actions.append("stripped_whitespace")
    s = stripped
    before = s
    s = s.strip('"').strip("'")
    s = s.strip("\u201c\u201d")
    if s != before:
        actions.append("removed_outer_quotes")
    if s in NULL_TOKENS:
        actions.append("null_token_to_missing")
        return None, actions
    return s, actions


def parse_numeric_series(series: pd.Series) -> pd.Series:
    normalized = series.map(lambda v: normalize_cell(v)[0])
    return pd.to_numeric(normalized, errors="coerce")


def load_raw_csv(csv_path: Path) -> pd.DataFrame:
    """Read CSV as strings; never overwrite the source file."""
    return pd.read_csv(csv_path, dtype=str)


def parse_dataset(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (parsed numeric frame with source_row_id, parse_issues table)."""
    parsed = pd.DataFrame(
        {c: parse_numeric_series(raw[c]) for c in raw.columns},
        index=raw.index,
    )
    parsed.insert(0, "source_row_id", parsed.index.astype(int))

    issues: list[dict] = []
    for row_idx in raw.index:
        for col in raw.columns:
            orig = raw.at[row_idx, col]
            norm, actions = normalize_cell(orig)
            orig_s = (
                ""
                if orig is None or (isinstance(orig, float) and np.isnan(orig))
                else str(orig)
            )
            if not actions and norm is not None:
                try:
                    float(norm)
                    continue
                except ValueError:
                    pass

            if "null_token_to_missing" in actions or (
                norm is None and actions and actions != ["already_missing"]
            ):
                issues.append(
                    {
                        "source_row_id": int(row_idx),
                        "column": col,
                        "original_value": orig_s,
                        "normalized_value": "",
                        "action": ";".join(actions),
                        "reason": "null_or_empty_to_missing",
                    }
                )
                continue

            if "removed_outer_quotes" in actions or "stripped_whitespace" in actions:
                try:
                    if norm is not None:
                        float(norm)
                    issues.append(
                        {
                            "source_row_id": int(row_idx),
                            "column": col,
                            "original_value": orig_s,
                            "normalized_value": "" if norm is None else norm,
                            "action": ";".join(actions),
                            "reason": "format_normalized",
                        }
                    )
                except (ValueError, TypeError):
                    issues.append(
                        {
                            "source_row_id": int(row_idx),
                            "column": col,
                            "original_value": orig_s,
                            "normalized_value": "" if norm is None else norm,
                            "action": ";".join(actions),
                            "reason": "non_numeric_after_normalization",
                        }
                    )
                continue

            if norm is not None:
                try:
                    float(norm)
                except ValueError:
                    issues.append(
                        {
                            "source_row_id": int(row_idx),
                            "column": col,
                            "original_value": orig_s,
                            "normalized_value": norm,
                            "action": ";".join(actions) if actions else "none",
                            "reason": "non_numeric_after_normalization",
                        }
                    )
    issues_df = pd.DataFrame(issues)
    if issues_df.empty:
        issues_df = pd.DataFrame(
            columns=[
                "source_row_id",
                "column",
                "original_value",
                "normalized_value",
                "action",
                "reason",
            ]
        )
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
    quarantine_out_of_range: bool = False,
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

    if quarantine_out_of_range:
        feature_cols = [c for c in parsed.columns if c not in ("source_row_id", "stress_level")]
        for i in label_rows:
            if any(r["source_row_id"] == int(i) for r in rows):
                continue
            for c in feature_cols:
                v = parsed.at[i, c]
                if pd.isna(v):
                    continue
                prov = PROVISIONAL_MAX.get(c, DEFAULT_FEATURE_MAX)
                if float(v) > prov:
                    rows.append(
                        {
                            "source_row_id": int(i),
                            "reason": "quarantine_out_of_range",
                            "detail": f"{c}={v} > provisional_max={prov}",
                        }
                    )
                    break

    if not rows:
        return pd.DataFrame(columns=["source_row_id", "reason", "detail"])
    return pd.DataFrame(rows).drop_duplicates(subset=["source_row_id"], keep="first")


def select_analysis_frame(
    parsed: pd.DataFrame,
    exclusions: pd.DataFrame,
) -> pd.DataFrame:
    excluded_ids = set(exclusions["source_row_id"].astype(int))
    keep = parsed[~parsed["source_row_id"].isin(excluded_ids)].copy()
    return keep.reset_index(drop=True)


def _is_missing(value: object) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value)


def _is_non_finite(value: object) -> bool:
    if _is_missing(value):
        return False
    try:
        fv = float(value)
    except (TypeError, ValueError):
        return False
    return not np.isfinite(fv)


def classify_cell_violations(column: str, value: object, rule: FieldRule | None) -> list[str]:
    """Return zero or more violation codes. Missing is not a domain violation."""
    if _is_missing(value):
        return []
    if _is_non_finite(value):
        return ["non_finite"]
    if rule is None:
        return []
    fv = float(value)
    codes: list[str] = []
    if rule.lower_bound is not None and fv < rule.lower_bound:
        codes.append("below_min")
    if rule.upper_bound is not None and fv > rule.upper_bound:
        codes.append("above_max")
    if rule.integer_required and not float(fv).is_integer():
        codes.append("non_integer")
    if rule.allowed_values is not None and fv not in rule.allowed_values:
        codes.append("invalid_category")
    return codes


def domain_violations_table(parsed: pd.DataFrame) -> pd.DataFrame:
    """Cell-level provisional-schema violations (reporting only)."""
    rows: list[dict] = []
    columns = [c for c in parsed.columns if c != "source_row_id"]
    for _, row in parsed.iterrows():
        for c in columns:
            rule = rule_for(c)
            v = row[c]
            if _is_missing(v):
                continue
            for code in classify_cell_violations(c, v, rule):
                rows.append(
                    {
                        "source_row_id": int(row["source_row_id"]),
                        "column": c,
                        "parsed_value": float(v) if not _is_non_finite(v) else str(v),
                        "violation": code,
                        "lower_bound": None if rule is None else rule.lower_bound,
                        "upper_bound": None if rule is None else rule.upper_bound,
                        "integer_required": None if rule is None else rule.integer_required,
                        "allowed_values": (
                            None
                            if rule is None or rule.allowed_values is None
                            else ",".join(str(int(x) if float(x).is_integer() else x) for x in sorted(rule.allowed_values))
                        ),
                        "rule_source": None if rule is None else rule.source,
                        "rule_status": None if rule is None else rule.status,
                    }
                )
    if not rows:
        return pd.DataFrame(
            columns=[
                "source_row_id",
                "column",
                "parsed_value",
                "violation",
                "lower_bound",
                "upper_bound",
                "integer_required",
                "allowed_values",
                "rule_source",
                "rule_status",
            ]
        )
    return pd.DataFrame(rows)


def domain_violation_counts(violations: pd.DataFrame, parsed: pd.DataFrame) -> dict[str, int]:
    if violations.empty:
        n_missing_cells = int(parsed.drop(columns=["source_row_id"], errors="ignore").isna().sum().sum())
        return {
            "n_violation_records": 0,
            "n_cells_with_any_violation": 0,
            "n_rows_with_any_violation": 0,
            "n_missing_cells": n_missing_cells,
        }
    cells = violations[["source_row_id", "column"]].drop_duplicates()
    n_missing_cells = int(parsed.drop(columns=["source_row_id"], errors="ignore").isna().sum().sum())
    return {
        "n_violation_records": int(len(violations)),
        "n_cells_with_any_violation": int(len(cells)),
        "n_rows_with_any_violation": int(cells["source_row_id"].nunique()),
        "n_missing_cells": n_missing_cells,
    }


def field_checks(parsed: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in parsed.columns if c not in ("source_row_id",)]
    violations = domain_violations_table(parsed)
    records = []
    for c in feature_cols:
        s = parsed[c]
        rule = rule_for(c)
        prov = PROVISIONAL_MAX.get(c, DEFAULT_FEATURE_MAX)
        gv = violations[violations["column"] == c] if not violations.empty else violations
        records.append(
            {
                "field": c,
                "min": float(s.min(skipna=True)) if s.notna().any() else np.nan,
                "max": float(s.max(skipna=True)) if s.notna().any() else np.nan,
                "n_missing": int(s.isna().sum()),
                "provisional_max": prov,
                "n_above_provisional_max": int((s > prov).sum(skipna=True)),
                "n_below_min": int((gv["violation"] == "below_min").sum()) if len(gv) else 0,
                "n_above_max": int((gv["violation"] == "above_max").sum()) if len(gv) else 0,
                "n_non_integer": int((gv["violation"] == "non_integer").sum()) if len(gv) else 0,
                "n_invalid_category": int((gv["violation"] == "invalid_category").sum()) if len(gv) else 0,
                "n_non_finite": int((gv["violation"] == "non_finite").sum()) if len(gv) else 0,
                "rule_status": None if rule is None else rule.status,
                "rule_source": None if rule is None else rule.source,
            }
        )
    return pd.DataFrame(records)


def assign_duplicate_group_ids(X: pd.DataFrame) -> np.ndarray:
    """Identical feature rows (including NaN pattern) share a duplicate_group_id."""
    filled = X.astype("string").fillna("<NA>")
    keys = filled.apply(lambda row: "|".join(row.tolist()), axis=1)
    codes, _ = pd.factorize(keys, sort=True)
    return codes.astype(int)


def feature_columns(parsed: pd.DataFrame) -> list[str]:
    return [c for c in parsed.columns if c not in ("source_row_id", "stress_level")]


def feature_values_differ(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise difference; matching NaN patterns count as equal."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    both_missing = np.isnan(a) & np.isnan(b)
    return ~both_missing & (a != b)


def assert_finite_or_nan(X: pd.DataFrame) -> None:
    """sklearn can impute NaN but not inf / other non-finite values."""
    arr = X.to_numpy(dtype=float)
    bad = ~np.isfinite(arr) & ~np.isnan(arr)
    if bad.any():
        rows, cols = np.where(bad)
        colname = X.columns[int(cols[0])]
        rid = rows[0]
        raise ValueError(
            f"non-finite value in feature {colname!r} at included-row index {int(rid)} "
            f"(value={arr[rid, cols[0]]!r}); refusing to silently replace it"
        )


def label_conflict_table(parsed: pd.DataFrame) -> pd.DataFrame:
    """Rows that share a feature group with more than one distinct valid label.

    Uses the same feature-group definition as ``assign_duplicate_group_ids``.
    Missing or illegal labels are not counted as a valid class; every member of
    a conflicting group is reported (including rows with missing/illegal labels).
    This is an audit table, not a confirmed mislabel list.
    """
    cols = feature_columns(parsed)
    gids = assign_duplicate_group_ids(parsed[cols])
    rows: list[dict] = []
    for gid in np.unique(gids):
        members = parsed.loc[gids == gid]
        valid: list[int] = []
        n_invalid = 0
        for v in members["stress_level"]:
            if stress_label_exclusion_reason(v if not pd.isna(v) else None) is None:
                valid.append(int(v))
            else:
                n_invalid += 1
        distinct = sorted(set(valid))
        if len(distinct) <= 1:
            continue
        counts = {str(k): valid.count(k) for k in distinct}
        count_text = ",".join(f"{k}:{counts[str(k)]}" for k in distinct)
        for _, rec in members.iterrows():
            raw_label = rec["stress_level"]
            rows.append(
                {
                    "conflict_group_id": int(gid),
                    "source_row_id": int(rec["source_row_id"]),
                    "stress_level": (None if pd.isna(raw_label) else float(raw_label)),
                    "group_size": int(len(members)),
                    "n_distinct_valid_labels": int(len(distinct)),
                    "valid_label_counts": count_text,
                    "n_invalid_or_missing_labels": int(n_invalid),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "conflict_group_id",
                "source_row_id",
                "stress_level",
                "group_size",
                "n_distinct_valid_labels",
                "valid_label_counts",
                "n_invalid_or_missing_labels",
            ]
        )
    return pd.DataFrame(rows)


def invalid_or_missing_label_table(parsed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for _, rec in parsed.iterrows():
        v = rec["stress_level"]
        reason = stress_label_exclusion_reason(None if pd.isna(v) else v)
        if reason:
            rows.append(
                {
                    "source_row_id": int(rec["source_row_id"]),
                    "stress_level": None if pd.isna(v) else float(v),
                    "reason": reason,
                }
            )
    if not rows:
        return pd.DataFrame(columns=["source_row_id", "stress_level", "reason"])
    return pd.DataFrame(rows)


def out_of_range_table(parsed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    feature_cols = [c for c in parsed.columns if c not in ("source_row_id", "stress_level")]
    for _, row in parsed.iterrows():
        for c in feature_cols:
            v = row[c]
            if pd.isna(v):
                continue
            prov = PROVISIONAL_MAX.get(c, DEFAULT_FEATURE_MAX)
            if float(v) > prov:
                rows.append(
                    {
                        "source_row_id": int(row["source_row_id"]),
                        "column": c,
                        "value": float(v),
                        "provisional_max": prov,
                        "note": "provisional_bound_not_data_dictionary",
                    }
                )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class PreparedData:
    X: pd.DataFrame
    y: np.ndarray
    source_row_ids: np.ndarray
    feature_names: list[str]
    exclusions: pd.DataFrame
    parse_issues: pd.DataFrame
    field_checks: pd.DataFrame
    csv_sha256_raw: str
    csv_sha256_normalized_lf: str
    scope: AnalysisScope
    raw: pd.DataFrame
    parsed: pd.DataFrame
    duplicate_group_ids: np.ndarray
    out_of_range: pd.DataFrame
    domain_violations: pd.DataFrame
    domain_violation_summary: dict

    @property
    def csv_sha256(self) -> str:
        return self.csv_sha256_raw


def prepare_data(
    csv_path: Path,
    scope: AnalysisScope = "primary",
    *,
    tail_quarantine_from: int = 1100,
    sensitivity_policy: SensitivityPolicy = "raw",
    raw: pd.DataFrame | None = None,
    parsed: pd.DataFrame | None = None,
    parse_issues: pd.DataFrame | None = None,
) -> PreparedData:
    csv_path = Path(csv_path).resolve()
    if raw is None:
        raw = load_raw_csv(csv_path)
    if parsed is None or parse_issues is None:
        parsed, parse_issues = parse_dataset(raw)

    quarantine_oor = scope == "sensitivity" and sensitivity_policy == "quarantine_out_of_range"
    exclusions = build_exclusions(
        parsed,
        scope,
        tail_quarantine_from=tail_quarantine_from,
        quarantine_out_of_range=quarantine_oor,
    )
    frame = select_analysis_frame(parsed, exclusions)

    valid_mask = [
        stress_label_exclusion_reason(row["stress_level"]) is None
        for _, row in frame.iterrows()
    ]
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
    X = X.reset_index(drop=True)
    dup_ids = assign_duplicate_group_ids(X)
    violations = domain_violations_table(parsed)
    assert_finite_or_nan(X)

    return PreparedData(
        X=X,
        y=y,
        source_row_ids=ids.astype(int),
        feature_names=list(X.columns),
        exclusions=exclusions,
        parse_issues=parse_issues,
        field_checks=field_checks(parsed),
        csv_sha256_raw=sha256_file(csv_path),
        csv_sha256_normalized_lf=sha256_file_normalized_lf(csv_path),
        scope=scope,
        raw=raw,
        parsed=parsed,
        duplicate_group_ids=dup_ids,
        out_of_range=out_of_range_table(parsed),
        domain_violations=violations,
        domain_violation_summary=domain_violation_counts(violations, parsed),
    )


# Audit-only: intentional label-driven zero map (never default training).
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
    labels = np.asarray(labels)
    for col, mapping in LEAK_MAP.items():
        if col not in Xm.columns:
            continue
        z = (Xm[col] == 0).to_numpy()
        if not z.any():
            continue
        Xm.loc[z, col] = [mapping[int(v)] for v in labels[z]]
    return Xm
