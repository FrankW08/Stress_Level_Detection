"""Provisional field schema for reporting only.

Rules come from DATA_CARD.md and the existing PROVISIONAL_MAX / 0–5 Likert
convention in this repository. They are not a confirmed instrument dictionary
and do not, by themselves, change primary inclusion or model inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RuleStatus = Literal["provisional"]

LIKERT_0_5 = (
    "headache",
    "sleep_quality",
    "breathing_problem",
    "noise_level",
    "living_conditions",
    "safety",
    "basic_needs",
    "academic_performance",
    "study_load",
    "teacher_student_relationship",
    "future_career_concerns",
    "peer_pressure",
    "extracurricular_activities",
    "bullying",
)

SOURCE_DATA_CARD = "DATA_CARD.md / repository provisional convention (not a confirmed data dictionary)"
SOURCE_PROVISIONAL_MAX = "existing PROVISIONAL_MAX in data.py (inferred from bulk data; not a confirmed data dictionary)"


@dataclass(frozen=True)
class FieldRule:
    name: str
    lower_bound: float | None
    upper_bound: float | None
    integer_required: bool
    allowed_values: frozenset[float] | None
    source: str
    status: RuleStatus = "provisional"


def _scale(name: str, upper: float) -> FieldRule:
    return FieldRule(
        name=name,
        lower_bound=0.0,
        upper_bound=upper,
        integer_required=False,
        allowed_values=None,
        source=SOURCE_PROVISIONAL_MAX,
    )


def _likert(name: str, upper: float = 5.0) -> FieldRule:
    return FieldRule(
        name=name,
        lower_bound=0.0,
        upper_bound=upper,
        integer_required=True,
        allowed_values=None,
        source=SOURCE_DATA_CARD,
    )


def build_provisional_schema() -> dict[str, FieldRule]:
    rules: dict[str, FieldRule] = {
        "anxiety_level": _scale("anxiety_level", 21),
        "self_esteem": _scale("self_esteem", 30),
        "mental_health_history": FieldRule(
            name="mental_health_history",
            lower_bound=0.0,
            upper_bound=1.0,
            integer_required=True,
            allowed_values=frozenset({0.0, 1.0}),
            source=SOURCE_DATA_CARD,
        ),
        "depression": _scale("depression", 27),
        "blood_pressure": FieldRule(
            name="blood_pressure",
            lower_bound=0.0,
            upper_bound=3.0,
            integer_required=True,
            allowed_values=None,
            source=SOURCE_PROVISIONAL_MAX,
        ),
        "social_support": FieldRule(
            name="social_support",
            lower_bound=0.0,
            upper_bound=3.0,
            integer_required=True,
            allowed_values=None,
            source=SOURCE_PROVISIONAL_MAX,
        ),
        "stress_level": FieldRule(
            name="stress_level",
            lower_bound=0.0,
            upper_bound=2.0,
            integer_required=True,
            allowed_values=frozenset({0.0, 1.0, 2.0}),
            source=SOURCE_DATA_CARD,
        ),
    }
    for name in LIKERT_0_5:
        rules[name] = _likert(name)
    return rules


PROVISIONAL_SCHEMA = build_provisional_schema()


def rule_for(column: str) -> FieldRule | None:
    return PROVISIONAL_SCHEMA.get(column)
