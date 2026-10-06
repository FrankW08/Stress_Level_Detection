# Data card (provisional)

## Files

- `StressLevelDataset_original.csv` — **immutable** source; SHA-256 recorded in each audit `results.json`.
- Parsed copies are never written back to this path.

## Record identity

- `source_row_id`: zero-based row index in the original CSV (column added in memory only).

## Target

- `stress_level`: ordinal labels **0, 1, 2** only.
- Missing or illegal labels are **excluded** from supervised training/evaluation (see `exclusions.csv`).

## Analysis scopes

| Scope | Rule |
|--------|------|
| **primary** | Rows `source_row_id` 0–1099, minus label exclusions. Tail rows ≥1100 quarantined by **audit convention** (not a confirmed target-population definition, and not proof of forgery). |
| **sensitivity** | All rows with valid labels; policies: `raw` (describe/mark; refuse CV if identical-feature duplicates cross folds), `quarantine_out_of_range` (**legacy / provisional**: exclude rows with any feature **above** the historical `PROVISIONAL_MAX` / default max 5; does not apply below-min or category rules), `grouped_duplicates` (StratifiedGroupKFold on identical-feature `duplicate_group_id` — **not** known subject IDs). |

The field schema in `src/stress_detection/schema.py` is **provisional**. It is used to **report** `below_min`, `above_max`, `non_integer`, `invalid_category` and `non_finite` cells. Missing values are counted separately and are not domain violations. A reported violation is **not** evidence that a row is fabricated or mislabelled, and these extra rules do **not** change primary inclusion or model inputs.

## Hashes

Audits record raw CSV SHA-256 and LF-normalized SHA-256 (see README line-ending note).


## Feature schema (provisional — verify against source)

Most Likert-style items use **0–5** unless noted. Zero means the lowest category on the instrument **only where confirmed**; otherwise treat as provisional.

| Column | Type (provisional) | Notes |
|--------|-------------------|--------|
| anxiety_level | continuous / scale | provisional max 21 |
| self_esteem | scale | provisional max 30 |
| mental_health_history | binary | 0/1 |
| depression | scale | provisional max 27 |
| blood_pressure | ordinal / categorical encoding | **non-monotonic** relation to stress; do not treat as mmHg |
| social_support | ordinal | provisional max 3 |
| headache, sleep_quality, … | Likert 0–5 | encoding is experimental config |

## Cleaning policy (current pipeline)

- Strip whitespace and outer quotes; map `Null`/empty to missing.
- **No** label-driven zero imputation in the default path.
- Imputation/scaling **inside sklearn Pipelines**, fit on training folds only.

## Open questions (unverified)

These are **not** established facts. Structured checklist: `docs/DATA_PROVENANCE.md`. Short list: `RESEARCH_TODO.md`.

- How `stress_level` was generated (self-report, sum of items, researcher coding, or something else).
- The original source file, collection protocol, and any ethics review.
- Whether any row identifies a unique person; identical feature groups are not confirmed subjects.
- Whether tail rows 1100–1120 share a common data-entry batch.
