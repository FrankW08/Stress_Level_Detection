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
| **primary** | Rows `source_row_id` 0–1099, minus label exclusions. Tail rows ≥1100 quarantined by **audit convention** (not proof of forgery). |
| **sensitivity** | All rows with valid labels; same feature schema checks. |

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

## Open questions

- Label generation process and primary source file lineage.
- Whether tail rows 1100–1120 share a common data-entry batch.
- Subject-level grouping / duplicate respondents.
