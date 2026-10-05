# Stress Level Detection (research / audit baseline)

Multiclass classification of student **stress_level** (0 / 1 / 2) from questionnaire-style features. This repository is a **reproducible research baseline**, not a clinical product or deployed detector.

## Research boundary

- **Primary question:** How well do standard models predict `stress_level` under nested preprocessing and fixed audit protocols?
- **Not claimed:** causal effects, synthetic-data status, theoretical performance ceiling, or external validity.
- **Invalid historical scores:** Old Notebook SVM figures near **94%** used **label-driven zero imputation** and other leakage; old LR figures near **88%** are **not** verified benchmarks.

## Data and line endings

- Source file: `StressLevelDataset_original.csv` (1121 rows; **never overwrite**).
- Git stores the CSV with **LF** newlines. With `core.autocrlf=true` on Windows, a checkout may use **CRLF** bytes on disk.
  - **Normalized-LF SHA-256** (content after CRLF→LF): `05145e0a27395e85f6ed062d6f89f99351bdd16f8fa7dc5e60243bd1083dab26`
  - **Raw checkout SHA-256** may be `4fc99679…` on Windows CRLF checkouts — **same records**, different newline bytes.
- Each `results.json` records both `csv_sha256_raw` and `csv_sha256_normalized_lf`.
- **Primary scope:** `source_row_id` 0–1099, valid labels only → **1098 rows**.
- **Sensitivity scope:** `--scope sensitivity` with `--sensitivity-policy` (`raw` | `quarantine_out_of_range` | `grouped_duplicates`). Provisional range bounds are **not** a confirmed data dictionary.
- See `DATA_CARD.md`.

## Install

```bash
cd Stress_Level_Detection
# Recommended
uv sync --extra dev --extra benchmark
# Or
python -m pip install -e ".[dev,benchmark]"
```

`uv.lock` pins the environment when using uv.

## Tests

```bash
python -m pytest -q
```

Tests write only under pytest `tmp_path` (no new files under `results/`).

## Smoke vs full audit (Experiment A)

Fixed-parameter models, shared **RepeatedStratifiedKFold** (default 5×10, `--seed` default 0), primary metric **macro-F1**. Pipelines impute/scale **inside training folds only**.

```bash
# Smoke — output directory is used exactly as given
python scripts/run_audit.py StressLevelDataset_original.csv /tmp/audit_smoke --smoke --config configs/audit_primary.yaml

# Seed check
python scripts/run_audit.py StressLevelDataset_original.csv /tmp/audit_seed123 --smoke --seed 123

# Sensitivity smoke (group-aware duplicates)
python scripts/run_audit.py StressLevelDataset_original.csv /tmp/audit_sens --smoke --scope sensitivity --sensitivity-policy grouped_duplicates

# Full (after a clean commit; write to a NEW directory — do not overwrite results/audit_full)
python scripts/run_audit.py StressLevelDataset_original.csv results/audit_full_clean_COMMIT --n-perm 100 --seed 0 --config configs/audit_primary.yaml
```

CLI flags override `--config` YAML. Outputs: `results.json`, folds, OOF, permutations, parse_issues, requirements-lock (non-empty), hashes for lock + code manifest.

**Experiment B** (nested LR/SVM/RF/XGB search): still a **stub** in `scripts/run_benchmark.py` — not completed.

## Historical full audit (`results/audit_full`)

The committed tree under `results/audit_full/` is **historical evidence** produced from git commit `cd80fc1` with a **dirty** working tree (pre-`5ee2152` packaging). Do **not** treat its `env.script_sha256` as a full code manifest.

| Model | macro-F1 (mean ± std, 50 outer folds, seed=0) |
|--------|-----------------------------------------------|
| Dummy (most frequent) | 0.169 |
| LR, all features | **0.886 ± 0.017** |
| Linear SVM, C=0.2 | 0.878 ± 0.017 |
| Random Forest | 0.878 ± 0.020 |
| LR, no 4 psych features | 0.883 ± 0.019 |
| Nested single-feature tree (depth 3) | 0.872 ± 0.020 |

After landing this code on a **clean** commit, regenerate a new full audit into a **new** directory and compare macro-F1 to the table above (float noise only expected for seed=0 primary).

## Leakage controls

- Default paths **never** apply label-driven zero replacement; audit-only helper: `stress_detection.data.apply_label_driven_zero_map`.
- Label-invariance tests: clean adapter ignores `y_valid`; deliberate leaky adapter must fail.
- `stress_level` is never a feature; production predict takes `X` only.

## Notebook

`Stress Level Classification.ipynb` — cleared outputs; uses `prepare_data` + package pipelines. Authority: `run_audit.py`. Optional XGBoost via `[benchmark]`.

```bash
python -m jupyter nbconvert --to notebook --execute "Stress Level Classification.ipynb" --output /tmp/executed_stress_nb.ipynb
```

## Roadmap (not done)

- Source-file lineage / label generation audit.
- Full Experiment B nested search.
- Clean-commit full audit regeneration into a new `results/` folder.
- Explanation stability, learning curves, external validation.
