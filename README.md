# Stress Level Detection (research / audit baseline)

Multiclass classification of student **stress_level** (0 / 1 / 2) from questionnaire-style features. This repository is a **reproducible research baseline**, not a clinical product or deployed detector.

## Research boundary

- **Primary question:** How well do standard models predict `stress_level` under nested preprocessing and fixed audit protocols?
- **Not claimed:** causal effects, synthetic-data status, theoretical performance ceiling, or external validity.
- **Invalid historical scores:** Notebook **~94% SVM** used **label-driven zero imputation** and other leakage; **~88% LR** from the old notebook is **not** a verified benchmark.

## Data

- Source file: `StressLevelDataset_original.csv` (1121 rows; **never overwrite**).
- **SHA-256 (local run, 2026-10-05):** `4fc99679af3bffc701129d09ed072bfbce79070073a2d5da55aff00b7eb50473`  
  If this differs from another environment, compare file bytes and git revision before comparing metrics.
- **Primary scope:** `source_row_id` 0–1099, valid labels only → **1098 rows** (21 tail rows quarantined by audit convention; 2 missing labels).
- **Sensitivity scope:** all rows with valid labels (`--scope sensitivity`) — configured but not run in the default full audit below.
- See `DATA_CARD.md` for provisional schema notes.

## Install

```bash
cd Stress_Level_Detection
python -m pip install -e ".[dev]"
```

## Tests

```bash
python -m pytest tests -q
```

## Smoke vs full audit (experiment A)

Fixed-parameter models, shared **RepeatedStratifiedKFold 5×10, seed=0**, macro-F1 primary metric. Pipelines impute/scale **inside training folds only**.

```bash
# Smoke (~seconds): 3 folds × 1 repeat, 5 permutations
python scripts/run_audit.py StressLevelDataset_original.csv results/audit_smoke --smoke

# Full (tens of seconds on this machine): 50 outer folds, 100 permutations
python scripts/run_audit.py StressLevelDataset_original.csv results/audit_full --n-perm 100
```

Outputs under the chosen directory: `results.json`, `folds.csv`, `fold_scores.csv`, `inner_candidates.csv`, `oof_predictions.csv`, `permutations.csv`, `per_sample_errors.csv`, `exclusions.csv`, `field_checks.csv`, `parse_issues.csv`, duplicate audit CSVs, `requirements-lock.txt`.

**Experiment B** (nested hyperparameter benchmark for LR/SVM/RF/XGB): stub status in `scripts/run_benchmark.py` — not completed this round.

## Actual full audit results (this workspace)

Environment: Python **3.12.5**, pandas **2.2.3**, scikit-learn **1.7.0**, numpy **2.2.6**, scipy **1.15.3**. Git: `cd80fc1` (**dirty** after local changes).

| Model | macro-F1 (mean ± std, 50 outer folds) |
|--------|----------------------------------------|
| Dummy (most frequent) | 0.169 (no std — constant) |
| LR, all features | **0.886 ± 0.017** |
| Linear SVM, C=0.2 | 0.878 ± 0.017 |
| Random Forest | 0.878 ± 0.020 |
| LR, no 4 psych features | 0.883 ± 0.019 |
| Nested single-feature tree (depth 3) | 0.872 ± 0.020 |

Nested single-feature picks: `blood_pressure` 33, `future_career_concerns` 11, `anxiety_level` 3, `sleep_quality` 3 (50 folds).

**Paired macro-F1 differences (LR_all − other), Nadeau–Bengio-style approx. 95% CI, df=49:**

| Comparison | Mean Δ | Approx. CI | Share folds LR higher |
|------------|--------|------------|------------------------|
| LR − nested single | +0.013 | [−0.002, +0.028] | 82% |
| LR − RF | +0.008 | [−0.006, +0.022] | 70% |
| LR − no psych | +0.003 | [−0.009, +0.014] | 60% |

**Permutation (label shuffling, SVM C=0.2):** observed clean **0.877**, leaky (label-zero map) **0.932**; paired null Δ mean **0.040** (see `permutations.csv`). **96%** of permutations had leaky > clean.

These numbers align with the second-version reference script (`audit_repro.py` / `new_README.md`) up to minor float formatting; **CSV hash in that document (`05145e0a…`) does not match this file** — treat attachment numbers as same protocol, different byte identity until the CSV is reconciled.

## Leakage fixes

- Removed default **label-driven zero replacement** in the notebook (`RUN_LEAKY_SVM_DEMO = False`); audit-only path in `stress_detection.data.LEAK_MAP`.
- Training uses **sklearn Pipelines** (median impute + scale + model for LR/SVM).
- **`stress_level` never enters features**; validation predictions do not take `y_valid`.
- Tests: `tests/test_validation.py`, `tests/test_model_persistence.py`.

## Notebook

`Stress Level Classification.ipynb` — EDA and legacy model cells retained with fixes (variable names, `fillna` assignment, classification report class keys, EDA bins, leakage guard). **Authoritative metrics:** last cell calls `run_audit_cv` / package API, or run `scripts/run_audit.py`.

Clear stale outputs and run top-to-bottom in a fresh kernel after `pip install -e .`.

## Roadmap (not done here)

- Sensitivity run including tail 21 rows; line-by-line source provenance.
- Full experiment B nested search with shared outer folds.
- Group-wise missingness, explanation stability, learning curves, external validation.

## Reference materials used

- `C:/Users/Boush/Downloads/audit_repro.py` — design reference (integrated into `src/stress_detection`).
- `C:/Users/Boush/Downloads/new_README.md` — expected reporting template (numbers verified against local `results/audit_full/results.json` where CSV matches).

No `AGENTS.md` existed upstream; see `AGENTS.md` in this repo for agent constraints.
