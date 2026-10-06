# Stress Level Detection (research / audit baseline)

Multiclass classification of student **stress_level** (0 / 1 / 2) from questionnaire-style features. This repository is a **reproducible research baseline**, not a clinical product or deployed detector.

## Research boundary

Current results support **prediction of the `stress_level` column that is already in this CSV**, plus a **leakage audit of the training code**. They do not establish how those labels were generated, who the respondents were, or whether the same scores would hold on new data.

- **Not claimed:** causal effects, clinical validity, synthetic-data status, a theoretical performance ceiling, or external generalization.
- Removing label-driven zero imputation from the default path removes that **code-level** leak. It does **not** prove that `stress_level` is independent of the input fields (the label could still have been derived from them).
- Rows 0–1099 are an **audit convention** for the primary scope, not a confirmed target population.
- `grouped_duplicates` groups **identical feature vectors**, not known subjects.
- Conflict groups, domain-rule violations and persistent-error flags are audit findings, **not** confirmed mislabels.
- **Invalid historical scores:** Old Notebook SVM figures near **94%** used **label-driven zero imputation** and other leakage; old LR figures near **88%** are **not** verified benchmarks.
- Experiment B (nested hyperparameter search) is still a stub. There is no external validation set. See `RESEARCH_TODO.md`.

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

`tests/test_oof_consistency.py` recomputes fold- and repeat-level metrics from the saved OOF predictions with scikit-learn and checks coverage and confusion matrices. It replaces `test_oof_metrics_match_fold_means` (removed in `314c444`), which lacked substantive consistency assertions: it only checked that the fold mean in `fold_scores.csv` matched `results.json` and that the pooled OOF macro-F1 lay in [0, 1]. It never required pooled OOF F1 to equal the fold mean (the `314c444` commit message describes it that way inaccurately).

## Smoke vs full audit (Experiment A)

Fixed-parameter models, shared **RepeatedStratifiedKFold** (default 5×10, `--seed` default 0), primary metric **macro-F1**. Pipelines impute/scale **inside training folds only**.

All commands below are run from the repository root with the project environment active (or prefixed with `uv run`).

The output directory is **required** and must be new or empty. The program refuses a non-empty directory (including `results/audit_full`) before writing anything. There is no `--overwrite`. A failed run may leave `run_status.json` as `in_progress` plus partial CSVs; that is not a completed audit — rerun into a new directory. `results.json` is written only at the end.

```bash
# Smoke — choose a new empty directory; do not omit the output path
python scripts/run_audit.py StressLevelDataset_original.csv /tmp/audit_smoke --smoke --config configs/audit_primary.yaml

# Seed check
python scripts/run_audit.py StressLevelDataset_original.csv /tmp/audit_seed123 --smoke --seed 123

# Sensitivity smoke (group-aware duplicates)
python scripts/run_audit.py StressLevelDataset_original.csv /tmp/audit_sens --smoke --scope sensitivity --sensitivity-policy grouped_duplicates
```

### Formal full run (clean working tree only)

The audit runs only if git succeeds, `git status --porcelain` prints nothing (no modified, staged or untracked files), and the output directory does not exist yet. The program itself also refuses a non-empty output directory. Each run writes to a new directory; `results/audit_full`, `results/audit_full_d12aa67_clean` and `results/audit_full_4865bb3_clean` are never overwritten.

PowerShell:

```powershell
& {
  $ErrorActionPreference = "Stop"
  $status = git status --porcelain
  if ($LASTEXITCODE -ne 0) { throw "git status failed" }
  if ($status) { throw "working tree is not clean:`n$($status -join "`n")" }
  $sha = git rev-parse --short HEAD
  if ($LASTEXITCODE -ne 0) { throw "git rev-parse failed" }
  $out = "results/audit_full_${sha}_clean"
  if (Test-Path $out) { throw "output directory already exists: $out" }
  python scripts/run_audit.py StressLevelDataset_original.csv $out --n-perm 100 --seed 0 --config configs/audit_primary.yaml
  if ($LASTEXITCODE -ne 0) { throw "audit failed" }
}
```

bash:

```bash
(
  set -euo pipefail
  status=$(git status --porcelain)
  if [ -n "$status" ]; then printf 'working tree is not clean:\n%s\n' "$status" >&2; exit 1; fi
  sha=$(git rev-parse --short HEAD)
  out="results/audit_full_${sha}_clean"
  if [ -e "$out" ]; then echo "output directory already exists: $out" >&2; exit 1; fi
  python scripts/run_audit.py StressLevelDataset_original.csv "$out" --n-perm 100 --seed 0 --config configs/audit_primary.yaml
)
```

Afterwards `results.json["git"]` should show `"dirty": false` and `"git_state_capture": "run_start"`.

### Settings precedence

For `n_splits` / `n_repeats` / `n_perm`:

1. `--n-splits` / `--n-repeats` / `--n-perm` on the command line (applies in any mode);
2. smoke defaults: 3 splits × 1 repeat and **5** permutations; YAML values are not applied, and the requested vs effective values are recorded;
3. config `eval.n_splits` / `eval.n_repeats` / `n_perm`;
4. full defaults: 5 splits × 10 repeats and 100 permutations.

Smoke does **not** override an explicit CLI `--n-perm` (or `--n-splits` / `--n-repeats`). It is therefore not true that every CLI flag always beats smoke, or that smoke always beats every CLI flag: smoke replaces unspecified eval/perm settings only.

Other settings: CLI flag > config value > default. `scope` is only `primary` or `sensitivity` (typos are rejected, not coerced). `sensitivity_policy` is only `raw`, `quarantine_out_of_range`, or `grouped_duplicates`. Integers (`seed`, `n_perm`, limits) reject bools, floats and numeric strings. `n_perm` must be ≥ 1; 0 is not treated as “skip the experiment”. Seed plus `n_repeats-1` must stay inside the sklearn/numpy seed range `[0, 2**32)`. Unknown keys, a conflicting `eval.seed`, and a non-default `persistent_error_threshold` are rejected **before** the output directory is created. Loading `--config` requires PyYAML; empty files and non-mapping roots (`[]`, `false`) are errors.

`results.json["config"]` stores the effective values; `config.settings_resolution.sources` records where each value came from.

Outputs: `results.json`, `folds.csv`, `fold_scores.csv`, `repeat_scores.csv`, `oof_predictions.csv`, permutations, parse_issues, requirements-lock (non-empty), hashes for lock + code manifest.

`grouped_duplicates` uses `StratifiedGroupKFold` for **outer and inner** splits (`inner_folds.csv` records group isolation). The primary permutation test (row-wise `StratifiedKFold(5)`) is unchanged. Under `grouped_duplicates` the permutation experiment is **`not_available`**: identical-feature groups are not confirmed subjects, and a group-level label permutation has not been justified. Do not read a missing `permutations.csv` in that mode as a completed test.

**Experiment B** (nested LR/SVM/RF/XGB search): still a **stub** in `scripts/run_benchmark.py` — not completed. That script also requires an explicit empty output directory.

## Run provenance

`results.json["git"]` is captured **at run start**, before the output directory is created or any artifact is written (`git_state_capture: "run_start"`). Writing results under `results/` therefore does not turn a clean starting tree into `dirty: true`, while uncommitted edits or untracked files that existed at start are still recorded as `dirty: true`. If git cannot be queried, `commit`/`dirty` are `"unknown"` and `available: false` — never reported as clean.

## How metrics are aggregated

Two summaries are reported; both are legitimate, they answer different questions:

| Field / file | Unit | Definition |
|---|---|---|
| `models` / `fold_scores.csv` | outer fold | Each metric is computed on one validation fold; mean/std over all 50 folds (5 × 10), std with ddof=1. |
| `models_repeat` / `repeat_scores.csv` | repeat | For each repeat, the 5 validation folds are merged by `source_row_id` into one complete OOF vector (every included sample exactly once); accuracy, macro-F1, per-class precision/recall/F1/support, the 3×3 confusion matrix (label order 0, 1, 2) and the 0↔2 error numerator/denominator/rate are recomputed from that pooled vector. Mean/std over the 10 repeats, std with ddof=1 (`null` if only one repeat). |

- **Fold std** describes fold-to-fold score variation.
- **Repeat std** describes how much the pooled OOF score moves when the same fixed dataset is re-partitioned. Repeats reuse the same 1098 rows and are **not** independent samples of new datasets, so repeat std cannot be used as a confidence interval for generalization. Fold std cannot either.
- Pooled OOF macro-F1 is generally not equal to the mean of fold macro-F1s.
- Confusion matrices summed over repeats count each sample once per repeat (10× in total); that total is not a number of independent samples.
- Paired model comparisons (`paired`) use per-fold paired differences with a Nadeau–Bengio-style corrected variance and df = 49. The result is an **approximate corrected interval**, not an exact one. An interval that contains 0 means the difference is not resolved under this protocol; it does **not** show the models are equivalent.

## Formal full audit (`results/audit_full_4865bb3_clean`)

Run from a clean working tree at commit `4865bb3` (`git.dirty: false`, `git_state_capture: "run_start"`), seed=0, 5 folds × 10 repeats from `configs/audit_primary.yaml`, 100 permutations, 1098 primary-scope rows. Fold-level metrics, OOF predictions, fold membership, exclusions, inner candidates and permutations are identical to `results/audit_full` and to `results/audit_full_d12aa67_clean`.

| Model | fold-level macro-F1 (50 folds) | repeat-level macro-F1 (10 repeats) | repeat-level accuracy | repeat-level 0↔2 error rate |
|-------|------|------|------|------|
| Dummy (most frequent) | 0.169 ± 0.001 | 0.169 ± 0.000 | 0.340 ± 0.000 | 0.496 ± 0.000 |
| LR, all features | 0.886 ± 0.017 | 0.886 ± 0.003 | 0.885 ± 0.003 | 0.074 ± 0.004 |
| LR, no 4 psych features | 0.883 ± 0.019 | 0.883 ± 0.003 | 0.883 ± 0.003 | 0.078 ± 0.004 |
| Linear SVM, C=0.2 | 0.878 ± 0.017 | 0.878 ± 0.004 | 0.878 ± 0.004 | 0.082 ± 0.006 |
| Random Forest | 0.878 ± 0.020 | 0.878 ± 0.006 | 0.878 ± 0.006 | 0.077 ± 0.006 |
| Nested single-feature tree (depth 3) | 0.872 ± 0.020 | 0.872 ± 0.006 | 0.870 ± 0.006 | 0.094 ± 0.009 |

Values are mean ± std (ddof=1); see "How metrics are aggregated" for what each std does and does not mean. The 0↔2 rate denominator is the number of samples with true label 0 or 2.

Paired macro-F1 differences (approximate corrected 95% intervals, df = 49):

| Comparison | mean difference | interval |
|---|---|---|
| LR all − nested single-feature tree | 0.0131 | [−0.0017, 0.0280] |
| LR all − Random Forest | 0.0077 | [−0.0064, 0.0217] |
| LR all − LR without psych features | 0.0026 | [−0.0085, 0.0138] |

All three intervals contain 0: the differences are not resolved under this protocol, which does not mean the models are equivalent.

Permutation check (linear SVM, one 5-fold CV, macro-F1): with the clean pipeline the observed score is 0.877 against a label-permutation null of 0.320 ± 0.020 (max 0.372). The audit-only label-driven zero-map counterexample scores 0.932, which illustrates the leakage this repository removed; it is not a valid result.

## Historical full audit (`results/audit_full`)

The committed tree under `results/audit_full/` is **historical evidence** produced from git commit `cd80fc1` with a **dirty** working tree (pre-`5ee2152` packaging). Do **not** treat its `env.script_sha256` as a full code manifest. It predates `repeat_scores.csv` / `models_repeat`.

Fold-level macro-F1 (mean ± std over 50 outer folds, seed=0):

| Model | macro-F1 |
|--------|----------|
| Dummy (most frequent) | 0.169 |
| LR, all features | **0.886 ± 0.017** |
| Linear SVM, C=0.2 | 0.878 ± 0.017 |
| Random Forest | 0.878 ± 0.020 |
| LR, no 4 psych features | 0.883 ± 0.019 |
| Nested single-feature tree (depth 3) | 0.872 ± 0.020 |

## Earlier verification run (not versioned)

A run on commit `066ea2b` with uncommitted changes (`dirty: true`, correctly recorded) was briefly committed as `results/audit_full_066ea2b_dirty_repeatlevel` in `314c444` and later moved out of the repository, because a dirty run is not a formal result. Its fold- and repeat-level results are identical to the formal run above.

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
- Explanation stability, learning curves, external validation.
