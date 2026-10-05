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

`tests/test_oof_consistency.py` recomputes fold- and repeat-level metrics from the saved OOF predictions with scikit-learn and checks coverage and confusion matrices. It replaces `test_oof_metrics_match_fold_means` (removed in `314c444`), which lacked substantive consistency assertions: it only checked that the fold mean in `fold_scores.csv` matched `results.json` and that the pooled OOF macro-F1 lay in [0, 1]. It never required pooled OOF F1 to equal the fold mean (the `314c444` commit message describes it that way inaccurately).

## Smoke vs full audit (Experiment A)

Fixed-parameter models, shared **RepeatedStratifiedKFold** (default 5×10, `--seed` default 0), primary metric **macro-F1**. Pipelines impute/scale **inside training folds only**.

All commands below are run from the repository root with the project environment active (or prefixed with `uv run`).

```bash
# Smoke — output directory is used exactly as given
python scripts/run_audit.py StressLevelDataset_original.csv /tmp/audit_smoke --smoke --config configs/audit_primary.yaml

# Seed check
python scripts/run_audit.py StressLevelDataset_original.csv /tmp/audit_seed123 --smoke --seed 123

# Sensitivity smoke (group-aware duplicates)
python scripts/run_audit.py StressLevelDataset_original.csv /tmp/audit_sens --smoke --scope sensitivity --sensitivity-policy grouped_duplicates
```

### Formal full run (clean working tree only)

The audit runs only if git succeeds, `git status --porcelain` prints nothing (no modified, staged or untracked files), and the output directory does not exist yet. Each run writes to a new directory; `results/audit_full` is never overwritten.

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

For outer CV `n_splits` / `n_repeats`:

1. `--n-splits` / `--n-repeats` on the command line (applies in any mode);
2. `--smoke` (or `smoke: true` in the config): 3 splits × 1 repeat; config `eval.*` values are not applied, and a note is printed and recorded;
3. config `eval.n_splits` / `eval.n_repeats`;
4. defaults: 5 splits × 10 repeats.

Other settings: CLI flag > config value > default. Unknown config keys, non-integer or out-of-range `eval` values, an `eval.seed` that disagrees with `seed`, and a non-default `persistent_error_threshold` (not configurable) are rejected before any output is written. `results.json["config"]["eval"]` holds the effective values; `config.settings_resolution` records the source of each `eval` value, any notes, and the config file path with its LF-normalized SHA-256.

Outputs: `results.json`, `folds.csv`, `fold_scores.csv`, `repeat_scores.csv`, `oof_predictions.csv`, permutations, parse_issues, requirements-lock (non-empty), hashes for lock + code manifest.

**Experiment B** (nested LR/SVM/RF/XGB search): still a **stub** in `scripts/run_benchmark.py` — not completed.

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

## Verification run with repeat-level summary (not versioned)

Same protocol (seed=0, 5 folds × 10 repeats, 100 permutations), run on commit `066ea2b` with **uncommitted** changes at run start (`dirty: true`, correctly recorded). It was briefly committed as `results/audit_full_066ea2b_dirty_repeatlevel` in `314c444` and has since been moved out of the repository as a local verification artifact, because a dirty run is not a formal result. Fold-level metrics, paired differences, permutation results, OOF predictions, fold membership and included rows are identical to `results/audit_full`.

Repeat-level pooled OOF metrics (mean ± std over 10 repeats):

| Model | macro-F1 | accuracy | 0↔2 error rate |
|-------|----------|----------|----------------|
| Dummy (most frequent) | 0.169 ± 0.000 | 0.340 ± 0.000 | 0.496 ± 0.000 |
| LR, all features | 0.886 ± 0.003 | 0.885 ± 0.003 | 0.074 ± 0.004 |
| Linear SVM, C=0.2 | 0.878 ± 0.004 | 0.878 ± 0.004 | 0.082 ± 0.006 |
| Random Forest | 0.878 ± 0.006 | 0.878 ± 0.006 | 0.077 ± 0.006 |
| LR, no 4 psych features | 0.883 ± 0.003 | 0.883 ± 0.003 | 0.078 ± 0.004 |
| Nested single-feature tree (depth 3) | 0.872 ± 0.006 | 0.870 ± 0.006 | 0.094 ± 0.009 |

The 0↔2 rate denominator is the number of samples with true label 0 or 2. A formal result should be regenerated from a clean commit (command below).

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
