# Research plan

## Primary question

After removing label-driven preprocessing leakage, **how much stable predictive gain do multi-feature models retain over simple baselines when predicting the `stress_level` column already in this CSV?**

## Scope of claims

- **In scope:** prediction of the stored label; leakage audits; comparison of models on shared outer folds.
- **Not in scope until provenance is settled:** measuring real-world stress, diagnosing mental health, causal effects, or treating this table as a confirmed clinical cohort (`docs/DATA_PROVENANCE.md`).
- This CSV has already been used for multiple exploratory rounds (notebook, historical audits). Later experiments **must not be described as a pre-registered prospective confirmation**.
- Repeat-CV on this table is **internal** evidence. External validation is a different claim and needs a second dataset with matching fields, encodings, and label definition.

## Experiment A — fixed-parameter baseline (implemented)

Models: Dummy (most frequent), nested single-feature depth-3 tree, LR (all features), linear SVM (C=0.2), Random Forest, LR without four psych features.

Protocol: primary rows 0–1099 with valid labels (1098 rows); `RepeatedStratifiedKFold` 5×10, seed=0; impute/scale inside pipelines on training folds only. 3-class metrics use labels `[0,1,2]` and `zero_division=0`. Every outer and inner train/valid fold must contain all three classes.

Two legitimate summaries:

- **Fold-level:** mean/std over 50 outer validation folds (fold-to-fold fluctuation).
- **Repeat-level:** pooled OOF vector per repeat (each included row once), then mean/std over 10 repeats (partition fluctuation on the **same** 1098 rows). Repeat std is **not** a confidence interval for external generalization.

Paired comparisons use per-fold differences and a Nadeau–Bengio-style corrected interval (df = 49). Folds are not independent. An interval containing 0 does **not** mean models are equivalent.

Pre-specified A comparisons (already reported): LR-all vs nested-single, LR-all vs RF, LR-all vs LR-no-psych.

Formal results: `results/audit_full_4865bb3_clean` (clean tree at `4865bb3`). Do not overwrite.

## Experiment B — limited nested search (design only; not executed)

**Do not run this search in the current engineering pass.** `scripts/run_benchmark.py` writes the design JSON only.

### Shared outer folds and rows

Use the **same** Experiment A outer splits (seed=0, 5×10) and the same included rows. All B procedures (`b_lr`, `b_select`) and A baselines used in the four comparisons must be scored on those folds. Do not retune the outer seed if a fold looks inconvenient.

### Procedures (not a post-hoc winner)

| Procedure | What is selected | Where |
|-----------|------------------|--------|
| `b_lr` | Hyperparameters **inside LR only** | Inner CV on the current outer training fold |
| `b_select` | **Family and** that family's hyperparameters, among designed LR, linear SVM, and RF | The **same** inner splits and the **same** inner scores as the family searches |

`b_select` may pick a different family on different outer folds. That is intended. Outer validation is not used to choose family, C, tree settings, thresholds, features, or the search space.

Picking “the family with the highest mean outer score” and then pairing that family with `lr_all` is **exploratory ranking only**. Writing the comparison down in advance does **not** remove winner-selection bias if the winner is chosen from the same outer scores used in the test.

### Models and grids

Default B does **not** add XGBoost, deep models, or extra feature selection.

| Family | Pipeline (unchanged preprocessing) | Candidates **in tie-break order** |
|--------|--------------------------------------|-----------------------------------|
| LR | median impute → scale → `LogisticRegression(max_iter=5000)` | `C=0.1`, `1.0`, `10.0` |
| linear SVM | median impute → scale → `SVC(kernel="linear")` | `C=0.05`, `0.2`, `0.8` |
| RF | median impute → `RandomForestClassifier` | `(n_estimators, max_depth, min_samples_leaf)`: `(100, 8, 4)`, `(100, 8, 1)`, `(100, None, 4)`, `(100, None, 1)`, `(300, 8, 4)`, `(300, 8, 1)`, `(300, None, 4)`, `(300, None, 1)` |

Family order for cross-family ties: **lr, then svm, then rf**.

Tie-break (single rule, no second pass):

1. Higher inner-CV mean macro-F1 (`labels=[0,1,2]`, `zero_division=0`) wins.
2. If still tied, the **earlier** candidate in the concatenated list (family order, then the table above) wins.

Fewer RF trees is a **compute-cost** preference in that list, not “stronger regularization.” There is no extra “then more regularized” rule after the list; such a rule would never run.

Inner splitter: 3-fold stratified (grouped: `StratifiedGroupKFold`) **on the outer training indices only**. Imputation, scaling, family, and hyperparameters are fit on inner training data only.

### Pre-specified comparisons (one set of four)

1. `b_lr` vs dummy  
2. `b_lr` vs nested-single  
3. `b_lr` vs Experiment A `lr_all`  
4. `b_select` vs Experiment A `lr_all`

These four pairs are one declared comparison set. Per-pair Nadeau–Bengio-style intervals (df = 49) may still be reported, but they are **not** simultaneous confidence intervals. An unadjusted interval that excludes 0 does **not** confirm the whole set. Do not add extra significance tests in this design. This CSV has already been used in multiple exploratory rounds; this design is **not** a prospective registration on untouched data.

Save per outer fold: selected family, params, every inner candidate score, and OOF predictions. Reuse Experiment A OOF for dummy / nested-single / `lr_all` only when that A run used the same rows, outer splits, seed, and estimators.

### Fit budget (`pipeline.fit` calls)

- Outer folds: 50. Inner folds: 3. Candidates: 3 + 3 + 8 = 14.
- **Inner fits:** `(3+3+8) × 3 × 50 = 2100`.
- **Family outer-train refits:** 3 families × 50 = **150**.
- **Search + family refit total: 2250.** This is **not** “2250 inner fits.”

`b_select` must reuse, on each outer fold, the inner scores already computed for those 14 candidates, and the outer-train refit/OOF of the winning family, whenever data, split membership, params, seed, preprocessor, estimator, and scorer are identical. Under that rule `b_select` needs **0** extra refits (its pick is one of the three family winners). If reuse is impossible, budget up to 50 extra outer-train fits.

**Not included in 2250** (list separately):

- Uncached Experiment A baseline fits (dummy, `lr_all`, SVM, RF, …).
- nested-single inner fits over features, plus its outer-train refit.
- Optional final **full-data** artifact fits after OOF is frozen. Do **not** retune those artifacts from outer validation scores. They are not performance estimates.
- Extra `b_select` refits only if reuse fails.

On this dataset size (~1100×20), a local 50-fold nested run is expected in minutes to tens of minutes.

### Artifacts

Save outer/inner membership, every inner candidate score, selected family and params per outer fold for `b_lr` and `b_select`, OOF, `requirements-lock.txt`, and code manifest. Distinguish **OOF performance** from a **final fit on all included rows**.

Grouped sensitivity, if ever run for B: same group isolation and 3-class coverage as A; no silent fallback to ungrouped CV. Grouped permutation remains `not_available`.

## Metrics and comparisons

- **Primary:** macro-F1 on `[0,1,2]`.
- **Secondary:** accuracy; per-class P/R/F1; 3×3 confusion matrix; 0↔2 error rate with denominator = count of true 0 or 2.
- Pre-specify B comparisons before looking at B scores: the four pairs in the Experiment B section. Do not fish all pairwise tests. Do not treat a post-hoc outer-score family ranking as a pre-specified test.
- Paired tests only on shared outer folds and the same rows.
- Repeat std ≠ generalization CI. CI containing 0 ≠ equivalence.

## Sensitivity protocols

| Policy | Who is included | What may run |
|--------|-----------------|--------------|
| `primary` | rows 0–1099, valid labels | Full A (and later B) with ungrouped repeated stratified CV |
| `sensitivity` + `raw` | all valid labels | Duplicate audit; **refuse CV** if identical-feature groups exist; report the refusal, do not invent scores |
| `sensitivity` + `quarantine_out_of_range` | valid labels minus **legacy upper-bound** hits only | Optional A-style CV; not a new cleaning standard |
| `sensitivity` + `grouped_duplicates` | all valid labels; `StratifiedGroupKFold` on identical-feature groups | A-style CV with group isolation + 3-class coverage; permutation **unavailable** |

Always report n included, class counts, n groups size>1, and exclusion reasons. Different policies are **different populations**; score changes are not “the effect of cleaning.”

Smoke (`--smoke`) only checks that a path runs (3×1, 5 perms or skip). Formal sensitivity needs the full 5×10 protocol.

Commands (do **not** start all of these in this pass):

```bash
python scripts/run_audit.py StressLevelDataset_original.csv results/sens_raw_full \
  --scope sensitivity --sensitivity-policy raw --n-perm 100 --seed 0
python scripts/run_audit.py StressLevelDataset_original.csv results/sens_oor_full \
  --scope sensitivity --sensitivity-policy quarantine_out_of_range --n-perm 100 --seed 0
python scripts/run_audit.py StressLevelDataset_original.csv results/sens_grouped_full \
  --scope sensitivity --sensitivity-policy grouped_duplicates --n-perm 100 --seed 0
```

Each must be a **new empty** directory. Grouped full run should record `permutation.status=not_available` and still write `inner_folds.csv`.

## Error analysis

Use **OOF** predictions, never training-set predictions as if they were test results. Describe class-wise errors, 0↔2 swaps, rows missed by all core models, and persistent-error flags. Subgroups found after seeing OOF are **exploratory**. Persistent error ≠ confirmed mislabel. If explanations are added later: fit explainers on training folds only, state which rows are explained, and treat coefficients/importances as associations, not causes.

## External validation

Only if a second table has documented field meanings, the same label definition, compatible encodings, and a stated population. Do not pool datasets because both filenames contain “stress”. Until then, all numerical claims are **internal** to this CSV.

## Execution conditions (what must be true before the next run)

Do **not** start Experiment B search until all of the following hold:

1. Working tree is clean at a recorded commit; output directory is new and empty.
2. Outer splits and included rows are the Experiment A primary protocol (or an explicitly documented sensitivity protocol).
3. Implementation matches `src/stress_detection/experiment_b.py` (four comparisons, `b_select` reuse, 3-class coverage, no outer-fold peeking).
4. Optional: a candidate public CSV has been compared with `scripts/compare_dataset_candidate.py`. A file match is **not** required to run B on this repo CSV; it is required before claiming that B evaluates a named public dataset.

Sensitivity full runs and Experiment B remain optional and must use **new** directories. Do not overwrite `results/audit_full_4865bb3_clean`.

## Dataset candidate comparison

```bash
python scripts/compare_dataset_candidate.py \
  --candidate PATH/TO/downloaded.csv \
  --reference StressLevelDataset_original.csv \
  --prefix-rows 1100 \
  --out-dir /tmp/dataset_compare_rxnach
```

`--out-dir` must be new or empty. Inputs are not modified. Reports: `comparison.json`, `comparison.md`, `cell_differences.csv`, `row_count_differences.csv`.

## Execution order for later work

1. Compare a candidate dump to this CSV with `scripts/compare_dataset_candidate.py` (prefix 1100 as well as full file). A hash mismatch of the whole files does not by itself rule out a prefix match.
2. Optional: formal grouped/raw/oor sensitivity full runs into new directories.
3. Implement Experiment B search **as specified here** (`b_lr` / `b_select`; no extra models; no outer-fold peeks; reuse inner scores).
4. Report the four pre-specified paired comparisons on shared outer folds; keep any outer-score family ranking exploratory.
5. OOF error analysis, explicitly exploratory.
6. External data only if step 1-style codebook **and** comparable label definition exist.
