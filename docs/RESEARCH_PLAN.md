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

### Shared outer folds

Use the **same** Experiment A outer splits (seed=0, 5×10). All B candidates and A baselines must be scored on those folds. Do not retune the outer seed if a fold looks inconvenient.

### Models and grids

Default B does **not** add XGBoost or deep models.

| Model | Pipeline (unchanged preprocessing) | Grid | Why these knobs |
|-------|--------------------------------------|------|-----------------|
| LR | median impute → scale → `LogisticRegression(max_iter=5000)` | `C ∈ {0.1, 1.0, 10.0}` | Inverse L2 strength; A uses sklearn default C=1 |
| linear SVM | median impute → scale → `SVC(kernel="linear")` | `C ∈ {0.05, 0.2, 0.8}` | Soft-margin; A used C=0.2 |
| RF | median impute → `RandomForestClassifier` | `n_estimators ∈ {100, 300}`, `max_depth ∈ {None, 8}`, `min_samples_leaf ∈ {1, 4}` | Tree count, depth cap, leaf size |

Inner splitter: 3-fold stratified (grouped: `StratifiedGroupKFold`) **on the outer training indices only**. Selection metric: macro-F1 with labels `[0,1,2]`, `zero_division=0`. Tie-break: higher inner mean, then earlier grid index, then more regularized settings (smaller C, fewer trees, finite depth before `None`, larger min_samples_leaf).

Hyperparameters, imputation, scaling, and any feature choice are fit on inner training data only. Outer validation is used once, after selection, for the OOF estimate. Do not use outer validation to expand the grid, pick thresholds, or drop features.

### Fit budget (pipeline.fit calls)

- Outer folds: 50.
- LR: 3 candidates × 3 inner × 50 = **450** inner fits + **50** outer-train refits.
- SVM: 3 × 3 × 50 = **450** + **50**.
- RF: 8 × 3 × 50 = **1200** + **50**.
- **Search total: 2250** inner+refit fits, plus up to 3 full-data artifact fits that **must not** be reported as performance.

On this dataset size (~1100×20), a local 50-fold nested run is expected on the order of several minutes to tens of minutes, not a cluster job.

### Artifacts

Save outer/inner membership, every inner candidate score, selected params per outer fold, OOF predictions, `requirements-lock.txt`, and code manifest. Distinguish **OOF performance** from a **final fit on all included rows** (deployment artifact only).

Grouped sensitivity, if ever run for B: same group isolation and 3-class coverage as A; no silent fallback to ungrouped CV. Grouped permutation remains `not_available`.

## Metrics and comparisons

- **Primary:** macro-F1 on `[0,1,2]`.
- **Secondary:** accuracy; per-class P/R/F1; 3×3 confusion matrix; 0↔2 error rate with denominator = count of true 0 or 2.
- Pre-specify B comparisons before looking at B scores: `b_lr` vs dummy; `b_lr` vs nested-single; `b_lr` vs A `lr_all`; winner-among-B vs A `lr_all`. Do not fish all pairwise tests.
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

## Execution order for later work

1. Finish provenance hash comparison against candidate public dumps (manual).
2. Optional: formal grouped/raw/oor sensitivity full runs into new directories.
3. Implement Experiment B search **as specified here** (no extra models, no outer-fold peeks).
4. Pre-specified paired comparisons on shared outer folds.
5. OOF error analysis, explicitly exploratory.
6. External data only if step 1–style codebook match exists.
