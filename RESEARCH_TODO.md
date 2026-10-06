# Research questions that are still open

This list is for work that the audit code cannot settle. Do not treat items here as completed, and do not invent answers (sources, ethics approvals, participant counts, or a label-generation procedure) when evidence is missing.

The structured provenance checklist is `docs/DATA_PROVENANCE.md`. The next-stage experimental design is `docs/RESEARCH_PLAN.md`. Experiment B is **designed but not executed**.

## Unverified

| Question | Evidence that would be needed | What we cannot conclude without it |
|----------|-------------------------------|------------------------------------|
| How was `stress_level` produced? | Instrument, codebook, or generation script that maps fields → label | That models are predicting an independent outcome rather than reconstructing a derived label |
| What is the original source file and lineage? | Provenance of `StressLevelDataset_original.csv` (export, scrape, or study dump), including a hash match to a public dump | That the 1121 rows are a well-defined sample from a stated population, or that they are the commonly cited 1100-row `StressLevelDataset.csv` |
| Are rows distinct people? | Subject IDs, timestamps, or a duplicate-respondent protocol | That `grouped_duplicates` is a subject-level split rather than an identical-feature split |
| Why are rows 1100–1120 different / excluded in primary? | Collection notes or a documented population rule | That the 0–1099 cut is anything more than an audit convention; that the extra rows are the only difference vs public 1100-row copies |
| Are conflict groups or domain-rule hits errors? | Recoding against source instruments | That those rows are confirmed mislabels or fabricated data |
| Do scores generalize? | A held-out external dataset collected under a stated protocol with matching fields and label definition | External validity |

## Limits that remain even after the code audit

- Removing label-driven zero mapping stops that **implementation** leak. It does not prove the CSV label is independent of the inputs.
- Persistent-error flags and same-data tree descriptions are descriptive of this table, not a diagnosis.
- Experiment B nested search is specified in `docs/RESEARCH_PLAN.md` and `src/stress_detection/experiment_b.py`; `scripts/run_benchmark.py` only writes the design JSON. The four pre-specified pairs use `b_lr` and `b_select` (not an outer-score winner).
- No external validation has been run.
- GitHub Actions on `main` at `b769e6b` completed successfully (run 37516150242). Later local edits are not covered by that run until they are pushed.

## Not in scope of the current engineering audit

- Clinical claims, treatment recommendations, or causal statements about stress.
- Declaring the dataset synthetic or authentic.
- Equating this 1121-row file with a 1100-row public listing because the column names match.
