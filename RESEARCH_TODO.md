# Research questions that are still open

This list is for work that the audit code cannot settle. Do not treat items here as completed, and do not invent answers (sources, ethics approvals, participant counts, or a label-generation procedure) when evidence is missing.

## Unverified

| Question | Evidence that would be needed | What we cannot conclude without it |
|----------|-------------------------------|------------------------------------|
| How was `stress_level` produced? | Instrument, codebook, or generation script that maps fields → label | That models are predicting an independent outcome rather than reconstructing a derived label |
| What is the original source file and lineage? | Provenance of `StressLevelDataset_original.csv` (export, scrape, or study dump) | That the 1121 rows are a well-defined sample from a stated population |
| Are rows distinct people? | Subject IDs, timestamps, or a duplicate-respondent protocol | That `grouped_duplicates` is a subject-level split rather than an identical-feature split |
| Why are rows 1100–1120 different / excluded in primary? | Collection notes or a documented population rule | That the 0–1099 cut is anything more than an audit convention |
| Are conflict groups or domain-rule hits errors? | Recoding against source instruments | That those rows are confirmed mislabels or fabricated data |
| Do scores generalize? | A held-out external dataset collected under a stated protocol | External validity |

## Limits that remain even after the code audit

- Removing label-driven zero mapping stops that **implementation** leak. It does not prove the CSV label is independent of the inputs.
- Persistent-error flags and same-data tree descriptions are descriptive of this table, not a diagnosis.
- Experiment B (full nested search) is not implemented.
- No external validation has been run.

## Not in scope of the current engineering audit

- Clinical claims, treatment recommendations, or causal statements about stress.
- Declaring the dataset synthetic or authentic.
