# Agent notes

- Do **not** overwrite `StressLevelDataset_original.csv` or delete prior `results/` runs.
- Default ML path: `src/stress_detection` + `scripts/run_audit.py` (experiment A).
- Label-driven zero mapping is **audit-only** (`LEAK_MAP` / `RUN_LEAKY_SVM_DEMO`); never ship in production pipelines.
- Do not treat old notebook accuracy figures as benchmarks.
