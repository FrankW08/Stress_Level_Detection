# Agent notes

- Do **not** overwrite `StressLevelDataset_original.csv` or delete prior `results/` historical runs (e.g. `results/audit_full`).
- Default ML path: `src/stress_detection` + `scripts/run_audit.py` (experiment A).
- Label-driven zero mapping is **audit-only** (`LEAK_MAP` / `apply_label_driven_zero_map` / leaky feature adapter); never ship in default training or inference.
- Do not treat old notebook accuracy figures (~94% SVM) as benchmarks.
- Prefer writing new audit outputs to new directories; do not overwrite committed `results/audit_full`.
- Prefer `uv sync --extra dev --extra benchmark` for locked environments.
