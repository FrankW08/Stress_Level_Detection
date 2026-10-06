# Agent notes

- Do **not** overwrite `StressLevelDataset_original.csv` or delete prior `results/` historical runs (e.g. `results/audit_full`, `results/audit_full_d12aa67_clean`, `results/audit_full_4865bb3_clean`).
- Default ML path: `src/stress_detection` + `scripts/run_audit.py` (experiment A).
- Label-driven zero mapping is **audit-only** (`LEAK_MAP` / `apply_label_driven_zero_map` / leaky feature adapter); never ship in default training or inference.
- Do not treat old notebook accuracy figures (~94% SVM) as benchmarks.
- Output directory is required and must be new or empty; the program refuses to write into a non-empty directory.
- 3-class protocol: labels `[0,1,2]`, `zero_division=0`, every outer/inner train and valid fold must cover all three classes. Fail clearly; do not fall back to ungrouped CV or resample seeds. Generate and validate splits before creating the output directory; training must reuse those splits.
- Persistent-error auxiliary classifier is binary; do not apply `[0,1,2]` to it.
- Experiment B: design only this round (`scripts/run_benchmark.py`, `docs/RESEARCH_PLAN.md`). Do not run the nested search unless explicitly asked.
- Do not invent data provenance, label-generation, or ethics facts. See `docs/DATA_PROVENANCE.md` and `RESEARCH_TODO.md`. Column-name similarity is not file identity.
- Prefer `uv sync --extra dev --extra benchmark` for locked environments. CI uses `uv sync --frozen --extra dev`.
- Do not auto commit or push unless the user asks.
