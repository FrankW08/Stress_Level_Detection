# Data provenance checklist

Status values: **known** (evidence in this repository), **candidate** (a public page with similar names or columns; identity not proven), **unknown**.

Matching column names, a shared filename `StressLevelDataset.csv`, or a paper that also studies “student stress” do **not** prove that `StressLevelDataset_original.csv` is that file, nor how `stress_level` was generated.

## Local file identity

| Item | Status | Known facts | Evidence | Unconfirmed | If this evidence is missing, cannot claim | Next check |
|------|--------|-------------|----------|-------------|-------------------------------------------|------------|
| File in this repo | known | `StressLevelDataset_original.csv`; 1121 data rows; 21 columns in the header | This path; `DATA_CARD.md`; audit `results.json` hashes | How the file was copied into the repo | A well-defined study sample | Ask the original committer; keep the file immutable |
| Content hash | known | LF-normalized SHA-256 `05145e0a27395e85f6ed062d6f89f99351bdd16f8fa7dc5e60243bd1083dab26` | README; each `results.json` | Whether any public dump has the same hash | Byte-level identity with another download | Hash a candidate download after LF normalization |
| Row identity | known | `source_row_id` is the zero-based CSV row index, added in memory only | `src/stress_detection/data.py` | Subject IDs | That a row is a person | Need a codebook with respondent IDs |
| Primary subset | known | Audit convention: rows 0–1099 with valid labels → 1098 rows; rows ≥1100 quarantined in primary | `prepare_data`; `exclusions.csv` | That 0–1099 is a population definition | Target-population inference | Collection notes |

## Public listings (identity not established)

Public copies often use the filename `StressLevelDataset.csv` and the same 21 column names, and they usually report **1100** records. This repository file has **1121** rows. The size mismatch alone means they are not automatically the same object.

| Item | Status | Known facts | Evidence | Unconfirmed | If this evidence is missing, cannot claim | Next check |
|------|--------|-------------|----------|-------------|-------------------------------------------|------------|
| Author / publisher of *this* CSV | unknown | This repo has no LICENSE, CITATION, or author file | Repository tree | Whether Kaggle user `rxnach` published this exact file | Authorship or permission | Download a candidate dump; compare LF-normalized SHA-256 and row counts |
| Candidate listing (same schema family) | candidate | Kaggle page title “Student Stress Factors: A Comprehensive Analysis” (URL user `rxnach`; page display name Chhabii). Column names overlap this header. Listing states **Apache 2.0**. Direct page fetch succeeded on 2026-10-06; the Download control sits behind Sign In | https://www.kaggle.com/datasets/rxnach/student-stress-factors-a-comprehensive-analysis | Row count of the Kaggle file; whether it equals this 1121-row object or its first 1100 rows | That published 1100-row analyses evaluated *this* file | User-provided CSV path; `scripts/compare_dataset_candidate.py`; do not merge by filename |
| Candidate paper (same filename, 1100 rows) | candidate | Springer *Discover Artificial Intelligence* article “Comprehensive analysis of stress factors affecting students: a machine learning approach” (2024) discusses a `StressLevelDataset.csv` of 1100 students and points to a GitHub copy | https://link.springer.com/article/10.1007/s44163-024-00169-6 | Whether that 1100-row table is a prefix, a subset, or unrelated to this 1121-row file | That this audit replicates that paper | Hash comparison against the paper’s linked file |
| Non-match (do not cite as source) | known as different schema | Kaggle `samyakb/student-stress-factors` is a small Google-Forms table (on the order of tens of rows, a handful of 1–5 ratings such as sleep / headaches / study load). Apache 2.0 is claimed on that listing | https://www.kaggle.com/datasets/samyakb/student-stress-factors | — | — | Do not treat a shared phrase “student stress factors” as identity |
| Version | unknown | — | — | Dataset version on Kaggle vs this CSV | Reproducible citation of a public dump | Record version/date of any matching dump |
| License of this repo file | unknown | This repo has no license file | Repository tree | License of a hash-matching dump | Redistribution rights | If a hash match is found, copy that license explicitly |

## Collection and respondents

| Item | Status | Known facts | Evidence | Unconfirmed | If this evidence is missing, cannot claim | Next check |
|------|--------|-------------|----------|-------------|-------------------------------------------|------------|
| How collected | unknown | — | Some candidate pages/READMEs say “nationwide survey” or “various educational institutions”; those are claims about *their* 1100-row copies | Instrument, dates, sampling frame for *this* file | Representativeness | Need a survey protocol tied to this file |
| When collected | unknown | — | — | Year / wave | Temporal validity | Need collection dates |
| Population | unknown | Feature names refer to students in ordinary language | Column names | Age, country, institution. Third-party READMEs sometimes mention “1100 students”, age 15–24, or a city; **none of that is evidenced for this file** | External generalization | Need inclusion criteria |
| Independent subjects vs repeated measures | unknown | Identical-feature groups exist and are used only as `duplicate_group_id` | `assign_duplicate_group_ids` | Whether duplicates are the same person | Subject-level inference; grouped CV as person-level CV | Need subject IDs or timestamps |
| Ethics / consent | unknown | — | The Springer candidate paper discusses de-identified data; that does not document an IRB for this repo | IRB or consent for this CSV | Ethical reuse claims | Do not invent an approval |

## Target and fields

| Item | Status | Known facts | Evidence | Unconfirmed | If this evidence is missing, cannot claim | Next check |
|------|--------|-------------|----------|-------------|-------------------------------------------|------------|
| `stress_level` coding in this file | known | Training uses integers {0,1,2} only; other values excluded | `VALID_STRESS_LABELS`; `exclusions.csv` | Semantic of 0/1/2 (low/medium/high is a common *description* on candidate pages, not verified here) | Clinical meaning | Codebook |
| How the label was generated | unknown | Default pipeline does not compute the label from features | Training code | Self-report vs sum of items vs researcher coding vs leakage from inputs | That models predict an independent construct | Script or instrument mapping |
| Label independent of inputs | unknown | Label-driven zero map was an implementation leak and was removed from the default path | `LEAK_MAP` is audit-only | Whether the CSV label was derived from the same fields | Causal or diagnostic claims | Provenance of the label column |
| Instrument mapping of scale fields | unknown | This repo’s schema maxima (anxiety 21, self-esteem 30, depression 27) are **provisional reporting bounds** | `DATA_CARD.md`; `schema.py` | Whether those fields are GAD-7, Rosenberg, and PHQ-9. Some third-party READMEs assert that mapping; it is **not** confirmed for this file | That scores are those named instruments | Codebook or item text |
| Field encodings / zero / missing | provisional | Parser maps Null/empty to missing; Likert 0–5 and scale maxima are **provisional** reporting rules | `DATA_CARD.md`; `schema.py` | Official instrument ranges; meaning of 0 | That a “violation” is an error | Data dictionary |
| `blood_pressure` | provisional | Treated as a small integer encoding, **not** mmHg; DATA_CARD notes a non-monotonic association with the label in this table | `DATA_CARD.md`; audit notes | Category labels (e.g. 1/2/3) | Physiological interpretation | Codebook |
| Tail rows 1100–1120 | known as excluded | Primary scope quarantines them by convention | `build_exclusions` | Whether they are a different batch, errors, extra copies, or the only difference vs 1100-row public files | That they are forgeries | Compare hashes of rows 0–1099 vs a 1100-row candidate; collection notes |
| External validation data | unknown | None in this repository | — | A second sample with the same fields and label definition | External validity | Only merge after codebook-level comparability |

## What current model scores mean

They are scores for **predicting the `stress_level` column already stored in this CSV** under the audit protocol. They are not a measurement of “true stress,” a diagnosis, or a claim about a named public dataset until hashes and a codebook match.

## Manual follow-up

Whole-file SHA-256 mismatch between a 1100-row download and this 1121-row file does **not** by itself prove the files are unrelated. Use:

```bash
python scripts/compare_dataset_candidate.py --candidate PATH/TO/candidate.csv --prefix-rows 1100 --out-dir /tmp/dataset_compare
```

A prefix or order-insensitive match still only supports **file content** correspondence, not sampling frame, label generation, or ethics.

1. Obtain the `rxnach` Kaggle CSV (login/terms may be required; this environment does not use account credentials).
2. Run the layered comparison (full file and prefix 1100).
3. If only a prefix matches, document the extra 21 rows as unexplained suffix — still not a reason to include them in primary without a codebook.
4. Record license and version from a matching dump only after a documented content match.
