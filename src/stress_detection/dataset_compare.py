"""Read-only layered comparison of a candidate CSV against the repo reference.

Content match supports file correspondence only. It does not prove sampling
frame, label generation, ethics review, or a named public source.
"""

from __future__ import annotations

import csv
import io
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stress_detection.hashes import (
    normalize_newlines_to_lf,
    sha256_bytes,
    sha256_file,
    sha256_file_normalized_lf,
)
from stress_detection.io_guard import ensure_empty_output_dir

DEFAULT_PREFIX_ROWS = 1100
DEFAULT_DETAIL_LIMIT = 10_000
ENCODING = "utf-8-sig"
# After strip; compared case-insensitively. Empty string is missing.
MISSING_TOKENS = frozenset({"", "null", "nan", "na", "none", "n/a"})

RULES = (
    "Read bytes first for hashes. Decode as UTF-8 with BOM skipped (utf-8-sig). "
    "Parse with Python csv.reader (Excel dialect, newline=''). "
    "Data row indices are 0-based and do not count the header. "
    "Raw comparison uses csv field strings after parse (quotes handled by csv). "
    "Format-normalize: strip surrounding whitespace; map CR/LF inside a cell to LF. "
    "Missing tokens after format-normalize, case-insensitive: empty, null, nan, na, none, n/a. "
    "Numeric parse: float() on the format-normalized string only if the cell is not missing. "
    "Non-numeric non-missing text is never coerced to NaN. "
    "No row drops, imputes, deduplication, or sorting except in the separate multiset analysis. "
    "Column-name alignment is used for cell and multiset compares; column-order mismatch is reported separately."
)


@dataclass
class LoadedCsv:
    path: Path
    raw_bytes: bytes
    text: str
    encoding: str
    columns: list[str]
    rows: list[list[str]]
    raw_sha256: str
    lf_sha256: str
    nbytes: int
    read_at_utc: str


def _format_normalize(cell: str) -> str:
    return cell.strip().replace("\r\n", "\n").replace("\r", "\n")


def _is_missing(cell: str) -> bool:
    return _format_normalize(cell).lower() in MISSING_TOKENS


def _try_float(cell: str) -> float | None:
    if _is_missing(cell):
        return None
    try:
        return float(_format_normalize(cell))
    except ValueError:
        return None


def load_csv(path: Path) -> LoadedCsv:
    path = Path(path).resolve()
    raw = path.read_bytes()
    read_at = datetime.now(timezone.utc).isoformat()
    text = raw.decode(ENCODING)
    reader = csv.reader(io.StringIO(text, newline=""))
    try:
        header = next(reader)
    except StopIteration:
        header = []
        body: list[list[str]] = []
    else:
        body = [list(row) for row in reader]
        width = len(header)
        for row in body:
            if len(row) < width:
                row.extend([""] * (width - len(row)))
            elif len(row) > width:
                # Keep extras attached as additional anonymous columns on first occurrence.
                pass
    extra = max((len(r) for r in body), default=len(header)) - len(header)
    columns = list(header)
    if extra > 0:
        columns.extend([f"__extra_col_{i}" for i in range(extra)])
        for row in body:
            if len(row) < len(columns):
                row.extend([""] * (len(columns) - len(row)))
    return LoadedCsv(
        path=path,
        raw_bytes=raw,
        text=text,
        encoding=ENCODING,
        columns=columns,
        rows=body,
        raw_sha256=sha256_bytes(raw),
        lf_sha256=sha256_bytes(normalize_newlines_to_lf(raw)),
        nbytes=len(raw),
        read_at_utc=read_at,
    )


def _file_layer(obj: LoadedCsv) -> dict[str, Any]:
    return {
        "path": str(obj.path),
        "nbytes": obj.nbytes,
        "read_at_utc": obj.read_at_utc,
        "encoding": obj.encoding,
        "raw_sha256": obj.raw_sha256,
        "lf_normalized_sha256": obj.lf_sha256,
        "n_data_rows": len(obj.rows),
        "n_columns": len(obj.columns),
        "columns": obj.columns,
        "row_index_base": 0,
        "row_index_note": "0-based data rows; header is not a data row",
    }


def _aligned_row(columns: list[str], row: list[str], col_to_idx: dict[str, int]) -> list[str]:
    out = []
    by_idx = {i: v for i, v in enumerate(row)}
    for c in columns:
        j = col_to_idx.get(c)
        out.append("" if j is None else by_idx.get(j, ""))
    return out


def _classify_pair(cand: str, ref: str) -> str | None:
    """Return a difference kind, or None if treated as equal for content matching."""
    if cand == ref:
        return None
    c_miss, r_miss = _is_missing(cand), _is_missing(ref)
    if c_miss and r_miss:
        return "format_only"
    if c_miss != r_miss:
        return "missing_vs_nonmissing"
    c_num, r_num = _try_float(cand), _try_float(ref)
    if c_num is not None and r_num is not None:
        if c_num == r_num:
            return "format_only"
        return "numeric_difference"
    if _format_normalize(cand) == _format_normalize(ref):
        return "format_only"
    if c_num is None and not c_miss and r_num is not None:
        return "unparseable"
    if r_num is None and not r_miss and c_num is not None:
        return "unparseable"
    if c_num is None and r_num is None and not c_miss and not r_miss:
        return "content_difference"
    return "content_difference"


def _cell_diff_rows(
    cand_rows: list[list[str]],
    ref_rows: list[list[str]],
    cand_cols: list[str],
    ref_cols: list[str],
    aligned_cols: list[str],
    table: str,
    detail_limit: int,
) -> tuple[list[dict[str, Any]], dict[str, int], bool]:
    cand_map = {c: i for i, c in enumerate(cand_cols)}
    ref_map = {c: i for i, c in enumerate(ref_cols)}
    counts: dict[str, int] = {
        "equal_raw": 0,
        "format_only": 0,
        "numeric_difference": 0,
        "missing_vs_nonmissing": 0,
        "unparseable": 0,
        "content_difference": 0,
        "n_cells_compared": 0,
    }
    details: list[dict[str, Any]] = []
    truncated = False
    n = min(len(cand_rows), len(ref_rows))
    for i in range(n):
        crow = _aligned_row(aligned_cols, cand_rows[i], cand_map)
        rrow = _aligned_row(aligned_cols, ref_rows[i], ref_map)
        for col, cv, rv in zip(aligned_cols, crow, rrow):
            counts["n_cells_compared"] += 1
            kind = _classify_pair(cv, rv)
            if kind is None:
                counts["equal_raw"] += 1
                continue
            counts[kind] = counts.get(kind, 0) + 1
            rec = {
                "table": table,
                "row_index_0based": i,
                "column": col,
                "candidate_raw": cv,
                "reference_raw": rv,
                "kind": kind,
            }
            if len(details) < detail_limit:
                details.append(rec)
            else:
                truncated = True
    return details, counts, truncated


def _raw_equal(cand_rows, ref_rows, cand_cols, ref_cols, aligned_cols) -> bool:
    if len(cand_rows) != len(ref_rows):
        return False
    cand_map = {c: i for i, c in enumerate(cand_cols)}
    ref_map = {c: i for i, c in enumerate(ref_cols)}
    for i in range(len(cand_rows)):
        if _aligned_row(aligned_cols, cand_rows[i], cand_map) != _aligned_row(
            aligned_cols, ref_rows[i], ref_map
        ):
            return False
    return True


def _format_equal(cand_rows, ref_rows, cand_cols, ref_cols, aligned_cols) -> bool:
    if len(cand_rows) != len(ref_rows):
        return False
    cand_map = {c: i for i, c in enumerate(cand_cols)}
    ref_map = {c: i for i, c in enumerate(ref_cols)}
    for i in range(len(cand_rows)):
        crow = _aligned_row(aligned_cols, cand_rows[i], cand_map)
        rrow = _aligned_row(aligned_cols, ref_rows[i], ref_map)
        for cv, rv in zip(crow, rrow):
            if _classify_pair(cv, rv) not in (None, "format_only"):
                return False
    return True


def _multiset(
    rows: list[list[str]], cols: list[str], aligned_cols: list[str]
) -> Counter[tuple[str, ...]]:
    col_map = {c: i for i, c in enumerate(cols)}
    counts: Counter[tuple[str, ...]] = Counter()
    for row in rows:
        aligned = _aligned_row(aligned_cols, row, col_map)
        counts[tuple(aligned)] += 1
    return counts


def _multiset_report(
    cand_rows, ref_rows, cand_cols, ref_cols, aligned_cols, table: str, detail_limit: int
) -> dict[str, Any]:
    cc = _multiset(cand_rows, cand_cols, aligned_cols)
    rc = _multiset(ref_rows, ref_cols, aligned_cols)
    same_multiset = cc == rc
    n_cand, n_ref = len(cand_rows), len(ref_rows)
    order_differs = False
    if same_multiset and n_cand == n_ref and n_cand > 0:
        order_differs = not _raw_equal(cand_rows, ref_rows, cand_cols, ref_cols, aligned_cols)
    extras: list[dict[str, Any]] = []
    truncated = False
    keys = set(cc) | set(rc)
    for key in sorted(keys, key=lambda t: (cc[t] != rc[t], t)):
        if cc[key] == rc[key]:
            continue
        rec = {
            "table": table,
            "candidate_count": cc[key],
            "reference_count": rc[key],
            "row_tuple": list(key),
        }
        if len(extras) < detail_limit:
            extras.append(rec)
        else:
            truncated = True
    status = "not_applicable"
    if n_cand == 0 and n_ref == 0:
        status = "both_empty"
    elif same_multiset and not order_differs:
        status = "same_rows_same_order_or_empty"
        if n_cand and _raw_equal(cand_rows, ref_rows, cand_cols, ref_cols, aligned_cols):
            status = "same_rows_same_order"
        elif n_cand == 0:
            status = "both_empty"
    elif same_multiset and order_differs:
        status = "same_rows_different_order"
    else:
        status = "row_or_multiplicity_differences"
    return {
        "table": table,
        "status": status,
        "same_multiset": same_multiset,
        "n_candidate_rows": n_cand,
        "n_reference_rows": n_ref,
        "n_distinct_row_keys_with_count_mismatch": sum(1 for k in keys if cc[k] != rc[k]),
        "details": extras,
        "details_truncated": truncated,
        "note": (
            "Multiset comparison uses raw csv fields aligned by column name. "
            "It does not replace positional row comparison."
        ),
    }


def compare_loaded(
    candidate: LoadedCsv,
    reference: LoadedCsv,
    *,
    prefix_rows: int = DEFAULT_PREFIX_ROWS,
    detail_limit: int = DEFAULT_DETAIL_LIMIT,
) -> dict[str, Any]:
    cand_cols, ref_cols = candidate.columns, reference.columns
    order_diff = cand_cols != ref_cols
    shared = [c for c in cand_cols if c in set(ref_cols)]
    only_cand = [c for c in cand_cols if c not in set(ref_cols)]
    only_ref = [c for c in ref_cols if c not in set(cand_cols)]
    aligned = shared
    comparable = bool(aligned)

    exact = candidate.raw_sha256 == reference.raw_sha256
    normalized = candidate.lf_sha256 == reference.lf_sha256

    full_shape_ok = comparable and len(candidate.rows) == len(reference.rows)
    prefix_n = prefix_rows
    cand_pref = candidate.rows[:prefix_n]
    ref_pref = reference.rows[:prefix_n]
    prefix_shape_ok = comparable and len(candidate.rows) >= prefix_n and len(reference.rows) >= prefix_n
    # If candidate is exactly the prefix length, still compare those rows to reference prefix.
    if comparable and len(candidate.rows) == prefix_n and len(reference.rows) >= prefix_n:
        prefix_shape_ok = True
        cand_pref = candidate.rows
        ref_pref = reference.rows[:prefix_n]
    elif comparable and len(candidate.rows) < prefix_n:
        prefix_shape_ok = False

    statuses: dict[str, Any] = {
        "exact_file_match": exact,
        "normalized_file_match": normalized,
        "column_order_differs": order_diff,
        "full_table_raw_match": False,
        "full_table_format_match": False,
        "prefix_raw_match": False,
        "prefix_format_match": False,
        "prefix_content_match": False,
        "full_same_rows_different_order": False,
        "prefix_same_rows_different_order": False,
        "content_differences": False,
        "not_comparable": not comparable,
    }

    full_details: list[dict[str, Any]] = []
    pref_details: list[dict[str, Any]] = []
    full_counts: dict[str, int] = {}
    pref_counts: dict[str, int] = {}
    full_trunc = pref_trunc = False
    full_ms: dict[str, Any] = {}
    pref_ms: dict[str, Any] = {}

    if comparable and full_shape_ok:
        full_details, full_counts, full_trunc = _cell_diff_rows(
            candidate.rows, reference.rows, cand_cols, ref_cols, aligned, "full", detail_limit
        )
        statuses["full_table_raw_match"] = _raw_equal(
            candidate.rows, reference.rows, cand_cols, ref_cols, aligned
        )
        statuses["full_table_format_match"] = _format_equal(
            candidate.rows, reference.rows, cand_cols, ref_cols, aligned
        )
    if comparable:
        full_ms = _multiset_report(
            candidate.rows, reference.rows, cand_cols, ref_cols, aligned, "full", detail_limit
        )
        if not full_shape_ok:
            full_ms["positional_status"] = (
                f"row counts differ (candidate={len(candidate.rows)}, "
                f"reference={len(reference.rows)}); positional full compare skipped"
            )
        statuses["full_same_rows_different_order"] = (
            full_ms["status"] == "same_rows_different_order"
        )

    if comparable and prefix_shape_ok:
        pref_details, pref_counts, pref_trunc = _cell_diff_rows(
            cand_pref, ref_pref, cand_cols, ref_cols, aligned, "prefix", detail_limit
        )
        statuses["prefix_raw_match"] = _raw_equal(
            cand_pref, ref_pref, cand_cols, ref_cols, aligned
        )
        statuses["prefix_format_match"] = _format_equal(
            cand_pref, ref_pref, cand_cols, ref_cols, aligned
        )
        statuses["prefix_content_match"] = statuses["prefix_raw_match"] or statuses[
            "prefix_format_match"
        ]
        pref_ms = _multiset_report(
            cand_pref, ref_pref, cand_cols, ref_cols, aligned, "prefix", detail_limit
        )
        statuses["prefix_same_rows_different_order"] = (
            pref_ms["status"] == "same_rows_different_order"
        )
    elif comparable:
        pref_ms = {
            "table": "prefix",
            "status": "not_comparable",
            "reason": (
                f"need at least {prefix_n} data rows in each file for a prefix-{prefix_n} "
                f"compare (candidate={len(candidate.rows)}, reference={len(reference.rows)})"
            ),
            "details": [],
            "details_truncated": False,
        }

    material = {"numeric_difference", "missing_vs_nonmissing", "unparseable", "content_difference"}
    n_material = 0
    for counts in (full_counts, pref_counts):
        n_material += sum(counts.get(k, 0) for k in material)
    statuses["content_differences"] = n_material > 0 or bool(only_cand or only_ref)

    return {
        "rules": RULES,
        "disclaimer": (
            "A content match supports correspondence of file bytes or parsed cells only. "
            "It does not by itself prove sampling population, how stress_level was generated, "
            "ethics approval, or that this file is a named public dataset."
        ),
        "prefix_rows": prefix_rows,
        "detail_limit": detail_limit,
        "candidate": _file_layer(candidate),
        "reference": _file_layer(reference),
        "columns": {
            "candidate_order": cand_cols,
            "reference_order": ref_cols,
            "order_differs": order_diff,
            "shared_in_candidate_order": shared,
            "only_in_candidate": only_cand,
            "only_in_reference": only_ref,
        },
        "statuses": statuses,
        "positional": {
            "full": {
                "comparable": bool(comparable and full_shape_ok),
                "counts": full_counts,
                "n_difference_rows_written": len(full_details),
                "details_truncated": full_trunc,
            },
            "prefix": {
                "comparable": bool(comparable and prefix_shape_ok),
                "n_candidate_rows_used": len(cand_pref) if prefix_shape_ok else 0,
                "n_reference_rows_used": len(ref_pref) if prefix_shape_ok else 0,
                "counts": pref_counts,
                "n_difference_rows_written": len(pref_details),
                "details_truncated": pref_trunc,
            },
        },
        "order_insensitive": {"full": full_ms, "prefix": pref_ms},
        "_cell_details": full_details + pref_details,
        "_row_count_details": list(full_ms.get("details") or []) + list(pref_ms.get("details") or []),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            out = dict(row)
            if "row_tuple" in out and not isinstance(out["row_tuple"], str):
                out["row_tuple"] = json.dumps(out["row_tuple"], ensure_ascii=False)
            w.writerow(out)


def _markdown(report: dict[str, Any]) -> str:
    st = report["statuses"]
    lines = [
        "# Dataset candidate comparison",
        "",
        report["disclaimer"],
        "",
        "## Files",
        "",
        f"- Candidate: `{report['candidate']['path']}` ({report['candidate']['n_data_rows']} data rows, "
        f"{report['candidate']['nbytes']} bytes)",
        f"- Reference: `{report['reference']['path']}` ({report['reference']['n_data_rows']} data rows, "
        f"{report['reference']['nbytes']} bytes)",
        f"- Candidate raw SHA-256: `{report['candidate']['raw_sha256']}`",
        f"- Candidate LF SHA-256: `{report['candidate']['lf_normalized_sha256']}`",
        f"- Reference raw SHA-256: `{report['reference']['raw_sha256']}`",
        f"- Reference LF SHA-256: `{report['reference']['lf_normalized_sha256']}`",
        f"- Prefix rows requested: {report['prefix_rows']}",
        "",
        "## Statuses (not collapsed into one label)",
        "",
    ]
    for k, v in st.items():
        lines.append(f"- `{k}`: {v}")
    lines += [
        "",
        "## Columns",
        "",
        f"- Order differs: {report['columns']['order_differs']}",
        f"- Only in candidate: {report['columns']['only_in_candidate'] or '(none)'}",
        f"- Only in reference: {report['columns']['only_in_reference'] or '(none)'}",
        "",
        "## Rules",
        "",
        report["rules"],
        "",
    ]
    return "\n".join(lines) + "\n"


def write_comparison(report: dict[str, Any], out_dir: Path) -> Path:
    out_dir = ensure_empty_output_dir(out_dir)
    cell_details = report.pop("_cell_details", [])
    row_details = report.pop("_row_count_details", [])
    (out_dir / "comparison.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "comparison.md").write_text(_markdown(report), encoding="utf-8")
    _write_csv(
        out_dir / "cell_differences.csv",
        cell_details,
        ["table", "row_index_0based", "column", "candidate_raw", "reference_raw", "kind"],
    )
    _write_csv(
        out_dir / "row_count_differences.csv",
        row_details,
        ["table", "candidate_count", "reference_count", "row_tuple"],
    )
    return out_dir


def compare_files(
    candidate: Path,
    reference: Path,
    out_dir: Path,
    *,
    prefix_rows: int = DEFAULT_PREFIX_ROWS,
    detail_limit: int = DEFAULT_DETAIL_LIMIT,
) -> dict[str, Any]:
    cand = load_csv(candidate)
    ref = load_csv(reference)
    before = (cand.raw_sha256, ref.raw_sha256)
    report = compare_loaded(cand, ref, prefix_rows=prefix_rows, detail_limit=detail_limit)
    write_comparison(report, out_dir)
    after_c = sha256_file(Path(candidate).resolve())
    after_r = sha256_file(Path(reference).resolve())
    if (after_c, after_r) != before:
        raise RuntimeError("input files changed on disk during comparison")
    report["input_hashes_unchanged"] = True
    return report
