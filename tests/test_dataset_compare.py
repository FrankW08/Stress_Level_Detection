"""Layered CSV comparison: synthetic cases only; does not touch the original dataset."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from stress_detection.dataset_compare import compare_files
from stress_detection.hashes import sha256_file as sha_file
from stress_detection.io_guard import OutputDirError

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "compare_dataset_candidate.py"


def _write_bytes(path: Path, text: str, newline: str = "\n") -> Path:
    body = text.replace("\n", newline)
    path.write_bytes(body.encode("utf-8"))
    return path


def test_exact_match(tmp_path):
    text = "a,b\n1,2\n3,4\n"
    cand = _write_bytes(tmp_path / "c.csv", text)
    ref = _write_bytes(tmp_path / "r.csv", text)
    before = (sha_file(cand), sha_file(ref))
    report = compare_files(cand, ref, tmp_path / "out", prefix_rows=2)
    assert report["statuses"]["exact_file_match"] is True
    assert report["statuses"]["normalized_file_match"] is True
    assert report["statuses"]["full_table_raw_match"] is True
    assert report["statuses"]["not_comparable"] is False
    assert before == (sha_file(cand), sha_file(ref))
    assert (tmp_path / "out" / "comparison.json").is_file()
    assert (tmp_path / "out" / "comparison.md").is_file()


def test_lf_vs_crlf(tmp_path):
    text = "a,b\n1,2\n3,4\n"
    cand = _write_bytes(tmp_path / "c.csv", text, newline="\n")
    ref = _write_bytes(tmp_path / "r.csv", text, newline="\r\n")
    report = compare_files(cand, ref, tmp_path / "out", prefix_rows=2)
    assert report["statuses"]["exact_file_match"] is False
    assert report["statuses"]["normalized_file_match"] is True
    assert report["statuses"]["full_table_raw_match"] is True


def test_numeric_text_format_only(tmp_path):
    cand = _write_bytes(tmp_path / "c.csv", "a,b\n1,2\n")
    ref = _write_bytes(tmp_path / "r.csv", "a,b\n1.0,2.0\n")
    report = compare_files(cand, ref, tmp_path / "out", prefix_rows=1)
    assert report["statuses"]["full_table_raw_match"] is False
    assert report["statuses"]["full_table_format_match"] is True
    cells = (tmp_path / "out" / "cell_differences.csv").read_text(encoding="utf-8")
    assert "format_only" in cells


def test_column_order_differs_but_aligned_match(tmp_path):
    cand = _write_bytes(tmp_path / "c.csv", "b,a\n2,1\n")
    ref = _write_bytes(tmp_path / "r.csv", "a,b\n1,2\n")
    report = compare_files(cand, ref, tmp_path / "out", prefix_rows=1)
    assert report["columns"]["order_differs"] is True
    assert report["statuses"]["full_table_raw_match"] is True


def test_row_order_multiset(tmp_path):
    cand = _write_bytes(tmp_path / "c.csv", "a,b\n1,2\n3,4\n")
    ref = _write_bytes(tmp_path / "r.csv", "a,b\n3,4\n1,2\n")
    report = compare_files(cand, ref, tmp_path / "out", prefix_rows=2)
    assert report["statuses"]["full_table_raw_match"] is False
    assert report["statuses"]["full_same_rows_different_order"] is True
    assert report["order_insensitive"]["full"]["status"] == "same_rows_different_order"


def test_duplicate_multiplicity_differs(tmp_path):
    cand = _write_bytes(tmp_path / "c.csv", "a,b\n1,2\n1,2\n")
    ref = _write_bytes(tmp_path / "r.csv", "a,b\n1,2\n")
    report = compare_files(cand, ref, tmp_path / "out", prefix_rows=1)
    assert report["positional"]["full"]["comparable"] is False
    assert report["statuses"]["prefix_raw_match"] is True
    assert report["order_insensitive"]["full"]["status"] == "row_or_multiplicity_differences"


def test_duplicate_same_length_multiplicity(tmp_path):
    cand = _write_bytes(tmp_path / "c.csv", "a,b\n1,2\n1,2\n3,4\n")
    ref = _write_bytes(tmp_path / "r.csv", "a,b\n1,2\n3,4\n3,4\n")
    report = compare_files(cand, ref, tmp_path / "out", prefix_rows=3)
    assert report["statuses"]["full_table_raw_match"] is False
    assert report["order_insensitive"]["full"]["status"] == "row_or_multiplicity_differences"
    counts = (tmp_path / "out" / "row_count_differences.csv").read_text(encoding="utf-8")
    assert "candidate_count" in counts
    assert counts.count("\n") > 1


def test_prefix_match_longer_reference(tmp_path):
    cand = _write_bytes(tmp_path / "c.csv", "a,b\n1,2\n3,4\n5,6\n")
    ref = _write_bytes(tmp_path / "r.csv", "a,b\n1,2\n3,4\n5,6\n7,8\n")
    report = compare_files(cand, ref, tmp_path / "out", prefix_rows=3)
    assert report["statuses"]["full_table_raw_match"] is False
    assert report["positional"]["full"]["comparable"] is False
    assert report["statuses"]["prefix_raw_match"] is True
    assert report["statuses"]["prefix_content_match"] is True


def test_missing_not_equal_to_illegal_string(tmp_path):
    cand = _write_bytes(tmp_path / "c.csv", "a,b\n,2\n")
    ref = _write_bytes(tmp_path / "r.csv", "a,b\nnot_a_number,2\n")
    report = compare_files(cand, ref, tmp_path / "out", prefix_rows=1)
    assert report["statuses"]["full_table_raw_match"] is False
    assert report["statuses"]["full_table_format_match"] is False
    kinds = (tmp_path / "out" / "cell_differences.csv").read_text(encoding="utf-8")
    assert "missing_vs_nonmissing" in kinds
    assert report["positional"]["full"]["counts"]["missing_vs_nonmissing"] == 1


def test_numeric_difference(tmp_path):
    cand = _write_bytes(tmp_path / "c.csv", "a,b\n1,2\n")
    ref = _write_bytes(tmp_path / "r.csv", "a,b\n9,2\n")
    report = compare_files(cand, ref, tmp_path / "out", prefix_rows=1)
    assert report["statuses"]["content_differences"] is True
    assert report["positional"]["full"]["counts"]["numeric_difference"] == 1
    md = (tmp_path / "out" / "comparison.md").read_text(encoding="utf-8")
    assert "sampling population" in md or "does not" in md.lower()


def test_nonempty_out_dir_rejected(tmp_path):
    cand = _write_bytes(tmp_path / "c.csv", "a,b\n1,2\n")
    ref = _write_bytes(tmp_path / "r.csv", "a,b\n1,2\n")
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "old.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(OutputDirError, match="not empty"):
        compare_files(cand, ref, occupied, prefix_rows=1)
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--candidate",
            str(cand),
            "--reference",
            str(ref),
            "--out-dir",
            str(occupied),
            "--prefix-rows",
            "1",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "not empty" in proc.stderr
    assert (occupied / "old.txt").read_text(encoding="utf-8") == "keep"


def test_inputs_unchanged_on_disk(tmp_path):
    cand = _write_bytes(tmp_path / "c.csv", "a,b\n1,2\n")
    ref = _write_bytes(tmp_path / "r.csv", "a,b\n1,2\n")
    h0c, h0r = sha_file(cand), sha_file(ref)
    compare_files(cand, ref, tmp_path / "out", prefix_rows=1)
    assert sha_file(cand) == h0c
    assert sha_file(ref) == h0r
