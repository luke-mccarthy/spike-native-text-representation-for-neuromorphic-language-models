#!/usr/bin/env python3
"""Validate frozen grids, raw result archives, audits, and checksums."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FRONT_END_COMMIT = "95f148159b9b340c2f679c0d10eedf1e35b936f9"
DECODER_COMMIT = "e20afbd166099210cca18be089e6aae447f6e56a"
EXPECTED_HASHES = {
    "configs/front_end_test.csv":
        "27a5e8108796bd10eb5f039449570d13595105114ee31d9b7f96b3f59de4e787",
    "configs/temporal_decoder_test.csv":
        "0adb5b3495873125e0a3da766194f38c0bca131c0dee2296c295d281d440a1ca",
    "configs/tokenizer_fits.csv":
        "b20fdb94ab5a0cbd95d939072ebdb8d89a41d0f0d49823733ae6fc8923db7135",
    "results/front_end/all_rows.csv":
        "5973bc8d1175fa403858606a10b39651cdfe8f138b9142495a4c3ddc8e1d60a2",
    "results/front_end/audit.json":
        "f2d717c16cabe3d330fc4d12620d3d6f34620e44f444870bdcb0d6123424ac29",
    "results/temporal_decoders/all_rows.csv":
        "70e64ea88cd49856572121cb7d8e67879a8e7de7c37d84c51523ab4abe5366c2",
    "results/temporal_decoders/audit.json":
        "82495d4405905edeef5137b7aa4469f8ecabaac3f1e2c978eb68fb195e82eec6",
    "results/temporal_decoders/grid_manifest.json":
        "40c4c5c171fbfa36a8996b533cb654e2f6bab573c3f0761d4b2931cf2f94dabb",
}


def csv_rows(path: str) -> list[dict[str, str]]:
    with (ROOT / path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha1(path: Path) -> str:
    content = path.read_bytes()
    header = f"blob {len(content)}\0".encode("ascii")
    return hashlib.sha1(header + content).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_front_end() -> None:
    grid = csv_rows("configs/front_end_test.csv")
    rows = csv_rows("results/front_end/all_rows.csv")
    audit = json.loads((ROOT / "results/front_end/audit.json").read_text())
    require(len(grid) == 90, "front-end grid must contain 90 rows")
    require(len(rows) == 90, "front-end archive must contain 90 rows")
    require(audit["status"] == "PASS", "front-end audit did not pass")
    require(audit["valid_rows"] == 90, "front-end audit row count differs")
    require(
        {row["expected_commit"] for row in rows} == {FRONT_END_COMMIT},
        "front-end rows do not identify the frozen runtime commit",
    )
    require(
        [row["run_name"] for row in grid] == [row["run_name"] for row in rows],
        "front-end grid and raw rows are not aligned",
    )
    require(
        {row["status"] for row in rows} <= {"", "PASS"},
        "front-end archive contains an explicit non-PASS row",
    )


def validate_temporal_decoders() -> None:
    grid = csv_rows("configs/temporal_decoder_test.csv")
    rows = csv_rows("results/temporal_decoders/all_rows.csv")
    audit = json.loads(
        (ROOT / "results/temporal_decoders/audit.json").read_text()
    )
    manifest = json.loads(
        (ROOT / "results/temporal_decoders/grid_manifest.json").read_text()
    )
    require(len(grid) == 162, "temporal-decoder grid must contain 162 rows")
    require(len(rows) == 162, "temporal-decoder archive must contain 162 rows")
    require(audit["status"] == "PASS", "temporal-decoder audit did not pass")
    require(audit["valid_rows"] == 162, "temporal-decoder audit row count differs")
    require(manifest["status"] == "PASS", "temporal-decoder grid did not pass")
    require(manifest["rows"] == 162, "temporal-decoder grid count differs")
    require(
        {row["expected_commit"] for row in rows} == {DECODER_COMMIT},
        "temporal-decoder rows do not identify the frozen runtime commit",
    )
    require(
        [row["run_name"] for row in grid] == [row["run_name"] for row in rows],
        "temporal-decoder grid and raw rows are not aligned",
    )
    require(
        {row["status"] for row in rows} == {"PASS"},
        "temporal-decoder archive contains a non-PASS row",
    )
    require(
        {row["dataset"] for row in rows}
        == {"text8", "tinystories_5000", "wikitext2"},
        "temporal-decoder dataset factorial differs",
    )
    require(
        {row["seed"] for row in rows} == {"1", "2", "3"},
        "temporal-decoder seed set differs",
    )


def validate_hashes() -> None:
    for relative, expected in EXPECTED_HASHES.items():
        actual = sha256(ROOT / relative)
        require(actual == expected, f"SHA-256 mismatch for {relative}")


def validate_source_manifest() -> None:
    manifest = json.loads((ROOT / "docs/source_manifest.json").read_text())
    patched = {
        row["path"]: row["original_copy"]
        for row in manifest["portability_patches"]
    }
    require(len(manifest["files"]) == 38, "source manifest file count differs")
    for relative, expected_blob in manifest["files"].items():
        runtime = ROOT / "paper_code" / relative
        require(runtime.is_file(), f"paper runtime file is missing: {relative}")
        identity_path = (
            ROOT / patched[relative]
            if relative in patched
            else runtime
        )
        require(
            git_blob_sha1(identity_path) == expected_blob,
            f"source blob identity differs: {relative}",
        )


def main() -> None:
    validate_hashes()
    validate_source_manifest()
    validate_front_end()
    validate_temporal_decoders()
    print("PASS: release grids, raw archives, audits, and hashes are consistent")


if __name__ == "__main__":
    main()
