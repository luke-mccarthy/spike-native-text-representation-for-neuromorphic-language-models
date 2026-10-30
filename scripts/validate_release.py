#!/usr/bin/env python3
"""Validate frozen grids, raw result archives, audits, and checksums."""

from __future__ import annotations

import csv
import hashlib
import json
import re
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
        "f7aa00ff8be391d496fa6c9d8dd8454d5a7fc232dd596a3df67c671a3af88e3a",
    "results/front_end/all_rows.csv":
        "5973bc8d1175fa403858606a10b39651cdfe8f138b9142495a4c3ddc8e1d60a2",
    "results/front_end/audit.json":
        "f2d717c16cabe3d330fc4d12620d3d6f34620e44f444870bdcb0d6123424ac29",
    "results/front_end/slurm_accounting.psv":
        "bec82ffa7882b520cea47c4cbf686e16db80896dfa2b67e84f59e8001d99820d",
    "results/temporal_decoders/all_rows.csv":
        "70e64ea88cd49856572121cb7d8e67879a8e7de7c37d84c51523ab4abe5366c2",
    "results/temporal_decoders/audit.json":
        "82495d4405905edeef5137b7aa4469f8ecabaac3f1e2c978eb68fb195e82eec6",
    "results/temporal_decoders/grid_manifest.json":
        "40c4c5c171fbfa36a8996b533cb654e2f6bab573c3f0761d4b2931cf2f94dabb",
    "results/temporal_decoders/SHA256SUMS":
        "7ce3de4f24a0a73b5d21a9ec8581d7fe22fa7bc47cae276c4067dc698c19b47a",
    "results/temporal_decoders/parts_sha256.txt":
        "0dbbca1f66dfcce3e4809c260aac50e02e971bfdd4dfbbfd2b3faa3001b4ed54",
    "results/temporal_decoders/slurm_accounting.psv":
        "9f22c70eebf1ddc224bca223360ae0da50061eac3a7a6fe3efb88a1ef618d1d5",
    "results/tokenizer_fits/audit.json":
        "278a49c7de36a53464a89469171f10b5f46c3e08732ab1dba268b3bacf6dc9f0",
    "results/tokenizer_fits/parts/text8_bpe16000.csv":
        "8c0048cd592d834d6c09086862c39804d6f185a888181cf9d0777b0dca25507a",
    "results/tokenizer_fits/parts/text8_bpe8000.csv":
        "60c9e712734a1dc554919529beeb2640044d04a227d735d6003cca914601b12d",
    "results/tokenizer_fits/parts/text8_unigram4000.csv":
        "7dc0ae40a6b660bf7df435d09de167e51be34478d14f84f485f70933527f22c5",
    "results/tokenizer_fits/parts/tinystories_5000_bpe16000.csv":
        "9ddcaaf42ea1e30046dc67ac79b0a07c42fd9f07d5bb45a942b5031e2bf50a44",
    "results/tokenizer_fits/parts/tinystories_5000_bpe8000.csv":
        "d84b95eee499a3f338b2b60fdf22b0b76f06ccf40891533ec1ef29f9ef775e6e",
    "results/tokenizer_fits/parts/tinystories_5000_unigram4000.csv":
        "68f595aadb9ba73b6b751014b4d914fe4e374c1aa0d67276eccfa58745701272",
    "results/tokenizer_fits/parts/wikitext2_bpe16000.csv":
        "a8ce55cde29d4db1b9164c97aca595e9a1aa9d0ae6822b8c0433b190d352ced6",
    "results/tokenizer_fits/parts/wikitext2_bpe8000.csv":
        "a2728408e5d07755035429d3ac014a8ca60c1df8e68b5943c6e4ab84babb9587",
    "results/tokenizer_fits/parts/wikitext2_unigram4000.csv":
        "755fbfebe59d462bd6327e65ed8ce7898bff5fc23c95b37d87ab06950d7526fc",
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


def validate_tokenizer_fits() -> None:
    grid = csv_rows("configs/tokenizer_fits.csv")
    part_paths = sorted((ROOT / "results/tokenizer_fits/parts").glob("*.csv"))
    rows = []
    for path in part_paths:
        rows.extend(csv_rows(str(path.relative_to(ROOT))))
    audit = json.loads((ROOT / "results/tokenizer_fits/audit.json").read_text())
    require(len(grid) == 9, "tokenizer-fit grid must contain 9 rows")
    require(len(part_paths) == 9, "tokenizer-fit archive must contain 9 parts")
    require(len(rows) == 9, "tokenizer-fit parts must contain 9 rows")
    require(audit["status"] == "PASS", "tokenizer-fit audit did not pass")
    require(audit["valid_rows"] == 9, "tokenizer-fit audit row count differs")
    require(audit["test_split_touched"] == 0, "tokenizer fitting touched test data")
    require(
        {row["task_id"] for row in rows} == {str(index) for index in range(1, 10)},
        "tokenizer-fit task identifiers differ",
    )


def validate_hashes() -> None:
    for relative, expected in EXPECTED_HASHES.items():
        actual = sha256(ROOT / relative)
        require(actual == expected, f"SHA-256 mismatch for {relative}")


def validate_source_manifest() -> None:
    manifest = json.loads((ROOT / "docs/source_manifest.json").read_text())
    patched = {
        row["runtime_path"]: row["original_copy"]
        for row in manifest["portability_changes"]
        if "original_copy" in row
    }
    require(
        manifest["format"] == "paper_code_source_manifest_v2",
        "unsupported source manifest format",
    )
    require(len(manifest["files"]) == 38, "source manifest file count differs")
    for relative, record in manifest["files"].items():
        runtime = ROOT / "paper_code" / relative
        require(runtime.is_file(), f"paper runtime file is missing: {relative}")
        require(
            sha256(runtime) == record["runtime_sha256"],
            f"public runtime checksum differs: {relative}",
        )
        require(record["source_path"], f"source path is missing: {relative}")
        require(
            re.fullmatch(r"[0-9a-f]{40}", record["source_blob_sha1"]) is not None,
            f"source blob identity is malformed: {relative}",
        )
        if relative in patched:
            identity_path = ROOT / patched[relative]
            require(
                git_blob_sha1(identity_path) == record["source_blob_sha1"],
                f"portability source blob identity differs: {relative}",
            )
        elif record["transformation"] == "unchanged":
            require(
                git_blob_sha1(runtime) == record["source_blob_sha1"],
                f"unchanged source blob identity differs: {relative}",
            )


def validate_canonical_data() -> None:
    """Compare local canonical data, if present, with the reference manifest."""
    local_manifest_path = ROOT / "data/canonical/manifest.json"
    if not local_manifest_path.is_file():
        print("SKIP: no local data/canonical/manifest.json")
        return
    local_files = {
        record["relative_path"]: record
        for record in json.loads(local_manifest_path.read_text())["files"]
    }
    reference_files = {
        record["relative_path"]: record
        for record in json.loads(
            (ROOT / "docs/canonical_file_manifest.json").read_text()
        )["files"]
    }
    mismatches = []
    for relative_path, reference in reference_files.items():
        local = local_files.get(relative_path)
        if local is None:
            mismatches.append(f"missing: {relative_path}")
        elif local["sha256"] != reference["sha256"] or local["bytes"] != reference["bytes"]:
            mismatches.append(
                f"hash/byte mismatch: {relative_path} "
                f"(local {local['bytes']} bytes / {local['sha256'][:12]}..., "
                f"reference {reference['bytes']} bytes / {reference['sha256'][:12]}...)"
            )
    if mismatches:
        detail = "\n  ".join(mismatches)
        raise AssertionError(
            "Local canonical data does not match docs/canonical_file_manifest.json:\n  "
            + detail
        )
    print(f"PASS: local canonical data matches docs/canonical_file_manifest.json for {len(reference_files)} files")


def validate_environment_evidence() -> None:
    evidence = json.loads((ROOT / "docs/bunya_environment.json").read_text())
    require(
        evidence["evidence_commit"]
        == "714d90b7f62e637c6f4682ded1ed55a2ab796fc5",
        "Bunya environment evidence commit differs",
    )
    require(
        evidence["captured_imports"]
        == {
            "numpy": "2.2.6",
            "sentencepiece": "0.2.1",
            "snntorch": "0.9.4",
            "torch": "2.11.0+cu126",
        },
        "captured import versions differ",
    )
    lock = (ROOT / "environment-lock.yml").read_text()
    require("/scratch/" not in lock, "archival environment contains a local path")
    require("snntorch==0.9.4" in lock, "archival environment omits snnTorch")
    require(
        "torch==2.11.0+cu126" in lock,
        "archival environment does not preserve the captured Torch build",
    )


def main() -> None:
    validate_hashes()
    validate_source_manifest()
    validate_environment_evidence()
    validate_tokenizer_fits()
    validate_front_end()
    validate_temporal_decoders()
    print("PASS: release grids, raw archives, audits, and hashes are consistent")
    validate_canonical_data()


if __name__ == "__main__":
    main()
