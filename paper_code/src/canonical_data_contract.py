"""Fail-closed canonical corpus and tokenizer provenance checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Mapping


PROTOCOL_ID = "canonical_frontend_protocol_2026_07_16_v1"


CANONICAL_TRAINING_RELATIVE_PATHS = {
    "development": {
        "tinystories_5000": (
            "tinystories_5000/development/train_1000_stories.txt"
        ),
        "text8": "text8/development/train_5m.txt",
        "wikitext2": "wikitext2/development/train_5m.txt",
    },
    "full_train_validation": {
        dataset: f"{dataset}/train.txt"
        for dataset in ("tinystories_5000", "text8", "wikitext2")
    },
}


def canonical_training_relative_path(*, dataset: str, data_tier: str) -> str:
    """Return the one manifest-declared training path for a tier/dataset."""
    try:
        return CANONICAL_TRAINING_RELATIVE_PATHS[data_tier][dataset]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported canonical training selection: {data_tier}/{dataset}"
        ) from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_files(manifest: Mapping[str, object]) -> Dict[str, Mapping[str, object]]:
    rows = manifest.get("files")
    if not isinstance(rows, list):
        raise ValueError("Canonical manifest is missing its file list")
    files: Dict[str, Mapping[str, object]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Canonical manifest contains an invalid file row")
        files[str(row["relative_path"])] = row
    return files


def assert_train_validation_contract(
    *,
    canonical_root: Path,
    dataset: str,
    data_tier: str,
    training_path: Path,
    validation_path: Path,
) -> Dict[str, object]:
    """Verify exact manifest paths, roles, hashes, and absence of test input."""
    root = canonical_root.resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(
            f"Unexpected canonical protocol: {manifest.get('protocol_id')!r}"
        )
    expected_relative = {
        "development": {
            "train": canonical_training_relative_path(
                dataset=dataset, data_tier="development"
            ),
            "validation": {
                "tinystories_5000": (
                    "tinystories_5000/development/validation_100_stories.txt"
                ),
                "text8": "text8/development/validation_1m.txt",
                "wikitext2": "wikitext2/development/validation_1m.txt",
            }[dataset],
        },
        "full_train_validation": {
            "train": canonical_training_relative_path(
                dataset=dataset, data_tier="full_train_validation"
            ),
            "validation": f"{dataset}/validation.txt",
        },
    }
    if data_tier not in expected_relative:
        raise ValueError(f"Unsupported data tier: {data_tier}")
    expected = expected_relative[data_tier]
    actual_paths = {
        "train": training_path.resolve(),
        "validation": validation_path.resolve(),
    }
    files = _manifest_files(manifest)
    output: Dict[str, object] = {
        "canonical_manifest_path": str(manifest_path),
        "canonical_manifest_sha256": sha256_file(manifest_path),
        "canonical_split_contract_pass": 1,
        "canonical_test_split_touched": 0,
    }
    for split in ("train", "validation"):
        expected_path = (root / expected[split]).resolve()
        if actual_paths[split] != expected_path:
            raise AssertionError(
                f"{split} path is not the declared {data_tier} canonical split: "
                f"{actual_paths[split]} != {expected_path}"
            )
        if "test" in expected[split].lower():
            raise AssertionError("Development/validation contract selected test data")
        record = files.get(expected[split])
        if record is None:
            raise AssertionError(
                f"Canonical manifest lacks {expected[split]}"
            )
        expected_role = (
            f"development_{split}"
            if data_tier == "development"
            else f"confirmation_{split}"
        )
        if str(record.get("role")) != expected_role:
            raise AssertionError(
                f"Canonical {split} role is {record.get('role')!r}, "
                f"expected {expected_role!r}"
            )
        digest = sha256_file(expected_path)
        size = expected_path.stat().st_size
        if digest != str(record["sha256"]) or size != int(record["bytes"]):
            raise AssertionError(
                f"Canonical {split} file differs from its immutable manifest"
            )
        output[f"canonical_{split}_relative_path"] = expected[split]
        output[f"canonical_{split}_role"] = str(record["role"])
        output[f"canonical_{split}_sha256"] = digest
        output[f"canonical_{split}_bytes"] = size
    if output["canonical_train_sha256"] == output["canonical_validation_sha256"]:
        raise AssertionError("Training and validation hashes unexpectedly match")
    return output


def assert_train_test_contract(
    *,
    canonical_root: Path,
    dataset: str,
    data_tier: str,
    training_path: Path,
    test_path: Path,
) -> Dict[str, object]:
    """Verify the immutable confirmation train/test pair without opening validation.

    This contract is intentionally separate from development/validation.  It is
    only for a pre-authorised final evaluation after configuration selection is
    frozen.  The function hashes and stats exactly the manifest, training file
    and test file; it neither resolves nor opens the validation payload.
    """

    if data_tier != "full_train_validation":
        raise AssertionError("confirmatory test requires the full canonical training tier")
    root = canonical_root.resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(
            f"Unexpected canonical protocol: {manifest.get('protocol_id')!r}"
        )
    expected = {
        "train": f"{dataset}/train.txt",
        "test": f"{dataset}/test.txt",
    }
    actual = {
        "train": training_path.resolve(),
        "test": test_path.resolve(),
    }
    files = _manifest_files(manifest)
    output: Dict[str, object] = {
        "canonical_manifest_path": str(manifest_path),
        "canonical_manifest_sha256": sha256_file(manifest_path),
        "canonical_split_contract_pass": 1,
        "canonical_validation_opened": 0,
        "canonical_test_split_touched": 1,
    }
    for split in ("train", "test"):
        expected_path = (root / expected[split]).resolve()
        if actual[split] != expected_path:
            raise AssertionError(
                f"{split} path is not the declared confirmation split: "
                f"{actual[split]} != {expected_path}"
            )
        record = files.get(expected[split])
        if record is None:
            raise AssertionError(f"Canonical manifest lacks {expected[split]}")
        expected_role = f"confirmation_{split}"
        if str(record.get("role")) != expected_role:
            raise AssertionError(
                f"Canonical {split} role is {record.get('role')!r}, "
                f"expected {expected_role!r}"
            )
        digest = sha256_file(expected_path)
        size = expected_path.stat().st_size
        if digest != str(record["sha256"]) or size != int(record["bytes"]):
            raise AssertionError(
                f"Canonical {split} file differs from its immutable manifest"
            )
        output[f"canonical_{split}_relative_path"] = expected[split]
        output[f"canonical_{split}_role"] = str(record["role"])
        output[f"canonical_{split}_sha256"] = digest
        output[f"canonical_{split}_bytes"] = size
    if output["canonical_train_sha256"] == output["canonical_test_sha256"]:
        raise AssertionError("Training and test hashes unexpectedly match")
    output.update({
        "canonical_validation_relative_path": "not_opened_confirmatory_test",
        "canonical_validation_role": "not_opened_confirmatory_test",
        "canonical_validation_sha256": "not_opened_confirmatory_test",
        "canonical_validation_bytes": "not_opened_confirmatory_test",
    })
    return output


def assert_training_only_contract(
    *,
    canonical_root: Path,
    dataset: str,
    data_tier: str,
    training_path: Path,
) -> Dict[str, object]:
    """Verify only the immutable training split for nested train-only tuning.

    This function deliberately neither hashes nor stats the canonical
    validation or test path. Its output makes that narrower scope explicit.
    """

    root = canonical_root.resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(
            f"Unexpected canonical protocol: {manifest.get('protocol_id')!r}"
        )
    expected = canonical_training_relative_path(
        dataset=dataset, data_tier=data_tier
    )
    expected_path = (root / expected).resolve()
    if training_path.resolve() != expected_path:
        raise AssertionError(
            "training path is not the declared canonical training split"
        )
    record = _manifest_files(manifest).get(expected)
    if record is None:
        raise AssertionError(f"Canonical manifest lacks {expected}")
    expected_role = (
        "development_train"
        if data_tier == "development"
        else "confirmation_train"
    )
    if str(record.get("role")) != expected_role:
        raise AssertionError(
            f"Canonical training role is {record.get('role')!r}, "
            f"expected {expected_role!r}"
        )
    digest = sha256_file(expected_path)
    size = expected_path.stat().st_size
    if digest != str(record["sha256"]) or size != int(record["bytes"]):
        raise AssertionError("Canonical training file differs from its manifest")
    return {
        "canonical_manifest_path": str(manifest_path),
        "canonical_manifest_sha256": sha256_file(manifest_path),
        "canonical_split_contract_pass": 1,
        "canonical_training_only_contract": 1,
        "canonical_train_relative_path": expected,
        "canonical_train_role": str(record["role"]),
        "canonical_train_sha256": digest,
        "canonical_train_bytes": size,
        "canonical_validation_relative_path": (
            "not_opened_nested_training_selection"
        ),
        "canonical_validation_role": "not_opened_nested_training_selection",
        "canonical_validation_sha256": "not_opened_nested_training_selection",
        "canonical_validation_bytes": "not_opened_nested_training_selection",
        "canonical_test_split_touched": 0,
    }


def sentencepiece_training_provenance(
    *,
    model_path: Path,
    dataset: str,
    expected_training_path: Path,
    expected_training_sha256: str,
) -> Dict[str, object]:
    """Require a tokenizer to have been fitted on the exact LM training split."""
    record_path = model_path.with_suffix(".train.json")
    if not record_path.is_file():
        raise FileNotFoundError(
            f"SentencePiece model lacks training provenance: {record_path}"
        )
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("protocol_id") != PROTOCOL_ID:
        raise AssertionError("Tokenizer protocol does not match the LM protocol")
    if record.get("dataset") != dataset:
        raise AssertionError("Tokenizer dataset does not match the LM dataset")
    if str(record.get("training_sha256")) != str(expected_training_sha256):
        raise AssertionError(
            "Tokenizer and LM/front-end fitting corpora have different hashes"
        )
    if Path(str(record.get("training_path"))).resolve() != expected_training_path.resolve():
        raise AssertionError(
            "Tokenizer and LM/front-end fitting corpora have different paths"
        )
    if sha256_file(expected_training_path) != expected_training_sha256:
        raise AssertionError("Expected tokenizer fitting file changed after hashing")
    return {
        "tokenizer_training_provenance_path": str(record_path),
        "tokenizer_training_path": str(expected_training_path.resolve()),
        "tokenizer_training_sha256": expected_training_sha256,
        "tokenizer_fit_corpus_matches_lm_training_corpus": 1,
    }
