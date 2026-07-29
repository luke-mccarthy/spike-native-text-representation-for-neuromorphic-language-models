#!/usr/bin/env python3
"""Measure one matched-corpus SentencePiece fitting job on a Slurm compute node."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import sentencepiece as spm

from scripts.make_2026_07_22_decisive_frontend_grids import read_fit_grid_row
from src.canonical_data_contract import (
    assert_training_only_contract,
    canonical_training_relative_path,
)


CANONICAL_ROOT = Path(
    "/scratch/user/lukemccarthy/datasets/spike_tokenizer_canonical_2026_07_16"
)
SPECS = {
    "bpe8000": ("bpe", 8000),
    "bpe16000": ("bpe", 16000),
    "unigram4000": ("unigram", 4000),
}
SENTENCEPIECE_DEFAULT_MAX_SENTENCE_LENGTH = 4_192


def sentencepiece_fit_settings(
    *,
    model_type: str,
    vocab_size: int,
    training_selection_raw_bytes: int,
) -> dict[str, object]:
    """Return lossless settings that accept an unbroken canonical prefix."""
    if int(training_selection_raw_bytes) <= 0:
        raise ValueError("training selection must be non-empty")
    if model_type not in {"bpe", "unigram"} or int(vocab_size) <= 0:
        raise ValueError("invalid SentencePiece fit specification")
    return {
        "model_type": model_type,
        "vocab_size": int(vocab_size),
        "character_coverage": 1.0,
        "normalization_rule_name": "identity",
        "add_dummy_prefix": False,
        "remove_extra_whitespaces": False,
        "byte_fallback": True,
        "hard_vocab_limit": False,
        "bos_id": -1,
        "eos_id": -1,
        "pad_id": -1,
        "unk_id": 0,
        "shuffle_input_sentence": False,
        "num_threads": 1,
        "input_sentence_size": 0,
        # Text8 is canonically one unbroken line. Raising this parser limit
        # preserves that exact byte stream; inserting synthetic newlines would
        # alter the tokenizer-fitting corpus and violate the hash contract.
        "max_sentence_length": max(
            SENTENCEPIECE_DEFAULT_MAX_SENTENCE_LENGTH,
            int(training_selection_raw_bytes) + 1,
        ),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_one(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted(row))
            writer.writeheader()
            writer.writerow(row)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.{os.getpid()}.tmp")
    try:
        shutil.copy2(source, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(value, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def utf8_safe_prefix(raw: bytes, requested_bytes: int) -> bytes:
    if requested_bytes <= 0:
        raise ValueError("training-byte-budget must be positive")
    selected = raw[: min(requested_bytes, len(raw))]
    try:
        selected.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        if exc.end != len(selected):
            raise AssertionError("canonical UTF-8 failed before prefix boundary") from exc
        selected = selected[: exc.start]
        selected.decode("utf-8", errors="strict")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", required=True, type=Path)
    parser.add_argument("--task-id", required=True, type=int)
    parser.add_argument("--fit-root", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    grid_row = read_fit_grid_row(args.grid, args.task_id)
    dataset = grid_row["dataset"]
    method = grid_row["method"]
    data_tier = grid_row["data_tier"]
    training_byte_budget = int(grid_row["training_byte_budget"])
    if dataset not in {"tinystories_5000", "text8", "wikitext2"}:
        raise ValueError(f"unsupported decisive dataset: {dataset}")
    if method not in SPECS:
        raise ValueError(f"unsupported decisive tokenizer: {method}")
    training_path = CANONICAL_ROOT / canonical_training_relative_path(
        dataset=dataset, data_tier=data_tier
    )
    contract = assert_training_only_contract(
        canonical_root=CANONICAL_ROOT,
        dataset=dataset,
        data_tier=data_tier,
        training_path=training_path,
    )
    model_output = args.fit_root / "models" / f"{dataset}_{method}.model"
    output = args.fit_root / "parts" / f"{dataset}_{method}.csv"
    model_type, vocab_size = SPECS[method]
    training_source_raw = training_path.read_bytes()
    training_selection = utf8_safe_prefix(
        training_source_raw, training_byte_budget
    )
    training_selection_sha256 = hashlib.sha256(training_selection).hexdigest()
    training_selection_path = (
        f"{training_path.resolve()}#utf8_bytes[0:{len(training_selection)}]"
    )
    settings = sentencepiece_fit_settings(
        model_type=model_type,
        vocab_size=vocab_size,
        training_selection_raw_bytes=len(training_selection),
    )
    with tempfile.TemporaryDirectory(prefix="decisive_spm_fit_") as directory:
        directory_path = Path(directory)
        fit_input = directory_path / "training_selection.txt"
        fit_input.write_bytes(training_selection)
        prefix = directory_path / "matched"
        wall_started = time.perf_counter()
        cpu_started = time.process_time()
        spm.SentencePieceTrainer.train(
            input=str(fit_input),
            model_prefix=str(prefix),
            **settings,
        )
        cpu_seconds = time.process_time() - cpu_started
        wall_seconds = time.perf_counter() - wall_started
        model_path = prefix.with_suffix(".model")
        processor = spm.SentencePieceProcessor(model_file=str(model_path))
        sample = training_selection.decode("utf-8", errors="strict")[:100_000]
        ids = processor.encode(sample, out_type=int)
        if processor.decode(ids) != sample:
            raise AssertionError("matched SentencePiece fit failed sample round-trip")
        atomic_copy(model_path, model_output)
        provenance = {
            "format": "decisive_exact_selection_sentencepiece_provenance_v1",
            "protocol_id": "canonical_frontend_protocol_2026_07_16_v1",
            "dataset": dataset,
            "method": method,
            "data_tier": data_tier,
            "git_commit": args.expected_commit,
            "training_source_path": str(training_path.resolve()),
            "training_source_sha256": sha256(training_path),
            "training_selection_path": training_selection_path,
            "training_selection_start_raw_byte": 0,
            "training_selection_end_raw_byte_exclusive": len(training_selection),
            "training_selection_raw_bytes": len(training_selection),
            "training_selection_sha256": training_selection_sha256,
            "test_split_touched": 0,
        }
        atomic_write_text(
            model_output.with_suffix(".train.json"),
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        )
        row = {
            "format": "matched_sentencepiece_fit_benchmark_v1",
            "status": "PASS",
            "dataset": dataset,
            "method": method,
            "data_tier": data_tier,
            "task_id": args.task_id,
            "git_commit": args.expected_commit,
            "training_path": training_selection_path,
            "training_sha256": training_selection_sha256,
            "training_source_path": str(training_path.resolve()),
            "training_source_sha256": sha256(training_path),
            "training_selection_raw_bytes": len(training_selection),
            "training_selection_utf8_complete": 1,
            "canonical_training_sha256": contract["canonical_train_sha256"],
            "canonical_validation_opened": 0,
            "test_split_touched": 0,
            "fit_cpu_seconds": cpu_seconds,
            "fit_wall_seconds": wall_seconds,
            "fitted_model_path": str(model_output.resolve()),
            "fitted_model_file_bytes": model_output.stat().st_size,
            "fitted_model_sha256": sha256(model_output),
            "fitted_piece_count": processor.get_piece_size(),
            "fit_settings_json": json.dumps(settings, sort_keys=True, separators=(",", ":")),
            "fit_settings_sha256": hashlib.sha256(
                json.dumps(settings, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "interpretation": (
                "matched_corpus_vocab_and_model_type_retraining_benchmark_"
                "not_historical_artifact_fit_measurement"
            ),
        }
        if row["training_source_sha256"] != row["canonical_training_sha256"]:
            raise AssertionError("SentencePiece fit benchmark used the wrong training source")
        # The row is the completion sentinel and is published last, after the
        # model and provenance have both been atomically replaced.
        write_one(output, row)
    print(
        f"PASS matched SentencePiece fit dataset={dataset} method={method} "
        f"cpu_seconds={cpu_seconds:.6f}"
    )


if __name__ == "__main__":
    main()
