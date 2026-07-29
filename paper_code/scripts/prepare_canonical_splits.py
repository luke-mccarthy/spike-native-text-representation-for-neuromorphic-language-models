#!/usr/bin/env python3
"""Prepare immutable, auditable train/validation/test corpus splits.

This script is deliberately independent of the model and tokenizer code.  It
splits raw data before any tokenizer, boundary rule, vocabulary, threshold, or
language model is fitted.  Existing output files are never silently replaced:
reruns verify byte-for-byte identity and fail if a file differs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


PROTOCOL_ID = "canonical_frontend_protocol_2026_07_16_v1"
MANIFEST_NAME = "manifest.json"
TINYSTORIES_MARKER = b"<|endoftext|>"
PUBLIC_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = Path(
    os.environ.get(
        "SPIKE_TOKENIZER_CANONICAL_ROOT",
        str(PUBLIC_ROOT / "data/canonical"),
    )
)


@dataclass(frozen=True)
class DocumentRecord:
    source_index: int
    content: bytes
    sha256: str


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def strict_utf8_character_count(data: bytes, label: str) -> int:
    try:
        return len(data.decode("utf-8", errors="strict"))
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not valid UTF-8: {exc}") from exc


def read_source(path: Path) -> bytes:
    if not path.is_file():
        raise FileNotFoundError(f"Required source file not found: {path}")
    data = path.read_bytes()
    if not data:
        raise ValueError(f"Required source file is empty: {path}")
    strict_utf8_character_count(data, str(path))
    return data


def source_record(path: Path, data: bytes) -> Dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": len(data),
        "characters": strict_utf8_character_count(data, str(path)),
        "sha256": sha256_bytes(data),
    }


def canonical_tinystories_documents(
    raw: bytes,
    expected_documents: int | None = 5000,
) -> List[bytes]:
    """Parse stories and remove only boundary-adjacent CR/LF characters."""
    pieces = raw.split(TINYSTORIES_MARKER)
    if pieces and not pieces[-1].strip(b"\r\n"):
        pieces = pieces[:-1]
    documents = [piece.strip(b"\r\n") for piece in pieces]
    if any(not document for document in documents):
        empty = [index for index, document in enumerate(documents) if not document]
        raise ValueError(f"TinyStories contains empty documents at indices {empty[:10]}")
    if any(TINYSTORIES_MARKER in document for document in documents):
        raise AssertionError("TinyStories marker remained inside a parsed document")
    if expected_documents is not None and len(documents) != expected_documents:
        raise ValueError(
            f"Expected {expected_documents} TinyStories documents, found {len(documents)}"
        )
    return documents


def deterministic_document_split(
    documents: Sequence[bytes],
    seed: int,
) -> Tuple[Dict[str, List[DocumentRecord]], Dict[str, object]]:
    """Deduplicate exact documents, order deterministically, and split 80/10/10."""
    unique: Dict[str, DocumentRecord] = {}
    duplicates: List[Dict[str, object]] = []
    for source_index, content in enumerate(documents):
        digest = sha256_bytes(content)
        if digest in unique:
            duplicates.append(
                {
                    "source_index": source_index,
                    "duplicate_of_source_index": unique[digest].source_index,
                    "sha256": digest,
                }
            )
            continue
        unique[digest] = DocumentRecord(source_index, content, digest)

    seed_bytes = str(seed).encode("ascii")

    def order_key(record: DocumentRecord) -> str:
        return sha256_bytes(seed_bytes + b":" + bytes.fromhex(record.sha256))

    ordered = sorted(unique.values(), key=order_key)
    validation_count = len(ordered) // 10
    test_count = len(ordered) // 10
    train_count = len(ordered) - validation_count - test_count
    split = {
        "train": ordered[:train_count],
        "validation": ordered[train_count : train_count + validation_count],
        "test": ordered[train_count + validation_count :],
    }

    hash_sets = {name: {record.sha256 for record in rows} for name, rows in split.items()}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = hash_sets[left] & hash_sets[right]
        if overlap:
            raise AssertionError(f"Exact TinyStories document leakage: {left}/{right}")

    audit = {
        "source_document_count": len(documents),
        "unique_document_count": len(ordered),
        "exact_duplicate_count_removed": len(duplicates),
        "exact_duplicates": duplicates,
        "split_document_counts": {name: len(rows) for name, rows in split.items()},
        "pairwise_exact_document_hash_overlap": {
            "train_validation": 0,
            "train_test": 0,
            "validation_test": 0,
        },
    }
    return split, audit


def join_tinystories(records: Sequence[DocumentRecord]) -> bytes:
    if not records:
        raise ValueError("Cannot write an empty TinyStories split")
    separator = b"\n" + TINYSTORIES_MARKER + b"\n"
    return separator.join(record.content for record in records) + b"\n"


def assignment_csv(split: Mapping[str, Sequence[DocumentRecord]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=("split", "order_in_split", "source_index", "bytes", "sha256"),
        lineterminator="\n",
    )
    writer.writeheader()
    for split_name in ("train", "validation", "test"):
        for order, record in enumerate(split[split_name]):
            writer.writerow(
                {
                    "split": split_name,
                    "order_in_split": order,
                    "source_index": record.source_index,
                    "bytes": len(record.content),
                    "sha256": record.sha256,
                }
            )
    return buffer.getvalue().encode("utf-8")


def split_text8(
    raw: bytes,
    expected_bytes: int = 100_000_000,
    train_bytes: int = 90_000_000,
    validation_bytes: int = 5_000_000,
) -> Dict[str, bytes]:
    if len(raw) != expected_bytes:
        raise ValueError(f"Expected text8 to contain {expected_bytes} bytes, found {len(raw)}")
    if train_bytes + validation_bytes >= expected_bytes:
        raise ValueError("text8 split sizes leave no test data")
    return {
        "train": raw[:train_bytes],
        "validation": raw[train_bytes : train_bytes + validation_bytes],
        "test": raw[train_bytes + validation_bytes :],
    }


def valid_utf8_prefix(data: bytes, maximum_bytes: int) -> bytes:
    """Return at most maximum_bytes without cutting a UTF-8 code point."""
    prefix = data[:maximum_bytes]
    while prefix:
        try:
            prefix.decode("utf-8", errors="strict")
            return prefix
        except UnicodeDecodeError as exc:
            if exc.start < len(prefix) - 4:
                raise
            prefix = prefix[:-1]
    return prefix


def file_record(
    relative_path: str,
    data: bytes,
    role: str,
    derived_from: str | None = None,
) -> Dict[str, object]:
    record: Dict[str, object] = {
        "relative_path": relative_path,
        "role": role,
        "bytes": len(data),
        "characters": strict_utf8_character_count(data, relative_path),
        "sha256": sha256_bytes(data),
    }
    if derived_from is not None:
        record["derived_from"] = derived_from
    return record


def write_or_verify(path: Path, data: bytes, verify_only: bool = False) -> str:
    """Atomically create a file, or verify an identical existing file."""
    if path.exists():
        if not path.is_file():
            raise ValueError(f"Expected a file but found another object: {path}")
        existing = path.read_bytes()
        if existing != data:
            raise ValueError(
                f"Refusing to replace non-identical canonical file: {path}\n"
                "Use a new versioned output directory after reviewing the protocol change."
            )
        return "verified"
    if verify_only:
        raise FileNotFoundError(f"Canonical file is missing during verification: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(data)
    temporary.replace(path)
    return "created"


def prepare(args: argparse.Namespace) -> Tuple[Dict[str, object], Dict[str, bytes]]:
    text8_path = Path(args.text8_full)
    tiny_path = Path(args.tinystories_5000)
    wiki_paths = {
        "train": Path(args.wikitext2_train),
        "validation": Path(args.wikitext2_validation),
        "test": Path(args.wikitext2_test),
    }

    text8_raw = read_source(text8_path)
    tiny_raw = read_source(tiny_path)
    wiki_raw = {name: read_source(path) for name, path in wiki_paths.items()}

    text8 = split_text8(text8_raw)
    tiny_documents = canonical_tinystories_documents(tiny_raw)
    tiny, tiny_audit = deterministic_document_split(tiny_documents, args.tinystories_seed)

    outputs: Dict[str, bytes] = {}
    roles: Dict[str, Tuple[str, str | None]] = {}

    def add(relative_path: str, data: bytes, role: str, parent: str | None = None) -> None:
        if relative_path in outputs:
            raise AssertionError(f"Duplicate output path: {relative_path}")
        outputs[relative_path] = data
        roles[relative_path] = (role, parent)

    for split_name, data in text8.items():
        add(f"text8/{split_name}.txt", data, f"confirmation_{split_name}")
    add(
        "text8/development/train_5m.txt",
        text8["train"][:5_000_000],
        "development_train",
        "text8/train.txt",
    )
    add(
        "text8/development/validation_1m.txt",
        text8["validation"][:1_000_000],
        "development_validation",
        "text8/validation.txt",
    )

    for split_name, data in wiki_raw.items():
        add(f"wikitext2/{split_name}.txt", data, f"confirmation_{split_name}")
    add(
        "wikitext2/development/train_5m.txt",
        valid_utf8_prefix(wiki_raw["train"], 5_000_000),
        "development_train",
        "wikitext2/train.txt",
    )
    add(
        "wikitext2/development/validation_1m.txt",
        valid_utf8_prefix(wiki_raw["validation"], 1_000_000),
        "development_validation",
        "wikitext2/validation.txt",
    )

    for split_name, records in tiny.items():
        add(
            f"tinystories_5000/{split_name}.txt",
            join_tinystories(records),
            f"confirmation_{split_name}",
        )
    add(
        "tinystories_5000/development/train_1000_stories.txt",
        join_tinystories(tiny["train"][: min(1000, len(tiny["train"]))]),
        "development_train",
        "tinystories_5000/train.txt",
    )
    add(
        "tinystories_5000/development/validation_100_stories.txt",
        join_tinystories(tiny["validation"][: min(100, len(tiny["validation"]))]),
        "development_validation",
        "tinystories_5000/validation.txt",
    )
    add(
        "tinystories_5000/document_assignments.csv",
        assignment_csv(tiny),
        "split_audit",
    )

    output_root = Path(args.output_root).resolve()
    manifest: Dict[str, object] = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "output_root": str(output_root),
        "non_negotiable_rules": {
            "split_raw_data_before_any_fit": True,
            "fit_tokenizers_frontends_and_codecs_on_train_only": True,
            "freeze_fitted_components_for_validation_and_test": True,
            "validation_for_selection": True,
            "test_locked_until_final_configuration": True,
            "development_views_do_not_include_test": True,
        },
        "sources": {
            "text8_full": source_record(text8_path, text8_raw),
            "tinystories_5000": source_record(tiny_path, tiny_raw),
            "wikitext2": {
                name: source_record(wiki_paths[name], wiki_raw[name])
                for name in ("train", "validation", "test")
            },
        },
        "split_rules": {
            "text8": {
                "rule": "standard contiguous 90,000,000/5,000,000/5,000,000 byte split",
                "boundaries": [0, 90_000_000, 95_000_000, 100_000_000],
            },
            "wikitext2": {"rule": "official train/validation/test source files"},
            "tinystories_5000": {
                "rule": "whole documents; exact-SHA256 deduplication; deterministic 80/10/10 split",
                "marker": TINYSTORIES_MARKER.decode("ascii"),
                "ordering_seed": args.tinystories_seed,
                **tiny_audit,
            },
        },
        "files": [
            file_record(path, outputs[path], roles[path][0], roles[path][1])
            for path in sorted(outputs)
        ],
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    outputs[MANIFEST_NAME] = manifest_bytes
    return manifest, outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--text8-full",
        required=True,
    )
    parser.add_argument(
        "--tinystories-5000",
        required=True,
    )
    parser.add_argument(
        "--wikitext2-train",
        required=True,
    )
    parser.add_argument(
        "--wikitext2-validation",
        required=True,
    )
    parser.add_argument(
        "--wikitext2-test",
        required=True,
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--tinystories-seed", type=int, default=20260716)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Reconstruct expected outputs and verify existing files without creating any.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest, outputs = prepare(args)
    output_root = Path(args.output_root).resolve()
    counts = {"created": 0, "verified": 0}
    for relative_path in sorted(outputs):
        status = write_or_verify(
            output_root / relative_path,
            outputs[relative_path],
            verify_only=args.verify_only,
        )
        counts[status] += 1

    print(f"Protocol: {manifest['protocol_id']}")
    print(f"Output root: {output_root}")
    print(f"Files created: {counts['created']}")
    print(f"Files verified: {counts['verified']}")
    print(
        "TinyStories documents (train/validation/test): "
        + "/".join(
            str(manifest["split_rules"]["tinystories_5000"]["split_document_counts"][name])
            for name in ("train", "validation", "test")
        )
    )
    print(f"Manifest: {output_root / MANIFEST_NAME}")


if __name__ == "__main__":
    main()
