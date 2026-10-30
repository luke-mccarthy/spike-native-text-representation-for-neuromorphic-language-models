#!/usr/bin/env python3
"""Check a tinystories_5000.txt source file against the recorded split hashes.

Applies the TinyStories steps of `reproduce.py prepare-data` (parsing,
deduplication and the deterministic 80/10/10 split), using the functions in
paper_code/scripts/prepare_canonical_splits.py, and compares the three
resulting files with the tinystories_5000 entries in
docs/canonical_file_manifest.json. Text8 and WikiText-2 are not needed.

Usage:
    python scripts/verify_tinystories_5000.py --source data/sources/tinystories_5000.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "paper_code" / "scripts"))

import prepare_canonical_splits as pcs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Candidate tinystories_5000.txt")
    parser.add_argument("--seed", type=int, default=20260716, help="Split seed (default: 20260716)")
    args = parser.parse_args()

    raw = pcs.read_source(args.source)
    print(f"Read {args.source}: {len(raw)} bytes")

    documents = pcs.canonical_tinystories_documents(raw)
    print(f"Parsed {len(documents)} <|endoftext|>-delimited documents (expected 5000)")

    split, _audit = pcs.deterministic_document_split(documents, args.seed)
    reference = {
        record["relative_path"]: record
        for record in json.loads((ROOT / "docs/canonical_file_manifest.json").read_text())["files"]
    }

    all_match = True
    for split_name in ("train", "validation", "test"):
        data = pcs.join_tinystories(split[split_name])
        digest = pcs.sha256_bytes(data)
        relative_path = f"tinystories_5000/{split_name}.txt"
        ref = reference[relative_path]
        match = digest == ref["sha256"] and len(data) == ref["bytes"]
        all_match &= match
        status = "MATCH" if match else "DOES NOT MATCH"
        print(
            f"{split_name:10s} {status}: "
            f"{len(data)} bytes (paper: {ref['bytes']}), "
            f"sha256 {digest[:16]}... (paper: {ref['sha256'][:16]}...)"
        )

    print()
    if all_match:
        print("PASS: the source file reproduces the recorded TinyStories-5000 splits.")
    else:
        print("FAIL: the source file does not reproduce the recorded TinyStories-5000 splits.")
        sys.exit(1)


if __name__ == "__main__":
    main()
