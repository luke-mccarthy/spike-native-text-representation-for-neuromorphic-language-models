#!/usr/bin/env python3
"""Check a candidate tinystories_5000.txt against the paper's recorded hashes.

This does the TinyStories-only half of `reproduce.py prepare-data` (parsing,
deduplication, and the deterministic 80/10/10 split) and compares the result
against the three tinystories_5000/*.txt entries in
docs/canonical_file_manifest.json -- without requiring the Text8 or
WikiText-2 source files, which the full `prepare-data` command also needs.

It imports its logic directly from paper_code/scripts/prepare_canonical_splits.py
rather than reimplementing it, so there is no risk of the two drifting apart.

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
    parser.add_argument("--seed", type=int, default=20260716, help="Matches prepare_canonical_splits.py's default")
    args = parser.parse_args()

    raw = pcs.read_source(args.source)
    print(f"Read {args.source}: {len(raw)} bytes")

    documents = pcs.canonical_tinystories_documents(raw)
    print(f"Parsed {len(documents)} <|endoftext|>-delimited documents (paper requires exactly 5000)")

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
        print("PASS: this source file reproduces the paper's exact TinyStories-5000 splits.")
    else:
        print(
            "FAIL: this source file does not reproduce the paper's TinyStories-5000 splits. "
            "This is not necessarily an error in your setup -- it means the reconstruction "
            "procedure in scripts/fetch_tinystories_5000.py does not (yet) match the original "
            "source file. Please report this so it can be corrected."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
