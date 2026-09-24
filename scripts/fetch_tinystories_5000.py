#!/usr/bin/env python3
"""Rebuild the raw TinyStories-5000 source file used by this study.

This is the exact procedure used to originally produce
``tinystories_5000.txt`` (preserved internally as
``scripts/prepare_2026_06_03_data.py``): take the first 5,000 examples of the
``roneneldan/TinyStories`` train split, in the dataset's own (unshuffled)
order, strip each story, keep the non-empty ones, and join them with the
``<|endoftext|>`` marker expected by
``paper_code/scripts/prepare_canonical_splits.py``.

Confirmed byte-for-byte: the author re-ran this exact procedure against the
live ``roneneldan/TinyStories`` dataset and diffed the result against the
original source file, which still exists at
``/scratch/user/lukemccarthy/datasets/tinystories/tinystories_5000.txt`` on
the Bunya HPC system and is recorded in the 2026-07-16 canonical manifest.
Both are 4,262,315 bytes with SHA-256
``afff3659756a01acd8c95ed8b16c67b0ad829cd8e3d7fc6161a1bc9982c04fb4``. This
script checks its own output against that hash and reports PASS/FAIL
immediately, before you spend any time on the canonical split step.

Requires the optional ``datasets`` package (``pip install datasets``), which
is not part of the pinned reproduction environment because it is only needed
once, to build this single input file.

Usage:
    python scripts/fetch_tinystories_5000.py --output data/sources/tinystories_5000.txt
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

KNOWN_GOOD_BYTES = 4_262_315
KNOWN_GOOD_SHA256 = "afff3659756a01acd8c95ed8b16c67b0ad829cd8e3d7fc6161a1bc9982c04fb4"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--count", type=int, default=5000,
        help="Number of leading TinyStories examples to include (default: 5000).",
    )
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "The 'datasets' package is required for this script: pip install datasets"
        ) from exc

    print("Loading roneneldan/TinyStories (train split) from Hugging Face...")
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    print(f"Total examples available: {len(dataset)}")

    texts = []
    for example in dataset.select(range(args.count)):
        text = str(example.get("text", "")).strip()
        if text:
            texts.append(text)

    if len(texts) != args.count:
        raise SystemExit(
            f"Expected {args.count} non-empty documents in the first {args.count} "
            f"examples, found {len(texts)}. This does not match what "
            "prepare_canonical_splits.py requires; do not use this output."
        )

    data = "\n<|endoftext|>\n".join(texts).encode("utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    print(f"Wrote {len(texts)} documents ({len(data)} bytes) to {args.output}")
    print(f"SHA-256: {digest}")

    if args.count == 5000:
        if len(data) == KNOWN_GOOD_BYTES and digest == KNOWN_GOOD_SHA256:
            print("PASS: byte-identical to the original tinystories_5000.txt used by the paper.")
        else:
            print(
                "FAIL: does not match the original source file "
                f"({KNOWN_GOOD_BYTES} bytes, sha256 {KNOWN_GOOD_SHA256}). "
                "The TinyStories dataset on Hugging Face may have changed since this "
                "was confirmed, or something in this environment altered the text "
                "(e.g. line-ending translation). Please report this before using the output."
            )

    print(
        "Next: run `python reproduce.py prepare-data --tinystories-5000 "
        f"{args.output} ...` (or `python scripts/verify_tinystories_5000.py "
        f"--source {args.output}` to check just this file)."
    )


if __name__ == "__main__":
    main()
