#!/usr/bin/env python3
"""Build the TinyStories-5000 source file used by the experiments.

Takes the first 5,000 examples of the ``roneneldan/TinyStories`` training
split in the dataset's own order, strips each story, and joins them with the
``<|endoftext|>`` marker read by
``paper_code/scripts/prepare_canonical_splits.py``. The output is compared
with the file used in the paper (4,262,315 bytes, SHA-256
``afff3659756a01acd8c95ed8b16c67b0ad829cd8e3d7fc6161a1bc9982c04fb4``).

Requires the ``datasets`` package (``pip install datasets``).

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
            f"examples, found {len(texts)}."
        )

    data = "\n<|endoftext|>\n".join(texts).encode("utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    print(f"Wrote {len(texts)} documents ({len(data)} bytes) to {args.output}")
    print(f"SHA-256: {digest}")

    if args.count == 5000:
        if len(data) == KNOWN_GOOD_BYTES and digest == KNOWN_GOOD_SHA256:
            print("PASS: identical to the tinystories_5000.txt used in the paper.")
        else:
            print(
                "FAIL: differs from the tinystories_5000.txt used in the paper "
                f"({KNOWN_GOOD_BYTES} bytes, SHA-256 {KNOWN_GOOD_SHA256})."
            )


if __name__ == "__main__":
    main()
