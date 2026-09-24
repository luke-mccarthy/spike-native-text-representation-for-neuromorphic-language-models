#!/usr/bin/env python3
"""Reconstruct the raw TinyStories-5000 source file.

This mirrors the internal development script that originally produced
``tinystories_5000.txt``: it takes the first 5,000 examples of the
``roneneldan/TinyStories`` train split, in the dataset's own (unshuffled)
order, and joins them with the ``<|endoftext|>`` marker expected by
``paper_code/scripts/prepare_canonical_splits.py``.

Provenance note: this script has been reconstructed from the private
development history rather than copied from a committed, run-logged
invocation, because no job file recorded the exact command used to build the
original source file. Its output format and story count have been checked
against the assertions in ``prepare_canonical_splits.py`` (exactly 5,000
non-empty ``<|endoftext|>``-delimited documents), but the byte-for-byte match
against the original ``tinystories_5000.txt`` has not been independently
confirmed by the authors as of this writing. After running this script, run
``python reproduce.py prepare-data`` and compare the resulting
``data/canonical/manifest.json`` against ``docs/canonical_file_manifest.json``
(same tool: ``python scripts/validate_release.py`` / ``pytest``). If the
hashes disagree, please open an issue rather than assuming your environment
is at fault -- it would mean this reconstruction needs correcting.

Requires the optional ``datasets`` package (``pip install datasets``), which
is not part of the pinned reproduction environment because it is only needed
once, to build this single input file.

Usage:
    python scripts/fetch_tinystories_5000.py --output data/sources/tinystories_5000.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path


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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n<|endoftext|>\n".join(texts), encoding="utf-8")
    print(f"Wrote {len(texts)} documents ({args.output.stat().st_size} bytes) to {args.output}")
    print(
        "Next: run `python reproduce.py prepare-data --tinystories-5000 "
        f"{args.output} ...` and diff the resulting manifest against "
        "docs/canonical_file_manifest.json."
    )


if __name__ == "__main__":
    main()
