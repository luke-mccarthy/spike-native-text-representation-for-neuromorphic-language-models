#!/usr/bin/env python3
"""Validate the release or execute one row from a frozen paper grid."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
FRONT_END_COMMIT = "95f148159b9b340c2f679c0d10eedf1e35b936f9"
PAPER_CODE_ROOT = ROOT / "paper_code"


def read_grid_row(path: Path, one_based_row: int) -> tuple[dict[str, str], int]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not 1 <= one_based_row <= len(rows):
        raise SystemExit(f"--row must be between 1 and {len(rows)}")
    return rows[one_based_row - 1], len(rows)


def front_end_command(args: argparse.Namespace) -> list[str]:
    grid = ROOT / "configs/front_end_test.csv"
    _, total = read_grid_row(grid, args.row)
    print(f"Selected front-end row {args.row}/{total}", file=sys.stderr)
    return [
        sys.executable,
        str(
            PAPER_CODE_ROOT
            / "scripts/run_2026_07_23_scalable_patch_frontend_row.py"
        ),
        "--grid",
        str(grid),
        "--task-id",
        str(args.row),
        "--parts-root",
        str(args.output_root),
        "--fit-root",
        str(args.tokenizer_root),
        "--fit-model-root",
        str(args.tokenizer_root / "models"),
        "--expected-commit",
        FRONT_END_COMMIT,
    ]


def temporal_decoder_command(args: argparse.Namespace) -> list[str]:
    grid_path = ROOT / "configs/temporal_decoder_test.csv"
    row, total = read_grid_row(grid_path, args.row)
    print(f"Selected temporal-decoder row {args.row}/{total}", file=sys.stderr)
    output = args.output_root / f"{row['run_name']}.csv"
    return [
        sys.executable,
        str(
            PAPER_CODE_ROOT
            / "scripts/train_2026_07_28_strict_ttfs_output_lm_r4.py"
        ),
        "--dataset", row["dataset"],
        "--data-tier", row["data_tier"],
        "--architecture", row["architecture"],
        "--output-head", row["output_head"],
        "--seed", row["seed"],
        "--train-byte-budget", row["train_byte_budget"],
        "--patches-per-block", row["patches_per_block"],
        "--target-boundary-rate-per-byte", row["target_boundary_rate_per_byte"],
        "--batch-size", row["batch_size"],
        "--hidden-dim", row["hidden_dim"],
        "--num-layers", row["num_layers"],
        "--decoder-hidden-dim", row["decoder_hidden_dim"],
        "--decoder-num-layers", row["decoder_num_layers"],
        "--output-microsteps", row["output_microsteps"],
        "--output-beta", row["output_beta"],
        "--output-threshold", row["output_threshold"],
        "--output-weight-scale", row["output_weight_scale"],
        "--output-clock-end", row["output_clock_end"],
        "--output-latency-tau", row["output_latency_tau"],
        "--output-score-scale", row["output_score_scale"],
        "--output-surrogate-temperature", row["output_surrogate_temperature"],
        "--output-margin-weight", row["output_margin_weight"],
        "--lr", row["lr"],
        "--test-evaluation-authorized", row["test_evaluation_authorized"],
        "--output-csv", str(output),
    ]


def execute(
    command: list[str],
    *,
    data_root: Path,
    dry_run: bool,
    working_directory: Path,
) -> None:
    print(shlex.join(command))
    if dry_run:
        return
    environment = os.environ.copy()
    environment["SPIKE_TOKENIZER_CANONICAL_ROOT"] = str(data_root.resolve())
    subprocess.run(command, cwd=working_directory, env=environment, check=True)


def add_common_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--row", type=int, required=True, help="One-indexed grid row")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/canonical")
    parser.add_argument("--output-root", type=Path, required=False)
    parser.add_argument("--dry-run", action="store_true")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="check archive integrity")

    prepare = subparsers.add_parser(
        "prepare-data", help="construct and verify canonical dataset splits"
    )
    prepare.add_argument("--text8-full", type=Path, required=True)
    prepare.add_argument("--tinystories-5000", type=Path, required=True)
    prepare.add_argument("--wikitext2-train", type=Path, required=True)
    prepare.add_argument("--wikitext2-validation", type=Path, required=True)
    prepare.add_argument("--wikitext2-test", type=Path, required=True)
    prepare.add_argument("--output-root", type=Path, default=ROOT / "data/canonical")
    prepare.add_argument("--verify-only", action="store_true")

    fit = subparsers.add_parser(
        "fit-tokenizer", help="fit one frozen SentencePiece grid row"
    )
    fit.add_argument("--row", type=int, required=True)
    fit.add_argument("--data-root", type=Path, default=ROOT / "data/canonical")
    fit.add_argument(
        "--tokenizer-root", type=Path, default=ROOT / "tokenizers/fitted"
    )
    fit.add_argument("--dry-run", action="store_true")

    front_end = subparsers.add_parser("front-end", help="run one front-end grid row")
    add_common_run_arguments(front_end)
    front_end.add_argument(
        "--tokenizer-root", type=Path, default=ROOT / "tokenizers/fitted"
    )

    decoder = subparsers.add_parser(
        "temporal-decoder", help="run one temporal-decoder grid row"
    )
    add_common_run_arguments(decoder)

    args = parser.parse_args()
    if args.command == "validate":
        subprocess.run(
            [sys.executable, "scripts/validate_release.py"], cwd=ROOT, check=True
        )
        return
    if args.command == "prepare-data":
        command = [
            sys.executable,
            str(
                PAPER_CODE_ROOT
                / "scripts/prepare_2026_07_16_canonical_splits.py"
            ),
            "--text8-full", str(args.text8_full),
            "--tinystories-5000", str(args.tinystories_5000),
            "--wikitext2-train", str(args.wikitext2_train),
            "--wikitext2-validation", str(args.wikitext2_validation),
            "--wikitext2-test", str(args.wikitext2_test),
            "--output-root", str(args.output_root),
        ]
        if args.verify_only:
            command.append("--verify-only")
        execute(
            command,
            data_root=args.output_root,
            dry_run=False,
            working_directory=PAPER_CODE_ROOT,
        )
        return
    if args.command == "fit-tokenizer":
        grid = ROOT / "configs/tokenizer_fits.csv"
        _, total = read_grid_row(grid, args.row)
        print(f"Selected tokenizer-fit row {args.row}/{total}", file=sys.stderr)
        command = [
            sys.executable,
            str(
                PAPER_CODE_ROOT
                / "scripts/measure_2026_07_22_sentencepiece_fit.py"
            ),
            "--grid", str(grid),
            "--task-id", str(args.row),
            "--fit-root", str(args.tokenizer_root),
            "--expected-commit", FRONT_END_COMMIT,
        ]
        execute(
            command,
            data_root=args.data_root,
            dry_run=args.dry_run,
            working_directory=PAPER_CODE_ROOT,
        )
        return

    default_output = (
        ROOT / "outputs/front_end"
        if args.command == "front-end"
        else ROOT / "outputs/temporal_decoders"
    )
    args.output_root = args.output_root or default_output
    args.output_root.mkdir(parents=True, exist_ok=True) if not args.dry_run else None
    command = (
        front_end_command(args)
        if args.command == "front-end"
        else temporal_decoder_command(args)
    )
    execute(
        command,
        data_root=args.data_root,
        dry_run=args.dry_run,
        working_directory=(
            PAPER_CODE_ROOT
            if args.command == "front-end"
            else PAPER_CODE_ROOT
        ),
    )


if __name__ == "__main__":
    main()
