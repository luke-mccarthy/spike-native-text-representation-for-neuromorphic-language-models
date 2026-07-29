#!/usr/bin/env python3
"""Run one mixed conventional/byte/patch development-grid row."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def read_one(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise AssertionError(f"expected one result row in {path}")
    return rows[0]


def atomic_row(path: Path, row: dict[str, object]) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted(row))
            writer.writeheader()
            writer.writerow(row)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", required=True, type=Path)
    parser.add_argument("--task-id", required=True, type=int)
    parser.add_argument("--parts-root", required=True, type=Path)
    parser.add_argument("--fit-root", required=True, type=Path)
    parser.add_argument("--fit-model-root", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    with args.grid.open(newline="", encoding="utf-8") as handle:
        grid = list(csv.DictReader(handle))
    if not 1 <= args.task_id <= len(grid):
        raise SystemExit("task ID lies outside grid")
    row = grid[args.task_id - 1]
    evaluation_source = row.get("evaluation_source", "canonical_validation")
    expected_test_touched = (
        "1" if evaluation_source == "canonical_test" else "0"
    )
    if (
        row["launch_authorized"] != "1"
        or row["test_split_touched"] != expected_test_touched
    ):
        raise AssertionError("row violates its authorised evaluation contract")
    output = args.parts_root / f"{row['run_name']}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    if row["runner_kind"] == "decisive":
        command = [
            sys.executable,
            "scripts/run_frontend_row.py",
            "--grid", str(args.grid),
            "--task-id", str(args.task_id),
            "--parts-root", str(args.parts_root),
            "--fit-root", str(args.fit_root / "parts"),
            "--fit-model-root", str(args.fit_model_root),
        ]
    elif row["runner_kind"] == "patch":
        command = [
            sys.executable,
            "scripts/train_patch_language_model.py",
            "--dataset", row["dataset"],
            "--data-tier", row["data_tier"],
            "--architecture", row["patch_architecture"],
            "--seed", row["seed"],
            "--evaluation-source", evaluation_source,
            "--train-byte-budget", row["train_byte_budget"],
            "--patches-per-block", row["patches_per_block"],
            "--batch-size", row["batch_size"],
            "--hidden-dim", row["hidden_dim"],
            "--num-layers", row["num_layers"],
            "--lr", row["lr"],
            "--output-csv", str(output),
        ]
    else:
        raise AssertionError(f"unknown runner kind: {row['runner_kind']}")
    print(f"Running scalable patch row {args.task_id}/{len(grid)}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    result: dict[str, object] = read_one(output)
    for key, value in row.items():
        result[f"grid_{key}"] = value
    result["run_id"] = row["run_id"]
    result["run_name"] = row["run_name"]
    result["name"] = row["name"]
    result["dataset"] = row["dataset"]
    result["seed"] = row["seed"]
    result["frontend_family"] = row["frontend_family"]
    result["architectural_stratum"] = row["architectural_stratum"]
    result["scientific_role"] = row["scientific_role"]
    result["evidence_class"] = row["evidence_class"]
    result["fully_spiking_frontend"] = row["fully_spiking_frontend"]
    result["runner_kind"] = row["runner_kind"]
    result["expected_commit"] = args.expected_commit
    result["evaluation_source"] = evaluation_source
    if result.get("test_split_touched") != expected_test_touched:
        raise AssertionError("produced row touched the wrong evaluation split")
    if row["runner_kind"] == "patch":
        result["bits_per_raw_byte"] = result["bpb"]
        result["eval_raw_bytes"] = result["evaluation_raw_bytes"]
        result["validation_bytes_per_unit"] = (
            1.0 / float(result["validation_units_per_raw_byte"])
        )
        result["interface_dimension"] = result["global_input_dimension"]
        result["vocab_size"] = 256
        for phase in ("training", "validation"):
            for scenario in ("low", "central", "high"):
                source = (
                    f"{phase}_patch_projected_end_to_end_energy_pj_"
                    f"per_raw_byte_{scenario}"
                )
                # Keep the decisive spelling for a single paired table.
                result[
                    f"{phase}_decisive_projected_end_to_end_energy_pj_"
                    f"per_raw_byte_{scenario}"
                ] = result[source]
        result["validation_decisive_projected_frontend_energy_pj_per_raw_byte_central"] = (
            float(result["validation_patch_projected_end_to_end_energy_pj_per_raw_byte_central"])
            - float(result["validation_patch_dense_global_compute_energy_pj_central"])
            / float(result["validation_raw_bytes"])
        )
        result["decisive_energy_projection_ready"] = 1
        result["evaluation_is_complete_canonical_validation"] = int(
            row["data_tier"] == "full_train_validation"
            and row["eval_byte_budget"] == "0"
            and evaluation_source == "canonical_validation"
        )
        result["evaluation_is_complete_confirmatory_test"] = int(
            row["data_tier"] == "full_train_validation"
            and row["eval_byte_budget"] == "0"
            and evaluation_source == "canonical_test"
        )
    atomic_row(output, result)


if __name__ == "__main__":
    main()
