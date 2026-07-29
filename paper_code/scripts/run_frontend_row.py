#!/usr/bin/env python3
"""Run and enrich one commit-pinned decisive frontend comparison row."""

from __future__ import annotations

import argparse
import csv
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.energy.frontend_energy_projection import frontend_energy_projection


BOUND_FIELDS = (
    "run_id", "run_name", "name", "frontend_family", "scientific_role",
    "fully_spiking_frontend", "fixed_spike_transducer",
    "recurrent_lif_payload_encoder", "spike_boundary_segmenter",
    "digital_patch_codec_present", "digital_tokenizer_present",
    "open_vocabulary_lossless_byte_coverage",
    "canonical_validation_is_development_evidence",
    "confirmatory_test_remains_sealed",
    "configuration_frozen_before_current_validation",
    "configuration_frozen_before_confirmatory_test",
    "frontend_fit_exact_lm_training_selection_required",
    "system_or_hardware_energy_claim_permitted", "launch_authorized",
)


def read_one(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise AssertionError(f"expected one row in {path}, found {len(rows)}")
    return rows[0]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", required=True, type=Path)
    parser.add_argument("--task-id", required=True, type=int)
    parser.add_argument("--parts-root", required=True, type=Path)
    parser.add_argument("--fit-root", required=True, type=Path)
    parser.add_argument("--fit-model-root", type=Path)
    args = parser.parse_args()
    with args.grid.open(newline="", encoding="utf-8") as handle:
        grid = list(csv.DictReader(handle))
    index = args.task_id - 1
    if index < 0 or index >= len(grid):
        raise SystemExit(f"task-id {args.task_id} outside grid with {len(grid)} rows")
    row = grid[index]
    if row.get("launch_authorized") != "1":
        raise AssertionError("grid row is not launch-authorized")
    evaluation_source = row.get("evaluation_source", "canonical_validation")
    expected_test_touched = "1" if evaluation_source == "canonical_test" else "0"
    if row.get("test_split_touched") != expected_test_touched:
        raise AssertionError("grid row violates its declared evaluation-split contract")
    output = args.parts_root / f"{row['run_name']}.csv"
    fit = None
    fit_seconds = None
    conventional_model = None
    if row["digital_tokenizer_present"] == "1":
        fit_path = args.fit_root / f"{row['dataset']}_{row['method']}.csv"
        fit = read_one(fit_path)
        if fit.get("status") != "PASS" or fit.get("test_split_touched") != "0":
            raise AssertionError("matched conventional fit benchmark is not valid")
        if fit.get("data_tier") != row["data_tier"]:
            raise AssertionError("matched tokenizer fit used a different data tier")
        if fit.get("training_selection_raw_bytes") != row["train_byte_budget"]:
            raise AssertionError("matched tokenizer fit used a different byte budget")
        conventional_model = (
            args.fit_model_root / f"{row['dataset']}_{row['method']}.model"
            if args.fit_model_root is not None
            else Path(fit["fitted_model_path"])
        )
        if not conventional_model.is_file():
            raise FileNotFoundError("matched tokenizer model is missing")
        if sha256_file(conventional_model) != fit.get("fitted_model_sha256"):
            raise AssertionError("matched tokenizer model hash differs from fit record")
        if not conventional_model.with_suffix(".train.json").is_file():
            raise FileNotFoundError("matched tokenizer provenance sidecar is missing")
        fit_seconds = float(fit["fit_cpu_seconds"])
    command = [
        sys.executable,
        "scripts/train_frontend_language_model.py",
        "--dataset", row["dataset"],
        "--data-tier", row["data_tier"],
        "--method", row["method"],
        "--patch-codec-vocab-size", row["patch_codec_vocab_size"],
        "--patch-codec-fallback", row["patch_codec_fallback"],
        "--interface-encoder", row["interface_encoder"],
        "--interface-temperature", row["interface_temperature"],
        "--interface-target-rate", row["interface_target_rate"],
        "--interface-rate-weight", row["interface_rate_weight"],
        "--seed", row["seed"],
        "--train-byte-budget", row["train_byte_budget"],
        "--training-schedule", row["training_schedule"],
        "--fit-frontends-on-lm-training-selection",
        "--evaluation-source", evaluation_source,
        "--eval-byte-budget", row["eval_byte_budget"],
        "--batch-size", row["batch_size"],
        "--seq-len", row["seq_len"],
        "--hidden-dim", row["hidden_dim"],
        "--num-layers", row["num_layers"],
        "--backbone", row["backbone"],
        "--spike-steps", row["spike_steps"],
        "--lr", row["lr"],
        "--output-csv", str(output),
    ]
    if conventional_model is not None:
        command.extend([
            "--conventional-tokenizer-model", str(conventional_model),
        ])
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Running decisive row {args.task_id}/{len(grid)}: {row['run_name']}", flush=True)
    subprocess.run(command, check=True)
    result = read_one(output)
    for key in BOUND_FIELDS:
        if key in row:
            result[key] = row[key]
    result["grid_dataset"] = row["dataset"]
    result["grid_method"] = row["method"]
    result["grid_interface_encoder"] = row["interface_encoder"]
    result["grid_seed"] = row["seed"]
    if result.get("test_split_touched") != expected_test_touched:
        raise AssertionError("produced row touched the wrong evaluation split")
    complete_evaluation = int(
        row["data_tier"] == "full_train_validation" and row["eval_byte_budget"] == "0"
    )
    result["evaluation_is_complete_canonical_validation"] = int(
        complete_evaluation and evaluation_source == "canonical_validation"
    )
    result["evaluation_is_complete_confirmatory_test"] = int(
        complete_evaluation and evaluation_source == "canonical_test"
    )
    if fit is not None:
        if fit.get("training_source_sha256") != result.get("training_sha256"):
            raise AssertionError("fit benchmark and LM row use different training source")
        if fit.get("training_sha256") != result.get("lm_training_selection_sha256"):
            raise AssertionError("fit benchmark and LM row use different byte selection")
        if fit.get("fitted_model_sha256") != result.get(
            "training_sequence_tokenizer_model_sha256"
        ):
            raise AssertionError("LM row did not use the matched fitted tokenizer model")
        result["matched_conventional_fit_benchmark_path"] = str(fit_path)
        result["matched_conventional_fit_settings_sha256"] = fit["fit_settings_sha256"]
        result["matched_conventional_fit_training_selection_sha256"] = fit[
            "training_sha256"
        ]
        result["matched_conventional_fit_training_selection_raw_bytes"] = fit[
            "training_selection_raw_bytes"
        ]
        result["matched_conventional_fit_model_sha256"] = fit[
            "fitted_model_sha256"
        ]
        result["matched_conventional_fit_model_used_by_row"] = 1
    else:
        result["matched_conventional_fit_benchmark_path"] = "not_applicable"
        result["matched_conventional_fit_settings_sha256"] = "not_applicable"
        result["matched_conventional_fit_training_selection_sha256"] = "not_applicable"
        result["matched_conventional_fit_training_selection_raw_bytes"] = "not_applicable"
        result["matched_conventional_fit_model_sha256"] = "not_applicable"
        result["matched_conventional_fit_model_used_by_row"] = "not_applicable"
    result.update(frontend_energy_projection(
        result,
        conventional_fit_cpu_seconds=fit_seconds,
    ))
    result["decisive_evaluation_energy_status"] = (
        "confirmatory_test_evaluated"
        if evaluation_source == "canonical_test"
        else "canonical_validation_evaluated"
    )
    temporary = output.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(result))
        writer.writeheader()
        writer.writerow(result)
    temporary.replace(output)


if __name__ == "__main__":
    main()
