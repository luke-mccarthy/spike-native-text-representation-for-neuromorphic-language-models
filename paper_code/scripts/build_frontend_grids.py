#!/usr/bin/env python3
"""Build the frozen, research-question-centred frontend comparison grids."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


DATASETS = ("tinystories_5000", "wikitext2")
SEEDS = (1, 2, 3)
FIT_METHODS = ("bpe8000", "bpe16000", "unigram4000")


@dataclass(frozen=True)
class Configuration:
    name: str
    method: str
    interface_encoder: str
    frontend_family: str
    scientific_role: str
    fully_spiking_frontend: int
    fixed_spike_transducer: int = 0
    recurrent_lif_payload_encoder: int = 0
    spike_boundary_segmenter: int = 0
    digital_patch_codec_present: int = 0
    digital_tokenizer_present: int = 0


# These are frozen before any new canonical-validation or test result is read.
# Existing weak methods remain available in immutable archives; the decisive
# grid carries only controls needed to answer compatibility, quality, event
# efficiency and end-to-end workload questions.
CONFIGURATIONS = (
    Configuration(
        "bpe8k", "bpe8000", "spikegpt_embedding64_atan",
        "conventional_subword", "corpus_matched_conventional_baseline", 0,
        digital_tokenizer_present=1,
    ),
    Configuration(
        "bpe16k", "bpe16000", "spikegpt_embedding64_atan",
        "conventional_subword", "corpus_matched_conventional_baseline", 0,
        digital_tokenizer_present=1,
    ),
    Configuration(
        "unigram4k", "unigram4000", "spikegpt_embedding64_atan",
        "conventional_subword", "corpus_matched_conventional_baseline", 0,
        digital_tokenizer_present=1,
    ),
    Configuration(
        "homeostatic_boundary_hybrid", "legacy_homeostatic_0.075",
        "spikegpt_embedding64_atan", "spike_boundary_hybrid",
        "best_existing_spike_boundary_with_digital_lossless_codec", 0,
        spike_boundary_segmenter=1, digital_patch_codec_present=1,
    ),
    Configuration(
        "byte_learned_control", "byte", "spikegpt_embedding64_atan",
        "codec_free_learned_byte_control",
        "isolates_byte_unit_rate_from_sparse_fixed_or_recurrent_encoding", 0,
    ),
    Configuration(
        "byte_fixed_4of32", "byte", "fixed_4of32_a",
        "codec_free_fixed_spike_transducer",
        "existing_injective_four_event_byte_anchor", 1,
        fixed_spike_transducer=1,
    ),
    Configuration(
        "byte_fixed_2of32", "byte", "fixed_2of32_byte",
        "codec_free_fixed_spike_transducer",
        "new_injective_half_activity_same_width_byte_code", 1,
        fixed_spike_transducer=1,
    ),
    Configuration(
        "byte_fixed_1of256", "byte", "fixed_1of256_byte",
        "codec_free_fixed_spike_transducer",
        "new_minimum_event_wider_bus_byte_code", 1,
        fixed_spike_transducer=1,
    ),
    Configuration(
        "byte_recurrent_lif_injective", "byte",
        "streaming_reservoir64_injective2",
        "codec_free_injective_recurrent_lif_snn",
        "streaming_recurrent_lif_context_plus_collision_free_raw_byte_spike_suffix", 1,
        recurrent_lif_payload_encoder=1,
    ),
)


def _rows(*, smoke: bool) -> list[dict[str, object]]:
    seeds = (1,) if smoke else SEEDS
    configurations = (
        tuple(
            configuration for configuration in CONFIGURATIONS
            if configuration.name in {
                "bpe8k",
                "homeostatic_boundary_hybrid",
                "byte_fixed_2of32",
                "byte_fixed_1of256",
                "byte_recurrent_lif_injective",
            }
        )
        if smoke else CONFIGURATIONS
    )
    rows: list[dict[str, object]] = []
    run_id = 0
    for dataset in DATASETS:
        for configuration in configurations:
            for seed in seeds:
                run_id += 1
                row = asdict(configuration)
                row.update({
                    "run_id": run_id,
                    "run_name": (
                        f"2026_07_22_decisive_frontend_{'smoke' if smoke else 'development'}_"
                        f"r{run_id:03d}_{dataset}_{configuration.name}_s{seed}"
                    ),
                    "dataset": dataset,
                    "data_tier": "development" if smoke else "full_train_validation",
                    "seed": seed,
                    "patch_codec_vocab_size": 8192,
                    "patch_codec_fallback": (
                        "optimal_training_subpatch"
                        if configuration.digital_patch_codec_present else "literal_bytes"
                    ),
                    "train_byte_budget": 100_000 if smoke else 1_000_000,
                    "training_schedule": "common_utf8_prefix_one_pass",
                    "eval_byte_budget": 100_000 if smoke else 0,
                    "batch_size": 2,
                    "seq_len": 32,
                    "hidden_dim": 64,
                    "num_layers": 2,
                    "backbone": "spikegpt_reference",
                    "spike_steps": 1,
                    "lr": 0.001,
                    "interface_temperature": 1.0,
                    "interface_target_rate": 0.125,
                    "interface_rate_weight": 0.1,
                    "open_vocabulary_lossless_byte_coverage": 1,
                    "test_split_touched": 0,
                    "canonical_validation_is_development_evidence": 1,
                    "confirmatory_test_remains_sealed": 1,
                    "configuration_frozen_before_current_validation": 1,
                    "frontend_fit_exact_lm_training_selection_required": 1,
                    "system_or_hardware_energy_claim_permitted": 0,
                    "launch_authorized": 1,
                })
                rows.append(row)
    return rows


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fit_rows(*, data_tier: str, byte_budget: int) -> list[dict[str, object]]:
    if data_tier not in {"development", "full_train_validation"}:
        raise ValueError(f"unsupported fit data tier: {data_tier}")
    if byte_budget <= 0:
        raise ValueError("fit byte budget must be positive")
    return [
        {
            "task_id": index,
            "data_tier": data_tier,
            "dataset": dataset,
            "method": method,
            "training_byte_budget": byte_budget,
        }
        for index, (dataset, method) in enumerate(
            (
                (dataset, method)
                for dataset in DATASETS
                for method in FIT_METHODS
            ),
            start=1,
        )
    ]


def read_fit_grid_row(path: Path, task_id: int) -> dict[str, str]:
    """Select one fit row by its declared ID, independent of CSV column order."""
    if task_id <= 0:
        raise ValueError("fit task ID must be positive")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "task_id", "data_tier", "dataset", "method", "training_byte_budget"
    }
    if not rows or any(set(row) != required for row in rows):
        raise AssertionError("fit grid header does not match the frozen schema")
    matches = [row for row in rows if int(row["task_id"]) == task_id]
    if len(matches) != 1:
        raise AssertionError(
            f"fit grid contains {len(matches)} rows for task ID {task_id}"
        )
    if {int(row["task_id"]) for row in rows} != set(range(1, len(rows) + 1)):
        raise AssertionError("fit grid task IDs are not contiguous and complete")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    smoke = _rows(smoke=True)
    development = _rows(smoke=False)
    smoke_fits = fit_rows(data_tier="development", byte_budget=100_000)
    development_fits = fit_rows(
        data_tier="full_train_validation", byte_budget=1_000_000
    )
    if len(smoke) != 10 or len(development) != 54:
        raise AssertionError("decisive grids are not complete paired factorials")
    _write(args.root / "smoke_grid.csv", smoke)
    _write(args.root / "development_grid.csv", development)
    _write(args.root / "smoke_fit_grid.csv", smoke_fits)
    _write(args.root / "development_fit_grid.csv", development_fits)
    identity = [asdict(configuration) for configuration in CONFIGURATIONS]
    manifest = {
        "format": "decisive_frontend_grid_manifest_v1",
        "status": "PASS",
        "smoke_rows": len(smoke),
        "development_rows": len(development),
        "matched_sentencepiece_smoke_fit_rows": len(smoke_fits),
        "matched_sentencepiece_development_fit_rows": len(development_fits),
        "datasets": list(DATASETS),
        "seeds": list(SEEDS),
        "configurations": [configuration.name for configuration in CONFIGURATIONS],
        "configuration_identity_sha256": hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "frozen_before_current_validation": 1,
        "test_split_touched": 0,
        "confirmatory_test_grid_not_generated": 1,
    }
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "grid_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"PASS decisive grids smoke={len(smoke)} development={len(development)}")


if __name__ == "__main__":
    main()
