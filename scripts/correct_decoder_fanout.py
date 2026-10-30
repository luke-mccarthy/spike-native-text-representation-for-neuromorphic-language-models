#!/usr/bin/env python3
"""Apply the decoder input-accumulation correction to the output-head archive.

The output-head runs counted layer-0 decoder input synaptic accumulations as
(active inputs) x (mean fan-in per destination neuron), truncated to an
integer in the dense decoder. Every input channel connects to exactly 8
destination neurons, so the correct count is (active inputs) x 8. Layer 1 of
the two-layer decoder (128 -> 128) has mean fan-in equal to fan-out and is
unaffected. The corrected counters are in
``paper_code/src/models/local_spiking_patch_decoder.py`` and
``paper_code/src/models/event_accounted_wta_ttfs_decoder.py``.

The counters are detached from training, so BPB and every other field are
unchanged. This script rewrites ``{train,validation}_decoder_input_synaptic_
accumulations`` and recomputes the energy projection with
``src.energy.wta_event_energy_projection``:

* one-layer decoders: corrected = archived x 8 / m0, where m0 is the archived
  per-input multiplier;
* two-layer decoders: corrected = archived + (8 - m0) x A0, where A0, the
  layer-0 active-input total, is rebuilt from the archived context, feedback
  and position event counts. A0 is exact for fixed patches; for adaptive
  patches the context term uses the mean patch length of the shared
  segmentation.

The original TTFS counter truncated each per-step product to an integer. Where
m0 is not a whole number (6.5 or 12.5), the corrected count can therefore be
lower than a rerun with the corrected counter by up to 0.5 per decoder step.

Inputs:  results/temporal_decoders/all_rows.csv (unchanged)
Outputs: results/temporal_decoders_corrected/all_rows.csv
         results/temporal_decoders_corrected/correction_audit.csv

Usage:
    python scripts/correct_decoder_fanout.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "paper_code"))

from src.energy.wta_event_energy_projection import wta_event_energy_projection  # noqa: E402

SOURCE = ROOT / "results/temporal_decoders/all_rows.csv"
OUTPUT_DIR = ROOT / "results/temporal_decoders_corrected"
FANOUT = 8
BACKBONE_DIM = 64
BYTE_CODE_DIM = 32
PATCH_BYTES = {
    "fixed_patch4_positional": 4,
    "fixed_patch8_hierarchical_lif": 8,
    "joint_patch8_hierarchical_lif": 8,
}
ADAPTIVE = "joint_patch8_hierarchical_lif"
VERSION = "decoder_input_fanout_correction_v1"


def value(row: dict, key: str) -> float:
    return float(row[key])


def run_key(row: dict, capacity: str | None = None) -> tuple:
    return (row["architecture"], row["dataset"], row["seed"], capacity or row["grid_decoder_capacity"])


def archived_multiplier(row: dict) -> float:
    layer0_inputs = BACKBONE_DIM + BYTE_CODE_DIM + PATCH_BYTES[row["architecture"]]
    multiplier = FANOUT * layer0_inputs / int(row["decoder_hidden_dim"])
    return int(multiplier) if row["output_head"] == "dense_reference" else multiplier


def layer0_active_inputs(row: dict, prefix: str, index: dict) -> float:
    patch_bytes = PATCH_BYTES[row["architecture"]]
    dense = row["output_head"] == "dense_reference"
    positions = value(row, f"{prefix}_decoder_position_spike_events")
    context = value(row, f"{prefix}_decoder_context_spike_events")
    if row["architecture"] != ADAPTIVE:
        feedback = (
            2 * (positions - positions / patch_bytes)
            if dense
            else value(row, f"{prefix}_decoder_teacher_byte_spike_events")
        )
        return context * patch_bytes + feedback + positions
    direct = index[(run_key(row), "ttfs_direct_256")]
    patches = (
        value(direct, f"{prefix}_decoder_position_spike_events")
        - value(direct, f"{prefix}_decoder_teacher_byte_spike_events") / 2
    )
    mean_length = positions / patches
    if not dense:
        return context * mean_length + value(row, f"{prefix}_decoder_teacher_byte_spike_events") + positions
    one_layer = index[(run_key(row, "matched_v1_capacity"), "dense_reference")]
    one_layer_inputs = value(one_layer, f"{prefix}_decoder_input_synaptic_accumulations") / archived_multiplier(one_layer)
    one_layer_context = value(one_layer, f"{prefix}_decoder_context_spike_events")
    full_patches = positions - (one_layer_inputs - one_layer_context * mean_length - positions) / 2
    return context * mean_length + 2 * (positions - full_patches) + positions


def main() -> None:
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    index = {(run_key(r), r["output_head"]): r for r in rows}
    fields += ["decoder_input_fanout_correction_version"]

    corrected_rows, audit = [], []
    for row in rows:
        out = dict(row)
        for prefix in ("train", "validation"):
            key = f"{prefix}_decoder_input_synaptic_accumulations"
            archived = value(row, key)
            multiplier = archived_multiplier(row)
            if row["decoder_num_layers"] == "1":
                corrected = archived * FANOUT / multiplier
            else:
                corrected = archived + (FANOUT - multiplier) * layer0_active_inputs(row, prefix, index)
            corrected = float(round(corrected))
            out[key] = repr(corrected)
            audit.append({
                "run_name": row["run_name"],
                "phase": prefix,
                "archived": int(archived),
                "corrected": int(corrected),
                "ratio": corrected / archived,
            })
        for name, projected in wta_event_energy_projection(dict(out)).items():
            if name in out:
                out[name] = repr(projected) if isinstance(projected, float) else str(projected)
        out["decoder_input_fanout_correction_version"] = VERSION
        corrected_rows.append(out)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "all_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(corrected_rows)
    with (OUTPUT_DIR / "correction_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(audit)
    print(f"Wrote {len(corrected_rows)} rows to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
