#!/usr/bin/env python3
"""Run the continuous-time WTA TTFS trainer using the audited base workflow."""

from __future__ import annotations

from typing import Dict

import scripts.train_ttfs_decoder as base
from src.energy.wta_event_energy_projection import wta_event_energy_projection
from src.models.spike_native_patch_lm import PositionalPatchSpikeEncoder
from src.models.event_accounted_wta_ttfs_decoder import (
    OUTPUT_HEADS,
    PATCH_ARCHITECTURES,
    EventAccountedWTATTFSPatchLanguageModel,
)

_original_metric_bundle = base.metric_bundle
_original_atomic_row = base.atomic_row


def _encoder_output_dimension(architecture: str) -> int:
    spec = PATCH_ARCHITECTURES[architecture]
    encoder = PositionalPatchSpikeEncoder(
        max_patch_bytes=int(spec["max_patch_bytes"]),
        hierarchical=bool(spec["hierarchical"]),
        seed=0,
    )
    return int(encoder.output_dim)


def _corrected_projection(row: Dict[str, object]) -> Dict[str, float | int | str]:
    expected = _encoder_output_dimension(str(row["architecture"]))
    existing = row.get("global_input_dimension")
    if existing not in {None, ""} and int(existing) != expected:
        raise AssertionError(f"global input dimension mismatch: {existing} != {expected}")
    row["global_input_dimension"] = expected
    row["global_input_dimension_source"] = "model_architecture_exact"
    return wta_event_energy_projection(row)


def _metric_bundle(
    prefix: str,
    nll: float,
    raw_bytes: int,
    stats: Dict[str, float],
    metrics: Dict[str, float],
) -> Dict[str, object]:
    output = _original_metric_bundle(prefix, nll, raw_bytes, stats, metrics)
    active = float(stats.get("decoder_output_active_bytes", 0.0))
    denominator = max(active, 1.0)
    emitted = float(stats.get("decoder_output_emitted_class_spikes", 0.0))
    latch = float(stats.get("decoder_output_inhibitory_latch_spikes", 0.0))
    candidates = float(stats.get("decoder_output_candidate_crossings", 0.0))
    suppressed = float(stats.get("decoder_output_suppressed_crossings", 0.0))
    valid = float(stats.get("decoder_output_valid_unique_winner_bytes", 0.0))
    attempted = float(stats.get("decoder_feedback_attempted_bytes", 0.0))
    fallbacks = float(stats.get("decoder_feedback_fallback_bytes", 0.0))
    output.update({
        f"{prefix}_output_emitted_class_spikes_per_byte": emitted / denominator,
        f"{prefix}_output_inhibitory_latch_spikes_per_byte": latch / denominator,
        f"{prefix}_output_class_plus_latch_spikes_per_byte": (emitted + latch) / denominator,
        f"{prefix}_output_candidate_crossings_per_byte": candidates / denominator,
        f"{prefix}_output_suppressed_crossings_per_byte": suppressed / denominator,
        f"{prefix}_output_valid_unique_winner_fraction": valid / denominator,
        f"{prefix}_feedback_fallback_fraction": fallbacks / max(attempted, 1.0),
    })
    return output


def _atomic_row(path, row: Dict[str, object]) -> None:
    strict = int(str(row["output_head"]) != "dense_reference")
    row["format"] = "strict_ttfs_output_decoder_row_v4"
    row["method"] = "byte_patch_autoregressive_continuous_ttfs_unique_wta_v9"
    row["output_dynamics"] = "continuous_submicrostep_race_with_inhibitory_latch"
    row["output_deadline_drive_mode"] = "late_quadratic_after_0p65_horizon"
    row["output_clock_end_semantics"] = "deadline_drive_end"
    row["output_wta_latch_is_spike_event"] = strict
    row["output_only_unique_winner_class_spike_is_routed"] = strict
    row["output_only_winner_class_spike_is_routed"] = strict
    row["output_candidate_times_are_local_not_routed"] = strict
    row["free_running_invalid_winner_feedback_is_silence"] = strict
    row["submicrostep_interpolation_is_local_neuron_state_arithmetic"] = strict
    _original_atomic_row(path, row)


base.OUTPUT_HEADS = OUTPUT_HEADS
base.PATCH_ARCHITECTURES = PATCH_ARCHITECTURES
base.StrictTTFSPatchLanguageModel = EventAccountedWTATTFSPatchLanguageModel
base.scalable_patch_projection = _corrected_projection
base.metric_bundle = _metric_bundle
base.atomic_row = _atomic_row
base.FORMAT = "strict_ttfs_output_decoder_row_v4"


if __name__ == "__main__":
    base.main()
