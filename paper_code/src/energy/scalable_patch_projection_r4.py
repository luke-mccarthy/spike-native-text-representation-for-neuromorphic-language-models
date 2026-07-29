"""r4 extension of the scalable patch projection for routed WTA events."""

from __future__ import annotations

from typing import Dict, Mapping

from src.energy.decisive_projection import SCENARIOS
from src.energy.scalable_patch_projection import scalable_patch_projection as base_projection


def _optional(row: Mapping[str, object], key: str) -> float:
    value = row.get(key, 0.0)
    if value in {None, "", "not_measured", "not_applicable"}:
        return 0.0
    result = float(value)
    if result < 0.0:
        raise ValueError(f"negative optional r4 ledger field: {key}")
    return result


def scalable_patch_projection_r4(
    row: Mapping[str, object],
) -> Dict[str, float | int | str]:
    """Add emitted class and inhibitory-latch packet energy to the base ledger."""

    result = dict(base_projection(row))
    for phase, prefix in (("training", "train"), ("validation", "validation")):
        multiplier = 3.0 if phase == "training" else 1.0
        emitted = _optional(row, f"{prefix}_decoder_output_emitted_class_spikes")
        latch = _optional(row, f"{prefix}_decoder_output_inhibitory_latch_spikes")
        extra_routed_events = multiplier * (emitted + latch)
        base = f"{phase}_patch_"
        result[f"{base}wta_output_routed_events"] = extra_routed_events
        for scenario in SCENARIOS:
            extra_event_energy = extra_routed_events * scenario.routed_spike_pj
            extra_memory_traffic = (
                8.0 * extra_routed_events if scenario.name == "high" else 0.0
            )
            extra_memory_energy = (
                extra_memory_traffic * scenario.compulsory_memory_byte_pj
            )
            result[f"{base}event_compute_energy_pj_{scenario.name}"] = (
                float(result[f"{base}event_compute_energy_pj_{scenario.name}"])
                + extra_event_energy
            )
            result[f"{base}memory_traffic_bytes_{scenario.name}"] = (
                float(result[f"{base}memory_traffic_bytes_{scenario.name}"])
                + extra_memory_traffic
            )
            result[f"{base}memory_energy_pj_{scenario.name}"] = (
                float(result[f"{base}memory_energy_pj_{scenario.name}"])
                + extra_memory_energy
            )
            extra_total = extra_event_energy + extra_memory_energy
            result[f"{base}projected_end_to_end_energy_pj_{scenario.name}"] = (
                float(result[f"{base}projected_end_to_end_energy_pj_{scenario.name}"])
                + extra_total
            )
            raw_bytes = float(
                row["training_raw_bytes"]
                if phase == "training"
                else row["validation_raw_bytes"]
            )
            result[
                f"{base}projected_end_to_end_energy_pj_per_raw_byte_{scenario.name}"
            ] = (
                float(result[f"{base}projected_end_to_end_energy_pj_{scenario.name}"])
                / raw_bytes
            )
    result["wta_output_routed_event_energy_included"] = 1
    result["wta_output_candidate_crossings_not_counted_as_routed_packets"] = 1
    return result


__all__ = ["scalable_patch_projection_r4"]
