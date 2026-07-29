"""Fail-closed deployment-scenario energy projections for the decisive study.

This module does not claim measured hardware energy.  It combines three
auditable evidence types without silently replacing missing work by zero:

* exact operation counts for the SpikeGPT reference equations and event
  frontends;
* measured process CPU time for opaque/digital tokenisation and codec work;
* compulsory memory-byte lower bounds plus active parameter storage.

The resulting low/central/high values describe an explicit heterogeneous
deployment scenario: digital tokenisers/codecs execute on a host CPU, sparse
frontends execute as routed events, and the common reference SpikeGPT compute
uses the repository's declared 45-nm arithmetic model.  The coefficient sweep
and paired break-even analysis are the scientific result; no value is a
measurement of Loihi, TrueNorth, SpiNNaker or another named device.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Dict, Mapping


LEDGER_VERSION = "decisive_heterogeneous_projection_v1"
HOROWITZ_FP32_MAC_PJ = 4.6


@dataclass(frozen=True)
class ProjectionScenario:
    name: str
    host_active_power_watts: float
    synaptic_accumulation_pj: float
    neuron_state_update_pj: float
    comparison_reset_pj: float
    routed_spike_pj: float
    compulsory_memory_byte_pj: float

    def validate(self) -> None:
        for key, value in asdict(self).items():
            if key != "name" and float(value) <= 0:
                raise ValueError(f"{key} must be positive")


# These deliberately broad values are sensitivity assumptions.  The high
# synaptic value is close to the published 12.7-pJ/SOP ODIN measurement; the
# sweep is not labelled as an ODIN/Loihi implementation of this network.
SCENARIOS = (
    ProjectionScenario("low", 5.0, 0.1, 0.1, 0.05, 0.1, 0.1),
    ProjectionScenario("central", 15.0, 1.0, 0.9, 0.1, 1.0, 1.25),
    ProjectionScenario("high", 30.0, 12.7, 4.6, 1.0, 10.0, 20.0),
)


def _number(row: Mapping[str, object], key: str) -> float:
    value = row.get(key)
    if value in {None, "", "not_measured", "not_applicable"}:
        raise ValueError(f"required decisive-ledger field is missing: {key}")
    result = float(value)
    if result < 0:
        raise ValueError(f"{key} must be non-negative")
    return result


def _optional(row: Mapping[str, object], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value in {None, "", "not_measured", "not_applicable"}:
        return float(default)
    result = float(value)
    if result < 0:
        raise ValueError(f"{key} must be non-negative")
    return result


def _integer_flag(row: Mapping[str, object], key: str) -> bool:
    return int(float(row.get(key, 0) or 0)) == 1


def _frontend_event_counts(row: Mapping[str, object], prefix: str) -> Dict[str, float]:
    synops = _optional(row, f"{prefix}_event_driven_input_accumulations_counterfactual")
    state_updates = 0.0
    comparisons = 0.0
    routed_events = _optional(row, f"{prefix}_binary_input_spike_events")
    if _integer_flag(row, "recurrent_lif_payload_encoder"):
        for suffix in (
            "input_synaptic_accumulations_event_driven",
            "recurrent_synaptic_accumulations_event_driven",
            "hidden_lif_state_updates",
            "hidden_threshold_evaluations",
            "hidden_reset_gate_elements",
            "raw_byte_input_spike_events",
            "hidden_spike_events",
        ):
            _number(row, f"{prefix}_frontend_{suffix}")
    for stem in (f"{prefix}_frontend_", f"{prefix}_unit_frontend_"):
        synops += sum(
            _optional(row, stem + suffix)
            for suffix in (
                "input_synaptic_accumulations_event_driven",
                "recurrent_synaptic_accumulations_event_driven",
                "readout_synaptic_accumulations_event_driven",
                "boundary_synaptic_accumulations_event_driven",
            )
        )
        state_updates += sum(
            _optional(row, stem + suffix)
            for suffix in (
                "hidden_lif_state_updates",
                "output_membrane_state_updates",
                "boundary_lif_state_updates",
            )
        )
        comparisons += sum(
            _optional(row, stem + suffix)
            for suffix in (
                "hidden_threshold_evaluations",
                "output_threshold_evaluations",
                "boundary_threshold_evaluations",
                "hidden_reset_gate_elements",
                "output_reset_gate_elements",
                "hamming_comparison_elements",
            )
        )
        routed_events += sum(
            _optional(row, stem + suffix)
            for suffix in (
                "raw_byte_input_spike_events",
                "prediction_spike_events",
                "prediction_error_spike_events",
                "natural_boundary_spikes",
            )
        )
    # Surrogate-gradient frontend training performs a forward pass plus two
    # adjoint traversals.  This is an explicit conservative algorithmic
    # estimate, not a hidden omission.
    backward_multiplier = 3.0 if (
        prefix == "train" and _integer_flag(row, "interface_trainable")
        and _integer_flag(row, "recurrent_lif_payload_encoder")
    ) else 1.0
    return {
        "synaptic_accumulations": synops * backward_multiplier,
        "state_updates": state_updates * backward_multiplier,
        "comparisons_and_resets": comparisons * backward_multiplier,
        "routed_spike_events": routed_events * backward_multiplier,
        "surrogate_training_multiplier": backward_multiplier,
    }


def _digital_cpu_seconds(
    row: Mapping[str, object],
    *,
    phase: str,
    fit_benchmark_cpu_seconds: float | None,
) -> tuple[float, str]:
    if phase == "training" and fit_benchmark_cpu_seconds is not None:
        fit_seconds = (
            float(fit_benchmark_cpu_seconds)
            + _optional(row, "frontend_model_load_cpu_seconds_current_run")
            + _optional(row, "interface_build_and_homeostasis_cpu_seconds_current_run")
        )
        fit_scope = "matched_sentencepiece_retraining_fit"
    elif phase == "training":
        fit_seconds = (
            _optional(row, "frontend_fit_cpu_seconds_current_run")
            + _optional(row, "interface_build_and_homeostasis_cpu_seconds_current_run")
        )
        fit_scope = "measured_current_frontend_fit_and_interface_initialisation"
    else:
        fit_seconds = 0.0
        fit_scope = "no_fit_during_frozen_inference"
    encode_key = f"frontend_{phase}_encode_cpu_seconds_current_run"
    encode_seconds = _number(row, encode_key)
    digital_tokenizer = _integer_flag(row, "digital_tokenizer_present")
    digital_codec = _integer_flag(row, "digital_patch_codec_present")
    spike_boundary = _integer_flag(row, "spike_boundary_segmenter")
    if digital_tokenizer:
        return fit_seconds + encode_seconds, f"{fit_scope}_plus_complete_tokenizer_host_path"
    if digital_codec and not spike_boundary:
        return fit_seconds + encode_seconds, f"{fit_scope}_plus_complete_digital_proxy_host_path"
    if digital_codec and spike_boundary:
        complete_host_path = _number(row, encode_key)
        return (
            fit_seconds + complete_host_path,
            f"{fit_scope}_plus_complete_hybrid_host_path_conservative_"
            "reference_snn_events_also_reported",
        )
    return fit_seconds, fit_scope


def _memory_counts(row: Mapping[str, object], phase: str) -> Dict[str, float]:
    prefix = f"{phase}_frontend_memory_"
    frontend_compulsory = _number(row, prefix + "compulsory_bytes_lower_bound")
    interface_parameters = _optional(
        row, prefix + "interface_parameter_bytes_fp32_once"
    )
    model_state = _number(row, "model_active_state_storage_bytes_reference")
    interface_reference_state = _number(
        row, "interface_state_storage_bytes_reference"
    )
    interface_deployment_state = _number(
        row, "interface_deployment_static_state_bytes"
    )
    # Replace the older interface-parameter-only term by explicitly declared
    # deployment and reference state.  This counts non-trainable codebooks,
    # masks and recurrent thresholds instead of silently dropping buffers.
    frontend_deployment = (
        frontend_compulsory - interface_parameters + interface_deployment_state
    )
    frontend_reference = (
        frontend_compulsory - interface_parameters + interface_reference_state
    )
    total_deployment = frontend_deployment + model_state
    total_reference = frontend_reference + model_state
    if min(frontend_deployment, frontend_reference, total_deployment, total_reference) <= 0:
        raise ValueError("decisive compulsory-memory total must be positive")
    return {
        "frontend_compulsory_bytes": frontend_compulsory,
        "model_active_state_bytes_once": model_state,
        "interface_deployment_state_bytes_once": interface_deployment_state,
        "interface_reference_state_bytes_once": interface_reference_state,
        "frontend_compulsory_deployment_bytes": frontend_deployment,
        "frontend_compulsory_reference_bytes": frontend_reference,
        "total_compulsory_bytes": total_deployment,
        "total_compulsory_reference_bytes": total_reference,
    }


def phase_projection(
    row: Mapping[str, object],
    *,
    phase: str,
    fit_benchmark_cpu_seconds: float | None = None,
) -> Dict[str, float | int | str]:
    if phase not in {"training", "validation"}:
        raise ValueError("phase must be training or validation")
    prefix = "train" if phase == "training" else "eval"
    raw_bytes = _number(row, f"{phase}_analytical_raw_bytes")
    common_compute = _number(
        row, f"{phase}_analytical_common_model_and_interface_compute_energy_pj"
    )
    dense_input_projection_macs = _number(row, f"{prefix}_dense_input_projection_macs")
    common_without_dense_input = (
        common_compute - dense_input_projection_macs * HOROWITZ_FP32_MAC_PJ
    )
    if common_without_dense_input < -1e-6:
        raise ValueError("input projection exceeds common compute total")
    common_without_dense_input = max(common_without_dense_input, 0.0)
    event_counts = _frontend_event_counts(row, prefix)
    memory_counts = _memory_counts(row, phase)
    model_dense_macs = (
        _number(row, f"{prefix}_dense_linear_macs_total")
        + (_optional(row, "train_backward_dense_linear_macs_estimate") if phase == "training" else 0.0)
        - dense_input_projection_macs
    )
    model_state_and_control_elements = sum(
        _number(row, f"{phase}_analytical_{suffix}")
        for suffix in (
            "lif_and_state_update_elements",
            "threshold_reset_control_elements",
            "ordinary_nonlinear_elements",
            "special_function_elements",
            "optimizer_update_elements",
        )
    )
    if model_dense_macs < 0:
        raise ValueError("model dense MAC count became negative after input replacement")
    digital_seconds, digital_scope = _digital_cpu_seconds(
        row,
        phase=phase,
        fit_benchmark_cpu_seconds=fit_benchmark_cpu_seconds,
    )
    base = f"{phase}_decisive_"
    result: Dict[str, float | int | str] = {
        f"{base}raw_bytes": raw_bytes,
        f"{base}common_model_compute_pj_excluding_dense_input_projection": (
            common_without_dense_input
        ),
        f"{base}removed_dense_input_projection_macs": dense_input_projection_macs,
        f"{base}frontend_synaptic_accumulations": event_counts["synaptic_accumulations"],
        f"{base}frontend_neuron_state_updates": event_counts["state_updates"],
        f"{base}frontend_comparisons_and_resets": event_counts["comparisons_and_resets"],
        f"{base}frontend_routed_spike_events": event_counts["routed_spike_events"],
        f"{base}frontend_surrogate_training_multiplier": (
            event_counts["surrogate_training_multiplier"]
        ),
        f"{base}digital_host_cpu_seconds": digital_seconds,
        f"{base}digital_host_scope": digital_scope,
        f"{base}frontend_compulsory_bytes": memory_counts["frontend_compulsory_bytes"],
        f"{base}model_active_state_bytes_once": memory_counts["model_active_state_bytes_once"],
        f"{base}interface_deployment_state_bytes_once": memory_counts["interface_deployment_state_bytes_once"],
        f"{base}interface_reference_state_bytes_once": memory_counts["interface_reference_state_bytes_once"],
        f"{base}total_compulsory_bytes": memory_counts["total_compulsory_bytes"],
        f"{base}total_compulsory_reference_bytes": memory_counts["total_compulsory_reference_bytes"],
        f"{base}model_dense_macs_after_event_input_replacement": model_dense_macs,
        f"{base}model_state_and_control_elements": model_state_and_control_elements,
    }
    for scenario in SCENARIOS:
        scenario.validate()
        event_energy = (
            event_counts["synaptic_accumulations"] * scenario.synaptic_accumulation_pj
            + event_counts["state_updates"] * scenario.neuron_state_update_pj
            + event_counts["comparisons_and_resets"] * scenario.comparison_reset_pj
            + event_counts["routed_spike_events"] * scenario.routed_spike_pj
        )
        host_energy = digital_seconds * scenario.host_active_power_watts * 1.0e12
        if scenario.name == "low":
            total_memory_traffic = memory_counts["total_compulsory_bytes"]
            frontend_memory_traffic = memory_counts["frontend_compulsory_deployment_bytes"]
            memory_scope = "compulsory_once_lower_bound"
        elif scenario.name == "central":
            total_memory_traffic = (
                memory_counts["total_compulsory_bytes"]
                + 4.0 * model_dense_macs
                + 4.0 * model_state_and_control_elements
                + 4.0 * event_counts["synaptic_accumulations"]
                + 4.0 * event_counts["state_updates"]
            )
            frontend_memory_traffic = (
                memory_counts["frontend_compulsory_deployment_bytes"]
                + 4.0 * event_counts["synaptic_accumulations"]
                + 4.0 * event_counts["state_updates"]
            )
            memory_scope = "on_chip_weight_and_state_word_per_operation"
        else:
            total_memory_traffic = (
                memory_counts["total_compulsory_reference_bytes"]
                + 12.0 * model_dense_macs
                + 12.0 * model_state_and_control_elements
                + 12.0 * event_counts["synaptic_accumulations"]
                + 12.0 * event_counts["state_updates"]
                + 8.0 * event_counts["routed_spike_events"]
            )
            frontend_memory_traffic = (
                memory_counts["frontend_compulsory_reference_bytes"]
                + 12.0 * event_counts["synaptic_accumulations"]
                + 12.0 * event_counts["state_updates"]
                + 8.0 * event_counts["routed_spike_events"]
            )
            memory_scope = "no_reuse_operand_state_and_event_packet_upper_scenario"
        memory_energy = total_memory_traffic * scenario.compulsory_memory_byte_pj
        frontend_memory_energy = frontend_memory_traffic * scenario.compulsory_memory_byte_pj
        frontend_total = event_energy + host_energy + frontend_memory_energy
        end_to_end = common_without_dense_input + event_energy + host_energy + memory_energy
        suffix = scenario.name
        result.update({
            f"{base}projected_frontend_energy_pj_{suffix}": frontend_total,
            f"{base}projected_frontend_energy_pj_per_raw_byte_{suffix}": (
                frontend_total / raw_bytes
            ),
            f"{base}projected_end_to_end_energy_pj_{suffix}": end_to_end,
            f"{base}projected_end_to_end_energy_pj_per_raw_byte_{suffix}": (
                end_to_end / raw_bytes
            ),
            f"{base}event_compute_energy_pj_{suffix}": event_energy,
            f"{base}digital_host_energy_pj_{suffix}": host_energy,
            f"{base}memory_energy_pj_{suffix}": memory_energy,
            f"{base}memory_traffic_bytes_{suffix}": total_memory_traffic,
            f"{base}frontend_memory_traffic_bytes_{suffix}": frontend_memory_traffic,
            f"{base}memory_traffic_scope_{suffix}": memory_scope,
        })
    result.update({
        f"{base}category_coverage_complete_within_declared_scope": 1,
        f"{base}projection_ready": 1,
        f"{base}physical_hardware_energy_measurement": 0,
    })
    return result


def decisive_energy_projection(
    row: Mapping[str, object],
    *,
    conventional_fit_cpu_seconds: float | None = None,
) -> Dict[str, float | int | str]:
    """Return phase-separated scenario projections and completeness metadata."""

    is_conventional = _integer_flag(row, "digital_tokenizer_present")
    if is_conventional and conventional_fit_cpu_seconds is None:
        raise ValueError(
            "conventional rows require the matched SentencePiece fit benchmark"
        )
    result: Dict[str, float | int | str] = {
        "decisive_energy_ledger_version": LEDGER_VERSION,
        "decisive_energy_projection_scope": (
            "heterogeneous_host_digital_frontend_plus_event_frontend_plus_"
            "reference_spikegpt_compute_plus_compulsory_memory_sensitivity"
        ),
        "decisive_energy_projection_is_hardware_measurement": 0,
        "decisive_energy_coefficient_sensitivity_not_named_hardware_claim": 1,
        "decisive_energy_scenarios_json": json.dumps(
            [asdict(scenario) for scenario in SCENARIOS],
            sort_keys=True,
            separators=(",", ":"),
        ),
        "decisive_common_compute_coefficient_source": (
            "Horowitz_ISSCC_2014_DOI_10.1109/ISSCC.2014.6757323"
        ),
        "decisive_high_synop_reference": (
            "ODIN_28nm_12.7_pJ_per_synaptic_operation_arXiv_1804.07858"
        ),
        "decisive_low_and_central_coefficients_are_sensitivity_assumptions": 1,
        "decisive_no_missing_category_encoded_as_zero": 1,
        "decisive_full_physical_system_energy_complete": 0,
        "decisive_static_idle_cache_allocator_and_physical_routing_outside_scope": 1,
        "decisive_training_fit_benchmark_included": int(
            conventional_fit_cpu_seconds is not None
        ),
    }
    result.update(phase_projection(
        row,
        phase="training",
        fit_benchmark_cpu_seconds=conventional_fit_cpu_seconds,
    ))
    result.update(phase_projection(row, phase="validation"))
    result["decisive_scientific_energy_comparison_ready"] = 1
    result["decisive_test_energy_status"] = "sealed_until_confirmatory_gate"
    return result
