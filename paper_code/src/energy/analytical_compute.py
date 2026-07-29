"""Phase-separated analytical compute-energy estimates.

The estimates in this module are deliberately not hardware measurements. They
convert an auditable operation ledger into a declared 45 nm compute-only
scenario commonly used for architecture-level comparisons:

* FP32 multiply-accumulate: 4.6 pJ (3.7 pJ multiply + 0.9 pJ add);
* FP32 accumulation/state update: 0.9 pJ;
* comparison/control element: 0.1 pJ;
* exponential, division or sigmoid element: 4.6 pJ.

The arithmetic coefficients are anchored to M. Horowitz, "Computing's energy
problem (and what we can do about it)", ISSCC 2014,
doi:10.1109/ISSCC.2014.6757323. Treating a special function as one FP32 MAC is
an explicit modelling convention, not a value measured by that source.

Memory hierarchy, routing, static power, host overhead, digital codecs and
opaque tokenizer internals are explicitly excluded; their phase timing and
resource measurements remain separate. Consequently the returned scalar is a
compute-energy estimate and never a hardware/system-energy claim.

The module also reports a second, deliberately separate empirical host-CPU
frontend proxy.  It converts measured process CPU time for one corpus encoding
to CPU ns per raw byte and to a declared single-core active-power sensitivity
scenario.  That proxy is applied to every frontend, including SentencePiece,
so conventional tokenization is not represented as free.  It describes the
current CPU implementations, not neuromorphic execution, and is never added to
the analytical event-compute scalar.

For comparison, the ledger also separates common language-model/interface
compute from explicitly counted frontend compute.  The conservative counted
total assigns opaque conventional-tokenizer arithmetic zero cost.  That value
is a lower-bound comparison deliberately biased against a claimed spiking
advantage; it must never be interpreted as conventional tokenization consuming
zero energy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Mapping


SCENARIO_ID = "horowitz_45nm_fp32_compute_only_v1"
SOURCE_DOI = "10.1109/ISSCC.2014.6757323"


@dataclass(frozen=True)
class ComputeEnergyScenario:
    fp32_mac_pj: float = 4.6
    fp32_accumulate_pj: float = 0.9
    comparison_pj: float = 0.1
    special_function_pj: float = 4.6

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")


REFERENCE_SCENARIO = ComputeEnergyScenario()
HOST_CPU_REFERENCE_POWER_WATTS = 15.0
HOST_CPU_POWER_SENSITIVITY_WATTS = (5.0, 15.0, 30.0)


def _number(row: Mapping[str, object], key: str) -> float:
    value = row.get(key, 0.0)
    if value in ("", None, "not_applicable"):
        return 0.0
    result = float(value)
    if result < 0:
        raise ValueError(f"{key} must be non-negative")
    return result


def _optional_number(row: Mapping[str, object], key: str) -> float | None:
    value = row.get(key, None)
    if value in ("", None, "not_applicable"):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result < 0:
        raise ValueError(f"{key} must be non-negative")
    return result


def empirical_frontend_cpu_work(
    row: Mapping[str, object],
    *,
    training_raw_bytes: int,
    validation_raw_bytes: int,
) -> Dict[str, float | int | str]:
    """Normalize measured one-time frontend encoding CPU work.

    ``time.process_time`` is collected around the same software path used to
    construct each modelling-unit sequence.  A fixed active-power coefficient
    makes the corresponding energy values a sensitivity scenario rather than a
    measurement.  CPU ns/byte is the primary implementation-neutral value.
    """

    if int(training_raw_bytes) <= 0 or int(validation_raw_bytes) <= 0:
        raise ValueError("raw byte denominators must be positive")
    result: Dict[str, float | int | str] = {
        "empirical_host_frontend_proxy_scope": (
            "one_time_current_cpu_implementation_corpus_encoding_"
            "separate_from_analytical_event_compute"
        ),
        "empirical_host_frontend_cpu_time_source": "python_process_time",
        "empirical_host_frontend_reference_power_watts": (
            HOST_CPU_REFERENCE_POWER_WATTS
        ),
        "empirical_host_frontend_power_sensitivity_watts": ";".join(
            f"{value:g}" for value in HOST_CPU_POWER_SENSITIVITY_WATTS
        ),
        "empirical_host_frontend_energy_proxy_is_hardware_measurement": 0,
        "empirical_host_frontend_proxy_combined_with_analytical_scalar": 0,
    }
    phase_fields = (
        (
            "training",
            "frontend_training_encode_cpu_seconds_current_run",
            int(training_raw_bytes),
        ),
        (
            "validation",
            "frontend_validation_encode_cpu_seconds_current_run",
            int(validation_raw_bytes),
        ),
    )
    for phase, field, raw_bytes in phase_fields:
        seconds = _optional_number(row, field)
        prefix = f"{phase}_empirical_host_frontend_"
        if seconds is None:
            result[f"{prefix}encoding_proxy_ready"] = 0
            result[f"{prefix}encode_cpu_seconds"] = "not_measured"
            result[f"{prefix}encode_cpu_ns_per_raw_byte"] = "not_measured"
            for watts in HOST_CPU_POWER_SENSITIVITY_WATTS:
                result[
                    f"{prefix}encode_energy_proxy_pj_per_raw_byte_at_{watts:g}w"
                ] = "not_measured"
            continue
        result[f"{prefix}encoding_proxy_ready"] = 1
        result[f"{prefix}encode_cpu_seconds"] = seconds
        result[f"{prefix}encode_cpu_ns_per_raw_byte"] = (
            seconds * 1.0e9 / raw_bytes
        )
        for watts in HOST_CPU_POWER_SENSITIVITY_WATTS:
            result[
                f"{prefix}encode_energy_proxy_pj_per_raw_byte_at_{watts:g}w"
            ] = seconds * watts * 1.0e12 / raw_bytes

    fit_seconds = _optional_number(
        row, "frontend_fit_cpu_seconds_current_run"
    )
    result["training_empirical_host_frontend_fit_cpu_time_ready"] = int(
        fit_seconds is not None
    )
    result["training_empirical_host_frontend_fit_cpu_seconds"] = (
        fit_seconds if fit_seconds is not None else "not_measured_precomputed_fit"
    )
    result["training_empirical_host_frontend_fit_not_added_to_encoding_proxy"] = 1
    return result


def phase_compute_energy(
    row: Mapping[str, object],
    *,
    prefix: str,
    raw_bytes: int,
    training: bool,
    scenario: ComputeEnergyScenario = REFERENCE_SCENARIO,
) -> Dict[str, float | int | str]:
    """Convert one phase operation ledger into auditable compute-only energy."""

    scenario.validate()
    if prefix not in {"train", "eval"}:
        raise ValueError("prefix must be train or eval")
    if int(raw_bytes) <= 0:
        raise ValueError("raw_bytes must be positive")

    dense_backbone_macs = _number(row, f"{prefix}_dense_linear_macs_total")
    backward_macs = (
        _number(row, "train_backward_dense_linear_macs_estimate")
        if training else 0.0
    )
    frontend_dense_macs = (
        _number(row, f"{prefix}_frontend_dense_reference_recurrent_macs_actual")
        + _number(row, f"{prefix}_frontend_dense_reference_readout_macs_actual")
        + _number(row, f"{prefix}_frontend_dense_reference_boundary_macs_actual")
        + _number(row, f"{prefix}_unit_frontend_dense_reference_recurrent_macs_actual")
        + _number(row, f"{prefix}_unit_frontend_dense_reference_readout_macs_actual")
        + _number(row, f"{prefix}_unit_frontend_dense_reference_boundary_macs_actual")
    )
    frontend_event_accumulations = (
        _number(row, f"{prefix}_frontend_input_synaptic_accumulations_event_driven")
        + _number(row, f"{prefix}_frontend_recurrent_synaptic_accumulations_event_driven")
        + _number(row, f"{prefix}_frontend_readout_synaptic_accumulations_event_driven")
        + _number(row, f"{prefix}_unit_frontend_input_synaptic_accumulations_event_driven")
        + _number(row, f"{prefix}_unit_frontend_recurrent_synaptic_accumulations_event_driven")
        + _number(row, f"{prefix}_unit_frontend_readout_synaptic_accumulations_event_driven")
    )
    model_lif_state_updates = _number(
        row, f"{prefix}_lif_neuron_state_updates"
    )
    frontend_lif_state_updates = (
        + _number(row, f"{prefix}_frontend_hidden_lif_state_updates")
        + _number(row, f"{prefix}_frontend_output_membrane_state_updates")
        + _number(row, f"{prefix}_unit_frontend_hidden_lif_state_updates")
        + _number(row, f"{prefix}_unit_frontend_output_membrane_state_updates")
        + _number(row, f"{prefix}_unit_frontend_boundary_lif_state_updates")
    )
    lif_state_updates = model_lif_state_updates + frontend_lif_state_updates
    model_threshold_and_reset_elements = (
        _number(row, f"{prefix}_lif_threshold_elements")
        + _number(row, f"{prefix}_lif_reset_elements")
        + _number(row, f"{prefix}_input_spike_generation_threshold_elements")
    )
    frontend_threshold_and_reset_elements = (
        + _number(row, f"{prefix}_frontend_hidden_threshold_evaluations")
        + _number(row, f"{prefix}_frontend_output_threshold_evaluations")
        + _number(row, f"{prefix}_frontend_hidden_reset_gate_elements")
        + _number(row, f"{prefix}_unit_frontend_hidden_threshold_evaluations")
        + _number(row, f"{prefix}_unit_frontend_output_threshold_evaluations")
        + _number(row, f"{prefix}_unit_frontend_boundary_threshold_evaluations")
        + _number(row, f"{prefix}_unit_frontend_hidden_reset_gate_elements")
        + _number(row, f"{prefix}_unit_frontend_output_reset_gate_elements")
        + _number(row, f"{prefix}_unit_frontend_hamming_comparison_elements")
    )
    threshold_and_reset_elements = (
        model_threshold_and_reset_elements
        + frontend_threshold_and_reset_elements
    )
    likelihood_multiplier = 3.0 if training else 1.0
    likelihood_ordinary_elements = likelihood_multiplier * (
        _number(row, f"{prefix}_likelihood_reduction_add_elements")
        + _number(row, f"{prefix}_likelihood_target_subtraction_elements")
    )
    likelihood_special_elements = likelihood_multiplier * (
        _number(row, f"{prefix}_likelihood_exponential_elements")
        + _number(row, f"{prefix}_likelihood_log_elements")
    )
    ordinary_nonlinear_elements = (
        _number(row, f"{prefix}_layer_norm_elements")
        + _number(row, f"{prefix}_channel_relu_square_elements")
        + _number(row, f"{prefix}_wkv_maximum_elements")
        + likelihood_ordinary_elements
    )
    special_function_elements = (
        _number(row, f"{prefix}_sigmoid_elements")
        + _number(row, f"{prefix}_wkv_exponential_elements")
        + _number(row, f"{prefix}_wkv_division_elements")
        + likelihood_special_elements
    )
    optimizer_elements = (
        _number(row, "train_optimizer_parameter_update_elements")
        if training else 0.0
    )
    unit_frontend_counted_elements = sum(
        _number(row, f"{prefix}_unit_frontend_{suffix}")
        for suffix in (
            "dense_reference_recurrent_macs_actual",
            "dense_reference_readout_macs_actual",
            "dense_reference_boundary_macs_actual",
            "input_synaptic_accumulations_event_driven",
            "recurrent_synaptic_accumulations_event_driven",
            "readout_synaptic_accumulations_event_driven",
            "hidden_lif_state_updates",
            "output_membrane_state_updates",
            "boundary_lif_state_updates",
            "hidden_threshold_evaluations",
            "output_threshold_evaluations",
            "boundary_threshold_evaluations",
            "hidden_reset_gate_elements",
            "output_reset_gate_elements",
            "hamming_comparison_elements",
        )
    )

    dense_reference_macs = dense_backbone_macs + backward_macs + frontend_dense_macs
    common_model_energy = (
        (model_lif_state_updates + optimizer_elements)
        * scenario.fp32_accumulate_pj
        + model_threshold_and_reset_elements * scenario.comparison_pj
        + ordinary_nonlinear_elements * scenario.fp32_accumulate_pj
        + special_function_elements * scenario.special_function_pj
        + (dense_backbone_macs + backward_macs) * scenario.fp32_mac_pj
    )
    explicit_frontend_common_energy = (
        frontend_lif_state_updates * scenario.fp32_accumulate_pj
        + frontend_threshold_and_reset_elements * scenario.comparison_pj
    )
    explicit_frontend_dense_energy = (
        frontend_dense_macs * scenario.fp32_mac_pj
        + explicit_frontend_common_energy
    )
    explicit_frontend_event_energy = (
        frontend_event_accumulations * scenario.fp32_accumulate_pj
        + explicit_frontend_common_energy
    )
    dense_reference_energy = common_model_energy + explicit_frontend_dense_energy
    event_frontend_energy = common_model_energy + explicit_frontend_event_energy
    phase = "training" if training else "validation"
    base = f"{phase}_analytical_"
    return {
        f"{base}energy_scenario": SCENARIO_ID,
        f"{base}energy_source_doi": SOURCE_DOI,
        f"{base}raw_bytes": int(raw_bytes),
        f"{base}dense_reference_macs": dense_reference_macs,
        f"{base}event_frontend_accumulations": frontend_event_accumulations,
        f"{base}lif_and_state_update_elements": lif_state_updates,
        f"{base}threshold_reset_control_elements": threshold_and_reset_elements,
        f"{base}ordinary_nonlinear_elements": ordinary_nonlinear_elements,
        f"{base}special_function_elements": special_function_elements,
        f"{base}likelihood_ordinary_elements": likelihood_ordinary_elements,
        f"{base}likelihood_special_elements": likelihood_special_elements,
        f"{base}likelihood_training_multiplier": likelihood_multiplier,
        f"{base}likelihood_normalization_operations_included": 1,
        f"{base}optimizer_update_elements": optimizer_elements,
        f"{base}common_model_and_interface_compute_energy_pj": common_model_energy,
        f"{base}common_model_and_interface_compute_energy_pj_per_raw_byte": common_model_energy / raw_bytes,
        f"{base}explicit_counted_frontend_dense_compute_energy_pj": explicit_frontend_dense_energy,
        f"{base}explicit_counted_frontend_dense_compute_energy_pj_per_raw_byte": explicit_frontend_dense_energy / raw_bytes,
        f"{base}explicit_counted_frontend_event_compute_energy_pj": explicit_frontend_event_energy,
        f"{base}explicit_counted_frontend_event_compute_energy_pj_per_raw_byte": explicit_frontend_event_energy / raw_bytes,
        f"{base}conservative_counted_total_compute_energy_pj": event_frontend_energy,
        f"{base}conservative_counted_total_compute_energy_pj_per_raw_byte": event_frontend_energy / raw_bytes,
        f"{base}conventional_opaque_frontend_compute_assumed_zero_in_conservative_total": int(
            str(row.get("method", ""))
            in {"bpe8000", "bpe16000", "unigram4000", "spikegpt_20b_bpe"}
        ),
        f"{base}digital_codec_compute_excluded_from_conservative_total": int(
            str(row.get("method", ""))
            not in {"bpe8000", "bpe16000", "unigram4000", "spikegpt_20b_bpe", "byte"}
        ),
        f"{base}dense_reference_compute_energy_pj": dense_reference_energy,
        f"{base}dense_reference_compute_energy_pj_per_raw_byte": dense_reference_energy / raw_bytes,
        f"{base}event_frontend_compute_energy_pj": event_frontend_energy,
        f"{base}event_frontend_compute_energy_pj_per_raw_byte": event_frontend_energy / raw_bytes,
        f"{base}compute_energy_estimate_ready": 1,
        f"{base}compute_operation_ledger_complete": 0,
        f"{base}unit_construction_counted_elements": (
            unit_frontend_counted_elements
        ),
        f"{base}unit_construction_operations_partially_included": int(
            unit_frontend_counted_elements > 0
        ),
        f"{base}opaque_conventional_tokenizer_operations_included": 0,
        f"{base}digital_codec_and_control_operations_included": 0,
        f"{base}frontend_fit_or_pretraining_operations_included": 0,
        f"{base}system_energy_estimate_ready": 0,
        f"{base}memory_energy_included": 0,
        f"{base}routing_static_and_host_energy_included": 0,
        f"{base}hardware_measured_energy": 0,
        f"{base}hardware_energy_claim_permitted": 0,
    }


def experiment_compute_energy(
    row: Mapping[str, object],
    *,
    training_raw_bytes: int,
    validation_raw_bytes: int,
) -> Dict[str, float | int | str]:
    """Return training and validation compute estimates under one scenario."""

    result: Dict[str, float | int | str] = {}
    result.update(phase_compute_energy(
        row, prefix="train", raw_bytes=training_raw_bytes, training=True
    ))
    result.update(phase_compute_energy(
        row, prefix="eval", raw_bytes=validation_raw_bytes, training=False
    ))
    result.update(empirical_frontend_cpu_work(
        row,
        training_raw_bytes=training_raw_bytes,
        validation_raw_bytes=validation_raw_bytes,
    ))
    result.update({
        "energy_estimate_scope": (
            "analytical_compute_only_phase_separated_memory_routing_static_"
            "host_codec_and_opaque_tokenizer_work_excluded_with_phase_"
            "timing_and_resources_reported_separately"
        ),
        "energy_missing_values_are_never_encoded_as_zero": 1,
        "likelihood_normalization_operations_included": 1,
        "test_analytical_energy_status": "not_evaluated_test_split_sealed",
    })
    return result
