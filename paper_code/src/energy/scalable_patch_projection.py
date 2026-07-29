"""Explicit operation and energy projections for codec-free patch language models.

The projection is a sensitivity analysis, not a hardware measurement.  Dense
SpikeGPT arithmetic retains the 45-nm FP32 MAC coefficient used by the
decisive comparison.  Binary local-interface activity is priced with the same
low/central/high event and memory scenarios.  The reference PyTorch MAC count
is reported separately and is never added to the event-driven counterfactual.
"""

from __future__ import annotations

from typing import Dict, Mapping

from src.energy.decisive_projection import HOROWITZ_FP32_MAC_PJ, SCENARIOS


PATCH_PROJECTION_VERSION = "scalable_patch_projection_v1"


def _value(row: Mapping[str, object], key: str) -> float:
    value = row.get(key)
    if value in {None, "", "not_measured", "not_applicable"}:
        raise ValueError(f"missing patch-ledger field: {key}")
    result = float(value)
    if result < 0:
        raise ValueError(f"negative patch-ledger field: {key}")
    return result


def project_patch_phase(
    row: Mapping[str, object],
    *,
    phase: str,
) -> Dict[str, float | int | str]:
    """Project one phase from exact local counts and declared global equations."""

    if phase not in {"training", "validation"}:
        raise ValueError("phase must be training or validation")
    prefix = "train" if phase == "training" else "validation"
    raw_bytes = _value(
        row,
        "training_raw_bytes" if phase == "training" else "validation_raw_bytes",
    )
    patch_steps = _value(row, f"{prefix}_global_patch_steps")
    hidden_dim = int(_value(row, "hidden_dim"))
    layers = int(_value(row, "num_layers"))
    input_dim = int(_value(row, "global_input_dimension"))

    # Per SpikeGPT block: four HxH time-mix projections and
    # Hx4H + HxH + 4HxH channel-mix projections.
    global_core_macs = patch_steps * layers * 13.0 * hidden_dim * hidden_dim
    global_lif_state = patch_steps * layers * 2.0 * hidden_dim
    # ln0 in the first block plus ln1/ln2 in each block.  The ordinary final
    # LayerNorm/readout path is bypassed because the local byte decoder
    # consumes the final binary channel spikes.
    global_layer_norm = patch_steps * hidden_dim * (2.0 * layers + 1.0)
    global_relu_square = patch_steps * layers * 4.0 * hidden_dim
    global_wkv_maximum = patch_steps * layers * 2.0 * hidden_dim
    global_special_functions = (
        patch_steps * layers * 2.0 * hidden_dim  # sigmoid
        + patch_steps * layers * 4.0 * hidden_dim  # WKV exponentials
        + patch_steps * layers * hidden_dim  # WKV division
    )
    global_input_synops = (
        _value(row, f"{prefix}_global_input_spike_events") * hidden_dim
        if input_dim != hidden_dim
        else 0.0
    )
    segmenter_synops = _value(row, f"{prefix}_segmenter_synaptic_accumulations")
    model_local_synops = sum(
        _value(row, f"{prefix}_{suffix}")
        for suffix in (
            "encoder_input_synaptic_accumulations",
            "encoder_recurrent_synaptic_accumulations",
            "decoder_input_synaptic_accumulations",
            "decoder_recurrent_synaptic_accumulations",
            "decoder_readout_synaptic_accumulations",
        )
    )
    segmenter_state = _value(row, f"{prefix}_segmenter_state_updates")
    model_local_state = sum(
        _value(row, f"{prefix}_{suffix}")
        for suffix in (
            "encoder_lif_state_updates",
            "decoder_lif_state_updates",
        )
    )
    segmenter_control = _value(
        row, f"{prefix}_segmenter_threshold_reset_elements"
    )
    model_local_control = sum(
        _value(row, f"{prefix}_{suffix}")
        for suffix in (
            "encoder_threshold_reset_elements",
            "decoder_threshold_reset_elements",
        )
    )
    decoder_likelihood_ordinary = sum(
        _value(row, f"{prefix}_{suffix}")
        for suffix in (
            "decoder_likelihood_reduction_add_elements",
            "decoder_likelihood_target_subtraction_elements",
        )
    )
    decoder_likelihood_special = sum(
        _value(row, f"{prefix}_{suffix}")
        for suffix in (
            "decoder_likelihood_exponential_elements",
            "decoder_likelihood_log_elements",
        )
    )
    decoder_event_readout_synops = _value(
        row, f"{prefix}_decoder_readout_synaptic_accumulations"
    )
    decoder_dense_reference_readout_macs = _value(
        row, f"{prefix}_decoder_dense_reference_readout_macs"
    )
    adapter_control = patch_steps * hidden_dim if input_dim != hidden_dim else 0.0
    segmenter_routed = _value(row, f"{prefix}_segmenter_spike_events")
    model_routed_events = sum(
        _value(row, f"{prefix}_{suffix}")
        for suffix in (
            "encoder_raw_byte_events",
            "encoder_patch_output_events",
            "encoder_hidden_events",
            "decoder_context_spike_events",
            "decoder_teacher_byte_spike_events",
            "decoder_position_spike_events",
            "decoder_hidden_spike_events",
        )
    )
    model_routed_events += _value(
        row, f"{prefix}_backbone_input_adapter_spikes"
    )
    model_routed_events += _value(
        row, f"{prefix}_backbone_total_hidden_spikes"
    )
    # Training is an explicit forward-plus-two-adjoint approximation, matching
    # the conservative multiplier used by the decisive event ledger.
    multiplier = 3.0 if phase == "training" else 1.0
    global_core_macs *= multiplier
    global_lif_state *= multiplier
    global_layer_norm *= multiplier
    global_relu_square *= multiplier
    global_wkv_maximum *= multiplier
    global_special_functions *= multiplier
    global_input_synops *= multiplier
    decoder_likelihood_ordinary *= multiplier
    decoder_likelihood_special *= multiplier
    decoder_event_readout_synops *= multiplier
    decoder_dense_reference_readout_macs *= multiplier
    local_synops = segmenter_synops + multiplier * model_local_synops
    local_state = segmenter_state + multiplier * model_local_state
    local_control = (
        segmenter_control
        + multiplier * (
            model_local_control
            + adapter_control
            + 2.0 * patch_steps * layers * hidden_dim
        )
    )
    routed_events = segmenter_routed + multiplier * model_routed_events
    parameter_bytes = _value(row, "model_parameter_bytes_fp32")
    buffer_bytes = _value(row, "model_buffer_bytes")
    optimizer_updates = (
        _value(row, "model_trainable_parameters")
        * _value(row, "training_optimizer_steps")
        if phase == "training"
        else 0.0
    )
    active_state_bytes = (
        patch_steps * hidden_dim * layers * 2.0 * 4.0
        + raw_bytes * hidden_dim * 4.0
    )
    base = f"{phase}_patch_"
    result: Dict[str, float | int | str] = {
        f"{base}global_dense_macs": global_core_macs,
        f"{base}global_lif_state_updates": global_lif_state,
        f"{base}global_ordinary_nonlinear_elements": (
            global_layer_norm + global_relu_square + global_wkv_maximum
        ),
        f"{base}global_special_function_elements": global_special_functions,
        f"{base}decoder_likelihood_ordinary_elements": (
            decoder_likelihood_ordinary
        ),
        f"{base}decoder_likelihood_special_elements": decoder_likelihood_special,
        f"{base}decoder_event_readout_synops": decoder_event_readout_synops,
        f"{base}decoder_dense_reference_readout_macs": (
            decoder_dense_reference_readout_macs
        ),
        f"{base}global_input_event_synops": global_input_synops,
        f"{base}local_event_synops": local_synops,
        f"{base}segmenter_event_synops_not_adjoint_multiplied": segmenter_synops,
        f"{base}local_state_updates": local_state,
        f"{base}local_threshold_reset_elements": local_control,
        f"{base}routed_spike_events": routed_events,
        f"{base}training_multiplier": multiplier,
        f"{base}optimizer_parameter_update_elements": optimizer_updates,
        f"{base}physical_hardware_energy_measurement": 0,
    }
    for scenario in SCENARIOS:
        event_energy = (
            (global_input_synops + local_synops)
            * scenario.synaptic_accumulation_pj
            + (local_state + global_lif_state + optimizer_updates)
            * scenario.neuron_state_update_pj
            + local_control * scenario.comparison_reset_pj
            + routed_events * scenario.routed_spike_pj
            + (
                global_layer_norm
                + global_relu_square
                + global_wkv_maximum
                + decoder_likelihood_ordinary
            )
            * scenario.neuron_state_update_pj
            + (
                global_special_functions + decoder_likelihood_special
            )
            * HOROWITZ_FP32_MAC_PJ
        )
        dense_energy = global_core_macs * HOROWITZ_FP32_MAC_PJ
        if scenario.name == "low":
            traffic = parameter_bytes + buffer_bytes + active_state_bytes
            memory_scope = "compulsory_once_lower_bound"
        elif scenario.name == "central":
            traffic = (
                parameter_bytes
                + buffer_bytes
                + active_state_bytes
                + 4.0 * global_core_macs
                + 4.0 * (
                    global_input_synops + local_synops + local_state
                    + global_lif_state + global_layer_norm
                    + global_relu_square + global_wkv_maximum
                    + global_special_functions + decoder_likelihood_ordinary
                    + decoder_likelihood_special + optimizer_updates
                )
            )
            memory_scope = "on_chip_weight_and_state_word_per_operation"
        else:
            traffic = (
                parameter_bytes
                + buffer_bytes
                + active_state_bytes
                + 12.0 * global_core_macs
                + 12.0 * (
                    global_input_synops + local_synops + local_state
                    + global_lif_state + global_layer_norm
                    + global_relu_square + global_wkv_maximum
                    + global_special_functions + decoder_likelihood_ordinary
                    + decoder_likelihood_special + optimizer_updates
                )
                + 8.0 * routed_events
            )
            memory_scope = "no_reuse_operand_state_and_event_packet_upper_scenario"
        memory_energy = traffic * scenario.compulsory_memory_byte_pj
        total = dense_energy + event_energy + memory_energy
        event_decoder_compute = (
            decoder_event_readout_synops
            * scenario.synaptic_accumulation_pj
        )
        dense_decoder_compute = (
            decoder_dense_reference_readout_macs * HOROWITZ_FP32_MAC_PJ
        )
        operation_traffic_factor = {
            "low": 0.0,
            "central": 4.0,
            "high": 12.0,
        }[scenario.name]
        dense_decoder_traffic = (
            traffic
            + operation_traffic_factor
            * (
                decoder_dense_reference_readout_macs
                - decoder_event_readout_synops
            )
        )
        dense_decoder_memory = (
            dense_decoder_traffic * scenario.compulsory_memory_byte_pj
        )
        dense_decoder_total = (
            total
            - event_decoder_compute
            - memory_energy
            + dense_decoder_compute
            + dense_decoder_memory
        )
        result.update({
            f"{base}projected_end_to_end_energy_pj_{scenario.name}": total,
            f"{base}projected_end_to_end_energy_pj_per_raw_byte_{scenario.name}": (
                total / raw_bytes
            ),
            f"{base}event_compute_energy_pj_{scenario.name}": event_energy,
            f"{base}dense_global_compute_energy_pj_{scenario.name}": dense_energy,
            f"{base}memory_energy_pj_{scenario.name}": memory_energy,
            f"{base}memory_traffic_bytes_{scenario.name}": traffic,
            f"{base}memory_scope_{scenario.name}": memory_scope,
            f"{base}projected_end_to_end_energy_pj_dense_decoder_reference_{scenario.name}": (
                dense_decoder_total
            ),
            f"{base}projected_end_to_end_energy_pj_per_raw_byte_dense_decoder_reference_{scenario.name}": (
                dense_decoder_total / raw_bytes
            ),
            f"{base}dense_decoder_reference_memory_traffic_bytes_{scenario.name}": (
                dense_decoder_traffic
            ),
        })
    result[f"{base}category_coverage_complete_within_declared_scope"] = 1
    return result


def scalable_patch_projection(
    row: Mapping[str, object],
) -> Dict[str, float | int | str]:
    result: Dict[str, float | int | str] = {
        "scalable_patch_energy_ledger_version": PATCH_PROJECTION_VERSION,
        "energy_projection_is_hardware_measurement": 0,
        "digital_tokenizer_host_energy_present": 0,
        "digital_patch_codec_energy_present": 0,
        "reference_cpu_runtime_not_used_as_neuromorphic_energy": 1,
        "event_and_dense_reference_counts_not_double_counted": 1,
        "likelihood_normalization_operations_included": 1,
        "decoder_readout_mapping_sensitivity_emitted": 1,
    }
    result.update(project_patch_phase(row, phase="training"))
    result.update(project_patch_phase(row, phase="validation"))
    return result
