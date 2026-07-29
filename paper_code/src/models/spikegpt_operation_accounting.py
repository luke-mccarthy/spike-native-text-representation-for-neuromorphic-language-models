"""Auditable algorithmic ledger for the reference-faithful SpikeGPT core."""

from __future__ import annotations

from typing import Dict


LEDGER_VERSION = "spikegpt_reference_algorithmic_ledger_v1"


def spikegpt_reference_operation_ledger(
    *,
    modelling_units: int,
    input_dimension: int,
    hidden_dimension: int,
    num_layers: int,
    vocabulary_size: int,
    input_spikes: float,
    input_adapter_spikes: float = 0.0,
    interface_parameter_count: int = 0,
    input_spike_generation_threshold_elements: int | None = None,
) -> Dict[str, int | float | str]:
    """Count dense linear MACs, LIF updates, nonlinear work, and state size.

    These are algorithmic counts for the PyTorch reference equations.  They
    are not a hardware-energy estimate.  In particular, layer-normalized
    residual streams remain dense, so hidden Linear operations are not
    re-labelled as sparse event accumulations.
    """
    values = {
        "modelling_units": modelling_units,
        "input_dimension": input_dimension,
        "hidden_dimension": hidden_dimension,
        "num_layers": num_layers,
        "vocabulary_size": vocabulary_size,
    }
    for name, value in values.items():
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive")
    if (
        float(input_spikes) < 0
        or float(input_adapter_spikes) < 0
        or int(interface_parameter_count) < 0
        or (
            input_spike_generation_threshold_elements is not None
            and int(input_spike_generation_threshold_elements) < 0
        )
    ):
        raise ValueError("spikes and parameter counts must be non-negative")

    units = int(modelling_units)
    input_dim = int(input_dimension)
    hidden = int(hidden_dimension)
    layers = int(num_layers)
    vocab = int(vocabulary_size)
    input_projection = 0 if input_dim == hidden else units * input_dim * hidden
    time_mix = units * layers * 4 * hidden * hidden
    channel_mix = units * layers * 9 * hidden * hidden
    readout = units * hidden * vocab
    total_dense = input_projection + time_mix + channel_mix + readout

    lif_updates = units * layers * 2 * hidden
    layer_norm_elements = units * hidden * (2 * layers + 2)
    channel_relu_square_elements = units * layers * 4 * hidden
    sigmoid_elements = units * layers * 2 * hidden
    wkv_exp_elements = units * layers * 4 * hidden
    wkv_maximum_elements = units * layers * 2 * hidden
    wkv_division_elements = units * layers * hidden
    # Exact negative-log-likelihood evaluation requires log-sum-exp over the
    # complete output vocabulary. Count it explicitly so a large conventional
    # vocabulary is not treated as having free probability normalisation.
    likelihood_exponential_elements = units * vocab
    likelihood_reduction_add_elements = units * (vocab - 1)
    likelihood_log_elements = units
    likelihood_target_subtraction_elements = units
    input_threshold_elements = (
        units * hidden
        if input_spike_generation_threshold_elements is None
        else int(input_spike_generation_threshold_elements)
    )
    recurrent_state_elements = layers * 7 * hidden
    interface_reads = units * hidden if input_dim == hidden else 0

    return {
        "operation_ledger_version": LEDGER_VERSION,
        "dense_input_projection_macs": input_projection,
        "dense_time_mix_projection_macs": time_mix,
        "dense_channel_mix_projection_macs": channel_mix,
        "dense_readout_macs": readout,
        "dense_linear_macs_total": total_dense,
        "lif_neuron_state_updates": lif_updates,
        "layer_norm_elements": layer_norm_elements,
        "channel_relu_square_elements": channel_relu_square_elements,
        "sigmoid_elements": sigmoid_elements,
        "wkv_exponential_elements": wkv_exp_elements,
        "wkv_maximum_elements": wkv_maximum_elements,
        "wkv_division_elements": wkv_division_elements,
        "likelihood_exponential_elements": likelihood_exponential_elements,
        "likelihood_reduction_add_elements": likelihood_reduction_add_elements,
        "likelihood_log_elements": likelihood_log_elements,
        "likelihood_target_subtraction_elements": (
            likelihood_target_subtraction_elements
        ),
        "likelihood_normalization_operations_included": 1,
        "lif_threshold_elements": lif_updates,
        "lif_reset_elements": lif_updates,
        "input_spike_generation_threshold_elements": input_threshold_elements,
        "binary_input_spike_events": float(input_spikes),
        "binary_input_adapter_spike_events": float(input_adapter_spikes),
        "interface_embedding_elements_read": interface_reads,
        "interface_parameter_elements": int(interface_parameter_count),
        "streaming_recurrent_state_elements_lower_bound": recurrent_state_elements,
        "streaming_recurrent_state_bytes_fp32_lower_bound": recurrent_state_elements * 4,
        "event_driven_input_accumulations_counterfactual": (
            float(input_spikes) * hidden if input_dim != hidden else 0.0
        ),
        "event_driven_input_counterfactual_is_actual_execution": 0,
        "hidden_linear_inputs_are_dense_after_layer_norm_and_residual": 1,
        "dense_linear_mac_count_complete": 1,
        "lif_update_count_complete": 1,
        "nonlinear_element_count_complete": 0,
        "memory_access_count_complete": 0,
        "frontend_operation_count_complete": 0,
        "hardware_energy_estimate_ready": 0,
        "hardware_energy_claim_permitted": 0,
    }
