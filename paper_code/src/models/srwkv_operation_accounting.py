"""Algorithmic operation ledger for the existing SRWKV-v2 proxy.

This module counts operations implied by the PyTorch model equations. It does
not estimate wall-plug energy and deliberately separates operations actually
executed by the dense proxy from hypothetical event-driven input projection.
"""

from __future__ import annotations

from typing import Dict


LEDGER_VERSION = "srwkv_v2_algorithmic_ledger_v1"


def srwkv_v2_operation_ledger(
    *,
    modelling_units: int,
    input_dimension: int,
    hidden_dimension: int,
    num_layers: int,
    vocabulary_size: int,
    spike_steps: int,
    input_spikes: float,
) -> Dict[str, int | float | str]:
    """Return auditable counts for one SRWKV-v2 forward workload.

    Dense MAC counts exactly match the Linear layers in
    StackedSpikingRWKVLM: input D-to-H, three H-to-H projections and an
    H-to-4H-to-H channel mixer per block, and the H-to-V readout.

    LIF neuron updates count one update per layer, hidden neuron, spike step,
    and modelling unit. Layer-normalization and nonlinear work are exposed as
    processed elements, not converted into invented MAC equivalents.

    event_driven_input_accumulations_counterfactual maps a binary input code to
    sparse fan-out. The current PyTorch model executes its dense input
    projection, so the counterfactual is never included in dense MAC totals.
    """
    values = {
        "modelling_units": modelling_units,
        "input_dimension": input_dimension,
        "hidden_dimension": hidden_dimension,
        "num_layers": num_layers,
        "vocabulary_size": vocabulary_size,
        "spike_steps": spike_steps,
    }
    for name, value in values.items():
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive")
    if float(input_spikes) < 0:
        raise ValueError("input_spikes must be non-negative")

    units = int(modelling_units)
    input_dim = int(input_dimension)
    hidden = int(hidden_dimension)
    layers = int(num_layers)
    vocab = int(vocabulary_size)
    steps = int(spike_steps)

    input_projection = units * input_dim * hidden
    time_mix_projections = units * layers * 3 * hidden * hidden
    channel_mix_projections = units * layers * 8 * hidden * hidden
    readout = units * hidden * vocab
    total_dense = (
        input_projection
        + time_mix_projections
        + channel_mix_projections
        + readout
    )

    bias_additions = units * (hidden + layers * 8 * hidden + vocab)
    lif_updates = units * layers * hidden * steps
    layer_norm_elements = units * hidden * (2 * layers + 1)
    channel_gelu_elements = units * layers * 4 * hidden

    return {
        "operation_ledger_version": LEDGER_VERSION,
        "dense_input_projection_macs": input_projection,
        "dense_time_mix_projection_macs": time_mix_projections,
        "dense_channel_mix_projection_macs": channel_mix_projections,
        "dense_readout_macs": readout,
        "dense_linear_macs_total": total_dense,
        "dense_bias_additions": bias_additions,
        "lif_neuron_state_updates": lif_updates,
        "layer_norm_elements": layer_norm_elements,
        "channel_gelu_elements": channel_gelu_elements,
        "binary_input_spike_events": float(input_spikes),
        "event_driven_input_accumulations_counterfactual": (
            float(input_spikes) * hidden
        ),
        "event_driven_input_counterfactual_is_actual_execution": 0,
        "dense_linear_mac_count_complete": 1,
        "lif_update_count_complete": 1,
        "nonlinear_operation_count_complete": 0,
        "memory_access_count_complete": 0,
        "frontend_operation_count_complete": 0,
        "hardware_energy_estimate_ready": 0,
        # Retained so pre-existing Stage A rows remain machine-auditable.
        "hardware_energy_claim_permitted": 0,
    }


def training_operation_estimates(
    forward_ledger: Dict[str, int | float | str],
    *,
    trainable_parameters: int,
    optimizer_steps: int,
) -> Dict[str, int | str]:
    """Add explicitly labelled training-only estimates.

    Linear-layer gradients with respect to activations and weights each require
    approximately one forward-sized matrix multiplication. The two-times-
    forward backward value is an estimate, not a measured hardware count.
    AdamW updates are parameter-update elements, not an energy value.
    """
    if int(trainable_parameters) <= 0 or int(optimizer_steps) <= 0:
        raise ValueError("training sizes must be positive")
    forward_macs = int(forward_ledger["dense_linear_macs_total"])
    return {
        "backward_dense_linear_macs_estimate": 2 * forward_macs,
        "backward_dense_linear_macs_estimate_multiplier": 2,
        "optimizer_parameter_update_elements": (
            int(trainable_parameters) * int(optimizer_steps)
        ),
        "training_estimate_status": (
            "forward_dense_macs_exact_backward_linear_macs_estimated_"
            "optimizer_elements_counted_energy_estimate_deferred"
        ),
    }
