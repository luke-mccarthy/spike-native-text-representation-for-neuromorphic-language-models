"""Second-generation codec-free spike-native patch interface.

This module extends ``src/models/spike_native_patch_lm.py`` (2026-07-23)
along the first two of three directions identified from the audited
90-row development archive at
``docs/experiment_records/2026_07_23_scalable_patch_frontends_development``:

1. The local byte decoder previously shared its hidden dimension and depth
   (one LIF layer) with the shared global backbone, despite doing genuinely
   new autoregressive work -- predicting up to eight bytes per patch -- that
   the original SpikeGPT readout never performed. ``LocalByteLIFDecoder``
   decouples ``decoder_hidden_dim`` and ``decoder_num_layers`` from the
   global backbone entirely.
2. Training-time knowledge distillation from a frozen, fully spiking
   byte-rate teacher is supported (``src/models/patch_distillation.py``),
   without adding any component to the deployed inference path: the teacher
   is never invoked outside the training loop, and nothing about the
   student's forward pass changes when distillation is disabled.

The original v1 module is left completely unmodified so every row in the
existing audited archive remains exactly reproducible from the commit that
produced it. This module deliberately reuses v1's patch encoder,
``PatchSequence``, and block-traversal helpers rather than duplicating them,
since those components were not implicated by the development-archive
findings.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from src.models.spikegpt_reference import SpikeGPTReferenceLM, atan_spike
from src.models.spike_native_patch_lm import (
    PATCH_ARCHITECTURES,
    PatchSequence,
    PositionalPatchSpikeEncoder,
    complete_patch_blocks,
    fixed_width_patches,
    merge_additive_stats,
    validate_patch_cover,
)
from src.tokenizers.spiking_blt_frontend import injective_byte_spike_codebook


PATCH_LEDGER_V2_VERSION = "codec_free_spike_native_patch_interface_v2_decoupled_decoder"


def _sparse_mask(rows: int, columns: int, fanout: int, *, generator: torch.Generator) -> torch.Tensor:
    if not 0 < fanout <= rows:
        raise ValueError("fanout must lie in [1, rows]")
    mask = torch.zeros(rows, columns)
    for source in range(columns):
        mask[torch.randperm(rows, generator=generator)[:fanout], source] = 1.0
    return mask


class _LIFDecoderLayer(nn.Module):
    """One masked, sparse, trainable recurrent LIF layer for the local decoder stack."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        input_fanout: int,
        recurrent_fanout: int,
        threshold: float,
        beta: float,
        seed: int,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("invalid decoder layer dimensions")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        input_mask = _sparse_mask(hidden_dim, input_dim, input_fanout, generator=generator)
        recurrent_mask = _sparse_mask(hidden_dim, hidden_dim, recurrent_fanout, generator=generator)
        self.register_buffer("input_mask", input_mask)
        self.register_buffer("recurrent_mask", recurrent_mask)
        # Scale synaptic weights down as the input fans out to more of the
        # layer (rather than a fixed magnitude regardless of dimension), so
        # summed input current stays in a comparable range as hidden_dim
        # changes and the layer does not default into all-neurons-fire
        # saturation, which would make every downstream position
        # indistinguishable regardless of its true input.
        input_scale = 0.35 / max(input_fanout, 1) ** 0.5
        recurrent_scale = 0.20 / max(recurrent_fanout, 1) ** 0.5
        self.input_weight = nn.Parameter(
            input_mask * input_scale
            * (0.5 + 1.0 * torch.rand(hidden_dim, input_dim, generator=generator))
        )
        recurrent_sign = torch.where(
            torch.rand(hidden_dim, hidden_dim, generator=generator) < 0.8,
            torch.ones(hidden_dim, hidden_dim),
            -torch.ones(hidden_dim, hidden_dim),
        )
        self.recurrent_weight = nn.Parameter(
            recurrent_mask * recurrent_sign * recurrent_scale
            * (0.5 + 1.0 * torch.rand(hidden_dim, hidden_dim, generator=generator))
        )
        self.hidden_dim = int(hidden_dim)
        self.beta = float(beta)
        self.threshold = float(threshold)

    def step(
        self, layer_input: torch.Tensor, membrane: torch.Tensor, previous_spikes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        candidate = (
            self.beta * membrane
            + F.linear(layer_input, self.input_weight * self.input_mask)
            + F.linear(previous_spikes, self.recurrent_weight * self.recurrent_mask)
        )
        spikes = atan_spike(candidate - self.threshold, alpha=2.0)
        next_membrane = candidate * (1.0 - spikes)
        return next_membrane, spikes

    def init_state(self, batch: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(batch, self.hidden_dim, device=device),
            torch.zeros(batch, self.hidden_dim, device=device),
        )


class LocalByteLIFDecoder(nn.Module):
    """Autoregressive local byte decoder with a decoupled capacity budget.

    Structurally identical in spirit to
    ``src.models.spike_native_patch_lm.LocalByteLIFDecoder`` (same causal
    teacher-forced within-patch recurrence, same binary global/local
    boundary, same dense 256-way likelihood readout) but with
    ``decoder_hidden_dim`` and ``decoder_num_layers`` set independently of
    the global backbone and of each other. Layer 0 receives the masked
    patch context, the previous decoded byte's fixed injective code, and a
    one-hot within-patch position code; layers above 0 receive only the
    previous layer's spikes, so added depth adds genuinely new recurrent
    transformation rather than repeating the same input.
    """

    def __init__(
        self,
        *,
        context_dim: int,
        max_patch_bytes: int,
        decoder_hidden_dim: int = 128,
        decoder_num_layers: int = 2,
        input_fanout: int = 8,
        recurrent_fanout: int = 4,
        seed: int = 1,
    ) -> None:
        super().__init__()
        if context_dim <= 0 or max_patch_bytes <= 1:
            raise ValueError("invalid local decoder dimensions")
        if decoder_hidden_dim <= 0 or decoder_num_layers <= 0:
            raise ValueError("decoder_hidden_dim and decoder_num_layers must be positive")
        self.context_dim = int(context_dim)
        self.max_patch_bytes = int(max_patch_bytes)
        self.decoder_hidden_dim = int(decoder_hidden_dim)
        self.decoder_num_layers = int(decoder_num_layers)
        self.register_buffer("byte_codebook", injective_byte_spike_codebook(32, 2))

        first_input_dim = context_dim + 32 + max_patch_bytes
        layers: List[_LIFDecoderLayer] = []
        for layer_index in range(decoder_num_layers):
            layer_input_dim = first_input_dim if layer_index == 0 else decoder_hidden_dim
            layers.append(
                _LIFDecoderLayer(
                    input_dim=layer_input_dim,
                    hidden_dim=decoder_hidden_dim,
                    input_fanout=min(input_fanout, layer_input_dim),
                    recurrent_fanout=min(recurrent_fanout, decoder_hidden_dim),
                    threshold=0.75,
                    beta=0.85,
                    seed=seed + 7919 + layer_index,
                )
            )
        self.layers = nn.ModuleList(layers)
        self.readout = nn.Linear(decoder_hidden_dim, 256, bias=False)
        nn.init.normal_(self.readout.weight, mean=0.0, std=0.02)
        self._last_stats: Dict[str, float | int | str] = {}

    def forward(
        self,
        context_spikes: torch.Tensor,
        target_bytes: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        if (
            context_spikes.ndim != 3
            or context_spikes.shape[:2] != target_bytes.shape[:2]
            or context_spikes.shape[-1] != self.context_dim
            or target_bytes.shape != target_mask.shape
            or target_bytes.shape[-1] != self.max_patch_bytes
        ):
            raise ValueError("local decoder received incompatible tensors")
        if not torch.all((context_spikes.detach() == 0) | (context_spikes.detach() == 1)):
            raise ValueError("local decoder context must be binary spikes")
        batch, steps, _ = context_spikes.shape
        flattened_context = context_spikes.reshape(batch * steps, self.context_dim)
        flattened_targets = target_bytes.reshape(batch * steps, self.max_patch_bytes)
        flattened_mask = target_mask.reshape(batch * steps, self.max_patch_bytes)
        rows = batch * steps
        device = context_spikes.device

        states = [layer.init_state(rows, device) for layer in self.layers]
        previous_byte = torch.zeros(rows, 32, device=device)
        logits: List[torch.Tensor] = []
        hidden_events = 0
        input_accumulations = 0
        recurrent_accumulations = 0
        layer_input_fanout = [
            layer.input_mask.sum(dim=0) for layer in self.layers
        ]  # outgoing synapses per input channel
        layer_recurrent_fanout = [layer.recurrent_mask.sum(dim=0) for layer in self.layers]
        for position in range(self.max_patch_bytes):
            active = flattened_mask[:, position].to(flattened_context.dtype).unsqueeze(1)
            position_spikes = torch.zeros(rows, self.max_patch_bytes, device=device)
            position_spikes[:, position] = active[:, 0]
            layer_input = torch.cat(
                (flattened_context * active, previous_byte, position_spikes), dim=-1
            )
            spikes = None
            for layer_index, layer in enumerate(self.layers):
                membrane, previous_spikes = states[layer_index]
                # Event-driven accumulation counts: one accumulate per active
                # input channel per destination neuron it fans out to, and
                # likewise for the previous step's recurrent spikes.
                input_accumulations += int(
                    (
                        (layer_input.detach() != 0).to(layer_input_fanout[layer_index].dtype)
                        * layer_input_fanout[layer_index]
                    ).sum().item()
                )
                recurrent_accumulations += int(
                    (previous_spikes.detach().sum(dim=-1) * layer_recurrent_fanout[layer_index].mean()).sum().item()
                )
                membrane, spikes = layer.step(layer_input, membrane, previous_spikes)
                membrane = membrane * active
                spikes = spikes * active
                states[layer_index] = (membrane, spikes)
                layer_input = spikes
            hidden_events += int(spikes.detach().sum().item())
            logits.append(self.readout(spikes))
            literal = flattened_targets[:, position]
            previous_byte = self.byte_codebook[literal] * active
        output = torch.stack(logits, dim=1).reshape(batch, steps, self.max_patch_bytes, 256)
        active_bytes = int(target_mask.sum().item())
        context_events = int((flattened_context.detach() * flattened_mask.any(dim=1, keepdim=True)).sum().item())
        readout_accumulations = hidden_events * 256
        # Teacher-forcing signal events: the previous decoded byte's fixed
        # 2-of-32 code is presented at every position after the first
        # active one in each patch, mirroring v1's accounting convention.
        teacher_events = max(active_bytes - rows, 0) * 2
        self._last_stats = {
            "frontend_operation_ledger_version": PATCH_LEDGER_V2_VERSION,
            "decoder_context_spike_events": context_events,
            "decoder_teacher_byte_spike_events": teacher_events,
            "decoder_position_spike_events": active_bytes,
            "decoder_hidden_spike_events": hidden_events,
            "decoder_hidden_dim": self.decoder_hidden_dim,
            "decoder_num_layers": self.decoder_num_layers,
            "decoder_input_synaptic_accumulations": input_accumulations,
            "decoder_recurrent_synaptic_accumulations": recurrent_accumulations,
            "decoder_readout_synaptic_accumulations": readout_accumulations,
            "decoder_lif_state_updates": active_bytes * self.decoder_hidden_dim * self.decoder_num_layers,
            "decoder_threshold_reset_elements": active_bytes * self.decoder_hidden_dim * self.decoder_num_layers * 2,
            "decoder_dense_reference_readout_macs": active_bytes * self.decoder_hidden_dim * 256,
            "decoder_likelihood_exponential_elements": active_bytes * 256,
            "decoder_likelihood_reduction_add_elements": active_bytes * 255,
            "decoder_likelihood_log_elements": active_bytes,
            "decoder_likelihood_target_subtraction_elements": active_bytes,
            "decoder_binary_internal_communication": 1,
            "decoder_capacity_decoupled_from_global_backbone": 1,
        }
        return output

    def last_operation_stats(self) -> Dict[str, float | int | str]:
        return dict(self._last_stats)


class LocalDecoderPatchLanguageModel(nn.Module):
    """Unchanged global SpikeGPT blocks plus a capacity-decoupled local decoder."""

    def __init__(
        self,
        *,
        max_patch_bytes: int,
        hierarchical_encoder: bool,
        hidden_dim: int = 64,
        num_layers: int = 2,
        decoder_hidden_dim: int = 128,
        decoder_num_layers: int = 2,
        seed: int = 1,
    ) -> None:
        super().__init__()
        self.encoder = PositionalPatchSpikeEncoder(
            max_patch_bytes=max_patch_bytes, hierarchical=hierarchical_encoder, seed=seed,
        )
        self.backbone = SpikeGPTReferenceLM(
            vocab_size=2, input_dim=self.encoder.output_dim, hidden_dim=hidden_dim,
            num_layers=num_layers, dropout=0.03,
        )
        self.backbone.final_ln = nn.Identity()
        self.backbone.readout = nn.Identity()
        self.decoder = LocalByteLIFDecoder(
            context_dim=hidden_dim,
            max_patch_bytes=max_patch_bytes,
            decoder_hidden_dim=decoder_hidden_dim,
            decoder_num_layers=decoder_num_layers,
            seed=seed,
        )
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.max_patch_bytes = int(max_patch_bytes)

    def forward(
        self,
        source_bytes: torch.Tensor,
        source_mask: torch.Tensor,
        target_bytes: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        input_spikes = self.encoder(source_bytes, source_mask)
        _, context_spikes, backbone_stats = self.backbone.forward_features(inputs_spikes=input_spikes)
        logits = self.decoder(context_spikes, target_bytes, target_mask)
        stats: Dict[str, object] = {
            "frontend_operation_ledger_version": PATCH_LEDGER_V2_VERSION,
            "global_patch_steps": int(source_bytes.shape[0] * source_bytes.shape[1]),
            "global_input_dimension": self.encoder.output_dim,
            "global_input_spike_events": int(input_spikes.detach().sum().item()),
            "fully_spike_communicating_patch_encoder": 1,
            "fully_spike_communicating_global_to_local_decoder": 1,
            "digital_tokenizer_present": 0,
            "digital_patch_codec_present": 0,
            **{k: v for k, v in self.encoder.last_operation_stats().items() if k != "frontend_operation_ledger_version"},
            **{k: v for k, v in self.decoder.last_operation_stats().items() if k != "frontend_operation_ledger_version"},
            **{f"backbone_{k}": v for k, v in backbone_stats.items()},
        }
        return logits, stats


__all__ = [
    "PATCH_ARCHITECTURES",
    "PATCH_LEDGER_V2_VERSION",
    "PatchSequence",
    "LocalByteLIFDecoder",
    "LocalDecoderPatchLanguageModel",
    "complete_patch_blocks",
    "fixed_width_patches",
    "merge_additive_stats",
    "validate_patch_cover",
]
