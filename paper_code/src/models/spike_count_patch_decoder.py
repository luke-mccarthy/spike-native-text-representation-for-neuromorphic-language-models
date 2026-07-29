"""Codec-free patch LM with an optional spike-count byte output head.

This additive v3 module leaves the audited v1/v2 implementations untouched.
The patch encoder and SpikeGPT reference backbone are unchanged. The dense
reference path reuses ``LocalByteLIFDecoder`` exactly. The spike-output path
reuses the same recurrent LIF decoder layers and replaces only the terminal
real-valued 256-way readout with 256 LIF output neurons whose spike counts are
used as training/evaluation scores and whose emitted events identify bytes at
inference.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from src.models.spikegpt_reference import SpikeGPTReferenceLM, atan_spike
from src.models.spike_native_patch_lm import PositionalPatchSpikeEncoder
from src.models.local_spiking_patch_decoder import (
    PATCH_ARCHITECTURES,
    LocalByteLIFDecoder,
    _LIFDecoderLayer,
)
from src.tokenizers.spiking_blt_frontend import injective_byte_spike_codebook


PATCH_LEDGER_V3_VERSION = "codec_free_patch_v3_spike_count_byte_output"
OUTPUT_HEADS = ("dense_reference", "spike_count_256")


class SpikeCountByteOutputHead(nn.Module):
    """Map decoder spikes to 256 byte neurons over local microsteps."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        microsteps: int = 4,
        beta: float = 0.85,
        threshold: float = 0.50,
        score_scale: float = 2.0,
        seed: int = 1,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or microsteps <= 0:
            raise ValueError("invalid spike-output dimensions")
        if not 0.0 < beta < 1.0 or threshold <= 0.0 or score_scale <= 0.0:
            raise ValueError("invalid spike-output dynamics")
        self.hidden_dim = int(hidden_dim)
        self.microsteps = int(microsteps)
        self.beta = float(beta)
        self.score_scale = float(score_scale)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) + 104729)
        weight = 0.08 * torch.randn(256, hidden_dim + 1, generator=generator)
        weight[:, -1] = 0.05 + 0.005 * torch.randn(256, generator=generator)
        self.output_weight = nn.Parameter(weight)
        self.output_threshold = nn.Parameter(torch.full((256,), float(threshold)))
        self._last_stats: Dict[str, float | int | str] = {}

    def forward(
        self,
        hidden_spikes: torch.Tensor,
        active: torch.Tensor,
        target_bytes: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_spikes.ndim != 2 or hidden_spikes.shape[-1] != self.hidden_dim:
            raise ValueError("hidden_spikes must be [rows, hidden_dim]")
        if active.shape != (hidden_spikes.shape[0], 1):
            raise ValueError("active mask must be [rows, 1]")
        if target_bytes.shape != (hidden_spikes.shape[0],):
            raise ValueError("target_bytes must be [rows]")
        if not torch.all((hidden_spikes.detach() == 0) | (hidden_spikes.detach() == 1)):
            raise ValueError("spike output head received non-binary input")

        rows = hidden_spikes.shape[0]
        device = hidden_spikes.device
        output_input = torch.cat((hidden_spikes * active, active), dim=-1)
        membrane = torch.zeros(rows, 256, device=device)
        spike_counts = torch.zeros_like(membrane)

        decided = torch.zeros(rows, dtype=torch.bool, device=device)
        first_time = torch.zeros(rows, dtype=torch.long, device=device)
        first_choice = torch.zeros(rows, dtype=torch.long, device=device)
        first_tied = torch.zeros(rows, dtype=torch.bool, device=device)

        output_spike_events = 0
        threshold = self.output_threshold.clamp_min(0.05).view(1, 256)
        for microstep in range(self.microsteps):
            candidate = self.beta * membrane + F.linear(output_input, self.output_weight)
            output_spikes = atan_spike(candidate - threshold, alpha=2.0) * active
            membrane = candidate * (1.0 - output_spikes) * active
            spike_counts = spike_counts + output_spikes

            detached = output_spikes.detach()
            output_spike_events += int(detached.sum().item())
            firing = detached.sum(dim=1) > 0
            new = (~decided) & firing & (active[:, 0] > 0)
            if bool(new.any()):
                first_time[new] = microstep + 1
                first_choice[new] = detached[new].argmax(dim=1)
                first_tied[new] = detached[new].sum(dim=1) > 1
            decided = decided | new

        detached_counts = spike_counts.detach()
        active_rows = active[:, 0] > 0
        total_counts = detached_counts.sum(dim=1)
        no_spike = active_rows & (total_counts == 0)
        maxima = detached_counts.max(dim=1).values
        winner_ties = active_rows & ((detached_counts == maxima.unsqueeze(1)).sum(dim=1) > 1)
        count_choice = detached_counts.argmax(dim=1)
        unique_first = decided & (~first_tied)

        active_bytes = int(active_rows.sum().item())
        input_events = int((output_input.detach() != 0).sum(dim=1)[active_rows].sum().item())
        self._last_stats = {
            "decoder_output_head": "spike_count_256",
            "decoder_output_microsteps": self.microsteps,
            "decoder_output_score_scale": self.score_scale,
            "decoder_output_spike_events": output_spike_events,
            "decoder_output_synaptic_accumulations": input_events * 256 * self.microsteps,
            "decoder_output_lif_state_updates": active_bytes * 256 * self.microsteps,
            "decoder_output_threshold_reset_elements": active_bytes * 256 * self.microsteps * 2,
            "decoder_output_no_spike_bytes": int(no_spike.sum().item()),
            "decoder_output_tied_winner_bytes": int(winner_ties.sum().item()),
            "decoder_output_first_spike_covered_bytes": int((decided & active_rows).sum().item()),
            "decoder_output_first_spike_tied_bytes": int((first_tied & active_rows).sum().item()),
            "decoder_output_first_spike_latency_sum": int(first_time[decided & active_rows].sum().item()),
            "decoder_output_first_spike_unique_correct_bytes": int((unique_first & active_rows & (first_choice == target_bytes)).sum().item()),
            "decoder_output_spike_count_correct_bytes": int((active_rows & (count_choice == target_bytes)).sum().item()),
            "decoder_output_active_bytes": active_bytes,
            "decoder_output_scores_are_spike_counts": 1,
            "decoder_output_neural_path_is_fully_spiking": 1,
            "decoder_output_dense_linear_readout_present": 0,
            "decoder_output_winner_selection_external_reference_control": 1,
        }
        return spike_counts * self.score_scale

    def last_operation_stats(self) -> Dict[str, float | int | str]:
        return dict(self._last_stats)


class LocalByteLIFDecoderSpikeOutput(nn.Module):
    """V2 recurrent LIF decoder with a spike-count terminal byte head."""

    def __init__(
        self,
        *,
        context_dim: int,
        max_patch_bytes: int,
        decoder_hidden_dim: int = 128,
        decoder_num_layers: int = 2,
        input_fanout: int = 8,
        recurrent_fanout: int = 4,
        output_microsteps: int = 4,
        output_score_scale: float = 2.0,
        seed: int = 1,
    ) -> None:
        super().__init__()
        if context_dim <= 0 or max_patch_bytes <= 1:
            raise ValueError("invalid local decoder dimensions")
        if decoder_hidden_dim <= 0 or decoder_num_layers <= 0:
            raise ValueError("decoder capacity must be positive")
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
        self.output_head = SpikeCountByteOutputHead(
            hidden_dim=decoder_hidden_dim,
            microsteps=output_microsteps,
            score_scale=output_score_scale,
            seed=seed,
        )
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
        rows = batch * steps
        flattened_context = context_spikes.reshape(rows, self.context_dim)
        flattened_targets = target_bytes.reshape(rows, self.max_patch_bytes)
        flattened_mask = target_mask.reshape(rows, self.max_patch_bytes)
        device = context_spikes.device

        states = [layer.init_state(rows, device) for layer in self.layers]
        previous_byte = torch.zeros(rows, 32, device=device)
        scores: List[torch.Tensor] = []
        hidden_events = 0
        input_accumulations = 0
        recurrent_accumulations = 0
        output_totals: Dict[str, float] = {}
        layer_input_fanout = [layer.input_mask.sum(dim=1) for layer in self.layers]
        layer_recurrent_fanout = [layer.recurrent_mask.sum(dim=0) for layer in self.layers]

        for position in range(self.max_patch_bytes):
            active = flattened_mask[:, position].to(flattened_context.dtype).unsqueeze(1)
            position_spikes = torch.zeros(rows, self.max_patch_bytes, device=device)
            position_spikes[:, position] = active[:, 0]
            layer_input = torch.cat((flattened_context * active, previous_byte, position_spikes), dim=-1)
            final_spikes = None
            for layer_index, layer in enumerate(self.layers):
                membrane, previous_spikes = states[layer_index]
                active_inputs = int((layer_input.detach() != 0).sum().item())
                input_accumulations += active_inputs * int(layer_input_fanout[layer_index].float().mean().item())
                recurrent_accumulations += int((previous_spikes.detach().sum(dim=-1) * layer_recurrent_fanout[layer_index].float().mean()).sum().item())
                membrane, final_spikes = layer.step(layer_input, membrane, previous_spikes)
                membrane = membrane * active
                final_spikes = final_spikes * active
                states[layer_index] = (membrane, final_spikes)
                layer_input = final_spikes
            if final_spikes is None:
                raise AssertionError("decoder produced no hidden layer")
            hidden_events += int(final_spikes.detach().sum().item())
            scores.append(self.output_head(final_spikes, active, flattened_targets[:, position]))
            for key, value in self.output_head.last_operation_stats().items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    output_totals[key] = output_totals.get(key, 0.0) + float(value)
            previous_byte = self.byte_codebook[flattened_targets[:, position]] * active

        output = torch.stack(scores, dim=1).reshape(batch, steps, self.max_patch_bytes, 256)
        active_bytes = int(target_mask.sum().item())
        context_events = int((flattened_context.detach() * flattened_mask.any(dim=1, keepdim=True)).sum().item())
        teacher_events = max(active_bytes - rows, 0) * 2
        output_synops = int(output_totals.get("decoder_output_synaptic_accumulations", 0.0))
        output_updates = int(output_totals.get("decoder_output_lif_state_updates", 0.0))
        output_control = int(output_totals.get("decoder_output_threshold_reset_elements", 0.0))
        self._last_stats = {
            "frontend_operation_ledger_version": PATCH_LEDGER_V3_VERSION,
            "decoder_context_spike_events": context_events,
            "decoder_teacher_byte_spike_events": teacher_events,
            "decoder_position_spike_events": active_bytes,
            "decoder_hidden_spike_events": hidden_events,
            "decoder_hidden_dim": self.decoder_hidden_dim,
            "decoder_num_layers": self.decoder_num_layers,
            "decoder_input_synaptic_accumulations": input_accumulations,
            "decoder_recurrent_synaptic_accumulations": recurrent_accumulations,
            "decoder_readout_synaptic_accumulations": output_synops,
            "decoder_lif_state_updates": active_bytes * self.decoder_hidden_dim * self.decoder_num_layers + output_updates,
            "decoder_threshold_reset_elements": active_bytes * self.decoder_hidden_dim * self.decoder_num_layers * 2 + output_control,
            "decoder_dense_reference_readout_macs": active_bytes * (self.decoder_hidden_dim + 1) * 256,
            "decoder_likelihood_exponential_elements": active_bytes * 256,
            "decoder_likelihood_reduction_add_elements": active_bytes * 255,
            "decoder_likelihood_log_elements": active_bytes,
            "decoder_likelihood_target_subtraction_elements": active_bytes,
            "decoder_binary_internal_communication": 1,
            "decoder_capacity_decoupled_from_global_backbone": 1,
            "decoder_dense_likelihood_readout_present": 0,
            **{
                key: int(value) if float(value).is_integer() else float(value)
                for key, value in output_totals.items()
            },
        }
        return output

    def last_operation_stats(self) -> Dict[str, float | int | str]:
        return dict(self._last_stats)


class SpikeCountPatchLanguageModel(nn.Module):
    """Unchanged patch encoder/backbone with selectable terminal byte head."""

    def __init__(
        self,
        *,
        max_patch_bytes: int,
        hierarchical_encoder: bool,
        output_head: str,
        hidden_dim: int = 64,
        num_layers: int = 2,
        decoder_hidden_dim: int = 128,
        decoder_num_layers: int = 2,
        output_microsteps: int = 4,
        output_score_scale: float = 2.0,
        seed: int = 1,
    ) -> None:
        super().__init__()
        if output_head not in OUTPUT_HEADS:
            raise ValueError(f"unsupported output head: {output_head}")
        self.output_head_name = output_head
        self.encoder = PositionalPatchSpikeEncoder(
            max_patch_bytes=max_patch_bytes,
            hierarchical=hierarchical_encoder,
            seed=seed,
        )
        self.backbone = SpikeGPTReferenceLM(
            vocab_size=2,
            input_dim=self.encoder.output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=0.03,
        )
        self.backbone.final_ln = nn.Identity()
        self.backbone.readout = nn.Identity()
        decoder_kwargs = dict(
            context_dim=hidden_dim,
            max_patch_bytes=max_patch_bytes,
            decoder_hidden_dim=decoder_hidden_dim,
            decoder_num_layers=decoder_num_layers,
            seed=seed,
        )
        if output_head == "dense_reference":
            self.decoder: nn.Module = LocalByteLIFDecoder(**decoder_kwargs)
        else:
            self.decoder = LocalByteLIFDecoderSpikeOutput(
                **decoder_kwargs,
                output_microsteps=output_microsteps,
                output_score_scale=output_score_scale,
            )
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.max_patch_bytes = int(max_patch_bytes)
        self.decoder_hidden_dim = int(decoder_hidden_dim)
        self.decoder_num_layers = int(decoder_num_layers)
        self.output_microsteps = int(output_microsteps)
        self.output_score_scale = float(output_score_scale)

    def forward(
        self,
        source_bytes: torch.Tensor,
        source_mask: torch.Tensor,
        target_bytes: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        input_spikes = self.encoder(source_bytes, source_mask)
        _, context_spikes, backbone_stats = self.backbone.forward_features(inputs_spikes=input_spikes)
        scores = self.decoder(context_spikes, target_bytes, target_mask)
        spike_head = int(self.output_head_name == "spike_count_256")
        stats: Dict[str, object] = {
            "frontend_operation_ledger_version": PATCH_LEDGER_V3_VERSION,
            "global_patch_steps": int(source_bytes.shape[0] * source_bytes.shape[1]),
            "global_input_dimension": self.encoder.output_dim,
            "global_input_spike_events": int(input_spikes.detach().sum().item()),
            "fully_spike_communicating_patch_encoder": 1,
            "fully_spike_communicating_global_to_local_decoder": 1,
            "digital_tokenizer_present": 0,
            "digital_patch_codec_present": 0,
            "spiking_byte_output_neurons_present": spike_head,
            "dense_final_byte_likelihood_readout_present": 1 - spike_head,
            "fully_spiking_decoder_neural_path": spike_head,
            "output_scores_derived_from_spike_counts": spike_head,
            "output_winner_selection_external_reference_control": spike_head,
            **{
                key: value
                for key, value in self.encoder.last_operation_stats().items()
                if key != "frontend_operation_ledger_version"
            },
            **{
                key: value
                for key, value in self.decoder.last_operation_stats().items()
                if key != "frontend_operation_ledger_version"
            },
            **{f"backbone_{key}": value for key, value in backbone_stats.items()},
        }
        return scores, stats


__all__ = [
    "OUTPUT_HEADS",
    "PATCH_ARCHITECTURES",
    "PATCH_LEDGER_V3_VERSION",
    "LocalByteLIFDecoderSpikeOutput",
    "SpikeCountByteOutputHead",
    "SpikeCountPatchLanguageModel",
]
