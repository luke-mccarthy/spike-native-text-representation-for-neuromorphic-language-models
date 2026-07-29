"""Strict spike-output patch LM with direct and hierarchical TTFS heads.

The deployed neural path for both strict heads is binary/event-driven at every
module boundary: patch spikes -> SpikeGPT channel spikes -> local LIF decoder
spikes -> output first-spike events.  Surrogate quantities exist only in the
backward graph.  The module also supports teacher-forced likelihood evaluation
and free-running spike-feedback evaluation with the same trained weights.
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
from src.models.hierarchical_ttfs_patch_decoder import HierarchicalFirstSpikeByteHead
from src.tokenizers.spiking_blt_frontend import injective_byte_spike_codebook

PATCH_LEDGER_V6_VERSION = "codec_free_patch_v6_strict_ttfs_direct_and_nibble"
OUTPUT_HEADS = ("dense_reference", "ttfs_direct_256", "ttfs_nibble_2x16")


class DirectFirstSpikeByteHead(nn.Module):
    """One deterministic one-spike LIF output neuron per byte class."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        microsteps: int = 16,
        beta: float = 0.90,
        threshold: float = 1.0,
        weight_scale: float = 0.04,
        clock_end: float = 0.20,
        latency_tau: float = 6.0,
        score_scale: float = 8.0,
        surrogate_temperature: float = 0.20,
        seed: int = 1,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or microsteps < 4:
            raise ValueError("invalid direct TTFS dimensions")
        if not 0.0 < beta < 1.0:
            raise ValueError("beta must lie in (0, 1)")
        for name, value in (
            ("threshold", threshold),
            ("weight_scale", weight_scale),
            ("clock_end", clock_end),
            ("latency_tau", latency_tau),
            ("score_scale", score_scale),
            ("surrogate_temperature", surrogate_temperature),
        ):
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
        self.hidden_dim = int(hidden_dim)
        self.microsteps = int(microsteps)
        self.beta = float(beta)
        self.threshold = float(threshold)
        self.weight_scale = float(weight_scale)
        self.clock_end = float(clock_end)
        self.latency_tau = float(latency_tau)
        self.score_scale = float(score_scale)
        self.surrogate_temperature = float(surrogate_temperature)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) + 32452843)
        self.input_weight = nn.Parameter(
            self.weight_scale * torch.randn(256, hidden_dim, generator=generator)
        )
        self.register_buffer("threshold_vector", torch.full((256,), self.threshold))
        self._last_stats: Dict[str, float | int | str] = {}
        self._last_margin_loss: torch.Tensor | None = None

    def forward(
        self,
        hidden_spikes: torch.Tensor,
        active: torch.Tensor,
        target_bytes: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_spikes.ndim != 2 or hidden_spikes.shape[-1] != self.hidden_dim:
            raise ValueError("hidden_spikes must be [rows, hidden_dim]")
        if active.shape != (hidden_spikes.shape[0], 1):
            raise ValueError("active must be [rows, 1]")
        if not torch.all((hidden_spikes.detach() == 0) | (hidden_spikes.detach() == 1)):
            raise ValueError("direct TTFS input must be binary")

        rows = hidden_spikes.shape[0]
        device = hidden_spikes.device
        active_rows = active[:, 0] > 0
        current = F.linear(hidden_spikes * active, self.input_weight)
        membrane = torch.zeros(rows, 256, device=device)
        hard_available = torch.ones_like(membrane)
        soft_survival = torch.ones_like(membrane)
        hard_score = torch.zeros_like(membrane)
        soft_score = torch.zeros_like(membrane)
        spike_count = torch.zeros_like(membrane)
        first_time = torch.zeros(rows, 256, dtype=torch.long, device=device)
        threshold = self.threshold_vector.view(1, 256)

        for step in range(self.microsteps):
            phase = float(step + 1) / float(self.microsteps)
            candidate = self.beta * membrane + current + self.clock_end * phase
            raw_spike = atan_spike(candidate - threshold, alpha=2.0)
            hard_spike = raw_spike * hard_available * active
            newly = (hard_spike.detach() > 0) & (first_time == 0)
            first_time[newly] = step + 1
            hard_available = hard_available * (1.0 - hard_spike)
            membrane = candidate * hard_available * active
            spike_count = spike_count + hard_spike
            latency_weight = torch.exp(
                torch.tensor(
                    -float(step + 1) / self.latency_tau,
                    device=device,
                    dtype=hidden_spikes.dtype,
                )
            )
            hard_score = hard_score + latency_weight * hard_spike
            hazard = torch.sigmoid(
                (candidate - threshold) / self.surrogate_temperature
            ) * soft_survival * active
            soft_score = soft_score + latency_weight * hazard
            soft_survival = soft_survival * (1.0 - hazard)

        scores = (
            hard_score.detach() + soft_score - soft_score.detach()
        ) * self.score_scale
        if bool(active_rows.any()):
            active_scores = scores[active_rows]
            active_targets = target_bytes[active_rows]
            correct = active_scores.gather(1, active_targets.unsqueeze(1)).squeeze(1)
            competitors = active_scores.clone()
            competitors.scatter_(1, active_targets.unsqueeze(1), float("-inf"))
            self._last_margin_loss = F.relu(
                0.25 - (correct - competitors.max(dim=1).values)
            ).mean()
        else:
            self._last_margin_loss = scores.sum() * 0.0

        detached = hard_score.detach() * self.score_scale
        earliest = torch.where(
            first_time > 0,
            first_time,
            torch.full_like(first_time, self.microsteps + 1),
        )
        minimum = earliest.min(dim=1).values
        covered = active_rows & (minimum <= self.microsteps)
        first_ties = active_rows & (
            (earliest == minimum.unsqueeze(1)).sum(dim=1) > 1
        )
        maxima = detached.max(dim=1).values
        score_ties = active_rows & (
            torch.isclose(detached, maxima.unsqueeze(1), atol=1e-7, rtol=0.0)
            .sum(dim=1) > 1
        )
        first_choice = earliest.argmin(dim=1)
        score_choice = detached.argmax(dim=1)
        active_bytes = int(active_rows.sum().item())
        input_events = int((hidden_spikes.detach() * active).sum().item())
        self._last_stats = {
            "decoder_output_head": "ttfs_direct_256",
            "decoder_output_microsteps": self.microsteps,
            "decoder_output_neurons": 256,
            "decoder_output_spike_events": int(spike_count.detach().sum().item()),
            "decoder_output_early_exit_spike_events": int(covered.sum().item()),
            "decoder_output_synaptic_accumulations": input_events * 256,
            "decoder_output_lif_state_updates": active_bytes * 256 * self.microsteps,
            "decoder_output_threshold_reset_elements": active_bytes * 256 * self.microsteps * 2,
            "decoder_output_no_spike_bytes": int((active_rows & (~covered)).sum().item()),
            "decoder_output_tied_winner_bytes": int(score_ties.sum().item()),
            "decoder_output_unique_winner_bytes": int((active_rows & (~score_ties)).sum().item()),
            "decoder_output_first_spike_covered_bytes": int(covered.sum().item()),
            "decoder_output_first_spike_tied_bytes": int(first_ties.sum().item()),
            "decoder_output_first_spike_latency_sum": int(minimum[covered].sum().item()),
            "decoder_output_first_spike_unique_correct_bytes": int((covered & (~first_ties) & (first_choice == target_bytes)).sum().item()),
            "decoder_output_temporal_score_correct_bytes": int((active_rows & (score_choice == target_bytes)).sum().item()),
            "decoder_output_active_bytes": active_bytes,
            "decoder_output_scores_are_first_spike_latency": 1,
            "decoder_output_exact_lossless_byte_code": 1,
            "decoder_output_neural_path_is_fully_spiking": 1,
            "decoder_output_dense_linear_readout_present": 0,
            "decoder_output_fixed_combinatorial_byte_decoder": 0,
            "decoder_output_winner_selection_external_reference_control": 1,
        }
        return scores

    def margin_loss(self) -> torch.Tensor:
        if self._last_margin_loss is None:
            raise RuntimeError("margin loss requested before forward")
        return self._last_margin_loss

    def last_operation_stats(self) -> Dict[str, float | int | str]:
        return dict(self._last_stats)


class LocalByteLIFDecoderStrictTTFS(nn.Module):
    """Sparse recurrent LIF decoder with teacher or predicted spike feedback."""

    def __init__(
        self,
        *,
        context_dim: int,
        max_patch_bytes: int,
        output_head: str,
        decoder_hidden_dim: int = 128,
        decoder_num_layers: int = 2,
        input_fanout: int = 8,
        recurrent_fanout: int = 4,
        output_microsteps: int = 16,
        output_beta: float = 0.90,
        output_threshold: float = 1.0,
        output_weight_scale: float = 0.04,
        output_clock_end: float = 0.20,
        output_latency_tau: float = 6.0,
        output_score_scale: float = 8.0,
        output_surrogate_temperature: float = 0.20,
        seed: int = 1,
    ) -> None:
        super().__init__()
        if output_head not in {"ttfs_direct_256", "ttfs_nibble_2x16"}:
            raise ValueError("strict decoder requires a TTFS output head")
        self.context_dim = int(context_dim)
        self.max_patch_bytes = int(max_patch_bytes)
        self.decoder_hidden_dim = int(decoder_hidden_dim)
        self.decoder_num_layers = int(decoder_num_layers)
        self.output_head_name = output_head
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
        common = dict(
            hidden_dim=decoder_hidden_dim,
            microsteps=output_microsteps,
            beta=output_beta,
            threshold=output_threshold,
            weight_scale=output_weight_scale,
            clock_end=output_clock_end,
            latency_tau=output_latency_tau,
            score_scale=output_score_scale,
            surrogate_temperature=output_surrogate_temperature,
            seed=seed,
        )
        self.output_head: nn.Module = (
            DirectFirstSpikeByteHead(**common)
            if output_head == "ttfs_direct_256"
            else HierarchicalFirstSpikeByteHead(**common)
        )
        self._last_stats: Dict[str, float | int | str] = {}
        self._last_margin_loss: torch.Tensor | None = None

    def forward(
        self,
        context_spikes: torch.Tensor,
        target_bytes: torch.Tensor,
        target_mask: torch.Tensor,
        *,
        feedback_mode: str = "teacher",
    ) -> torch.Tensor:
        if feedback_mode not in {"teacher", "free_running"}:
            raise ValueError("feedback_mode must be teacher or free_running")
        if context_spikes.ndim != 3 or context_spikes.shape[-1] != self.context_dim:
            raise ValueError("invalid context shape")
        if target_bytes.shape != target_mask.shape or target_bytes.shape[:2] != context_spikes.shape[:2]:
            raise ValueError("invalid target tensors")
        if not torch.all((context_spikes.detach() == 0) | (context_spikes.detach() == 1)):
            raise ValueError("decoder context must be binary")

        batch, steps, _ = context_spikes.shape
        rows = batch * steps
        context = context_spikes.reshape(rows, self.context_dim)
        targets = target_bytes.reshape(rows, self.max_patch_bytes)
        mask = target_mask.reshape(rows, self.max_patch_bytes)
        device = context_spikes.device
        states = [layer.init_state(rows, device) for layer in self.layers]
        previous_byte = torch.zeros(rows, 32, device=device)
        scores: List[torch.Tensor] = []
        hidden_events = 0
        input_accumulations = 0
        recurrent_accumulations = 0
        output_totals: Dict[str, float] = {}
        margin_loss = torch.zeros((), device=device)
        fanin = [layer.input_mask.sum(dim=1).float().mean() for layer in self.layers]
        recurrent_fanin = [layer.recurrent_mask.sum(dim=0).float().mean() for layer in self.layers]

        for position in range(self.max_patch_bytes):
            active = mask[:, position].to(context.dtype).unsqueeze(1)
            position_spikes = torch.zeros(rows, self.max_patch_bytes, device=device)
            position_spikes[:, position] = active[:, 0]
            layer_input = torch.cat((context * active, previous_byte, position_spikes), dim=-1)
            final_spikes = None
            for index, layer in enumerate(self.layers):
                membrane, previous_spikes = states[index]
                input_accumulations += int((layer_input.detach() != 0).sum().item() * fanin[index].item())
                recurrent_accumulations += int((previous_spikes.detach().sum().item()) * recurrent_fanin[index].item())
                membrane, final_spikes = layer.step(layer_input, membrane, previous_spikes)
                membrane = membrane * active
                final_spikes = final_spikes * active
                states[index] = (membrane, final_spikes)
                layer_input = final_spikes
            if final_spikes is None:
                raise AssertionError("decoder produced no hidden spikes")
            hidden_events += int(final_spikes.detach().sum().item())
            position_scores = self.output_head(final_spikes, active, targets[:, position])
            scores.append(position_scores)
            margin_loss = margin_loss + self.output_head.margin_loss()
            for key, value in self.output_head.last_operation_stats().items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    output_totals[key] = output_totals.get(key, 0.0) + float(value)
            feedback_byte = (
                targets[:, position]
                if feedback_mode == "teacher"
                else position_scores.detach().argmax(dim=1)
            )
            previous_byte = self.byte_codebook[feedback_byte] * active
            if not torch.all((previous_byte.detach() == 0) | (previous_byte.detach() == 1)):
                raise AssertionError("feedback ceased to be binary")

        self._last_margin_loss = margin_loss / self.max_patch_bytes
        output = torch.stack(scores, dim=1).reshape(batch, steps, self.max_patch_bytes, 256)
        active_bytes = int(target_mask.sum().item())
        output_synops = int(output_totals.get("decoder_output_synaptic_accumulations", 0.0))
        output_updates = int(output_totals.get("decoder_output_lif_state_updates", 0.0))
        output_control = int(output_totals.get("decoder_output_threshold_reset_elements", 0.0))
        self._last_stats = {
            "frontend_operation_ledger_version": PATCH_LEDGER_V6_VERSION,
            "decoder_feedback_mode": feedback_mode,
            "decoder_feedback_is_binary_spike_code": 1,
            "decoder_context_spike_events": int((context.detach() * mask.any(dim=1, keepdim=True)).sum().item()),
            "decoder_teacher_byte_spike_events": (max(active_bytes - rows, 0) * 2 if feedback_mode == "teacher" else 0),
            "decoder_predicted_byte_feedback_events": (max(active_bytes - rows, 0) * 2 if feedback_mode == "free_running" else 0),
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
            "decoder_binary_internal_communication": 1,
            "decoder_dense_likelihood_readout_present": 0,
            **{key: int(value) if float(value).is_integer() else float(value) for key, value in output_totals.items()},
        }
        return output

    def margin_loss(self) -> torch.Tensor:
        if self._last_margin_loss is None:
            raise RuntimeError("margin loss requested before forward")
        return self._last_margin_loss

    def last_operation_stats(self) -> Dict[str, float | int | str]:
        return dict(self._last_stats)


class StrictTTFSPatchLanguageModel(nn.Module):
    """Unchanged patch encoder/backbone with dense, direct TTFS or nibble TTFS output."""

    def __init__(self, *, max_patch_bytes: int, hierarchical_encoder: bool, output_head: str,
                 hidden_dim: int = 64, num_layers: int = 2, decoder_hidden_dim: int = 128,
                 decoder_num_layers: int = 2, output_microsteps: int = 16,
                 output_beta: float = 0.90, output_threshold: float = 1.0,
                 output_weight_scale: float = 0.04, output_clock_end: float = 0.20,
                 output_latency_tau: float = 6.0, output_score_scale: float = 8.0,
                 output_surrogate_temperature: float = 0.20, seed: int = 1) -> None:
        super().__init__()
        if output_head not in OUTPUT_HEADS:
            raise ValueError(f"unsupported output head: {output_head}")
        self.output_head_name = output_head
        self.encoder = PositionalPatchSpikeEncoder(max_patch_bytes=max_patch_bytes, hierarchical=hierarchical_encoder, seed=seed)
        self.backbone = SpikeGPTReferenceLM(vocab_size=2, input_dim=self.encoder.output_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=0.03)
        self.backbone.final_ln = nn.Identity()
        self.backbone.readout = nn.Identity()
        common = dict(context_dim=hidden_dim, max_patch_bytes=max_patch_bytes,
                      decoder_hidden_dim=decoder_hidden_dim, decoder_num_layers=decoder_num_layers, seed=seed)
        if output_head == "dense_reference":
            self.decoder: nn.Module = LocalByteLIFDecoder(**common)
        else:
            self.decoder = LocalByteLIFDecoderStrictTTFS(
                **common, output_head=output_head, output_microsteps=output_microsteps,
                output_beta=output_beta, output_threshold=output_threshold,
                output_weight_scale=output_weight_scale, output_clock_end=output_clock_end,
                output_latency_tau=output_latency_tau, output_score_scale=output_score_scale,
                output_surrogate_temperature=output_surrogate_temperature,
            )

    def forward(self, source_bytes: torch.Tensor, source_mask: torch.Tensor,
                target_bytes: torch.Tensor, target_mask: torch.Tensor, *,
                feedback_mode: str = "teacher") -> Tuple[torch.Tensor, Dict[str, object]]:
        input_spikes = self.encoder(source_bytes, source_mask)
        _, context_spikes, backbone_stats = self.backbone.forward_features(inputs_spikes=input_spikes)
        if self.output_head_name == "dense_reference":
            if feedback_mode != "teacher":
                raise ValueError("dense reference free-running mode is not implemented in v6")
            scores = self.decoder(context_spikes, target_bytes, target_mask)
        else:
            scores = self.decoder(context_spikes, target_bytes, target_mask, feedback_mode=feedback_mode)
        strict = int(self.output_head_name != "dense_reference")
        stats: Dict[str, object] = {
            "frontend_operation_ledger_version": PATCH_LEDGER_V6_VERSION,
            "global_patch_steps": int(source_bytes.shape[0] * source_bytes.shape[1]),
            "global_input_dimension": self.encoder.output_dim,
            "global_input_spike_events": int(input_spikes.detach().sum().item()),
            "fully_spike_communicating_patch_encoder": 1,
            "fully_spike_communicating_global_to_local_decoder": 1,
            "digital_tokenizer_present": 0,
            "digital_patch_codec_present": 0,
            "spiking_byte_output_neurons_present": strict,
            "dense_final_byte_likelihood_readout_present": 1 - strict,
            "fully_spiking_decoder_neural_path": strict,
            "surrogate_used_only_for_backward_credit_assignment": strict,
            **{key: value for key, value in self.encoder.last_operation_stats().items() if key != "frontend_operation_ledger_version"},
            **{key: value for key, value in self.decoder.last_operation_stats().items() if key != "frontend_operation_ledger_version"},
            **{f"backbone_{key}": value for key, value in backbone_stats.items()},
        }
        return scores, stats

    def output_auxiliary_loss(self, *, margin_weight: float) -> torch.Tensor:
        if self.output_head_name == "dense_reference":
            parameter = next(self.parameters())
            return torch.zeros((), device=parameter.device)
        return float(margin_weight) * self.decoder.margin_loss()


__all__ = ["OUTPUT_HEADS", "PATCH_ARCHITECTURES", "PATCH_LEDGER_V6_VERSION",
           "DirectFirstSpikeByteHead", "LocalByteLIFDecoderStrictTTFS",
           "StrictTTFSPatchLanguageModel"]
