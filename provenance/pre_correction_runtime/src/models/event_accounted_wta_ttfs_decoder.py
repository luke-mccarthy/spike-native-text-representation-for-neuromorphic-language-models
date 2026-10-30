"""Event emission and feedback accounting for the continuous WTA workflow.

The continuous race dynamics and latency likelihoods are unchanged.
Only unique earliest winners emit a routed class spike and inhibitory latch.
Tied or silent races emit no class identity and therefore provide silent
free-running feedback.  Feedback events are counted only when they are routed
to an active next byte position.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import nn

from src.models.continuous_wta_ttfs_decoder import (
    OUTPUT_HEADS,
    PATCH_ARCHITECTURES,
    ContinuousRacePopulation,
    DirectRaceToFirstSpikeByteHead,
    LocalByteLIFDecoderContinuousTTFS,
    NibbleRaceToFirstSpikeByteHead,
    ContinuousWTATTFSPatchLanguageModel,
)

PATCH_LEDGER_V9_VERSION = "codec_free_patch_v9_continuous_ttfs_unique_wta"


class EventAccountedRacePopulation(ContinuousRacePopulation):
    """Continuous race with events restricted to valid unique winners."""

    def forward(
        self,
        hidden_spikes: torch.Tensor,
        active: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        scores, detail = super().forward(hidden_spikes, active)
        stats = self._last_stats
        old_emitted = int(stats["population_emitted_class_spikes"])
        old_latch = int(stats["population_inhibitory_latch_spikes"])
        candidate_total = int(stats["population_candidate_crossings"])
        emitted = int(detail["valid_unique"].sum().item())
        latch = emitted
        suppressed = max(candidate_total - emitted, 0)
        old_inhibitory_synops = old_latch * self.classes
        new_inhibitory_synops = latch * self.classes
        stats["population_emitted_class_spikes"] = emitted
        stats["population_inhibitory_latch_spikes"] = latch
        stats["population_suppressed_crossings"] = suppressed
        stats["population_inhibitory_fanout_synops"] = new_inhibitory_synops
        stats["population_synaptic_accumulations"] = (
            int(stats["population_synaptic_accumulations"])
            - old_inhibitory_synops
            + new_inhibitory_synops
        )
        stats["population_threshold_reset_elements"] = (
            int(stats["population_threshold_reset_elements"])
            - old_inhibitory_synops
            + new_inhibitory_synops
        )
        if old_emitted != old_latch:
            raise AssertionError("v8 emitted/latch counts were inconsistent")
        self._last_stats = stats
        return scores, detail


class EventAccountedDirectTTFSHead(DirectRaceToFirstSpikeByteHead):
    def __init__(self, *, hidden_dim: int, seed: int = 1, **kwargs: object) -> None:
        nn.Module.__init__(self)
        self.population = EventAccountedRacePopulation(
            hidden_dim=hidden_dim,
            classes=256,
            seed=int(seed) + 32452843,
            **kwargs,
        )
        self._last_stats: Dict[str, float | int | str] = {}
        self._last_margin_loss: torch.Tensor | None = None


class EventAccountedNibbleTTFSHead(NibbleRaceToFirstSpikeByteHead):
    def __init__(self, *, hidden_dim: int, seed: int = 1, **kwargs: object) -> None:
        nn.Module.__init__(self)
        self.high = EventAccountedRacePopulation(
            hidden_dim=hidden_dim,
            classes=16,
            seed=int(seed) + 15485863,
            **kwargs,
        )
        self.low = EventAccountedRacePopulation(
            hidden_dim=hidden_dim,
            classes=16,
            seed=int(seed) + 32452843,
            **kwargs,
        )
        byte_ids = torch.arange(256, dtype=torch.long)
        self.register_buffer("byte_high", byte_ids // 16)
        self.register_buffer("byte_low", byte_ids % 16)
        self._last_stats: Dict[str, float | int | str] = {}
        self._last_margin_loss: torch.Tensor | None = None
        self._last_winner: torch.Tensor | None = None
        self._last_valid_unique: torch.Tensor | None = None


class EventAccountedTTFSPatchDecoder(LocalByteLIFDecoderContinuousTTFS):
    """v8 decoder with exact next-position feedback event accounting."""

    def __init__(self, *, output_head: str, **kwargs: object) -> None:
        super().__init__(output_head=output_head, **kwargs)
        common = dict(
            hidden_dim=self.decoder_hidden_dim,
            microsteps=int(kwargs.get("output_microsteps", 24)),
            beta=float(kwargs.get("output_beta", 0.90)),
            threshold=float(kwargs.get("output_threshold", 1.0)),
            weight_scale=float(kwargs.get("output_weight_scale", 0.04)),
            deadline_drive_end=float(kwargs.get("output_clock_end", 0.06)),
            deadline_start_fraction=0.65,
            latency_tau=float(kwargs.get("output_latency_tau", 6.0)),
            score_scale=float(kwargs.get("output_score_scale", 8.0)),
            surrogate_temperature=float(kwargs.get("output_surrogate_temperature", 0.20)),
            seed=int(kwargs.get("seed", 1)),
        )
        self.output_head = (
            EventAccountedDirectTTFSHead(**common)
            if output_head == "ttfs_direct_256"
            else EventAccountedNibbleTTFSHead(**common)
        )

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
            raise ValueError("invalid decoder context")
        if target_bytes.shape != target_mask.shape or target_bytes.shape[:2] != context_spikes.shape[:2]:
            raise ValueError("invalid decoder target tensors")
        if not torch.all((context_spikes.detach() == 0) | (context_spikes.detach() == 1)):
            raise ValueError("decoder context must be binary spikes")

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
        teacher_feedback_events = 0
        predicted_feedback_events = 0
        feedback_fallback_bytes = 0
        feedback_attempted_bytes = 0
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
                recurrent_accumulations += int(previous_spikes.detach().sum().item() * recurrent_fanin[index].item())
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

            if position + 1 < self.max_patch_bytes:
                next_active = mask[:, position + 1]
            else:
                next_active = torch.zeros(rows, dtype=torch.bool, device=device)
            if feedback_mode == "teacher":
                previous_byte = (
                    self.byte_codebook[targets[:, position]]
                    * next_active.to(context.dtype).unsqueeze(1)
                )
                teacher_feedback_events += int(previous_byte.detach().sum().item())
            else:
                winner, valid_unique = self.output_head.last_decision()
                feedback_attempted_bytes += int(next_active.sum().item())
                feedback_fallback_bytes += int((next_active & (~valid_unique)).sum().item())
                routed = next_active & valid_unique
                previous_byte = (
                    self.byte_codebook[winner]
                    * routed.to(context.dtype).unsqueeze(1)
                )
                predicted_feedback_events += int(previous_byte.detach().sum().item())
            if not torch.all((previous_byte.detach() == 0) | (previous_byte.detach() == 1)):
                raise AssertionError("feedback ceased to be binary")

        self._last_margin_loss = margin_loss / self.max_patch_bytes
        output = torch.stack(scores, dim=1).reshape(batch, steps, self.max_patch_bytes, 256)
        active_bytes = int(target_mask.sum().item())
        output_synops = int(output_totals.get("decoder_output_synaptic_accumulations", 0.0))
        output_updates = int(output_totals.get("decoder_output_lif_state_updates", 0.0))
        output_control = int(output_totals.get("decoder_output_threshold_reset_elements", 0.0))
        self._last_stats = {
            "frontend_operation_ledger_version": PATCH_LEDGER_V9_VERSION,
            "decoder_context_spike_events": int((context.detach() * mask.any(dim=1, keepdim=True)).sum().item()),
            "decoder_teacher_byte_spike_events": teacher_feedback_events,
            "decoder_predicted_byte_feedback_events": predicted_feedback_events,
            "decoder_feedback_fallback_bytes": feedback_fallback_bytes,
            "decoder_feedback_attempted_bytes": feedback_attempted_bytes,
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
            **{
                key: int(value) if float(value).is_integer() else float(value)
                for key, value in output_totals.items()
            },
        }
        return output


class EventAccountedWTATTFSPatchLanguageModel(ContinuousWTATTFSPatchLanguageModel):
    """v8 latency dynamics with unique-only WTA routing and exact feedback."""

    def __init__(self, *, output_head: str, **kwargs: object) -> None:
        super().__init__(output_head=output_head, **kwargs)
        if output_head != "dense_reference":
            self.decoder = EventAccountedTTFSPatchDecoder(
                context_dim=int(kwargs.get("hidden_dim", 64)),
                max_patch_bytes=int(kwargs["max_patch_bytes"]),
                output_head=output_head,
                decoder_hidden_dim=int(kwargs.get("decoder_hidden_dim", 128)),
                decoder_num_layers=int(kwargs.get("decoder_num_layers", 2)),
                output_microsteps=int(kwargs.get("output_microsteps", 24)),
                output_beta=float(kwargs.get("output_beta", 0.90)),
                output_threshold=float(kwargs.get("output_threshold", 1.0)),
                output_weight_scale=float(kwargs.get("output_weight_scale", 0.04)),
                output_clock_end=float(kwargs.get("output_clock_end", 0.06)),
                output_latency_tau=float(kwargs.get("output_latency_tau", 6.0)),
                output_score_scale=float(kwargs.get("output_score_scale", 8.0)),
                output_surrogate_temperature=float(kwargs.get("output_surrogate_temperature", 0.20)),
                seed=int(kwargs.get("seed", 1)),
            )

    def forward(
        self,
        source_bytes: torch.Tensor,
        source_mask: torch.Tensor,
        target_bytes: torch.Tensor,
        target_mask: torch.Tensor,
        *,
        feedback_mode: str = "teacher",
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        scores, stats = super().forward(
            source_bytes,
            source_mask,
            target_bytes,
            target_mask,
            feedback_mode=feedback_mode,
        )
        stats["frontend_operation_ledger_version"] = PATCH_LEDGER_V9_VERSION
        return scores, stats


__all__ = [
    "OUTPUT_HEADS",
    "PATCH_ARCHITECTURES",
    "PATCH_LEDGER_V9_VERSION",
    "EventAccountedRacePopulation",
    "EventAccountedDirectTTFSHead",
    "EventAccountedNibbleTTFSHead",
    "EventAccountedTTFSPatchDecoder",
    "EventAccountedWTATTFSPatchLanguageModel",
]
