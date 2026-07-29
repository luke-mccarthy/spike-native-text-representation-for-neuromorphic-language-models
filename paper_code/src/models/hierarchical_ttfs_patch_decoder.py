"""Codec-free patch LM with a strict spike-timing byte output.

The v5 output replaces the failed 256-way spike-count population with two
16-neuron first-spike populations.  One population represents the high nibble
and one the low nibble of each byte.  Each output neuron emits at most one
spike, and byte scores are fixed sums of latency-derived nibble scores.  There
is no trainable dense 256-way readout after the output neurons.

The forward values used for likelihoods are derived only from hard output
spikes.  A differentiable first-event hazard supplies a straight-through
surrogate gradient so silent/late neurons still receive useful credit during
training.  A shared rising synchronisation current ensures that evidence is
expressed as relative first-spike latency rather than population spike count.
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


PATCH_LEDGER_V5_VERSION = "codec_free_patch_v5_hierarchical_first_spike_output"
OUTPUT_HEADS = ("dense_reference", "ttfs_nibble_2x16")


class FirstSpikeNibblePopulation(nn.Module):
    """Sixteen deterministic LIF neurons scored by first-spike latency."""

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
            raise ValueError("invalid first-spike population dimensions")
        if not 0.0 < beta < 1.0:
            raise ValueError("beta must lie in (0, 1)")
        if threshold <= 0.0 or weight_scale <= 0.0 or clock_end <= 0.0:
            raise ValueError("invalid first-spike dynamics")
        if latency_tau <= 0.0 or score_scale <= 0.0:
            raise ValueError("invalid latency score parameters")
        if surrogate_temperature <= 0.0:
            raise ValueError("surrogate temperature must be positive")

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
        generator.manual_seed(int(seed) + 15485863)
        weight = self.weight_scale * torch.randn(
            16,
            hidden_dim,
            generator=generator,
        )
        self.input_weight = nn.Parameter(weight)
        self.register_buffer(
            "threshold_vector",
            torch.full((16,), self.threshold),
        )
        self._last_stats: Dict[str, float | int | str] = {}

    def forward(
        self,
        hidden_spikes: torch.Tensor,
        active: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if hidden_spikes.ndim != 2 or hidden_spikes.shape[-1] != self.hidden_dim:
            raise ValueError("hidden_spikes must be [rows, hidden_dim]")
        if active.shape != (hidden_spikes.shape[0], 1):
            raise ValueError("active mask must be [rows, 1]")
        if not torch.all((hidden_spikes.detach() == 0) | (hidden_spikes.detach() == 1)):
            raise ValueError("first-spike population received non-binary input")

        rows = hidden_spikes.shape[0]
        device = hidden_spikes.device
        active_rows = active[:, 0] > 0
        synaptic_current = F.linear(hidden_spikes * active, self.input_weight)
        membrane = torch.zeros(rows, 16, device=device)
        hard_available = torch.ones_like(membrane)
        soft_survival = torch.ones_like(membrane)
        hard_score = torch.zeros_like(membrane)
        soft_score = torch.zeros_like(membrane)
        spike_count = torch.zeros_like(membrane)
        first_time = torch.zeros(rows, 16, dtype=torch.long, device=device)

        threshold = self.threshold_vector.view(1, 16)
        for step in range(self.microsteps):
            phase = float(step + 1) / float(self.microsteps)
            synchronisation_current = self.clock_end * phase
            candidate = (
                self.beta * membrane
                + synaptic_current
                + synchronisation_current
            )
            raw_spike = atan_spike(candidate - threshold, alpha=2.0)
            hard_spike = raw_spike * hard_available * active
            newly_fired = (hard_spike.detach() > 0) & (first_time == 0)
            first_time[newly_fired] = step + 1
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

        # Forward values are purely hard-spike latency scores.  Backward values
        # use the differentiable first-event hazard to avoid silent gradients.
        score = hard_score.detach() + soft_score - soft_score.detach()
        detached_score = hard_score.detach()
        detached_count = spike_count.detach()
        no_spike = active_rows.unsqueeze(1) & (detached_count == 0)
        earliest = torch.where(
            first_time > 0,
            first_time,
            torch.full_like(first_time, self.microsteps + 1),
        )
        minimum_time = earliest.min(dim=1).values
        first_ties = active_rows & (
            (earliest == minimum_time.unsqueeze(1)).sum(dim=1) > 1
        )
        first_choice = earliest.argmin(dim=1)
        score_max = detached_score.max(dim=1).values
        score_ties = active_rows & (
            torch.isclose(
                detached_score,
                score_max.unsqueeze(1),
                atol=1e-7,
                rtol=0.0,
            ).sum(dim=1) > 1
        )
        covered = active_rows & (minimum_time <= self.microsteps)
        early_spikes = torch.zeros(rows, device=device)
        if bool(covered.any()):
            early_spikes[covered] = (
                earliest[covered] == minimum_time[covered].unsqueeze(1)
            ).sum(dim=1).to(hidden_spikes.dtype)

        input_events = int((hidden_spikes.detach() * active).sum().item())
        active_bytes = int(active_rows.sum().item())
        self._last_stats = {
            "population_output_microsteps": self.microsteps,
            "population_output_beta": self.beta,
            "population_output_threshold": self.threshold,
            "population_output_weight_scale_initial": self.weight_scale,
            "population_output_clock_end": self.clock_end,
            "population_output_latency_tau": self.latency_tau,
            "population_output_score_scale": self.score_scale,
            "population_output_spike_events": int(detached_count.sum().item()),
            "population_output_early_exit_spike_events": int(early_spikes.sum().item()),
            "population_output_synaptic_accumulations": input_events * 16,
            "population_output_lif_state_updates": active_bytes * 16 * self.microsteps,
            "population_output_threshold_reset_elements": active_bytes * 16 * self.microsteps * 2,
            "population_output_no_spike_neurons": int(no_spike.sum().item()),
            "population_output_covered_rows": int(covered.sum().item()),
            "population_output_first_tied_rows": int((first_ties & active_rows).sum().item()),
            "population_output_score_tied_rows": int((score_ties & active_rows).sum().item()),
            "population_output_first_latency_sum": int(minimum_time[covered].sum().item()),
            "population_output_active_rows": active_bytes,
        }
        detail = {
            "hard_score": detached_score,
            "first_time": earliest,
            "first_choice": first_choice,
            "first_tied": first_ties,
            "covered": covered,
            "spike_count": detached_count,
        }
        return score * self.score_scale, detail

    def last_operation_stats(self) -> Dict[str, float | int | str]:
        return dict(self._last_stats)


class HierarchicalFirstSpikeByteHead(nn.Module):
    """Two 16-neuron TTFS populations forming an exact 256-byte code."""

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
        common = dict(
            hidden_dim=hidden_dim,
            microsteps=microsteps,
            beta=beta,
            threshold=threshold,
            weight_scale=weight_scale,
            clock_end=clock_end,
            latency_tau=latency_tau,
            score_scale=score_scale,
            surrogate_temperature=surrogate_temperature,
        )
        self.high_population = FirstSpikeNibblePopulation(
            **common,
            seed=seed + 1009,
        )
        self.low_population = FirstSpikeNibblePopulation(
            **common,
            seed=seed + 2017,
        )
        byte_ids = torch.arange(256, dtype=torch.long)
        self.register_buffer("byte_high_nibble", byte_ids // 16)
        self.register_buffer("byte_low_nibble", byte_ids % 16)
        self.microsteps = int(microsteps)
        self._last_stats: Dict[str, float | int | str] = {}
        self._last_margin_loss: torch.Tensor | None = None

    def forward(
        self,
        hidden_spikes: torch.Tensor,
        active: torch.Tensor,
        target_bytes: torch.Tensor,
    ) -> torch.Tensor:
        high_score, high = self.high_population(hidden_spikes, active)
        low_score, low = self.low_population(hidden_spikes, active)
        byte_score = (
            high_score[:, self.byte_high_nibble]
            + low_score[:, self.byte_low_nibble]
        )

        active_rows = active[:, 0] > 0
        target_high = target_bytes // 16
        target_low = target_bytes % 16
        if bool(active_rows.any()):
            active_high = high_score[active_rows]
            active_low = low_score[active_rows]
            active_target_high = target_high[active_rows]
            active_target_low = target_low[active_rows]
            high_correct = active_high.gather(1, active_target_high.unsqueeze(1)).squeeze(1)
            low_correct = active_low.gather(1, active_target_low.unsqueeze(1)).squeeze(1)
            high_masked = active_high.clone()
            low_masked = active_low.clone()
            high_masked.scatter_(1, active_target_high.unsqueeze(1), float("-inf"))
            low_masked.scatter_(1, active_target_low.unsqueeze(1), float("-inf"))
            high_margin = high_correct - high_masked.max(dim=1).values
            low_margin = low_correct - low_masked.max(dim=1).values
            self._last_margin_loss = (
                F.relu(0.25 - high_margin).mean()
                + F.relu(0.25 - low_margin).mean()
            )
        else:
            self._last_margin_loss = byte_score.sum() * 0.0

        detached_byte = byte_score.detach()
        maxima = detached_byte.max(dim=1).values
        byte_ties = active_rows & (
            torch.isclose(
                detached_byte,
                maxima.unsqueeze(1),
                atol=1e-7,
                rtol=0.0,
            ).sum(dim=1) > 1
        )
        byte_choice = detached_byte.argmax(dim=1)
        covered = high["covered"] & low["covered"] & active_rows
        first_tied = high["first_tied"] | low["first_tied"]
        first_choice = high["first_choice"] * 16 + low["first_choice"]
        first_latency = torch.maximum(
            high["first_time"].min(dim=1).values,
            low["first_time"].min(dim=1).values,
        )

        high_stats = self.high_population.last_operation_stats()
        low_stats = self.low_population.last_operation_stats()
        active_bytes = int(active_rows.sum().item())
        self._last_stats = {
            "decoder_output_head": "ttfs_nibble_2x16",
            "decoder_output_microsteps": self.microsteps,
            "decoder_output_neurons": 32,
            "decoder_output_spike_events": int(high_stats["population_output_spike_events"]) + int(low_stats["population_output_spike_events"]),
            "decoder_output_early_exit_spike_events": int(high_stats["population_output_early_exit_spike_events"]) + int(low_stats["population_output_early_exit_spike_events"]),
            "decoder_output_synaptic_accumulations": int(high_stats["population_output_synaptic_accumulations"]) + int(low_stats["population_output_synaptic_accumulations"]),
            "decoder_output_lif_state_updates": int(high_stats["population_output_lif_state_updates"]) + int(low_stats["population_output_lif_state_updates"]),
            "decoder_output_threshold_reset_elements": int(high_stats["population_output_threshold_reset_elements"]) + int(low_stats["population_output_threshold_reset_elements"]),
            "decoder_output_no_spike_bytes": int((active_rows & (~covered)).sum().item()),
            "decoder_output_tied_winner_bytes": int(byte_ties.sum().item()),
            "decoder_output_unique_winner_bytes": int((active_rows & (~byte_ties)).sum().item()),
            "decoder_output_first_spike_covered_bytes": int(covered.sum().item()),
            "decoder_output_first_spike_tied_bytes": int((first_tied & active_rows).sum().item()),
            "decoder_output_first_spike_latency_sum": int(first_latency[covered].sum().item()),
            "decoder_output_first_spike_unique_correct_bytes": int((covered & (~first_tied) & (first_choice == target_bytes)).sum().item()),
            "decoder_output_temporal_score_correct_bytes": int((active_rows & (byte_choice == target_bytes)).sum().item()),
            "decoder_output_active_bytes": active_bytes,
            "decoder_output_high_population_spikes": int(high_stats["population_output_spike_events"]),
            "decoder_output_low_population_spikes": int(low_stats["population_output_spike_events"]),
            "decoder_output_scores_are_first_spike_latency": 1,
            "decoder_output_hierarchical_nibble_code": 1,
            "decoder_output_exact_lossless_byte_code": 1,
            "decoder_output_neural_path_is_fully_spiking": 1,
            "decoder_output_dense_linear_readout_present": 0,
            "decoder_output_fixed_combinatorial_byte_decoder": 1,
            "decoder_output_winner_selection_external_reference_control": 1,
        }
        return byte_score

    def margin_loss(self) -> torch.Tensor:
        if self._last_margin_loss is None:
            raise RuntimeError("margin loss requested before forward")
        return self._last_margin_loss

    def last_operation_stats(self) -> Dict[str, float | int | str]:
        return dict(self._last_stats)


class LocalByteLIFDecoderFirstSpikeOutput(nn.Module):
    """Recurrent local LIF decoder with strict TTFS nibble output."""

    def __init__(
        self,
        *,
        context_dim: int,
        max_patch_bytes: int,
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
        self.output_head = HierarchicalFirstSpikeByteHead(
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
        self._last_stats: Dict[str, float | int | str] = {}
        self._last_margin_loss: torch.Tensor | None = None

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
        margin_loss = torch.zeros((), device=device)
        layer_input_fanout = [layer.input_mask.sum(dim=1) for layer in self.layers]
        layer_recurrent_fanout = [layer.recurrent_mask.sum(dim=0) for layer in self.layers]

        for position in range(self.max_patch_bytes):
            active = flattened_mask[:, position].to(flattened_context.dtype).unsqueeze(1)
            position_spikes = torch.zeros(rows, self.max_patch_bytes, device=device)
            position_spikes[:, position] = active[:, 0]
            layer_input = torch.cat(
                (flattened_context * active, previous_byte, position_spikes),
                dim=-1,
            )
            final_spikes = None
            for layer_index, layer in enumerate(self.layers):
                membrane, previous_spikes = states[layer_index]
                active_inputs = int((layer_input.detach() != 0).sum().item())
                input_accumulations += active_inputs * int(
                    layer_input_fanout[layer_index].float().mean().item()
                )
                recurrent_accumulations += int(
                    (
                        previous_spikes.detach().sum(dim=-1)
                        * layer_recurrent_fanout[layer_index].float().mean()
                    ).sum().item()
                )
                membrane, final_spikes = layer.step(
                    layer_input,
                    membrane,
                    previous_spikes,
                )
                membrane = membrane * active
                final_spikes = final_spikes * active
                states[layer_index] = (membrane, final_spikes)
                layer_input = final_spikes
            if final_spikes is None:
                raise AssertionError("decoder produced no hidden layer")
            hidden_events += int(final_spikes.detach().sum().item())
            scores.append(
                self.output_head(
                    final_spikes,
                    active,
                    flattened_targets[:, position],
                )
            )
            margin_loss = margin_loss + self.output_head.margin_loss()
            for key, value in self.output_head.last_operation_stats().items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    output_totals[key] = output_totals.get(key, 0.0) + float(value)
            previous_byte = self.byte_codebook[flattened_targets[:, position]] * active

        self._last_margin_loss = margin_loss / self.max_patch_bytes
        output = torch.stack(scores, dim=1).reshape(
            batch,
            steps,
            self.max_patch_bytes,
            256,
        )
        active_bytes = int(target_mask.sum().item())
        context_events = int(
            (
                flattened_context.detach()
                * flattened_mask.any(dim=1, keepdim=True)
            ).sum().item()
        )
        teacher_events = max(active_bytes - rows, 0) * 2
        output_synops = int(output_totals.get("decoder_output_synaptic_accumulations", 0.0))
        output_updates = int(output_totals.get("decoder_output_lif_state_updates", 0.0))
        output_control = int(output_totals.get("decoder_output_threshold_reset_elements", 0.0))
        self._last_stats = {
            "frontend_operation_ledger_version": PATCH_LEDGER_V5_VERSION,
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

    def margin_loss(self) -> torch.Tensor:
        if self._last_margin_loss is None:
            raise RuntimeError("margin loss requested before forward")
        return self._last_margin_loss

    def last_operation_stats(self) -> Dict[str, float | int | str]:
        return dict(self._last_stats)


class HierarchicalTTFSPatchLanguageModel(nn.Module):
    """Unchanged patch encoder/backbone with dense or strict TTFS output."""

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
            self.decoder = LocalByteLIFDecoderFirstSpikeOutput(
                **decoder_kwargs,
                output_microsteps=output_microsteps,
                output_beta=output_beta,
                output_threshold=output_threshold,
                output_weight_scale=output_weight_scale,
                output_clock_end=output_clock_end,
                output_latency_tau=output_latency_tau,
                output_score_scale=output_score_scale,
                output_surrogate_temperature=output_surrogate_temperature,
            )

    def forward(
        self,
        source_bytes: torch.Tensor,
        source_mask: torch.Tensor,
        target_bytes: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        input_spikes = self.encoder(source_bytes, source_mask)
        _, context_spikes, backbone_stats = self.backbone.forward_features(
            inputs_spikes=input_spikes
        )
        scores = self.decoder(context_spikes, target_bytes, target_mask)
        spike_head = int(self.output_head_name == "ttfs_nibble_2x16")
        stats: Dict[str, object] = {
            "frontend_operation_ledger_version": PATCH_LEDGER_V5_VERSION,
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
            "output_scores_derived_from_first_spike_latency": spike_head,
            "output_fixed_nibble_code_present": spike_head,
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

    def output_auxiliary_loss(self, *, margin_weight: float) -> torch.Tensor:
        if self.output_head_name == "dense_reference":
            parameter = next(self.parameters())
            return torch.zeros((), device=parameter.device)
        if not isinstance(self.decoder, LocalByteLIFDecoderFirstSpikeOutput):
            raise AssertionError("unexpected first-spike decoder type")
        return float(margin_weight) * self.decoder.margin_loss()


__all__ = [
    "OUTPUT_HEADS",
    "PATCH_ARCHITECTURES",
    "PATCH_LEDGER_V5_VERSION",
    "FirstSpikeNibblePopulation",
    "HierarchicalFirstSpikeByteHead",
    "LocalByteLIFDecoderFirstSpikeOutput",
    "HierarchicalTTFSPatchLanguageModel",
]
