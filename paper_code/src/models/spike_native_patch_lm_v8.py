"""Continuous-time race-to-first-spike output heads with event WTA latching.

The v8 strict output path keeps all inter-module communication binary and
spike-based.  Each output population integrates binary decoder spikes locally,
estimates the first threshold-crossing time within a microstep, emits only the
unique earliest class spike, and emits one inhibitory latch spike that closes
the race.  Counterfactual candidate crossing times are retained locally for
surrogate-gradient likelihood training and external BPB evaluation; they are
not routed as deployed output events.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from src.models.spike_native_patch_lm_v2 import LocalByteLIFDecoderV2, _LIFDecoderLayer
from src.models.spike_native_patch_lm_v6 import OUTPUT_HEADS, PATCH_ARCHITECTURES
from src.models.spike_native_patch_lm_v7 import SpikeNativePatchLanguageModelV7
from src.models.spikegpt_reference import atan_spike
from src.tokenizers.spiking_blt_frontend import injective_byte_spike_codebook

PATCH_LEDGER_V8_VERSION = "codec_free_patch_v8_continuous_ttfs_wta"


class ContinuousRacePopulation(nn.Module):
    """One-spike-per-decision TTFS population with an inhibitory event latch."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        classes: int,
        microsteps: int = 24,
        beta: float = 0.90,
        threshold: float = 1.0,
        weight_scale: float = 0.04,
        deadline_drive_end: float = 0.06,
        deadline_start_fraction: float = 0.65,
        latency_tau: float = 6.0,
        score_scale: float = 8.0,
        surrogate_temperature: float = 0.20,
        tie_tolerance: float = 1e-6,
        seed: int = 1,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or classes <= 1 or microsteps < 4:
            raise ValueError("invalid continuous TTFS population dimensions")
        if not 0.0 < beta < 1.0:
            raise ValueError("beta must lie in (0, 1)")
        if threshold <= 0.0 or weight_scale <= 0.0:
            raise ValueError("threshold and weight scale must be positive")
        if deadline_drive_end < 0.0:
            raise ValueError("deadline drive must be non-negative")
        if not 0.0 <= deadline_start_fraction < 1.0:
            raise ValueError("deadline start fraction must lie in [0, 1)")
        if latency_tau <= 0.0 or score_scale <= 0.0:
            raise ValueError("latency score parameters must be positive")
        if surrogate_temperature <= 0.0 or tie_tolerance <= 0.0:
            raise ValueError("surrogate temperature and tie tolerance must be positive")

        self.hidden_dim = int(hidden_dim)
        self.classes = int(classes)
        self.microsteps = int(microsteps)
        self.beta = float(beta)
        self.threshold = float(threshold)
        self.weight_scale = float(weight_scale)
        self.deadline_drive_end = float(deadline_drive_end)
        self.deadline_start_fraction = float(deadline_start_fraction)
        self.latency_tau = float(latency_tau)
        self.score_scale = float(score_scale)
        self.surrogate_temperature = float(surrogate_temperature)
        self.tie_tolerance = float(tie_tolerance)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        self.input_weight = nn.Parameter(
            self.weight_scale
            * torch.randn(self.classes, self.hidden_dim, generator=generator)
        )
        self.register_buffer(
            "threshold_vector",
            torch.full((self.classes,), self.threshold),
        )
        self._last_stats: Dict[str, float | int | str] = {}
        self._last_winner: torch.Tensor | None = None
        self._last_valid_unique: torch.Tensor | None = None

    def _deadline_drive(self, step: int) -> float:
        phase = float(step + 1) / float(self.microsteps)
        if phase <= self.deadline_start_fraction:
            return 0.0
        progress = (
            (phase - self.deadline_start_fraction)
            / (1.0 - self.deadline_start_fraction)
        )
        return self.deadline_drive_end * progress * progress

    def forward(
        self,
        hidden_spikes: torch.Tensor,
        active: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if hidden_spikes.ndim != 2 or hidden_spikes.shape[-1] != self.hidden_dim:
            raise ValueError("hidden_spikes must be [rows, hidden_dim]")
        if active.shape != (hidden_spikes.shape[0], 1):
            raise ValueError("active must be [rows, 1]")
        if not torch.all((hidden_spikes.detach() == 0) | (hidden_spikes.detach() == 1)):
            raise ValueError("continuous TTFS population requires binary input spikes")

        rows = hidden_spikes.shape[0]
        device = hidden_spikes.device
        dtype = hidden_spikes.dtype
        active_rows = active[:, 0] > 0
        threshold = self.threshold_vector.view(1, self.classes)
        current = F.linear(hidden_spikes * active, self.input_weight)

        membrane = torch.zeros(rows, self.classes, device=device, dtype=dtype)
        hard_available = torch.ones_like(membrane)
        hard_time = torch.full(
            (rows, self.classes),
            float(self.microsteps + 1),
            device=device,
            dtype=dtype,
        )
        soft_survival = torch.ones_like(membrane)
        soft_expected_time = torch.zeros_like(membrane)

        for step in range(self.microsteps):
            previous = membrane
            candidate = (
                self.beta * membrane
                + current
                + self._deadline_drive(step)
            ) * active
            raw_crossing = atan_spike(candidate - threshold, alpha=2.0)
            newly = (
                (raw_crossing.detach() > 0)
                & (hard_available.detach() > 0)
                & active_rows.unsqueeze(1)
            )
            denominator = candidate - previous
            fraction = (
                (threshold - previous)
                / torch.where(
                    denominator.abs() > 1e-7,
                    denominator,
                    torch.full_like(denominator, 1e-7),
                )
            ).clamp(0.0, 1.0)
            crossing_time = float(step) + fraction
            hard_time = torch.where(newly, crossing_time.detach(), hard_time)
            hard_available = hard_available * (~newly).to(dtype)
            membrane = candidate * hard_available

            conditional_hazard = torch.sigmoid(
                (candidate - threshold) / self.surrogate_temperature
            ) * active
            event_mass = soft_survival * conditional_hazard
            soft_expected_time = soft_expected_time + event_mass * float(step + 1)
            soft_survival = soft_survival * (1.0 - conditional_hazard)

        horizon = float(self.microsteps + 1)
        soft_expected_time = soft_expected_time + soft_survival * horizon
        hard_score = -hard_time / self.latency_tau * self.score_scale
        soft_score = -soft_expected_time / self.latency_tau * self.score_scale
        scores = hard_score.detach() + soft_score - soft_score.detach()

        minimum_time = hard_time.min(dim=1).values
        covered = active_rows & (minimum_time <= float(self.microsteps))
        near_minimum = (
            (hard_time - minimum_time.unsqueeze(1)).abs() <= self.tie_tolerance
        )
        earliest_count = near_minimum.sum(dim=1)
        tied = covered & (earliest_count > 1)
        valid_unique = covered & (~tied)
        winner = hard_time.argmin(dim=1)
        candidate_crossings = (
            (hard_time <= float(self.microsteps)) & active_rows.unsqueeze(1)
        ).sum(dim=1)
        emitted_class = covered.to(torch.long)
        latch_spikes = covered.to(torch.long)
        suppressed = torch.clamp(candidate_crossings - emitted_class, min=0)

        input_events = int((hidden_spikes.detach() * active).sum().item())
        active_count = int(active_rows.sum().item())
        candidate_total = int(candidate_crossings.sum().item())
        emitted_total = int(emitted_class.sum().item())
        latch_total = int(latch_spikes.sum().item())
        suppressed_total = int(suppressed.sum().item())
        interpolation_elements = candidate_total * 4
        inhibitory_synops = latch_total * self.classes

        self._last_winner = winner.detach()
        self._last_valid_unique = valid_unique.detach()
        self._last_stats = {
            "population_active_rows": active_count,
            "population_candidate_crossings": candidate_total,
            "population_emitted_class_spikes": emitted_total,
            "population_inhibitory_latch_spikes": latch_total,
            "population_suppressed_crossings": suppressed_total,
            "population_no_decision_rows": int((active_rows & (~covered)).sum().item()),
            "population_continuous_tied_rows": int(tied.sum().item()),
            "population_valid_unique_rows": int(valid_unique.sum().item()),
            "population_first_latency_sum": float(minimum_time[covered].sum().item()),
            "population_synaptic_accumulations": input_events * self.classes + inhibitory_synops,
            "population_lif_state_updates": active_count * self.classes * self.microsteps,
            "population_threshold_reset_elements": (
                active_count * self.classes * self.microsteps * 2
                + interpolation_elements
                + inhibitory_synops
            ),
            "population_crossing_interpolation_elements": interpolation_elements,
            "population_inhibitory_fanout_synops": inhibitory_synops,
        }
        detail = {
            "hard_time": hard_time.detach(),
            "winner": winner.detach(),
            "covered": covered.detach(),
            "tied": tied.detach(),
            "valid_unique": valid_unique.detach(),
            "candidate_crossings": candidate_crossings.detach(),
        }
        return scores, detail

    def last_operation_stats(self) -> Dict[str, float | int | str]:
        return dict(self._last_stats)

    def last_decision(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._last_winner is None or self._last_valid_unique is None:
            raise RuntimeError("decision requested before forward")
        return self._last_winner, self._last_valid_unique


class DirectRaceToFirstSpikeByteHead(nn.Module):
    """One output neuron per byte, with one winning spike and one latch spike."""

    def __init__(self, *, hidden_dim: int, seed: int = 1, **kwargs: object) -> None:
        super().__init__()
        self.population = ContinuousRacePopulation(
            hidden_dim=hidden_dim,
            classes=256,
            seed=int(seed) + 32452843,
            **kwargs,
        )
        self._last_stats: Dict[str, float | int | str] = {}
        self._last_margin_loss: torch.Tensor | None = None

    def forward(
        self,
        hidden_spikes: torch.Tensor,
        active: torch.Tensor,
        target_bytes: torch.Tensor,
    ) -> torch.Tensor:
        scores, detail = self.population(hidden_spikes, active)
        active_rows = active[:, 0] > 0
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

        stats = self.population.last_operation_stats()
        winner = detail["winner"]
        valid_unique = detail["valid_unique"]
        active_count = int(stats["population_active_rows"])
        self._last_stats = {
            "decoder_output_spike_events": int(stats["population_emitted_class_spikes"]),
            "decoder_output_early_exit_spike_events": int(stats["population_emitted_class_spikes"]) + int(stats["population_inhibitory_latch_spikes"]),
            "decoder_output_emitted_class_spikes": int(stats["population_emitted_class_spikes"]),
            "decoder_output_inhibitory_latch_spikes": int(stats["population_inhibitory_latch_spikes"]),
            "decoder_output_candidate_crossings": int(stats["population_candidate_crossings"]),
            "decoder_output_suppressed_crossings": int(stats["population_suppressed_crossings"]),
            "decoder_output_crossing_interpolation_elements": int(stats["population_crossing_interpolation_elements"]),
            "decoder_output_inhibitory_fanout_synops": int(stats["population_inhibitory_fanout_synops"]),
            "decoder_output_synaptic_accumulations": int(stats["population_synaptic_accumulations"]),
            "decoder_output_lif_state_updates": int(stats["population_lif_state_updates"]),
            "decoder_output_threshold_reset_elements": int(stats["population_threshold_reset_elements"]),
            "decoder_output_no_spike_bytes": int(stats["population_no_decision_rows"]),
            "decoder_output_tied_winner_bytes": int(stats["population_continuous_tied_rows"]),
            "decoder_output_unique_winner_bytes": int(stats["population_valid_unique_rows"]),
            "decoder_output_first_spike_covered_bytes": active_count - int(stats["population_no_decision_rows"]),
            "decoder_output_first_spike_tied_bytes": int(stats["population_continuous_tied_rows"]),
            "decoder_output_first_spike_latency_sum": float(stats["population_first_latency_sum"]),
            "decoder_output_first_spike_unique_correct_bytes": int((valid_unique & (winner == target_bytes)).sum().item()),
            "decoder_output_temporal_score_correct_bytes": int((active_rows & (scores.detach().argmax(dim=1) == target_bytes)).sum().item()),
            "decoder_output_active_bytes": active_count,
            "decoder_output_valid_unique_winner_bytes": int(stats["population_valid_unique_rows"]),
        }
        return scores

    def margin_loss(self) -> torch.Tensor:
        if self._last_margin_loss is None:
            raise RuntimeError("margin loss requested before forward")
        return self._last_margin_loss

    def last_operation_stats(self) -> Dict[str, float | int | str]:
        return dict(self._last_stats)

    def last_decision(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.population.last_decision()


class NibbleRaceToFirstSpikeByteHead(nn.Module):
    """Two independently latched 16-way races forming an exact byte code."""

    def __init__(self, *, hidden_dim: int, seed: int = 1, **kwargs: object) -> None:
        super().__init__()
        self.high = ContinuousRacePopulation(
            hidden_dim=hidden_dim,
            classes=16,
            seed=int(seed) + 15485863,
            **kwargs,
        )
        self.low = ContinuousRacePopulation(
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

    def forward(
        self,
        hidden_spikes: torch.Tensor,
        active: torch.Tensor,
        target_bytes: torch.Tensor,
    ) -> torch.Tensor:
        high_score, high = self.high(hidden_spikes, active)
        low_score, low = self.low(hidden_spikes, active)
        byte_score = high_score[:, self.byte_high] + low_score[:, self.byte_low]
        active_rows = active[:, 0] > 0
        target_high = target_bytes // 16
        target_low = target_bytes % 16
        if bool(active_rows.any()):
            ah = high_score[active_rows]
            al = low_score[active_rows]
            th = target_high[active_rows]
            tl = target_low[active_rows]
            ch = ah.gather(1, th.unsqueeze(1)).squeeze(1)
            cl = al.gather(1, tl.unsqueeze(1)).squeeze(1)
            mh = ah.clone()
            ml = al.clone()
            mh.scatter_(1, th.unsqueeze(1), float("-inf"))
            ml.scatter_(1, tl.unsqueeze(1), float("-inf"))
            self._last_margin_loss = (
                F.relu(0.25 - (ch - mh.max(dim=1).values)).mean()
                + F.relu(0.25 - (cl - ml.max(dim=1).values)).mean()
            )
        else:
            self._last_margin_loss = byte_score.sum() * 0.0

        high_stats = self.high.last_operation_stats()
        low_stats = self.low.last_operation_stats()
        winner = high["winner"] * 16 + low["winner"]
        valid_unique = high["valid_unique"] & low["valid_unique"] & active_rows
        covered = high["covered"] & low["covered"] & active_rows
        tied = covered & (~valid_unique)
        active_count = int(active_rows.sum().item())
        self._last_winner = winner.detach()
        self._last_valid_unique = valid_unique.detach()

        def total(key: str) -> int:
            return int(high_stats[key]) + int(low_stats[key])

        emitted = total("population_emitted_class_spikes")
        latch = total("population_inhibitory_latch_spikes")
        self._last_stats = {
            "decoder_output_spike_events": emitted,
            "decoder_output_early_exit_spike_events": emitted + latch,
            "decoder_output_emitted_class_spikes": emitted,
            "decoder_output_inhibitory_latch_spikes": latch,
            "decoder_output_candidate_crossings": total("population_candidate_crossings"),
            "decoder_output_suppressed_crossings": total("population_suppressed_crossings"),
            "decoder_output_crossing_interpolation_elements": total("population_crossing_interpolation_elements"),
            "decoder_output_inhibitory_fanout_synops": total("population_inhibitory_fanout_synops"),
            "decoder_output_synaptic_accumulations": total("population_synaptic_accumulations"),
            "decoder_output_lif_state_updates": total("population_lif_state_updates"),
            "decoder_output_threshold_reset_elements": total("population_threshold_reset_elements"),
            "decoder_output_no_spike_bytes": int((active_rows & (~covered)).sum().item()),
            "decoder_output_tied_winner_bytes": int(tied.sum().item()),
            "decoder_output_unique_winner_bytes": int(valid_unique.sum().item()),
            "decoder_output_first_spike_covered_bytes": int(covered.sum().item()),
            "decoder_output_first_spike_tied_bytes": int(tied.sum().item()),
            "decoder_output_first_spike_latency_sum": float(torch.maximum(high["hard_time"].min(dim=1).values, low["hard_time"].min(dim=1).values)[covered].sum().item()),
            "decoder_output_first_spike_unique_correct_bytes": int((valid_unique & (winner == target_bytes)).sum().item()),
            "decoder_output_temporal_score_correct_bytes": int((active_rows & (byte_score.detach().argmax(dim=1) == target_bytes)).sum().item()),
            "decoder_output_active_bytes": active_count,
            "decoder_output_valid_unique_winner_bytes": int(valid_unique.sum().item()),
        }
        return byte_score

    def margin_loss(self) -> torch.Tensor:
        if self._last_margin_loss is None:
            raise RuntimeError("margin loss requested before forward")
        return self._last_margin_loss

    def last_operation_stats(self) -> Dict[str, float | int | str]:
        return dict(self._last_stats)

    def last_decision(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._last_winner is None or self._last_valid_unique is None:
            raise RuntimeError("decision requested before forward")
        return self._last_winner, self._last_valid_unique


class LocalByteLIFDecoderContinuousTTFS(nn.Module):
    """Sparse recurrent decoder with tie-aware spike-only predicted feedback."""

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
        output_microsteps: int = 24,
        output_beta: float = 0.90,
        output_threshold: float = 1.0,
        output_weight_scale: float = 0.04,
        output_clock_end: float = 0.06,
        output_latency_tau: float = 6.0,
        output_score_scale: float = 8.0,
        output_surrogate_temperature: float = 0.20,
        seed: int = 1,
    ) -> None:
        super().__init__()
        if output_head not in {"ttfs_direct_256", "ttfs_nibble_2x16"}:
            raise ValueError("continuous TTFS decoder requires a strict output head")
        self.context_dim = int(context_dim)
        self.max_patch_bytes = int(max_patch_bytes)
        self.decoder_hidden_dim = int(decoder_hidden_dim)
        self.decoder_num_layers = int(decoder_num_layers)
        self.output_head_name = output_head
        self.output_microsteps = int(output_microsteps)
        self.output_clock_end = float(output_clock_end)
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
            deadline_drive_end=output_clock_end,
            deadline_start_fraction=0.65,
            latency_tau=output_latency_tau,
            score_scale=output_score_scale,
            surrogate_temperature=output_surrogate_temperature,
            seed=seed,
        )
        self.output_head: nn.Module = (
            DirectRaceToFirstSpikeByteHead(**common)
            if output_head == "ttfs_direct_256"
            else NibbleRaceToFirstSpikeByteHead(**common)
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

            if feedback_mode == "teacher":
                previous_byte = self.byte_codebook[targets[:, position]] * active
            else:
                winner, valid_unique = self.output_head.last_decision()
                valid = valid_unique & (active[:, 0] > 0)
                feedback_attempted_bytes += int((active[:, 0] > 0).sum().item())
                feedback_fallback_bytes += int(((active[:, 0] > 0) & (~valid)).sum().item())
                previous_byte = (
                    self.byte_codebook[winner]
                    * valid.to(context.dtype).unsqueeze(1)
                )
            if not torch.all((previous_byte.detach() == 0) | (previous_byte.detach() == 1)):
                raise AssertionError("feedback ceased to be binary")

        self._last_margin_loss = margin_loss / self.max_patch_bytes
        output = torch.stack(scores, dim=1).reshape(batch, steps, self.max_patch_bytes, 256)
        active_bytes = int(target_mask.sum().item())
        output_synops = int(output_totals.get("decoder_output_synaptic_accumulations", 0.0))
        output_updates = int(output_totals.get("decoder_output_lif_state_updates", 0.0))
        output_control = int(output_totals.get("decoder_output_threshold_reset_elements", 0.0))
        self._last_stats = {
            "frontend_operation_ledger_version": PATCH_LEDGER_V8_VERSION,
            "decoder_context_spike_events": int((context.detach() * mask.any(dim=1, keepdim=True)).sum().item()),
            "decoder_teacher_byte_spike_events": max(active_bytes - rows, 0) * 2 if feedback_mode == "teacher" else 0,
            "decoder_predicted_byte_feedback_events": max(active_bytes - rows - feedback_fallback_bytes, 0) * 2 if feedback_mode == "free_running" else 0,
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

    def margin_loss(self) -> torch.Tensor:
        if self._last_margin_loss is None:
            raise RuntimeError("margin loss requested before forward")
        return self._last_margin_loss

    def last_operation_stats(self) -> Dict[str, float | int | str]:
        return dict(self._last_stats)


class SpikeNativePatchLanguageModelV8(SpikeNativePatchLanguageModelV7):
    """v7 patch/backbone path with continuous-time WTA TTFS output heads."""

    def __init__(self, *, output_head: str, **kwargs: object) -> None:
        super().__init__(output_head=output_head, **kwargs)
        self.output_head_name = output_head
        if output_head != "dense_reference":
            self.decoder = LocalByteLIFDecoderContinuousTTFS(
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
        stats["frontend_operation_ledger_version"] = PATCH_LEDGER_V8_VERSION
        return scores, stats


__all__ = [
    "OUTPUT_HEADS",
    "PATCH_ARCHITECTURES",
    "PATCH_LEDGER_V8_VERSION",
    "ContinuousRacePopulation",
    "DirectRaceToFirstSpikeByteHead",
    "NibbleRaceToFirstSpikeByteHead",
    "LocalByteLIFDecoderContinuousTTFS",
    "SpikeNativePatchLanguageModelV8",
]
