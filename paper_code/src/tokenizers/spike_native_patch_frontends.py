"""Byte-event SNN frontends for lossless patch language modelling.

This module separates three functions:

* segmentation decides where a byte patch ends;
* ``LosslessPatchCodec`` serialises that patch as a prediction target;
* this module converts the *raw bytes represented by each modelling unit* to
  binary spikes for the language-model input.

The encoder receives one 1-of-256 byte event per internal time step.  Sparse
masked synapses drive a recurrent LIF reservoir and a spiking readout.  The
only continuous neural state is membrane potential.  Channel zero is a
patch-completion spike, while the remaining channels are thresholded content
spikes.  No byte histogram, dense token embedding, patch hash, or arbitrary
token-ID codebook is used.

The PyTorch reference uses dense masked matrix multiplications for recurrent
and readout synapses.  Runtime statistics therefore disclose both those dense
reference MACs and the sparse event-accumulation count that a compatible
event-driven implementation could execute.  Neither is a hardware-energy
claim.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from src.models.spikegpt_reference import atan_spike


FRONTEND_LEDGER_VERSION = "spike_native_patch_snn_algorithmic_ledger_v1"
SPIKEPATCH_INTERFACE_MODES = {
    "spikepatch_reservoir_fixed64": "fixed",
    "spikepatch_reservoir_homeostatic64": "homeostatic",
    "spikepatch_reservoir_surrogate64": "surrogate",
}


@dataclass(frozen=True)
class SpikePatchSNNConfig:
    """Topology and dynamics shared by the controlled Stage C candidates."""

    hidden_dim: int = 64
    output_dim: int = 64
    input_fanout: int = 12
    recurrent_fanout: int = 4
    readout_fanout: int = 8
    hidden_beta: float = 0.85
    output_beta: float = 0.90
    boundary_beta: float = 0.90
    hidden_threshold: float = 0.55
    output_threshold: float = 1.50
    boundary_threshold: float = 2.00
    input_scale: float = 1.00
    recurrent_scale: float = 0.20
    readout_scale: float = 0.35
    boundary_scale: float = 0.025
    min_patch_bytes: int = 4
    max_patch_bytes: int = 24
    target_hidden_rate: float = 0.125
    target_content_rate: float = 0.125
    target_boundary_rate_per_byte: float = 0.100
    homeostatic_hidden_lr: float = 0.50
    homeostatic_output_lr: float = 0.50
    homeostatic_boundary_lr: float = 2.00
    surrogate_alpha: float = 2.0
    seed: int = 1

    def validate(self) -> None:
        if self.hidden_dim <= 0 or self.output_dim < 2:
            raise ValueError("hidden_dim must be positive and output_dim must be at least two")
        if not 0 < self.input_fanout <= self.hidden_dim:
            raise ValueError("input_fanout must lie in [1, hidden_dim]")
        if not 0 < self.recurrent_fanout <= self.hidden_dim:
            raise ValueError("recurrent_fanout must lie in [1, hidden_dim]")
        if not 0 < self.readout_fanout <= self.output_dim - 1:
            raise ValueError("readout_fanout must lie in [1, output_dim - 1]")
        if not 0 < self.hidden_beta < 1 or not 0 < self.output_beta < 1:
            raise ValueError("hidden and output beta must lie strictly between zero and one")
        if not 0 < self.boundary_beta < 1:
            raise ValueError("boundary beta must lie strictly between zero and one")
        if self.min_patch_bytes <= 0 or self.max_patch_bytes < self.min_patch_bytes:
            raise ValueError("invalid minimum/maximum patch lengths")
        for name in (
            "hidden_threshold", "output_threshold", "boundary_threshold",
            "input_scale", "recurrent_scale", "readout_scale", "boundary_scale",
            "surrogate_alpha",
        ):
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "target_hidden_rate", "target_content_rate", "target_boundary_rate_per_byte",
        ):
            value = float(getattr(self, name))
            if not 0 < value < 1:
                raise ValueError(f"{name} must lie strictly between zero and one")


def payload_table_from_patch_codec(codec: object) -> Tuple[bytes, ...]:
    """Return the exact raw-byte payload for each reachable codec ID."""

    model_vocab_size = int(getattr(codec, "model_vocab_size"))
    id_to_patch = getattr(codec, "id_to_patch")
    payloads: List[bytes] = []
    for token_id in range(model_vocab_size):
        if token_id < 256:
            payloads.append(bytes((token_id,)))
        else:
            patch = id_to_patch.get(token_id)
            if patch is None:
                raise AssertionError(f"Missing payload for reachable patch ID {token_id}")
            payloads.append(bytes(patch))
    return tuple(payloads)


def _masked_topology(config: SpikePatchSNNConfig) -> Dict[str, torch.Tensor]:
    """Create deterministic sparse masks and signed initial synapses."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(config.seed))
    hidden = config.hidden_dim
    content = config.output_dim - 1

    input_mask = torch.zeros(256, hidden)
    for byte in range(256):
        channels = torch.randperm(hidden, generator=generator)[: config.input_fanout]
        input_mask[byte, channels] = 1.0
    input_weight = input_mask * config.input_scale

    recurrent_mask = torch.zeros(hidden, hidden)
    for source in range(hidden):
        destinations = torch.randperm(hidden, generator=generator)[: config.recurrent_fanout]
        recurrent_mask[destinations, source] = 1.0
    recurrent_sign = torch.where(
        torch.rand(hidden, hidden, generator=generator) < 0.75,
        torch.ones(hidden, hidden),
        -torch.ones(hidden, hidden),
    )
    recurrent_jitter = 0.75 + 0.50 * torch.rand(hidden, hidden, generator=generator)
    recurrent_weight = recurrent_mask * recurrent_sign * recurrent_jitter * config.recurrent_scale

    readout_mask = torch.zeros(content, hidden)
    for source in range(hidden):
        destinations = torch.randperm(content, generator=generator)[: config.readout_fanout]
        readout_mask[destinations, source] = 1.0
    readout_sign = torch.where(
        torch.rand(content, hidden, generator=generator) < 0.80,
        torch.ones(content, hidden),
        -torch.ones(content, hidden),
    )
    readout_jitter = 0.75 + 0.50 * torch.rand(content, hidden, generator=generator)
    readout_weight = readout_mask * readout_sign * readout_jitter * config.readout_scale

    boundary_mask = torch.ones(hidden)
    boundary_jitter = 0.75 + 0.50 * torch.rand(hidden, generator=generator)
    boundary_weight = boundary_mask * boundary_jitter * config.boundary_scale
    return {
        "input_mask": input_mask,
        "input_weight": input_weight,
        "recurrent_mask": recurrent_mask,
        "recurrent_weight": recurrent_weight,
        "readout_mask": readout_mask,
        "readout_weight": readout_weight,
        "boundary_mask": boundary_mask,
        "boundary_weight": boundary_weight,
    }


class SpikeNativePatchEncoder(nn.Module):
    """Recurrent LIF encoder and optional joint spiking segmenter.

    ``learning_rule`` is one of ``fixed``, ``homeostatic``, or ``surrogate``.
    Fixed and homeostatic modes have no gradient-trained parameters.
    Surrogate mode uses the same hard Heaviside forward spikes but permits
    gradients through the ATan surrogate.  Structural masks remain fixed, so
    formerly absent synapses cannot become dense during optimisation.
    """

    def __init__(
        self,
        config: SpikePatchSNNConfig | None = None,
        *,
        learning_rule: str = "fixed",
    ) -> None:
        super().__init__()
        self.config = config or SpikePatchSNNConfig()
        self.config.validate()
        if learning_rule not in {"fixed", "homeostatic", "surrogate"}:
            raise ValueError(f"Unsupported learning rule: {learning_rule}")
        self.learning_rule = learning_rule
        topology = _masked_topology(self.config)
        for name in ("input_mask", "recurrent_mask", "readout_mask", "boundary_mask"):
            self.register_buffer(name, topology[name])
        for name in ("input_weight", "recurrent_weight", "readout_weight", "boundary_weight"):
            value = topology[name]
            if learning_rule == "surrogate":
                setattr(self, name, nn.Parameter(value))
            else:
                self.register_buffer(name, value)

        hidden_threshold = torch.full((self.config.hidden_dim,), self.config.hidden_threshold)
        output_threshold = torch.full((self.config.output_dim - 1,), self.config.output_threshold)
        boundary_threshold = torch.tensor(float(self.config.boundary_threshold))
        if learning_rule == "surrogate":
            self.hidden_threshold = nn.Parameter(hidden_threshold)
            self.output_threshold = nn.Parameter(output_threshold)
            self.boundary_threshold = nn.Parameter(boundary_threshold)
        else:
            self.register_buffer("hidden_threshold", hidden_threshold)
            self.register_buffer("output_threshold", output_threshold)
            self.register_buffer("boundary_threshold", boundary_threshold)

        self.homeostasis_fitted = False
        self.homeostasis_report: Dict[str, object] = {}
        self._last_stats: Dict[str, float | int | str] = {}
        self._last_hidden_counts = torch.zeros(self.config.hidden_dim)
        self._last_content_counts = torch.zeros(self.config.output_dim - 1)

    @property
    def output_dim(self) -> int:
        return self.config.output_dim

    @property
    def is_trainable(self) -> bool:
        return self.learning_rule == "surrogate"

    def _weights(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.input_weight * self.input_mask,
            self.recurrent_weight * self.recurrent_mask,
            self.readout_weight * self.readout_mask,
            self.boundary_weight * self.boundary_mask,
        )

    def _thresholds(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.hidden_threshold.clamp_min(0.05),
            self.output_threshold.clamp_min(0.05),
            self.boundary_threshold.clamp_min(0.05),
        )

    def _spike(self, value: torch.Tensor) -> torch.Tensor:
        return atan_spike(value, alpha=self.config.surrogate_alpha)

    @staticmethod
    def _padded_payloads(
        patches: Sequence[bytes],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        values = [bytes(patch) for patch in patches]
        max_length = max((len(patch) for patch in values), default=0)
        padded = torch.zeros(len(values), max(max_length, 1), dtype=torch.long, device=device)
        lengths = torch.tensor([len(patch) for patch in values], dtype=torch.long, device=device)
        for row, patch in enumerate(values):
            if patch:
                padded[row, : len(patch)] = torch.tensor(
                    list(patch), dtype=torch.long, device=device
                )
        return padded, lengths

    def encode_patches(
        self,
        patches: Sequence[bytes],
    ) -> Tuple[torch.Tensor, Dict[str, float | int | str]]:
        """Encode reset-delimited byte payloads to one binary row per payload."""

        device = self.hidden_threshold.device
        padded, lengths = self._padded_payloads(patches, device)
        batch = int(lengths.numel())
        hidden = self.config.hidden_dim
        content = self.config.output_dim - 1
        if batch == 0:
            empty = torch.zeros(0, self.config.output_dim, device=device)
            self._last_stats = self._stats_template(0, 0, 0, 0, 0, 0)
            return empty, dict(self._last_stats)

        input_weight, recurrent_weight, readout_weight, _ = self._weights()
        hidden_threshold, output_threshold, _ = self._thresholds()
        hidden_membrane = torch.zeros(batch, hidden, device=device)
        hidden_spikes = torch.zeros(batch, hidden, device=device)
        output_membrane = torch.zeros(batch, content, device=device)
        hidden_counts = torch.zeros(hidden, device=device)
        recurrent_accumulations = hidden_membrane.new_zeros(())
        readout_accumulations = hidden_membrane.new_zeros(())
        input_accumulations = hidden_membrane.new_zeros(())
        recurrent_fanout = self.recurrent_mask.sum(dim=0)
        readout_fanout = self.readout_mask.sum(dim=0)
        max_length = int(lengths.max().item()) if batch else 0

        for step in range(max_length):
            active = (lengths > step).to(hidden_membrane.dtype).unsqueeze(1)
            byte_ids = padded[:, step]
            input_current = input_weight[byte_ids]
            recurrent_current = F.linear(hidden_spikes, recurrent_weight)
            candidate = (
                self.config.hidden_beta * hidden_membrane
                + input_current
                + recurrent_current
            )
            new_spikes = self._spike(candidate - hidden_threshold) * active
            candidate = candidate * (1.0 - new_spikes)
            hidden_membrane = active * candidate + (1.0 - active) * hidden_membrane
            previous_spikes = hidden_spikes
            hidden_spikes = new_spikes

            output_candidate = (
                self.config.output_beta * output_membrane
                + F.linear(hidden_spikes, readout_weight)
            )
            output_membrane = active * output_candidate + (1.0 - active) * output_membrane
            hidden_counts = hidden_counts + hidden_spikes.sum(dim=0)
            input_accumulations = input_accumulations + (
                self.input_mask[byte_ids].sum(dim=1, keepdim=True) * active
            ).sum()
            recurrent_accumulations = recurrent_accumulations + (
                previous_spikes * recurrent_fanout
            ).sum()
            readout_accumulations = readout_accumulations + (
                hidden_spikes * readout_fanout
            ).sum()

        content_spikes = self._spike(output_membrane - output_threshold)
        completion_spikes = torch.ones(batch, 1, device=device)
        spikes = torch.cat((completion_spikes, content_spikes), dim=1)
        if not torch.all((spikes.detach() == 0) | (spikes.detach() == 1)):
            raise AssertionError("Spike-native patch encoder emitted a non-binary value")

        raw_bytes = int(lengths.sum().item())
        content_events = int(content_spikes.detach().sum().item())
        hidden_events = int(hidden_counts.detach().sum().item())
        stats = self._stats_template(
            raw_bytes,
            batch,
            hidden_events,
            content_events,
            int(input_accumulations.detach().item()),
            int(recurrent_accumulations.detach().item()),
        )
        stats["readout_synaptic_accumulations_event_driven"] = int(
            readout_accumulations.detach().item()
        )
        stats["padded_reference_rows"] = batch * max_length
        stats["padded_reference_extra_rows"] = batch * max_length - raw_bytes
        stats["dense_reference_recurrent_macs_actual"] = batch * max_length * hidden * hidden
        stats["dense_reference_readout_macs_actual"] = batch * max_length * hidden * content
        stats["content_spike_rate"] = content_events / max(batch * content, 1)
        stats["hidden_spike_rate"] = hidden_events / max(raw_bytes * hidden, 1)
        stats["silent_content_rows"] = int((content_spikes.detach().sum(dim=1) == 0).sum().item())
        self._last_stats = stats
        self._last_hidden_counts = hidden_counts.detach().cpu()
        self._last_content_counts = content_spikes.detach().sum(dim=0).cpu()
        return spikes, dict(stats)

    def _stats_template(
        self,
        raw_bytes: int,
        patches: int,
        hidden_events: int,
        content_events: int,
        input_accumulations: int,
        recurrent_accumulations: int,
    ) -> Dict[str, float | int | str]:
        hidden = self.config.hidden_dim
        content = self.config.output_dim - 1
        return {
            "frontend_operation_ledger_version": FRONTEND_LEDGER_VERSION,
            "raw_byte_input_spike_events": raw_bytes,
            "patch_completion_spike_events": patches,
            "hidden_spike_events": hidden_events,
            "content_spike_events": content_events,
            "input_synaptic_accumulations_event_driven": input_accumulations,
            "recurrent_synaptic_accumulations_event_driven": recurrent_accumulations,
            "readout_synaptic_accumulations_event_driven": 0,
            "hidden_lif_state_updates": raw_bytes * hidden,
            "hidden_threshold_evaluations": raw_bytes * hidden,
            "hidden_reset_gate_elements": raw_bytes * hidden,
            "output_membrane_state_updates": raw_bytes * content,
            "output_threshold_evaluations": patches * content,
            "output_reset_gate_elements": patches * content,
            "dense_reference_recurrent_macs_actual": raw_bytes * hidden * hidden,
            "dense_reference_readout_macs_actual": raw_bytes * hidden * content,
            "dense_reference_uses_masked_synapses": 1,
            "event_driven_accumulations_are_reference_execution": 0,
            "input_is_one_hot_raw_byte_event": 1,
            "internal_communication_is_binary_spikes": 1,
            "continuous_neural_state_is_membrane_only": 1,
            "patch_target_serialisation_is_digital": 1,
            "dense_lm_readout_is_outside_frontend": 1,
            "frontend_synaptic_lif_and_threshold_count_complete": 1,
            "frontend_algorithmic_count_complete": 0,
            "frontend_memory_access_count_complete": 0,
            "frontend_hardware_energy_estimate_ready": 0,
            "frontend_hardware_energy_claim_permitted": 0,
        }

    @torch.no_grad()
    def segment_and_encode(
        self,
        data: bytes | bytearray | Sequence[int],
    ) -> Tuple[List[bytes], torch.Tensor, List[str], Dict[str, float | int | str]]:
        """Jointly emit neural/forced patch boundaries and patch content spikes."""

        values = [int(value) for value in data]
        if any(value < 0 or value > 255 for value in values):
            raise ValueError("Raw byte events must lie in [0, 255]")
        if not values:
            raise ValueError("Cannot segment an empty byte stream")
        device = self.hidden_threshold.device
        input_weight, recurrent_weight, readout_weight, boundary_weight = self._weights()
        hidden_threshold, output_threshold, boundary_threshold = self._thresholds()
        hidden = self.config.hidden_dim
        content = self.config.output_dim - 1
        hidden_membrane = torch.zeros(hidden, device=device)
        hidden_spikes = torch.zeros(hidden, device=device)
        output_membrane = torch.zeros(content, device=device)
        boundary_membrane = torch.zeros((), device=device)
        recurrent_fanout = self.recurrent_mask.sum(dim=0)
        readout_fanout = self.readout_mask.sum(dim=0)
        patches: List[bytes] = []
        codes: List[torch.Tensor] = []
        boundary_kinds: List[str] = []
        current: List[int] = []
        hidden_counts = torch.zeros(hidden, device=device)
        content_counts = torch.zeros(content, device=device)
        input_accumulations = 0
        recurrent_accumulations = 0
        readout_accumulations = 0
        natural_boundaries = 0
        forced_boundaries = 0

        def emit(kind: str) -> None:
            nonlocal hidden_membrane, hidden_spikes, output_membrane, boundary_membrane
            nonlocal current, natural_boundaries, forced_boundaries, content_counts
            content_code = self._spike(output_membrane - output_threshold)
            code = torch.cat((torch.ones(1, device=device), content_code))
            patches.append(bytes(current))
            codes.append(code)
            boundary_kinds.append(kind)
            content_counts = content_counts + content_code
            if kind == "neural_spike":
                natural_boundaries += 1
            elif kind == "max_length_guard":
                forced_boundaries += 1
            hidden_membrane = torch.zeros_like(hidden_membrane)
            hidden_spikes = torch.zeros_like(hidden_spikes)
            output_membrane = torch.zeros_like(output_membrane)
            boundary_membrane = torch.zeros_like(boundary_membrane)
            current = []

        for value in values:
            current.append(value)
            previous_spikes = hidden_spikes
            candidate = (
                self.config.hidden_beta * hidden_membrane
                + input_weight[value]
                + F.linear(hidden_spikes, recurrent_weight)
            )
            hidden_spikes = self._spike(candidate - hidden_threshold)
            hidden_membrane = candidate * (1.0 - hidden_spikes)
            output_membrane = (
                self.config.output_beta * output_membrane
                + F.linear(hidden_spikes, readout_weight)
            )
            boundary_membrane = (
                self.config.boundary_beta * boundary_membrane
                + torch.dot(hidden_spikes, boundary_weight)
            )
            boundary_spike = self._spike(boundary_membrane - boundary_threshold)
            hidden_counts = hidden_counts + hidden_spikes
            input_accumulations += int(self.input_mask[value].sum().item())
            recurrent_accumulations += int(
                (previous_spikes * recurrent_fanout).sum().item()
            )
            readout_accumulations += int((hidden_spikes * readout_fanout).sum().item())

            eligible = len(current) >= self.config.min_patch_bytes
            if eligible and bool(boundary_spike.item()):
                emit("neural_spike")
            elif len(current) >= self.config.max_patch_bytes:
                emit("max_length_guard")

        if current:
            emit("end_of_stream_flush")
        code_tensor = torch.stack(codes, dim=0)
        hidden_events = int(hidden_counts.sum().item())
        content_events = int(content_counts.sum().item())
        stats = self._stats_template(
            len(values), len(patches), hidden_events, content_events,
            input_accumulations, recurrent_accumulations,
        )
        stats.update({
            "readout_synaptic_accumulations_event_driven": readout_accumulations,
            "boundary_synaptic_accumulations_event_driven": hidden_events,
            "boundary_lif_state_updates": len(values),
            "boundary_threshold_evaluations": len(values),
            "dense_reference_boundary_macs_actual": len(values) * hidden,
            "natural_boundary_spikes": natural_boundaries,
            "max_length_guard_boundaries": forced_boundaries,
            "end_of_stream_flushes": int(boundary_kinds[-1] == "end_of_stream_flush"),
            "mean_patch_bytes": len(values) / max(len(patches), 1),
            "natural_boundary_rate_per_byte": natural_boundaries / max(len(values), 1),
            "emitted_boundary_rate_per_byte": len(patches) / max(len(values), 1),
            "content_spike_rate": content_events / max(len(patches) * content, 1),
            "hidden_spike_rate": hidden_events / max(len(values) * hidden, 1),
        })
        self._last_stats = stats
        self._last_hidden_counts = hidden_counts.cpu()
        self._last_content_counts = content_counts.cpu()
        return patches, code_tensor, boundary_kinds, dict(stats)

    @torch.no_grad()
    def fit_patch_homeostasis(
        self,
        patches: Sequence[bytes],
        *,
        batch_size: int = 512,
        corpus_sha256: str | None = None,
    ) -> Dict[str, object]:
        """Fit local firing thresholds from training patch occurrences only."""

        if self.learning_rule != "homeostatic":
            raise RuntimeError("Patch homeostasis is available only in homeostatic mode")
        if batch_size <= 0 or not patches:
            raise ValueError("Homeostatic fitting requires patches and a positive batch size")
        hidden_counts = torch.zeros_like(self.hidden_threshold, device="cpu")
        content_counts = torch.zeros_like(self.output_threshold, device="cpu")
        operation_totals: Dict[str, float] = {}
        raw_bytes = 0
        patch_count = 0
        for start in range(0, len(patches), batch_size):
            batch = patches[start : start + batch_size]
            _, stats = self.encode_patches(batch)
            raw_bytes += int(stats["raw_byte_input_spike_events"])
            patch_count += len(batch)
            hidden_counts += self._last_hidden_counts
            content_counts += self._last_content_counts
            merge_frontend_operation_stats(operation_totals, stats)
        hidden_rates = hidden_counts / max(raw_bytes, 1)
        content_rates = content_counts / max(patch_count, 1)
        self.hidden_threshold.add_(
            self.config.homeostatic_hidden_lr
            * (hidden_rates.to(self.hidden_threshold.device) - self.config.target_hidden_rate)
        ).clamp_(0.05, 20.0)
        self.output_threshold.add_(
            self.config.homeostatic_output_lr
            * (content_rates.to(self.output_threshold.device) - self.config.target_content_rate)
        ).clamp_(0.05, 20.0)
        self.homeostasis_fitted = True
        self.homeostasis_report = {
            "homeostasis_rule": "per_neuron_firing_threshold_rate_error_one_training_pass",
            "homeostasis_training_only": 1,
            "homeostasis_fit_patches": patch_count,
            "homeostasis_fit_raw_bytes": raw_bytes,
            "homeostasis_fit_corpus_sha256": corpus_sha256 or "caller_not_supplied",
            "homeostasis_preupdate_hidden_rate_mean": float(hidden_rates.mean().item()),
            "homeostasis_preupdate_content_rate_mean": float(content_rates.mean().item()),
            "homeostasis_target_hidden_rate": self.config.target_hidden_rate,
            "homeostasis_target_content_rate": self.config.target_content_rate,
            "homeostasis_state_digest": self.state_digest(),
            **{
                f"homeostasis_fit_{key}": value
                for key, value in operation_totals.items()
            },
        }
        return dict(self.homeostasis_report)

    @torch.no_grad()
    def fit_joint_homeostasis(
        self,
        training_bytes: bytes,
        *,
        corpus_sha256: str | None = None,
    ) -> Dict[str, object]:
        """Fit reservoir, content, and boundary thresholds on training bytes."""

        if self.learning_rule != "homeostatic":
            raise RuntimeError("Joint homeostasis is available only in homeostatic mode")
        _, _, _, stats = self.segment_and_encode(training_bytes)
        raw_bytes = int(stats["raw_byte_input_spike_events"])
        patches = int(stats["patch_completion_spike_events"])
        hidden_rates = self._last_hidden_counts / max(raw_bytes, 1)
        content_rates = self._last_content_counts / max(patches, 1)
        emitted_boundary_rate = patches / max(raw_bytes, 1)
        self.hidden_threshold.add_(
            self.config.homeostatic_hidden_lr
            * (hidden_rates.to(self.hidden_threshold.device) - self.config.target_hidden_rate)
        ).clamp_(0.05, 20.0)
        self.output_threshold.add_(
            self.config.homeostatic_output_lr
            * (content_rates.to(self.output_threshold.device) - self.config.target_content_rate)
        ).clamp_(0.05, 20.0)
        self.boundary_threshold.add_(
            self.config.homeostatic_boundary_lr
            * (emitted_boundary_rate - self.config.target_boundary_rate_per_byte)
        ).clamp_(0.05, 20.0)
        self.homeostasis_fitted = True
        self.homeostasis_report = {
            "homeostasis_rule": "local_threshold_rate_error_one_training_pass",
            "homeostasis_training_only": 1,
            "homeostasis_fit_raw_bytes": raw_bytes,
            "homeostasis_fit_corpus_sha256": corpus_sha256 or hashlib.sha256(training_bytes).hexdigest(),
            "homeostasis_preupdate_hidden_rate_mean": float(hidden_rates.mean().item()),
            "homeostasis_preupdate_content_rate_mean": float(content_rates.mean().item()),
            "homeostasis_preupdate_boundary_rate_per_byte": emitted_boundary_rate,
            "homeostasis_target_hidden_rate": self.config.target_hidden_rate,
            "homeostasis_target_content_rate": self.config.target_content_rate,
            "homeostasis_target_boundary_rate_per_byte": self.config.target_boundary_rate_per_byte,
            "homeostasis_state_digest": self.state_digest(),
            **{
                f"homeostasis_fit_{key}": value
                for key, value in stats.items()
                if isinstance(value, (int, float))
                and not isinstance(value, bool)
                and key in {
                    "raw_byte_input_spike_events",
                    "patch_completion_spike_events",
                    "hidden_spike_events",
                    "content_spike_events",
                    "input_synaptic_accumulations_event_driven",
                    "recurrent_synaptic_accumulations_event_driven",
                    "readout_synaptic_accumulations_event_driven",
                    "boundary_synaptic_accumulations_event_driven",
                    "hidden_lif_state_updates",
                    "boundary_lif_state_updates",
                    "dense_reference_recurrent_macs_actual",
                    "dense_reference_readout_macs_actual",
                    "dense_reference_boundary_macs_actual",
                }
            },
        }
        return dict(self.homeostasis_report)

    @torch.no_grad()
    def state_digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.learning_rule.encode("ascii"))
        digest.update(repr(asdict(self.config)).encode("utf-8"))
        for name, tensor in sorted(self.state_dict().items()):
            value = tensor.detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(value.numpy().tobytes())
        return digest.hexdigest()

    def last_operation_stats(self) -> Dict[str, float | int | str]:
        return dict(self._last_stats)


def audit_patch_spike_codes(
    patches: Sequence[bytes],
    codes: torch.Tensor,
) -> Dict[str, float | int]:
    """Measure determinism, collisions, silence, and reversal sensitivity."""

    if codes.ndim != 2 or len(patches) != int(codes.shape[0]):
        raise ValueError("Patch/code audit expects aligned [patch, channel] rows")
    hard = codes.detach().cpu()
    if not torch.all((hard == 0) | (hard == 1)):
        raise AssertionError("Patch code audit received a non-binary code")
    first_code: Dict[bytes, Tuple[int, ...]] = {}
    deterministic = True
    for patch, row in zip(patches, hard):
        key = bytes(patch)
        code = tuple(int(value) for value in row.tolist())
        previous = first_code.setdefault(key, code)
        deterministic = deterministic and previous == code
    unique_payloads = len(first_code)
    unique_codewords = len(set(first_code.values()))
    row_spikes = hard.sum(dim=1) if hard.numel() else torch.zeros(0)
    return {
        "audit_binary_codes": 1,
        "audit_completion_channel_always_spikes": int(
            hard.shape[0] == 0 or bool(torch.all(hard[:, 0] == 1))
        ),
        "audit_duplicate_payloads_are_deterministic": int(deterministic),
        "audit_unique_payloads": unique_payloads,
        "audit_unique_codewords": unique_codewords,
        "audit_distinct_payload_collisions": unique_payloads - unique_codewords,
        "audit_distinct_payload_collision_fraction": (
            (unique_payloads - unique_codewords) / max(unique_payloads, 1)
        ),
        "audit_spikes_per_patch_min": float(row_spikes.min().item()) if len(row_spikes) else 0.0,
        "audit_spikes_per_patch_mean": float(row_spikes.mean().item()) if len(row_spikes) else 0.0,
        "audit_spikes_per_patch_max": float(row_spikes.max().item()) if len(row_spikes) else 0.0,
    }


class SpikePatchPayloadInterface(nn.Module):
    """LM interface that executes an SNN on each modelling unit's raw payload."""

    def __init__(
        self,
        payloads_by_id: Sequence[bytes],
        *,
        mode: str,
        seed: int = 1,
        target_rate: float = 0.125,
        homeostatic_fit_ids: Sequence[int] | None = None,
        fit_corpus_sha256: str | None = None,
    ) -> None:
        super().__init__()
        if mode not in SPIKEPATCH_INTERFACE_MODES:
            raise ValueError(f"Unsupported spike-patch interface: {mode}")
        if not payloads_by_id:
            raise ValueError("Spike-patch interface requires an ID-to-payload table")
        self.payloads_by_id = tuple(bytes(value) for value in payloads_by_id)
        self.vocab_size = len(self.payloads_by_id)
        self.mode = mode
        self.target_rate = float(target_rate)
        config = SpikePatchSNNConfig(target_content_rate=target_rate, seed=seed)
        self.encoder = SpikeNativePatchEncoder(
            config,
            learning_rule=SPIKEPATCH_INTERFACE_MODES[mode],
        )
        if mode == "spikepatch_reservoir_homeostatic64":
            if homeostatic_fit_ids is None:
                raise ValueError("Homeostatic spike-patch interface requires training IDs")
            fitting_patches = [self.payloads_by_id[int(token_id)] for token_id in homeostatic_fit_ids]
            self.encoder.fit_patch_homeostasis(
                fitting_patches,
                corpus_sha256=fit_corpus_sha256,
            )
        self._last_stats: Dict[str, float | int | str] = {}

    @property
    def is_binary(self) -> bool:
        return True

    def forward(self, token_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        flat_ids = token_ids.detach().reshape(-1).cpu().tolist()
        if any(token_id < 0 or token_id >= self.vocab_size for token_id in flat_ids):
            raise ValueError("Token ID is outside the spike-patch payload vocabulary")
        payloads = [self.payloads_by_id[int(token_id)] for token_id in flat_ids]
        spikes, stats = self.encoder.encode_patches(payloads)
        features = spikes.reshape(*token_ids.shape, self.encoder.output_dim)
        if self.mode == "spikepatch_reservoir_surrogate64" and self.training:
            content_rate = features[..., 1:].mean()
            regularizer = (content_rate - self.target_rate).square()
        else:
            regularizer = features.new_zeros(())
        self._last_stats = stats
        return features, regularizer

    @torch.no_grad()
    def audit(self, *, fail_on_collision: bool = False) -> Dict[str, object]:
        codes: List[torch.Tensor] = []
        for start in range(0, self.vocab_size, 512):
            batch, _ = self.encoder.encode_patches(
                self.payloads_by_id[start : start + 512]
            )
            codes.append(batch.cpu())
        code_tensor = torch.cat(codes, dim=0)
        measured = audit_patch_spike_codes(self.payloads_by_id, code_tensor)
        # Content collisions are an experimental outcome, unlike collisions in
        # an arbitrary control codebook.  ``fail_on_collision`` is accepted for
        # API compatibility but deliberately does not suppress the experiment.
        return {
            "frontend_operation_ledger_version": FRONTEND_LEDGER_VERSION,
            "interface_encoder": self.mode,
            "interface_binary": 1,
            "interface_dimension": self.encoder.output_dim,
            "interface_unique_codewords": measured["audit_unique_codewords"],
            "interface_collisions": measured["audit_distinct_payload_collisions"],
            "interface_collision_fraction": measured["audit_distinct_payload_collision_fraction"],
            "interface_spikes_per_unit_min": measured["audit_spikes_per_patch_min"],
            "interface_spikes_per_unit_mean": measured["audit_spikes_per_patch_mean"],
            "interface_spikes_per_unit_max": measured["audit_spikes_per_patch_max"],
            "interface_channel_usage_cv": "reported_in_stage_c_aggregator",
            "interface_sampled_hamming_min": "reported_in_stage_c_aggregator",
            "interface_sampled_hamming_mean": "reported_in_stage_c_aggregator",
            "interface_trainable": int(self.encoder.is_trainable),
            "interface_parameter_count": sum(p.numel() for p in self.parameters()),
            "interface_target_rate": self.target_rate,
            "interface_temperature": "not_applicable_lif_dynamics",
            "interface_surrogate": (
                "atan_alpha_2_heaviside_forward"
                if self.encoder.is_trainable else "not_applicable"
            ),
            "interface_role": "raw_byte_event_recurrent_lif_patch_content_encoder",
            "interface_rank_multiplier": "not_applicable",
            "interface_rank_offset": "not_applicable",
            "interface_collision_gate": "measured_outcome_not_fail_closed_control_gate",
            "interface_uses_arbitrary_token_id_code": 0,
            "interface_uses_dense_byte_histogram": 0,
            "interface_uses_patch_hash": 0,
            "interface_executes_payload_snn_per_occurrence": 1,
            "interface_completion_spike_channel": 0,
            "interface_learning_rule": self.encoder.learning_rule,
            "interface_state_digest": self.encoder.state_digest(),
            "frontend_synaptic_lif_and_threshold_count_complete": 1,
            "frontend_operation_count_complete": 0,
            "frontend_memory_access_count_complete": 0,
            "frontend_hardware_energy_estimate_ready": 0,
            "frontend_hardware_energy_claim_permitted": 0,
            **self.encoder.homeostasis_report,
        }

    def last_operation_stats(self) -> Dict[str, float | int | str]:
        return dict(self._last_stats)


class JointSpikingPatchSegmenter:
    """Compatibility wrapper for the canonical native-patch segmentation API."""

    def __init__(
        self,
        *,
        learning_rule: str = "fixed",
        seed: int = 1,
        training_bytes: bytes | None = None,
        config: SpikePatchSNNConfig | None = None,
        training_sha256: str | None = None,
    ) -> None:
        chosen = config or SpikePatchSNNConfig(seed=seed)
        self.encoder = SpikeNativePatchEncoder(chosen, learning_rule=learning_rule)
        if learning_rule == "homeostatic":
            if training_bytes is None:
                raise ValueError("Homeostatic joint segmenter requires training bytes")
            self.encoder.fit_joint_homeostasis(
                training_bytes,
                corpus_sha256=training_sha256,
            )

    def segment_bytes(self, data: bytes | Sequence[int]):
        patches, _, _, stats = self.encoder.segment_and_encode(data)
        steps = [1 for _ in patches]
        return patches, steps, stats

    def segment_text(self, text: str):
        return self.segment_bytes(text.encode("utf-8", errors="strict"))


def merge_frontend_operation_stats(
    totals: Dict[str, float],
    stats: Mapping[str, object],
) -> None:
    """Accumulate numeric per-call frontend counts without double counting labels."""

    additive_keys = {
        "raw_byte_input_spike_events",
        "patch_completion_spike_events",
        "hidden_spike_events",
        "content_spike_events",
        "input_synaptic_accumulations_event_driven",
        "recurrent_synaptic_accumulations_event_driven",
        "readout_synaptic_accumulations_event_driven",
        "boundary_synaptic_accumulations_event_driven",
        "hidden_lif_state_updates",
        "hidden_threshold_evaluations",
        "hidden_reset_gate_elements",
        "output_membrane_state_updates",
        "output_threshold_evaluations",
        "output_reset_gate_elements",
        "natural_boundary_spikes",
        "dense_reference_recurrent_macs_actual",
        "dense_reference_readout_macs_actual",
        "dense_reference_boundary_macs_actual",
        "padded_reference_rows",
        "padded_reference_extra_rows",
        "silent_content_rows",
    }
    for key in additive_keys:
        value = stats.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            totals[key] = totals.get(key, 0.0) + float(value)
