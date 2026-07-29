"""BLT-inspired spiking next-byte-surprise patch segmenters.

This is not a conversion of the complete Byte Latent Transformer (BLT).  BLT
also contains trainable local byte encoder/decoder networks and a global latent
transformer.  The implementation here isolates one BLT principle: dynamic
patch boundaries generated from next-byte predictability.  Raw byte events
drive a recurrent LIF predictor, prediction-error spikes drive a boundary LIF,
and the emitted patches are later serialised by the repository's lossless
digital codec before a SpikeGPT language-model backbone consumes them.

Communication between the predictor stages is binary.  Continuous state is
limited to membrane potential and trainable synaptic weights.  The PyTorch
reference executes masked dense operations and reports them separately from
event-conditioned accumulation counts; neither count is a hardware-energy
claim.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from src.models.spikegpt_reference import atan_spike


SPIKING_BLT_METHODS = {
    "spike_blt_surprise_fixed": "fixed",
    "spike_blt_surprise_surrogate_0.100": "predictive_surrogate",
}
SPIKING_BLT_LEDGER_VERSION = "spiking_blt_surprise_snn_algorithmic_ledger_v1"


def spiking_blt_method_spec(method: str) -> Tuple[str, float]:
    """Return learning rule and boundary-rate target for declared/tuning names."""

    if method == "spike_blt_surprise_fixed":
        return "fixed", 0.100
    prefix = "spike_blt_surprise_surrogate_"
    if not method.startswith(prefix):
        raise ValueError(f"unsupported spiking BLT method: {method}")
    try:
        target = float(method[len(prefix):])
    except ValueError as exc:
        raise ValueError(f"invalid spiking BLT target in {method}") from exc
    if target not in {0.075, 0.100, 0.125, 0.150, 0.200}:
        raise ValueError(f"spiking BLT target is outside the predeclared grid: {target}")
    return "predictive_surrogate", target


@dataclass(frozen=True)
class SpikingBLTConfig:
    """Declared dynamics for the controlled Stage C-BLT extension."""

    hidden_dim: int = 64
    prediction_dim: int = 32
    prediction_active_bits: int = 4
    input_fanout: int = 12
    recurrent_fanout: int = 4
    prediction_fanout: int = 8
    hidden_beta: float = 0.85
    prediction_beta: float = 0.50
    boundary_beta: float = 0.90
    hidden_threshold: float = 0.55
    prediction_threshold: float = 0.75
    boundary_threshold: float = 2.00
    input_scale: float = 1.00
    recurrent_scale: float = 0.20
    prediction_scale: float = 0.35
    surprise_hamming_threshold: int = 4
    boundary_surprise_scale: float = 1.0
    min_patch_bytes: int = 4
    max_patch_bytes: int = 24
    target_boundary_rate_per_byte: float = 0.100
    target_prediction_rate: float = 0.125
    fit_byte_limit: int = 1_000_000
    fit_sequence_length: int = 64
    fit_batch_size: int = 64
    fit_learning_rate: float = 0.002
    fit_rate_weight: float = 0.10
    surrogate_alpha: float = 2.0
    seed: int = 1

    def validate(self) -> None:
        if self.hidden_dim <= 0 or self.prediction_dim <= 0:
            raise ValueError("hidden and prediction dimensions must be positive")
        if not 0 < self.prediction_active_bits < self.prediction_dim:
            raise ValueError("prediction active bits must lie inside prediction_dim")
        if math.comb(self.prediction_dim, self.prediction_active_bits) < 256:
            raise ValueError("prediction spike-code capacity must cover all bytes")
        if not 0 < self.input_fanout <= self.hidden_dim:
            raise ValueError("invalid input fanout")
        if not 0 < self.recurrent_fanout <= self.hidden_dim:
            raise ValueError("invalid recurrent fanout")
        if not 0 < self.prediction_fanout <= self.prediction_dim:
            raise ValueError("invalid prediction fanout")
        if self.min_patch_bytes <= 0 or self.max_patch_bytes < self.min_patch_bytes:
            raise ValueError("invalid patch-length guards")
        if self.fit_byte_limit < 2 or self.fit_sequence_length < 2:
            raise ValueError("predictor fitting requires at least two bytes")
        if self.fit_batch_size <= 0 or self.fit_learning_rate <= 0:
            raise ValueError("invalid fitting batch size or learning rate")
        for name in ("hidden_beta", "prediction_beta", "boundary_beta"):
            if not 0 < float(getattr(self, name)) < 1:
                raise ValueError(f"{name} must lie strictly between zero and one")
        if not 0 < self.target_boundary_rate_per_byte < 1:
            raise ValueError("target boundary rate must lie strictly between zero and one")


def injective_byte_spike_codebook(
    dimension: int = 32,
    active_bits: int = 4,
) -> torch.Tensor:
    """Return the first 256 lexicographic constant-weight binary codes."""

    if dimension <= 0 or not 0 < active_bits < dimension:
        raise ValueError("invalid constant-weight code dimensions")
    if math.comb(dimension, active_bits) < 256:
        raise ValueError("constant-weight codebook cannot encode 256 bytes")
    rows = torch.zeros(256, dimension)
    combination = list(range(active_bits))
    for byte_value in range(256):
        rows[byte_value, combination] = 1.0
        for index in range(active_bits - 1, -1, -1):
            if combination[index] != index + dimension - active_bits:
                break
        else:
            raise AssertionError("constant-weight code enumeration exhausted early")
        if byte_value == 255:
            continue
        combination[index] += 1
        for tail in range(index + 1, active_bits):
            combination[tail] = combination[tail - 1] + 1
    if torch.unique(rows, dim=0).shape[0] != 256:
        raise AssertionError("byte prediction spike codebook is not injective")
    return rows


def _masked_weights(config: SpikingBLTConfig) -> Dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(config.seed))
    hidden = config.hidden_dim
    prediction = config.prediction_dim

    input_mask = torch.zeros(256, hidden)
    for byte_value in range(256):
        destinations = torch.randperm(hidden, generator=generator)[: config.input_fanout]
        input_mask[byte_value, destinations] = 1.0
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
    recurrent_weight = recurrent_mask * recurrent_sign * config.recurrent_scale

    prediction_mask = torch.zeros(prediction, hidden)
    for source in range(hidden):
        destinations = torch.randperm(prediction, generator=generator)[: config.prediction_fanout]
        prediction_mask[destinations, source] = 1.0
    prediction_weight = prediction_mask * config.prediction_scale
    return {
        "input_mask": input_mask,
        "input_weight": input_weight,
        "recurrent_mask": recurrent_mask,
        "recurrent_weight": recurrent_weight,
        "prediction_mask": prediction_mask,
        "prediction_weight": prediction_weight,
    }


class SpikingNextBytePredictor(nn.Module):
    """Masked recurrent LIF network that predicts a sparse next-byte code."""

    def __init__(
        self,
        config: SpikingBLTConfig,
        *,
        trainable: bool,
    ) -> None:
        super().__init__()
        config.validate()
        self.config = config
        topology = _masked_weights(config)
        for name in ("input_mask", "recurrent_mask", "prediction_mask"):
            self.register_buffer(name, topology[name])
        for name in ("input_weight", "recurrent_weight", "prediction_weight"):
            value = topology[name]
            if trainable:
                setattr(self, name, nn.Parameter(value))
            else:
                self.register_buffer(name, value)
        self.register_buffer(
            "byte_codebook",
            injective_byte_spike_codebook(
                config.prediction_dim,
                config.prediction_active_bits,
            ),
        )
        self.trainable_predictor = bool(trainable)

    def _spike(self, value: torch.Tensor) -> torch.Tensor:
        return atan_spike(value, alpha=self.config.surrogate_alpha)

    def step(
        self,
        byte_ids: torch.Tensor,
        hidden_membrane: torch.Tensor,
        hidden_spikes: torch.Tensor,
        prediction_membrane: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        input_weight = self.input_weight * self.input_mask
        recurrent_weight = self.recurrent_weight * self.recurrent_mask
        prediction_weight = self.prediction_weight * self.prediction_mask
        candidate = (
            self.config.hidden_beta * hidden_membrane
            + input_weight[byte_ids]
            + F.linear(hidden_spikes, recurrent_weight)
        )
        next_hidden_spikes = self._spike(candidate - self.config.hidden_threshold)
        next_hidden_membrane = candidate * (1.0 - next_hidden_spikes)
        next_prediction_membrane = (
            self.config.prediction_beta * prediction_membrane
            + F.linear(next_hidden_spikes, prediction_weight)
        )
        prediction_spikes = self._spike(
            next_prediction_membrane - self.config.prediction_threshold
        )
        next_prediction_membrane = next_prediction_membrane * (1.0 - prediction_spikes)
        return next_hidden_membrane, next_hidden_spikes, next_prediction_membrane, prediction_spikes

    def state_digest(self) -> str:
        digest = hashlib.sha256(repr(asdict(self.config)).encode("utf-8"))
        for name, tensor in sorted(self.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()


class SpikingBLTSurpriseSegmenter:
    """Frozen next-byte-error SNN plus boundary LIF and length guards."""

    def __init__(
        self,
        *,
        learning_rule: str = "fixed",
        seed: int = 1,
        training_bytes: bytes | None = None,
        training_sha256: str | None = None,
        config: SpikingBLTConfig | None = None,
    ) -> None:
        if learning_rule not in {"fixed", "predictive_surrogate"}:
            raise ValueError(f"unsupported spiking BLT learning rule: {learning_rule}")
        chosen = config or SpikingBLTConfig(seed=seed)
        chosen.validate()
        self.config = chosen
        self.learning_rule = learning_rule
        self.predictor = SpikingNextBytePredictor(
            chosen,
            trainable=learning_rule == "predictive_surrogate",
        )
        # Compatibility with the repository's frozen-front-end digest helper.
        self.model = self.predictor
        self.boundary_threshold = float(chosen.boundary_threshold)
        self.fit_report: Dict[str, object] = {}
        if learning_rule == "predictive_surrogate":
            if training_bytes is None:
                raise ValueError("predictive spiking BLT requires training bytes")
            self.fit_predictor(
                training_bytes,
                corpus_sha256=training_sha256,
            )

    def _training_blocks(self, values: torch.Tensor) -> List[torch.Tensor]:
        usable = values[: self.config.fit_byte_limit]
        block = self.config.fit_sequence_length + 1
        return [usable[start : start + block] for start in range(0, len(usable) - 1, block - 1)]

    def fit_predictor(
        self,
        training_bytes: bytes,
        *,
        corpus_sha256: str | None = None,
    ) -> Dict[str, object]:
        if self.learning_rule != "predictive_surrogate":
            raise RuntimeError("predictor fitting is available only in predictive_surrogate mode")
        if len(training_bytes) < 2:
            raise ValueError("predictor fitting requires at least two training bytes")
        values = torch.tensor(list(training_bytes), dtype=torch.long)
        blocks = self._training_blocks(values)
        optimizer = torch.optim.Adam(
            self.predictor.parameters(),
            lr=self.config.fit_learning_rate,
            betas=(0.9, 0.99),
            weight_decay=0.0,
        )
        total_targets = 0
        total_loss = 0.0
        update_steps = 0
        self.predictor.train()
        for offset in range(0, len(blocks), self.config.fit_batch_size):
            batch = blocks[offset : offset + self.config.fit_batch_size]
            if not batch:
                continue
            width = min(len(value) for value in batch) - 1
            if width <= 0:
                continue
            source = torch.stack([value[:width] for value in batch])
            target = torch.stack([value[1 : width + 1] for value in batch])
            hidden_membrane = torch.zeros(len(batch), self.config.hidden_dim)
            hidden_spikes = torch.zeros_like(hidden_membrane)
            prediction_membrane = torch.zeros(len(batch), self.config.prediction_dim)
            losses: List[torch.Tensor] = []
            for step in range(width):
                hidden_membrane, hidden_spikes, prediction_membrane, prediction_spikes = self.predictor.step(
                    source[:, step], hidden_membrane, hidden_spikes, prediction_membrane
                )
                target_spikes = self.predictor.byte_codebook[target[:, step]]
                prediction_error = F.mse_loss(prediction_spikes, target_spikes)
                rate_error = (
                    prediction_spikes.mean() - self.config.target_prediction_rate
                ).square()
                losses.append(prediction_error + self.config.fit_rate_weight * rate_error)
            loss = torch.stack(losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                self.predictor.input_weight.mul_(self.predictor.input_mask)
                self.predictor.recurrent_weight.mul_(self.predictor.recurrent_mask)
                self.predictor.prediction_weight.mul_(self.predictor.prediction_mask)
            total_targets += len(batch) * width
            total_loss += float(loss.detach().item()) * len(batch) * width
            update_steps += 1
        self.predictor.eval()
        fitting_selection = training_bytes[: self.config.fit_byte_limit]
        self.boundary_threshold = self._calibrate_boundary_threshold(
            fitting_selection
        )
        self.fit_report = {
            "spiking_blt_fit_training_only": 1,
            "spiking_blt_fit_source_sha256": corpus_sha256 or hashlib.sha256(training_bytes).hexdigest(),
            "spiking_blt_fit_selection_sha256": hashlib.sha256(fitting_selection).hexdigest(),
            "spiking_blt_fit_raw_bytes_available": len(training_bytes),
            "spiking_blt_fit_raw_bytes_used": len(fitting_selection),
            "spiking_blt_fit_targets": total_targets,
            "spiking_blt_fit_optimizer_steps": update_steps,
            "spiking_blt_fit_loss_mean": total_loss / max(total_targets, 1),
            "spiking_blt_boundary_threshold": self.boundary_threshold,
            "spiking_blt_target_boundary_rate_per_byte": self.config.target_boundary_rate_per_byte,
            "spiking_blt_predictor_state_digest": self.predictor.state_digest(),
        }
        return dict(self.fit_report)

    @torch.no_grad()
    def _surprise_trace(self, data: bytes) -> Tuple[List[int], Dict[str, int]]:
        hidden_membrane = torch.zeros(1, self.config.hidden_dim)
        hidden_spikes = torch.zeros_like(hidden_membrane)
        prediction_membrane = torch.zeros(1, self.config.prediction_dim)
        previous_prediction: torch.Tensor | None = None
        surprise_events: List[int] = []
        hidden_events = 0
        prediction_events = 0
        recurrent_accumulations = 0
        prediction_accumulations = 0
        for value in data:
            if previous_prediction is None:
                surprise_events.append(0)
            else:
                target = self.predictor.byte_codebook[int(value)]
                hamming = int(torch.ne(previous_prediction[0], target).sum().item())
                surprise_events.append(int(hamming >= self.config.surprise_hamming_threshold))
            byte_tensor = torch.tensor([int(value)], dtype=torch.long)
            recurrent_accumulations += (
                int(hidden_spikes.sum().item()) * self.config.recurrent_fanout
            )
            hidden_membrane, hidden_spikes, prediction_membrane, previous_prediction = self.predictor.step(
                byte_tensor, hidden_membrane, hidden_spikes, prediction_membrane
            )
            current_hidden_events = int(hidden_spikes.sum().item())
            hidden_events += current_hidden_events
            prediction_accumulations += (
                current_hidden_events * self.config.prediction_fanout
            )
            prediction_events += int(previous_prediction.sum().item())
        return surprise_events, {
            "hidden_spike_events": hidden_events,
            "prediction_spike_events": prediction_events,
            "input_synaptic_accumulations_event_driven": (
                len(data) * self.config.input_fanout
            ),
            "recurrent_synaptic_accumulations_event_driven": (
                recurrent_accumulations
            ),
            "readout_synaptic_accumulations_event_driven": (
                prediction_accumulations
            ),
            "hidden_lif_state_updates": len(data) * self.config.hidden_dim,
            "hidden_threshold_evaluations": len(data) * self.config.hidden_dim,
            "hidden_reset_gate_elements": len(data) * self.config.hidden_dim,
            "output_membrane_state_updates": (
                len(data) * self.config.prediction_dim
            ),
            "output_threshold_evaluations": (
                len(data) * self.config.prediction_dim
            ),
            "output_reset_gate_elements": (
                len(data) * self.config.prediction_dim
            ),
            "dense_reference_recurrent_macs_actual": (
                len(data) * self.config.hidden_dim * self.config.hidden_dim
            ),
            "dense_reference_readout_macs_actual": (
                len(data) * self.config.hidden_dim * self.config.prediction_dim
            ),
            "hamming_comparison_elements": (
                max(len(data) - 1, 0) * self.config.prediction_dim
            ),
        }

    @torch.no_grad()
    def _emitted_boundary_rate(self, surprise: Sequence[int], threshold: float) -> float:
        membrane = 0.0
        patch_length = 0
        emitted = 0
        for event in surprise:
            patch_length += 1
            membrane = self.config.boundary_beta * membrane + self.config.boundary_surprise_scale * event
            eligible = patch_length >= self.config.min_patch_bytes
            if (eligible and membrane >= threshold) or patch_length >= self.config.max_patch_bytes:
                emitted += 1
                membrane = 0.0
                patch_length = 0
        if patch_length:
            emitted += 1
        return emitted / max(len(surprise), 1)

    @torch.no_grad()
    def _calibrate_boundary_threshold(self, training_bytes: bytes) -> float:
        surprise, _ = self._surprise_trace(training_bytes)
        candidates = [0.25 + 0.05 * index for index in range(1, 241)]
        return min(
            candidates,
            key=lambda threshold: (
                abs(self._emitted_boundary_rate(surprise, threshold) - self.config.target_boundary_rate_per_byte),
                threshold,
            ),
        )

    @torch.no_grad()
    def segment_bytes(
        self,
        data: bytes | bytearray | Sequence[int],
    ) -> Tuple[List[bytes], List[int], Dict[str, object]]:
        raw = bytes(int(value) for value in data)
        if not raw:
            raise ValueError("cannot segment an empty byte stream")
        surprise, predictor_stats = self._surprise_trace(raw)
        patches: List[bytes] = []
        current = bytearray()
        membrane = 0.0
        neural_boundaries = 0
        guard_boundaries = 0
        for value, event in zip(raw, surprise):
            current.append(value)
            membrane = self.config.boundary_beta * membrane + self.config.boundary_surprise_scale * event
            eligible = len(current) >= self.config.min_patch_bytes
            if eligible and membrane >= self.boundary_threshold:
                patches.append(bytes(current))
                current.clear()
                membrane = 0.0
                neural_boundaries += 1
            elif len(current) >= self.config.max_patch_bytes:
                patches.append(bytes(current))
                current.clear()
                membrane = 0.0
                guard_boundaries += 1
        flushes = int(bool(current))
        if current:
            patches.append(bytes(current))
        if b"".join(patches) != raw:
            raise AssertionError("spiking BLT segmenter did not preserve exact raw bytes")
        stats: Dict[str, object] = {
            "frontend_operation_ledger_version": SPIKING_BLT_LEDGER_VERSION,
            "raw_byte_input_spike_events": len(raw),
            "prediction_error_spike_events": sum(surprise),
            "boundary_lif_state_updates": len(raw),
            "boundary_threshold_evaluations": len(raw),
            "natural_boundary_spikes": neural_boundaries,
            "max_length_guard_boundaries": guard_boundaries,
            "end_of_stream_flushes": flushes,
            "patch_completion_spike_events": len(patches),
            "mean_patch_bytes": len(raw) / len(patches),
            "prediction_error_spike_rate_per_byte": sum(surprise) / len(raw),
            "emitted_boundary_rate_per_byte": len(patches) / len(raw),
            "boundary_threshold": self.boundary_threshold,
            "input_is_one_hot_raw_byte_event": 1,
            "predictor_communication_is_binary_spikes": 1,
            "continuous_neural_state_is_membrane_only": 1,
            "patch_target_serialisation_is_digital": 1,
            "dense_reference_uses_masked_synapses": 1,
            "frontend_synaptic_lif_threshold_and_hamming_count_complete": 1,
            "operation_scope": (
                "spiking_predictor_synaptic_lif_boundary_and_hamming_"
                "inference_counted_fit_codec_and_memory_excluded"
            ),
            "frontend_algorithmic_count_complete": 0,
            "frontend_memory_access_count_complete": 0,
            "frontend_hardware_energy_estimate_ready": 0,
            "frontend_hardware_energy_claim_permitted": 0,
            **predictor_stats,
            **self.fit_report,
        }
        return patches, [1 for _ in patches], stats

    def segment_text(self, text: str):
        return self.segment_bytes(text.encode("utf-8", errors="strict"))

    def state_digest(self) -> str:
        digest = hashlib.sha256(self.learning_rule.encode("ascii"))
        digest.update(self.predictor.state_digest().encode("ascii"))
        digest.update(repr(self.boundary_threshold).encode("ascii"))
        return digest.hexdigest()


def merge_spiking_blt_stats(
    totals: Dict[str, float],
    stats: Mapping[str, object],
) -> None:
    """Accumulate additive numeric counts for external qualification tools."""

    for key in (
        "raw_byte_input_spike_events",
        "hidden_spike_events",
        "prediction_spike_events",
        "prediction_error_spike_events",
        "boundary_lif_state_updates",
        "boundary_threshold_evaluations",
        "natural_boundary_spikes",
        "max_length_guard_boundaries",
        "end_of_stream_flushes",
        "patch_completion_spike_events",
    ):
        value = stats.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            totals[key] = totals.get(key, 0.0) + float(value)
