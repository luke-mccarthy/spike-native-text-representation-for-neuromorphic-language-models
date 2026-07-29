"""Auditable fixed and learned input interfaces for controlled SNN LM studies."""

from __future__ import annotations

import hashlib
import json
from typing import Dict, Sequence, Tuple

import torch
from torch import nn

from src.models.spikegpt_reference import atan_spike
from src.tokenizers.lossless_codecs import CombinationSpikeCodebook
from src.tokenizers.spike_native_patch_frontends import (
    SPIKEPATCH_INTERFACE_MODES,
    SpikeNativePatchEncoder,
    SpikePatchSNNConfig,
    SpikePatchPayloadInterface,
)


INTERFACE_DIMENSION = 32
INTERFACE_ACTIVE_BITS = 4

# name -> (dimension, active bits, rank multiplier, rank offset).  The two
# byte-native additions deliberately span the event/storage trade-off: 2-of-32
# keeps the existing narrow bus while halving events, whereas 1-of-256 uses the
# minimum possible number of input events at the cost of a wider input map.
FIXED_CODE_SPECS = {
    "fixed_4of32_a": (32, 4, 2654435761, 1013904223),
    "fixed_4of32_b": (32, 4, 2246822519, 3266489917),
    "fixed_4of32_c": (32, 4, 668265263, 374761393),
    "fixed_2of32_byte": (32, 2, 2654435761, 1013904223),
    "fixed_1of256_byte": (256, 1, 1, 0),
}

# The fixed codebooks are experimental controls whose stated purpose requires
# an injective ID-to-spike mapping.  The SpikeGPT embedding is different: its
# hard binary rows are trained by the LM objective, so convergence to the same
# sign pattern for two IDs is a measured representation outcome.  Stage B2
# records such collisions instead of censoring the whole trial after training.
MEASURED_COLLISION_MODES = {"spikegpt_embedding64_atan"}
INJECTIVE_RECURRENT_BYTE_MODE = "streaming_reservoir64_injective2"
FAIL_CLOSED_COLLISION_POLICY = "fail_closed_injective_control_codebook"
MEASURED_COLLISION_POLICY = "measured_trainable_representation_outcome"


def _fixed_code_tensor(
    vocab_size: int,
    name: str,
    *,
    device: torch.device | None = None,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    dimension, active_bits, multiplier, offset = FIXED_CODE_SPECS[name]
    if name == "fixed_1of256_byte":
        if vocab_size > 257:
            raise ValueError(
                "fixed_1of256_byte supports 256 literal bytes and one zero-event BOS"
            )
        tensor = torch.zeros(
            vocab_size, dimension, dtype=torch.float32, device=device
        )
        literal_rows = min(vocab_size, 256)
        indices = torch.arange(literal_rows, device=device)
        tensor[indices, indices] = 1.0
        return tensor, {
            "dimension": dimension,
            "active_bits": active_bits,
            "vocab_size": vocab_size,
            "rank_multiplier": multiplier,
            "rank_offset": offset,
            "interface_encoder": name,
            "shared_bos_uses_zero_event_code": int(vocab_size == 257),
        }
    codebook = CombinationSpikeCodebook(
        vocab_size=vocab_size,
        dimension=dimension,
        active_bits=active_bits,
        rank_multiplier=multiplier,
        rank_offset=offset,
    )
    tensor = torch.zeros(
        vocab_size, dimension, dtype=torch.float32, device=device
    )
    for token_id in range(vocab_size):
        tensor[token_id, list(codebook.channels(token_id))] = 1.0
    audit = dict(codebook.audit())
    audit["interface_encoder"] = name
    return tensor, audit


def _sampled_hamming(codes: torch.Tensor, pairs: int = 4096) -> Tuple[float, float]:
    rows = int(codes.shape[0])
    if rows < 2:
        return 0.0, 0.0
    count = min(pairs, rows)
    left = torch.arange(count, device=codes.device)
    right = (left * 104729 + 37) % rows
    right = torch.where(right == left, (right + 1) % rows, right)
    distances = (codes[left] != codes[right]).float().sum(dim=1)
    return float(distances.min().item()), float(distances.mean().item())


def audit_binary_codes(codes: torch.Tensor) -> Dict[str, object]:
    if codes.ndim != 2:
        raise ValueError("Interface code audit expects [vocab, channels]")
    hard = codes.detach().float().cpu()
    if not torch.all((hard == 0) | (hard == 1)):
        raise AssertionError("Binary interface emitted a non-binary value")
    unique_rows = int(torch.unique(hard, dim=0).shape[0])
    row_spikes = hard.sum(dim=1)
    channel_usage = hard.sum(dim=0)
    usage_mean = float(channel_usage.mean().item())
    usage_std = float(channel_usage.std(unbiased=False).item())
    sampled_min, sampled_mean = _sampled_hamming(hard)
    code_to_ids: Dict[Tuple[int, ...], list[int]] = {}
    for token_id, row in enumerate(hard):
        key = tuple(int(value) for value in row.tolist())
        code_to_ids.setdefault(key, []).append(token_id)
    collision_groups = [
        token_ids for token_ids in code_to_ids.values() if len(token_ids) > 1
    ]
    collision_groups.sort(key=lambda token_ids: token_ids[0])
    preview = [token_ids[:32] for token_ids in collision_groups[:64]]
    preview_truncated = int(
        len(collision_groups) > len(preview)
        or any(len(group) > 32 for group in collision_groups)
    )
    codebook_digest = hashlib.sha256(
        hard.to(torch.uint8).contiguous().numpy().tobytes()
    ).hexdigest()
    return {
        "interface_binary": 1,
        "interface_dimension": int(hard.shape[1]),
        "interface_unique_codewords": unique_rows,
        "interface_collisions": int(hard.shape[0]) - unique_rows,
        "interface_collision_fraction": (int(hard.shape[0]) - unique_rows) / max(int(hard.shape[0]), 1),
        "interface_collision_group_count": len(collision_groups),
        "interface_collision_token_ids_json": json.dumps(
            preview, separators=(",", ":")
        ),
        "interface_collision_token_ids_truncated": preview_truncated,
        "interface_codebook_sha256": codebook_digest,
        "interface_spikes_per_unit_min": float(row_spikes.min().item()),
        "interface_spikes_per_unit_mean": float(row_spikes.mean().item()),
        "interface_spikes_per_unit_max": float(row_spikes.max().item()),
        "interface_channel_usage_cv": usage_std / max(usage_mean, 1e-12),
        "interface_sampled_hamming_min": sampled_min,
        "interface_sampled_hamming_mean": sampled_mean,
    }


class SpikeInputInterface(nn.Module):
    """ID-to-feature interface that leaves the recurrent backbone unchanged."""

    def __init__(
        self,
        vocab_size: int,
        mode: str = "fixed_4of32_a",
        temperature: float = 1.0,
        target_rate: float = 0.125,
    ) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0 < target_rate < 1:
            raise ValueError("target_rate must lie strictly between zero and one")
        supported = set(FIXED_CODE_SPECS) | {
            "learned_topk_4of32_ste",
            "learned_threshold32_rate",
            "spikegpt_embedding64_atan",
            "dense32_ceiling",
        }
        if mode not in supported:
            raise ValueError(f"Unsupported spike input interface: {mode}")
        if mode in {"fixed_2of32_byte", "fixed_1of256_byte"} and vocab_size > 257:
            raise ValueError(
                f"{mode} is a codec-free raw-byte interface and only supports "
                "the 256 byte values plus the shared BOS"
            )
        self.vocab_size = int(vocab_size)
        self.mode = mode
        self.temperature = float(temperature)
        self.target_rate = float(target_rate)
        initial, fixed_audit = _fixed_code_tensor(vocab_size, "fixed_4of32_a")
        self._fixed_audit = fixed_audit
        if mode in FIXED_CODE_SPECS:
            fixed, audit = _fixed_code_tensor(vocab_size, mode)
            self.register_buffer("fixed_codes", fixed)
            self._fixed_audit = audit
            self.logits = None
            self.dense = None
        elif mode in {"learned_topk_4of32_ste", "learned_threshold32_rate"}:
            inactive_logit = -4.0 if mode == "learned_threshold32_rate" else -2.0
            logits = torch.where(
                initial > 0,
                torch.full_like(initial, 2.0),
                torch.full_like(initial, inactive_logit),
            )
            self.logits = nn.Parameter(logits)
            self.register_buffer("fixed_codes", torch.empty(0))
            self.dense = None
        elif mode == "spikegpt_embedding64_atan":
            self.dense = nn.Embedding(vocab_size, 64)
            nn.init.normal_(self.dense.weight, mean=0.0, std=1.0e-5)
            self.register_buffer("fixed_codes", torch.empty(0))
            self.logits = None
        else:
            self.dense = nn.Embedding(vocab_size, INTERFACE_DIMENSION)
            nn.init.normal_(self.dense.weight, mean=0.0, std=0.02)
            self.register_buffer("fixed_codes", torch.empty(0))
            self.logits = None

    @property
    def is_binary(self) -> bool:
        return self.mode != "dense32_ceiling"

    def _all_features(self, differentiable: bool) -> torch.Tensor:
        if self.mode in FIXED_CODE_SPECS:
            return self.fixed_codes
        if self.mode == "dense32_ceiling":
            return self.dense.weight
        if self.mode == "spikegpt_embedding64_atan":
            if differentiable:
                return atan_spike(self.dense.weight, alpha=2.0)
            return (self.dense.weight >= 0).to(self.dense.weight.dtype)
        if self.logits is None:
            raise AssertionError("Learned binary interface is missing logits")
        if self.mode == "learned_topk_4of32_ste":
            indices = self.logits.topk(INTERFACE_ACTIVE_BITS, dim=-1).indices
            hard = torch.zeros_like(self.logits).scatter_(1, indices, 1.0)
            soft = torch.softmax(self.logits / self.temperature, dim=-1) * INTERFACE_ACTIVE_BITS
        else:
            soft = torch.sigmoid(self.logits / self.temperature)
            hard = (soft >= 0.5).to(soft.dtype)
        if differentiable:
            # Parenthesise the zero-valued surrogate term so the forward
            # tensor remains bit-exactly hard rather than retaining round-off.
            return hard + (soft - soft.detach())
        return hard

    def forward(self, token_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self._all_features(differentiable=self.training)[token_ids]
        if self.mode == "learned_threshold32_rate" and self.training:
            probabilities = torch.sigmoid(self.logits / self.temperature)
            row_rates = probabilities.mean(dim=-1)
            regularizer = (row_rates - self.target_rate).square().mean()
        else:
            regularizer = features.new_zeros(())
        return features, regularizer

    @torch.no_grad()
    def audit(self, *, fail_on_collision: bool = True) -> Dict[str, object]:
        features = self._all_features(differentiable=False)
        collision_policy = (
            MEASURED_COLLISION_POLICY
            if self.mode in MEASURED_COLLISION_MODES
            else FAIL_CLOSED_COLLISION_POLICY
        )
        if self.is_binary:
            audit = audit_binary_codes(features)
            if (
                fail_on_collision
                and collision_policy == FAIL_CLOSED_COLLISION_POLICY
                and int(audit["interface_collisions"]) != 0
            ):
                raise RuntimeError(
                    f"Interface {self.mode} produced {audit['interface_collisions']} collisions"
                )
        else:
            collision_policy = "not_applicable_continuous_interface"
            audit = {
                "interface_binary": 0,
                "interface_dimension": int(features.shape[1]),
                "interface_unique_codewords": "not_applicable_continuous",
                "interface_collisions": "not_applicable_continuous",
                "interface_collision_fraction": "not_applicable_continuous",
                "interface_collision_group_count": "not_applicable_continuous",
                "interface_collision_token_ids_json": "not_applicable_continuous",
                "interface_collision_token_ids_truncated": "not_applicable_continuous",
                "interface_codebook_sha256": "not_applicable_continuous",
                "interface_spikes_per_unit_min": 0.0,
                "interface_spikes_per_unit_mean": 0.0,
                "interface_spikes_per_unit_max": 0.0,
                "interface_channel_usage_cv": "not_applicable_continuous",
                "interface_sampled_hamming_min": "not_applicable_continuous",
                "interface_sampled_hamming_mean": "not_applicable_continuous",
            }
        audit.update({
            "interface_encoder": self.mode,
            "interface_trainable": int(any(parameter.requires_grad for parameter in self.parameters())),
            "interface_parameter_count": sum(parameter.numel() for parameter in self.parameters()),
            "interface_target_rate": self.target_rate if self.mode == "learned_threshold32_rate" else "not_applicable",
            "interface_temperature": self.temperature if self.mode.startswith("learned_") else "not_applicable",
            "interface_surrogate": (
                "atan_alpha_2_heaviside_forward"
                if self.mode == "spikegpt_embedding64_atan"
                else "not_applicable"
            ),
            "interface_role": (
                "spikegpt_learned_binary_token_embedding"
                if self.mode == "spikegpt_embedding64_atan"
                else "controlled_frontend_interface"
            ),
            "interface_collision_policy": collision_policy,
        })
        reference_state_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in self.state_dict().values()
        )
        packed_codebook_bytes = (
            (int(features.numel()) + 7) // 8 if self.is_binary else 0
        )
        audit["interface_state_storage_reference_bytes"] = reference_state_bytes
        audit["interface_packed_binary_codebook_bytes"] = packed_codebook_bytes
        audit["interface_deployment_static_state_bytes"] = (
            packed_codebook_bytes
            if self.mode in FIXED_CODE_SPECS
            else reference_state_bytes
        )
        audit["interface_deployment_static_state_scope"] = (
            "packed_fixed_binary_routing_table"
            if self.mode in FIXED_CODE_SPECS
            else "reference_learned_interface_state"
        )
        if self.mode in FIXED_CODE_SPECS:
            audit["interface_rank_multiplier"] = self._fixed_audit["rank_multiplier"]
            audit["interface_rank_offset"] = self._fixed_audit["rank_offset"]
            audit["interface_fixed_code_active_bits"] = FIXED_CODE_SPECS[self.mode][1]
            audit["interface_codec_free_raw_byte_map"] = int(
                self.mode in {"fixed_2of32_byte", "fixed_1of256_byte"}
            )
        else:
            audit["interface_rank_multiplier"] = "not_applicable"
            audit["interface_rank_offset"] = "not_applicable"
            audit["interface_fixed_code_active_bits"] = "not_applicable"
            audit["interface_codec_free_raw_byte_map"] = 0
        return audit


class InjectiveRecurrentByteSpikeInterface(nn.Module):
    """Streaming recurrent-LIF byte features plus an injective spike suffix.

    State is reset at each independent LM batch sequence, not at each byte.
    The fixed suffix guarantees byte identity at every context while the
    64-neuron reservoir provides genuinely temporal spike features.  No
    digital tokenizer, patch vocabulary or payload codec is involved.
    """

    def __init__(
        self,
        payloads_by_id: Sequence[bytes],
        *,
        seed: int,
        target_rate: float,
        homeostatic_fit_ids: Sequence[int],
        fit_corpus_sha256: str | None,
    ) -> None:
        super().__init__()
        if len(payloads_by_id) != 257:
            raise ValueError(
                "injective recurrent byte interface requires 256 bytes plus BOS"
            )
        expected = tuple(bytes((value,)) for value in range(256)) + (b"",)
        if tuple(payloads_by_id) != expected:
            raise ValueError("injective recurrent byte interface received non-byte payloads")
        # Reuse the frozen sparse topology generator, but retain only the
        # input/recurrent tensors actually executed by this streaming path.
        config = SpikePatchSNNConfig(seed=seed)
        template = SpikeNativePatchEncoder(config, learning_rule="fixed")
        self.register_buffer("input_weight", template.input_weight.detach().clone())
        self.register_buffer(
            "recurrent_weight", template.recurrent_weight.detach().clone()
        )
        self.register_buffer(
            "hidden_threshold", template.hidden_threshold.detach().clone()
        )
        fixed, _ = _fixed_code_tensor(len(payloads_by_id), "fixed_2of32_byte")
        self.register_buffer("fixed_codes", fixed)
        self.hidden_beta = float(config.hidden_beta)
        self.vocab_size = len(payloads_by_id)
        self.payloads_by_id = tuple(payloads_by_id)
        self.target_rate = float(target_rate)
        self.fit_corpus_sha256 = fit_corpus_sha256 or "caller_not_supplied"
        self._last_stats: Dict[str, float | int | str] = {}
        self.mode = INJECTIVE_RECURRENT_BYTE_MODE

    @property
    def is_binary(self) -> bool:
        return True

    def forward(self, token_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if token_ids.ndim != 2:
            raise ValueError("streaming recurrent byte interface expects [batch, time]")
        if torch.any(token_ids < 0) or torch.any(token_ids >= self.vocab_size):
            raise ValueError("streaming recurrent byte interface received invalid ID")
        batch, steps = token_ids.shape
        hidden = int(self.hidden_threshold.numel())
        membrane = torch.zeros(
            batch, hidden, dtype=self.input_weight.dtype, device=token_ids.device
        )
        spikes = torch.zeros_like(membrane)
        outputs = []
        raw_events = 0
        hidden_events = 0
        input_accumulations = 0
        recurrent_accumulations = 0
        input_fanout = torch.count_nonzero(self.input_weight, dim=1)
        recurrent_fanout = torch.count_nonzero(self.recurrent_weight, dim=0)
        for index in range(steps):
            ids = token_ids[:, index]
            literal = ids < 256
            safe_ids = torch.where(literal, ids, torch.zeros_like(ids))
            current = self.input_weight[safe_ids] * literal.unsqueeze(-1)
            previous = spikes
            candidate = (
                self.hidden_beta * membrane
                + current
                + torch.nn.functional.linear(previous, self.recurrent_weight)
            )
            spikes = (candidate >= self.hidden_threshold).to(candidate.dtype)
            membrane = candidate * (1.0 - spikes)
            outputs.append(spikes)
            raw_events += int(literal.sum().item())
            hidden_events += int(spikes.detach().sum().item())
            input_accumulations += int(
                input_fanout[safe_ids].mul(literal).sum().item()
            )
            recurrent_accumulations += int(
                (previous.detach() * recurrent_fanout).sum().item()
            )
        recurrent = torch.stack(outputs, dim=1)
        fixed = self.fixed_codes[token_ids]
        self._last_stats = {
            "frontend_operation_ledger_version": (
                "streaming_recurrent_byte_event_ledger_v1"
            ),
            "raw_byte_input_spike_events": raw_events,
            "hidden_spike_events": hidden_events,
            "input_synaptic_accumulations_event_driven": input_accumulations,
            "recurrent_synaptic_accumulations_event_driven": recurrent_accumulations,
            "hidden_lif_state_updates": batch * steps * hidden,
            "hidden_threshold_evaluations": batch * steps * hidden,
            "hidden_reset_gate_elements": batch * steps * hidden,
        }
        features = torch.cat((recurrent, fixed), dim=-1)
        return features, features.new_zeros(())

    @torch.no_grad()
    def audit(self, *, fail_on_collision: bool = True) -> Dict[str, object]:
        ids = torch.arange(self.vocab_size, device=self.fixed_codes.device).unsqueeze(1)
        features, _ = self.forward(ids)
        # forward() already concatenates the 64 recurrent spikes and the
        # 32-channel injective suffix.  Audit that emitted 96-D interface
        # directly rather than appending the suffix a second time.
        combined = features[:, 0].cpu()
        audit = audit_binary_codes(combined)
        if fail_on_collision and int(audit["interface_collisions"]) != 0:
            raise RuntimeError("injective recurrent byte interface collided")
        sparse_synapses = sum(
            int(torch.count_nonzero(tensor).item())
            for tensor in (
                self.input_weight,
                self.recurrent_weight,
            )
        )
        neuron_state_elements = int(self.hidden_threshold.numel())
        fixed_packed_bytes = (int(self.fixed_codes.numel()) + 7) // 8
        deployment_state_bytes = (
            sparse_synapses * 6 + neuron_state_elements * 4 + fixed_packed_bytes
        )
        reference_state_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in self.state_dict().values()
        )
        audit.update({
            "frontend_operation_ledger_version": (
                "streaming_recurrent_byte_event_ledger_v1"
            ),
            "interface_encoder": self.mode,
            "interface_trainable": 0,
            "interface_parameter_count": 0,
            "interface_target_rate": "not_applicable_training_free_reservoir",
            "interface_temperature": "not_applicable_lif_dynamics",
            "interface_surrogate": "not_applicable_fixed_streaming_reservoir",
            "interface_role": "codec_free_streaming_recurrent_lif_plus_injective_raw_byte_spikes",
            "interface_rank_multiplier": FIXED_CODE_SPECS["fixed_2of32_byte"][2],
            "interface_rank_offset": FIXED_CODE_SPECS["fixed_2of32_byte"][3],
            "interface_collision_policy": FAIL_CLOSED_COLLISION_POLICY,
            "interface_collision_gate": "fail_closed_injective_recurrent_byte_code",
            "interface_uses_arbitrary_token_id_code": 0,
            "interface_uses_dense_byte_histogram": 0,
            "interface_uses_patch_hash": 0,
            "interface_executes_payload_snn_per_occurrence": 0,
            "interface_executes_streaming_snn_per_occurrence": 1,
            "interface_completion_spike_channel": 0,
            "interface_learning_rule": "fixed_sparse_streaming_reservoir",
            "interface_state_digest": hashlib.sha256(
                b"".join(
                    tensor.detach().cpu().contiguous().numpy().tobytes()
                    for tensor in (
                        self.input_weight,
                        self.recurrent_weight,
                        self.hidden_threshold,
                        self.fixed_codes,
                    )
                )
            ).hexdigest(),
            "homeostasis_fit_corpus_sha256": "not_applicable_training_free_reservoir",
            "interface_fixed_code_active_bits": 2,
            "interface_codec_free_raw_byte_map": 1,
            "interface_sparse_synapse_count": sparse_synapses,
            "interface_neuron_state_elements": neuron_state_elements,
            "interface_packed_binary_codebook_bytes": fixed_packed_bytes,
            "interface_state_storage_reference_bytes": reference_state_bytes,
            "interface_deployment_static_state_bytes": deployment_state_bytes,
            "interface_deployment_static_state_scope": (
                "six_bytes_per_active_synapse_plus_fp32_threshold_state_plus_"
                "packed_injective_suffix"
            ),
            "frontend_synaptic_lif_and_threshold_count_complete": 1,
            "frontend_operation_count_complete": 1,
            "frontend_memory_access_count_complete": 0,
            "frontend_hardware_energy_estimate_ready": 0,
            "frontend_hardware_energy_claim_permitted": 0,
        })
        return audit

    def last_operation_stats(self) -> Dict[str, float | int | str]:
        return dict(self._last_stats)


def build_spike_input_interface(
    vocab_size: int,
    mode: str,
    *,
    temperature: float = 1.0,
    target_rate: float = 0.125,
    payloads_by_id: Sequence[bytes] | None = None,
    homeostatic_fit_ids: Sequence[int] | None = None,
    fit_corpus_sha256: str | None = None,
    seed: int = 1,
) -> SpikeInputInterface | SpikePatchPayloadInterface | InjectiveRecurrentByteSpikeInterface:
    if mode == INJECTIVE_RECURRENT_BYTE_MODE:
        if payloads_by_id is None or homeostatic_fit_ids is None:
            raise ValueError(f"Interface {mode} requires raw-byte payloads and fit IDs")
        return InjectiveRecurrentByteSpikeInterface(
            payloads_by_id,
            seed=seed,
            target_rate=target_rate,
            homeostatic_fit_ids=homeostatic_fit_ids,
            fit_corpus_sha256=fit_corpus_sha256,
        )
    if mode in SPIKEPATCH_INTERFACE_MODES:
        if payloads_by_id is None:
            raise ValueError(
                f"Interface {mode} requires exact raw-byte payloads for every ID"
            )
        if len(payloads_by_id) != vocab_size:
            raise ValueError("Spike-patch payload table does not match vocab_size")
        return SpikePatchPayloadInterface(
            payloads_by_id,
            mode=mode,
            seed=seed,
            target_rate=target_rate,
            homeostatic_fit_ids=homeostatic_fit_ids,
            fit_corpus_sha256=fit_corpus_sha256,
        )
    return SpikeInputInterface(
        vocab_size=vocab_size,
        mode=mode,
        temperature=temperature,
        target_rate=target_rate,
    )
