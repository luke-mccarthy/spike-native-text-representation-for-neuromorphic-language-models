"""Codec-free spike-native patch interface around the SpikeGPT reference core.

The expensive global recurrent core advances once per patch, not once per
byte.  A deterministic positional 2-of-32 byte code makes every bounded patch
injective without a learned vocabulary, hash table or fallback codec.  An
optional local recurrent LIF layer adds hierarchical context while retaining
the injective suffix.  A lightweight recurrent LIF decoder predicts the bytes
inside the next patch autoregressively.

This is deliberately a *local interface module* around an unchanged SpikeGPT
core.  It is more than an input-only tokenizer: reducing global steps while
retaining exact byte-level likelihood requires a local output decoder.  All
communication into the global core and from its final spiking path into the
local decoder is binary.  The final 256-way likelihood readout is dense, just
as the ordinary SpikeGPT vocabulary readout is dense.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from src.models.spikegpt_reference import (
    SpikeGPTReferenceLM,
    atan_spike,
)
from src.tokenizers.spiking_blt_frontend import injective_byte_spike_codebook


PATCH_LEDGER_VERSION = "codec_free_spike_native_patch_interface_v1"
PATCH_ARCHITECTURES = {
    "fixed_patch4_positional": {
        "patch_mode": "fixed",
        "max_patch_bytes": 4,
        "hierarchical": False,
    },
    "fixed_patch8_hierarchical_lif": {
        "patch_mode": "fixed",
        "max_patch_bytes": 8,
        "hierarchical": True,
    },
    "joint_patch8_hierarchical_lif": {
        "patch_mode": "joint_snn_homeostatic",
        "max_patch_bytes": 8,
        "hierarchical": True,
    },
}


def fixed_width_patches(raw: bytes, patch_bytes: int) -> List[bytes]:
    if patch_bytes <= 0:
        raise ValueError("patch_bytes must be positive")
    if not raw:
        raise ValueError("cannot patch an empty byte stream")
    patches = [
        raw[start : start + patch_bytes]
        for start in range(0, len(raw), patch_bytes)
    ]
    if b"".join(patches) != raw:
        raise AssertionError("fixed patching changed the raw byte stream")
    return patches


def validate_patch_cover(
    patches: Sequence[bytes],
    raw: bytes,
    *,
    max_patch_bytes: int,
) -> None:
    if not patches or any(not patch for patch in patches):
        raise AssertionError("patch sequence contains an empty data patch")
    if any(len(patch) > max_patch_bytes for patch in patches):
        raise AssertionError("patch exceeds the declared maximum length")
    if b"".join(patches) != raw:
        raise AssertionError("patch sequence does not exactly cover raw bytes")


@dataclass(frozen=True)
class PatchSequence:
    patches: Tuple[bytes, ...]
    max_patch_bytes: int
    raw_sha256: str

    @property
    def raw_bytes(self) -> int:
        return sum(len(patch) for patch in self.patches)

    @property
    def mean_patch_bytes(self) -> float:
        return self.raw_bytes / len(self.patches)

    def source_and_targets(
        self,
        starts: Sequence[int],
        length: int,
        *,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return previous-patch inputs and exact next-patch byte targets."""
        if length <= 0:
            raise ValueError("block length must be positive")
        batch = len(starts)
        source = torch.zeros(
            batch,
            length,
            self.max_patch_bytes,
            dtype=torch.long,
            device=device,
        )
        source_mask = torch.zeros_like(source, dtype=torch.bool)
        target = torch.zeros_like(source)
        target_mask = torch.zeros_like(source_mask)
        for row, start in enumerate(starts):
            if start < 0 or start + length > len(self.patches):
                raise IndexError("patch block lies outside the sequence")
            for offset in range(length):
                target_patch = self.patches[start + offset]
                target[row, offset, : len(target_patch)] = torch.tensor(
                    list(target_patch), dtype=torch.long, device=device
                )
                target_mask[row, offset, : len(target_patch)] = True
                source_index = start + offset - 1
                if source_index >= 0:
                    source_patch = self.patches[source_index]
                    source[row, offset, : len(source_patch)] = torch.tensor(
                        list(source_patch), dtype=torch.long, device=device
                    )
                    source_mask[row, offset, : len(source_patch)] = True
        return source, source_mask, target, target_mask


def complete_patch_blocks(num_patches: int, patches_per_block: int) -> List[Tuple[int, int]]:
    if num_patches <= 0 or patches_per_block <= 0:
        raise ValueError("patch and block counts must be positive")
    return [
        (start, min(patches_per_block, num_patches - start))
        for start in range(0, num_patches, patches_per_block)
    ]


def _sparse_mask(
    rows: int,
    columns: int,
    fanout: int,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    if not 0 < fanout <= rows:
        raise ValueError("fanout must lie in [1, rows]")
    mask = torch.zeros(rows, columns)
    for source in range(columns):
        mask[torch.randperm(rows, generator=generator)[:fanout], source] = 1.0
    return mask


class PositionalPatchSpikeEncoder(nn.Module):
    """Injective bounded-patch spikes with an optional local LIF hierarchy."""

    def __init__(
        self,
        *,
        max_patch_bytes: int,
        hierarchical: bool,
        hidden_dim: int = 64,
        input_fanout: int = 8,
        recurrent_fanout: int = 4,
        seed: int = 1,
    ) -> None:
        super().__init__()
        if max_patch_bytes <= 1 or hidden_dim <= 0:
            raise ValueError("invalid patch encoder dimensions")
        self.max_patch_bytes = int(max_patch_bytes)
        self.hierarchical = bool(hierarchical)
        self.hidden_dim = int(hidden_dim)
        self.input_fanout = int(input_fanout)
        self.recurrent_fanout = int(recurrent_fanout)
        self.register_buffer(
            "byte_codebook",
            injective_byte_spike_codebook(32, 2),
        )
        if hierarchical:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(seed))
            input_mask = _sparse_mask(
                hidden_dim, 32, input_fanout, generator=generator
            )
            recurrent_mask = _sparse_mask(
                hidden_dim, hidden_dim, recurrent_fanout, generator=generator
            )
            input_weight = input_mask * (
                0.75 + 0.5 * torch.rand(hidden_dim, 32, generator=generator)
            )
            recurrent_sign = torch.where(
                torch.rand(hidden_dim, hidden_dim, generator=generator) < 0.8,
                torch.ones(hidden_dim, hidden_dim),
                -torch.ones(hidden_dim, hidden_dim),
            )
            recurrent_weight = (
                recurrent_mask
                * recurrent_sign
                * 0.15
                * (0.75 + 0.5 * torch.rand(
                    hidden_dim, hidden_dim, generator=generator
                ))
            )
            self.register_buffer("input_mask", input_mask)
            self.register_buffer("input_weight", input_weight)
            self.register_buffer("recurrent_mask", recurrent_mask)
            self.register_buffer("recurrent_weight", recurrent_weight)
        else:
            self.register_buffer("input_mask", torch.empty(0))
            self.register_buffer("input_weight", torch.empty(0))
            self.register_buffer("recurrent_mask", torch.empty(0))
            self.register_buffer("recurrent_weight", torch.empty(0))
        self._last_stats: Dict[str, float | int | str] = {}

    @property
    def output_dim(self) -> int:
        positional = self.max_patch_bytes * 32
        return positional + (self.hidden_dim if self.hierarchical else 0)

    def forward(
        self,
        patch_bytes: torch.Tensor,
        patch_mask: torch.Tensor,
    ) -> torch.Tensor:
        if (
            patch_bytes.ndim != 3
            or patch_mask.shape != patch_bytes.shape
            or patch_bytes.shape[-1] != self.max_patch_bytes
        ):
            raise ValueError("patch encoder expects aligned [batch, time, position]")
        if torch.any(patch_bytes < 0) or torch.any(patch_bytes > 255):
            raise ValueError("patch byte lies outside [0, 255]")
        mask = patch_mask.to(dtype=self.byte_codebook.dtype)
        byte_spikes = self.byte_codebook[patch_bytes] * mask.unsqueeze(-1)
        batch, steps, positions, _ = byte_spikes.shape
        positional = byte_spikes.reshape(batch, steps, positions * 32)
        raw_bytes = int(patch_mask.sum().item())
        hidden_events = 0
        recurrent_accumulations = 0
        if self.hierarchical:
            flattened = byte_spikes.reshape(batch * steps, positions, 32)
            flat_mask = patch_mask.reshape(batch * steps, positions)
            membrane = torch.zeros(
                batch * steps,
                self.hidden_dim,
                device=patch_bytes.device,
            )
            previous = torch.zeros_like(membrane)
            summary = torch.zeros_like(membrane)
            recurrent_fanout = self.recurrent_mask.sum(dim=0)
            for position in range(positions):
                active = flat_mask[:, position].to(membrane.dtype).unsqueeze(1)
                candidate = (
                    0.85 * membrane
                    + F.linear(
                        flattened[:, position],
                        self.input_weight * self.input_mask,
                    )
                    + F.linear(
                        previous,
                        self.recurrent_weight * self.recurrent_mask,
                    )
                )
                spikes = atan_spike(candidate - 0.55, alpha=2.0) * active
                membrane = active * candidate * (1.0 - spikes)
                summary = torch.maximum(summary, spikes)
                recurrent_accumulations += int(
                    (previous.detach() * recurrent_fanout).sum().item()
                )
                previous = spikes
            hidden_events = int(summary.detach().sum().item())
            output = torch.cat(
                (positional, summary.reshape(batch, steps, self.hidden_dim)),
                dim=-1,
            )
        else:
            output = positional
        if not torch.all((output.detach() == 0) | (output.detach() == 1)):
            raise AssertionError("patch encoder emitted a non-binary value")
        self._last_stats = {
            "frontend_operation_ledger_version": PATCH_LEDGER_VERSION,
            "encoder_raw_byte_events": raw_bytes * 2,
            "encoder_patch_output_events": int(output.detach().sum().item()),
            "encoder_hidden_events": hidden_events,
            "encoder_input_synaptic_accumulations": (
                raw_bytes * 2 * self.input_fanout if self.hierarchical else 0
            ),
            "encoder_recurrent_synaptic_accumulations": recurrent_accumulations,
            "encoder_lif_state_updates": (
                raw_bytes * self.hidden_dim if self.hierarchical else 0
            ),
            "encoder_threshold_reset_elements": (
                raw_bytes * self.hidden_dim * 2 if self.hierarchical else 0
            ),
            "encoder_dense_reference_macs": (
                batch
                * steps
                * positions
                * self.hidden_dim
                * (32 + self.hidden_dim)
                if self.hierarchical
                else 0
            ),
            "encoder_injective_positional_suffix": 1,
            "encoder_digital_tokenizer_present": 0,
            "encoder_digital_patch_codec_present": 0,
            "encoder_binary_communication": 1,
        }
        return output

    def last_operation_stats(self) -> Dict[str, float | int | str]:
        return dict(self._last_stats)


class LocalByteLIFDecoder(nn.Module):
    """Autoregressive local byte decoder driven by global binary spikes."""

    def __init__(
        self,
        *,
        context_dim: int,
        max_patch_bytes: int,
        hidden_dim: int = 64,
        input_fanout: int = 8,
        recurrent_fanout: int = 4,
        seed: int = 1,
    ) -> None:
        super().__init__()
        if context_dim <= 0 or max_patch_bytes <= 1 or hidden_dim <= 0:
            raise ValueError("invalid local decoder dimensions")
        self.context_dim = int(context_dim)
        self.max_patch_bytes = int(max_patch_bytes)
        self.hidden_dim = int(hidden_dim)
        self.input_fanout = int(input_fanout)
        self.recurrent_fanout = int(recurrent_fanout)
        self.register_buffer(
            "byte_codebook",
            injective_byte_spike_codebook(32, 2),
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) + 7919)
        input_dim = context_dim + 32 + max_patch_bytes
        input_mask = _sparse_mask(
            hidden_dim, input_dim, input_fanout, generator=generator
        )
        recurrent_mask = _sparse_mask(
            hidden_dim, hidden_dim, recurrent_fanout, generator=generator
        )
        self.register_buffer("input_mask", input_mask)
        self.register_buffer("recurrent_mask", recurrent_mask)
        self.input_weight = nn.Parameter(
            input_mask
            * (
                0.25
                + 0.10
                * torch.rand(hidden_dim, input_dim, generator=generator)
            )
        )
        recurrent_sign = torch.where(
            torch.rand(hidden_dim, hidden_dim, generator=generator) < 0.8,
            torch.ones(hidden_dim, hidden_dim),
            -torch.ones(hidden_dim, hidden_dim),
        )
        self.recurrent_weight = nn.Parameter(
            recurrent_mask
            * recurrent_sign
            * 0.10
            * (
                0.75
                + 0.5
                * torch.rand(hidden_dim, hidden_dim, generator=generator)
            )
        )
        self.readout = nn.Linear(hidden_dim, 256, bias=False)
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
        if not torch.all(
            (context_spikes.detach() == 0) | (context_spikes.detach() == 1)
        ):
            raise ValueError("local decoder context must be binary spikes")
        batch, steps, _ = context_spikes.shape
        flattened_context = context_spikes.reshape(
            batch * steps, self.context_dim
        )
        flattened_targets = target_bytes.reshape(
            batch * steps, self.max_patch_bytes
        )
        flattened_mask = target_mask.reshape(
            batch * steps, self.max_patch_bytes
        )
        membrane = torch.zeros(
            batch * steps, self.hidden_dim, device=context_spikes.device
        )
        previous_hidden = torch.zeros_like(membrane)
        previous_byte = torch.zeros(
            batch * steps, 32, device=context_spikes.device
        )
        logits: List[torch.Tensor] = []
        hidden_events = 0
        recurrent_accumulations = 0
        readout_accumulations = 0
        recurrent_fanout = self.recurrent_mask.sum(dim=0)
        for position in range(self.max_patch_bytes):
            active = flattened_mask[:, position].to(
                flattened_context.dtype
            ).unsqueeze(1)
            position_spikes = torch.zeros(
                batch * steps,
                self.max_patch_bytes,
                device=context_spikes.device,
            )
            position_spikes[:, position] = active[:, 0]
            decoder_input = torch.cat(
                (flattened_context * active, previous_byte, position_spikes),
                dim=-1,
            )
            candidate = (
                0.85 * membrane
                + F.linear(
                    decoder_input,
                    self.input_weight * self.input_mask,
                )
                + F.linear(
                    previous_hidden,
                    self.recurrent_weight * self.recurrent_mask,
                )
            )
            hidden = atan_spike(candidate - 0.75, alpha=2.0)
            hidden = hidden * active
            membrane = candidate * (1.0 - hidden) * active
            logits.append(self.readout(hidden))
            hidden_events += int(hidden.detach().sum().item())
            recurrent_accumulations += int(
                (previous_hidden.detach() * recurrent_fanout).sum().item()
            )
            readout_accumulations += int(hidden.detach().sum().item()) * 256
            literal = flattened_targets[:, position]
            previous_byte = self.byte_codebook[literal] * active
            previous_hidden = hidden
        output = torch.stack(logits, dim=1).reshape(
            batch, steps, self.max_patch_bytes, 256
        )
        active_bytes = int(target_mask.sum().item())
        context_events = int(context_spikes.detach().sum().item())
        patch_lengths = flattened_mask.sum(dim=1).to(context_spikes.dtype)
        repeated_context_events = int(
            (
                flattened_context.detach().sum(dim=1)
                * patch_lengths
            ).sum().item()
        )
        teacher_events = max(active_bytes - batch * steps, 0) * 2
        position_events = active_bytes
        decoder_input_events = (
            repeated_context_events
            + teacher_events
            + position_events
        )
        self._last_stats = {
            "frontend_operation_ledger_version": PATCH_LEDGER_VERSION,
            "decoder_context_spike_events": context_events,
            "decoder_teacher_byte_spike_events": teacher_events,
            "decoder_position_spike_events": position_events,
            "decoder_hidden_spike_events": hidden_events,
            "decoder_input_synaptic_accumulations": (
                decoder_input_events * self.input_fanout
            ),
            "decoder_recurrent_synaptic_accumulations": recurrent_accumulations,
            "decoder_readout_synaptic_accumulations": readout_accumulations,
            "decoder_lif_state_updates": active_bytes * self.hidden_dim,
            "decoder_threshold_reset_elements": active_bytes * self.hidden_dim * 2,
            "decoder_dense_reference_readout_macs": (
                active_bytes * self.hidden_dim * 256
            ),
            "decoder_likelihood_exponential_elements": active_bytes * 256,
            "decoder_likelihood_reduction_add_elements": active_bytes * 255,
            "decoder_likelihood_log_elements": active_bytes,
            "decoder_likelihood_target_subtraction_elements": active_bytes,
            "decoder_likelihood_normalization_operations_included": 1,
            "decoder_binary_internal_communication": 1,
            "decoder_dense_likelihood_readout_present": 1,
            "decoder_autoregressive_teacher_forcing_is_causal": 1,
        }
        return output

    def last_operation_stats(self) -> Dict[str, float | int | str]:
        return dict(self._last_stats)


class SpikeNativePatchLanguageModel(nn.Module):
    """Unchanged global SpikeGPT blocks plus codec-free local patch modules."""

    def __init__(
        self,
        *,
        max_patch_bytes: int,
        hierarchical_encoder: bool,
        hidden_dim: int = 64,
        num_layers: int = 2,
        seed: int = 1,
    ) -> None:
        super().__init__()
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
        # The global model no longer predicts a patch-vocabulary ID.  Removing
        # this unused head avoids hiding a vocabulary-sized parameter/compute
        # cost; byte likelihoods are produced by the local decoder below.
        self.backbone.final_ln = nn.Identity()
        self.backbone.readout = nn.Identity()
        self.decoder = LocalByteLIFDecoder(
            context_dim=hidden_dim,
            max_patch_bytes=max_patch_bytes,
            hidden_dim=hidden_dim,
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
        _, context_spikes, backbone_stats = self.backbone.forward_features(
            inputs_spikes=input_spikes
        )
        logits = self.decoder(context_spikes, target_bytes, target_mask)
        stats: Dict[str, object] = {
            "frontend_operation_ledger_version": PATCH_LEDGER_VERSION,
            "global_patch_steps": int(source_bytes.shape[0] * source_bytes.shape[1]),
            "global_input_dimension": self.encoder.output_dim,
            "global_input_spike_events": int(input_spikes.detach().sum().item()),
            "fully_spike_communicating_patch_encoder": 1,
            "fully_spike_communicating_global_to_local_decoder": 1,
            "digital_tokenizer_present": 0,
            "digital_patch_codec_present": 0,
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
        return logits, stats


def merge_additive_stats(
    totals: Dict[str, float],
    stats: Mapping[str, object],
) -> None:
    for key, value in stats.items():
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and key
            not in {
                "fully_spike_communicating_patch_encoder",
                "fully_spike_communicating_global_to_local_decoder",
                "digital_tokenizer_present",
                "digital_patch_codec_present",
            }
        ):
            totals[key] = totals.get(key, 0.0) + float(value)
