"""Corrected strict-TTFS model ledger for the r2 calibration workflow.

The v6 neural dynamics are retained unchanged.  This additive wrapper corrects
only the operation-ledger fields required by the scalable patch projection:
strict output heads now report the per-byte log and target-subtraction elements
used by numerical likelihood evaluation.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn

from src.models.spike_native_patch_lm_v2 import LocalByteLIFDecoderV2
from src.models.spike_native_patch_lm_v6 import (
    OUTPUT_HEADS,
    PATCH_ARCHITECTURES,
    LocalByteLIFDecoderStrictTTFS,
    SpikeNativePatchLanguageModelV6,
)


PATCH_LEDGER_V7_VERSION = "codec_free_patch_v7_strict_ttfs_corrected_ledger"


class LocalByteLIFDecoderStrictTTFSV7(LocalByteLIFDecoderStrictTTFS):
    """v6 strict decoder with complete likelihood-operation accounting."""

    def forward(
        self,
        context_spikes: torch.Tensor,
        target_bytes: torch.Tensor,
        target_mask: torch.Tensor,
        *,
        feedback_mode: str = "teacher",
    ) -> torch.Tensor:
        scores = super().forward(
            context_spikes,
            target_bytes,
            target_mask,
            feedback_mode=feedback_mode,
        )
        active_bytes = int(target_mask.sum().item())
        self._last_stats["frontend_operation_ledger_version"] = PATCH_LEDGER_V7_VERSION
        self._last_stats["decoder_likelihood_log_elements"] = active_bytes
        self._last_stats["decoder_likelihood_target_subtraction_elements"] = active_bytes
        self._last_stats["decoder_likelihood_normalization_operations_included"] = 1
        return scores


class SpikeNativePatchLanguageModelV7(SpikeNativePatchLanguageModelV6):
    """v6 neural model with the corrected strict decoder ledger."""

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
        super().__init__(
            max_patch_bytes=max_patch_bytes,
            hierarchical_encoder=hierarchical_encoder,
            output_head=output_head,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            decoder_hidden_dim=decoder_hidden_dim,
            decoder_num_layers=decoder_num_layers,
            output_microsteps=output_microsteps,
            output_beta=output_beta,
            output_threshold=output_threshold,
            output_weight_scale=output_weight_scale,
            output_clock_end=output_clock_end,
            output_latency_tau=output_latency_tau,
            output_score_scale=output_score_scale,
            output_surrogate_temperature=output_surrogate_temperature,
            seed=seed,
        )
        if output_head == "dense_reference":
            if not isinstance(self.decoder, LocalByteLIFDecoderV2):
                raise AssertionError("dense reference decoder construction changed")
        else:
            self.decoder = LocalByteLIFDecoderStrictTTFSV7(
                context_dim=hidden_dim,
                max_patch_bytes=max_patch_bytes,
                output_head=output_head,
                decoder_hidden_dim=decoder_hidden_dim,
                decoder_num_layers=decoder_num_layers,
                output_microsteps=output_microsteps,
                output_beta=output_beta,
                output_threshold=output_threshold,
                output_weight_scale=output_weight_scale,
                output_clock_end=output_clock_end,
                output_latency_tau=output_latency_tau,
                output_score_scale=output_score_scale,
                output_surrogate_temperature=output_surrogate_temperature,
                seed=seed,
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
        stats["frontend_operation_ledger_version"] = PATCH_LEDGER_V7_VERSION
        return scores, stats


__all__ = [
    "OUTPUT_HEADS",
    "PATCH_ARCHITECTURES",
    "PATCH_LEDGER_V7_VERSION",
    "LocalByteLIFDecoderStrictTTFSV7",
    "SpikeNativePatchLanguageModelV7",
]
