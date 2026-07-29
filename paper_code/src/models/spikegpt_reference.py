"""Small, CPU-portable reference implementation of the SpikeGPT block.

The architecture follows the archived SpikeGPT implementation: binary input
spikes, RWKV time mixing with the WKV recurrence, LIF neurons after both the
time- and channel-mixing paths, squared-ReLU channel mixing, residual paths,
and a dense vocabulary readout.  The original CUDA WKV kernel and CuPy LIF
kernel are replaced by differentiable PyTorch reference recurrences so small
controlled studies can run on CPU.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
from torch import nn


class _ATanSpike(torch.autograd.Function):
    """Heaviside forward with the SpikingJelly ATan(alpha=2) surrogate."""

    @staticmethod
    def forward(ctx, value: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.save_for_backward(value)
        ctx.alpha = float(alpha)
        return (value >= 0).to(value.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (value,) = ctx.saved_tensors
        alpha = ctx.alpha
        scale = math.pi * alpha / 2.0
        gradient = alpha / 2.0 / (1.0 + (scale * value).square())
        return grad_output * gradient, None


def atan_spike(value: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
    return _ATanSpike.apply(value, float(alpha))


class SequenceLIF(nn.Module):
    """Multi-step LIF compatible with SpikeGPT's tau=2, hard-reset use."""

    def __init__(
        self,
        *,
        tau: float = 2.0,
        threshold: float = 1.0,
        surrogate_alpha: float = 2.0,
    ) -> None:
        super().__init__()
        if tau <= 1.0:
            raise ValueError("tau must be greater than one")
        self.tau = float(tau)
        self.threshold = float(threshold)
        self.surrogate_alpha = float(surrogate_alpha)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3:
            raise ValueError("SequenceLIF expects [batch, time, channels]")
        membrane = torch.zeros_like(values[:, 0, :])
        outputs = []
        for value in values.unbind(dim=1):
            # SpikingJelly LIF with decay_input=True and v_reset=0.
            membrane = membrane + (value - membrane) / self.tau
            spike = atan_spike(
                membrane - self.threshold,
                alpha=self.surrogate_alpha,
            )
            membrane = membrane * (1.0 - spike)
            outputs.append(spike)
        return torch.stack(outputs, dim=1)


def _time_shift(values: torch.Tensor) -> torch.Tensor:
    return torch.cat((torch.zeros_like(values[:, :1, :]), values[:, :-1, :]), dim=1)


def reference_wkv(
    time_decay: torch.Tensor,
    time_first: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    """Stable recurrent form of the causal RWKV-v2 WKV operator."""
    if keys.shape != values.shape or keys.ndim != 3:
        raise ValueError("keys and values must share [batch, time, channels]")
    batch, _, channels = keys.shape
    if time_decay.shape != (channels,) or time_first.shape != (channels,):
        raise ValueError("WKV parameter dimensions do not match channels")

    decay = -torch.exp(time_decay.float()).to(keys.dtype).view(1, channels)
    first = time_first.to(keys.dtype).view(1, channels)
    aa = torch.zeros(batch, channels, dtype=keys.dtype, device=keys.device)
    bb = torch.zeros_like(aa)
    pp = torch.full_like(aa, -1.0e30)
    outputs = []
    for key, value in zip(keys.unbind(dim=1), values.unbind(dim=1)):
        current = first + key
        maximum = torch.maximum(pp, current)
        previous_weight = torch.exp(pp - maximum)
        current_weight = torch.exp(current - maximum)
        numerator = previous_weight * aa + current_weight * value
        denominator = previous_weight * bb + current_weight
        outputs.append(numerator / denominator.clamp_min(1.0e-30))

        decayed = pp + decay
        maximum = torch.maximum(decayed, key)
        previous_weight = torch.exp(decayed - maximum)
        current_weight = torch.exp(key - maximum)
        aa = previous_weight * aa + current_weight * value
        bb = previous_weight * bb + current_weight
        pp = maximum
    return torch.stack(outputs, dim=1)


class SpikeGPTTimeMix(nn.Module):
    def __init__(self, hidden_dim: int, layer_id: int, num_layers: int) -> None:
        super().__init__()
        if hidden_dim < 2 or num_layers < 1:
            raise ValueError("invalid SpikeGPT dimensions")
        denominator = max(num_layers - 1, 1)
        ratio_0_to_1 = layer_id / denominator
        ratio_1_to_almost0 = 1.0 - layer_id / num_layers
        channel_ratio = torch.arange(hidden_dim, dtype=torch.float32) / hidden_dim

        exponent = 0.7 + 1.3 * ratio_0_to_1
        decay = -5.0 + 8.0 * torch.pow(
            torch.arange(hidden_dim, dtype=torch.float32) / (hidden_dim - 1),
            exponent,
        )
        zigzag = torch.tensor(
            [((index + 1) % 3 - 1) * 0.5 for index in range(hidden_dim)],
            dtype=torch.float32,
        )
        self.time_decay = nn.Parameter(decay)
        self.time_first = nn.Parameter(torch.full((hidden_dim,), math.log(0.3)) + zigzag)
        self.time_mix_k = nn.Parameter(channel_ratio.pow(ratio_1_to_almost0).view(1, 1, -1))
        self.time_mix_v = nn.Parameter(
            (channel_ratio.pow(ratio_1_to_almost0) + 0.3 * ratio_0_to_1).view(1, 1, -1)
        )
        self.time_mix_r = nn.Parameter(
            channel_ratio.pow(0.5 * ratio_1_to_almost0).view(1, 1, -1)
        )

        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.receptance = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.zeros_(self.key.weight)
        nn.init.zeros_(self.receptance.weight)
        nn.init.zeros_(self.output.weight)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        shifted = _time_shift(values)
        mixed_k = values * self.time_mix_k + shifted * (1.0 - self.time_mix_k)
        mixed_v = values * self.time_mix_v + shifted * (1.0 - self.time_mix_v)
        mixed_r = values * self.time_mix_r + shifted * (1.0 - self.time_mix_r)
        key = self.key(mixed_k)
        value = self.value(mixed_v)
        receptance = torch.sigmoid(self.receptance(mixed_r))
        return self.output(
            receptance * reference_wkv(
                self.time_decay,
                self.time_first,
                key,
                value,
            )
        )


class SpikeGPTChannelMix(nn.Module):
    def __init__(self, hidden_dim: int, layer_id: int, num_layers: int) -> None:
        super().__init__()
        ratio_1_to_almost0 = 1.0 - layer_id / num_layers
        channel_ratio = torch.arange(hidden_dim, dtype=torch.float32) / hidden_dim
        initial = channel_ratio.pow(ratio_1_to_almost0).view(1, 1, -1)
        self.time_mix_k = nn.Parameter(initial.clone())
        self.time_mix_r = nn.Parameter(initial.clone())
        self.key = nn.Linear(hidden_dim, 4 * hidden_dim, bias=False)
        self.receptance = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(4 * hidden_dim, hidden_dim, bias=False)
        nn.init.zeros_(self.receptance.weight)
        nn.init.zeros_(self.value.weight)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        shifted = _time_shift(values)
        mixed_k = values * self.time_mix_k + shifted * (1.0 - self.time_mix_k)
        mixed_r = values * self.time_mix_r + shifted * (1.0 - self.time_mix_r)
        key = torch.relu(self.key(mixed_k)).square()
        return torch.sigmoid(self.receptance(mixed_r)) * self.value(key)


class SpikeGPTBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        layer_id: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.ln0 = nn.LayerNorm(hidden_dim) if layer_id == 0 else nn.Identity()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.time_mix = SpikeGPTTimeMix(hidden_dim, layer_id, num_layers)
        self.channel_mix = SpikeGPTChannelMix(hidden_dim, layer_id, num_layers)
        self.lif1 = SequenceLIF(tau=2.0, threshold=1.0, surrogate_alpha=2.0)
        self.lif2 = SequenceLIF(tau=2.0, threshold=1.0, surrogate_alpha=2.0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        values = self.ln0(values)
        time_spikes = self.lif1(self.time_mix(self.ln1(values)))
        values = values + time_spikes
        channel_spikes = self.lif2(self.channel_mix(self.ln2(values)))
        # Keep the differentiable binary output of the final spiking path
        # available to local codec-free decoders.  This does not change the
        # ordinary SpikeGPT forward result or its dense residual stream.
        self.last_channel_spikes = channel_spikes
        values = self.dropout(values + channel_spikes)
        return values, {
            "time_mix_spikes": float(time_spikes.detach().sum().item()),
            "channel_mix_spikes": float(channel_spikes.detach().sum().item()),
            "hidden_spikes": float(
                time_spikes.detach().sum().item()
                + channel_spikes.detach().sum().item()
            ),
            "hidden_neuron_updates": float(time_spikes.numel() + channel_spikes.numel()),
        }


class SpikeGPTReferenceLM(nn.Module):
    """Reference-faithful SpikeGPT core accepting a direct spike interface."""

    def __init__(
        self,
        *,
        vocab_size: int,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.03,
    ) -> None:
        super().__init__()
        if vocab_size <= 1 or input_dim <= 0 or hidden_dim < 2 or num_layers < 1:
            raise ValueError("invalid SpikeGPT reference configuration")
        self.vocab_size = int(vocab_size)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        if input_dim == hidden_dim:
            self.input_projection = nn.Identity()
            self.input_projection_emits_spikes = False
        else:
            self.input_projection = nn.Linear(input_dim, hidden_dim, bias=False)
            self.input_projection_emits_spikes = True
        self.blocks = nn.ModuleList(
            [
                SpikeGPTBlock(hidden_dim, layer, num_layers, dropout)
                for layer in range(num_layers)
            ]
        )
        self.final_ln = nn.LayerNorm(hidden_dim)
        self.readout = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward_features(self, *, inputs_spikes: torch.Tensor):
        """Return the dense residual state and last binary channel spikes.

        The ordinary language-model head still consumes the dense residual
        state.  A patch-local decoder may instead consume
        ``last_channel_spikes`` so communication across the global/local
        boundary remains binary without changing the SpikeGPT blocks.
        """
        if inputs_spikes.ndim != 3 or inputs_spikes.shape[-1] != self.input_dim:
            raise ValueError("inputs_spikes must be [batch, time, input_dim]")
        if not torch.all((inputs_spikes == 0) | (inputs_spikes == 1)):
            raise ValueError("SpikeGPT reference input must be binary")
        values = self.input_projection(inputs_spikes.float())
        adapter_spikes = 0.0
        if self.input_projection_emits_spikes:
            values = atan_spike(values, alpha=2.0)
            adapter_spikes = float(values.detach().sum().item())

        input_spikes = float(inputs_spikes.detach().sum().item())
        hidden_spikes = 0.0
        hidden_updates = 0.0
        stats: Dict[str, float] = {}
        for layer, block in enumerate(self.blocks):
            values, block_stats = block(values)
            hidden_spikes += block_stats["hidden_spikes"]
            hidden_updates += block_stats["hidden_neuron_updates"]
            stats[f"layer{layer}_time_mix_spikes"] = block_stats["time_mix_spikes"]
            stats[f"layer{layer}_channel_mix_spikes"] = block_stats["channel_mix_spikes"]
            stats[f"layer{layer}_hidden_spikes"] = block_stats["hidden_spikes"]
        stats.update({
            "input_spike_rate": input_spikes / max(inputs_spikes.numel(), 1),
            "total_input_spikes": input_spikes,
            "input_adapter_spikes": adapter_spikes,
            "hidden_spike_rate": hidden_spikes / max(hidden_updates, 1.0),
            "total_hidden_spikes": hidden_spikes,
            "actual_lif_neuron_updates": hidden_updates,
            "actual_compute_steps": float(inputs_spikes.shape[0] * inputs_spikes.shape[1]),
        })
        last_channel_spikes = self.blocks[-1].last_channel_spikes
        if not torch.all(
            (last_channel_spikes == 0) | (last_channel_spikes == 1)
        ):
            raise AssertionError("SpikeGPT block exposed a non-binary spike tensor")
        return values, last_channel_spikes, stats

    def forward(self, *, inputs_spikes: torch.Tensor):
        values, _, stats = self.forward_features(inputs_spikes=inputs_spikes)
        logits = self.readout(self.final_ln(values))
        return logits, stats


def build_spikegpt_reference_lm(
    *,
    vocab_size: int,
    input_dim: int,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.03,
) -> SpikeGPTReferenceLM:
    return SpikeGPTReferenceLM(
        vocab_size=vocab_size,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
