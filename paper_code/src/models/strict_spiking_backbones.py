from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, membrane_minus_threshold: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(membrane_minus_threshold)
        return (membrane_minus_threshold > 0).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor]:
        (x,) = ctx.saved_tensors
        width = 1.0
        surrogate = torch.clamp(1.0 - x.abs() / width, min=0.0)
        return grad_output * surrogate


def spike_fn(x: torch.Tensor) -> torch.Tensor:
    return SurrogateSpike.apply(x)


@dataclass
class StrictSpikeRWKVV2Config:
    vocab_size: int
    input_dim: int = 32
    hidden_dim: int = 64
    num_layers: int = 2
    beta: float = 0.95
    threshold: float = 1.0
    dropout: float = 0.0


class StrictSpikeRWKVV2LM(nn.Module):
    """
    Strict spike-communicating SRWKV-style small language model.

    Strict behaviour:
      - input is spike/event features [B, T, input_dim]
      - each recurrent layer has continuous membrane state
      - each recurrent layer emits binary threshold spikes
      - recurrent layers pass only binary spikes to the next layer
      - recurrent feedback uses previous spikes
      - optional compute_steps repeats spiking recurrent updates per patch
      - dense vocabulary readout occurs only at final LM output

    Learned synaptic weights are retained. Dense hidden activations are not
    communicated between recurrent layers.
    """

    def __init__(self, config: StrictSpikeRWKVV2Config):
        super().__init__()
        self.config = config

        self.input_syn = nn.Linear(config.input_dim, config.hidden_dim, bias=False)

        self.in_syn = nn.ModuleList()
        self.rec_syn = nn.ModuleList()
        for _ in range(config.num_layers):
            self.in_syn.append(nn.Linear(config.hidden_dim, config.hidden_dim, bias=False))
            self.rec_syn.append(nn.Linear(config.hidden_dim, config.hidden_dim, bias=False))

        self.dropout = nn.Dropout(config.dropout)
        self.readout = nn.Linear(config.hidden_dim, config.vocab_size)

    def forward(
        self,
        inputs_spikes: torch.Tensor,
        compute_steps: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if inputs_spikes is None:
            raise ValueError("StrictSpikeRWKVV2LM requires inputs_spikes=[B,T,D].")

        batch_size, seq_len, _ = inputs_spikes.shape
        device = inputs_spikes.device
        dtype = inputs_spikes.dtype

        if compute_steps is None:
            compute_steps = torch.ones(batch_size, seq_len, device=device, dtype=torch.long)
        else:
            compute_steps = compute_steps.to(device=device, dtype=torch.long).clamp(min=1)

        mem = [
            torch.zeros(batch_size, self.config.hidden_dim, device=device, dtype=dtype)
            for _ in range(self.config.num_layers)
        ]
        prev_spikes = [
            torch.zeros(batch_size, self.config.hidden_dim, device=device, dtype=dtype)
            for _ in range(self.config.num_layers)
        ]

        logits_steps = []

        total_input_spikes = float(inputs_spikes.detach().sum().item())
        total_hidden_spikes = 0.0
        layer_spike_totals = [0.0 for _ in range(self.config.num_layers)]
        actual_compute_steps = int(compute_steps.detach().sum().item())

        max_compute = int(compute_steps.max().item()) if compute_steps.numel() else 1

        for t in range(seq_len):
            final_h_spike = None

            for micro_step in range(max_compute):
                active = (compute_steps[:, t] > micro_step).float().view(batch_size, 1)

                input_current = self.input_syn(inputs_spikes[:, t, :])
                h_spike = spike_fn(input_current - self.config.threshold) * active

                for layer in range(self.config.num_layers):
                    current = self.in_syn[layer](h_spike) + self.rec_syn[layer](prev_spikes[layer])
                    proposed_mem = self.config.beta * mem[layer] + current

                    mem[layer] = active * proposed_mem + (1.0 - active) * mem[layer]

                    out_spike = spike_fn(mem[layer] - self.config.threshold) * active
                    mem[layer] = mem[layer] - out_spike * self.config.threshold

                    out_spike = self.dropout(out_spike)
                    prev_spikes[layer] = active * out_spike + (1.0 - active) * prev_spikes[layer]
                    h_spike = out_spike

                    spike_count = float(out_spike.detach().sum().item())
                    total_hidden_spikes += spike_count
                    layer_spike_totals[layer] += spike_count

                final_h_spike = h_spike if final_h_spike is None else torch.where(active.bool(), h_spike, final_h_spike)

            if final_h_spike is None:
                final_h_spike = torch.zeros(batch_size, self.config.hidden_dim, device=device, dtype=dtype)

            logits_steps.append(self.readout(final_h_spike))

        logits = torch.stack(logits_steps, dim=1)

        denom_input = max(inputs_spikes.numel(), 1)
        denom_hidden = max(batch_size * seq_len * self.config.hidden_dim * self.config.num_layers, 1)

        stats: Dict[str, float] = {
            "input_spike_rate": float(total_input_spikes / denom_input),
            "total_input_spikes": float(total_input_spikes),
            "hidden_spike_rate": float(total_hidden_spikes / denom_hidden),
            "total_hidden_spikes": float(total_hidden_spikes),
            "spikes_per_token": float((total_input_spikes + total_hidden_spikes) / max(batch_size * seq_len, 1)),
            "actual_compute_steps": float(actual_compute_steps),
            "mean_actual_compute_steps": float(actual_compute_steps / max(batch_size * seq_len, 1)),
        }

        for layer, total in enumerate(layer_spike_totals):
            stats[f"layer{layer + 1}_hidden_spikes"] = float(total)
            stats[f"layer{layer + 1}_hidden_spike_rate"] = float(
                total / max(batch_size * seq_len * self.config.hidden_dim, 1)
            )

        return logits, stats


def build_strict_spike_rwkv_lm(
    vocab_size: int,
    input_dim: int = 32,
    hidden_dim: int = 64,
    num_layers: int = 2,
    beta: float = 0.95,
    threshold: float = 1.0,
    dropout: float = 0.0,
) -> StrictSpikeRWKVV2LM:
    return StrictSpikeRWKVV2LM(
        StrictSpikeRWKVV2Config(
            vocab_size=vocab_size,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            beta=beta,
            threshold=threshold,
            dropout=dropout,
        )
    )
