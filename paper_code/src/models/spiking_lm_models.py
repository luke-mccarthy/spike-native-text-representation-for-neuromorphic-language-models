"""
Reusable spiking language model architectures.

These models are intentionally tokenizer-agnostic. They accept either:
  - input_ids: integer symbol IDs, e.g. char/byte/BPE/Unigram/spike-native IDs
  - inputs_embeds: precomputed embeddings, e.g. future BLT-style patch embeddings

The goal is to keep the model core compatible with future front ends:
  - fixed vocabulary tokenizers
  - byte/patch tokenizers
  - spike-native tokenizer/front-end modules
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

try:
    import snntorch as snn
except ImportError as exc:
    raise ImportError(
        "snntorch is required for spiking_lm_models.py. "
        "Activate /scratch/user/lukemccarthy/conda-envs/spikegpt-gpu-env."
    ) from exc


@dataclass
class SpikeStats:
    input_spike_rate: float
    hidden_spike_rate: float
    total_hidden_spikes: float
    total_input_spikes: float


def _rate_from_tensor(x: torch.Tensor) -> float:
    if x.numel() == 0:
        return 0.0
    return float(x.detach().float().mean().item())


class SpikeInputEncoder(nn.Module):
    """
    Token/embedding-to-spike adapter.

    This is deliberately simple and shared across model types so tokenizer
    comparisons use consistent input encoding logic.

    Supported modes:
      threshold: spike where projected activation is positive
      rate: sigmoid activation sampled as Bernoulli spikes
      temporal: deterministic sparse spike from top activations
      none: no input spikes; pass dense projected activations onward
    """

    def __init__(self, emb_dim: int, spike_dim: int, mode: str = "threshold"):
        super().__init__()
        self.proj = nn.Linear(emb_dim, spike_dim)
        self.mode = mode

        if mode not in {"threshold", "rate", "temporal", "none"}:
            raise ValueError(f"Unsupported encoder mode: {mode}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        x: [batch, seq, emb_dim]
        returns:
            encoded: [batch, seq, spike_dim]
        """
        z = self.proj(x)

        if self.mode == "none":
            encoded = z
        elif self.mode == "threshold":
            encoded = (z > 0).float()
        elif self.mode == "rate":
            probs = torch.sigmoid(z)
            encoded = torch.bernoulli(probs)
        elif self.mode == "temporal":
            # Deterministic sparse approximation: keep positive activations above
            # each sample/time-step mean. This gives lower spike activity while
            # preserving a stable smoke-test-friendly implementation.
            thresh = z.mean(dim=-1, keepdim=True)
            encoded = (z > thresh).float()
        else:
            raise RuntimeError("Unreachable encoder mode")

        stats = {
            "input_spike_rate": _rate_from_tensor(encoded) if self.mode != "none" else 0.0,
            "total_input_spikes": float(encoded.detach().float().sum().item())
            if self.mode != "none"
            else 0.0,
        }
        return encoded, stats


class SymbolEmbeddingMixin:
    """
    Shared helper for accepting either input_ids or inputs_embeds.

    This is the key future-compatibility hook:
      - input_ids for current tokenizers
      - inputs_embeds for future BLT/spike-native front ends
    """

    embedding: nn.Embedding

    def embed_inputs(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            return inputs_embeds

        if input_ids is None:
            raise ValueError("Either input_ids or inputs_embeds must be provided.")

        return self.embedding(input_ids)


class BaselineSpikingLM(nn.Module, SymbolEmbeddingMixin):
    """
    Stable baseline spiking LM.

    This is the forward-compatible baseline to use from now on. It is intentionally
    simple and tokenizer-agnostic, while preserving the same metric interface as
    the larger models.
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 32,
        spike_dim: int = 32,
        hidden_dim: int = 64,
        spike_steps: int = 4,
        beta: float = 0.95,
        encoder: str = "threshold",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.spike_dim = spike_dim
        self.hidden_dim = hidden_dim
        self.spike_steps = spike_steps

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.input_encoder = SpikeInputEncoder(emb_dim, spike_dim, mode=encoder)
        self.in_to_hidden = nn.Linear(spike_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.lif = snn.Leaky(beta=beta)
        self.readout = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        x = self.embed_inputs(input_ids=input_ids, inputs_embeds=inputs_embeds)
        encoded, input_stats = self.input_encoder(x)

        batch, seq, _ = encoded.shape
        mem = torch.zeros(batch, self.hidden_dim, device=encoded.device)

        logits_steps = []
        hidden_spike_sum = 0.0
        hidden_count = 0

        for t in range(seq):
            h_in = self.in_to_hidden(encoded[:, t, :])
            h_in = self.dropout(h_in)

            spike_accum = 0.0
            for _ in range(self.spike_steps):
                spk, mem = self.lif(h_in, mem)
                spike_accum = spike_accum + spk

            spk_mean = spike_accum / float(self.spike_steps)
            logits_steps.append(self.readout(spk_mean))

            hidden_spike_sum += float(spike_accum.detach().float().sum().item())
            hidden_count += spike_accum.numel()

        logits = torch.stack(logits_steps, dim=1)

        stats = {
            **input_stats,
            "hidden_spike_rate": hidden_spike_sum / max(hidden_count, 1),
            "total_hidden_spikes": hidden_spike_sum,
        }
        return logits, stats


class RecurrentSpikingLM(nn.Module, SymbolEmbeddingMixin):
    """
    Minimal recurrent SNN LM.

    This is an ablation model: recurrence and decay, but no RWKV-style
    receptance/key/value time mixing. It helps determine whether changes are due
    to generic recurrence or RWKV-like structure.
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 32,
        spike_dim: int = 32,
        hidden_dim: int = 64,
        spike_steps: int = 4,
        beta: float = 0.95,
        encoder: str = "threshold",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.spike_steps = spike_steps

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.input_encoder = SpikeInputEncoder(emb_dim, spike_dim, mode=encoder)
        self.input_proj = nn.Linear(spike_dim, hidden_dim)
        self.recurrent_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.decay_logit = nn.Parameter(torch.ones(hidden_dim) * 2.0)
        self.dropout = nn.Dropout(dropout)
        self.lif = snn.Leaky(beta=beta)
        self.readout = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        x = self.embed_inputs(input_ids=input_ids, inputs_embeds=inputs_embeds)
        encoded, input_stats = self.input_encoder(x)

        batch, seq, _ = encoded.shape
        state = torch.zeros(batch, self.hidden_dim, device=encoded.device)
        mem = torch.zeros(batch, self.hidden_dim, device=encoded.device)

        decay = torch.sigmoid(self.decay_logit).view(1, -1)

        logits_steps = []
        hidden_spike_sum = 0.0
        hidden_count = 0

        for t in range(seq):
            inp = self.input_proj(encoded[:, t, :])
            rec = self.recurrent_proj(state)
            state = decay * state + inp + rec
            state = self.dropout(state)

            spike_accum = 0.0
            for _ in range(self.spike_steps):
                spk, mem = self.lif(state, mem)
                spike_accum = spike_accum + spk

            spk_mean = spike_accum / float(self.spike_steps)
            state = spk_mean
            logits_steps.append(self.readout(spk_mean))

            hidden_spike_sum += float(spike_accum.detach().float().sum().item())
            hidden_count += spike_accum.numel()

        logits = torch.stack(logits_steps, dim=1)

        stats = {
            **input_stats,
            "hidden_spike_rate": hidden_spike_sum / max(hidden_count, 1),
            "total_hidden_spikes": hidden_spike_sum,
        }
        return logits, stats


class SpikingRWKVLM(nn.Module, SymbolEmbeddingMixin):
    """
    Tiny SpikeGPT/RWKV-inspired spiking LM.

    This is not a full SpikeGPT reproduction. It is a small controlled model
    that preserves the relevant scaling-compatible ingredients:
      - recurrent state
      - learned decay
      - receptance/gating
      - key/value content update
      - channel mixing
      - spiking hidden activity
      - tokenizer/front-end agnostic input interface
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 32,
        spike_dim: int = 32,
        hidden_dim: int = 64,
        spike_steps: int = 4,
        beta: float = 0.95,
        encoder: str = "threshold",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.spike_steps = spike_steps

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.input_encoder = SpikeInputEncoder(emb_dim, spike_dim, mode=encoder)

        # Project encoded input into RWKV-like hidden space.
        self.x_proj = nn.Linear(spike_dim, hidden_dim)

        # Learned time mixing between current and previous token representation.
        self.time_mix = nn.Parameter(torch.full((hidden_dim,), 0.5))

        # RWKV-like projections.
        self.receptance = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

        # Learned decay, constrained to (0, 1).
        self.decay_logit = nn.Parameter(torch.ones(hidden_dim) * 2.0)

        # Channel mixing block.
        self.channel_mix = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)
        self.lif = snn.Leaky(beta=beta)
        self.readout = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        x = self.embed_inputs(input_ids=input_ids, inputs_embeds=inputs_embeds)
        encoded, input_stats = self.input_encoder(x)

        batch, seq, _ = encoded.shape

        prev_x = torch.zeros(batch, self.hidden_dim, device=encoded.device)
        rwkv_state = torch.zeros(batch, self.hidden_dim, device=encoded.device)
        lif_mem = torch.zeros(batch, self.hidden_dim, device=encoded.device)

        decay = torch.sigmoid(self.decay_logit).view(1, -1)
        mix = torch.sigmoid(self.time_mix).view(1, -1)

        logits_steps = []
        hidden_spike_sum = 0.0
        hidden_count = 0

        for t in range(seq):
            x_t = self.x_proj(encoded[:, t, :])

            # RWKV-style time mixing: interpolate current and previous hidden input.
            x_mix = mix * x_t + (1.0 - mix) * prev_x
            prev_x = x_t

            r = torch.sigmoid(self.receptance(x_mix))
            k = self.key(x_mix)
            v = self.value(x_mix)

            # Small controlled key-value state update.
            rwkv_state = decay * rwkv_state + torch.tanh(k) * v

            time_out = r * rwkv_state
            time_out = self.dropout(time_out)

            spike_accum = 0.0
            for _ in range(self.spike_steps):
                spk, lif_mem = self.lif(time_out, lif_mem)
                spike_accum = spike_accum + spk

            spk_mean = spike_accum / float(self.spike_steps)

            # Channel mixing after spiking time-mix output.
            channel_out = spk_mean + self.channel_mix(spk_mean)

            logits_steps.append(self.readout(channel_out))

            hidden_spike_sum += float(spike_accum.detach().float().sum().item())
            hidden_count += spike_accum.numel()

        logits = torch.stack(logits_steps, dim=1)

        stats = {
            **input_stats,
            "hidden_spike_rate": hidden_spike_sum / max(hidden_count, 1),
            "total_hidden_spikes": hidden_spike_sum,
        }
        return logits, stats



class SpikingRWKVBlock(nn.Module):
    """
    One residual SpikeGPT/RWKV-inspired spiking block.

    This is a scalable proxy block rather than a full SpikeGPT reproduction.
    It preserves the key ingredients we care about for tokenizer/front-end tests:
      - layer norm
      - time mixing
      - receptance/gating
      - key/value-like recurrent state update
      - learned decay
      - LIF spiking activation
      - channel mixing
      - residual connections
    """

    def __init__(
        self,
        hidden_dim: int,
        spike_steps: int = 4,
        beta: float = 0.95,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.spike_steps = spike_steps

        self.ln_time = nn.LayerNorm(hidden_dim)
        self.ln_channel = nn.LayerNorm(hidden_dim)

        self.time_mix = nn.Parameter(torch.full((hidden_dim,), 0.5))

        self.receptance = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

        self.decay_logit = nn.Parameter(torch.ones(hidden_dim) * 2.0)

        self.lif = snn.Leaky(beta=beta)

        self.channel_mix = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

    def initial_state(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        return {
            "prev_x": torch.zeros(batch_size, self.hidden_dim, device=device),
            "rwkv_state": torch.zeros(batch_size, self.hidden_dim, device=device),
            "lif_mem": torch.zeros(batch_size, self.hidden_dim, device=device),
        }

    def forward_step(
        self,
        x_t: torch.Tensor,
        state: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, float]]:
        residual = x_t

        x_norm = self.ln_time(x_t)

        mix = torch.sigmoid(self.time_mix).view(1, -1)
        x_mix = mix * x_norm + (1.0 - mix) * state["prev_x"]
        state["prev_x"] = x_norm

        r = torch.sigmoid(self.receptance(x_mix))
        k = self.key(x_mix)
        v = self.value(x_mix)

        decay = torch.sigmoid(self.decay_logit).view(1, -1)
        state["rwkv_state"] = decay * state["rwkv_state"] + torch.tanh(k) * v

        time_out = r * state["rwkv_state"]
        time_out = self.dropout(time_out)

        spike_accum = 0.0
        for _ in range(self.spike_steps):
            spk, state["lif_mem"] = self.lif(time_out, state["lif_mem"])
            spike_accum = spike_accum + spk

        spk_mean = spike_accum / float(self.spike_steps)

        # Residual spiking time-mix path.
        x_mid = residual + spk_mean

        # Residual channel-mix path.
        channel_in = self.ln_channel(x_mid)
        channel_out = self.channel_mix(channel_in)
        out = x_mid + channel_out

        stats = {
            "hidden_spikes": float(spike_accum.detach().float().sum().item()),
            "hidden_count": spike_accum.numel(),
        }

        return out, state, stats


class StackedSpikingRWKVLM(nn.Module, SymbolEmbeddingMixin):
    """
    Stacked residual SpikeGPT/RWKV-inspired spiking LM.

    This is the scalable SRWKV proxy model. It keeps the same front-end interface
    as the baseline and SRWKV v1, but adds depth through repeated residual
    RWKV-like spiking blocks.
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 32,
        spike_dim: int = 32,
        hidden_dim: int = 64,
        spike_steps: int = 4,
        beta: float = 0.95,
        encoder: str = "threshold",
        dropout: float = 0.1,
        num_layers: int = 2,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.spike_steps = spike_steps
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.input_encoder = SpikeInputEncoder(emb_dim, spike_dim, mode=encoder)
        self.input_proj = nn.Linear(spike_dim, hidden_dim)

        self.blocks = nn.ModuleList([
            SpikingRWKVBlock(
                hidden_dim=hidden_dim,
                spike_steps=spike_steps,
                beta=beta,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.final_ln = nn.LayerNorm(hidden_dim)
        self.readout = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        inputs_spikes: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if inputs_spikes is not None:
            encoded = inputs_spikes.float()
            input_stats = {
                "input_spike_rate": float(encoded.detach().mean().item()) if encoded.numel() else 0.0,
                "total_input_spikes": float(encoded.detach().sum().item()) if encoded.numel() else 0.0,
            }
        else:
            x = self.embed_inputs(input_ids=input_ids, inputs_embeds=inputs_embeds)
            encoded, input_stats = self.input_encoder(x)

        batch, seq, _ = encoded.shape
        device = encoded.device

        block_states = [
            block.initial_state(batch, device)
            for block in self.blocks
        ]

        logits_steps = []
        hidden_spike_sum = 0.0
        hidden_count = 0

        for t in range(seq):
            h = self.input_proj(encoded[:, t, :])

            for i, block in enumerate(self.blocks):
                h, block_states[i], block_stats = block.forward_step(h, block_states[i])
                hidden_spike_sum += block_stats["hidden_spikes"]
                hidden_count += block_stats["hidden_count"]

            h = self.final_ln(h)
            logits_steps.append(self.readout(h))

        logits = torch.stack(logits_steps, dim=1)

        stats = {
            **input_stats,
            "hidden_spike_rate": hidden_spike_sum / max(hidden_count, 1),
            "total_hidden_spikes": hidden_spike_sum,
        }

        return logits, stats

def build_spiking_lm(
    model_type: str,
    vocab_size: int,
    emb_dim: int = 32,
    spike_dim: int = 32,
    hidden_dim: int = 64,
    spike_steps: int = 4,
    beta: float = 0.95,
    encoder: str = "threshold",
    dropout: float = 0.1,
    num_layers: int = 1,
) -> nn.Module:
    if model_type == "baseline":
        return BaselineSpikingLM(
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            spike_dim=spike_dim,
            hidden_dim=hidden_dim,
            spike_steps=spike_steps,
            beta=beta,
            encoder=encoder,
            dropout=dropout,
        )

    if model_type == "recurrent":
        return RecurrentSpikingLM(
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            spike_dim=spike_dim,
            hidden_dim=hidden_dim,
            spike_steps=spike_steps,
            beta=beta,
            encoder=encoder,
            dropout=dropout,
        )

    if model_type in {"srwkv", "spiking_rwkv"}:
        return SpikingRWKVLM(
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            spike_dim=spike_dim,
            hidden_dim=hidden_dim,
            spike_steps=spike_steps,
            beta=beta,
            encoder=encoder,
            dropout=dropout,
        )

    if model_type in {"srwkv_v2", "stacked_srwkv"}:
        return StackedSpikingRWKVLM(
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            spike_dim=spike_dim,
            hidden_dim=hidden_dim,
            spike_steps=spike_steps,
            beta=beta,
            encoder=encoder,
            dropout=dropout,
            num_layers=num_layers,
        )

    if model_type in {"ssm_snn", "selective_ssm", "ssm_snn_v1"}:
        return SelectiveSSMSpikingLM(
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            spike_dim=spike_dim,
            hidden_dim=hidden_dim,
            spike_steps=spike_steps,
            beta=beta,
            encoder=encoder,
            dropout=dropout,
            num_layers=num_layers,
        )

    raise ValueError(f"Unknown model_type: {model_type}")


class SelectiveSSMSpikingLM(nn.Module, SymbolEmbeddingMixin):
    """
    Small NeuronSpark-like selective state-space spiking LM proxy.

    This is not a reproduction of NeuronSpark. It is a compact backbone
    designed to test whether tokenizer/front-end effects transfer beyond
    the existing baseline and SRWKV-style models.

    Core idea:
      x_t -> input projection
      previous state -> recurrent projection
      gate controls selective state update
      LIF-like threshold produces spikes
      spikes/state feed vocabulary readout
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 32,
        spike_dim: int = 32,
        hidden_dim: int = 64,
        spike_steps: int = 4,
        beta: float = 0.95,
        encoder: str = "threshold",
        dropout: float = 0.1,
        num_layers: int = 1,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.spike_dim = spike_dim
        self.hidden_dim = hidden_dim
        self.spike_steps = spike_steps
        self.beta = beta
        self.encoder = encoder
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, emb_dim)

        self.input_proj = nn.Linear(emb_dim, spike_dim)

        self.in_proj = nn.ModuleList([
            nn.Linear(spike_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])

        self.state_proj = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=False)
            for _ in range(num_layers)
        ])

        self.gate_proj = nn.ModuleList([
            nn.Linear(spike_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Linear(hidden_dim, vocab_size)

    def _encode_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, emb_dim]
        returns spike-like input: [B, T, spike_dim]
        """
        z = self.input_proj(x)

        if self.encoder == "threshold":
            return (z > 0).float()

        if self.encoder == "temporal":
            return (torch.sigmoid(z) > 0.5).float()

        if self.encoder == "rate":
            probs = torch.sigmoid(z)
            return torch.bernoulli(probs)

        return (z > 0).float()

    def forward(self, input_ids=None, inputs_embeds=None):
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Either input_ids or inputs_embeds must be provided.")
            x = self.embedding(input_ids)
        else:
            x = inputs_embeds

        input_spikes = self._encode_input(x)
        batch_size, seq_len, _ = input_spikes.shape

        states = [
            torch.zeros(
                batch_size,
                self.hidden_dim,
                device=input_spikes.device,
                dtype=input_spikes.dtype,
            )
            for _ in range(self.num_layers)
        ]

        outputs = []
        total_hidden_spikes = 0.0

        for t in range(seq_len):
            h = input_spikes[:, t, :]

            for layer in range(self.num_layers):
                gate = torch.sigmoid(self.gate_proj[layer](h))
                candidate = torch.tanh(
                    self.in_proj[layer](h) + self.state_proj[layer](states[layer])
                )

                states[layer] = self.beta * states[layer] + gate * candidate

                spike = (states[layer] > 1.0).float()
                states[layer] = states[layer] - spike

                h = self.norms[layer](states[layer] + spike)
                h = self.dropout(h)

                total_hidden_spikes += float(spike.detach().sum().item())

            outputs.append(h)

        hidden = torch.stack(outputs, dim=1)
        logits = self.readout(hidden)

        total_input_spikes = float(input_spikes.detach().sum().item())
        input_rate = float(input_spikes.detach().mean().item())
        hidden_rate = total_hidden_spikes / max(
            batch_size * seq_len * self.hidden_dim * self.num_layers,
            1,
        )

        stats = {
            "input_spike_rate": input_rate,
            "total_input_spikes": total_input_spikes,
            "hidden_spike_rate": hidden_rate,
            "total_hidden_spikes": total_hidden_spikes,
        }

        return logits, stats
