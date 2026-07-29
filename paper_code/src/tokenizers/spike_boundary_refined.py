from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Optional

import random
import torch
import torch.nn.functional as F
from torch import nn


def bytes_to_ints(text: str) -> List[int]:
    return list(text.encode("utf-8", errors="ignore"))


def stable_hash_patch(patch: bytes, buckets: int) -> int:
    h = 2166136261
    for b in patch:
        h ^= int(b)
        h = (h * 16777619) & 0xFFFFFFFF
    return int(h % max(buckets, 1))


def local_byte_novelty(prev_b: Optional[int], b: int) -> float:
    if prev_b is None:
        return 1.0
    return 1.0 if b != prev_b else 0.25


def summarise_patches(
    patches: Sequence[bytes],
    num_chars: int,
    adaptive_steps: Optional[Sequence[int]] = None,
    segmenter_spike_native: bool = True,
    segmenter_params: int = 0,
    segmenter_trainable_params: int = 0,
) -> Dict[str, float]:
    lengths = [len(p) for p in patches]
    n_patches = len(patches)
    boundary_spikes = max(n_patches - 1, 0)

    if adaptive_steps is None:
        adaptive_steps = [1 for _ in patches]

    adaptive_steps = list(adaptive_steps)

    return {
        "num_chars": float(num_chars),
        "num_patches": float(n_patches),
        "num_boundary_spikes": float(boundary_spikes),
        "mean_patch_bytes": float(sum(lengths) / max(len(lengths), 1)),
        "min_patch_bytes": float(min(lengths) if lengths else 0),
        "max_patch_bytes": float(max(lengths) if lengths else 0),
        "patches_per_char": float(n_patches / max(num_chars, 1)),
        "chars_per_patch": float(num_chars / max(n_patches, 1)),
        "boundary_spikes_per_char": float(boundary_spikes / max(num_chars, 1)),
        "boundary_rate": float(boundary_spikes / max(n_patches, 1)),
        "mean_adaptive_steps": float(sum(adaptive_steps) / max(len(adaptive_steps), 1)),
        "min_adaptive_steps": float(min(adaptive_steps) if adaptive_steps else 0),
        "max_adaptive_steps": float(max(adaptive_steps) if adaptive_steps else 0),
        "adaptive_steps_per_char": float(sum(adaptive_steps) / max(num_chars, 1)),
        "adaptive_steps_per_patch": float(sum(adaptive_steps) / max(n_patches, 1)),
        "segmenter_spike_native": float(1.0 if segmenter_spike_native else 0.0),
        "segmenter_params": float(segmenter_params),
        "segmenter_trainable_params": float(segmenter_trainable_params),
    }


class ProxyBLTRefinedSegmenter:
    def __init__(self, patcher):
        self.patcher = patcher

    def segment_text(self, text: str) -> Tuple[List[bytes], List[int], Dict[str, float]]:
        patches = self.patcher.patch_text(text)
        steps = [1 for _ in patches]
        stats = summarise_patches(
            patches,
            len(text),
            adaptive_steps=steps,
            segmenter_spike_native=False,
        )
        return patches, steps, stats


class SpikeBoundaryLIFRefinedSegmenter:
    def __init__(
        self,
        threshold: float = 7.5,
        beta: float = 0.85,
        min_patch_bytes: int = 2,
        max_patch_bytes: int = 16,
        novelty_gain: float = 1.0,
        punctuation_gain: float = 2.0,
    ):
        self.threshold = threshold
        self.beta = beta
        self.min_patch_bytes = min_patch_bytes
        self.max_patch_bytes = max_patch_bytes
        self.novelty_gain = novelty_gain
        self.punctuation_gain = punctuation_gain

    def _current(self, prev_b: Optional[int], b: int) -> float:
        novelty = local_byte_novelty(prev_b, b)
        punctuation = 1.0 if b in {32, 10, 13, 9, 44, 46, 59, 58, 33, 63} else 0.0
        return self.novelty_gain * novelty + self.punctuation_gain * punctuation + 0.25

    def _step_for_patch(self, peak_mem: float) -> int:
        return 1

    def segment_bytes(self, data: Sequence[int]) -> Tuple[List[bytes], List[int], Dict[str, float]]:
        patches: List[bytes] = []
        steps: List[int] = []
        cur: List[int] = []
        mem = 0.0
        peak_mem = 0.0
        prev_b = None

        for b in data:
            b = int(b)
            cur.append(b)
            mem = self.beta * mem + self._current(prev_b, b)
            peak_mem = max(peak_mem, mem)
            prev_b = b

            force = len(cur) >= self.max_patch_bytes
            spike = mem >= self.threshold and len(cur) >= self.min_patch_bytes

            if force or spike:
                patches.append(bytes(cur))
                steps.append(self._step_for_patch(peak_mem))
                cur = []
                mem = 0.0
                peak_mem = 0.0

        if cur:
            patches.append(bytes(cur))
            steps.append(self._step_for_patch(peak_mem))

        stats = summarise_patches(patches, len(data), adaptive_steps=steps, segmenter_spike_native=True)
        return patches, steps, stats

    def segment_text(self, text: str) -> Tuple[List[bytes], List[int], Dict[str, float]]:
        return self.segment_bytes(bytes_to_ints(text))


class SpikeBoundaryAdaptiveRefinedSegmenter(SpikeBoundaryLIFRefinedSegmenter):
    def __init__(
        self,
        threshold: float = 7.5,
        beta: float = 0.85,
        min_patch_bytes: int = 2,
        max_patch_bytes: int = 16,
        min_steps: int = 1,
        max_steps: int = 4,
    ):
        super().__init__(
            threshold=threshold,
            beta=beta,
            min_patch_bytes=min_patch_bytes,
            max_patch_bytes=max_patch_bytes,
        )
        self.min_steps = min_steps
        self.max_steps = max_steps

    def _step_for_patch(self, peak_mem: float) -> int:
        frac = min(max(peak_mem / max(self.threshold, 1e-9), 0.0), 2.0) / 2.0
        steps = int(round(self.min_steps + frac * (self.max_steps - self.min_steps)))
        return max(self.min_steps, min(self.max_steps, steps))


class SurrogateBoundarySpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, membrane_minus_threshold: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(membrane_minus_threshold)
        return (membrane_minus_threshold > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return grad_output * torch.clamp(1.0 - x.abs(), min=0.0)


def boundary_spike_fn(x: torch.Tensor) -> torch.Tensor:
    return SurrogateBoundarySpike.apply(x)


class TrainableBoundaryNeuron(nn.Module):
    def __init__(self, emb_dim: int = 16, beta: float = 0.85, threshold: float = 1.0):
        super().__init__()
        self.emb = nn.Embedding(256, emb_dim)
        self.current = nn.Linear(emb_dim, 1)
        self.beta = beta
        self.threshold = threshold

    def forward_teacher_forced(self, byte_seq: torch.Tensor, teacher_boundaries: torch.Tensor) -> torch.Tensor:
        mem = torch.zeros((), device=byte_seq.device)
        logits = []

        for i in range(byte_seq.numel()):
            emb = self.emb(byte_seq[i])
            current = F.softplus(self.current(emb)).squeeze() + 0.01
            mem = self.beta * mem + current
            logit = mem - self.threshold
            logits.append(logit)

            if float(teacher_boundaries[i].detach().item()) > 0.5:
                mem = torch.zeros_like(mem)

        return torch.stack(logits)

    @torch.no_grad()
    def current_for_byte(self, b: int) -> float:
        device = next(self.parameters()).device
        bt = torch.tensor([int(b)], dtype=torch.long, device=device)
        val = F.softplus(self.current(self.emb(bt))).squeeze() + 0.01
        return float(val.item())


class SpikeBoundaryLearnedTrainableSegmenter:
    """
    Trainable spike-native boundary segmenter.

    A small learned synaptic current model is trained to imitate ProxyBLT
    boundary locations. After this fitting step, segmentation is generated
    by a LIF-style boundary membrane and hard threshold boundary spikes.
    ProxyBLT is used only as a training teacher, not as the inference segmenter.
    """

    def __init__(
        self,
        teacher_patches: Sequence[bytes],
        full_text: str,
        train_batches: int = 80,
        window: int = 512,
        lr: float = 1e-2,
        seed: int = 1,
        threshold: float = 1.0,
        beta: float = 0.85,
        min_patch_bytes: int = 2,
        max_patch_bytes: int = 16,
        device: str = "cpu",
    ):
        random.seed(seed)
        torch.manual_seed(seed)

        self.threshold = threshold
        self.beta = beta
        self.min_patch_bytes = min_patch_bytes
        self.max_patch_bytes = max_patch_bytes
        self.device = torch.device(device)

        self.data = bytes_to_ints(full_text)
        self.labels = self._labels_from_teacher(teacher_patches, len(self.data))

        self.model = TrainableBoundaryNeuron(beta=beta, threshold=threshold).to(self.device)
        self._fit(train_batches=train_batches, window=window, lr=lr)

    def _labels_from_teacher(self, teacher_patches: Sequence[bytes], n: int) -> torch.Tensor:
        labels = torch.zeros(n, dtype=torch.float32)
        idx = 0
        for p in teacher_patches:
            idx += len(p)
            if 0 < idx < n:
                labels[idx - 1] = 1.0
        return labels

    def _fit(self, train_batches: int, window: int, lr: float) -> None:
        if len(self.data) < 4:
            return

        opt = torch.optim.AdamW(self.model.parameters(), lr=lr)

        max_start = max(1, len(self.data) - window - 1)

        for _ in range(train_batches):
            start = random.randint(0, max_start)
            stop = min(start + window, len(self.data))

            x = torch.tensor(self.data[start:stop], dtype=torch.long, device=self.device)
            y = self.labels[start:stop].to(self.device)

            logits = self.model.forward_teacher_forced(x, y)
            loss = F.binary_cross_entropy_with_logits(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            opt.step()

    def segment_text(self, text: str) -> Tuple[List[bytes], List[int], Dict[str, float]]:
        data = bytes_to_ints(text)

        patches: List[bytes] = []
        steps: List[int] = []
        cur: List[int] = []
        mem = 0.0

        for b in data:
            b = int(b)
            cur.append(b)
            mem = self.beta * mem + self.model.current_for_byte(b)

            force = len(cur) >= self.max_patch_bytes
            spike = mem >= self.threshold and len(cur) >= self.min_patch_bytes

            if force or spike:
                patches.append(bytes(cur))
                steps.append(1)
                cur = []
                mem = 0.0

        if cur:
            patches.append(bytes(cur))
            steps.append(1)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        stats = summarise_patches(
            patches,
            len(data),
            adaptive_steps=steps,
            segmenter_spike_native=True,
            segmenter_params=total_params,
            segmenter_trainable_params=trainable_params,
        )
        return patches, steps, stats


class SpikePatchBinaryRefinedEncoder(nn.Module):
    def __init__(self, output_dim: int = 32, max_patch_bytes: int = 16):
        super().__init__()
        self.output_dim = output_dim
        self.max_patch_bytes = max_patch_bytes
        self.proj = nn.Linear(257, output_dim)

    def forward(self, patches: Sequence[bytes]) -> Tuple[torch.Tensor, Dict[str, float]]:
        rows = []

        for patch in patches:
            hist = torch.zeros(256, dtype=torch.float32)
            if patch:
                vals = torch.tensor(list(patch), dtype=torch.long)
                hist.scatter_add_(0, vals, torch.ones_like(vals, dtype=torch.float32))
                hist = hist / max(len(patch), 1)

            length = torch.tensor(
                [min(len(patch), self.max_patch_bytes) / max(self.max_patch_bytes, 1)],
                dtype=torch.float32,
            )
            rows.append(torch.cat([hist, length], dim=0))

        if rows:
            features = torch.stack(rows, dim=0).to(self.proj.weight.device)
        else:
            features = torch.zeros(0, 257, dtype=torch.float32, device=self.proj.weight.device)

        raw = self.proj(features)
        spikes = (torch.sigmoid(raw) >= 0.5).float()

        unique_events = float((spikes > 0).sum().item())

        return spikes, {
            "frontend_spike_rate": float(spikes.mean().item()) if spikes.numel() else 0.0,
            "frontend_total_spikes": float(spikes.sum().item()),
            "frontend_spikes_per_patch": float(spikes.sum().item() / max(len(patches), 1)),
            "frontend_unique_events": unique_events,
            "frontend_unique_events_per_patch": float(unique_events / max(len(patches), 1)),
            "frontend_nonzero_channels_per_patch": float(unique_events / max(len(patches), 1)),
        }


class SpikePatchEventNgramRefinedEncoder(nn.Module):
    def __init__(self, output_dim: int = 32, ngram: int = 3, events_per_patch: int = 4):
        super().__init__()
        self.output_dim = output_dim
        self.ngram = ngram
        self.events_per_patch = events_per_patch

    def _patch_code(self, patch: bytes) -> torch.Tensor:
        out = torch.zeros(self.output_dim, dtype=torch.float32)
        data = list(patch)

        if not data:
            return out

        ids = []
        n = max(1, self.ngram)

        if len(data) < n:
            ids = [stable_hash_patch(bytes(data), self.output_dim)]
        else:
            for i in range(len(data) - n + 1):
                ids.append(stable_hash_patch(bytes(data[i:i+n]), self.output_dim))

        for idx in ids[: self.events_per_patch]:
            out[int(idx) % self.output_dim] = 1.0

        return out

    def forward(self, patches: Sequence[bytes]) -> Tuple[torch.Tensor, Dict[str, float]]:
        rows = [self._patch_code(p) for p in patches]
        spikes = torch.stack(rows, dim=0) if rows else torch.zeros(0, self.output_dim)

        unique_events = float((spikes > 0).sum().item())

        return spikes, {
            "frontend_spike_rate": float(spikes.mean().item()) if spikes.numel() else 0.0,
            "frontend_total_spikes": float(spikes.sum().item()),
            "frontend_spikes_per_patch": float(spikes.sum().item() / max(len(patches), 1)),
            "frontend_unique_events": unique_events,
            "frontend_unique_events_per_patch": float(unique_events / max(len(patches), 1)),
            "frontend_nonzero_channels_per_patch": float(unique_events / max(len(patches), 1)),
        }


def build_refined_patch_encoder(patch_repr: str, output_dim: int):
    if patch_repr == "spikepatch_binary":
        return SpikePatchBinaryRefinedEncoder(output_dim=output_dim)
    if patch_repr == "spikepatch_event_ngram":
        return SpikePatchEventNgramRefinedEncoder(output_dim=output_dim)
    raise ValueError(f"Unknown patch_repr: {patch_repr}")
