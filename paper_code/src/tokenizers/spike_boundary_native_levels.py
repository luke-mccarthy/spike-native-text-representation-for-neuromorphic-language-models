from __future__ import annotations

from typing import Dict, List, Sequence, Tuple
import math
import random

import torch
import torch.nn.functional as F
from torch import nn

from src.tokenizers.spike_boundary_refined import (
    bytes_to_ints,
    local_byte_novelty,
    summarise_patches,
    SpikeBoundaryLIFRefinedSegmenter,
    SpikeBoundaryLearnedTrainableSegmenter,
)


def qvalue(xs: Sequence[float], q: float) -> float:
    if not xs:
        return 0.0
    vals = sorted(float(x) for x in xs)
    idx = int(round((len(vals) - 1) * q))
    idx = max(0, min(len(vals) - 1, idx))
    return vals[idx]


def causal_byte_surprise(data: Sequence[int], window: int = 64) -> List[float]:
    data = [int(x) for x in data]
    counts = [0 for _ in range(256)]
    total = 0
    scores: List[float] = []

    for i, b in enumerate(data):
        prob = (counts[b] + 1.0) / (total + 256.0)
        scores.append(float(-math.log(prob)))

        counts[b] += 1
        total += 1

        if total > window:
            old = data[i - window]
            counts[old] -= 1
            total -= 1

    return scores


def intrinsic_boundary_targets(
    data: Sequence[int],
    surprise: Sequence[float],
    min_patch_bytes: int = 2,
    max_patch_bytes: int = 16,
    surprise_quantile: float = 0.88,
) -> torch.Tensor:
    thr = qvalue(surprise, surprise_quantile)
    y = torch.zeros(len(data), dtype=torch.float32)

    since = 0
    punctuation_set = {32, 10, 13, 9, 44, 46, 59, 58, 33, 63}

    for i, b in enumerate(data):
        b = int(b)
        since += 1

        punctuation = b in punctuation_set
        high_surprise = surprise[i] >= thr and since >= min_patch_bytes
        force = since >= max_patch_bytes

        if force or punctuation or high_surprise:
            if since >= min_patch_bytes:
                y[i] = 1.0
                since = 0

    return y


def patch_interval_stats(patches: Sequence[bytes]) -> Dict[str, float]:
    lengths = [len(p) for p in patches]
    if not lengths:
        return {
            "mean_boundary_interval_chars": 0.0,
            "std_boundary_interval_chars": 0.0,
            "min_boundary_interval_chars": 0.0,
            "max_boundary_interval_chars": 0.0,
        }

    mean = sum(lengths) / len(lengths)
    var = sum((x - mean) ** 2 for x in lengths) / len(lengths)

    return {
        "mean_boundary_interval_chars": float(mean),
        "std_boundary_interval_chars": float(math.sqrt(var)),
        "min_boundary_interval_chars": float(min(lengths)),
        "max_boundary_interval_chars": float(max(lengths)),
    }


class SpikeBoundaryLIFRefractorySegmenter(SpikeBoundaryLIFRefinedSegmenter):
    def __init__(
        self,
        threshold: float = 7.5,
        beta: float = 0.85,
        min_patch_bytes: int = 2,
        max_patch_bytes: int = 16,
        refractory_chars: int = 3,
    ):
        super().__init__(
            threshold=threshold,
            beta=beta,
            min_patch_bytes=min_patch_bytes,
            max_patch_bytes=max_patch_bytes,
        )
        self.refractory_chars = int(refractory_chars)

    def segment_bytes(self, data: Sequence[int]) -> Tuple[List[bytes], List[int], Dict[str, float]]:
        patches: List[bytes] = []
        steps: List[int] = []
        cur: List[int] = []

        mem = 0.0
        peak_mem = 0.0
        prev_b = None
        refractory_left = 0
        suppressed = 0

        for b in data:
            b = int(b)
            cur.append(b)

            mem = self.beta * mem + self._current(prev_b, b)
            peak_mem = max(peak_mem, mem)
            prev_b = b

            force = len(cur) >= self.max_patch_bytes
            candidate_spike = mem >= self.threshold and len(cur) >= self.min_patch_bytes

            if candidate_spike and refractory_left > 0 and not force:
                suppressed += 1
                candidate_spike = False

            if force or candidate_spike:
                patches.append(bytes(cur))
                steps.append(self._step_for_patch(peak_mem))
                cur = []
                mem = 0.0
                peak_mem = 0.0
                refractory_left = self.refractory_chars
            else:
                refractory_left = max(0, refractory_left - 1)

        if cur:
            patches.append(bytes(cur))
            steps.append(self._step_for_patch(peak_mem))

        stats = summarise_patches(patches, len(data), adaptive_steps=steps, segmenter_spike_native=True)
        stats.update(patch_interval_stats(patches))
        stats.update({
            "refractory_chars": float(self.refractory_chars),
            "refractory_suppressed_count": float(suppressed),
            "refractory_suppressed_per_char": float(suppressed / max(len(data), 1)),
            "hard_reset_count": float(len(patches)),
            "soft_reset_count": 0.0,
            "uses_proxy_boundary_teacher": 0.0,
            "proxy_free_boundary_targets": 1.0,
            "surrogate_spike_boundary_training": 0.0,
            "local_online_boundary_learning": 0.0,
            "spike_native_inference": 1.0,
            "end_to_end_lm_boundary_training": 0.0,
        })
        return patches, steps, stats


class TeacherBLTRefractorySegmenter(SpikeBoundaryLearnedTrainableSegmenter):
    def __init__(self, *args, refractory_chars: int = 3, **kwargs):
        self.refractory_chars = int(refractory_chars)
        super().__init__(*args, **kwargs)

    def segment_text(self, text: str) -> Tuple[List[bytes], List[int], Dict[str, float]]:
        data = bytes_to_ints(text)

        patches: List[bytes] = []
        steps: List[int] = []
        cur: List[int] = []

        mem = 0.0
        refractory_left = 0
        suppressed = 0

        for b in data:
            b = int(b)
            cur.append(b)

            mem = self.beta * mem + self.model.current_for_byte(b)

            force = len(cur) >= self.max_patch_bytes
            candidate_spike = mem >= self.threshold and len(cur) >= self.min_patch_bytes

            if candidate_spike and refractory_left > 0 and not force:
                suppressed += 1
                candidate_spike = False

            if force or candidate_spike:
                patches.append(bytes(cur))
                steps.append(1)
                cur = []
                mem = 0.0
                refractory_left = self.refractory_chars
            else:
                refractory_left = max(0, refractory_left - 1)

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
        stats.update(patch_interval_stats(patches))
        stats.update({
            "refractory_chars": float(self.refractory_chars),
            "refractory_suppressed_count": float(suppressed),
            "refractory_suppressed_per_char": float(suppressed / max(len(data), 1)),
            "hard_reset_count": float(len(patches)),
            "soft_reset_count": 0.0,
            "uses_proxy_boundary_teacher": 1.0,
            "proxy_free_boundary_targets": 0.0,
            "surrogate_spike_boundary_training": 0.0,
            "local_online_boundary_learning": 0.0,
            "spike_native_inference": 1.0,
            "end_to_end_lm_boundary_training": 0.0,
        })
        return patches, steps, stats


class ByteCurrentNeuron(nn.Module):
    def __init__(self, emb_dim: int = 16, beta: float = 0.85, threshold: float = 1.0):
        super().__init__()
        self.emb = nn.Embedding(256, emb_dim)
        self.current = nn.Linear(emb_dim, 1)
        self.beta = float(beta)
        self.threshold = float(threshold)

    def current_value(self, b: int) -> float:
        device = next(self.parameters()).device
        bt = torch.tensor([int(b)], dtype=torch.long, device=device)
        with torch.no_grad():
            cur = F.softplus(self.current(self.emb(bt))).squeeze() + 0.01
        return float(cur.item())

    def membrane_logits_teacher_reset(self, byte_seq: torch.Tensor, target_boundaries: torch.Tensor) -> torch.Tensor:
        mem = torch.zeros((), device=byte_seq.device)
        logits = []

        for i in range(byte_seq.numel()):
            cur = F.softplus(self.current(self.emb(byte_seq[i]))).squeeze() + 0.01
            mem = self.beta * mem + cur
            logits.append(mem - self.threshold)

            if float(target_boundaries[i].detach().item()) > 0.5:
                mem = torch.zeros_like(mem)

        return torch.stack(logits)


class SurrogateStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, width: float = 1.0):
        ctx.save_for_backward(x)
        ctx.width = width
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        width = ctx.width
        grad = torch.clamp(1.0 - torch.abs(x / width), min=0.0)
        return grad_output * grad, None


def surrogate_step(x: torch.Tensor, width: float = 1.0) -> torch.Tensor:
    return SurrogateStep.apply(x, width)


class ByteSurpriseLogitBoundarySegmenter:
    """
    Level 2.

    No ProxyBLT. Boundary targets are created from causal byte-surprise,
    punctuation, and max-patch constraints. Training uses membrane logits,
    not surrogate spikes.
    """

    def __init__(
        self,
        full_text: str,
        train_batches: int = 40,
        window: int = 512,
        lr: float = 1e-2,
        seed: int = 1,
        threshold: float = 1.0,
        beta: float = 0.85,
        min_patch_bytes: int = 2,
        max_patch_bytes: int = 16,
        surprise_window: int = 64,
        surprise_quantile: float = 0.88,
        device: str = "cpu",
    ):
        random.seed(seed)
        torch.manual_seed(seed)

        self.threshold = float(threshold)
        self.beta = float(beta)
        self.min_patch_bytes = int(min_patch_bytes)
        self.max_patch_bytes = int(max_patch_bytes)
        self.device = torch.device(device)

        self.data = bytes_to_ints(full_text)
        self.surprise = causal_byte_surprise(self.data, window=surprise_window)
        self.targets = intrinsic_boundary_targets(
            self.data,
            self.surprise,
            min_patch_bytes=min_patch_bytes,
            max_patch_bytes=max_patch_bytes,
            surprise_quantile=surprise_quantile,
        )

        self.model = ByteCurrentNeuron(beta=beta, threshold=threshold).to(self.device)
        self._fit(train_batches=train_batches, window=window, lr=lr)

    def _fit(self, train_batches: int, window: int, lr: float) -> None:
        if len(self.data) < 4:
            return

        opt = torch.optim.AdamW(self.model.parameters(), lr=lr)
        max_start = max(1, len(self.data) - window - 1)

        for _ in range(train_batches):
            start = random.randint(0, max_start)
            stop = min(start + window, len(self.data))

            x = torch.tensor(self.data[start:stop], dtype=torch.long, device=self.device)
            y = self.targets[start:stop].to(self.device)

            logits = self.model.membrane_logits_teacher_reset(x, y)
            pos = float(y.mean().item())
            pos_weight = torch.tensor([(1.0 - pos) / max(pos, 1e-3)], device=self.device)

            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)

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

            mem = self.beta * mem + self.model.current_value(b)

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
        stats.update(patch_interval_stats(patches))
        stats.update({
            "uses_proxy_boundary_teacher": 0.0,
            "proxy_free_boundary_targets": 1.0,
            "surrogate_spike_boundary_training": 0.0,
            "local_online_boundary_learning": 0.0,
            "spike_native_inference": 1.0,
            "end_to_end_lm_boundary_training": 0.0,
            "intrinsic_target_boundary_rate": float(self.targets.mean().item()) if len(self.targets) else 0.0,
            "byte_surprise_mean": float(sum(self.surprise) / max(len(self.surprise), 1)),
            "hard_reset_count": float(len(patches)),
            "soft_reset_count": 0.0,
        })
        return patches, steps, stats


class SurrogateSurpriseSpikeBoundarySegmenter(ByteSurpriseLogitBoundarySegmenter):
    """
    Level 3.

    No ProxyBLT. Boundary targets are created from byte-surprise.
    Training uses surrogate gradients through emitted boundary spikes.
    """

    def _training_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run hard spike/reset dynamics while retaining surrogate reset gradients."""
        mem = torch.zeros((), device=self.device)
        logits = []
        spikes = []
        since = 0

        for i in range(x.numel()):
            since += 1
            cur = F.softplus(self.model.current(self.model.emb(x[i]))).squeeze() + 0.01
            mem = self.beta * mem + cur
            logit = mem - self.threshold
            candidate = surrogate_step(logit, width=1.0)
            eligible = since >= self.min_patch_bytes
            force = since >= self.max_patch_bytes

            if force:
                spike = logit * 0.0 + 1.0
                emitted = True
            elif eligible:
                spike = candidate
                emitted = bool(float(candidate.detach().item()) > 0.5)
            else:
                spike = logit * 0.0
                emitted = False

            logits.append(logit)
            spikes.append(spike)
            mem = mem * (1.0 - spike)
            if emitted:
                since = 0

        return torch.stack(logits), torch.stack(spikes)

    def _fit(self, train_batches: int, window: int, lr: float) -> None:
        self.training_diagnostics: Dict[str, float] = {
            "surrogate_train_batches": float(train_batches),
        }
        if len(self.data) < 4:
            return

        opt = torch.optim.AdamW(self.model.parameters(), lr=lr)
        max_start = max(1, len(self.data) - window - 1)
        losses: List[float] = []
        gradient_norms: List[float] = []
        hard_rates: List[float] = []
        target_rates: List[float] = []

        for _ in range(train_batches):
            start = random.randint(0, max_start)
            stop = min(start + window, len(self.data))

            x = torch.tensor(self.data[start:stop], dtype=torch.long, device=self.device)
            y = self.targets[start:stop].to(self.device)

            logits, spikes = self._training_forward(x)
            positive_rate = y.mean().clamp(min=1e-3, max=1.0 - 1e-3)
            pos_weight = (1.0 - positive_rate) / positive_rate
            classification_loss = F.binary_cross_entropy_with_logits(
                logits,
                y,
                pos_weight=pos_weight,
            )
            soft_rate = torch.sigmoid(logits).mean()
            rate_loss = torch.abs(soft_rate - y.mean())
            loss = classification_loss + 0.01 * rate_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_squared = torch.zeros((), device=self.device)
            for parameter in self.model.parameters():
                if parameter.grad is not None:
                    grad_squared = grad_squared + parameter.grad.detach().pow(2).sum()
            gradient_norm = float(torch.sqrt(grad_squared).item())
            if not math.isfinite(gradient_norm) or gradient_norm <= 0.0:
                raise RuntimeError(
                    "Surrogate boundary training produced a zero or non-finite gradient"
                )
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            opt.step()

            losses.append(float(loss.detach().item()))
            gradient_norms.append(gradient_norm)
            hard_rates.append(float(spikes.detach().mean().item()))
            target_rates.append(float(y.detach().mean().item()))

        if losses:
            self.training_diagnostics.update({
                "surrogate_train_loss_initial": losses[0],
                "surrogate_train_loss_final": losses[-1],
                "surrogate_train_loss_min": min(losses),
                "surrogate_train_gradient_norm_min": min(gradient_norms),
                "surrogate_train_gradient_norm_max": max(gradient_norms),
                "surrogate_train_hard_spike_rate_mean": sum(hard_rates) / len(hard_rates),
                "surrogate_train_target_rate_mean": sum(target_rates) / len(target_rates),
                "surrogate_train_uses_logit_bce": 1.0,
                "surrogate_train_rate_auxiliary_weight": 0.01,
            })

    def segment_text(self, text: str) -> Tuple[List[bytes], List[int], Dict[str, float]]:
        patches, steps, stats = super().segment_text(text)
        stats.update({
            "surrogate_spike_boundary_training": 1.0,
            **self.training_diagnostics,
        })
        return patches, steps, stats


class SurpriseResetBoundarySegmenter:
    """
    Level 2 proxy-free rule method.

    No ProxyBLT and no learned boundary model. Causal byte surprise drives
    hard/soft reset choice. This is a local text analogue of entropy-aware
    reset, but uses byte-surprise rather than LLM token entropy.
    """

    def __init__(
        self,
        threshold: float = 7.5,
        beta: float = 0.85,
        min_patch_bytes: int = 2,
        max_patch_bytes: int = 16,
        surprise_window: int = 64,
        hard_reset_quantile: float = 0.80,
        soft_reset_scale: float = 0.5,
        refractory_chars: int = 2,
    ):
        self.threshold = float(threshold)
        self.beta = float(beta)
        self.min_patch_bytes = int(min_patch_bytes)
        self.max_patch_bytes = int(max_patch_bytes)
        self.surprise_window = int(surprise_window)
        self.hard_reset_quantile = float(hard_reset_quantile)
        self.soft_reset_scale = float(soft_reset_scale)
        self.refractory_chars = int(refractory_chars)

    def segment_text(self, text: str) -> Tuple[List[bytes], List[int], Dict[str, float]]:
        data = bytes_to_ints(text)
        surprise = causal_byte_surprise(data, window=self.surprise_window)
        hard_thr = qvalue(surprise, self.hard_reset_quantile)

        min_s = min(surprise) if surprise else 0.0
        max_s = max(surprise) if surprise else 1.0
        denom = max(max_s - min_s, 1e-9)

        patches: List[bytes] = []
        steps: List[int] = []
        cur: List[int] = []

        mem = 0.0
        prev_b = None
        hard_resets = 0
        soft_resets = 0
        suppressed = 0
        refractory_left = 0

        punctuation_set = {32, 10, 13, 9, 44, 46, 59, 58, 33, 63}

        for i, b in enumerate(data):
            b = int(b)
            cur.append(b)

            novelty = local_byte_novelty(prev_b, b)
            punctuation = 1.0 if b in punctuation_set else 0.0
            snorm = (surprise[i] - min_s) / denom
            current = 0.25 + novelty + 1.25 * punctuation + 1.25 * snorm

            mem = self.beta * mem + current
            prev_b = b

            force = len(cur) >= self.max_patch_bytes
            candidate_spike = mem >= self.threshold and len(cur) >= self.min_patch_bytes

            if candidate_spike and refractory_left > 0 and not force:
                suppressed += 1
                candidate_spike = False

            if force or candidate_spike:
                high_surprise = surprise[i] >= hard_thr
                patches.append(bytes(cur))
                steps.append(1)
                cur = []

                if high_surprise:
                    hard_resets += 1
                    mem = 0.0
                else:
                    soft_resets += 1
                    mem = max(0.0, mem - self.soft_reset_scale * self.threshold)

                refractory_left = self.refractory_chars
            else:
                refractory_left = max(0, refractory_left - 1)

        if cur:
            patches.append(bytes(cur))
            steps.append(1)

        stats = summarise_patches(patches, len(data), adaptive_steps=steps, segmenter_spike_native=True)
        stats.update(patch_interval_stats(patches))
        stats.update({
            "uses_proxy_boundary_teacher": 0.0,
            "proxy_free_boundary_targets": 1.0,
            "surrogate_spike_boundary_training": 0.0,
            "local_online_boundary_learning": 0.0,
            "spike_native_inference": 1.0,
            "end_to_end_lm_boundary_training": 0.0,
            "byte_surprise_mean": float(sum(surprise) / max(len(surprise), 1)),
            "byte_surprise_hard_reset_threshold": float(hard_thr),
            "hard_reset_count": float(hard_resets),
            "soft_reset_count": float(soft_resets),
            "hard_reset_rate": float(hard_resets / max(hard_resets + soft_resets, 1)),
            "soft_reset_rate": float(soft_resets / max(hard_resets + soft_resets, 1)),
            "soft_reset_scale": float(self.soft_reset_scale),
            "refractory_chars": float(self.refractory_chars),
            "refractory_suppressed_count": float(suppressed),
            "refractory_suppressed_per_char": float(suppressed / max(len(data), 1)),
        })
        return patches, steps, stats


class LocalHomeostaticBoundarySegmenter:
    """
    Level 4.

    No ProxyBLT. No backpropagation. A local online update adjusts per-byte
    currents and threshold using only causal byte surprise and recent boundary
    behaviour.
    """

    def __init__(
        self,
        full_text: str,
        passes: int = 2,
        seed: int = 1,
        threshold: float = 7.5,
        beta: float = 0.85,
        min_patch_bytes: int = 2,
        max_patch_bytes: int = 16,
        surprise_window: int = 64,
        surprise_quantile: float = 0.88,
        lr_current: float = 0.02,
        lr_threshold: float = 0.005,
        target_boundary_rate: float = 0.12,
    ):
        random.seed(seed)

        self.threshold = float(threshold)
        self.beta = float(beta)
        self.min_patch_bytes = int(min_patch_bytes)
        self.max_patch_bytes = int(max_patch_bytes)
        self.lr_current = float(lr_current)
        self.lr_threshold = float(lr_threshold)
        self.target_boundary_rate = float(target_boundary_rate)

        self.currents = [0.25 for _ in range(256)]
        for b in {32, 10, 13, 9, 44, 46, 59, 58, 33, 63}:
            self.currents[b] = 1.25

        data = bytes_to_ints(full_text)
        surprise = causal_byte_surprise(data, window=surprise_window)
        targets = intrinsic_boundary_targets(
            data,
            surprise,
            min_patch_bytes=min_patch_bytes,
            max_patch_bytes=max_patch_bytes,
            surprise_quantile=surprise_quantile,
        )

        self._local_fit(data, targets, passes=passes)

    def _local_fit(self, data: Sequence[int], targets: torch.Tensor, passes: int = 2) -> None:
        for _ in range(passes):
            mem = 0.0
            since = 0
            spike_count = 0
            seen = 0

            for i, b in enumerate(data):
                b = int(b)
                seen += 1
                since += 1

                mem = self.beta * mem + self.currents[b]

                force = since >= self.max_patch_bytes
                spike = (mem >= self.threshold and since >= self.min_patch_bytes) or force
                desired = bool(float(targets[i].item()) > 0.5)

                if desired and not spike:
                    self.currents[b] += self.lr_current
                elif spike and not desired and not force:
                    self.currents[b] = max(0.01, self.currents[b] - self.lr_current)

                if spike:
                    spike_count += 1
                    mem = 0.0
                    since = 0

                observed_rate = spike_count / max(seen, 1)
                self.threshold += self.lr_threshold * (observed_rate - self.target_boundary_rate)
                self.threshold = max(1.0, min(20.0, self.threshold))

    def segment_text(self, text: str) -> Tuple[List[bytes], List[int], Dict[str, float]]:
        data = bytes_to_ints(text)

        patches: List[bytes] = []
        steps: List[int] = []
        cur: List[int] = []

        mem = 0.0

        for b in data:
            b = int(b)
            cur.append(b)
            mem = self.beta * mem + self.currents[b]

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

        stats = summarise_patches(patches, len(data), adaptive_steps=steps, segmenter_spike_native=True)
        stats.update(patch_interval_stats(patches))
        stats.update({
            "uses_proxy_boundary_teacher": 0.0,
            "proxy_free_boundary_targets": 1.0,
            "surrogate_spike_boundary_training": 0.0,
            "local_online_boundary_learning": 1.0,
            "spike_native_inference": 1.0,
            "end_to_end_lm_boundary_training": 0.0,
            "local_final_threshold": float(self.threshold),
            "local_mean_current": float(sum(self.currents) / len(self.currents)),
            "hard_reset_count": float(len(patches)),
            "soft_reset_count": 0.0,
        })
        return patches, steps, stats


class EMAThresholdHomeostaticBoundarySegmenter:
    """Proxy-free local threshold homeostasis fitted on training bytes only.

    The controller adapts a scalar LIF threshold from an exponential moving
    average of emitted boundary activity.  The fitted threshold is frozen by
    ``segment_text``; evaluation therefore performs no parameter or state
    learning.  This is a rate-control mechanism, not a semantic boundary
    teacher.
    """

    def __init__(
        self,
        full_text: str,
        target_boundary_rate: float = 0.10,
        passes: int = 1,
        threshold: float = 7.5,
        beta: float = 0.85,
        min_patch_bytes: int = 4,
        max_patch_bytes: int = 24,
        ema_alpha: float = 0.001,
        threshold_lr: float = 0.05,
        threshold_min: float = 1.0,
        threshold_max: float = 60.0,
    ):
        if not 0.0 < target_boundary_rate < 1.0:
            raise ValueError("target_boundary_rate must be between zero and one")
        if passes <= 0:
            raise ValueError("passes must be positive")
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1]")
        if threshold_lr <= 0.0:
            raise ValueError("threshold_lr must be positive")
        if min_patch_bytes <= 0 or max_patch_bytes < min_patch_bytes:
            raise ValueError("Invalid patch-length constraints")
        if threshold_min <= 0.0 or threshold_max <= threshold_min:
            raise ValueError("Invalid threshold bounds")

        self.target_boundary_rate = float(target_boundary_rate)
        self.passes = int(passes)
        self.threshold = float(threshold)
        self.initial_threshold = float(threshold)
        self.beta = float(beta)
        self.min_patch_bytes = int(min_patch_bytes)
        self.max_patch_bytes = int(max_patch_bytes)
        self.ema_alpha = float(ema_alpha)
        self.threshold_lr = float(threshold_lr)
        self.threshold_min = float(threshold_min)
        self.threshold_max = float(threshold_max)
        self.training_boundary_rate = 0.0
        self.training_ema_rate = self.target_boundary_rate
        self.training_forced_boundary_fraction = 0.0
        self._fit(bytes_to_ints(full_text))

    @staticmethod
    def _current(prev_b: int | None, b: int) -> float:
        novelty = local_byte_novelty(prev_b, b)
        punctuation = 1.0 if b in {32, 10, 13, 9, 44, 46, 59, 58, 33, 63} else 0.0
        return 0.25 + novelty + 2.0 * punctuation

    def _fit(self, data: Sequence[int]) -> None:
        if not data:
            raise ValueError("Cannot fit homeostasis on empty training text")
        final_spikes = 0
        final_forced = 0
        final_ema = self.target_boundary_rate
        for _ in range(self.passes):
            mem = 0.0
            since = 0
            prev_b = None
            ema = self.target_boundary_rate
            spike_count = 0
            forced_count = 0
            for value in data:
                b = int(value)
                since += 1
                mem = self.beta * mem + self._current(prev_b, b)
                prev_b = b
                force = since >= self.max_patch_bytes
                spike = force or (mem >= self.threshold and since >= self.min_patch_bytes)
                indicator = 1.0 if spike else 0.0
                ema = (1.0 - self.ema_alpha) * ema + self.ema_alpha * indicator
                self.threshold += self.threshold_lr * (ema - self.target_boundary_rate)
                self.threshold = max(self.threshold_min, min(self.threshold_max, self.threshold))
                if spike:
                    spike_count += 1
                    forced_count += int(force)
                    mem = 0.0
                    since = 0
            final_spikes = spike_count
            final_forced = forced_count
            final_ema = ema

        self.training_boundary_rate = final_spikes / max(len(data), 1)
        self.training_ema_rate = final_ema
        self.training_forced_boundary_fraction = final_forced / max(final_spikes, 1)

    def segment_text(self, text: str) -> Tuple[List[bytes], List[int], Dict[str, float]]:
        data = bytes_to_ints(text)
        patches: List[bytes] = []
        current_patch: List[int] = []
        mem = 0.0
        prev_b = None
        forced_count = 0

        for value in data:
            b = int(value)
            current_patch.append(b)
            mem = self.beta * mem + self._current(prev_b, b)
            prev_b = b
            force = len(current_patch) >= self.max_patch_bytes
            spike = mem >= self.threshold and len(current_patch) >= self.min_patch_bytes
            if force or spike:
                forced_count += int(force)
                patches.append(bytes(current_patch))
                current_patch = []
                mem = 0.0
        if current_patch:
            patches.append(bytes(current_patch))

        steps = [1 for _ in patches]
        stats = summarise_patches(
            patches,
            len(data),
            adaptive_steps=steps,
            segmenter_spike_native=True,
        )
        stats.update(patch_interval_stats(patches))
        achieved_rate = max(len(patches) - 1, 0) / max(len(data), 1)
        stats.update({
            "uses_proxy_boundary_teacher": 0.0,
            "proxy_free_boundary_targets": 1.0,
            "surrogate_spike_boundary_training": 0.0,
            "local_online_boundary_learning": 1.0,
            "spike_native_inference": 1.0,
            "end_to_end_lm_boundary_training": 0.0,
            "homeostatic_controller_ema_threshold": 1.0,
            "homeostatic_target_boundary_rate": self.target_boundary_rate,
            "homeostatic_training_boundary_rate": self.training_boundary_rate,
            "homeostatic_training_ema_rate": self.training_ema_rate,
            "homeostatic_validation_boundary_rate": achieved_rate,
            "homeostatic_absolute_target_rate_error": abs(achieved_rate - self.target_boundary_rate),
            "homeostatic_initial_threshold": self.initial_threshold,
            "homeostatic_final_threshold": self.threshold,
            "homeostatic_ema_alpha": self.ema_alpha,
            "homeostatic_threshold_lr": self.threshold_lr,
            "homeostatic_training_forced_boundary_fraction": self.training_forced_boundary_fraction,
            "homeostatic_validation_forced_boundary_fraction": forced_count / max(len(patches), 1),
            "evaluation_parameter_updates": 0.0,
        })
        return patches, steps, stats


class LocalAdaptiveSelectivityBoundarySegmenter:
    """
    Level 4 local non-homeostatic boundary adaptation.

    This method is inspired by adaptive activity regulation rather than
    fixed homeostasis. It does not use ProxyBLT and does not use
    backpropagation.

    The rule adapts:
    - byte currents through local error between desired and observed boundary events;
    - an activity-regulation term Ath that acts as spike-triggered inhibition;
    - Ath is adapted over logging intervals using recent boundary activity.

    Interpretation:
    - proxy-free;
    - local online boundary adaptation;
    - spike-native inference;
    - not end-to-end LM-trained;
    - not surrogate-gradient trained.
    """

    def __init__(
        self,
        full_text: str,
        passes: int = 2,
        seed: int = 1,
        threshold: float = 7.5,
        beta: float = 0.85,
        min_patch_bytes: int = 4,
        max_patch_bytes: int = 16,
        surprise_window: int = 64,
        surprise_quantile: float = 0.94,
        lr_current: float = 0.01,
        ath_init: float = 1.0,
        ath_alpha: float = 0.05,
        ath_min: float = 0.0,
        ath_max: float = 12.0,
        logging_interval: int = 10000,
    ):
        random.seed(seed)

        self.threshold = float(threshold)
        self.beta = float(beta)
        self.min_patch_bytes = int(min_patch_bytes)
        self.max_patch_bytes = int(max_patch_bytes)
        self.lr_current = float(lr_current)
        self.ath = float(ath_init)
        self.ath_alpha = float(ath_alpha)
        self.ath_min = float(ath_min)
        self.ath_max = float(ath_max)
        self.logging_interval = int(logging_interval)

        self.currents = [0.20 for _ in range(256)]
        for b in {32, 10, 13, 9, 44, 46, 59, 58, 33, 63}:
            self.currents[b] = 1.00

        data = bytes_to_ints(full_text)
        surprise = causal_byte_surprise(data, window=surprise_window)
        targets = intrinsic_boundary_targets(
            data,
            surprise,
            min_patch_bytes=min_patch_bytes,
            max_patch_bytes=max_patch_bytes,
            surprise_quantile=surprise_quantile,
        )

        self._local_fit(data, targets, passes=passes)

    def _local_fit(self, data: Sequence[int], targets: torch.Tensor, passes: int = 2) -> None:
        for _ in range(passes):
            mem = 0.0
            since = 0

            interval_spikes = 0
            interval_targets = 0
            interval_seen = 0

            for i, b in enumerate(data):
                b = int(b)
                since += 1
                interval_seen += 1

                desired = bool(float(targets[i].item()) > 0.5)
                if desired:
                    interval_targets += 1

                effective_threshold = self.threshold + self.ath
                mem = self.beta * mem + self.currents[b]

                force = since >= self.max_patch_bytes
                spike = (mem >= effective_threshold and since >= self.min_patch_bytes) or force

                if desired and not spike:
                    self.currents[b] += self.lr_current
                elif spike and not desired and not force:
                    self.currents[b] = max(0.01, self.currents[b] - self.lr_current)

                if spike:
                    interval_spikes += 1
                    mem = 0.0
                    since = 0

                if interval_seen >= self.logging_interval:
                    desired_rate = interval_targets / max(interval_seen, 1)
                    observed_rate = interval_spikes / max(interval_seen, 1)

                    if desired_rate > 0:
                        relative_error = (observed_rate - desired_rate) / desired_rate
                    else:
                        relative_error = observed_rate

                    self.ath *= (1.0 + self.ath_alpha * relative_error)
                    self.ath = max(self.ath_min, min(self.ath_max, self.ath))

                    interval_spikes = 0
                    interval_targets = 0
                    interval_seen = 0

    def segment_text(self, text: str) -> Tuple[List[bytes], List[int], Dict[str, float]]:
        data = bytes_to_ints(text)

        patches: List[bytes] = []
        steps: List[int] = []
        cur: List[int] = []

        mem = 0.0

        for b in data:
            b = int(b)
            cur.append(b)
            mem = self.beta * mem + self.currents[b]

            effective_threshold = self.threshold + self.ath
            force = len(cur) >= self.max_patch_bytes
            spike = mem >= effective_threshold and len(cur) >= self.min_patch_bytes

            if force or spike:
                patches.append(bytes(cur))
                steps.append(1)
                cur = []
                mem = 0.0

        if cur:
            patches.append(bytes(cur))
            steps.append(1)

        stats = summarise_patches(patches, len(data), adaptive_steps=steps, segmenter_spike_native=True)
        stats.update(patch_interval_stats(patches))
        stats.update({
            "uses_proxy_boundary_teacher": 0.0,
            "proxy_free_boundary_targets": 1.0,
            "surrogate_spike_boundary_training": 0.0,
            "local_online_boundary_learning": 1.0,
            "local_homeostatic_boundary_learning": 0.0,
            "local_adaptive_selectivity_boundary_learning": 1.0,
            "spike_native_inference": 1.0,
            "end_to_end_lm_boundary_training": 0.0,
            "local_final_ath": float(self.ath),
            "local_effective_threshold": float(self.threshold + self.ath),
            "local_mean_current": float(sum(self.currents) / len(self.currents)),
            "hard_reset_count": float(len(patches)),
            "soft_reset_count": 0.0,
        })
        return patches, steps, stats
