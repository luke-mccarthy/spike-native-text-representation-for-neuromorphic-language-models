"""
BLT-style and Spike-BLT front-end utilities.

This file separates two related but different ideas:

1. BLT-style patch tokenizer
   raw text -> UTF-8 bytes -> dynamic byte patches -> patch IDs

   This behaves like a tokenizer and still requires model-side spike encoding.
   It can be tested with rate, threshold, and temporal encoders.

2. Spike-BLT front end
   raw text -> UTF-8 bytes -> dynamic byte patches -> patch-level spike embeddings

   This is a bridge toward spike-native byte/patch front ends. It can later be
   passed to models through inputs_embeds=..., bypassing or reducing a generic
   spike encoder.

This is not a full Byte Latent Transformer implementation. Full BLT uses a
learned next-byte entropy model for boundary selection. This first version uses
fixed, boundary-aware, and entropy-proxy patching so the interface can be tested
before adding learned entropy.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn


Patch = bytes


@dataclass
class BLTPatcherConfig:
    mode: str = "entropy_proxy"
    min_patch_bytes: int = 2
    max_patch_bytes: int = 16
    entropy_window: int = 16
    entropy_threshold: float = 3.0
    vocab_size: int = 8192
    unk_token: str = "<unk>"
    pad_token: str = "<pad>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"


@dataclass
class BLTEncoding:
    input_ids: Optional[List[int]]
    patch_lengths: List[int]
    num_bytes: int
    num_patches: int
    bytes_per_patch: float
    patches_per_byte: float


class BLTPatchVocabProxy:
    """
    BLT-style byte patch tokenizer.

    Produces discrete patch IDs, so it can be compared against char, byte,
    BPE, and Unigram tokenizers using the existing language-model pipeline.
    """

    def __init__(self, config: Optional[BLTPatcherConfig] = None):
        self.config = config or BLTPatcherConfig()
        self.patch_to_id: Dict[Patch, int] = {}
        self.id_to_patch: Dict[int, Patch] = {}

        if self.config.mode not in {"fixed", "boundary", "entropy_proxy"}:
            raise ValueError(f"Unsupported patch mode: {self.config.mode}")

        self.special_tokens = [
            self.config.pad_token,
            self.config.unk_token,
            self.config.bos_token,
            self.config.eos_token,
        ]

    @staticmethod
    def text_to_bytes(text: str) -> bytes:
        return text.encode("utf-8", errors="ignore")

    @staticmethod
    def bytes_to_text(data: bytes) -> str:
        return data.decode("utf-8", errors="ignore")

    @staticmethod
    def _is_boundary_byte(b: int) -> bool:
        return b in set(b" \n\r\t.,;:!?()[]{}\"'")

    @staticmethod
    def _shannon_entropy(byte_values: Sequence[int]) -> float:
        if not byte_values:
            return 0.0

        counts = Counter(byte_values)
        n = len(byte_values)
        ent = 0.0

        for c in counts.values():
            p = c / n
            ent -= p * math.log2(p)

        return ent

    def patch_text(self, text: str) -> List[Patch]:
        return self.patch_bytes(self.text_to_bytes(text))

    def patch_bytes(self, data: bytes) -> List[Patch]:
        if self.config.mode == "fixed":
            return self._patch_fixed(data)
        if self.config.mode == "boundary":
            return self._patch_boundary(data)
        if self.config.mode == "entropy_proxy":
            return self._patch_entropy_proxy(data)
        raise RuntimeError("Unreachable patch mode")

    def _patch_fixed(self, data: bytes) -> List[Patch]:
        max_len = self.config.max_patch_bytes
        return [
            data[i : i + max_len]
            for i in range(0, len(data), max_len)
            if data[i : i + max_len]
        ]

    def _patch_boundary(self, data: bytes) -> List[Patch]:
        patches: List[Patch] = []
        start = 0

        for i, b in enumerate(data):
            length = i - start + 1

            should_split = False

            if length >= self.config.max_patch_bytes:
                should_split = True
            elif length >= self.config.min_patch_bytes and self._is_boundary_byte(b):
                should_split = True

            if should_split:
                patch = data[start : i + 1]
                if patch:
                    patches.append(patch)
                start = i + 1

        if start < len(data):
            patches.append(data[start:])

        return patches

    def _patch_entropy_proxy(self, data: bytes) -> List[Patch]:
        """
        Dynamic byte patching using a cheap entropy proxy.

        True BLT uses learned next-byte entropy. This proxy uses local byte
        entropy plus boundary bytes. It is for early integration and ablations,
        not for final BLT claims.
        """
        patches: List[Patch] = []
        start = 0

        for i in range(len(data)):
            length = i - start + 1

            if length < self.config.min_patch_bytes:
                continue

            window_start = max(0, i - self.config.entropy_window + 1)
            window = data[window_start : i + 1]
            ent = self._shannon_entropy(window)

            should_split = False

            if length >= self.config.max_patch_bytes:
                should_split = True
            elif ent >= self.config.entropy_threshold and self._is_boundary_byte(data[i]):
                should_split = True

            if should_split:
                patch = data[start : i + 1]
                if patch:
                    patches.append(patch)
                start = i + 1

        if start < len(data):
            patches.append(data[start:])

        return patches

    def train_from_texts(self, texts: Iterable[str]) -> None:
        counter: Counter[Patch] = Counter()
        total_bytes = 0
        total_patches = 0

        for text in texts:
            data = self.text_to_bytes(str(text))
            patches = self.patch_bytes(data)
            counter.update(patches)
            total_bytes += len(data)
            total_patches += len(patches)

        self.patch_to_id.clear()
        self.id_to_patch.clear()

        for token in self.special_tokens:
            patch = token.encode("utf-8")
            idx = len(self.patch_to_id)
            self.patch_to_id[patch] = idx
            self.id_to_patch[idx] = patch

        remaining = max(self.config.vocab_size - len(self.special_tokens), 0)

        for patch, _count in counter.most_common(remaining):
            if patch in self.patch_to_id:
                continue
            idx = len(self.patch_to_id)
            self.patch_to_id[patch] = idx
            self.id_to_patch[idx] = patch

        print("Trained BLT-style patch vocabulary proxy")
        print(f"mode: {self.config.mode}")
        print(f"target_vocab_size: {self.config.vocab_size}")
        print(f"actual_vocab_size: {len(self.patch_to_id)}")
        print(f"observed_unique_patches: {len(counter)}")
        print(f"total_bytes: {total_bytes}")
        print(f"total_patches: {total_patches}")
        if total_patches:
            print(f"bytes_per_patch: {total_bytes / total_patches:.4f}")

    def encode(self, text: str, add_special: bool = False) -> List[int]:
        if not self.patch_to_id:
            raise RuntimeError("Patch vocabulary is empty. Train or load the patch vocabulary proxy first.")

        unk_id = self.patch_to_id[self.config.unk_token.encode("utf-8")]
        ids: List[int] = []

        if add_special:
            ids.append(self.patch_to_id[self.config.bos_token.encode("utf-8")])

        for patch in self.patch_text(text):
            ids.append(self.patch_to_id.get(patch, unk_id))

        if add_special:
            ids.append(self.patch_to_id[self.config.eos_token.encode("utf-8")])

        return ids

    def decode(self, ids: Sequence[int], skip_special: bool = True) -> str:
        special_ids = {
            self.patch_to_id[t.encode("utf-8")]
            for t in self.special_tokens
            if t.encode("utf-8") in self.patch_to_id
        }

        chunks: List[bytes] = []

        for idx in ids:
            idx = int(idx)
            if skip_special and idx in special_ids:
                continue
            chunks.append(self.id_to_patch.get(idx, self.config.unk_token.encode("utf-8")))

        return self.bytes_to_text(b"".join(chunks))

    def stats_for_text(self, text: str) -> Dict[str, float]:
        data = self.text_to_bytes(text)
        patches = self.patch_bytes(data)
        lengths = [len(p) for p in patches]

        if not patches:
            return {
                "num_bytes": len(data),
                "num_patches": 0,
                "bytes_per_patch": 0.0,
                "patches_per_byte": 0.0,
                "min_patch_bytes": 0,
                "max_patch_bytes": 0,
                "mean_patch_bytes": 0.0,
            }

        return {
            "num_bytes": len(data),
            "num_patches": len(patches),
            "bytes_per_patch": len(data) / len(patches),
            "patches_per_byte": len(patches) / max(len(data), 1),
            "min_patch_bytes": min(lengths),
            "max_patch_bytes": max(lengths),
            "mean_patch_bytes": sum(lengths) / len(lengths),
        }

    def encode_with_stats(self, text: str, add_special: bool = False) -> BLTEncoding:
        ids = self.encode(text, add_special=add_special)
        patches = self.patch_text(text)
        stats = self.stats_for_text(text)

        return BLTEncoding(
            input_ids=ids,
            patch_lengths=[len(p) for p in patches],
            num_bytes=int(stats["num_bytes"]),
            num_patches=int(stats["num_patches"]),
            bytes_per_patch=float(stats["bytes_per_patch"]),
            patches_per_byte=float(stats["patches_per_byte"]),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        obj = {
            "config": asdict(self.config),
            "id_to_patch": {
                str(idx): list(patch)
                for idx, patch in self.id_to_patch.items()
            },
        }

        path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
        print(f"Wrote {path}")

    @classmethod
    def load(cls, path: str | Path) -> "BLTPatchVocabProxy":
        path = Path(path)
        obj = json.loads(path.read_text(encoding="utf-8"))

        patcher = cls(BLTPatcherConfig(**obj["config"]))
        patcher.id_to_patch = {
            int(idx): bytes(values)
            for idx, values in obj["id_to_patch"].items()
        }
        patcher.patch_to_id = {
            patch: idx
            for idx, patch in patcher.id_to_patch.items()
        }

        return patcher


# Backwards-compatible alias for earlier scripts/imports.
# Prefer BLTPatchVocabProxy in new code.
BLTPatcher = BLTPatchVocabProxy


class SpikeBLTFrontend(nn.Module):
    """
    Spike-compatible BLT front end.

    This turns byte patches directly into patch-level spike embeddings. It is
    intended for future model calls using inputs_embeds=..., bypassing or
    reducing the need for a separate generic spike encoder.

    Current feature representation:
      - byte histogram over 256 possible byte values
      - patch length feature
      - projection to embedding dimension
      - threshold/rate spike generation
    """

    def __init__(
        self,
        embed_dim: int,
        max_patch_bytes: int = 16,
        spike_mode: str = "threshold",
    ):
        super().__init__()

        if spike_mode not in {"threshold", "rate", "none", "binary"}:
            raise ValueError(f"Unsupported spike_mode: {spike_mode}")

        self.embed_dim = embed_dim
        self.max_patch_bytes = max_patch_bytes
        self.spike_mode = spike_mode

        self.proj = nn.Linear(257, embed_dim)

    def patch_features(
        self,
        patches: Sequence[Patch],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        rows = []

        for patch in patches:
            hist = torch.zeros(256, dtype=torch.float32)

            if patch:
                vals = torch.tensor(list(patch), dtype=torch.long)
                hist.scatter_add_(0, vals, torch.ones_like(vals, dtype=torch.float32))
                hist = hist / max(len(patch), 1)

            length_feature = torch.tensor(
                [min(len(patch), self.max_patch_bytes) / max(self.max_patch_bytes, 1)],
                dtype=torch.float32,
            )

            rows.append(torch.cat([hist, length_feature], dim=0))

        if not rows:
            out = torch.zeros(0, 257, dtype=torch.float32)
        else:
            out = torch.stack(rows, dim=0)

        if device is not None:
            out = out.to(device)

        return out

    def forward(self, patches: Sequence[Patch]) -> torch.Tensor:
        features = self.patch_features(patches, device=self.proj.weight.device)
        dense = self.proj(features)

        if self.spike_mode == "none":
            return dense

        if self.spike_mode == "threshold":
            return (dense > 0).float()

        if self.spike_mode == "rate":
            return torch.bernoulli(torch.sigmoid(dense))

        if self.spike_mode == "binary":
            return (torch.sigmoid(dense) >= 0.5).float()

        raise RuntimeError("Unreachable spike mode")
