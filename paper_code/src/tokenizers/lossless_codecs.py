"""Lossless target and collision-free spike-interface codecs.

The default patch fallback is deliberately the historical literal-byte rule.
The optional ``optimal_training_subpatch`` rule is an explicitly selected,
training-vocabulary-only codec stabilization experiment: an unseen patch is
decomposed into the minimum number of known training patches and literal bytes.
It never learns from validation or test data and remains exactly reversible.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


BYTE_VOCAB_SIZE = 256
CODEC_FORMAT = "lossless_patch_codec_v1"
PATCH_FALLBACK_MODES = {"literal_bytes", "optimal_training_subpatch"}


@dataclass(frozen=True)
class PatchEncodingStats:
    input_patches: int
    input_bytes: int
    output_ids: int
    vocabulary_patch_hits: int
    fallback_patch_count: int
    fallback_byte_ids: int
    vocabulary_bytes_covered: int
    subpatch_vocabulary_hits: int = 0
    subpatch_vocabulary_bytes_covered: int = 0

    def as_dict(self) -> Dict[str, float | int]:
        effective_hits = self.vocabulary_patch_hits + self.subpatch_vocabulary_hits
        effective_bytes = (
            self.vocabulary_bytes_covered
            + self.subpatch_vocabulary_bytes_covered
        )
        return {
            "input_patches": self.input_patches,
            "input_bytes": self.input_bytes,
            "output_ids": self.output_ids,
            "vocabulary_patch_hits": self.vocabulary_patch_hits,
            "fallback_patch_count": self.fallback_patch_count,
            "fallback_byte_ids": self.fallback_byte_ids,
            "vocabulary_patch_coverage": self.vocabulary_patch_hits / max(self.input_patches, 1),
            "vocabulary_byte_coverage": self.vocabulary_bytes_covered / max(self.input_bytes, 1),
            "subpatch_vocabulary_hits": self.subpatch_vocabulary_hits,
            "subpatch_vocabulary_bytes_covered": self.subpatch_vocabulary_bytes_covered,
            "effective_vocabulary_id_hits": effective_hits,
            "effective_vocabulary_byte_coverage": effective_bytes / max(self.input_bytes, 1),
            "fallback_patch_rate": self.fallback_patch_count / max(self.input_patches, 1),
            "literal_fallback_byte_rate": self.fallback_byte_ids / max(self.input_bytes, 1),
            "ids_per_byte": self.output_ids / max(self.input_bytes, 1),
            "bytes_per_id": self.input_bytes / max(self.output_ids, 1),
        }


class LosslessPatchCodec:
    """Reversible patch ID codec with explicit fallback policy.

    IDs 0..255 always mean their corresponding raw byte. IDs from 256 upward
    represent frequent multi-byte patches observed in training. The historical
    fallback emits every byte of an unseen patch literally. The opt-in optimal
    fallback uses only this fitted training vocabulary and dynamic programming
    to minimize emitted IDs; ties minimize literal bytes and then use a stable
    ID ordering.
    """

    def __init__(
        self,
        patch_to_id: Mapping[bytes, int],
        target_vocab_size: int,
    ) -> None:
        if target_vocab_size < BYTE_VOCAB_SIZE:
            raise ValueError("target_vocab_size must reserve all 256 literal byte IDs")
        normalised = {bytes(patch): int(token_id) for patch, token_id in patch_to_id.items()}
        if any(len(patch) <= 1 for patch in normalised):
            raise ValueError("Vocabulary patch IDs are reserved for multi-byte patches")
        if len(set(normalised.values())) != len(normalised):
            raise ValueError("Patch IDs must be unique")
        expected_ids = list(range(BYTE_VOCAB_SIZE, BYTE_VOCAB_SIZE + len(normalised)))
        if sorted(normalised.values()) != expected_ids:
            raise ValueError("Patch IDs must be contiguous starting at 256")
        if BYTE_VOCAB_SIZE + len(normalised) > target_vocab_size:
            raise ValueError("Patch vocabulary exceeds target_vocab_size")
        self.patch_to_id = normalised
        self.id_to_patch = {token_id: patch for patch, token_id in normalised.items()}
        self.target_vocab_size = int(target_vocab_size)
        by_first: Dict[int, List[Tuple[bytes, int]]] = {}
        for patch, token_id in normalised.items():
            by_first.setdefault(patch[0], []).append((patch, token_id))
        self._patches_by_first_byte = {
            value: tuple(sorted(rows, key=lambda item: (-len(item[0]), item[1])))
            for value, rows in by_first.items()
        }

    @classmethod
    def fit(
        cls,
        training_patches: Sequence[bytes],
        target_vocab_size: int = 8192,
    ) -> "LosslessPatchCodec":
        if target_vocab_size < BYTE_VOCAB_SIZE:
            raise ValueError("target_vocab_size must be at least 256")
        if not training_patches:
            raise ValueError("Cannot fit a patch codec on an empty patch sequence")
        patches = [bytes(patch) for patch in training_patches]
        if any(not patch for patch in patches):
            raise ValueError("Training patches must be non-empty")
        counts = Counter(patch for patch in patches if len(patch) > 1)
        capacity = target_vocab_size - BYTE_VOCAB_SIZE
        ranked = sorted(counts, key=lambda patch: (-counts[patch], -len(patch), patch))
        selected = ranked[:capacity]
        mapping = {patch: BYTE_VOCAB_SIZE + index for index, patch in enumerate(selected)}
        return cls(mapping, target_vocab_size=target_vocab_size)

    @property
    def used_vocab_size(self) -> int:
        return BYTE_VOCAB_SIZE + len(self.patch_to_id)

    @property
    def model_vocab_size(self) -> int:
        return self.used_vocab_size

    def encode_patch_segments(
        self,
        patch: bytes,
        *,
        fallback_mode: str = "literal_bytes",
    ) -> List[Tuple[int, bytes]]:
        """Return aligned ``(id, raw payload)`` segments for one input patch."""

        value = bytes(patch)
        if not value:
            raise ValueError("Cannot encode an empty patch")
        if fallback_mode not in PATCH_FALLBACK_MODES:
            raise ValueError(f"unsupported patch fallback mode: {fallback_mode}")
        direct = self.patch_to_id.get(value)
        if direct is not None:
            return [(direct, value)]
        if fallback_mode == "literal_bytes":
            return [(byte, bytes((byte,))) for byte in value]

        # best[i] is (number of IDs, literal bytes, stable ID tuple, segments)
        # for value[i:]. Every suffix always has a literal-byte solution.
        best: List[Tuple[int, int, Tuple[int, ...], List[Tuple[int, bytes]]] | None] = [
            None for _ in range(len(value) + 1)
        ]
        best[len(value)] = (0, 0, (), [])
        for start in range(len(value) - 1, -1, -1):
            literal_id = int(value[start])
            suffix = best[start + 1]
            if suffix is None:
                raise AssertionError("literal fallback suffix is unavailable")
            candidates: List[
                Tuple[int, int, Tuple[int, ...], List[Tuple[int, bytes]]]
            ] = [
                (
                    1 + suffix[0],
                    1 + suffix[1],
                    (literal_id, *suffix[2]),
                    [(literal_id, bytes((literal_id,))), *suffix[3]],
                )
            ]
            for known, token_id in self._patches_by_first_byte.get(value[start], ()):
                end = start + len(known)
                if end > len(value) or value[start:end] != known:
                    continue
                known_suffix = best[end]
                if known_suffix is None:
                    continue
                candidates.append((
                    1 + known_suffix[0],
                    known_suffix[1],
                    (token_id, *known_suffix[2]),
                    [(token_id, known), *known_suffix[3]],
                ))
            best[start] = min(candidates, key=lambda row: (row[0], row[1], row[2]))
        result = best[0]
        if result is None:
            raise AssertionError("optimal subpatch encoding failed")
        if b"".join(payload for _, payload in result[3]) != value:
            raise AssertionError("optimal subpatch encoding lost raw bytes")
        return result[3]

    def encode_patches(
        self,
        patches: Sequence[bytes],
        *,
        fallback_mode: str = "literal_bytes",
    ) -> Tuple[List[int], PatchEncodingStats]:
        ids: List[int] = []
        input_bytes = 0
        vocabulary_hits = 0
        fallback_patches = 0
        fallback_byte_ids = 0
        vocabulary_bytes = 0
        subpatch_hits = 0
        subpatch_bytes = 0
        for value in patches:
            patch = bytes(value)
            if not patch:
                raise ValueError("Cannot encode an empty patch")
            input_bytes += len(patch)
            direct = self.patch_to_id.get(patch)
            segments = self.encode_patch_segments(patch, fallback_mode=fallback_mode)
            ids.extend(token_id for token_id, _ in segments)
            if direct is not None:
                vocabulary_hits += 1
                vocabulary_bytes += len(patch)
            else:
                fallback_patches += 1
                for token_id, payload in segments:
                    if token_id < BYTE_VOCAB_SIZE:
                        fallback_byte_ids += 1
                    else:
                        subpatch_hits += 1
                        subpatch_bytes += len(payload)
        stats = PatchEncodingStats(
            input_patches=len(patches),
            input_bytes=input_bytes,
            output_ids=len(ids),
            vocabulary_patch_hits=vocabulary_hits,
            fallback_patch_count=fallback_patches,
            fallback_byte_ids=fallback_byte_ids,
            vocabulary_bytes_covered=vocabulary_bytes,
            subpatch_vocabulary_hits=subpatch_hits,
            subpatch_vocabulary_bytes_covered=subpatch_bytes,
        )
        return ids, stats

    def decode_ids(self, ids: Iterable[int]) -> bytes:
        output = bytearray()
        for value in ids:
            token_id = int(value)
            if 0 <= token_id < BYTE_VOCAB_SIZE:
                output.append(token_id)
                continue
            patch = self.id_to_patch.get(token_id)
            if patch is None:
                raise ValueError(f"Unknown or unreachable patch token ID: {token_id}")
            output.extend(patch)
        return bytes(output)

    def assert_roundtrip(
        self,
        patches: Sequence[bytes],
        *,
        fallback_mode: str = "literal_bytes",
    ) -> PatchEncodingStats:
        ids, stats = self.encode_patches(patches, fallback_mode=fallback_mode)
        expected = b"".join(bytes(patch) for patch in patches)
        if self.decode_ids(ids) != expected:
            raise AssertionError("LosslessPatchCodec failed exact raw-byte round-trip")
        return stats

    def to_dict(self) -> Dict[str, object]:
        ordered = sorted(self.patch_to_id.items(), key=lambda item: item[1])
        return {
            "format": CODEC_FORMAT,
            "target_vocab_size": self.target_vocab_size,
            "used_vocab_size": self.used_vocab_size,
            "byte_id_range": [0, BYTE_VOCAB_SIZE - 1],
            "patches": [
                {"id": token_id, "hex": patch.hex(), "bytes": len(patch)}
                for patch, token_id in ordered
            ],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "LosslessPatchCodec":
        if payload.get("format") != CODEC_FORMAT:
            raise ValueError(f"Unsupported codec format: {payload.get('format')!r}")
        raw_patches = payload.get("patches")
        if not isinstance(raw_patches, list):
            raise ValueError("Codec payload is missing its patch list")
        mapping: Dict[bytes, int] = {}
        for row in raw_patches:
            if not isinstance(row, dict):
                raise ValueError("Invalid codec patch row")
            patch = bytes.fromhex(str(row["hex"]))
            token_id = int(row["id"])
            if patch in mapping:
                raise ValueError("Duplicate patch in codec payload")
            mapping[patch] = token_id
        return cls(mapping, target_vocab_size=int(payload["target_vocab_size"]))

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "LosslessPatchCodec":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def combination_capacity(dimension: int, active_bits: int) -> int:
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    if active_bits <= 0 or active_bits > dimension:
        raise ValueError("active_bits must satisfy 0 < active_bits <= dimension")
    return math.comb(dimension, active_bits)


class CombinationSpikeCodebook:
    """Deterministic collision-free fixed-weight binary spike codebook."""

    def __init__(
        self,
        vocab_size: int,
        dimension: int = 32,
        active_bits: int = 4,
        rank_multiplier: int = 2654435761,
        rank_offset: int = 1013904223,
    ) -> None:
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        capacity = combination_capacity(dimension, active_bits)
        if vocab_size > capacity:
            raise ValueError(
                f"vocab_size={vocab_size} exceeds C({dimension},{active_bits})={capacity}"
            )
        self.vocab_size = int(vocab_size)
        self.dimension = int(dimension)
        self.active_bits = int(active_bits)
        self.capacity = int(capacity)
        self.rank_multiplier = int(rank_multiplier) % self.capacity
        self.rank_offset = int(rank_offset) % self.capacity
        if math.gcd(self.rank_multiplier, self.capacity) != 1:
            raise ValueError("rank_multiplier must be coprime with codebook capacity")

    def channels(self, token_id: int) -> Tuple[int, ...]:
        token_id = int(token_id)
        if token_id < 0 or token_id >= self.vocab_size:
            raise ValueError(f"token_id outside codebook: {token_id}")
        rank = (self.rank_multiplier * token_id + self.rank_offset) % self.capacity
        selected: List[int] = []
        start = 0
        remaining = self.active_bits
        while remaining:
            for candidate in range(start, self.dimension):
                suffix = self.dimension - candidate - 1
                count = math.comb(suffix, remaining - 1) if suffix >= remaining - 1 else 0
                if rank < count:
                    selected.append(candidate)
                    start = candidate + 1
                    remaining -= 1
                    break
                rank -= count
            else:
                raise AssertionError("Combination unranking failed")
        return tuple(selected)

    def encode_ids(self, token_ids: Iterable[int]) -> List[Tuple[int, ...]]:
        return [self.channels(token_id) for token_id in token_ids]

    def channel_usage(self) -> List[int]:
        counts = [0 for _ in range(self.dimension)]
        for token_id in range(self.vocab_size):
            for channel in self.channels(token_id):
                counts[channel] += 1
        return counts

    def audit(self) -> Dict[str, int | float | str]:
        usage = self.channel_usage()
        usage_mean = sum(usage) / max(len(usage), 1)
        variance = sum((value - usage_mean) ** 2 for value in usage) / max(len(usage), 1)
        return {
            "encoding": "lexicographic_combination_code",
            "vocab_size": self.vocab_size,
            "dimension": self.dimension,
            "active_bits": self.active_bits,
            "capacity": self.capacity,
            "unused_patterns": self.capacity - self.vocab_size,
            "rank_multiplier": self.rank_multiplier,
            "rank_offset": self.rank_offset,
            "collisions": 0,
            "collision_fraction": 0.0,
            "channel_usage_min": min(usage),
            "channel_usage_max": max(usage),
            "channel_usage_mean": usage_mean,
            "channel_usage_coefficient_of_variation": math.sqrt(variance) / max(usage_mean, 1e-12),
        }
