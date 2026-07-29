#!/usr/bin/env python3
"""Audit canonical frontend representations without touching the locked test split.

This is a development-tier representation gate, not a language-model benchmark.
Every adaptive native frontend is fitted only on the canonical development
training view, frozen, and then applied twice to the canonical development
validation view. Conventional tokenizers must already have passed the canonical
training-only, exact-roundtrip audit.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import sentencepiece as spm
import torch

from src.tokenizers.lossless_codecs import CombinationSpikeCodebook, LosslessPatchCodec
from src.tokenizers.spike_boundary_refined import (
    SpikeBoundaryLIFRefinedSegmenter,
    stable_hash_patch,
)
from src.tokenizers.spike_boundary_native_levels import (
    EMAThresholdHomeostaticBoundarySegmenter,
    LocalAdaptiveSelectivityBoundarySegmenter,
    LocalHomeostaticBoundarySegmenter,
    SpikeBoundaryLIFRefractorySegmenter,
    SurrogateSurpriseSpikeBoundarySegmenter,
)


PROTOCOL_ID = "canonical_frontend_protocol_2026_07_16_v1"
CANONICAL_ROOT = Path(
    "/scratch/user/lukemccarthy/datasets/spike_tokenizer_canonical_2026_07_16"
)
TOKENIZER_ROOT = ROOT / "tokenizers/2026_07_16_canonical"
TARGET_RATES = (0.075, 0.100, 0.125)
TARGET_PATCH_VOCAB = 8192
SPIKE_DIMENSION = 32
ACTIVE_BITS = 4

DATASETS: Mapping[str, Mapping[str, str]] = {
    "tinystories_5000": {
        "train": "tinystories_5000/development/train_1000_stories.txt",
        "validation": "tinystories_5000/development/validation_100_stories.txt",
    },
    "text8": {
        "train": "text8/development/train_5m.txt",
        "validation": "text8/development/validation_1m.txt",
    },
    "wikitext2": {
        "train": "wikitext2/development/train_5m.txt",
        "validation": "wikitext2/development/validation_1m.txt",
    },
}
TOKENIZER_MODELS = ("bpe8000", "bpe16000", "unigram4000")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_strict_text(path: Path) -> Tuple[bytes, str]:
    data = path.read_bytes()
    if not data:
        raise ValueError(f"Empty canonical split: {path}")
    return data, data.decode("utf-8", errors="strict")


def percentile(values: Sequence[int], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * q))
    return float(ordered[max(0, min(len(ordered) - 1, index))])


def event_signature(patch: bytes, dimension: int = SPIKE_DIMENSION) -> Tuple[int, ...]:
    """Return the deterministic sparse event-ngram signature used for diagnostics."""
    if not patch:
        raise ValueError("Event signatures require non-empty patches")
    ngram = 3
    if len(patch) < ngram:
        hashes = [stable_hash_patch(patch, dimension)]
    else:
        hashes = [
            stable_hash_patch(patch[index : index + ngram], dimension)
            for index in range(len(patch) - ngram + 1)
        ]
    return tuple(sorted(set(hashes[:ACTIVE_BITS])))


def fitted_state_digest(segmenter: object) -> str:
    """Fingerprint fitted parameters that must remain unchanged at evaluation."""
    digest = hashlib.sha256()
    scalar_names = (
        "threshold", "initial_threshold", "beta", "ath", "target_boundary_rate",
        "training_boundary_rate", "training_ema_rate", "refractory_chars",
    )
    for name in scalar_names:
        if hasattr(segmenter, name):
            digest.update(name.encode("utf-8"))
            digest.update(repr(getattr(segmenter, name)).encode("utf-8"))
    currents = getattr(segmenter, "currents", None)
    if currents is not None:
        digest.update(json.dumps([float(value) for value in currents]).encode("utf-8"))
    model = getattr(segmenter, "model", None)
    if model is not None and hasattr(model, "state_dict"):
        for name, tensor in sorted(model.state_dict().items()):
            value = tensor.detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(repr(tuple(value.shape)).encode("ascii"))
            digest.update(value.numpy().tobytes())
    encoder = getattr(segmenter, "encoder", None)
    if encoder is not None and hasattr(encoder, "state_digest"):
        digest.update(b"nested_spike_native_encoder")
        digest.update(str(encoder.state_digest()).encode("ascii"))
    return digest.hexdigest()


def segment_rate(segmenter: object, text: str) -> float:
    patches, _, _ = segmenter.segment_text(text)
    raw_bytes = len(text.encode("utf-8", errors="strict"))
    return max(len(patches) - 1, 0) / max(raw_bytes, 1)


def calibrate_lif_threshold(
    training_text: str,
    target_rate: float,
    calibration_characters: int,
) -> Tuple[float, float]:
    sample = training_text[:calibration_characters]
    low, high = 0.5, 60.0
    for _ in range(12):
        threshold = 0.5 * (low + high)
        segmenter = SpikeBoundaryLIFRefinedSegmenter(
            threshold=threshold,
            beta=0.85,
            min_patch_bytes=4,
            max_patch_bytes=24,
        )
        if segment_rate(segmenter, sample) > target_rate:
            low = threshold
        else:
            high = threshold
    threshold = 0.5 * (low + high)
    final = SpikeBoundaryLIFRefinedSegmenter(
        threshold=threshold,
        beta=0.85,
        min_patch_bytes=4,
        max_patch_bytes=24,
    )
    return threshold, segment_rate(final, sample)


def verify_tokenizer_gate() -> Mapping[str, object]:
    summary_path = TOKENIZER_ROOT / "audit_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"Missing {summary_path}; run aggregate_2026_07_16_canonical_tokenizer_audits.py first"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    required = (
        summary.get("format") == "canonical_tokenizer_audit_summary_v1",
        int(summary.get("completed_models", 0)) == 9,
        bool(summary.get("all_fitted_on_training_only")),
        bool(summary.get("all_vocab_files_valid")),
        bool(summary.get("all_split_roundtrips_exact")),
        bool(summary.get("all_unknown_id_counts_zero")),
    )
    if not all(required):
        raise AssertionError("Canonical tokenizer audit summary did not pass every gate")
    return summary


def conventional_rows(
    dataset: str,
    validation_bytes: bytes,
    validation_text: str,
    common: Mapping[str, object],
) -> Iterable[Dict[str, object]]:
    specifications: List[Tuple[str, List[int], int, str]] = [
        ("byte", list(validation_bytes), 256, ""),
    ]
    for model_name in TOKENIZER_MODELS:
        model_path = TOKENIZER_ROOT / f"{dataset}_{model_name}.model"
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        processor = spm.SentencePieceProcessor(model_file=str(model_path))
        ids = list(processor.encode(validation_text, out_type=int))
        decoded = processor.decode(ids).encode("utf-8", errors="strict")
        if decoded != validation_bytes:
            raise AssertionError(f"Validation round-trip failed for {model_path}")
        specifications.append((model_name, ids, int(processor.get_piece_size()), str(model_path)))

    for method, ids, vocab_size, model_path in specifications:
        codebook = CombinationSpikeCodebook(
            vocab_size=vocab_size,
            dimension=SPIKE_DIMENSION,
            active_bits=ACTIVE_BITS,
        )
        code_audit = codebook.audit()
        yield {
            **common,
            "family": "conventional_tokenizer",
            "method": method,
            "training_signal": "training-free" if method == "byte" else "corpus-unsupervised",
            "tokenizer_model": model_path,
            "lossless_raw_byte_roundtrip": 1,
            "evaluation_frozen": 1,
            "evaluation_repeat_deterministic": 1,
            "num_modelling_units": len(ids),
            "bytes_per_modelling_unit": len(validation_bytes) / max(len(ids), 1),
            "modelling_units_per_byte": len(ids) / max(len(validation_bytes), 1),
            "model_vocab_size": vocab_size,
            "interface_code": "collision_free_4_of_32",
            "interface_spikes_per_byte": ACTIVE_BITS * len(ids) / max(len(validation_bytes), 1),
            "interface_code_collisions": int(code_audit["collisions"]),
            "interface_channel_usage_cv": code_audit["channel_usage_coefficient_of_variation"],
            "quality_metric_status": "not_computed_representation_gate_only",
        }


def native_row(
    dataset: str,
    method: str,
    training_signal: str,
    segmenter_factory: Callable[[], object],
    training_bytes: bytes,
    training_text: str,
    validation_bytes: bytes,
    validation_text: str,
    common: Mapping[str, object],
    target_rate: float | None = None,
    fit_extra: Mapping[str, object] | None = None,
) -> Dict[str, object]:
    started = time.perf_counter()
    segmenter = segmenter_factory()
    fit_seconds = time.perf_counter() - started

    train_started = time.perf_counter()
    training_patches, _, _ = segmenter.segment_text(training_text)
    train_segment_seconds = time.perf_counter() - train_started
    if b"".join(training_patches) != training_bytes:
        raise AssertionError(f"Training segmentation is not lossless: {dataset}/{method}")

    codec_started = time.perf_counter()
    codec = LosslessPatchCodec.fit(training_patches, target_vocab_size=TARGET_PATCH_VOCAB)
    codec_fit_seconds = time.perf_counter() - codec_started
    del training_patches
    gc.collect()

    state_before = fitted_state_digest(segmenter)
    validation_started = time.perf_counter()
    validation_patches, _, stats = segmenter.segment_text(validation_text)
    validation_segment_seconds = time.perf_counter() - validation_started
    state_after_first = fitted_state_digest(segmenter)
    repeated_patches, _, _ = segmenter.segment_text(validation_text)
    state_after_second = fitted_state_digest(segmenter)

    if state_before != state_after_first or state_before != state_after_second:
        raise AssertionError(f"Evaluation mutated fitted state: {dataset}/{method}")
    if validation_patches != repeated_patches:
        raise AssertionError(f"Evaluation is not deterministic: {dataset}/{method}")
    if b"".join(validation_patches) != validation_bytes:
        raise AssertionError(f"Validation segmentation is not lossless: {dataset}/{method}")

    ids, codec_stats = codec.encode_patches(validation_patches)
    if codec.decode_ids(ids) != validation_bytes:
        raise AssertionError(f"Codec round-trip failed: {dataset}/{method}")
    codebook = CombinationSpikeCodebook(
        vocab_size=codec.model_vocab_size,
        dimension=SPIKE_DIMENSION,
        active_bits=ACTIVE_BITS,
    )
    code_audit = codebook.audit()

    lengths = [len(patch) for patch in validation_patches]
    counts = Counter(validation_patches)
    signatures: Dict[Tuple[int, ...], int] = Counter(
        event_signature(patch) for patch in counts
    )
    colliding_types = sum(count for count in signatures.values() if count > 1)
    event_spikes = sum(len(event_signature(patch)) for patch in validation_patches)
    boundary_rate = max(len(validation_patches) - 1, 0) / max(len(validation_bytes), 1)

    row: Dict[str, object] = {
        **common,
        "family": "spike_native_boundary",
        "method": method,
        "training_signal": training_signal,
        "lossless_raw_byte_roundtrip": 1,
        "evaluation_frozen": 1,
        "evaluation_repeat_deterministic": 1,
        "fitted_state_sha256": state_before,
        "target_boundary_rate": "" if target_rate is None else target_rate,
        "validation_boundary_rate": boundary_rate,
        "absolute_target_rate_error": "" if target_rate is None else abs(boundary_rate - target_rate),
        "num_patches": len(validation_patches),
        "mean_patch_bytes": statistics.fmean(lengths) if lengths else 0.0,
        "median_patch_bytes": percentile(lengths, 0.50),
        "p95_patch_bytes": percentile(lengths, 0.95),
        "max_patch_bytes_observed": max(lengths) if lengths else 0,
        "unique_patch_types": len(counts),
        "model_vocab_size": codec.model_vocab_size,
        **codec_stats.as_dict(),
        "interface_code": "lossless_patch_codec_plus_collision_free_4_of_32",
        "interface_spikes_per_byte": ACTIVE_BITS * len(ids) / max(len(validation_bytes), 1),
        "interface_code_collisions": int(code_audit["collisions"]),
        "interface_channel_usage_cv": code_audit["channel_usage_coefficient_of_variation"],
        "event_ngram_spikes_per_byte": event_spikes / max(len(validation_bytes), 1),
        "event_ngram_unique_type_collision_fraction": colliding_types / max(len(counts), 1),
        "fit_seconds": fit_seconds,
        "training_segmentation_seconds": train_segment_seconds,
        "codec_fit_seconds": codec_fit_seconds,
        "validation_segmentation_seconds": validation_segment_seconds,
        "quality_metric_status": "not_computed_representation_gate_only",
    }
    for name in (
        "homeostatic_final_threshold", "homeostatic_training_boundary_rate",
        "homeostatic_training_ema_rate", "homeostatic_validation_boundary_rate",
        "intrinsic_target_boundary_rate", "surrogate_train_loss_initial",
        "surrogate_train_loss_final", "surrogate_train_gradient_norm_min",
        "surrogate_train_gradient_norm_max", "refractory_suppressed_per_char",
    ):
        if name in stats:
            row[name] = stats[name]
    if fit_extra:
        row.update(fit_extra)
    del repeated_patches, validation_patches, ids, segmenter
    gc.collect()
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--calibration-characters", type=int, default=300_000)
    parser.add_argument("--surrogate-batches", type=int, default=20)
    args = parser.parse_args()

    verify_tokenizer_gate()
    manifest = json.loads((CANONICAL_ROOT / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("protocol_id") != PROTOCOL_ID:
        raise AssertionError("Canonical manifest protocol mismatch")

    paths = DATASETS[args.dataset]
    training_path = CANONICAL_ROOT / paths["train"]
    validation_path = CANONICAL_ROOT / paths["validation"]
    training_bytes, training_text = load_strict_text(training_path)
    validation_bytes, validation_text = load_strict_text(validation_path)
    common: Dict[str, object] = {
        "protocol_id": PROTOCOL_ID,
        "tier": "development_representation_gate",
        "dataset": args.dataset,
        "training_path": str(training_path),
        "validation_path": str(validation_path),
        "training_bytes": len(training_bytes),
        "validation_bytes": len(validation_bytes),
        "training_sha256": sha256_bytes(training_bytes),
        "validation_sha256": sha256_bytes(validation_bytes),
        "fit_uses_training_only": 1,
        "test_split_touched": 0,
    }
    rows: List[Dict[str, object]] = list(
        conventional_rows(args.dataset, validation_bytes, validation_text, common)
    )

    rows.append(native_row(
        args.dataset, "lif_fixed_default", "training-free",
        lambda: SpikeBoundaryLIFRefinedSegmenter(
            threshold=7.5, beta=0.85, min_patch_bytes=4, max_patch_bytes=24
        ), training_bytes, training_text, validation_bytes, validation_text, common,
    ))
    rows.append(native_row(
        args.dataset, "lif_refractory6", "training-free",
        lambda: SpikeBoundaryLIFRefractorySegmenter(
            threshold=7.5, beta=0.85, min_patch_bytes=4,
            max_patch_bytes=24, refractory_chars=6,
        ), training_bytes, training_text, validation_bytes, validation_text, common,
    ))

    for target_rate in TARGET_RATES:
        threshold, calibration_rate = calibrate_lif_threshold(
            training_text, target_rate, args.calibration_characters
        )
        rows.append(native_row(
            args.dataset, f"lif_rate_calibrated_{target_rate:.3f}",
            "corpus-unsupervised activity calibration",
            lambda threshold=threshold: SpikeBoundaryLIFRefinedSegmenter(
                threshold=threshold, beta=0.85, min_patch_bytes=4, max_patch_bytes=24
            ), training_bytes, training_text, validation_bytes, validation_text, common,
            target_rate=target_rate,
            fit_extra={"calibrated_threshold": threshold, "calibration_boundary_rate": calibration_rate},
        ))
        rows.append(native_row(
            args.dataset, f"ema_homeostatic_{target_rate:.3f}",
            "activity-homeostatic unsupervised",
            lambda target_rate=target_rate: EMAThresholdHomeostaticBoundarySegmenter(
                full_text=training_text,
                target_boundary_rate=target_rate,
                passes=1,
                threshold=7.5,
                beta=0.85,
                min_patch_bytes=4,
                max_patch_bytes=24,
                ema_alpha=0.001,
                threshold_lr=0.05,
            ), training_bytes, training_text, validation_bytes, validation_text, common,
            target_rate=target_rate,
        ))

    rows.append(native_row(
        args.dataset, "legacy_intrinsic_homeostatic_0.100",
        "intrinsic pseudo-target self-supervision",
        lambda: LocalHomeostaticBoundarySegmenter(
            full_text=training_text, passes=1, seed=1, threshold=7.5, beta=0.85,
            min_patch_bytes=4, max_patch_bytes=24, surprise_quantile=0.88,
            lr_current=0.005, lr_threshold=0.0003, target_boundary_rate=0.100,
        ), training_bytes, training_text, validation_bytes, validation_text, common,
        target_rate=0.100,
    ))
    rows.append(native_row(
        args.dataset, "intrinsic_adaptive_selectivity",
        "intrinsic pseudo-target self-supervision",
        lambda: LocalAdaptiveSelectivityBoundarySegmenter(
            full_text=training_text, passes=1, seed=1, threshold=7.5, beta=0.85,
            min_patch_bytes=4, max_patch_bytes=24, surprise_quantile=0.88,
            lr_current=0.005, ath_init=1.0, ath_alpha=0.05,
            ath_max=20.0, logging_interval=10_000,
        ), training_bytes, training_text, validation_bytes, validation_text, common,
    ))
    for seed in (1, 2, 3):
        rows.append(native_row(
            args.dataset, f"surrogate_surprise_seed{seed}",
            "intrinsic pseudo-target surrogate backpropagation",
            lambda seed=seed: SurrogateSurpriseSpikeBoundarySegmenter(
                full_text=training_text, train_batches=args.surrogate_batches,
                window=512, lr=0.01, seed=seed, threshold=7.5, beta=0.85,
                min_patch_bytes=4, max_patch_bytes=24,
                surprise_quantile=0.88, device="cpu",
            ), training_bytes, training_text, validation_bytes, validation_text, common,
            fit_extra={"seed": seed, "surrogate_batches": args.surrogate_batches},
        ))

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({name for row in rows for name in row})
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "format": "canonical_frontend_representation_audit_v1",
        "protocol_id": PROTOCOL_ID,
        "dataset": args.dataset,
        "test_split_touched": False,
        "rows": len(rows),
        "output_csv": str(output),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "quality_metrics_computed": False,
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"PASS: canonical frozen representation audit for {args.dataset}")
    print(f"Rows: {len(rows)}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
