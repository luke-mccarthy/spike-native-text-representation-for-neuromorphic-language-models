#!/usr/bin/env python3
"""Canonical lossless frontend LM pilot with held-out BPB/BPC evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import resource
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import sentencepiece as spm
import torch
import torch.nn.functional as F

from scripts.audit_2026_07_16_canonical_frontend_representations import (
    calibrate_lif_threshold,
    fitted_state_digest,
)
from src.canonical_data_contract import (
    assert_train_test_contract,
    assert_training_only_contract,
    assert_train_validation_contract,
    sentencepiece_training_provenance,
)
from src.energy.analytical_compute import experiment_compute_energy
from src.energy.frontend_memory import experiment_frontend_memory_ledger
from src.models.spike_input_interfaces import (
    INJECTIVE_RECURRENT_BYTE_MODE,
    build_spike_input_interface,
)
from src.tokenizers.spike_native_patch_frontends import (
    JointSpikingPatchSegmenter,
    SPIKEPATCH_INTERFACE_MODES,
    merge_frontend_operation_stats,
    payload_table_from_patch_codec,
)
from src.tokenizers.spiking_blt_frontend import (
    SpikingBLTConfig,
    SpikingBLTSurpriseSegmenter,
    spiking_blt_method_spec,
)
from src.models.spikegpt_operation_accounting import (
    spikegpt_reference_operation_ledger,
)
from src.models.spikegpt_reference import build_spikegpt_reference_lm
from src.models.spiking_lm_models import build_spiking_lm
from src.models.srwkv_operation_accounting import (
    srwkv_v2_operation_ledger,
    training_operation_estimates,
)
from src.models.strict_spiking_backbones_v2 import build_strict_spike_rwkv_v2_lm
from src.tokenizers.blt_patcher import BLTPatcherConfig, BLTPatchVocabProxy
from src.tokenizers.lossless_codecs import LosslessPatchCodec
from src.tokenizers.gpt_neox_20b import (
    OFFICIAL_SPIKEGPT_TOKENIZER_BLOB_SHA,
    OFFICIAL_SPIKEGPT_TOKENIZER_SOURCE,
    OfficialSpikeGPT20BTokenizer,
)
from src.tokenizers.spike_boundary_refined import SpikeBoundaryLIFRefinedSegmenter
from src.tokenizers.spike_boundary_native_levels import (
    EMAThresholdHomeostaticBoundarySegmenter,
    LocalAdaptiveSelectivityBoundarySegmenter,
    LocalHomeostaticBoundarySegmenter,
    SpikeBoundaryLIFRefractorySegmenter,
    TeacherBLTRefractorySegmenter,
)


PROTOCOL_ID = "canonical_frontend_protocol_2026_07_16_v1"
CANONICAL_ROOT = Path(
    "/scratch/user/lukemccarthy/datasets/spike_tokenizer_canonical_2026_07_16"
)
TOKENIZER_ROOT = ROOT / "tokenizers/2026_07_16_canonical"
OFFICIAL_SPIKEGPT_TOKENIZER_PATH = (
    ROOT / "assets/tokenizers/spikegpt_20B_tokenizer.json"
)
DATASET_TIERS: Mapping[str, Mapping[str, Mapping[str, str]]] = {
    "development": {
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
    },
    "full_train_validation": {
        dataset: {
            "train": f"{dataset}/train.txt",
            "validation": f"{dataset}/validation.txt",
            "test": f"{dataset}/test.txt",
        }
        for dataset in ("tinystories_5000", "text8", "wikitext2")
    },
}
# Backward-compatible dataset-name view used by older importers and tests.
DATASETS = DATASET_TIERS["development"]
CONVENTIONAL_MODELS = {
    "bpe8000": "bpe8000",
    "bpe16000": "bpe16000",
    "unigram4000": "unigram4000",
}
EXTERNAL_CONVENTIONAL_MODELS = {
    "spikegpt_20b_bpe": OFFICIAL_SPIKEGPT_TOKENIZER_PATH,
}
ALL_CONVENTIONAL_METHODS = frozenset(
    (*CONVENTIONAL_MODELS, *EXTERNAL_CONVENTIONAL_MODELS)
)


@dataclass
class UnitSequence:
    ids: List[int]
    byte_lengths: List[int]
    character_lengths: List[int]
    vocab_size: int
    representation: str
    metadata: Dict[str, object]
    payloads_by_id: Tuple[bytes, ...] | None = None

    def validate(self, expected_bytes: int, expected_characters: int) -> None:
        if not (len(self.ids) == len(self.byte_lengths) == len(self.character_lengths)):
            raise AssertionError("Unit IDs and raw-length accounting are misaligned")
        if len(self.ids) < 3:
            raise ValueError("Unit sequence is too short")
        if min(self.ids) < 0 or max(self.ids) >= self.vocab_size:
            raise AssertionError("Unit ID falls outside the model vocabulary")
        if sum(self.byte_lengths) != expected_bytes:
            raise AssertionError("Unit byte lengths do not cover the exact raw bytes")
        if sum(self.character_lengths) != expected_characters:
            raise AssertionError("Unit character lengths do not cover the exact text")
        if self.payloads_by_id is not None:
            if len(self.payloads_by_id) != self.vocab_size:
                raise AssertionError("Raw payload table must align with every reachable ID")
            if any(not isinstance(payload, bytes) for payload in self.payloads_by_id):
                raise AssertionError("Raw payload table entries must be byte strings")


def prepend_shared_bos(sequence: UnitSequence) -> UnitSequence:
    """Add a shared zero-byte BOS target context without losing raw accounting."""
    bos_id = sequence.vocab_size
    metadata = dict(sequence.metadata)
    metadata.update({
        "shared_bos_added": 1,
        "bos_id": bos_id,
        "pre_bos_vocab_size": sequence.vocab_size,
    })
    return UnitSequence(
        ids=[bos_id, *sequence.ids],
        byte_lengths=[0, *sequence.byte_lengths],
        character_lengths=[0, *sequence.character_lengths],
        vocab_size=sequence.vocab_size + 1,
        representation=sequence.representation,
        metadata=metadata,
        payloads_by_id=(
            (*sequence.payloads_by_id, b"")
            if sequence.payloads_by_id is not None
            else None
        ),
    )


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


OPENED_DATA_PATHS: List[Tuple[str, str]] = []


def read_strict(path: Path, *, role: str = "unspecified") -> Tuple[bytes, str]:
    if role not in {"training", "validation", "test", "unspecified"}:
        raise ValueError(f"unsupported data role: {role}")
    resolved = str(path.resolve())
    OPENED_DATA_PATHS.append((role, resolved))
    data = path.read_bytes()
    if not data:
        raise ValueError(f"Empty canonical split: {path}")
    return data, data.decode("utf-8", errors="strict")


def utf8_safe_prefix(raw: bytes, requested_bytes: int) -> Tuple[bytes, str]:
    """Return the longest valid UTF-8 prefix within a raw-byte budget."""
    if requested_bytes <= 0:
        raise ValueError("common-prefix training requires a positive byte budget")
    selected = raw[: min(int(requested_bytes), len(raw))]
    try:
        return selected, selected.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        if exc.end != len(selected):
            raise AssertionError("canonical UTF-8 failed before the prefix boundary") from exc
        selected = selected[: exc.start]
        return selected, selected.decode("utf-8", errors="strict")


def utf8_byte_character_completions(text: str) -> List[int]:
    flags: List[int] = []
    for character in text:
        encoded = character.encode("utf-8", errors="strict")
        flags.extend([0] * (len(encoded) - 1))
        flags.append(1)
    return flags


def byte_sequence(raw: bytes, text: str) -> UnitSequence:
    flags = utf8_byte_character_completions(text)
    if len(flags) != len(raw):
        raise AssertionError("UTF-8 character accounting mismatch")
    return UnitSequence(
        ids=list(raw),
        byte_lengths=[1] * len(raw),
        character_lengths=flags,
        vocab_size=256,
        representation="raw_byte",
        metadata={"lossless_raw_byte_roundtrip": 1},
        payloads_by_id=tuple(bytes((value,)) for value in range(256)),
    )


def sentencepiece_sequence(
    method: str,
    raw: bytes,
    text: str,
    *,
    processor,
    model_path: Path,
    tokenizer_provenance: Mapping[str, object],
) -> Tuple[UnitSequence, Dict[str, float]]:
    started = time.perf_counter()
    cpu_started = time.process_time()
    proto = processor.encode_as_immutable_proto(text)
    ids: List[int] = []
    surfaces: List[str] = []
    for piece in proto.pieces:
        ids.append(int(piece.id))
        surfaces.append(str(piece.surface))
    byte_lengths = [len(surface.encode("utf-8", errors="strict")) for surface in surfaces]
    character_lengths = [len(surface) for surface in surfaces]
    encode_seconds = time.perf_counter() - started
    encode_cpu_seconds = time.process_time() - cpu_started
    if "".join(surfaces) != text:
        raise AssertionError(f"SentencePiece surfaces do not reconstruct input: {model_path}")
    if processor.decode(ids).encode("utf-8", errors="strict") != raw:
        raise AssertionError(f"SentencePiece decode is not an exact raw-byte round-trip: {model_path}")
    sequence = UnitSequence(
        ids=ids,
        byte_lengths=byte_lengths,
        character_lengths=character_lengths,
        vocab_size=int(processor.get_piece_size()),
        representation=method,
        metadata={
            "lossless_raw_byte_roundtrip": 1,
            "tokenizer_model": str(model_path),
            "tokenizer_model_file_bytes": model_path.stat().st_size,
            "tokenizer_model_sha256": sha256(model_path.read_bytes()),
            "tokenizer_fitted_on_canonical_training_only": 1,
            **tokenizer_provenance,
        },
    )
    return sequence, {
        "encode_seconds": encode_seconds,
        "encode_cpu_seconds": encode_cpu_seconds,
    }


def official_spikegpt_sequence(
    tokenizer: OfficialSpikeGPT20BTokenizer,
    raw: bytes,
    text: str,
) -> UnitSequence:
    """Encode with the exact 20B tokenizer configuration vendored by SpikeGPT."""

    encoded = tokenizer.encode(text)
    byte_lengths = [len(payload) for payload in encoded.token_bytes]
    flags = utf8_byte_character_completions(text)
    character_lengths: List[int] = []
    offset = 0
    for length in byte_lengths:
        character_lengths.append(sum(flags[offset:offset + length]))
        offset += length
    if offset != len(raw) or b"".join(encoded.token_bytes) != raw:
        raise AssertionError("SpikeGPT 20B tokenization changed canonical bytes")
    return UnitSequence(
        ids=list(encoded.ids),
        byte_lengths=byte_lengths,
        character_lengths=character_lengths,
        vocab_size=tokenizer.vocab_size,
        representation="spikegpt_20b_bpe",
        metadata={
            "lossless_raw_byte_roundtrip": 1,
            "tokenizer_model": str(OFFICIAL_SPIKEGPT_TOKENIZER_PATH),
            "tokenizer_model_file_bytes": OFFICIAL_SPIKEGPT_TOKENIZER_PATH.stat().st_size,
            "tokenizer_model_sha256": tokenizer.sha256,
            "tokenizer_model_git_blob_sha": OFFICIAL_SPIKEGPT_TOKENIZER_BLOB_SHA,
            "tokenizer_model_source": OFFICIAL_SPIKEGPT_TOKENIZER_SOURCE,
            "tokenizer_fitted_on_canonical_training_only": 0,
            "tokenizer_is_external_pretrained_baseline": 1,
            "external_tokenizer_training_overlap_with_evaluation_not_audited": 1,
            **{
                f"tokenizer_{key}": value
                for key, value in encoded.statistics.items()
            },
        },
    )


def proxy_patcher() -> BLTPatchVocabProxy:
    return BLTPatchVocabProxy(BLTPatcherConfig(
        mode="entropy_proxy",
        min_patch_bytes=4,
        max_patch_bytes=24,
        entropy_window=16,
        entropy_threshold=3.0,
        vocab_size=8192,
    ))


def build_segmenter(method: str, training_text: str, seed: int):
    if method == "spike_blt_surprise_fixed" or method.startswith(
        "spike_blt_surprise_surrogate_"
    ):
        training_bytes = training_text.encode("utf-8", errors="strict")
        learning_rule, target_boundary_rate = spiking_blt_method_spec(method)
        segmenter = SpikingBLTSurpriseSegmenter(
            learning_rule=learning_rule,
            seed=seed,
            training_bytes=(
                training_bytes if learning_rule == "predictive_surrogate" else None
            ),
            training_sha256=sha256(training_bytes),
            config=SpikingBLTConfig(
                seed=seed,
                target_boundary_rate_per_byte=target_boundary_rate,
                fit_byte_limit=len(training_bytes),
            ),
        )
        return segmenter, {
            "training_signal": (
                "training_free_fixed_spiking_predictor"
                if learning_rule == "fixed"
                else "training_only_surrogate_next_byte_spike_prediction"
            ),
            "paper_role": "stage_c_blt_inspired_extension",
            "segmenter_spike_native": 1,
            "blt_inspired_not_complete_blt_conversion": 1,
            "boundary_current_source": "next_byte_prediction_error_spikes",
            "predictor_communication_is_binary_spikes": 1,
            "lossless_patch_target_serialisation_is_digital": 1,
            "target_boundary_rate": target_boundary_rate,
            **segmenter.fit_report,
        }
    if method == "joint_snn_fixed":
        return JointSpikingPatchSegmenter(
            learning_rule="fixed",
            seed=seed,
        ), {
            "training_signal": "training-free deterministic sparse reservoir",
            "paper_role": "stage_c_joint_spike_native_candidate",
            "segmenter_spike_native": 1,
            "boundary_current_source": "recurrent_hidden_spikes_only",
            "joint_content_codes_currently_qualification_only": 1,
        }
    if method == "joint_snn_homeostatic":
        training_bytes = training_text.encode("utf-8", errors="strict")
        return JointSpikingPatchSegmenter(
            learning_rule="homeostatic",
            seed=seed,
            training_bytes=training_bytes,
            training_sha256=sha256(training_bytes),
        ), {
            "training_signal": "training-only local firing-rate homeostasis",
            "paper_role": "stage_c_joint_spike_native_candidate",
            "segmenter_spike_native": 1,
            "boundary_current_source": "recurrent_hidden_spikes_only",
            "joint_content_codes_currently_qualification_only": 1,
        }
    if method == "proxy_blt":
        return proxy_patcher(), {
            "training_signal": "algorithmic entropy proxy",
            "paper_role": "bridge_ablation",
            "segmenter_spike_native": 0,
        }
    if method == "teacher_blt_refrac":
        patcher = proxy_patcher()
        teacher_patches = patcher.patch_text(training_text)
        return TeacherBLTRefractorySegmenter(
            teacher_patches=teacher_patches,
            full_text=training_text,
            train_batches=40,
            window=512,
            lr=0.01,
            seed=seed,
            device="cpu",
            refractory_chars=6,
            threshold=1.0,
            beta=0.85,
            min_patch_bytes=4,
            max_patch_bytes=24,
        ), {
            "training_signal": "external proxy teacher distillation",
            "paper_role": "bridge_upper_bound_ablation",
            "segmenter_spike_native": 1,
        }
    if method == "lif_fixed":
        return SpikeBoundaryLIFRefinedSegmenter(
            threshold=7.5, beta=0.85, min_patch_bytes=4, max_patch_bytes=24
        ), {"training_signal": "training-free", "paper_role": "primary_spike_native", "segmenter_spike_native": 1}
    if method == "lif_refrac":
        return SpikeBoundaryLIFRefractorySegmenter(
            threshold=7.5, beta=0.85, min_patch_bytes=4,
            max_patch_bytes=24, refractory_chars=6,
        ), {"training_signal": "training-free", "paper_role": "primary_spike_native", "segmenter_spike_native": 1}
    if method == "lif_calibrated_0.125":
        threshold, rate = calibrate_lif_threshold(training_text, 0.125, 300_000)
        return SpikeBoundaryLIFRefinedSegmenter(
            threshold=threshold, beta=0.85, min_patch_bytes=4, max_patch_bytes=24
        ), {
            "training_signal": "corpus-unsupervised activity calibration",
            "paper_role": "primary_spike_native",
            "segmenter_spike_native": 1,
            "calibrated_threshold": threshold,
            "calibration_boundary_rate": rate,
            "target_boundary_rate": 0.125,
        }
    if method == "ema_homeostatic_0.125":
        return EMAThresholdHomeostaticBoundarySegmenter(
            full_text=training_text,
            target_boundary_rate=0.125,
            passes=1,
            threshold=7.5,
            beta=0.85,
            min_patch_bytes=4,
            max_patch_bytes=24,
            ema_alpha=0.001,
            threshold_lr=0.05,
        ), {
            "training_signal": "activity-homeostatic unsupervised",
            "paper_role": "primary_spike_native",
            "segmenter_spike_native": 1,
            "target_boundary_rate": 0.125,
        }
    if method.startswith("legacy_homeostatic_"):
        try:
            target_boundary_rate = float(method.rsplit("_", 1)[1])
        except ValueError as exc:
            raise ValueError(f"invalid homeostatic target in {method}") from exc
        if target_boundary_rate not in {0.075, 0.100, 0.125}:
            raise ValueError(
                "homeostatic target is outside the predeclared tuning grid"
            )
        return LocalHomeostaticBoundarySegmenter(
            full_text=training_text, passes=1, seed=seed,
            threshold=7.5, beta=0.85, min_patch_bytes=4, max_patch_bytes=24,
            surprise_quantile=0.88, lr_current=0.005,
            lr_threshold=0.0003, target_boundary_rate=target_boundary_rate,
        ), {
            "training_signal": "intrinsic pseudo-target self-supervision",
            "paper_role": "spike_native_self_supervised_ablation",
            "segmenter_spike_native": 1,
            "target_boundary_rate": target_boundary_rate,
        }
    if method == "adaptive_selectivity":
        return LocalAdaptiveSelectivityBoundarySegmenter(
            full_text=training_text, passes=1, seed=seed,
            threshold=7.5, beta=0.85, min_patch_bytes=4, max_patch_bytes=24,
            surprise_quantile=0.88, lr_current=0.005,
            ath_init=1.0, ath_alpha=0.05, ath_max=20.0,
            logging_interval=10_000,
        ), {
            "training_signal": "intrinsic pseudo-target self-supervision",
            "paper_role": "spike_native_self_supervised_ablation",
            "segmenter_spike_native": 1,
        }
    raise ValueError(f"Unsupported native/bridge method: {method}")


def segment(segmenter, method: str, text: str):
    if method == "proxy_blt":
        patches = segmenter.patch_text(text)
        return patches, {
            "evaluation_parameter_updates": 0,
            "operation_scope": (
                "digital_entropy_proxy_operations_not_yet_counted"
            ),
        }
    patches, _, stats = segmenter.segment_text(text)
    if method.startswith("legacy_homeostatic_"):
        raw_bytes = len(text.encode("utf-8", errors="strict"))
        stats = dict(stats)
        stats.update({
            "boundary_lif_state_updates": raw_bytes,
            "boundary_threshold_evaluations": raw_bytes,
            "operation_scope": (
                "boundary_lif_updates_counted_digital_surprise_and_codec_"
                "operations_not_yet_counted"
            ),
        })
    return patches, stats


def encode_lossless_patches(
    patches: Sequence[bytes],
    raw: bytes,
    character_flags: Sequence[int],
    codec: LosslessPatchCodec,
    *,
    fallback_mode: str = "literal_bytes",
    audit_roundtrip: bool = True,
) -> Tuple[UnitSequence, Dict[str, object]]:
    if b"".join(patches) != raw:
        raise AssertionError("Patch sequence does not reconstruct exact raw bytes")
    ids: List[int] = []
    byte_lengths: List[int] = []
    character_lengths: List[int] = []
    offset = 0
    vocabulary_hits = 0
    fallback_patches = 0
    fallback_byte_ids = 0
    subpatch_vocabulary_hits = 0
    subpatch_vocabulary_bytes_covered = 0
    vocabulary_bytes_covered = 0
    for patch_value in patches:
        patch = bytes(patch_value)
        flags = character_flags[offset : offset + len(patch)]
        offset += len(patch)
        token_id = codec.patch_to_id.get(patch)
        if token_id is not None:
            ids.append(token_id)
            byte_lengths.append(len(patch))
            character_lengths.append(sum(flags))
            vocabulary_hits += 1
            vocabulary_bytes_covered += len(patch)
        else:
            fallback_patches += 1
            segment_offset = 0
            for segment_id, payload in codec.encode_patch_segments(
                patch,
                fallback_mode=fallback_mode,
            ):
                segment_flags = flags[
                    segment_offset : segment_offset + len(payload)
                ]
                segment_offset += len(payload)
                ids.append(segment_id)
                byte_lengths.append(len(payload))
                character_lengths.append(sum(segment_flags))
                if segment_id < 256:
                    fallback_byte_ids += 1
                else:
                    subpatch_vocabulary_hits += 1
                    subpatch_vocabulary_bytes_covered += len(payload)
            if segment_offset != len(patch):
                raise AssertionError("Subpatch fallback did not cover its input patch")
    sequence = UnitSequence(
        ids=ids,
        byte_lengths=byte_lengths,
        character_lengths=character_lengths,
        vocab_size=codec.model_vocab_size,
        representation="lossless_patch_codec",
        metadata={"lossless_raw_byte_roundtrip": 1},
        payloads_by_id=payload_table_from_patch_codec(codec),
    )
    if audit_roundtrip and codec.decode_ids(ids) != raw:
        raise AssertionError("Lossless patch codec failed exact raw-byte round-trip")
    return sequence, {
        "input_patches": len(patches),
        "vocabulary_patch_hits": vocabulary_hits,
        "fallback_patch_count": fallback_patches,
        "fallback_patch_rate": fallback_patches / max(len(patches), 1),
        "vocabulary_patch_coverage": vocabulary_hits / max(len(patches), 1),
        "vocabulary_bytes_covered": vocabulary_bytes_covered,
        "vocabulary_byte_coverage": vocabulary_bytes_covered / max(len(raw), 1),
        "fallback_byte_ids": fallback_byte_ids,
        "literal_fallback_byte_rate": fallback_byte_ids / max(len(raw), 1),
        "subpatch_vocabulary_hits": subpatch_vocabulary_hits,
        "subpatch_vocabulary_bytes_covered": subpatch_vocabulary_bytes_covered,
        "effective_vocabulary_byte_coverage": (
            vocabulary_bytes_covered + subpatch_vocabulary_bytes_covered
        ) / max(len(raw), 1),
        "ids_per_byte": len(ids) / max(len(raw), 1),
        "bytes_per_id": len(raw) / max(len(ids), 1),
        "codec_vocab_size": codec.model_vocab_size,
        "patch_codec_fallback_mode": fallback_mode,
    }


def native_sequences(
    method: str,
    fitting_raw: bytes,
    fitting_text: str,
    validation_raw: bytes,
    validation_text: str,
    seed: int,
    *,
    modelling_training_raw: bytes | None = None,
    modelling_training_text: str | None = None,
    patch_codec_vocab_size: int = 8192,
    patch_codec_fallback: str = "literal_bytes",
) -> Tuple[UnitSequence, UnitSequence, Dict[str, object]]:
    if modelling_training_raw is None:
        modelling_training_raw = fitting_raw
    if modelling_training_text is None:
        modelling_training_text = fitting_text

    started = time.perf_counter()
    cpu_started = time.process_time()
    # Every fitted frontend sees the same declared training-only fitting
    # corpus. In train-only tuning ``fitting_text`` is the inner fit prefix;
    # in the paper comparison it is the canonical training split. The LM
    # optimization schedule remains the separate common raw-byte prefix.
    segmenter_fit_text = fitting_text
    segmenter, metadata = build_segmenter(method, segmenter_fit_text, seed)
    segmenter_fit_seconds = time.perf_counter() - started
    segmenter_fit_cpu_seconds = time.process_time() - cpu_started
    state_before = fitted_state_digest(segmenter)

    started = time.perf_counter()
    cpu_started = time.process_time()
    fitting_patches, fitting_stats = segment(segmenter, method, fitting_text)
    fitting_segmentation_seconds = time.perf_counter() - started
    fitting_segmentation_cpu_seconds = time.process_time() - cpu_started
    state_after_training_application = fitted_state_digest(segmenter)

    started = time.perf_counter()
    cpu_started = time.process_time()
    codec = LosslessPatchCodec.fit(
        fitting_patches,
        target_vocab_size=patch_codec_vocab_size,
    )
    codec_fit_seconds = time.perf_counter() - started
    codec_fit_cpu_seconds = time.process_time() - cpu_started

    if modelling_training_raw == fitting_raw:
        training_patches = fitting_patches
        training_stats = fitting_stats
        training_segmentation_seconds = 0.0
        training_segmentation_cpu_seconds = 0.0
    else:
        started = time.perf_counter()
        cpu_started = time.process_time()
        training_patches, training_stats = segment(
            segmenter, method, modelling_training_text
        )
        training_segmentation_seconds = time.perf_counter() - started
        training_segmentation_cpu_seconds = time.process_time() - cpu_started
    state_after_selected_training = fitted_state_digest(segmenter)

    started = time.perf_counter()
    cpu_started = time.process_time()
    validation_patches, validation_stats = segment(segmenter, method, validation_text)
    validation_segmentation_seconds = time.perf_counter() - started
    validation_segmentation_cpu_seconds = time.process_time() - cpu_started
    state_after_validation = fitted_state_digest(segmenter)
    repeat_started = time.perf_counter()
    repeat_cpu_started = time.process_time()
    repeated_patches, _ = segment(segmenter, method, validation_text)
    validation_repeat_audit_seconds = time.perf_counter() - repeat_started
    validation_repeat_audit_cpu_seconds = (
        time.process_time() - repeat_cpu_started
    )
    state_after_repeat = fitted_state_digest(segmenter)
    if (
        state_before != state_after_training_application
        or state_before != state_after_selected_training
        or state_before != state_after_validation
        or state_before != state_after_repeat
    ):
        raise AssertionError(f"Frozen segmenter state mutated during application: {method}")
    if validation_patches != repeated_patches:
        raise AssertionError(f"Validation segmentation is not deterministic: {method}")

    started = time.perf_counter()
    cpu_started = time.process_time()
    training, training_codec = encode_lossless_patches(
        training_patches, modelling_training_raw,
        utf8_byte_character_completions(modelling_training_text), codec,
        fallback_mode=patch_codec_fallback,
        audit_roundtrip=False,
    )
    training_codec_seconds = time.perf_counter() - started
    training_codec_cpu_seconds = time.process_time() - cpu_started
    started = time.perf_counter()
    cpu_started = time.process_time()
    validation, validation_codec = encode_lossless_patches(
        validation_patches, validation_raw,
        utf8_byte_character_completions(validation_text), codec,
        fallback_mode=patch_codec_fallback,
        audit_roundtrip=False,
    )
    validation_codec_seconds = time.perf_counter() - started
    validation_codec_cpu_seconds = time.process_time() - cpu_started
    if codec.decode_ids(training.ids) != modelling_training_raw:
        raise AssertionError("Training patch codec failed exact raw-byte round-trip")
    if codec.decode_ids(validation.ids) != validation_raw:
        raise AssertionError("Validation patch codec failed exact raw-byte round-trip")
    fit_seconds = (
        segmenter_fit_seconds
        + fitting_segmentation_seconds
        + codec_fit_seconds
    )
    fit_cpu_seconds = (
        segmenter_fit_cpu_seconds
        + fitting_segmentation_cpu_seconds
        + codec_fit_cpu_seconds
    )
    metadata.update({
        "frontend_fit_seconds": fit_seconds,
        "frontend_fit_seconds_current_run": fit_seconds,
        "frontend_fit_precomputed": 0,
        "frontend_fit_cpu_seconds_current_run": fit_cpu_seconds,
        "frontend_training_encode_seconds_current_run": (
            training_segmentation_seconds + training_codec_seconds
        ),
        "frontend_training_encode_cpu_seconds_current_run": (
            training_segmentation_cpu_seconds + training_codec_cpu_seconds
        ),
        "frontend_validation_encode_seconds_current_run": (
            validation_segmentation_seconds + validation_codec_seconds
        ),
        "frontend_validation_encode_cpu_seconds_current_run": (
            validation_segmentation_cpu_seconds + validation_codec_cpu_seconds
        ),
        "frontend_validation_repeat_audit_seconds_current_run": (
            validation_repeat_audit_seconds
        ),
        "frontend_validation_repeat_audit_cpu_seconds_current_run": (
            validation_repeat_audit_cpu_seconds
        ),
        "frontend_fit_operation_count_complete": 0,
        "frontend_wall_time_accounting_complete": 1,
        "frontend_segmenter_fit_seconds": segmenter_fit_seconds,
        "frontend_segmenter_fit_cpu_seconds": segmenter_fit_cpu_seconds,
        "frontend_fitting_corpus_segmentation_seconds": fitting_segmentation_seconds,
        "frontend_fitting_corpus_segmentation_cpu_seconds": (
            fitting_segmentation_cpu_seconds
        ),
        "frontend_codec_fit_seconds": codec_fit_seconds,
        "frontend_codec_fit_cpu_seconds": codec_fit_cpu_seconds,
        "frontend_training_segmentation_cpu_seconds": (
            training_segmentation_cpu_seconds
        ),
        "frontend_training_codec_cpu_seconds": training_codec_cpu_seconds,
        "frontend_validation_segmentation_cpu_seconds": (
            validation_segmentation_cpu_seconds
        ),
        "frontend_validation_codec_cpu_seconds": validation_codec_cpu_seconds,
        "patch_codec_target_vocab_size": patch_codec_vocab_size,
        "patch_codec_used_vocab_size": codec.model_vocab_size,
        "patch_codec_payload_storage_bytes": sum(
            len(payload) for payload in payload_table_from_patch_codec(codec)
        ),
        "patch_codec_fallback_mode": patch_codec_fallback,
        "patch_codec_fallback_uses_training_vocabulary_only": 1,
        "evaluation_frozen": 1,
        "evaluation_repeat_deterministic": 1,
        "fitting_corpus_boundary_rate": max(len(fitting_patches) - 1, 0) / max(len(fitting_raw), 1),
        "training_boundary_rate": max(len(training_patches) - 1, 0) / max(len(modelling_training_raw), 1),
        "validation_boundary_rate": max(len(validation_patches) - 1, 0) / max(len(validation_raw), 1),
        "fitting_corpus_mean_patch_bytes": len(fitting_raw) / max(len(fitting_patches), 1),
        "training_mean_patch_bytes": len(modelling_training_raw) / max(len(training_patches), 1),
        "validation_mean_patch_bytes": len(validation_raw) / max(len(validation_patches), 1),
        **{f"training_codec_{key}": value for key, value in training_codec.items()},
        **{f"validation_codec_{key}": value for key, value in validation_codec.items()},
        **{
            f"train_unit_frontend_{key}": value
            for key, value in training_stats.items()
            if isinstance(value, (int, float, str)) and not isinstance(value, bool)
        },
        **{
            f"eval_unit_frontend_{key}": value
            for key, value in validation_stats.items()
            if isinstance(value, (int, float, str)) and not isinstance(value, bool)
        },
    })
    for key in (
        "homeostatic_final_threshold", "homeostatic_training_boundary_rate",
        "homeostatic_training_ema_rate", "intrinsic_target_boundary_rate",
        "local_final_ath", "refractory_suppressed_per_char",
        "prediction_error_spike_rate_per_byte", "emitted_boundary_rate_per_byte",
        "boundary_threshold", "hidden_spike_events", "prediction_spike_events",
        "prediction_error_spike_events", "natural_boundary_spikes",
        "max_length_guard_boundaries", "end_of_stream_flushes",
    ):
        if key in validation_stats:
            metadata[key] = validation_stats[key]
        elif key in training_stats:
            metadata[key] = training_stats[key]
    return training, validation, metadata


def build_sequences(
    args,
    training_raw,
    training_text,
    validation_raw,
    validation_text,
    *,
    training_path: Path,
    training_sha256: str,
    modelling_training_raw: bytes | None = None,
    modelling_training_text: str | None = None,
):
    if modelling_training_raw is None:
        modelling_training_raw = training_raw
    if modelling_training_text is None:
        modelling_training_text = training_text
    if args.method == "byte":
        started = time.perf_counter()
        cpu_started = time.process_time()
        training = byte_sequence(modelling_training_raw, modelling_training_text)
        training_seconds = time.perf_counter() - started
        training_cpu_seconds = time.process_time() - cpu_started
        started = time.perf_counter()
        cpu_started = time.process_time()
        validation = byte_sequence(validation_raw, validation_text)
        validation_seconds = time.perf_counter() - started
        validation_cpu_seconds = time.process_time() - cpu_started
        return training, validation, {
            "training_signal": "training-free", "paper_role": "primary_baseline",
            "segmenter_spike_native": 0,
            "frontend_fit_seconds": 0.0,
            "frontend_fit_seconds_current_run": 0.0,
            "frontend_fit_precomputed": 0,
            "frontend_fit_cpu_seconds_current_run": 0.0,
            "frontend_training_encode_seconds_current_run": training_seconds,
            "frontend_training_encode_cpu_seconds_current_run": training_cpu_seconds,
            "frontend_validation_encode_seconds_current_run": validation_seconds,
            "frontend_validation_encode_cpu_seconds_current_run": validation_cpu_seconds,
            "frontend_validation_repeat_audit_seconds_current_run": 0.0,
            "frontend_validation_repeat_audit_cpu_seconds_current_run": 0.0,
            "frontend_fit_operation_count_complete": 1,
            "frontend_wall_time_accounting_complete": 1,
            "unit_frontend_operation_scope": (
                "digital_byte_indexing_operations_not_counted_phase_time_recorded"
            ),
        }
    if args.method in EXTERNAL_CONVENTIONAL_MODELS:
        tokenizer = OfficialSpikeGPT20BTokenizer.from_path(
            EXTERNAL_CONVENTIONAL_MODELS[args.method]
        )
        started = time.perf_counter()
        cpu_started = time.process_time()
        training = official_spikegpt_sequence(
            tokenizer, modelling_training_raw, modelling_training_text
        )
        training_seconds = time.perf_counter() - started
        training_cpu_seconds = time.process_time() - cpu_started
        started = time.perf_counter()
        cpu_started = time.process_time()
        validation = official_spikegpt_sequence(
            tokenizer, validation_raw, validation_text
        )
        validation_seconds = time.perf_counter() - started
        validation_cpu_seconds = time.process_time() - cpu_started
        return training, validation, {
            "training_signal": "external-pretrained-fixed-tokenizer",
            "paper_role": "external_spikegpt_tokenizer_fidelity_baseline",
            "segmenter_spike_native": 0,
            "frontend_fit_seconds": 0.0,
            "frontend_fit_seconds_current_run": 0.0,
            "frontend_fit_precomputed": 1,
            "frontend_fit_cpu_seconds_current_run": "external_fit_not_repeated",
            "frontend_training_encode_seconds_current_run": training_seconds,
            "frontend_training_encode_cpu_seconds_current_run": training_cpu_seconds,
            "frontend_validation_encode_seconds_current_run": validation_seconds,
            "frontend_validation_encode_cpu_seconds_current_run": validation_cpu_seconds,
            "frontend_validation_repeat_audit_seconds_current_run": 0.0,
            "frontend_validation_repeat_audit_cpu_seconds_current_run": 0.0,
            "frontend_fit_operation_count_complete": 0,
            "frontend_wall_time_accounting_complete": 0,
            "unit_frontend_operation_scope": (
                "dependency_free_exact_bytelevel_bpe_encoding_logical_merge_"
                "counts_and_phase_time_recorded_external_fit_not_repeated"
            ),
            "external_pretrained_tokenizer_baseline": 1,
            "external_tokenizer_training_overlap_with_evaluation_not_audited": 1,
        }
    if args.method in CONVENTIONAL_MODELS:
        if args.conventional_tokenizer_model:
            model_path = Path(args.conventional_tokenizer_model).resolve()
            record_path = model_path.with_suffix(".train.json")
            if not model_path.is_file() or not record_path.is_file():
                raise FileNotFoundError(
                    "matched conventional tokenizer model/provenance is missing"
                )
            record = json.loads(record_path.read_text(encoding="utf-8"))
            modelling_digest = sha256(modelling_training_raw)
            required = {
                "format": "decisive_exact_selection_sentencepiece_provenance_v1",
                "protocol_id": PROTOCOL_ID,
                "dataset": args.dataset,
                "method": args.method,
                "training_source_path": str(training_path.resolve()),
                "training_source_sha256": training_sha256,
                "training_selection_start_raw_byte": 0,
                "training_selection_end_raw_byte_exclusive": len(modelling_training_raw),
                "training_selection_raw_bytes": len(modelling_training_raw),
                "training_selection_sha256": modelling_digest,
                "test_split_touched": 0,
            }
            mismatches = {
                key: (record.get(key), expected)
                for key, expected in required.items()
                if record.get(key) != expected
            }
            if mismatches:
                raise AssertionError(
                    f"matched tokenizer provenance mismatch: {mismatches}"
                )
            tokenizer_provenance = {
                "tokenizer_training_provenance_path": str(record_path),
                "tokenizer_training_path": record["training_selection_path"],
                "tokenizer_training_sha256": modelling_digest,
                "tokenizer_fit_corpus_matches_lm_training_corpus": 1,
                "tokenizer_fit_exact_lm_training_selection": 1,
            }
        else:
            model_path = TOKENIZER_ROOT / (
                f"{args.dataset}_{CONVENTIONAL_MODELS[args.method]}.model"
            )
            tokenizer_provenance = sentencepiece_training_provenance(
                model_path=model_path,
                dataset=args.dataset,
                expected_training_path=training_path,
                expected_training_sha256=training_sha256,
            )
        model_file_bytes = model_path.stat().st_size
        model_sha256 = sha256(model_path.read_bytes())
        load_started = time.perf_counter()
        load_cpu_started = time.process_time()
        processor = spm.SentencePieceProcessor(model_file=str(model_path))
        model_load_seconds = time.perf_counter() - load_started
        model_load_cpu_seconds = time.process_time() - load_cpu_started
        training, training_timing = sentencepiece_sequence(
            args.method,
            modelling_training_raw,
            modelling_training_text,
            processor=processor,
            model_path=model_path,
            tokenizer_provenance=tokenizer_provenance,
        )
        validation, validation_timing = sentencepiece_sequence(
            args.method,
            validation_raw,
            validation_text,
            processor=processor,
            model_path=model_path,
            tokenizer_provenance=tokenizer_provenance,
        )
        return training, validation, {
            "training_signal": "corpus-unsupervised",
            "paper_role": "primary_baseline",
            "segmenter_spike_native": 0,
            "frontend_fit_seconds": 0.0,
            "frontend_fit_seconds_current_run": 0.0,
            "frontend_fit_precomputed": 1,
            "frontend_fit_cpu_seconds_current_run": (
                "not_measured_precomputed_sentencepiece_fit"
            ),
            "frontend_model_load_seconds_current_run": model_load_seconds,
            "frontend_model_load_cpu_seconds_current_run": model_load_cpu_seconds,
            "frontend_training_encode_seconds_current_run": training_timing[
                "encode_seconds"
            ],
            "frontend_training_encode_cpu_seconds_current_run": training_timing[
                "encode_cpu_seconds"
            ],
            "frontend_validation_encode_seconds_current_run": validation_timing[
                "encode_seconds"
            ],
            "frontend_validation_encode_cpu_seconds_current_run": validation_timing[
                "encode_cpu_seconds"
            ],
            "frontend_validation_repeat_audit_seconds_current_run": 0.0,
            "frontend_validation_repeat_audit_cpu_seconds_current_run": 0.0,
            "frontend_fit_operation_count_complete": 0,
            "frontend_wall_time_accounting_complete": 0,
            "tokenizer_model_file_bytes_current_run": model_file_bytes,
            "tokenizer_model_sha256_current_run": model_sha256,
            "unit_frontend_operation_scope": (
                "sentencepiece_internal_operations_opaque_single_loaded_"
                "processor_encode_cpu_time_recorded_offline_fit_time_unavailable"
            ),
        }
    return native_sequences(
        args.method, training_raw, training_text,
        validation_raw, validation_text, args.seed,
        modelling_training_raw=modelling_training_raw,
        modelling_training_text=modelling_training_text,
        patch_codec_vocab_size=args.patch_codec_vocab_size,
        patch_codec_fallback=args.patch_codec_fallback,
    )



def build_backbone(args, vocab_size: int, input_dim: int = 32):
    if args.backbone == "strict_spike_rwkv_v2":
        return build_strict_spike_rwkv_v2_lm(
            vocab_size=vocab_size,
            input_dim=32,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            beta=0.90,
            threshold=0.75,
            dropout=0.0,
        )
    if args.backbone == "srwkv_v2":
        return build_spiking_lm(
            model_type="srwkv_v2",
            vocab_size=vocab_size,
            emb_dim=32,
            spike_dim=32,
            hidden_dim=args.hidden_dim,
            spike_steps=args.spike_steps,
            beta=0.90,
            encoder="none",
            dropout=0.0,
            num_layers=args.num_layers,
        )
    if args.backbone == "spikegpt_reference":
        return build_spikegpt_reference_lm(
            vocab_size=vocab_size,
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=0.03,
        )
    raise ValueError(f"Unsupported backbone: {args.backbone}")


def run_backbone(model, args, input_spikes, compute_steps):
    if args.backbone == "strict_spike_rwkv_v2":
        return model(inputs_spikes=input_spikes, compute_steps=compute_steps)
    if args.backbone == "srwkv_v2":
        logits, stats = model(inputs_spikes=input_spikes)
        stats = dict(stats)
        stats["actual_compute_steps"] = float(
            input_spikes.shape[0] * input_spikes.shape[1] * args.spike_steps
        )
        return logits, stats
    if args.backbone == "spikegpt_reference":
        return model(inputs_spikes=input_spikes)
    raise ValueError(f"Unsupported backbone: {args.backbone}")


def backbone_accounting_status(backbone: str) -> str:
    if backbone == "strict_spike_rwkv_v2":
        return "event_lower_bounds_plus_dense_readout_hardware_energy_estimate_deferred"
    if backbone == "srwkv_v2":
        return (
            "dense_linear_macs_and_lif_updates_counted_"
            "nonlinear_memory_frontend_incomplete_hardware_energy_estimate_deferred"
        )
    if backbone == "spikegpt_reference":
        return (
            "reference_faithful_spikegpt_dense_macs_lif_updates_and_"
            "nonlinear_elements_counted_memory_frontend_incomplete_"
            "hardware_energy_estimate_deferred"
        )
    raise ValueError(f"Unsupported backbone: {backbone}")


def batch_from_starts(sequence: UnitSequence, starts: Sequence[int], seq_len: int, device):
    x = torch.tensor(
        [sequence.ids[start : start + seq_len] for start in starts],
        dtype=torch.long, device=device,
    )
    y = torch.tensor(
        [sequence.ids[start + 1 : start + seq_len + 1] for start in starts],
        dtype=torch.long, device=device,
    )
    byte_lengths = torch.tensor(
        [sequence.byte_lengths[start + 1 : start + seq_len + 1] for start in starts],
        dtype=torch.long, device=device,
    )
    character_lengths = torch.tensor(
        [sequence.character_lengths[start + 1 : start + seq_len + 1] for start in starts],
        dtype=torch.long, device=device,
    )
    return x, y, byte_lengths, character_lengths


def complete_target_blocks(num_targets: int, seq_len: int) -> List[Tuple[int, int]]:
    """Partition every target exactly once into contiguous reset blocks."""
    if num_targets <= 0 or seq_len <= 0:
        raise ValueError("target and block sizes must be positive")
    return [
        (start, min(seq_len, num_targets - start))
        for start in range(0, num_targets, seq_len)
    ]


def accumulate_model_stats(totals: Dict[str, float], stats: Mapping[str, float]) -> None:
    totals["input_spikes"] += float(stats.get("total_input_spikes", 0.0))
    totals["hidden_spikes"] += float(stats.get("total_hidden_spikes", 0.0))
    totals["compute_steps"] += float(stats.get("actual_compute_steps", 0.0))
    totals["input_adapter_spikes"] = totals.get("input_adapter_spikes", 0.0) + float(
        stats.get("input_adapter_spikes", 0.0)
    )
    for key, value in stats.items():
        if key.startswith("layer") and key.endswith("_hidden_spikes"):
            totals[key] = totals.get(key, 0.0) + float(value)


def evaluate(model, interface, sequence, args, device):
    """Evaluate all target units, including the final partial context block."""
    model.eval()
    interface.eval()
    starts = list(range(0, len(sequence.ids) - 1, args.seq_len))
    full_starts = [
        start for start in starts
        if start + args.seq_len < len(sequence.ids)
    ]
    partial_starts = [
        start for start in starts
        if start + args.seq_len >= len(sequence.ids)
    ]
    if len(partial_starts) > 1:
        raise AssertionError("Evaluation produced more than one partial block")

    total_nll = 0.0
    total_bytes = 0
    total_characters = 0
    total_units = 0
    totals: Dict[str, float] = {
        "input_spikes": 0.0,
        "hidden_spikes": 0.0,
        "compute_steps": 0.0,
    }
    frontend_totals: Dict[str, float] = {}

    def process(selected: Sequence[int], block_length: int) -> None:
        nonlocal total_nll, total_bytes, total_characters, total_units
        x_ids, y, byte_lengths, character_lengths = batch_from_starts(
            sequence, selected, block_length, device
        )
        x, _ = interface(x_ids)
        if hasattr(interface, "last_operation_stats"):
            merge_frontend_operation_stats(
                frontend_totals,
                interface.last_operation_stats(),
            )
        compute_steps = torch.ones_like(x_ids, dtype=torch.float32)
        logits, stats = run_backbone(model, args, x, compute_steps)
        if not interface.is_binary:
            stats = dict(stats)
            stats["total_input_spikes"] = 0.0
            stats["input_spike_rate"] = 0.0
        losses = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            reduction="none",
        )
        total_nll += float(losses.sum().item())
        total_bytes += int(byte_lengths.sum().item())
        total_characters += int(character_lengths.sum().item())
        total_units += int(y.numel())
        accumulate_model_stats(totals, stats)

    budget_reached = False
    with torch.no_grad():
        for offset in range(0, len(full_starts), args.batch_size):
            selected = full_starts[offset : offset + args.batch_size]
            if not selected:
                break
            process(selected, args.seq_len)
            if args.eval_byte_budget > 0 and total_bytes >= args.eval_byte_budget:
                budget_reached = True
                break

        if not budget_reached and partial_starts:
            start = partial_starts[0]
            block_length = len(sequence.ids) - start - 1
            if block_length <= 0 or block_length >= args.seq_len:
                raise AssertionError("Invalid final evaluation block length")
            process([start], block_length)

    if total_bytes <= 0 or total_characters <= 0:
        raise AssertionError("Evaluation denominator is empty")
    expected_bytes = sum(sequence.byte_lengths)
    expected_characters = sum(sequence.character_lengths)
    expected_units = len(sequence.ids) - 1
    complete = int(
        total_bytes == expected_bytes
        and total_characters == expected_characters
        and total_units == expected_units
    )
    if args.eval_byte_budget <= 0 and not complete:
        raise AssertionError(
            "Full-split evaluation did not cover every held-out raw byte, "
            "character, and modelling unit"
        )
    return {
        "eval_nll_nats": total_nll,
        "eval_raw_bytes": total_bytes,
        "eval_characters": total_characters,
        "eval_modelling_units": total_units,
        "evaluation_complete_raw_split": complete,
        "evaluation_uses_shared_bos": 1,
        "evaluation_recurrent_reset_interval_units": args.seq_len,
        "bits_per_raw_byte": total_nll / math.log(2.0) / total_bytes,
        "bits_per_character": total_nll / math.log(2.0) / total_characters,
        "unit_perplexity": math.exp(
            min(total_nll / max(total_units, 1), 20.0)
        ),
        **{f"eval_{key}": value for key, value in totals.items()},
        **{f"eval_frontend_{key}": value for key, value in frontend_totals.items()},
    }

def write_csv(path: Path, row: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(row))
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument(
        "--data-tier",
        choices=sorted(DATASET_TIERS),
        default="development",
    )
    parser.add_argument("--method", required=True)
    parser.add_argument(
        "--patch-codec-vocab-size",
        type=int,
        default=8192,
        help="Native patch-codec target vocabulary; ignored by byte/SentencePiece methods.",
    )
    parser.add_argument(
        "--patch-codec-fallback",
        choices=("literal_bytes", "optimal_training_subpatch"),
        default="literal_bytes",
        help="Explicit native codec fallback; the historical default remains literal bytes.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--train-byte-budget", type=int, default=100_000)
    parser.add_argument(
        "--training-schedule",
        choices=("random_unit_windows", "common_utf8_prefix_one_pass"),
        default="random_unit_windows",
    )
    parser.add_argument(
        "--fit-frontends-on-lm-training-selection",
        action="store_true",
        help=(
            "Fit every train-fitted frontend on the exact LM byte selection; "
            "used by the decisive fairness-gated comparison."
        ),
    )
    parser.add_argument(
        "--conventional-tokenizer-model",
        default="",
        help="Exact-selection SentencePiece model produced by the decisive fit job.",
    )
    parser.add_argument("--eval-byte-budget", type=int, default=20_000)
    parser.add_argument(
        "--evaluation-source",
        choices=(
            "canonical_validation",
            "canonical_test",
            "training_inner_holdout",
        ),
        default="canonical_validation",
    )
    parser.add_argument("--inner-fit-byte-budget", type=int, default=800_000)
    parser.add_argument("--inner-holdout-end-byte", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument(
        "--backbone",
        choices=("strict_spike_rwkv_v2", "srwkv_v2", "spikegpt_reference"),
        default="strict_spike_rwkv_v2",
    )
    parser.add_argument("--spike-steps", type=int, default=4)
    parser.add_argument(
        "--interface-encoder",
        choices=(
            "fixed_4of32_a", "fixed_4of32_b", "fixed_4of32_c",
            "fixed_2of32_byte", "fixed_1of256_byte",
            "learned_topk_4of32_ste", "learned_threshold32_rate",
            "spikegpt_embedding64_atan",
            *sorted(SPIKEPATCH_INTERFACE_MODES),
            INJECTIVE_RECURRENT_BYTE_MODE,
            "dense32_ceiling",
        ),
        default="fixed_4of32_a",
    )
    parser.add_argument("--interface-temperature", type=float, default=1.0)
    parser.add_argument("--interface-target-rate", type=float, default=0.125)
    parser.add_argument("--interface-rate-weight", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()
    if args.conventional_tokenizer_model and args.method not in CONVENTIONAL_MODELS:
        parser.error(
            "--conventional-tokenizer-model is only valid for matched SentencePiece methods"
        )
    if args.conventional_tokenizer_model and not args.fit_frontends_on_lm_training_selection:
        parser.error(
            "a matched tokenizer model requires --fit-frontends-on-lm-training-selection"
        )
    overall_started = time.perf_counter()
    process_started = time.process_time()
    OPENED_DATA_PATHS.clear()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    device = torch.device("cpu")
    selected_paths = DATASET_TIERS[args.data_tier][args.dataset]
    training_path = CANONICAL_ROOT / selected_paths["train"]
    validation_path = CANONICAL_ROOT / selected_paths["validation"]
    if args.evaluation_source == "training_inner_holdout":
        canonical_contract = assert_training_only_contract(
            canonical_root=CANONICAL_ROOT,
            dataset=args.dataset,
            data_tier=args.data_tier,
            training_path=training_path,
        )
    elif args.evaluation_source == "canonical_validation":
        canonical_contract = assert_train_validation_contract(
            canonical_root=CANONICAL_ROOT,
            dataset=args.dataset,
            data_tier=args.data_tier,
            training_path=training_path,
            validation_path=validation_path,
        )
    else:
        if args.data_tier != "full_train_validation":
            raise AssertionError(
                "confirmatory test evaluation requires the full training tier"
            )
        test_path = CANONICAL_ROOT / selected_paths["test"]
        canonical_contract = assert_train_test_contract(
            canonical_root=CANONICAL_ROOT,
            dataset=args.dataset,
            data_tier=args.data_tier,
            training_path=training_path,
            test_path=test_path,
        )
    training_raw, training_text = read_strict(training_path, role="training")
    training_digest = sha256(training_raw)
    if training_digest != canonical_contract["canonical_train_sha256"]:
        raise AssertionError("Loaded training bytes differ from the canonical contract")
    if args.evaluation_source == "canonical_validation":
        validation_raw, validation_text = read_strict(
            validation_path, role="validation"
        )
        validation_digest = sha256(validation_raw)
        if validation_digest != canonical_contract["canonical_validation_sha256"]:
            raise AssertionError(
                "Loaded validation bytes differ from the canonical contract"
            )
        validation_source_description = str(validation_path)
        inner_fit_end = "not_applicable"
        inner_holdout_end = "not_applicable"
    elif args.evaluation_source == "canonical_test":
        validation_raw, validation_text = read_strict(test_path, role="test")
        validation_digest = sha256(validation_raw)
        if validation_digest != canonical_contract["canonical_test_sha256"]:
            raise AssertionError("Loaded test bytes differ from the canonical contract")
        validation_source_description = str(test_path)
        inner_fit_end = "not_applicable"
        inner_holdout_end = "not_applicable"
    else:
        if args.method in ALL_CONVENTIONAL_METHODS:
            raise AssertionError(
                "Pretrained conventional tokenizers cannot enter train-only "
                "tuning because their fit corpus contains the inner holdout"
            )
        if not (
            0 < args.inner_fit_byte_budget < args.inner_holdout_end_byte
            <= len(training_raw)
        ):
            raise ValueError("invalid inner fit/holdout byte boundaries")
        tuning_raw, tuning_text = utf8_safe_prefix(
            training_raw, args.inner_holdout_end_byte
        )
        inner_training_raw, inner_training_text = utf8_safe_prefix(
            tuning_raw, args.inner_fit_byte_budget
        )
        validation_raw = tuning_raw[len(inner_training_raw):]
        if not validation_raw:
            raise ValueError("inner training holdout is empty")
        validation_text = validation_raw.decode("utf-8", errors="strict")
        validation_digest = sha256(validation_raw)
        inner_fit_end = len(inner_training_raw)
        inner_holdout_end = len(tuning_raw)
        validation_source_description = (
            f"{training_path.resolve()}#utf8_bytes["
            f"{inner_fit_end}:{inner_holdout_end}]"
        )
    if args.evaluation_source == "training_inner_holdout":
        modelling_training_raw = inner_training_raw
        modelling_training_text = inner_training_text
    elif args.training_schedule == "common_utf8_prefix_one_pass":
        modelling_training_raw, modelling_training_text = utf8_safe_prefix(
            training_raw,
            args.train_byte_budget,
        )
    else:
        modelling_training_raw = training_raw
        modelling_training_text = training_text
    modelling_training_digest = sha256(modelling_training_raw)
    exact_frontend_fit_selection = (
        args.evaluation_source == "training_inner_holdout"
        or args.fit_frontends_on_lm_training_selection
    )
    frontend_source_raw = (
        modelling_training_raw if exact_frontend_fit_selection else training_raw
    )
    frontend_source_text = (
        modelling_training_text if exact_frontend_fit_selection else training_text
    )
    frontend_prepare_cpu_started = time.process_time()
    training, validation, frontend_metadata = build_sequences(
        args,
        frontend_source_raw,
        frontend_source_text,
        validation_raw,
        validation_text,
        training_path=training_path,
        training_sha256=training_digest,
        modelling_training_raw=modelling_training_raw,
        modelling_training_text=modelling_training_text,
    )
    frontend_prepare_cpu_seconds = time.process_time() - frontend_prepare_cpu_started
    if args.method in CONVENTIONAL_MODELS:
        frontend_fit_path = str(training.metadata["tokenizer_training_path"])
        frontend_fit_sha256 = str(training.metadata["tokenizer_training_sha256"])
    elif args.method in EXTERNAL_CONVENTIONAL_MODELS:
        frontend_fit_path = "external_pretrained_tokenizer_fit_not_repeated"
        frontend_fit_sha256 = "external_pretrained_tokenizer_fit_not_repeated"
    elif args.method == "byte":
        frontend_fit_path = "not_applicable_training_free"
        frontend_fit_sha256 = "not_applicable_training_free"
    elif exact_frontend_fit_selection:
        frontend_fit_path = (
            f"{training_path.resolve()}#utf8_bytes[0:{len(modelling_training_raw)}]"
        )
        frontend_fit_sha256 = modelling_training_digest
    else:
        frontend_fit_path = str(training_path.resolve())
        frontend_fit_sha256 = training_digest
    if args.method in EXTERNAL_CONVENTIONAL_MODELS:
        frontend_fit_match = 1
    elif exact_frontend_fit_selection:
        frontend_fit_match = int(
            args.method == "byte"
            or frontend_fit_sha256 == modelling_training_digest
        )
    else:
        frontend_fit_match = int(
            args.method == "byte"
            or (
                Path(frontend_fit_path).resolve() == training_path.resolve()
                and frontend_fit_sha256 == training_digest
            )
        )
    if frontend_fit_match != 1:
        raise AssertionError(
            "Front-end fitting corpus does not match the LM training corpus"
        )
    frontend_metadata.update({
        "frontend_fit_corpus_path": frontend_fit_path,
        "frontend_fit_corpus_sha256": frontend_fit_sha256,
        "frontend_fit_corpus_matches_lm_training_corpus": int(
            frontend_fit_match == 1
            and args.method not in EXTERNAL_CONVENTIONAL_MODELS
            and (
                args.method == "byte"
                or modelling_training_raw == frontend_source_raw
            )
        ),
        "frontend_fit_corpus_matches_canonical_training_split": int(
            frontend_fit_match == 1
            and args.evaluation_source in {
                "canonical_validation", "canonical_test"
            }
            and args.method not in EXTERNAL_CONVENTIONAL_MODELS
        ),
        "frontend_fit_exact_lm_training_selection": int(
            frontend_fit_match == 1
            and exact_frontend_fit_selection
            and args.method not in EXTERNAL_CONVENTIONAL_MODELS
        ),
        "frontend_fit_corpus_contains_lm_training_selection": (
            0 if args.method in EXTERNAL_CONVENTIONAL_MODELS else frontend_fit_match
        ),
        "frontend_is_external_pretrained_fixed_baseline": int(
            args.method in EXTERNAL_CONVENTIONAL_MODELS
        ),
    })
    if training.vocab_size != validation.vocab_size:
        raise AssertionError("Training and validation vocabularies differ before BOS")
    if training.payloads_by_id != validation.payloads_by_id:
        raise AssertionError("Training and validation ID payload tables differ")
    pre_bos_vocab_size = training.vocab_size
    training = prepend_shared_bos(training)
    validation = prepend_shared_bos(validation)
    training.validate(len(modelling_training_raw), len(modelling_training_text))
    validation.validate(len(validation_raw), len(validation_text))
    if training.vocab_size != validation.vocab_size:
        raise AssertionError("Training and validation vocabularies differ after BOS")

    interface_build_started = time.perf_counter()
    interface_build_cpu_started = time.process_time()
    interface = build_spike_input_interface(
        training.vocab_size,
        args.interface_encoder,
        temperature=args.interface_temperature,
        target_rate=args.interface_target_rate,
        payloads_by_id=training.payloads_by_id,
        homeostatic_fit_ids=training.ids[1:],
        fit_corpus_sha256=modelling_training_digest,
        seed=args.seed,
    ).to(device)
    interface_build_seconds = time.perf_counter() - interface_build_started
    interface_build_cpu_seconds = time.process_time() - interface_build_cpu_started
    interface_initial_audit_started = time.perf_counter()
    initial_interface_audit = interface.audit(fail_on_collision=True)
    interface_initial_audit_seconds = time.perf_counter() - interface_initial_audit_started
    model = build_backbone(
        args,
        training.vocab_size,
        input_dim=int(initial_interface_audit["interface_dimension"]),
    ).to(device)
    trainable_parameters = list(model.parameters()) + list(interface.parameters())
    if args.backbone == "spikegpt_reference":
        optimiser = torch.optim.Adam(
            trainable_parameters,
            lr=args.lr,
            betas=(0.9, 0.99),
            weight_decay=0.0,
        )
        optimizer_name = "adam_spikegpt_compatible_no_weight_decay"
    else:
        optimiser = torch.optim.AdamW(trainable_parameters, lr=args.lr)
        optimizer_name = "adamw"
    bytes_seen = 0
    units_seen = 0
    steps = 0
    losses: List[float] = []
    interface_regularizers: List[float] = []
    train_stats: Dict[str, float] = {"input_spikes": 0.0, "hidden_spikes": 0.0, "compute_steps": 0.0}
    train_frontend_stats: Dict[str, float] = {}
    started = time.perf_counter()
    train_cpu_started = time.process_time()
    model.train()
    interface.train()
    train_target_units_covered_once = 0

    def train_batch(starts: Sequence[int], block_length: int) -> None:
        nonlocal bytes_seen, units_seen, steps
        x_ids, y, byte_lengths, _ = batch_from_starts(
            training,
            starts,
            block_length,
            device,
        )
        x, interface_regularizer = interface(x_ids)
        if hasattr(interface, "last_operation_stats"):
            merge_frontend_operation_stats(
                train_frontend_stats,
                interface.last_operation_stats(),
            )
        compute_steps = torch.ones_like(x_ids, dtype=torch.float32)
        logits, stats = run_backbone(model, args, x, compute_steps)
        if not interface.is_binary:
            stats = dict(stats)
            stats["total_input_spikes"] = 0.0
            stats["input_spike_rate"] = 0.0
        lm_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss = lm_loss + args.interface_rate_weight * interface_regularizer
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
        if not math.isfinite(float(gradient_norm)):
            raise RuntimeError("LM training produced a non-finite gradient")
        optimiser.step()
        losses.append(float(lm_loss.item()))
        interface_regularizers.append(float(interface_regularizer.detach().item()))
        bytes_seen += int(byte_lengths.sum().item())
        units_seen += int(y.numel())
        steps += 1
        accumulate_model_stats(train_stats, stats)

    if args.training_schedule == "common_utf8_prefix_one_pass":
        blocks = complete_target_blocks(len(training.ids) - 1, args.seq_len)
        full_starts = [
            start for start, length in blocks
            if length == args.seq_len
        ]
        partial_starts = [
            start for start, length in blocks
            if length < args.seq_len
        ]
        if len(partial_starts) > 1:
            raise AssertionError("training schedule produced multiple partial blocks")
        for offset in range(0, len(full_starts), args.batch_size):
            selected = full_starts[offset : offset + args.batch_size]
            if selected:
                train_batch(selected, args.seq_len)
        if partial_starts:
            start = partial_starts[0]
            block_length = len(training.ids) - start - 1
            if block_length <= 0 or block_length >= args.seq_len:
                raise AssertionError("invalid final training block length")
            train_batch([start], block_length)
        expected_training_bytes = sum(training.byte_lengths)
        expected_training_units = len(training.ids) - 1
        if bytes_seen != expected_training_bytes or units_seen != expected_training_units:
            raise AssertionError("common-prefix training did not cover each target exactly once")
        train_target_units_covered_once = 1
    else:
        max_start = len(training.ids) - args.seq_len - 1
        while bytes_seen < args.train_byte_budget:
            starts = [random.randrange(max_start) for _ in range(args.batch_size)]
            train_batch(starts, args.seq_len)
            if steps > 20_000:
                raise RuntimeError(
                    "Raw-byte training budget did not converge within step guard"
                )
    train_seconds = time.perf_counter() - started
    train_cpu_seconds = time.process_time() - train_cpu_started
    interface_final_audit_started = time.perf_counter()
    final_interface_audit = interface.audit(fail_on_collision=True)
    interface_final_audit_seconds = time.perf_counter() - interface_final_audit_started
    evaluation_started = time.perf_counter()
    evaluation_cpu_started = time.process_time()
    evaluation = evaluate(model, interface, validation, args, device)
    evaluation_seconds = time.perf_counter() - evaluation_started
    evaluation_cpu_seconds = time.process_time() - evaluation_cpu_started

    eval_input_spikes = float(evaluation["eval_input_spikes"])
    eval_hidden_spikes = float(evaluation["eval_hidden_spikes"])
    eval_units = int(evaluation["eval_modelling_units"])
    model_params = sum(parameter.numel() for parameter in model.parameters())
    interface_params = sum(parameter.numel() for parameter in interface.parameters())
    unused_builtin_input_params = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.startswith("embedding.") or name.startswith("input_encoder.")
    )
    active_trainable_params = (
        model_params - unused_builtin_input_params + interface_params
    )
    model_active_state_storage_bytes = sum(
        tensor.numel() * tensor.element_size()
        for name, tensor in model.state_dict().items()
        if not (
            name.startswith("embedding.")
            or name.startswith("input_encoder.")
        )
    )
    interface_state_storage_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in interface.state_dict().values()
    )
    if args.backbone == "srwkv_v2":
        train_operation_ledger = srwkv_v2_operation_ledger(
            modelling_units=units_seen,
            input_dimension=int(final_interface_audit["interface_dimension"]),
            hidden_dimension=args.hidden_dim,
            num_layers=args.num_layers,
            vocabulary_size=training.vocab_size,
            spike_steps=args.spike_steps,
            input_spikes=train_stats["input_spikes"],
        )
        eval_operation_ledger = srwkv_v2_operation_ledger(
            modelling_units=eval_units,
            input_dimension=int(final_interface_audit["interface_dimension"]),
            hidden_dimension=args.hidden_dim,
            num_layers=args.num_layers,
            vocabulary_size=training.vocab_size,
            spike_steps=args.spike_steps,
            input_spikes=eval_input_spikes,
        )
        train_operation_estimates = training_operation_estimates(
            train_operation_ledger,
            trainable_parameters=active_trainable_params,
            optimizer_steps=steps,
        )
    elif args.backbone == "spikegpt_reference":
        learned_interface_threshold_elements_per_unit = (
            int(final_interface_audit["interface_dimension"])
            if args.interface_encoder in {
                "learned_topk_4of32_ste",
                "learned_threshold32_rate",
                "spikegpt_embedding64_atan",
            }
            else 0
        )
        train_operation_ledger = spikegpt_reference_operation_ledger(
            modelling_units=units_seen,
            input_dimension=int(final_interface_audit["interface_dimension"]),
            hidden_dimension=args.hidden_dim,
            num_layers=args.num_layers,
            vocabulary_size=training.vocab_size,
            input_spikes=train_stats["input_spikes"],
            input_adapter_spikes=train_stats.get("input_adapter_spikes", 0.0),
            interface_parameter_count=interface_params,
            input_spike_generation_threshold_elements=(
                units_seen * learned_interface_threshold_elements_per_unit
            ),
        )
        eval_operation_ledger = spikegpt_reference_operation_ledger(
            modelling_units=eval_units,
            input_dimension=int(final_interface_audit["interface_dimension"]),
            hidden_dimension=args.hidden_dim,
            num_layers=args.num_layers,
            vocabulary_size=training.vocab_size,
            input_spikes=eval_input_spikes,
            input_adapter_spikes=float(evaluation.get("eval_input_adapter_spikes", 0.0)),
            interface_parameter_count=interface_params,
            input_spike_generation_threshold_elements=(
                eval_units * learned_interface_threshold_elements_per_unit
            ),
        )
        train_operation_estimates = training_operation_estimates(
            train_operation_ledger,
            trainable_parameters=active_trainable_params,
            optimizer_steps=steps,
        )
    else:
        train_operation_ledger = {}
        eval_operation_ledger = {}
        train_operation_estimates = {}
    if (
        args.interface_encoder in SPIKEPATCH_INTERFACE_MODES
        or args.interface_encoder == INJECTIVE_RECURRENT_BYTE_MODE
    ):
        train_operation_ledger["frontend_operation_count_complete"] = 0
        eval_operation_ledger["frontend_operation_count_complete"] = 0
        train_operation_ledger["frontend_synaptic_lif_and_threshold_count_complete"] = 1
        eval_operation_ledger["frontend_synaptic_lif_and_threshold_count_complete"] = 1
        train_operation_ledger["frontend_execution_is_dense_masked_pytorch_reference"] = 1
        eval_operation_ledger["frontend_execution_is_dense_masked_pytorch_reference"] = 1
    row: Dict[str, object] = {
        "format": "canonical_lossless_lm_v4_common_schedule_spikegpt_ledger",
        "protocol_id": PROTOCOL_ID,
        "tier": (
            (
                "stage_b_spikegpt_full_train_validation"
                if args.backbone == "spikegpt_reference"
                else "stage_a_full_train_validation"
            )
            if args.data_tier == "full_train_validation"
            else "development_pipeline_pilot"
        ),
        "data_tier": args.data_tier,
        "git_commit": git_commit(),
        "dataset": args.dataset,
        "method": args.method,
        "seed": args.seed,
        "training_path": str(training_path),
        "validation_path": validation_source_description,
        "evaluation_path": validation_source_description,
        "training_sha256": training_digest,
        "validation_sha256": validation_digest,
        "evaluation_sha256": validation_digest,
        "evaluation_split_role": (
            "test" if args.evaluation_source == "canonical_test" else "validation"
        ),
        "training_schedule": args.training_schedule,
        "evaluation_source": args.evaluation_source,
        "outer_validation_split_touched": int(
            args.evaluation_source == "canonical_validation"
        ),
        "outer_validation_opened_for_hyperparameter_selection": 0,
        "inner_fit_end_raw_byte_exclusive": inner_fit_end,
        "inner_holdout_end_raw_byte_exclusive": inner_holdout_end,
        "inner_holdout_sha256": (
            validation_digest
            if args.evaluation_source == "training_inner_holdout"
            else "not_applicable"
        ),
        "lm_training_selection_source_path": str(training_path),
        "lm_training_selection_source_sha256": training_digest,
        "lm_training_selection_start_raw_byte": 0,
        "lm_training_selection_end_raw_byte_exclusive": len(modelling_training_raw),
        "lm_training_selection_raw_bytes": len(modelling_training_raw),
        "lm_training_selection_characters": len(modelling_training_text),
        "lm_training_selection_sha256": modelling_training_digest,
        "lm_training_selection_utf8_complete": 1,
        "lm_training_selection_shared_across_methods_by_contract": int(
            args.training_schedule == "common_utf8_prefix_one_pass"
        ),
        **canonical_contract,
        "fit_uses_training_only": int(
            args.method not in EXTERNAL_CONVENTIONAL_MODELS
        ),
        "external_pretrained_tokenizer_fit_not_repeated": int(
            args.method in EXTERNAL_CONVENTIONAL_MODELS
        ),
        "opened_data_paths_json": json.dumps(
            [
                {"role": role, "path": path}
                for role, path in OPENED_DATA_PATHS
            ],
            separators=(",", ":"),
        ),
        "opened_data_roles_json": json.dumps(
            [role for role, _ in OPENED_DATA_PATHS],
            separators=(",", ":"),
        ),
        "opened_data_path_count": len(OPENED_DATA_PATHS),
        "data_access_ledger_scope": (
            "trainer_declared_corpus_payload_opens_with_repository_"
            "provenance_audit_for_active_entrypoints"
        ),
        "test_split_touched": int(
            any(role == "test" for role, _ in OPENED_DATA_PATHS)
        ),
        "test_split_path_available_to_trainer": int(
            args.evaluation_source == "canonical_test"
        ),
        "evaluation_parameter_updates": 0,
        "confirmatory_test_evaluation": int(
            args.evaluation_source == "canonical_test"
        ),
        "lossless_raw_byte_units": 1,
        "next_target": "next_modelling_unit_id",
        "output_is_spike_sequence": 0,
        "interface_code": args.interface_encoder,
        "interface_encoder": args.interface_encoder,
        "interface_initial_collisions": initial_interface_audit["interface_collisions"],
        "interface_code_collisions": final_interface_audit["interface_collisions"],
        "interface_dimension": final_interface_audit["interface_dimension"],
        "interface_active_bits": final_interface_audit["interface_spikes_per_unit_mean"],
        "interface_rate_regularizer_weight": args.interface_rate_weight,
        "interface_rate_regularizer_mean": sum(interface_regularizers) / max(len(interface_regularizers), 1),
        "interface_build_and_homeostasis_seconds_current_run": interface_build_seconds,
        "interface_build_and_homeostasis_cpu_seconds_current_run": interface_build_cpu_seconds,
        "interface_initial_audit_seconds_current_run": interface_initial_audit_seconds,
        "interface_final_audit_seconds_current_run": interface_final_audit_seconds,
        "vocab_size": training.vocab_size,
        "pre_bos_vocab_size": pre_bos_vocab_size,
        "shared_bos_added": 1,
        "bos_id": pre_bos_vocab_size,
        "bos_is_prediction_target": 0,
        "training_num_units": len(training.ids) - 1,
        "validation_num_units": len(validation.ids) - 1,
        "training_bytes_per_unit": len(modelling_training_raw) / (len(training.ids) - 1),
        "validation_bytes_per_unit": len(validation_raw) / (len(validation.ids) - 1),
        "train_raw_byte_budget_requested": args.train_byte_budget,
        "train_raw_bytes_seen": bytes_seen,
        "train_target_units_covered_once": train_target_units_covered_once,
        "train_modelling_units_seen": units_seen,
        "train_optimizer_steps": steps,
        "train_loss_mean_per_unit": sum(losses) / len(losses),
        "train_loss_initial": losses[0],
        "train_loss_final": losses[-1],
        "train_seconds": train_seconds,
        "train_cpu_seconds_current_run": train_cpu_seconds,
        "evaluation_seconds": evaluation_seconds,
        "evaluation_cpu_seconds_current_run": evaluation_cpu_seconds,
        "frontend_prepare_cpu_seconds_current_run": frontend_prepare_cpu_seconds,
        "optimizer": optimizer_name,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "backbone": args.backbone,
        "spike_steps": args.spike_steps,
        "backbone_operation_accounting_status": backbone_accounting_status(args.backbone),
        "total_params": model_params + interface_params,
        "model_params_including_bypassed_builtin_input_adapter": model_params,
        "interface_params": interface_params,
        "active_trainable_params": active_trainable_params,
        "active_parameter_storage_bytes_fp32": active_trainable_params * 4,
        "model_active_state_storage_bytes_reference": model_active_state_storage_bytes,
        "interface_state_storage_bytes_reference": interface_state_storage_bytes,
        "active_state_storage_bytes_reference": (
            model_active_state_storage_bytes + interface_state_storage_bytes
        ),
        "bypassed_builtin_input_adapter_params": unused_builtin_input_params,
        "train_interface_spikes": train_stats["input_spikes"],
        "train_hidden_spikes": train_stats["hidden_spikes"],
        **{f"train_frontend_{key}": value for key, value in train_frontend_stats.items()},
        **evaluation,
        "eval_neural_spikes_no_frontend_input_double_count": (
            eval_input_spikes + eval_hidden_spikes
        ),
        "eval_dense_readout_macs": eval_operation_ledger.get(
            "dense_readout_macs",
            eval_units * args.hidden_dim * training.vocab_size,
        ),
        "energy_status": (
            "hardware_energy_estimate_deferred_until_operation_"
            "and_memory_ledger_is_complete"
        ),
        **{
            f"train_{key}": value
            for key, value in train_operation_ledger.items()
        },
        **{
            f"train_{key}": value
            for key, value in train_operation_estimates.items()
        },
        **{
            f"eval_{key}": value
            for key, value in eval_operation_ledger.items()
        },
        **final_interface_audit,
        **frontend_metadata,
        **{f"training_sequence_{key}": value for key, value in training.metadata.items()},
        **{f"validation_sequence_{key}": value for key, value in validation.metadata.items()},
    }
    if args.evaluation_source == "canonical_test":
        if row["test_split_touched"] != 1:
            raise AssertionError("Authorized confirmatory test path was not opened")
        if any(role == "validation" for role, _ in OPENED_DATA_PATHS):
            raise AssertionError("Validation was reopened during confirmatory testing")
    elif row["test_split_touched"] != 0:
        raise AssertionError("A test-labelled data path was opened before final selection")
    row.update(
        experiment_compute_energy(
            row,
            training_raw_bytes=len(modelling_training_raw),
            validation_raw_bytes=len(validation_raw),
        )
    )
    row.update(
        experiment_frontend_memory_ledger(
            row,
            training_raw_bytes=len(modelling_training_raw),
            validation_raw_bytes=len(validation_raw),
        )
    )
    row.update({
        "process_wall_seconds_current_run": time.perf_counter() - overall_started,
        "process_cpu_seconds_current_run": time.process_time() - process_started,
        "process_peak_rss_kib": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        ),
        "process_peak_rss_platform_semantics": "linux_ru_maxrss_kib",
    })
    write_csv(Path(args.output_csv), row)
    print(
        f"PASS canonical LM pilot method={args.method} "
        f"BPB={row['bits_per_raw_byte']:.6f} BPC={row['bits_per_character']:.6f}"
    )
    print(f"Output: {args.output_csv}")


if __name__ == "__main__":
    main()
