#!/usr/bin/env python3
"""Train one codec-free patch interface and evaluate one authorised split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import time
from pathlib import Path
import sys
from typing import Dict, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from scripts.train_frontend_language_model import (
    DATASET_TIERS,
    read_strict,
    utf8_safe_prefix,
)
from src.canonical_data_contract import (
    assert_train_test_contract,
    assert_train_validation_contract,
)
from src.energy.scalable_patch_projection import scalable_patch_projection
from src.models.spike_native_patch_lm import (
    PATCH_ARCHITECTURES,
    PatchSequence,
    SpikeNativePatchLanguageModel,
    complete_patch_blocks,
    fixed_width_patches,
    merge_additive_stats,
    validate_patch_cover,
)
from src.tokenizers.spike_native_patch_frontends import (
    JointSpikingPatchSegmenter,
    SpikePatchSNNConfig,
)


CANONICAL_ROOT = Path(
    os.environ.get(
        "SPIKE_TOKENIZER_CANONICAL_ROOT",
        str(ROOT.parent / "data/canonical"),
    )
)


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def atomic_row(path: Path, row: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted(row))
            writer.writeheader()
            writer.writerow(row)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _numeric(stats: Mapping[str, object], key: str) -> float:
    value = stats.get(key, 0)
    return float(value) if isinstance(value, (int, float)) else 0.0


def segmenter_counts(
    stats: Mapping[str, object],
    *,
    fixed_raw_bytes: int = 0,
    fixed_patches: int = 0,
) -> Dict[str, float]:
    if fixed_raw_bytes:
        return {
            "segmenter_synaptic_accumulations": 0.0,
            "segmenter_state_updates": float(fixed_raw_bytes),
            "segmenter_threshold_reset_elements": float(fixed_raw_bytes),
            "segmenter_spike_events": float(fixed_patches),
        }
    return {
        "segmenter_synaptic_accumulations": sum(
            _numeric(stats, key)
            for key in (
                "input_synaptic_accumulations_event_driven",
                "recurrent_synaptic_accumulations_event_driven",
                "readout_synaptic_accumulations_event_driven",
                "boundary_synaptic_accumulations_event_driven",
            )
        ),
        "segmenter_state_updates": sum(
            _numeric(stats, key)
            for key in (
                "hidden_lif_state_updates",
                "output_membrane_state_updates",
                "boundary_lif_state_updates",
            )
        ),
        "segmenter_threshold_reset_elements": sum(
            _numeric(stats, key)
            for key in (
                "hidden_threshold_evaluations",
                "output_threshold_evaluations",
                "boundary_threshold_evaluations",
                "hidden_reset_gate_elements",
                "output_reset_gate_elements",
            )
        ),
        "segmenter_spike_events": sum(
            _numeric(stats, key)
            for key in (
                "raw_byte_input_spike_events",
                "hidden_spike_events",
                "content_spike_events",
                "patch_completion_spike_events",
                "natural_boundary_spikes",
            )
        ),
    }


def add_counts(target: Dict[str, float], values: Mapping[str, float]) -> None:
    for key, value in values.items():
        target[key] = target.get(key, 0.0) + float(value)


def build_patch_sequence(
    raw: bytes,
    *,
    architecture: str,
    raw_sha256: str,
    segmenter: JointSpikingPatchSegmenter | None,
) -> tuple[PatchSequence, Dict[str, float]]:
    spec = PATCH_ARCHITECTURES[architecture]
    maximum = int(spec["max_patch_bytes"])
    if spec["patch_mode"] == "fixed":
        patches = fixed_width_patches(raw, maximum)
        counts = segmenter_counts(
            {}, fixed_raw_bytes=len(raw), fixed_patches=len(patches)
        )
    else:
        if segmenter is None:
            raise AssertionError("adaptive patch architecture lacks its segmenter")
        patches, steps, stats = segmenter.segment_bytes(raw)
        if len(steps) != len(patches) or any(step != 1 for step in steps):
            raise AssertionError("joint segmenter did not emit one global step per patch")
        counts = segmenter_counts(stats)
    validate_patch_cover(patches, raw, max_patch_bytes=maximum)
    return (
        PatchSequence(tuple(patches), maximum, raw_sha256),
        counts,
    )


def run_blocks(
    *,
    model: SpikeNativePatchLanguageModel,
    sequence: PatchSequence,
    patches_per_block: int,
    batch_size: int,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, int, Dict[str, float]]:
    blocks = complete_patch_blocks(len(sequence.patches), patches_per_block)
    grouped: Dict[int, list[int]] = {}
    for start, length in blocks:
        grouped.setdefault(length, []).append(start)
    total_nll = 0.0
    total_bytes = 0
    totals: Dict[str, float] = {}
    training = optimizer is not None
    model.train(training)
    for length, starts_for_length in sorted(grouped.items(), reverse=True):
        for offset in range(0, len(starts_for_length), batch_size):
            starts = starts_for_length[offset : offset + batch_size]
            source, source_mask, target, target_mask = sequence.source_and_targets(
                starts, length, device=torch.device("cpu")
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits, stats = model(source, source_mask, target, target_mask)
            selected_logits = logits[target_mask]
            selected_targets = target[target_mask]
            loss_sum = F.cross_entropy(
                selected_logits, selected_targets, reduction="sum"
            )
            if training:
                (loss_sum / max(int(target_mask.sum().item()), 1)).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_nll += float(loss_sum.detach().item())
            total_bytes += int(target_mask.sum().item())
            merge_additive_stats(totals, stats)
    if total_bytes != sequence.raw_bytes:
        raise AssertionError(
            f"block traversal covered {total_bytes} bytes, expected {sequence.raw_bytes}"
        )
    return total_nll, total_bytes, totals


def optimizer_step_count(
    sequence: PatchSequence,
    *,
    patches_per_block: int,
    batch_size: int,
) -> int:
    lengths: Dict[int, int] = {}
    for _, length in complete_patch_blocks(
        len(sequence.patches), patches_per_block
    ):
        lengths[length] = lengths.get(length, 0) + 1
    return sum(
        (count + batch_size - 1) // batch_size
        for count in lengths.values()
    )


def prefix_stats(
    row: Dict[str, object],
    prefix: str,
    stats: Mapping[str, float],
) -> None:
    for key, value in stats.items():
        row[f"{prefix}_{key}"] = value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_TIERS["development"]))
    parser.add_argument(
        "--data-tier",
        required=True,
        choices=("development", "full_train_validation"),
    )
    parser.add_argument("--architecture", required=True, choices=sorted(PATCH_ARCHITECTURES))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--evaluation-source",
        choices=("canonical_validation", "canonical_test"),
        default="canonical_validation",
    )
    parser.add_argument("--train-byte-budget", type=int, default=1_000_000)
    parser.add_argument("--patches-per-block", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--output-csv", required=True, type=Path)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)

    paths = DATASET_TIERS[args.data_tier][args.dataset]
    training_path = CANONICAL_ROOT / paths["train"]
    if args.evaluation_source == "canonical_validation":
        evaluation_path = CANONICAL_ROOT / paths["validation"]
        evaluation_role = "validation"
        contract = assert_train_validation_contract(
            canonical_root=CANONICAL_ROOT,
            dataset=args.dataset,
            data_tier=args.data_tier,
            training_path=training_path,
            validation_path=evaluation_path,
        )
    else:
        if args.data_tier != "full_train_validation":
            raise AssertionError(
                "canonical test evaluation requires full_train_validation"
            )
        evaluation_path = CANONICAL_ROOT / paths["test"]
        evaluation_role = "test"
        contract = assert_train_test_contract(
            canonical_root=CANONICAL_ROOT,
            dataset=args.dataset,
            data_tier=args.data_tier,
            training_path=training_path,
            test_path=evaluation_path,
        )
    training_source, _ = read_strict(training_path, role="training")
    evaluation_raw, _ = read_strict(
        evaluation_path, role=evaluation_role
    )
    training_raw, _ = utf8_safe_prefix(
        training_source, args.train_byte_budget
    )
    training_sha = sha256(training_raw)
    evaluation_sha = sha256(evaluation_raw)
    spec = PATCH_ARCHITECTURES[args.architecture]
    segmenter = None
    fit_counts: Dict[str, float] = {}
    fit_seconds = 0.0
    segmenter_state_before_evaluation = "not_applicable"
    if spec["patch_mode"] == "joint_snn_homeostatic":
        fit_started = time.process_time()
        segmenter = JointSpikingPatchSegmenter(
            learning_rule="homeostatic",
            seed=args.seed,
            training_bytes=training_raw,
            training_sha256=training_sha,
            config=SpikePatchSNNConfig(
                min_patch_bytes=2,
                max_patch_bytes=int(spec["max_patch_bytes"]),
                target_boundary_rate_per_byte=0.125,
                seed=args.seed,
            ),
        )
        fit_seconds = time.process_time() - fit_started
        fit_report = segmenter.encoder.homeostasis_report
        fit_stats = {
            key.removeprefix("homeostasis_fit_"): value
            for key, value in fit_report.items()
            if key.startswith("homeostasis_fit_")
        }
        fit_counts = segmenter_counts(fit_stats)
        segmenter_state_before_evaluation = segmenter.encoder.state_digest()
    training_sequence, training_segment_counts = build_patch_sequence(
        training_raw,
        architecture=args.architecture,
        raw_sha256=training_sha,
        segmenter=segmenter,
    )
    evaluation_sequence, evaluation_segment_counts = build_patch_sequence(
        evaluation_raw,
        architecture=args.architecture,
        raw_sha256=evaluation_sha,
        segmenter=segmenter,
    )
    # Evaluation is deterministic and may not update frozen homeostasis.
    repeat, repeat_counts = build_patch_sequence(
        evaluation_raw,
        architecture=args.architecture,
        raw_sha256=evaluation_sha,
        segmenter=segmenter,
    )
    if repeat.patches != evaluation_sequence.patches:
        raise AssertionError("evaluation patching is not deterministic")
    if repeat_counts != evaluation_segment_counts:
        raise AssertionError("evaluation segmenter counts are not deterministic")
    if (
        segmenter is not None
        and segmenter.encoder.state_digest() != segmenter_state_before_evaluation
    ):
        raise AssertionError("evaluation changed frozen segmenter state")

    model = SpikeNativePatchLanguageModel(
        max_patch_bytes=int(spec["max_patch_bytes"]),
        hierarchical_encoder=bool(spec["hierarchical"]),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        seed=args.seed,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_started = time.process_time()
    train_nll, train_bytes, train_stats = run_blocks(
        model=model,
        sequence=training_sequence,
        patches_per_block=args.patches_per_block,
        batch_size=args.batch_size,
        optimizer=optimizer,
    )
    train_cpu = time.process_time() - train_started
    with torch.no_grad():
        evaluation_started = time.process_time()
        evaluation_nll, evaluation_bytes, evaluation_stats = run_blocks(
            model=model,
            sequence=evaluation_sequence,
            patches_per_block=args.patches_per_block,
            batch_size=args.batch_size,
            optimizer=None,
        )
        evaluation_cpu = time.process_time() - evaluation_started
    add_counts(training_segment_counts, fit_counts)
    add_counts(train_stats, training_segment_counts)
    add_counts(evaluation_stats, evaluation_segment_counts)

    parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    buffer_bytes = sum(
        buffer.numel() * buffer.element_size()
        for buffer in model.buffers()
    )
    if segmenter is not None:
        buffer_bytes += sum(
            value.numel() * value.element_size()
            for value in segmenter.encoder.state_dict().values()
        )
    row: Dict[str, object] = {
        "format": "scalable_spike_native_patch_row_v1",
        "status": "PASS",
        "dataset": args.dataset,
        "data_tier": args.data_tier,
        "architecture": args.architecture,
        "frontend_family": (
            "codec_free_adaptive_spike_native_patch_local_decoder"
            if spec["patch_mode"] != "fixed"
            else "codec_free_fixed_spike_native_patch_local_decoder"
        ),
        "method": "byte_patch_autoregressive",
        "name": args.architecture,
        "seed": args.seed,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "patches_per_block": args.patches_per_block,
        "train_byte_budget": args.train_byte_budget,
        "training_raw_bytes": train_bytes,
        # The projection retains the legacy ``validation_`` prefix so rows can
        # share one comparison schema.  ``evaluation_*`` and ``test_*`` aliases
        # below state the actual split without ambiguity.
        "validation_raw_bytes": evaluation_bytes,
        "evaluation_raw_bytes": evaluation_bytes,
        "training_source_sha256": sha256(training_source),
        "training_selection_sha256": training_sha,
        "validation_source_sha256": evaluation_sha,
        "evaluation_source_sha256": evaluation_sha,
        "evaluation_source": args.evaluation_source,
        "evaluation_split_role": evaluation_role,
        "evaluation_path": str(evaluation_path),
        "training_nll_nats": train_nll,
        "validation_nll_nats": evaluation_nll,
        "evaluation_nll_nats": evaluation_nll,
        "training_bpb": train_nll / train_bytes / math.log(2.0),
        "bpb": evaluation_nll / evaluation_bytes / math.log(2.0),
        "evaluation_bpb": evaluation_nll / evaluation_bytes / math.log(2.0),
        "training_patch_count": len(training_sequence.patches),
        "validation_patch_count": len(evaluation_sequence.patches),
        "evaluation_patch_count": len(evaluation_sequence.patches),
        "training_mean_patch_bytes": training_sequence.mean_patch_bytes,
        "validation_mean_patch_bytes": evaluation_sequence.mean_patch_bytes,
        "evaluation_mean_patch_bytes": evaluation_sequence.mean_patch_bytes,
        "training_units_per_raw_byte": (
            len(training_sequence.patches) / train_bytes
        ),
        "validation_units_per_raw_byte": (
            len(evaluation_sequence.patches) / evaluation_bytes
        ),
        "evaluation_units_per_raw_byte": (
            len(evaluation_sequence.patches) / evaluation_bytes
        ),
        "training_effective_raw_context_bytes": (
            args.patches_per_block * training_sequence.mean_patch_bytes
        ),
        "validation_effective_raw_context_bytes": (
            args.patches_per_block * evaluation_sequence.mean_patch_bytes
        ),
        "evaluation_effective_raw_context_bytes": (
            args.patches_per_block * evaluation_sequence.mean_patch_bytes
        ),
        "global_input_dimension": model.encoder.output_dim,
        "model_parameter_bytes_fp32": parameter_bytes,
        "model_trainable_parameters": trainable_parameters,
        "training_optimizer_steps": optimizer_step_count(
            training_sequence,
            patches_per_block=args.patches_per_block,
            batch_size=args.batch_size,
        ),
        "model_buffer_bytes": buffer_bytes,
        "frontend_fit_cpu_seconds_current_run": fit_seconds,
        "lm_training_cpu_seconds_current_run": train_cpu,
        "validation_cpu_seconds_current_run": evaluation_cpu,
        "evaluation_cpu_seconds_current_run": evaluation_cpu,
        "fully_spiking_frontend": 1,
        "fully_spike_communicating_patch_encoder": 1,
        "fully_spike_communicating_global_to_local_decoder": 1,
        "dense_final_byte_likelihood_readout_present": 1,
        "digital_tokenizer_present": 0,
        "digital_patch_codec_present": 0,
        "open_vocabulary_lossless_byte_coverage": 1,
        "one_global_backbone_step_per_patch": 1,
        "one_global_backbone_step_per_byte": 0,
        "local_decoder_required_for_exact_byte_likelihood": 1,
        "canonical_validation_is_development_evidence": int(
            args.evaluation_source == "canonical_validation"
        ),
        "test_split_touched": int(
            args.evaluation_source == "canonical_test"
        ),
        "outer_validation_parameter_updates": 0,
        "evaluation_parameter_updates": 0,
        "validation_repeated_deterministically": 1,
        "evaluation_repeated_deterministically": 1,
        "confirmatory_test_evaluation": int(
            args.evaluation_source == "canonical_test"
        ),
        "configuration_selected_on_current_validation": 0,
        "physical_hardware_energy_measurement": 0,
        **contract,
    }
    prefix_stats(row, "train", train_stats)
    prefix_stats(row, "validation", evaluation_stats)
    if args.evaluation_source == "canonical_test":
        row.update({
            "test_raw_bytes": evaluation_bytes,
            "test_source_sha256": evaluation_sha,
            "test_nll_nats": evaluation_nll,
            "test_bpb": row["evaluation_bpb"],
            "test_patch_count": len(evaluation_sequence.patches),
            "test_mean_patch_bytes": evaluation_sequence.mean_patch_bytes,
            "test_units_per_raw_byte": (
                len(evaluation_sequence.patches) / evaluation_bytes
            ),
        })
    row.update(scalable_patch_projection(row))
    atomic_row(args.output_csv, row)
    print(
        f"PASS {args.dataset}/{args.architecture}/s{args.seed} "
        f"{evaluation_role}_BPB={row['bpb']:.6f} "
        f"mean_patch={evaluation_sequence.mean_patch_bytes:.3f}"
    )


if __name__ == "__main__":
    main()
