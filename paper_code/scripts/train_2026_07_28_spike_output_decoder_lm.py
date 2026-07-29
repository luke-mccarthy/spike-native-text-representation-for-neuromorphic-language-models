#!/usr/bin/env python3
"""Train one 2026-07-28 codec-free patch decoder/output-head row.

The patch encoder and SpikeGPT reference backbone remain unchanged. This study
crosses decoder capacity with either the existing dense 256-way reference
readout or a 256-neuron spike-count byte output head. It reports BPB and direct
next-byte/whole-patch prediction metrics plus the existing operation/energy
projection fields.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Dict, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from scripts.train_2026_07_17_canonical_lm import DATASET_TIERS, read_strict, utf8_safe_prefix
from scripts.train_2026_07_23_scalable_patch_lm import (
    build_patch_sequence,
    optimizer_step_count,
    segmenter_counts,
)
from src.canonical_data_contract import assert_train_test_contract, assert_train_validation_contract
from src.energy.scalable_patch_projection import scalable_patch_projection
from src.models.spike_native_patch_lm import complete_patch_blocks, merge_additive_stats
from src.models.spike_native_patch_lm_v3 import (
    OUTPUT_HEADS,
    PATCH_ARCHITECTURES,
    SpikeNativePatchLanguageModelV3,
)
from src.tokenizers.spike_native_patch_frontends import JointSpikingPatchSegmenter, SpikePatchSNNConfig


CANONICAL_ROOT = Path(
    os.environ.get(
        "SPIKE_TOKENIZER_CANONICAL_ROOT",
        "/scratch/user/lukemccarthy/datasets/spike_tokenizer_canonical_2026_07_16",
    )
)
FORMAT = "spike_output_decoder_row_v1"


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


def add_counts(target: Dict[str, float], values: Mapping[str, float]) -> None:
    for key, value in values.items():
        target[key] = target.get(key, 0.0) + float(value)


def configure_reproducibility(seed: int) -> Dict[str, object]:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    return {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "deterministic_algorithms_enabled": int(torch.are_deterministic_algorithms_enabled()),
        "torch_num_threads": torch.get_num_threads(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "unset"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS", "unset"),
    }


def run_blocks(
    *,
    model: SpikeNativePatchLanguageModelV3,
    sequence,
    patches_per_block: int,
    batch_size: int,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, int, Dict[str, float], Dict[str, float]]:
    blocks = complete_patch_blocks(len(sequence.patches), patches_per_block)
    grouped: Dict[int, list[int]] = {}
    for start, length in blocks:
        grouped.setdefault(length, []).append(start)

    total_nll = 0.0
    total_bytes = 0
    totals: Dict[str, float] = {}
    metrics = {
        "top1_correct_bytes": 0.0,
        "top5_correct_bytes": 0.0,
        "exact_patch_correct": 0.0,
        "patches_evaluated": 0.0,
        "correct_prefix_bytes_sum": 0.0,
    }
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
            scores, stats = model(source, source_mask, target, target_mask)
            selected_scores = scores[target_mask]
            selected_targets = target[target_mask]
            loss_sum = F.cross_entropy(selected_scores, selected_targets, reduction="sum")
            if training:
                (loss_sum / max(int(target_mask.sum().item()), 1)).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_nll += float(loss_sum.detach().item())
            active_bytes = int(target_mask.sum().item())
            total_bytes += active_bytes
            merge_additive_stats(totals, stats)

            with torch.no_grad():
                predictions = scores.argmax(dim=-1)
                top5 = scores.topk(k=5, dim=-1).indices
                metrics["top1_correct_bytes"] += float(
                    ((predictions == target) & target_mask).sum().item()
                )
                metrics["top5_correct_bytes"] += float(
                    ((top5 == target.unsqueeze(-1)).any(dim=-1) & target_mask).sum().item()
                )
                flat_pred = predictions.reshape(-1, predictions.shape[-1])
                flat_target = target.reshape(-1, target.shape[-1])
                flat_mask = target_mask.reshape(-1, target_mask.shape[-1])
                for pred_row, target_row, mask_row in zip(flat_pred, flat_target, flat_mask):
                    patch_length = int(mask_row.sum().item())
                    if patch_length == 0:
                        continue
                    correct = (pred_row[:patch_length] == target_row[:patch_length])
                    metrics["patches_evaluated"] += 1.0
                    metrics["exact_patch_correct"] += float(bool(correct.all().item()))
                    prefix = 0
                    for value in correct.tolist():
                        if not value:
                            break
                        prefix += 1
                    metrics["correct_prefix_bytes_sum"] += float(prefix)

    if total_bytes != sequence.raw_bytes:
        raise AssertionError(
            f"block traversal covered {total_bytes} bytes, expected {sequence.raw_bytes}"
        )
    return total_nll, total_bytes, totals, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_TIERS["development"]))
    parser.add_argument("--data-tier", required=True, choices=("development", "full_train_validation"))
    parser.add_argument("--architecture", required=True, choices=sorted(PATCH_ARCHITECTURES))
    parser.add_argument("--output-head", required=True, choices=OUTPUT_HEADS)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--evaluation-source", choices=("canonical_validation", "canonical_test"),
        default="canonical_validation",
    )
    parser.add_argument("--train-byte-budget", type=int, default=1_000_000)
    parser.add_argument("--patches-per-block", type=int, required=True)
    parser.add_argument("--target-boundary-rate-per-byte", type=float, default=0.200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--decoder-hidden-dim", type=int, required=True)
    parser.add_argument("--decoder-num-layers", type=int, required=True)
    parser.add_argument("--output-microsteps", type=int, default=4)
    parser.add_argument("--output-score-scale", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--output-csv", required=True, type=Path)
    args = parser.parse_args()

    reproducibility = configure_reproducibility(args.seed)
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
            raise AssertionError("canonical test evaluation requires full_train_validation")
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
    evaluation_raw, _ = read_strict(evaluation_path, role=evaluation_role)
    training_raw, _ = utf8_safe_prefix(training_source, args.train_byte_budget)
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
                target_boundary_rate_per_byte=args.target_boundary_rate_per_byte,
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
    repeat_sequence, repeat_counts = build_patch_sequence(
        evaluation_raw,
        architecture=args.architecture,
        raw_sha256=evaluation_sha,
        segmenter=segmenter,
    )
    if repeat_sequence.patches != evaluation_sequence.patches:
        raise AssertionError("evaluation patching is not deterministic")
    if repeat_counts != evaluation_segment_counts:
        raise AssertionError("evaluation segmenter counts are not deterministic")
    if segmenter is not None and segmenter.encoder.state_digest() != segmenter_state_before_evaluation:
        raise AssertionError("evaluation changed frozen segmenter state")

    model = SpikeNativePatchLanguageModelV3(
        max_patch_bytes=int(spec["max_patch_bytes"]),
        hierarchical_encoder=bool(spec["hierarchical"]),
        output_head=args.output_head,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        decoder_hidden_dim=args.decoder_hidden_dim,
        decoder_num_layers=args.decoder_num_layers,
        output_microsteps=args.output_microsteps,
        output_score_scale=args.output_score_scale,
        seed=args.seed,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_started = time.process_time()
    train_nll, train_bytes, train_stats, train_metrics = run_blocks(
        model=model,
        sequence=training_sequence,
        patches_per_block=args.patches_per_block,
        batch_size=args.batch_size,
        optimizer=optimizer,
    )
    train_cpu = time.process_time() - train_started

    with torch.no_grad():
        evaluation_started = time.process_time()
        evaluation_nll, evaluation_bytes, evaluation_stats, evaluation_metrics = run_blocks(
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

    parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    buffer_bytes = sum(buffer.numel() * buffer.element_size() for buffer in model.buffers())
    if segmenter is not None:
        buffer_bytes += sum(
            value.numel() * value.element_size()
            for value in segmenter.encoder.state_dict().values()
        )

    evaluation_patch_count = max(int(evaluation_metrics["patches_evaluated"]), 1)
    spike_head = int(args.output_head == "spike_count_256")
    row: Dict[str, object] = {
        "format": FORMAT,
        "status": "PASS",
        "dataset": args.dataset,
        "data_tier": args.data_tier,
        "architecture": args.architecture,
        "method": "byte_patch_autoregressive_spike_output_v3",
        "name": f"{args.architecture}_{args.output_head}_dh{args.decoder_hidden_dim}_dl{args.decoder_num_layers}",
        "seed": args.seed,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "decoder_hidden_dim": args.decoder_hidden_dim,
        "decoder_num_layers": args.decoder_num_layers,
        "decoder_capacity_parameters": args.decoder_hidden_dim * args.decoder_num_layers,
        "output_head": args.output_head,
        "output_microsteps": args.output_microsteps,
        "output_score_scale": args.output_score_scale,
        "target_boundary_rate_per_byte": args.target_boundary_rate_per_byte,
        "patches_per_block": args.patches_per_block,
        "train_byte_budget": args.train_byte_budget,
        "training_raw_bytes": train_bytes,
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
        "training_top1_accuracy": train_metrics["top1_correct_bytes"] / train_bytes,
        "training_top5_accuracy": train_metrics["top5_correct_bytes"] / train_bytes,
        "evaluation_top1_accuracy": evaluation_metrics["top1_correct_bytes"] / evaluation_bytes,
        "evaluation_top5_accuracy": evaluation_metrics["top5_correct_bytes"] / evaluation_bytes,
        "evaluation_exact_patch_accuracy": evaluation_metrics["exact_patch_correct"] / evaluation_patch_count,
        "evaluation_mean_correct_prefix_bytes": evaluation_metrics["correct_prefix_bytes_sum"] / evaluation_patch_count,
        "training_patch_count": len(training_sequence.patches),
        "validation_patch_count": len(evaluation_sequence.patches),
        "evaluation_patch_count": len(evaluation_sequence.patches),
        "training_mean_patch_bytes": training_sequence.mean_patch_bytes,
        "validation_mean_patch_bytes": evaluation_sequence.mean_patch_bytes,
        "evaluation_mean_patch_bytes": evaluation_sequence.mean_patch_bytes,
        "training_units_per_raw_byte": len(training_sequence.patches) / train_bytes,
        "validation_units_per_raw_byte": len(evaluation_sequence.patches) / evaluation_bytes,
        "evaluation_units_per_raw_byte": len(evaluation_sequence.patches) / evaluation_bytes,
        "training_effective_raw_context_bytes": args.patches_per_block * training_sequence.mean_patch_bytes,
        "validation_effective_raw_context_bytes": args.patches_per_block * evaluation_sequence.mean_patch_bytes,
        "global_input_dimension": model.encoder.output_dim,
        "model_parameter_bytes_fp32": parameter_bytes,
        "model_trainable_parameters": trainable_parameters,
        "model_buffer_bytes": buffer_bytes,
        "training_optimizer_steps": optimizer_step_count(
            training_sequence,
            patches_per_block=args.patches_per_block,
            batch_size=args.batch_size,
        ),
        "frontend_fit_cpu_seconds_current_run": fit_seconds,
        "lm_training_cpu_seconds_current_run": train_cpu,
        "validation_cpu_seconds_current_run": evaluation_cpu,
        "evaluation_cpu_seconds_current_run": evaluation_cpu,
        "fully_spiking_frontend": 1,
        "fully_spike_communicating_patch_encoder": 1,
        "fully_spike_communicating_global_to_local_decoder": 1,
        "fully_spiking_decoder_neural_path": spike_head,
        "spiking_byte_output_neurons_present": spike_head,
        "dense_final_byte_likelihood_readout_present": 1 - spike_head,
        "external_numerical_likelihood_evaluation_present": 1,
        "digital_tokenizer_present": 0,
        "digital_patch_codec_present": 0,
        "open_vocabulary_lossless_byte_coverage": 1,
        "one_global_backbone_step_per_patch": 1,
        "one_global_backbone_step_per_byte": 0,
        "canonical_validation_is_development_evidence": int(args.evaluation_source == "canonical_validation"),
        "test_split_touched": int(args.evaluation_source == "canonical_test"),
        "outer_validation_parameter_updates": 0,
        "evaluation_parameter_updates": 0,
        "validation_repeated_deterministically": 1,
        "evaluation_repeated_deterministically": 1,
        "configuration_selected_on_current_validation": 0,
        "physical_hardware_energy_measurement": 0,
        **reproducibility,
        **contract,
    }
    for key, value in train_stats.items():
        row[f"train_{key}"] = value
    for key, value in evaluation_stats.items():
        row[f"validation_{key}"] = value

    active_output_bytes = float(evaluation_stats.get("decoder_output_active_bytes", 0.0))
    if active_output_bytes > 0:
        row.update({
            "evaluation_output_no_spike_fraction": float(evaluation_stats.get("decoder_output_no_spike_bytes", 0.0)) / active_output_bytes,
            "evaluation_output_tied_winner_fraction": float(evaluation_stats.get("decoder_output_tied_winner_bytes", 0.0)) / active_output_bytes,
            "evaluation_first_spike_coverage": float(evaluation_stats.get("decoder_output_first_spike_covered_bytes", 0.0)) / active_output_bytes,
            "evaluation_first_spike_tied_fraction": float(evaluation_stats.get("decoder_output_first_spike_tied_bytes", 0.0)) / active_output_bytes,
            "evaluation_first_spike_unique_accuracy": float(evaluation_stats.get("decoder_output_first_spike_unique_correct_bytes", 0.0)) / active_output_bytes,
            "evaluation_mean_first_spike_latency_microsteps": float(evaluation_stats.get("decoder_output_first_spike_latency_sum", 0.0)) / max(float(evaluation_stats.get("decoder_output_first_spike_covered_bytes", 0.0)), 1.0),
            "evaluation_output_spikes_per_byte": float(evaluation_stats.get("decoder_output_spike_events", 0.0)) / active_output_bytes,
        })
    else:
        row.update({
            "evaluation_output_no_spike_fraction": 0.0,
            "evaluation_output_tied_winner_fraction": 0.0,
            "evaluation_first_spike_coverage": 0.0,
            "evaluation_first_spike_tied_fraction": 0.0,
            "evaluation_first_spike_unique_accuracy": 0.0,
            "evaluation_mean_first_spike_latency_microsteps": 0.0,
            "evaluation_output_spikes_per_byte": 0.0,
        })

    if args.evaluation_source == "canonical_test":
        row.update({
            "test_raw_bytes": evaluation_bytes,
            "test_source_sha256": evaluation_sha,
            "test_nll_nats": evaluation_nll,
            "test_bpb": row["evaluation_bpb"],
            "test_top1_accuracy": row["evaluation_top1_accuracy"],
            "test_top5_accuracy": row["evaluation_top5_accuracy"],
            "test_exact_patch_accuracy": row["evaluation_exact_patch_accuracy"],
        })

    row.update(scalable_patch_projection(row))
    atomic_row(args.output_csv, row)
    print(
        f"PASS {args.dataset}/{args.architecture}/{args.output_head}/s{args.seed} "
        f"decoder={args.decoder_hidden_dim}x{args.decoder_num_layers} "
        f"{evaluation_role}_BPB={row['bpb']:.6f} "
        f"top1={row['evaluation_top1_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
