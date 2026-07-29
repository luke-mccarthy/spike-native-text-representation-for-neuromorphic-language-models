#!/usr/bin/env python3
"""Train one dense, direct-TTFS or nibble-TTFS patch-language-model row."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from scripts.train_frontend_language_model import DATASET_TIERS, read_strict, utf8_safe_prefix
from scripts.train_patch_language_model import (
    build_patch_sequence,
    optimizer_step_count,
    segmenter_counts,
)
from scripts.train_spike_count_decoder import (
    CANONICAL_ROOT,
    add_counts,
    atomic_row,
    configure_reproducibility,
    sha256,
)
from src.canonical_data_contract import (
    assert_train_test_contract,
    assert_train_validation_contract,
)
from src.energy.scalable_patch_projection import scalable_patch_projection
from src.models.spike_native_patch_lm import complete_patch_blocks, merge_additive_stats
from src.models.strict_ttfs_patch_decoder import (
    OUTPUT_HEADS,
    PATCH_ARCHITECTURES,
    StrictTTFSPatchLanguageModel,
)
from src.tokenizers.spike_native_patch_frontends import (
    JointSpikingPatchSegmenter,
    SpikePatchSNNConfig,
)

FORMAT = "strict_ttfs_output_decoder_row_v1"


def run_blocks(
    *,
    model: StrictTTFSPatchLanguageModel,
    sequence,
    patches_per_block: int,
    batch_size: int,
    optimizer: torch.optim.Optimizer | None,
    margin_weight: float,
    feedback_mode: str,
) -> tuple[float, int, Dict[str, float], Dict[str, float], float]:
    grouped: Dict[int, list[int]] = {}
    for start, length in complete_patch_blocks(len(sequence.patches), patches_per_block):
        grouped.setdefault(length, []).append(start)
    total_nll = 0.0
    total_bytes = 0
    total_aux = 0.0
    totals: Dict[str, float] = {}
    metrics = {
        "top1_correct_bytes": 0.0,
        "top5_correct_bytes": 0.0,
        "exact_patch_correct": 0.0,
        "patches_evaluated": 0.0,
        "correct_prefix_bytes_sum": 0.0,
    }
    training = optimizer is not None
    if training and feedback_mode != "teacher":
        raise ValueError("training must use teacher feedback")
    model.train(training)
    for length, starts_for_length in sorted(grouped.items(), reverse=True):
        for offset in range(0, len(starts_for_length), batch_size):
            starts = starts_for_length[offset : offset + batch_size]
            source, source_mask, target, target_mask = sequence.source_and_targets(
                starts, length, device=torch.device("cpu")
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
            scores, stats = model(
                source,
                source_mask,
                target,
                target_mask,
                feedback_mode=feedback_mode,
            )
            selected_scores = scores[target_mask]
            selected_targets = target[target_mask]
            loss_sum = F.cross_entropy(selected_scores, selected_targets, reduction="sum")
            auxiliary = model.output_auxiliary_loss(margin_weight=margin_weight)
            if training:
                objective = loss_sum / max(int(target_mask.sum().item()), 1) + auxiliary
                objective.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_nll += float(loss_sum.detach().item())
            total_aux += float(auxiliary.detach().item())
            active_bytes = int(target_mask.sum().item())
            total_bytes += active_bytes
            merge_additive_stats(totals, stats)
            with torch.no_grad():
                predictions = scores.argmax(dim=-1)
                top5 = scores.topk(k=5, dim=-1).indices
                metrics["top1_correct_bytes"] += float(((predictions == target) & target_mask).sum().item())
                metrics["top5_correct_bytes"] += float(((top5 == target.unsqueeze(-1)).any(dim=-1) & target_mask).sum().item())
                flat_pred = predictions.reshape(-1, predictions.shape[-1])
                flat_target = target.reshape(-1, target.shape[-1])
                flat_mask = target_mask.reshape(-1, target_mask.shape[-1])
                for pred_row, target_row, mask_row in zip(flat_pred, flat_target, flat_mask):
                    patch_length = int(mask_row.sum().item())
                    if patch_length == 0:
                        continue
                    correct = pred_row[:patch_length] == target_row[:patch_length]
                    metrics["patches_evaluated"] += 1.0
                    metrics["exact_patch_correct"] += float(bool(correct.all().item()))
                    prefix = 0
                    for value in correct.tolist():
                        if not value:
                            break
                        prefix += 1
                    metrics["correct_prefix_bytes_sum"] += float(prefix)
    if total_bytes != sequence.raw_bytes:
        raise AssertionError(f"covered {total_bytes} bytes, expected {sequence.raw_bytes}")
    return total_nll, total_bytes, totals, metrics, total_aux


def metric_bundle(prefix: str, nll: float, raw_bytes: int, stats: Dict[str, float], metrics: Dict[str, float]) -> Dict[str, object]:
    patches = max(int(metrics["patches_evaluated"]), 1)
    active = float(stats.get("decoder_output_active_bytes", 0.0))
    covered = float(stats.get("decoder_output_first_spike_covered_bytes", 0.0))
    output: Dict[str, object] = {
        f"{prefix}_nll_nats": nll,
        f"{prefix}_bpb": nll / raw_bytes / math.log(2.0),
        f"{prefix}_top1_accuracy": metrics["top1_correct_bytes"] / raw_bytes,
        f"{prefix}_top5_accuracy": metrics["top5_correct_bytes"] / raw_bytes,
        f"{prefix}_exact_patch_accuracy": metrics["exact_patch_correct"] / patches,
        f"{prefix}_mean_correct_prefix_bytes": metrics["correct_prefix_bytes_sum"] / patches,
    }
    if active > 0:
        output.update({
            f"{prefix}_output_no_spike_fraction": float(stats.get("decoder_output_no_spike_bytes", 0.0)) / active,
            f"{prefix}_output_tied_winner_fraction": float(stats.get("decoder_output_tied_winner_bytes", 0.0)) / active,
            f"{prefix}_output_unique_winner_fraction": float(stats.get("decoder_output_unique_winner_bytes", 0.0)) / active,
            f"{prefix}_first_spike_coverage": covered / active,
            f"{prefix}_first_spike_tied_fraction": float(stats.get("decoder_output_first_spike_tied_bytes", 0.0)) / active,
            f"{prefix}_first_spike_unique_accuracy": float(stats.get("decoder_output_first_spike_unique_correct_bytes", 0.0)) / active,
            f"{prefix}_mean_first_spike_latency_microsteps": float(stats.get("decoder_output_first_spike_latency_sum", 0.0)) / max(covered, 1.0),
            f"{prefix}_output_spikes_per_byte": float(stats.get("decoder_output_spike_events", 0.0)) / active,
            f"{prefix}_early_exit_spikes_per_byte": float(stats.get("decoder_output_early_exit_spike_events", 0.0)) / active,
        })
    else:
        for suffix in (
            "output_no_spike_fraction", "output_tied_winner_fraction",
            "output_unique_winner_fraction", "first_spike_coverage",
            "first_spike_tied_fraction", "first_spike_unique_accuracy",
            "mean_first_spike_latency_microsteps", "output_spikes_per_byte",
            "early_exit_spikes_per_byte",
        ):
            output[f"{prefix}_{suffix}"] = 0.0
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_TIERS["development"]))
    parser.add_argument("--data-tier", required=True, choices=("development", "full_train_validation"))
    parser.add_argument("--architecture", required=True, choices=sorted(PATCH_ARCHITECTURES))
    parser.add_argument("--output-head", required=True, choices=OUTPUT_HEADS)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--train-byte-budget", required=True, type=int)
    parser.add_argument("--patches-per-block", required=True, type=int)
    parser.add_argument("--target-boundary-rate-per-byte", required=True, type=float)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--hidden-dim", required=True, type=int)
    parser.add_argument("--num-layers", required=True, type=int)
    parser.add_argument("--decoder-hidden-dim", required=True, type=int)
    parser.add_argument("--decoder-num-layers", required=True, type=int)
    parser.add_argument("--output-microsteps", required=True, type=int)
    parser.add_argument("--output-beta", required=True, type=float)
    parser.add_argument("--output-threshold", required=True, type=float)
    parser.add_argument("--output-weight-scale", required=True, type=float)
    parser.add_argument("--output-clock-end", required=True, type=float)
    parser.add_argument("--output-latency-tau", required=True, type=float)
    parser.add_argument("--output-score-scale", required=True, type=float)
    parser.add_argument("--output-surrogate-temperature", required=True, type=float)
    parser.add_argument("--output-margin-weight", required=True, type=float)
    parser.add_argument("--lr", required=True, type=float)
    parser.add_argument("--test-evaluation-authorized", required=True, type=int, choices=(0, 1))
    parser.add_argument("--output-csv", required=True, type=Path)
    args = parser.parse_args()

    if args.test_evaluation_authorized and args.data_tier != "full_train_validation":
        raise AssertionError("test evaluation requires full_train_validation")
    reproducibility = configure_reproducibility(args.seed)
    paths = DATASET_TIERS[args.data_tier][args.dataset]
    training_path = CANONICAL_ROOT / paths["train"]
    validation_path = CANONICAL_ROOT / paths["validation"]
    validation_contract = assert_train_validation_contract(
        canonical_root=CANONICAL_ROOT,
        dataset=args.dataset,
        data_tier=args.data_tier,
        training_path=training_path,
        validation_path=validation_path,
    )
    training_source, _ = read_strict(training_path, role="training")
    validation_raw, _ = read_strict(validation_path, role="validation")
    training_raw, _ = utf8_safe_prefix(training_source, args.train_byte_budget)
    training_sha = sha256(training_raw)
    validation_sha = sha256(validation_raw)

    test_raw = None
    test_contract: Dict[str, object] = {}
    if args.test_evaluation_authorized:
        test_path = CANONICAL_ROOT / paths["test"]
        test_contract = assert_train_test_contract(
            canonical_root=CANONICAL_ROOT,
            dataset=args.dataset,
            data_tier=args.data_tier,
            training_path=training_path,
            test_path=test_path,
        )
        test_raw, _ = read_strict(test_path, role="test")

    spec = PATCH_ARCHITECTURES[args.architecture]
    segmenter = None
    fit_counts: Dict[str, float] = {}
    fit_seconds = 0.0
    segmenter_state = "not_applicable"
    if spec["patch_mode"] == "joint_snn_homeostatic":
        started = time.process_time()
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
        fit_seconds = time.process_time() - started
        fit_counts = segmenter_counts({
            key.removeprefix("homeostasis_fit_"): value
            for key, value in segmenter.encoder.homeostasis_report.items()
            if key.startswith("homeostasis_fit_")
        })
        segmenter_state = segmenter.encoder.state_digest()

    training_sequence, training_segment_counts = build_patch_sequence(
        training_raw, architecture=args.architecture, raw_sha256=training_sha, segmenter=segmenter
    )
    validation_sequence, validation_segment_counts = build_patch_sequence(
        validation_raw, architecture=args.architecture, raw_sha256=validation_sha, segmenter=segmenter
    )
    repeat_sequence, repeat_counts = build_patch_sequence(
        validation_raw, architecture=args.architecture, raw_sha256=validation_sha, segmenter=segmenter
    )
    if repeat_sequence.patches != validation_sequence.patches or repeat_counts != validation_segment_counts:
        raise AssertionError("validation patching is not deterministic")
    if segmenter is not None and segmenter.encoder.state_digest() != segmenter_state:
        raise AssertionError("validation changed frozen segmenter state")
    test_sequence = None
    test_segment_counts: Dict[str, float] = {}
    if test_raw is not None:
        test_sha = sha256(test_raw)
        test_sequence, test_segment_counts = build_patch_sequence(
            test_raw, architecture=args.architecture, raw_sha256=test_sha, segmenter=segmenter
        )
        if segmenter is not None and segmenter.encoder.state_digest() != segmenter_state:
            raise AssertionError("test changed frozen segmenter state")

    model = StrictTTFSPatchLanguageModel(
        max_patch_bytes=int(spec["max_patch_bytes"]),
        hierarchical_encoder=bool(spec["hierarchical"]),
        output_head=args.output_head,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        decoder_hidden_dim=args.decoder_hidden_dim,
        decoder_num_layers=args.decoder_num_layers,
        output_microsteps=args.output_microsteps,
        output_beta=args.output_beta,
        output_threshold=args.output_threshold,
        output_weight_scale=args.output_weight_scale,
        output_clock_end=args.output_clock_end,
        output_latency_tau=args.output_latency_tau,
        output_score_scale=args.output_score_scale,
        output_surrogate_temperature=args.output_surrogate_temperature,
        seed=args.seed,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    started = time.process_time()
    train_nll, train_bytes, train_stats, train_metrics, train_aux = run_blocks(
        model=model, sequence=training_sequence, patches_per_block=args.patches_per_block,
        batch_size=args.batch_size, optimizer=optimizer,
        margin_weight=args.output_margin_weight, feedback_mode="teacher",
    )
    train_cpu = time.process_time() - started
    with torch.no_grad():
        started = time.process_time()
        val_nll, val_bytes, val_stats, val_metrics, val_aux = run_blocks(
            model=model, sequence=validation_sequence, patches_per_block=args.patches_per_block,
            batch_size=args.batch_size, optimizer=None,
            margin_weight=args.output_margin_weight, feedback_mode="teacher",
        )
        val_cpu = time.process_time() - started
        free_val = None
        if args.output_head != "dense_reference":
            free_val = run_blocks(
                model=model, sequence=validation_sequence, patches_per_block=args.patches_per_block,
                batch_size=args.batch_size, optimizer=None,
                margin_weight=args.output_margin_weight, feedback_mode="free_running",
            )
        test_result = None
        free_test = None
        if test_sequence is not None:
            test_result = run_blocks(
                model=model, sequence=test_sequence, patches_per_block=args.patches_per_block,
                batch_size=args.batch_size, optimizer=None,
                margin_weight=args.output_margin_weight, feedback_mode="teacher",
            )
            if args.output_head != "dense_reference":
                free_test = run_blocks(
                    model=model, sequence=test_sequence, patches_per_block=args.patches_per_block,
                    batch_size=args.batch_size, optimizer=None,
                    margin_weight=args.output_margin_weight, feedback_mode="free_running",
                )

    add_counts(training_segment_counts, fit_counts)
    add_counts(train_stats, training_segment_counts)
    add_counts(val_stats, validation_segment_counts)
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    strict = int(args.output_head != "dense_reference")
    row: Dict[str, object] = {
        "format": FORMAT,
        "status": "PASS",
        "dataset": args.dataset,
        "data_tier": args.data_tier,
        "architecture": args.architecture,
        "method": "byte_patch_autoregressive_strict_ttfs_output_v6",
        "seed": args.seed,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "decoder_hidden_dim": args.decoder_hidden_dim,
        "decoder_num_layers": args.decoder_num_layers,
        "output_head": args.output_head,
        "output_microsteps": args.output_microsteps,
        "output_beta": args.output_beta,
        "output_threshold": args.output_threshold,
        "output_weight_scale": args.output_weight_scale,
        "output_clock_end": args.output_clock_end,
        "output_latency_tau": args.output_latency_tau,
        "output_score_scale": args.output_score_scale,
        "output_surrogate_temperature": args.output_surrogate_temperature,
        "output_margin_weight": args.output_margin_weight,
        "target_boundary_rate_per_byte": args.target_boundary_rate_per_byte,
        "patches_per_block": args.patches_per_block,
        "train_byte_budget": args.train_byte_budget,
        "training_raw_bytes": train_bytes,
        "validation_raw_bytes": val_bytes,
        "training_source_sha256": sha256(training_source),
        "training_selection_sha256": training_sha,
        "validation_source_sha256": validation_sha,
        "evaluation_source": "canonical_validation_and_authorized_test" if test_sequence is not None else "canonical_validation",
        "evaluation_split_role": "validation_plus_confirmatory_test" if test_sequence is not None else "validation",
        "training_auxiliary_loss_sum": train_aux,
        "validation_auxiliary_loss_sum": val_aux,
        "training_patch_count": len(training_sequence.patches),
        "validation_patch_count": len(validation_sequence.patches),
        "training_mean_patch_bytes": training_sequence.mean_patch_bytes,
        "validation_mean_patch_bytes": validation_sequence.mean_patch_bytes,
        "training_optimizer_steps": optimizer_step_count(training_sequence, patches_per_block=args.patches_per_block, batch_size=args.batch_size),
        "frontend_fit_cpu_seconds_current_run": fit_seconds,
        "lm_training_cpu_seconds_current_run": train_cpu,
        "validation_cpu_seconds_current_run": val_cpu,
        "model_parameter_bytes_fp32": parameter_bytes,
        "model_trainable_parameters": parameters,
        "model_buffer_bytes": buffer_bytes,
        "fully_spiking_frontend": 1,
        "fully_spike_communicating_patch_encoder": 1,
        "fully_spike_communicating_global_to_local_decoder": 1,
        "fully_spiking_decoder_neural_path": strict,
        "spiking_byte_output_neurons_present": strict,
        "dense_final_byte_likelihood_readout_present": 1 - strict,
        "surrogate_gradient_training_present": 1,
        "hard_binary_spikes_preserved_in_forward_pass": 1,
        "surrogate_used_only_for_backward_credit_assignment": 1,
        "training_executed_on_neuromorphic_hardware": 0,
        "external_numerical_likelihood_evaluation_present": 1,
        "digital_tokenizer_present": 0,
        "digital_patch_codec_present": 0,
        "open_vocabulary_lossless_byte_coverage": 1,
        "test_split_touched": int(test_sequence is not None),
        "test_evaluation_authorized": args.test_evaluation_authorized,
        "outer_validation_parameter_updates": 0,
        "test_parameter_updates": 0,
        **reproducibility,
        **{f"validation_contract_{key}": value for key, value in validation_contract.items()},
        **{f"test_contract_{key}": value for key, value in test_contract.items()},
        **metric_bundle("training", train_nll, train_bytes, train_stats, train_metrics),
        **metric_bundle("evaluation", val_nll, val_bytes, val_stats, val_metrics),
    }
    if free_val is not None:
        row.update(metric_bundle("free_running_evaluation", free_val[0], free_val[1], free_val[2], free_val[3]))
    if test_result is not None:
        row.update(metric_bundle("test", test_result[0], test_result[1], test_result[2], test_result[3]))
        row["test_raw_bytes"] = test_result[1]
        row["test_patch_count"] = len(test_sequence.patches)
        add_counts(test_result[2], test_segment_counts)
    if free_test is not None:
        row.update(metric_bundle("free_running_test", free_test[0], free_test[1], free_test[2], free_test[3]))
    for key, value in train_stats.items():
        row[f"train_{key}"] = value
    for key, value in val_stats.items():
        row[f"validation_{key}"] = value
    row["bpb"] = row["evaluation_bpb"]
    row.update(scalable_patch_projection(row))
    atomic_row(args.output_csv, row)
    print(
        f"PASS {args.dataset}/{args.architecture}/{args.output_head}/s{args.seed} "
        f"val_BPB={row['evaluation_bpb']:.6f} "
        f"ties={row['evaluation_output_tied_winner_fraction']:.4f} "
        f"coverage={row['evaluation_first_spike_coverage']:.4f}"
    )


if __name__ == "__main__":
    main()
