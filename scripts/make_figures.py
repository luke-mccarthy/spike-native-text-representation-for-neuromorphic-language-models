#!/usr/bin/env python3
"""Regenerate the paper's result figures directly from the archived raw CSVs.

This script reads only the committed, hash-verified raw row archives in
``results/front_end/all_rows.csv`` and ``results/temporal_decoders/all_rows.csv``
and renders the same comparisons reported in the paper (front-end quality,
temporal compression, quality-energy frontier, and strict TTFS output-head
trade-offs). No values are hand-entered: every plotted number is aggregated
directly from the raw rows at run time.

Unlike the original manuscript figures, method identity is always shown
through a shared colour/marker legend rather than per-point inline text
labels, so nothing overlaps regardless of how close two points sit together.

Usage:
    python scripts/make_figures.py --output-dir outputs/figures
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
FRONT_END_CSV = ROOT / "results/front_end/all_rows.csv"
TEMPORAL_CSV = ROOT / "results/temporal_decoders/all_rows.csv"

DATASETS = ["tinystories_5000", "text8", "wikitext2"]
DATASET_LABELS = {
    "tinystories_5000": "TinyStories",
    "text8": "Text8",
    "wikitext2": "WikiText-2",
}

# (method, interface_encoder-or-None, patch-position-in-byte_patch_autoregressive-or-None)
FRONT_END_METHODS = [
    ("bpe16000", None, None, "BPE-16k", "#1B3A6B", "o"),
    ("bpe8000", None, None, "BPE-8k", "#2F5597", "s"),
    ("unigram4000", None, None, "Unigram-4k", "#6E9BD1", "D"),
    ("legacy_homeostatic_0.075", None, None, "Boundary + digital codec", "#59A14F", "^"),
    ("byte", "fixed_1of256_byte", None, "Byte 1-of-256", "#F28E2B", "P"),
    ("byte", "fixed_2of32_byte", None, "Byte 2-of-32", "#E0A458", "X"),
    ("byte", "streaming_reservoir64_injective2", None, "Recurrent-LIF byte", "#C97A2B", "v"),
    ("byte_patch_autoregressive", None, 0, "Fixed patch-4", "#B07AA1", "*"),
    ("byte_patch_autoregressive", None, 1, "Fixed patch-8", "#8E4E9E", "h"),
    ("byte_patch_autoregressive", None, 2, "Adaptive patch-8", "#5A3E7A", "H"),
]

TTFS_HEADS = [
    ("dense_reference", "Dense readout", "#1B3A6B", "o"),
    ("ttfs_direct_256", "Direct TTFS", "#F28E2B", "s"),
    ("ttfs_nibble_2x16", "Nibble TTFS", "#8E4E9E", "^"),
]

ENERGY_COL = "validation_decisive_projected_end_to_end_energy_pj_per_raw_byte_central"
PATCH_ENERGY_COL = "validation_patch_projected_end_to_end_energy_pj_per_raw_byte_central"
TTFS_ENERGY_COL = "validation_patch_projected_end_to_end_energy_pj_per_raw_byte_central"


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean_sd(values: Sequence[float]) -> tuple[float, float]:
    n = len(values)
    m = sum(values) / n
    if n < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in values) / (n - 1)
    return m, var ** 0.5


def front_end_rows_by_method(rows: List[Dict[str, str]], dataset: str) -> Dict[str, List[Dict[str, str]]]:
    """Group front-end rows for one dataset into the 10 canonical methods."""
    out: Dict[str, List[Dict[str, str]]] = {}
    ds_rows = [r for r in rows if r["dataset"] == dataset]
    for method, sub, patch_index, label, _color, _marker in FRONT_END_METHODS:
        if method == "byte_patch_autoregressive":
            group = [r for r in ds_rows if r["method"] == method]
            # Grid order within the 9-row group is 3x fixed-4, 3x fixed-8, 3x adaptive-8.
            chunk = group[patch_index * 3 : patch_index * 3 + 3]
            out[label] = chunk
        else:
            group = [
                r for r in ds_rows
                if r["method"] == method and (sub is None or r.get("interface_encoder") == sub)
            ]
            out[label] = group
    return out


def bpb_values(group: Sequence[Mapping[str, str]]) -> List[float]:
    out = []
    for row in group:
        raw = row.get("test_bpb") or row.get("bits_per_raw_byte")
        out.append(float(raw))
    return out


def steps_per_byte(group: Sequence[Mapping[str, str]]) -> float:
    """Raw bytes represented per global recurrent step (paper's rho_step)."""
    col = "validation_bytes_per_unit"
    vals = [float(r[col]) for r in group if r.get(col) not in (None, "")]
    return sum(vals) / len(vals)


def energy_values(group: Sequence[Mapping[str, str]], scenario: str = "central") -> List[float]:
    patch_col = f"validation_patch_projected_end_to_end_energy_pj_per_raw_byte_{scenario}"
    dense_col = f"validation_decisive_projected_end_to_end_energy_pj_per_raw_byte_{scenario}"
    out = []
    for row in group:
        raw = row.get(patch_col) or row.get(dense_col)
        out.append(float(raw))
    return out


def figure1_quality(rows: List[Dict[str, str]], output_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(10.5, 6.5))
    n_methods = len(FRONT_END_METHODS)
    y_positions = list(range(n_methods))[::-1]
    dataset_offsets = {"tinystories_5000": 0.22, "text8": 0.0, "wikitext2": -0.22}
    dataset_colors = {"tinystories_5000": "#2F5597", "text8": "#E0A458", "wikitext2": "#8E4E9E"}
    dataset_markers = {"tinystories_5000": "o", "text8": "^", "wikitext2": "D"}

    for dataset in DATASETS:
        grouped = front_end_rows_by_method(rows, dataset)
        for index, (_m, _s, _p, label, _c, _mk) in enumerate(FRONT_END_METHODS):
            values = bpb_values(grouped[label])
            mean, sd = mean_sd(values)
            y = y_positions[index] + dataset_offsets[dataset]
            axis.errorbar(
                mean, y, xerr=sd, fmt=dataset_markers[dataset], color=dataset_colors[dataset],
                markersize=6, markeredgecolor="black", markeredgewidth=0.4, capsize=3, elinewidth=1,
            )

    axis.set_yticks(y_positions)
    axis.set_yticklabels([label for *_x, label, _c, _mk in FRONT_END_METHODS])
    axis.set_xlabel("Test bits per raw UTF-8 byte (mean ± SD across three seeds)")
    axis.set_title("Predictive quality of the front-end families")
    axis.grid(alpha=0.25, axis="x")
    legend_handles = [
        Line2D([0], [0], marker=dataset_markers[d], color="w", markerfacecolor=dataset_colors[d],
               markeredgecolor="black", markersize=8, label=DATASET_LABELS[d])
        for d in DATASETS
    ]
    axis.set_ylim(-0.6, n_methods - 0.4)
    axis.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "figure1_front_end_quality.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def method_legend_handles() -> List[Line2D]:
    return [
        Line2D([0], [0], marker=marker, color="w", markerfacecolor=color, markeredgecolor="black",
               markersize=8, label=label)
        for _m, _s, _p, label, color, marker in FRONT_END_METHODS
    ]


def figure2_rate_quality(rows: List[Dict[str, str]], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    for axis, dataset in zip(axes, DATASETS):
        grouped = front_end_rows_by_method(rows, dataset)
        for method, _s, _p, label, color, marker in FRONT_END_METHODS:
            group = grouped[label]
            bpb_mean, bpb_sd = mean_sd(bpb_values(group))
            rate = steps_per_byte(group)
            axis.errorbar(
                rate, bpb_mean, yerr=bpb_sd, fmt=marker, color=color, markersize=9,
                markeredgecolor="black", markeredgewidth=0.5, capsize=3,
            )
        axis.set_xlabel("Raw bytes represented per global recurrent step")
        axis.set_ylabel("Test BPB (lower is better)")
        axis.set_title(DATASET_LABELS[dataset])
        axis.grid(alpha=0.25)
    fig.suptitle("Temporal compression is obtained by accepting a quality-rate trade-off")
    fig.legend(
        handles=method_legend_handles(), loc="lower center", ncol=5, frameon=False,
        bbox_to_anchor=(0.5, -0.08), fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(output_dir / "figure2_rate_quality_tradeoff.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure3_energy_frontier(rows: List[Dict[str, str]], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, dataset in zip(axes, DATASETS):
        grouped = front_end_rows_by_method(rows, dataset)
        points = []
        for method, _s, _p, label, color, marker in FRONT_END_METHODS:
            group = grouped[label]
            bpb_mean, bpb_sd = mean_sd(bpb_values(group))
            energy_mean, _ = mean_sd(energy_values(group))
            axis.errorbar(
                energy_mean, bpb_mean, yerr=bpb_sd, fmt=marker, color=color, markersize=9,
                markeredgecolor="black", markeredgewidth=0.5, capsize=3, zorder=3,
            )
            points.append((energy_mean, bpb_mean))
        # Non-dominated (Pareto) frontier: lower energy AND lower BPB is better.
        points_sorted = sorted(points)
        frontier = []
        best_bpb = float("inf")
        for energy, bpb in points_sorted:
            if bpb < best_bpb:
                frontier.append((energy, bpb))
                best_bpb = bpb
        fx, fy = zip(*frontier)
        axis.plot(fx, fy, linestyle="--", color="grey", linewidth=1.2, zorder=1, label="Non-dominated frontier")
        axis.set_xscale("log")
        axis.set_xlabel("Projected central energy (pJ per raw byte, log scale)")
        axis.set_ylabel("Test bits per raw UTF-8 byte (lower is better)")
        axis.set_title(DATASET_LABELS[dataset])
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right", fontsize=8, frameon=False)
    # Note: this is the validation-split (single forward pass) energy projection,
    # matching the paper's Table 3 "Central pJ/raw byte" column. It is not the
    # training-phase projection (which would include the x3 surrogate-gradient
    # backward-pass factor described in Methods S3.8); see the audit notes.
    fig.suptitle("Front-end trade-off: predictive quality versus projected central energy")
    fig.legend(
        handles=method_legend_handles(), loc="lower center", ncol=5, frameon=False,
        bbox_to_anchor=(0.5, -0.08), fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(output_dir / "figure3_quality_energy_frontier.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure4_ttfs_operating_points(rows: List[Dict[str, str]], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, dataset in zip(axes, DATASETS):
        ds_rows = [r for r in rows if r["dataset"] == dataset]
        for head, label, color, marker in TTFS_HEADS:
            group = [r for r in ds_rows if r["output_head"] == head]
            bpb_mean, bpb_sd = mean_sd([float(r["test_bpb"]) for r in group])
            energy_mean, _ = mean_sd([float(r[TTFS_ENERGY_COL]) for r in group])
            axis.errorbar(
                energy_mean, bpb_mean, yerr=bpb_sd, fmt=marker, color=color, markersize=11,
                markeredgecolor="black", markeredgewidth=0.6, capsize=3,
            )
        axis.set_xlabel("Projected validation energy (pJ per raw byte)")
        axis.set_ylabel("Teacher-forced test BPB")
        axis.set_title(DATASET_LABELS[dataset])
        axis.grid(alpha=0.25)
    fig.suptitle("Output representation changes the quality-energy operating point")
    legend_handles = [
        Line2D([0], [0], marker=marker, color="w", markerfacecolor=color, markeredgecolor="black",
               markersize=9, label=label)
        for _h, label, color, marker in TTFS_HEADS
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(output_dir / "figure4_ttfs_operating_points.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure5_ttfs_paired_bars(rows: List[Dict[str, str]], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    bar_width = 0.35
    x = list(range(len(DATASETS)))
    colors = {"ttfs_direct_256": "#2F8F9D", "ttfs_nibble_2x16": "#D98E04"}
    labels = {"ttfs_direct_256": "Direct TTFS", "ttfs_nibble_2x16": "Nibble TTFS"}

    delta_bpb: Dict[str, List[float]] = defaultdict(list)
    energy_saved: Dict[str, List[float]] = defaultdict(list)
    for dataset in DATASETS:
        ds_rows = [r for r in rows if r["dataset"] == dataset]
        dense_group = [r for r in ds_rows if r["output_head"] == "dense_reference"]
        dense_bpb_mean, _ = mean_sd([float(r["test_bpb"]) for r in dense_group])

        def keyfn(row: Mapping[str, str]) -> tuple:
            return (row["architecture"], row["grid_decoder_capacity"], row.get("grid_seed") or row.get("seed"))

        dense_energy = {keyfn(r): float(r[TTFS_ENERGY_COL]) for r in dense_group}
        for head in ("ttfs_direct_256", "ttfs_nibble_2x16"):
            group = [r for r in ds_rows if r["output_head"] == head]
            head_bpb_mean, _ = mean_sd([float(r["test_bpb"]) for r in group])
            delta_bpb[head].append(head_bpb_mean - dense_bpb_mean)
            ratios = [
                1 - float(r[TTFS_ENERGY_COL]) / dense_energy[keyfn(r)]
                for r in group if keyfn(r) in dense_energy
            ]
            energy_saved[head].append(100 * sum(ratios) / len(ratios))

    axis = axes[0]
    for offset, head in zip((-bar_width / 2, bar_width / 2), ("ttfs_direct_256", "ttfs_nibble_2x16")):
        values = delta_bpb[head]
        bars = axis.bar([xi + offset for xi in x], values, width=bar_width,
                         color=colors[head], edgecolor="black", linewidth=0.5, label=labels[head])
        for bar, value in zip(bars, values):
            axis.annotate(f"+{value:.2f}", (bar.get_x() + bar.get_width() / 2, value),
                           textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8)
    axis.set_xticks(x)
    axis.set_xticklabels([DATASET_LABELS[d] for d in DATASETS])
    axis.set_ylabel("Paired increase in test BPB")
    axis.set_title("Quality cost")
    axis.grid(alpha=0.25, axis="y")
    axis.set_ylim(0, max(max(delta_bpb["ttfs_direct_256"]), max(delta_bpb["ttfs_nibble_2x16"])) * 1.15)

    axis = axes[1]
    for offset, head in zip((-bar_width / 2, bar_width / 2), ("ttfs_direct_256", "ttfs_nibble_2x16")):
        values = energy_saved[head]
        bars = axis.bar([xi + offset for xi in x], values, width=bar_width,
                         color=colors[head], edgecolor="black", linewidth=0.5, label=labels[head])
        for bar, value in zip(bars, values):
            axis.annotate(f"{value:.1f}%", (bar.get_x() + bar.get_width() / 2, value),
                           textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8)
    axis.set_xticks(x)
    axis.set_xticklabels([DATASET_LABELS[d] for d in DATASETS])
    axis.set_ylabel("Projected central energy saved (%)")
    axis.set_title("Energy benefit")
    axis.grid(alpha=0.25, axis="y")
    axis.set_ylim(0, max(max(energy_saved["ttfs_direct_256"]), max(energy_saved["ttfs_nibble_2x16"])) * 1.15)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colors[h], edgecolor="black", linewidth=0.5, label=labels[h])
        for h in ("ttfs_direct_256", "ttfs_nibble_2x16")
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Strict spike-time output heads versus the paired dense readout")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(output_dir / "figure5_ttfs_paired_tradeoffs.png", dpi=300)
    plt.close(fig)


SCENARIOS = ["low", "central", "high"]
SCENARIO_COLORS = {"low": "#8FBF8F", "central": "#2F5597", "high": "#C0504D"}
SENSITIVITY_METHODS = [
    ("byte", "fixed_1of256_byte", None, "Byte 1-of-256 vs BPE-16k"),
    ("byte_patch_autoregressive", None, 1, "Fixed patch-8 vs BPE-16k"),
]


def figure6_energy_scenario_sensitivity(rows: List[Dict[str, str]], output_dir: Path) -> None:
    """Supplementary figure: how much does the headline energy-reduction claim
    move if the low/high coefficient scenarios from Methods S3.8 are used
    instead of the central scenario shown in the paper's main figures? Both
    low/central/high are defined in the paper but only central is ever
    plotted; this makes the sensitivity visible."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=True)
    bar_width = 0.25
    n_groups = len(SENSITIVITY_METHODS)
    x = list(range(n_groups))

    for axis, dataset in zip(axes, DATASETS):
        grouped = front_end_rows_by_method(rows, dataset)
        bpe16_group = grouped["BPE-16k"]
        max_value = 0.0
        for scenario_index, scenario in enumerate(SCENARIOS):
            offset = (scenario_index - (len(SCENARIOS) - 1) / 2) * bar_width
            bpe16_energy = sum(energy_values(bpe16_group, scenario)) / len(bpe16_group)
            reductions = []
            for method, sub, patch_index, label in SENSITIVITY_METHODS:
                if method == "byte_patch_autoregressive":
                    method_group = grouped["Fixed patch-8"]
                else:
                    method_group = grouped[label.split(" vs ")[0]]
                method_energy = sum(energy_values(method_group, scenario)) / len(method_group)
                reduction = 100 * (1 - method_energy / bpe16_energy)
                reductions.append(reduction)
            max_value = max(max_value, max(reductions))
            bars = axis.bar(
                [xi + offset for xi in x], reductions, width=bar_width,
                color=SCENARIO_COLORS[scenario], edgecolor="black", linewidth=0.5,
            )
            for bar, value in zip(bars, reductions):
                axis.annotate(
                    f"{value:.0f}%", (bar.get_x() + bar.get_width() / 2, value),
                    textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7.5, rotation=90,
                )
        axis.set_xticks(x)
        axis.set_xticklabels([label for _m, _s, _p, label in SENSITIVITY_METHODS], rotation=12, ha="right")
        axis.set_title(DATASET_LABELS[dataset])
        axis.set_ylim(0, max_value * 1.35)
        axis.grid(alpha=0.25, axis="y")
    axes[0].set_ylabel("Projected end-to-end energy saved (%)")

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=SCENARIO_COLORS[s], edgecolor="black", linewidth=0.5,
                       label=f"{s.capitalize()} coefficients")
        for s in SCENARIOS
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        "Sensitivity of the energy-reduction claim to the low/central/high coefficient scenarios (Methods S3.8)"
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    fig.savefig(output_dir / "figure6_energy_scenario_sensitivity.png", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--front-end-csv", type=Path, default=FRONT_END_CSV)
    parser.add_argument("--temporal-decoder-csv", type=Path, default=TEMPORAL_CSV)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/figures")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    front_end_rows = read_csv(args.front_end_csv)
    temporal_rows = read_csv(args.temporal_decoder_csv)

    figure1_quality(front_end_rows, args.output_dir)
    figure2_rate_quality(front_end_rows, args.output_dir)
    figure3_energy_frontier(front_end_rows, args.output_dir)
    figure4_ttfs_operating_points(temporal_rows, args.output_dir)
    figure5_ttfs_paired_bars(temporal_rows, args.output_dir)
    figure6_energy_scenario_sensitivity(front_end_rows, args.output_dir)
    print(f"Wrote 6 figures to {args.output_dir}")


if __name__ == "__main__":
    main()
