#!/usr/bin/env python3
"""Generate the result figures of the paper from the committed raw CSVs.

Reads ``results/front_end/all_rows.csv`` and
``results/temporal_decoders/all_rows.csv`` and writes three vector PDFs,
sized to the single-column text width of the Elsevier CAS template:

    rate_quality_tradeoff.pdf     test BPB against raw bytes per global step
    quality_energy_frontier.pdf   test BPB against projected training energy
    ttfs_paired_tradeoffs.pdf     paired TTFS output-head effects by corpus

Every plotted value is aggregated from the raw rows when the script runs.

Usage:
    pip install matplotlib==3.11.2
    python scripts/make_figures.py --output-dir outputs/figures
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
FRONT_END_CSV = ROOT / "results/front_end/all_rows.csv"
TEMPORAL_CSV = ROOT / "results/temporal_decoders/all_rows.csv"

# Single-column text width of cas-sc (468.33 pt) in inches.
TEXT_WIDTH_IN = 468.33 / 72.27

DATASETS = ["tinystories_5000", "text8", "wikitext2"]
DATASET_LABELS = {
    "tinystories_5000": "TinyStories",
    "text8": "Text8",
    "wikitext2": "WikiText-2",
}

# (method, interface_encoder or None, architecture or None, label, colour, marker)
FRONT_END_METHODS = [
    ("bpe16000", None, None, "BPE-16k", "#1B3A6B", "o"),
    ("bpe8000", None, None, "BPE-8k", "#2F5597", "s"),
    ("unigram4000", None, None, "Unigram-4k", "#6E9BD1", "D"),
    ("legacy_homeostatic_0.075", None, None, "Boundary + digital codec", "#59A14F", "^"),
    ("byte", "fixed_1of256_byte", None, "Byte 1-of-256", "#F28E2B", "P"),
    ("byte", "fixed_2of32_byte", None, "Byte 2-of-32", "#E0A458", "X"),
    ("byte", "streaming_reservoir64_injective2", None, "Recurrent-LIF byte", "#C97A2B", "v"),
    ("byte_patch_autoregressive", None, "fixed_patch4_positional", "Fixed patch-4", "#B07AA1", "*"),
    ("byte_patch_autoregressive", None, "fixed_patch8_hierarchical_lif", "Fixed patch-8", "#8E4E9E", "h"),
    ("byte_patch_autoregressive", None, "joint_patch8_hierarchical_lif", "Adaptive patch-8", "#5A3E7A", "H"),
]

TTFS_HEADS = [
    ("ttfs_direct_256", "Direct TTFS", "#2F8F9D"),
    ("ttfs_nibble_2x16", "Nibble TTFS", "#D98E04"),
]
TTFS_ENERGY_COL = "validation_patch_projected_end_to_end_energy_pj_per_raw_byte_central"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7.5,
    "axes.titlesize": 7.5,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 0.8,
    "pdf.fonttype": 42,
})


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean_sd(values: Sequence[float]) -> Tuple[float, float]:
    n = len(values)
    m = sum(values) / n
    if n < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in values) / (n - 1)
    return m, var ** 0.5


def front_end_rows_by_method(
    rows: List[Dict[str, str]], dataset: str
) -> Dict[str, List[Dict[str, str]]]:
    out: Dict[str, List[Dict[str, str]]] = {}
    ds_rows = [r for r in rows if r["dataset"] == dataset]
    for method, encoder, architecture, label, _colour, _marker in FRONT_END_METHODS:
        out[label] = [
            r for r in ds_rows
            if r["method"] == method
            and (encoder is None or r.get("interface_encoder") == encoder)
            and (architecture is None or r.get("architecture") == architecture)
        ]
        if len(out[label]) != 3:
            raise ValueError(f"expected 3 seeds for {label} on {dataset}")
    return out


def bpb_values(group: Sequence[Mapping[str, str]]) -> List[float]:
    return [float(r.get("test_bpb") or r["bits_per_raw_byte"]) for r in group]


def bytes_per_step(group: Sequence[Mapping[str, str]]) -> float:
    values = [float(r["validation_bytes_per_unit"]) for r in group]
    return sum(values) / len(values)


def training_energy(group: Sequence[Mapping[str, str]], scenario: str = "central") -> float:
    patch_col = f"training_patch_projected_end_to_end_energy_pj_per_raw_byte_{scenario}"
    dense_col = f"training_decisive_projected_end_to_end_energy_pj_per_raw_byte_{scenario}"
    values = [float(r.get(patch_col) or r[dense_col]) for r in group]
    return sum(values) / len(values)


def style_axis(axis) -> None:
    axis.grid(alpha=0.25, linewidth=0.5)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)


def method_legend(fig, extra: Sequence[Line2D] = ()) -> None:
    handles = [
        Line2D([0], [0], marker=marker, color="w", markerfacecolor=colour,
               markeredgecolor="black", markeredgewidth=0.4, markersize=5.5, label=label)
        for _m, _e, _a, label, colour, marker in FRONT_END_METHODS
    ]
    handles.extend(extra)
    fig.legend(handles=handles, loc="lower center", ncol=6 if extra else 5, frameon=False,
               bbox_to_anchor=(0.5, 0.0), handletextpad=0.3, columnspacing=1.0)


def plot_front_end_panels(rows, output: Path, x_of, xlabel: str, log_x: bool, frontier: bool) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH_IN, 2.55))
    for axis, dataset in zip(axes, DATASETS):
        grouped = front_end_rows_by_method(rows, dataset)
        points = []
        for _m, _e, _a, label, colour, marker in FRONT_END_METHODS:
            group = grouped[label]
            bpb_mean, bpb_sd = mean_sd(bpb_values(group))
            x = x_of(group)
            axis.errorbar(
                x, bpb_mean, yerr=bpb_sd, fmt=marker, color=colour, markersize=5,
                markeredgecolor="black", markeredgewidth=0.4, capsize=1.5,
                elinewidth=0.6, zorder=3,
            )
            points.append((x, bpb_mean))
        if frontier:
            best, front = float("inf"), []
            for x, y in sorted(points):
                if y < best:
                    front.append((x, y))
                    best = y
            fx, fy = zip(*front)
            axis.plot(fx, fy, linestyle="--", color="grey", linewidth=0.8, zorder=1)
        if log_x:
            axis.set_xscale("log")
        axis.set_title(DATASET_LABELS[dataset], pad=3)
        axis.set_xlabel(xlabel)
        style_axis(axis)
    axes[0].set_ylabel("Test BPB (lower is better)")
    extra = (
        [Line2D([0], [0], linestyle="--", color="grey", linewidth=0.8, label="Non-dominated")]
        if frontier else []
    )
    method_legend(fig, extra)
    fig.tight_layout(rect=(0, 0.13, 1, 1), w_pad=0.8)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def figure_rate_quality(rows: List[Dict[str, str]], output_dir: Path) -> None:
    plot_front_end_panels(
        rows, output_dir / "rate_quality_tradeoff.pdf", bytes_per_step,
        "Raw bytes per global step", log_x=False, frontier=False,
    )


def figure_energy_frontier(rows: List[Dict[str, str]], output_dir: Path) -> None:
    plot_front_end_panels(
        rows, output_dir / "quality_energy_frontier.pdf", training_energy,
        "Training energy (pJ per raw byte)", log_x=True, frontier=True,
    )


def figure_ttfs_paired(rows: List[Dict[str, str]], output_dir: Path) -> None:
    def key(row: Mapping[str, str]) -> tuple:
        return (row["architecture"], row["grid_decoder_capacity"], row.get("grid_seed") or row.get("seed"))

    delta_bpb: Dict[str, List[float]] = {head: [] for head, _l, _c in TTFS_HEADS}
    energy_saved: Dict[str, List[float]] = {head: [] for head, _l, _c in TTFS_HEADS}
    for dataset in DATASETS:
        ds_rows = [r for r in rows if r["dataset"] == dataset]
        dense = [r for r in ds_rows if r["output_head"] == "dense_reference"]
        dense_bpb, _ = mean_sd([float(r["test_bpb"]) for r in dense])
        dense_energy = {key(r): float(r[TTFS_ENERGY_COL]) for r in dense}
        for head, _label, _colour in TTFS_HEADS:
            group = [r for r in ds_rows if r["output_head"] == head]
            head_bpb, _ = mean_sd([float(r["test_bpb"]) for r in group])
            delta_bpb[head].append(head_bpb - dense_bpb)
            ratios = [1 - float(r[TTFS_ENERGY_COL]) / dense_energy[key(r)] for r in group]
            energy_saved[head].append(100 * sum(ratios) / len(ratios))

    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 2.35))
    width = 0.36
    x = list(range(len(DATASETS)))
    panels = [
        (axes[0], delta_bpb, "Paired increase in test BPB", "+{:.2f}"),
        (axes[1], energy_saved, "Projected central energy saved (%)", "{:.1f}%"),
    ]
    for axis, values_by_head, ylabel, fmt in panels:
        top = 0.0
        for offset, (head, label, colour) in zip((-width / 2, width / 2), TTFS_HEADS):
            values = values_by_head[head]
            bars = axis.bar([xi + offset for xi in x], values, width=width, color=colour,
                            edgecolor="black", linewidth=0.4, label=label)
            for bar, value in zip(bars, values):
                axis.annotate(fmt.format(value), (bar.get_x() + bar.get_width() / 2, value),
                              textcoords="offset points", xytext=(0, 2), ha="center", fontsize=6.5)
            top = max(top, max(values))
        axis.set_xticks(x)
        axis.set_xticklabels([DATASET_LABELS[d] for d in DATASETS])
        axis.set_ylabel(ylabel)
        axis.set_ylim(0, top * 1.15)
        style_axis(axis)
        axis.grid(axis="x", visible=False)
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colour, edgecolor="black", linewidth=0.4, label=label)
        for _h, label, colour in TTFS_HEADS
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=2.0)
    fig.savefig(output_dir / "ttfs_paired_tradeoffs.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the paper's result figures.")
    parser.add_argument("--front-end-csv", type=Path, default=FRONT_END_CSV)
    parser.add_argument("--temporal-decoder-csv", type=Path, default=TEMPORAL_CSV)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/figures")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    front_end_rows = read_csv(args.front_end_csv)
    temporal_rows = read_csv(args.temporal_decoder_csv)

    figure_rate_quality(front_end_rows, args.output_dir)
    figure_energy_frontier(front_end_rows, args.output_dir)
    figure_ttfs_paired(temporal_rows, args.output_dir)
    print(f"Wrote 3 figures to {args.output_dir}")


if __name__ == "__main__":
    main()
