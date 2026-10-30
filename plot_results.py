#!/usr/bin/env python3
"""Plot DeepSpeed communication benchmark results from results/results.csv.

Usage: uv run --with matplotlib plot_results.py
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
PLOTS = ROOT / "plots"

# (label, system, nodes, bar color). Color follows the series in every figure.
SERIES = {
    "mi300a_1": ("MI300A, 1 node (4 APUs)", "mi300a", 1, "#2a78d6"),
    "mi300a_2": ("MI300A, 2 nodes (8 APUs)", "mi300a", 2, "#eb6834"),
    "a100_1": ("A100, 1 node (8 GPUs)", "a100", 1, "#1baf7a"),
    "a100_2": ("A100, 2 nodes (16 GPUs)", "a100", 2, "#eda100"),
}
OPS = ["all_reduce", "all_gather", "all_to_all", "broadcast", "pt2pt"]
METRICS = [("duration_ms", "Duration (ms)"), ("throughput_GBps", "Throughput (GB/s)"), ("busbw_GBps", "Bus bandwidth (GB/s)")]
def load():
    """Return {series: {op: row}} for the runs marked plotted in the CSV."""
    with open(ROOT / "results" / "results.csv", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["plotted"] == "True"]
    data = {}
    for key, (_, system, nodes, _) in SERIES.items():
        selected = [r for r in rows if r["system"] == system and int(r["nodes"]) == nodes]
        data[key] = {r["op"]: {m: float(r[m]) for m, _ in METRICS} for r in selected}
        assert sorted(data[key]) == sorted(OPS), f"incomplete results for {key}"
    return data


def plot(data, keys, title, filename):
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": "#0b0b0b", "axes.labelcolor": "#52514e"})
    x = np.arange(len(OPS))
    width = 0.8 / len(keys)
    fig, axes = plt.subplots(1, len(METRICS), figsize=(17, 5.4), facecolor="#fcfcfb")
    for ax, (metric, ylabel) in zip(axes, METRICS):
        for i, key in enumerate(keys):
            values = [data[key][op][metric] for op in OPS]
            offset = (i - (len(keys) - 1) / 2) * width
            bars = ax.bar(x + offset, values, width * 0.94, color=SERIES[key][3], label=SERIES[key][0])
            ax.bar_label(bars, fmt="%.0f", padding=3, fontsize=9, color="#0b0b0b")
        ax.set_facecolor("#fcfcfb")
        ax.set_title(ylabel, fontsize=13, fontweight="bold", loc="left")
        ax.set_xticks(x, OPS, fontsize=10.5)
        ax.tick_params(length=0, colors="#52514e")
        ax.grid(axis="y", color="#e7e6e2", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#b5b4ae")
        ax.margins(y=0.12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=len(keys), frameon=False, fontsize=11, bbox_to_anchor=(0.99, 0.97))
    fig.suptitle(title, x=0.01, y=0.97, ha="left", fontsize=16, fontweight="bold")
    fig.text(0.01, 0.01, "Message sizes differ between series (see results/results.csv). "
             "pt2pt runs rank 0 to rank 1, which share a node even in 2-node jobs.", fontsize=9.5, color="#52514e")
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    fig.savefig(PLOTS / filename, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    data = load()
    PLOTS.mkdir(exist_ok=True)
    plot(data, ["mi300a_1", "mi300a_2"], "DeepSpeed communication: MI300A, 1 vs 2 nodes", "mi300a_1_vs_2_nodes.png")
    plot(data, ["a100_1", "a100_2"], "DeepSpeed communication: A100, 1 vs 2 nodes", "a100_1_vs_2_nodes.png")
    plot(data, ["mi300a_1", "a100_1"], "DeepSpeed communication: MI300A vs A100, 1 node", "mi300a_vs_a100_1_node.png")
    plot(data, ["mi300a_2", "a100_2"], "DeepSpeed communication: MI300A vs A100, 2 nodes", "mi300a_vs_a100_2_nodes.png")


if __name__ == "__main__":
    main()
