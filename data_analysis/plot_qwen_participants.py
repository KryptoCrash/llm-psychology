#!/usr/bin/env python3
"""Create paper figures for Qwen participant-count experiments."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "paper" / "generated"
FIGURES = ROOT / "paper" / "figures"


DATASET_LABELS = {"mmlu": "MMLU", "bbh": "BBH"}
PROMPT_LABELS = {"base": "Base", "explain": "Explanation"}
COLORS = {"base": "#1f77b4", "explain": "#d62728", "qd": "#2ca02c", "da": "#9467bd"}

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    }
)


def load_conditions() -> dict[tuple[str, str, str], dict]:
    rows = {}
    with (GENERATED / "qwen_condition_bootstrap.csv").open() as f:
        for row in csv.DictReader(f):
            rows[(row["dataset"], row["mode"], row["prompt"])] = row
    return rows


def f(row: dict, key: str) -> float:
    value = row.get(key, "")
    return float(value) if value != "" else float("nan")


def metric_series(rows: dict, dataset: str, prompt: str, metric: str, modes: list[str]) -> tuple[list[int], list[float], list[float], list[float]]:
    xs, ys, lows, highs = [], [], [], []
    for mode in modes:
        row = rows[(dataset, mode, prompt)]
        xs.append(int(mode))
        ys.append(f(row, metric))
        lows.append(f(row, f"{metric}_ci_low"))
        highs.append(f(row, f"{metric}_ci_high"))
    return xs, ys, lows, highs


def style_axis(ax, ylabel: str | None = None) -> None:
    ax.set_xlim(1, 10)
    ax.set_xticks(range(1, 11))
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis="y", color="#d9d9d9", linewidth=0.7)
    ax.grid(False, axis="x")
    if ylabel:
        ax.set_ylabel(ylabel)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_metric_grid(rows: dict, metric: str, ylabel: str, filename: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    for ax, dataset in zip(axes, ("mmlu", "bbh")):
        for prompt in ("base", "explain"):
            modes = [str(i) for i in range(1, 11)] if metric != "conformity" else [str(i) for i in range(2, 11)]
            xs, ys, lows, highs = metric_series(rows, dataset, prompt, metric, modes)
            color = COLORS[prompt]
            ax.plot(xs, ys, marker="o", markersize=3.5, linewidth=1.8, color=color, label=PROMPT_LABELS[prompt])
            ax.fill_between(xs, lows, highs, color=color, alpha=0.14, linewidth=0)

        ax.set_title(DATASET_LABELS[dataset], fontsize=10, pad=5)
        ax.set_xlabel("Participants")
        style_axis(ax, ylabel if ax is axes[0] else None)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"{filename}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_mitigation_markers(rows: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    x_positions = {"MA10": 0, "QD": 1, "DA": 2}
    offset = {"base": -0.08, "explain": 0.08}

    for ax, dataset in zip(axes, ("mmlu", "bbh")):
        for prompt in ("base", "explain"):
            for label, mode in (("MA10", "10"), ("QD", "qd"), ("DA", "da")):
                row = rows[(dataset, mode, prompt)]
                x = x_positions[label] + offset[prompt]
                y = f(row, "conformity")
                low = f(row, "conformity_ci_low")
                high = f(row, "conformity_ci_high")
                ax.errorbar(
                    [x],
                    [y],
                    yerr=[[y - low], [high - y]],
                    fmt="o",
                    markersize=5,
                    capsize=2.5,
                    color=COLORS[prompt],
                    label=PROMPT_LABELS[prompt] if label == "MA10" else None,
                )
        ax.set_xticks(list(x_positions.values()), list(x_positions.keys()))
        ax.set_ylim(0, 0.55)
        ax.set_title(DATASET_LABELS[dataset], fontsize=10, pad=5)
        ax.grid(True, axis="y", color="#d9d9d9", linewidth=0.7)
        ax.grid(False, axis="x")
        if ax is axes[0]:
            ax.set_ylabel("Conformity rate")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"qwen_mitigation_conformity.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows = load_conditions()
    plot_metric_grid(rows, "conformity", "Conformity rate", "qwen_conformity_by_participants")
    plot_metric_grid(rows, "strict_accuracy", "Strict accuracy", "qwen_strict_accuracy_by_participants")
    plot_metric_grid(rows, "parse_rate", "Parse rate", "qwen_parse_rate_by_participants")
    plot_mitigation_markers(rows)
    print(f"Wrote figures to {FIGURES}")


if __name__ == "__main__":
    main()
