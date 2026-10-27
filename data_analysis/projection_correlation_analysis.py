"""Summarize correlations in the Qwen activation projection reports.

The inputs are the JSON reports emitted by project_experiment_activations.py.
This script only uses saved activations/projections; it does not rerun model
inference.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = [
    ROOT
    / "projection_results"
    / "qwen_experiment_mean_activation_projection_layer23_onto_layer_sweep_qwen_mmlu_mode_8_base_residual_assistant_start_all_wrong_minus_random_answers_n8_100_diffmeans.json",
    ROOT
    / "projection_results"
    / "qwen_experiment_mean_activation_projection_layer23_onto_layer_sweep_qwen_mmlu_mode_10_base_residual_assistant_start_all_wrong_minus_random_answers_n10_100_diffmeans.json",
]
OUT_DIR = ROOT / "paper" / "generated"


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    denom = math.sqrt(var_x * var_y)
    return cov / denom if denom else float("nan")


def numeric_mode(row: dict) -> int | None:
    mode = row.get("mode")
    if isinstance(mode, str) and mode.isdigit():
        return int(mode)
    if isinstance(mode, int):
        return mode
    return None


def load_numeric_rows(path: Path) -> tuple[str, list[dict]]:
    with path.open() as f:
        report = json.load(f)
    ref_n = report["reference_metadata"]["n"]
    ref_label = f"mmlu n={ref_n}"
    rows = [row for row in report["rows"] if numeric_mode(row) is not None]
    return ref_label, rows


def summarize_correlations(ref_label: str, rows: list[dict]) -> list[dict]:
    summaries = []
    for dataset in ("mmlu", "bbh"):
        for prompt_style in ("base", "explain"):
            group = [
                row
                for row in rows
                if row["dataset"] == dataset and row["prompt_style"] == prompt_style
            ]
            group = sorted(group, key=lambda row: numeric_mode(row) or -1)
            ns = [float(numeric_mode(row)) for row in group]
            projections = [float(row["projection_onto_unit_reference"]) for row in group]
            cosines = [float(row["cosine_with_reference"]) for row in group]
            summaries.append(
                {
                    "reference": ref_label,
                    "dataset": dataset,
                    "prompt_style": prompt_style,
                    "n_count": len(group),
                    "r_n_projection": pearson(ns, projections),
                    "r_n_cosine": pearson(ns, cosines),
                    "mean_projection": sum(projections) / len(projections),
                    "mean_cosine": sum(cosines) / len(cosines),
                }
            )
    return summaries


def summarize_prompt_deltas(ref_label: str, rows: list[dict]) -> list[dict]:
    deltas = []
    for dataset in ("mmlu", "bbh"):
        base_by_n = {
            numeric_mode(row): row
            for row in rows
            if row["dataset"] == dataset and row["prompt_style"] == "base"
        }
        explain_by_n = {
            numeric_mode(row): row
            for row in rows
            if row["dataset"] == dataset and row["prompt_style"] == "explain"
        }
        common_ns = sorted(set(base_by_n) & set(explain_by_n))
        projection_deltas = [
            float(base_by_n[n]["projection_onto_unit_reference"])
            - float(explain_by_n[n]["projection_onto_unit_reference"])
            for n in common_ns
        ]
        cosine_deltas = [
            float(base_by_n[n]["cosine_with_reference"])
            - float(explain_by_n[n]["cosine_with_reference"])
            for n in common_ns
        ]
        deltas.append(
            {
                "reference": ref_label,
                "dataset": dataset,
                "n_count": len(common_ns),
                "mean_delta_projection_base_minus_explain": sum(projection_deltas)
                / len(projection_deltas),
                "min_delta_projection_base_minus_explain": min(projection_deltas),
                "max_delta_projection_base_minus_explain": max(projection_deltas),
                "r_n_delta_projection": pearson([float(n) for n in common_ns], projection_deltas),
                "mean_delta_cosine_base_minus_explain": sum(cosine_deltas) / len(cosine_deltas),
                "r_n_delta_cosine": pearson([float(n) for n in common_ns], cosine_deltas),
            }
        )
    return deltas


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    correlations = []
    prompt_deltas = []
    for report_path in REPORTS:
        ref_label, rows = load_numeric_rows(report_path)
        correlations.extend(summarize_correlations(ref_label, rows))
        prompt_deltas.extend(summarize_prompt_deltas(ref_label, rows))

    write_csv(OUT_DIR / "qwen_projection_correlations.csv", correlations)
    write_csv(OUT_DIR / "qwen_projection_prompt_deltas.csv", prompt_deltas)
    with (OUT_DIR / "qwen_projection_correlation_summary.json").open("w") as f:
        json.dump(
            {"correlations": correlations, "prompt_deltas": prompt_deltas},
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
