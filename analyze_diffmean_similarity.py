import argparse
import csv
import json
from pathlib import Path

import torch


CONSENSUS_PROMPT = (
    "Does this activation represent pressure toward an incorrect consensus? "
    "Answer only True or False."
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("diffmeans_pt", type=Path)
    parser.add_argument(
        "--layer-results-json",
        type=Path,
        default=None,
        help="Optional layer_sweep JSON with oracle responses.",
    )
    parser.add_argument("--output-prefix", type=Path, default=None)
    return parser.parse_args()


def load_oracle_true_layers(path: Path | None) -> set[int]:
    if path is None:
        return set()
    data = json.loads(path.read_text())
    true_layers = set()
    for row in data["layers"]:
        response = row.get("oracle_responses", {}).get(CONSENSUS_PROMPT, "")
        if response.strip().lower().startswith("true"):
            true_layers.add(int(row["layer"]))
    return true_layers


def mean_offdiag(matrix: torch.Tensor, indices: list[int]) -> float | None:
    if len(indices) < 2:
        return None
    sub = matrix[indices][:, indices]
    mask = ~torch.eye(len(indices), dtype=torch.bool)
    return float(sub[mask].mean().item())


def write_csv(path: Path, layers: list[int], matrix: torch.Tensor) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer"] + layers)
        for layer, row in zip(layers, matrix.tolist()):
            writer.writerow([layer] + [f"{value:.8f}" for value in row])


def contiguous_true_runs(layers: list[int], true_layers: set[int]) -> list[list[int]]:
    runs = []
    current = []
    for layer in layers:
        if layer in true_layers:
            current.append(layer)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs


def main():
    args = parse_args()
    data = torch.load(args.diffmeans_pt, map_location="cpu")
    layers = [int(layer) for layer in data["layers"]]
    vectors = torch.stack([data["diffmeans"][layer].float() for layer in layers])
    norms = vectors.norm(dim=1)
    unit = vectors / norms.clamp_min(1e-12).unsqueeze(1)
    cosine = unit @ unit.T

    prefix = args.output_prefix
    if prefix is None:
        prefix = args.diffmeans_pt.with_suffix("")

    csv_file = prefix.with_name(prefix.name + "_cosine_similarity.csv")
    json_file = prefix.with_name(prefix.name + "_similarity_summary.json")
    md_file = prefix.with_name(prefix.name + "_similarity_analysis.md")

    write_csv(csv_file, layers, cosine)

    true_layers = load_oracle_true_layers(args.layer_results_json)
    true_layer_indices = [i for i, layer in enumerate(layers) if layer in true_layers]
    false_layer_indices = [i for i, layer in enumerate(layers) if layer not in true_layers]
    true_runs = contiguous_true_runs(layers, true_layers)

    adjacent = [
        {
            "from_layer": layers[i],
            "to_layer": layers[i + 1],
            "cosine": float(cosine[i, i + 1].item()),
            "norm_ratio_to_prev": float((norms[i + 1] / norms[i].clamp_min(1e-12)).item()),
        }
        for i in range(len(layers) - 1)
    ]
    top_adjacent_jumps = sorted(adjacent, key=lambda row: row["cosine"])[:10]

    # For each layer, measure how aligned it is to the average direction of all later oracle-true layers.
    later_true_alignment = []
    if true_layer_indices:
        true_centroid = unit[true_layer_indices].mean(dim=0)
        true_centroid = true_centroid / true_centroid.norm().clamp_min(1e-12)
        for i, layer in enumerate(layers):
            later_true_alignment.append(
                {
                    "layer": layer,
                    "cosine_to_true_centroid": float((unit[i] @ true_centroid).item()),
                    "oracle_true": layer in true_layers,
                }
            )

    summary = {
        "source": str(args.diffmeans_pt),
        "direction": data.get("direction"),
        "sample_count": data.get("sample_count"),
        "layers": layers,
        "norms": {str(layer): float(norm.item()) for layer, norm in zip(layers, norms)},
        "oracle_true_layers": sorted(true_layers),
        "oracle_true_runs": true_runs,
        "mean_true_true_cosine": mean_offdiag(cosine, true_layer_indices),
        "mean_false_false_cosine": mean_offdiag(cosine, false_layer_indices),
        "mean_false_true_cosine": (
            float(cosine[false_layer_indices][:, true_layer_indices].mean().item())
            if false_layer_indices and true_layer_indices
            else None
        ),
        "adjacent_layer_cosines": adjacent,
        "lowest_adjacent_cosines": top_adjacent_jumps,
        "cosine_to_true_centroid": later_true_alignment,
        "csv_file": str(csv_file),
    }
    json_file.write_text(json.dumps(summary, indent=2))

    lines = [
        "# Diffmean Layer Similarity Analysis",
        "",
        f"- source: `{args.diffmeans_pt}`",
        f"- direction: `{data.get('direction')}`",
        f"- sample_count: {data.get('sample_count')}",
        f"- cosine matrix: `{csv_file}`",
        "",
        "## Oracle True Layers",
        "",
        ", ".join(map(str, sorted(true_layers))) if true_layers else "No oracle labels supplied.",
        "",
        "## Block Similarity",
        "",
        f"- true/true mean off-diagonal cosine: {summary['mean_true_true_cosine']}",
        f"- false/false mean off-diagonal cosine: {summary['mean_false_false_cosine']}",
        f"- false/true mean cosine: {summary['mean_false_true_cosine']}",
        "",
        "## Lowest Adjacent Layer Cosines",
        "",
        "| from | to | cosine | norm ratio |",
        "|---:|---:|---:|---:|",
    ]
    for row in top_adjacent_jumps:
        lines.append(
            f"| {row['from_layer']} | {row['to_layer']} | "
            f"{row['cosine']:.6f} | {row['norm_ratio_to_prev']:.3f} |"
        )

    if later_true_alignment:
        lines.extend(
            [
                "",
                "## Cosine To Oracle-True Centroid",
                "",
                "| layer | cosine | oracle true |",
                "|---:|---:|:---:|",
            ]
        )
        for row in later_true_alignment:
            lines.append(
                f"| {row['layer']} | {row['cosine_to_true_centroid']:.6f} | "
                f"{row['oracle_true']} |"
            )

    md_file.write_text("\n".join(lines))
    print(f"Wrote {csv_file}")
    print(f"Wrote {json_file}")
    print(f"Wrote {md_file}")


if __name__ == "__main__":
    main()
