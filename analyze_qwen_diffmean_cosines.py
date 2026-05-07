from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


DEFAULT_SOURCE_DIRS = [Path("layer_sweep_results/qwen")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute pairwise cosine similarities between Qwen layer-sweep "
            "diffmean vectors."
        )
    )
    parser.add_argument(
        "--source-dirs",
        nargs="+",
        type=Path,
        default=DEFAULT_SOURCE_DIRS,
        help="Directories to scan for Qwen *_diffmeans.pt artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("diffmean_cosine_results/qwen_layer_sweep"),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=25,
        help="Number of strongest positive/negative/absolute non-self pairs to report.",
    )
    parser.add_argument(
        "--neighbor-k",
        type=int,
        default=5,
        help="Number of nearest other layers used for per-layer coherence summaries.",
    )
    return parser.parse_args()


def load_diffmean_artifacts(source_dirs: list[Path]) -> list[dict[str, Any]]:
    paths: list[Path] = []
    for source_dir in source_dirs:
        paths.extend(sorted(source_dir.glob("*_diffmeans.pt")))
    paths = sorted(set(paths))

    artifacts = []
    for path in paths:
        data = torch.load(path, map_location="cpu", weights_only=False)
        if data.get("model") != "qwen":
            continue
        diffmeans = data.get("diffmeans")
        if not isinstance(diffmeans, dict):
            continue

        layers = sorted(int(layer) for layer in diffmeans)
        vectors = torch.stack([diffmeans[layer].detach().cpu().float().flatten() for layer in layers])
        if vectors.ndim != 2:
            raise ValueError(f"{path} did not load as a 2D layer-vector tensor")

        artifacts.append(
            {
                "path": path,
                "label": artifact_label(path),
                "data": data,
                "layers": layers,
                "vectors": vectors,
                "norms": vectors.norm(dim=1),
            }
        )

    if not artifacts:
        searched = ", ".join(str(path) for path in source_dirs)
        raise FileNotFoundError(f"No Qwen *_diffmeans.pt files found in: {searched}")
    return artifacts


def artifact_label(path: Path) -> str:
    label = re.sub(r"_diffmeans$", "", path.stem)
    return label


def safe_cosine_matrix(vectors: torch.Tensor) -> torch.Tensor:
    norms = vectors.norm(dim=1)
    nonzero = norms > 0
    normalized = torch.zeros_like(vectors)
    normalized[nonzero] = F.normalize(vectors[nonzero], dim=1)
    return normalized @ normalized.T


def row_metadata(artifact: dict[str, Any]) -> dict[str, Any]:
    data = artifact["data"]
    keys = [
        "model",
        "model_name",
        "dataset",
        "mode",
        "explain",
        "qd",
        "n",
        "sample_count",
        "position",
        "activation_kind",
        "direction",
        "baseline",
    ]
    metadata = {key: data.get(key) for key in keys if key in data}
    metadata["source_file"] = str(artifact["path"])
    metadata["label"] = artifact["label"]
    metadata["layer_count"] = len(artifact["layers"])
    metadata["hidden_size"] = int(artifact["vectors"].shape[1])
    metadata["zero_norm_layers"] = [
        layer
        for layer, norm in zip(artifact["layers"], artifact["norms"].tolist(), strict=True)
        if norm == 0.0
    ]
    return metadata


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2))


def write_per_artifact_outputs(
    artifacts: list[dict[str, Any]],
    matrices_dir: Path,
) -> list[dict[str, Any]]:
    summaries = []
    matrices_dir.mkdir(parents=True, exist_ok=True)

    for artifact in artifacts:
        cosine = safe_cosine_matrix(artifact["vectors"])
        matrix_payload = {
            "metadata": row_metadata(artifact),
            "layers": artifact["layers"],
            "norms": [float(norm) for norm in artifact["norms"].tolist()],
            "cosine_matrix": cosine.tolist(),
        }
        json_path = matrices_dir / f"{artifact['label']}_layer_cosines.json"
        csv_path = matrices_dir / f"{artifact['label']}_layer_cosines.csv"
        write_json(json_path, matrix_payload)
        write_matrix_csv(csv_path, artifact["layers"], cosine)

        summaries.append(
            {
                **row_metadata(artifact),
                "diffmean_norm_min": float(artifact["norms"].min().item()),
                "diffmean_norm_max": float(artifact["norms"].max().item()),
                "diffmean_norm_mean": float(artifact["norms"].mean().item()),
                "matrix_json": str(json_path),
                "matrix_csv": str(csv_path),
            }
        )
    return summaries


def write_matrix_csv(path: Path, layers: list[int], matrix: torch.Tensor) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", *layers])
        for layer, row in zip(layers, matrix.tolist(), strict=True):
            writer.writerow([layer, *[f"{value:.8f}" for value in row]])


def layer_neighbor_rows(
    artifact: dict[str, Any],
    cosine: torch.Tensor,
    neighbor_k: int,
) -> list[dict[str, Any]]:
    rows = []
    layers = artifact["layers"]
    if neighbor_k < 1:
        raise ValueError("--neighbor-k must be at least 1")
    if neighbor_k >= len(layers):
        raise ValueError("--neighbor-k must be smaller than the number of layers")

    for i, layer in enumerate(layers):
        neighbors = sorted(
            (
                (float(cosine[i, j].item()), layers[j])
                for j in range(len(layers))
                if j != i
            ),
            reverse=True,
        )[:neighbor_k]
        rows.append(
            {
                **row_metadata(artifact),
                "layer": layer,
                "neighbor_k": neighbor_k,
                "mean_topk_cosine": sum(value for value, _ in neighbors) / neighbor_k,
                "top_neighbor_layers": [neighbor for _, neighbor in neighbors],
                "top_neighbor_cosines": [value for value, _ in neighbors],
            }
        )
    return rows


def write_layer_neighbor_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "label",
        "dataset",
        "mode",
        "explain",
        "n",
        "position",
        "activation_kind",
        "direction",
        "baseline",
        "layer",
        "neighbor_k",
        "mean_topk_cosine",
        "top_neighbor_layers",
        "top_neighbor_cosines",
        "source_file",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{key: row.get(key) for key in fieldnames},
                    "mean_topk_cosine": f"{row['mean_topk_cosine']:.8f}",
                    "top_neighbor_layers": " ".join(
                        str(layer) for layer in row["top_neighbor_layers"]
                    ),
                    "top_neighbor_cosines": " ".join(
                        f"{value:.8f}" for value in row["top_neighbor_cosines"]
                    ),
                }
            )


def global_bundle(artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    labels = []
    rows = []
    vectors = []
    for artifact in artifacts:
        for idx, layer in enumerate(artifact["layers"]):
            label = f"{artifact['label']}::layer_{layer}"
            labels.append(label)
            rows.append(
                {
                    "label": label,
                    "artifact": artifact["label"],
                    "source_file": str(artifact["path"]),
                    "dataset": artifact["data"].get("dataset"),
                    "mode": artifact["data"].get("mode"),
                    "n": artifact["data"].get("n"),
                    "position": artifact["data"].get("position"),
                    "activation_kind": artifact["data"].get("activation_kind"),
                    "layer": layer,
                    "norm": float(artifact["norms"][idx].item()),
                }
            )
            vectors.append(artifact["vectors"][idx])

    vector_tensor = torch.stack(vectors)
    cosine = safe_cosine_matrix(vector_tensor)
    return {
        "labels": labels,
        "rows": rows,
        "vectors": vector_tensor,
        "cosine": cosine,
    }


def pair_rows(
    rows: list[dict[str, Any]],
    cosine: torch.Tensor,
    *,
    include_same_artifact: bool = False,
) -> list[dict[str, Any]]:
    out = []
    n_rows = len(rows)
    for i in range(n_rows):
        if rows[i]["norm"] == 0.0:
            continue
        for j in range(i + 1, n_rows):
            if rows[j]["norm"] == 0.0:
                continue
            if not include_same_artifact and rows[i]["artifact"] == rows[j]["artifact"]:
                continue
            out.append(
                {
                    "left": rows[i]["label"],
                    "right": rows[j]["label"],
                    "left_artifact": rows[i]["artifact"],
                    "right_artifact": rows[j]["artifact"],
                    "left_layer": rows[i]["layer"],
                    "right_layer": rows[j]["layer"],
                    "left_dataset": rows[i]["dataset"],
                    "right_dataset": rows[j]["dataset"],
                    "left_mode": rows[i]["mode"],
                    "right_mode": rows[j]["mode"],
                    "cosine": float(cosine[i, j].item()),
                }
            )
    return out


def top_pairs(pairs: list[dict[str, Any]], top_k: int) -> dict[str, list[dict[str, Any]]]:
    return {
        "strongest_positive": sorted(pairs, key=lambda row: row["cosine"], reverse=True)[:top_k],
        "strongest_negative": sorted(pairs, key=lambda row: row["cosine"])[:top_k],
        "strongest_absolute": sorted(pairs, key=lambda row: abs(row["cosine"]), reverse=True)[:top_k],
    }


def write_pair_csv(path: Path, pairs: list[dict[str, Any]]) -> None:
    fieldnames = [
        "left",
        "right",
        "left_artifact",
        "right_artifact",
        "left_layer",
        "right_layer",
        "left_dataset",
        "right_dataset",
        "left_mode",
        "right_mode",
        "cosine",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in pairs:
            writer.writerow({**row, "cosine": f"{row['cosine']:.8f}"})


def write_activation_metadata(artifacts: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "raw_condition_activations_available": False,
        "note": (
            "The cached layer_sweep.py artifacts store diffmeans, examples, token "
            "positions, and periodic diff_sums checkpoints. They do not store raw "
            "per-example condition A/B activations."
        ),
        "sources": [],
    }
    checkpoint_bundle = {
        "raw_condition_activations_available": False,
        "note": (
            "These are aggregate activation-difference sums from layer_sweep.py "
            "checkpoints, not raw condition A/B activations."
        ),
        "checkpoints": {},
    }
    for artifact in artifacts:
        examples = artifact["data"].get("examples", [])
        checkpoint = artifact["path"].with_name(
            artifact["path"].name.replace("_diffmeans.pt", "_checkpoint.pt")
        )
        checkpoint_file = str(checkpoint) if checkpoint.exists() else None
        payload["sources"].append(
            {
                "source_file": str(artifact["path"]),
                "checkpoint_file": checkpoint_file,
                "sample_count": artifact["data"].get("sample_count"),
                "position": artifact["data"].get("position"),
                "activation_kind": artifact["data"].get("activation_kind"),
                "example_count": len(examples),
                "examples": examples,
            }
        )
        if checkpoint.exists():
            checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
            checkpoint_bundle["checkpoints"][artifact["label"]] = {
                "source_file": str(artifact["path"]),
                "checkpoint_file": str(checkpoint),
                "processed": checkpoint_data.get("processed"),
                "layers": checkpoint_data.get("layers"),
                "diff_sums": {
                    int(layer): tensor.detach().cpu().float()
                    for layer, tensor in checkpoint_data.get("diff_sums", {}).items()
                },
            }
    write_json(output_dir / "cached_activation_metadata.json", payload)
    torch.save(checkpoint_bundle, output_dir / "cached_activation_diffsum_checkpoints.pt")


def write_summary(
    path: Path,
    artifact_summaries: list[dict[str, Any]],
    global_rows: list[dict[str, Any]],
    report: dict[str, list[dict[str, Any]]],
    layer_neighbors: list[dict[str, Any]],
) -> None:
    lines = [
        "# Qwen Diffmean Layer Cosine Analysis",
        "",
        f"- source artifacts: {len(artifact_summaries)}",
        f"- layer vectors: {len(global_rows)}",
        f"- zero-norm vectors: {sum(1 for row in global_rows if row['norm'] == 0.0)}",
        "",
        "## Sources",
        "",
        "| artifact | dataset | mode | n | samples | position | norm_max | norm_mean |",
        "|---|---|---:|---:|---:|---|---:|---:|",
    ]
    for row in artifact_summaries:
        lines.append(
            "| {label} | {dataset} | {mode} | {n} | {samples} | {position} | {norm_max:.6f} | {norm_mean:.6f} |".format(
                label=row["label"],
                dataset=row.get("dataset"),
                mode=row.get("mode"),
                n=row.get("n"),
                samples=row.get("sample_count"),
                position=row.get("position"),
                norm_max=row["diffmean_norm_max"],
                norm_mean=row["diffmean_norm_mean"],
            )
        )

    lines.extend(
        [
            "",
            "## Strongest Per-Layer Neighborhoods",
            "",
            "| rank | artifact | layer | mean top-k cosine | neighbor layers |",
            "|---:|---|---:|---:|---|",
        ]
    )
    for rank, row in enumerate(
        sorted(layer_neighbors, key=lambda item: item["mean_topk_cosine"], reverse=True)[
            :25
        ],
        1,
    ):
        neighbor_layers = ", ".join(str(layer) for layer in row["top_neighbor_layers"])
        lines.append(
            f"| {rank} | {row['label']} | {row['layer']} | "
            f"{row['mean_topk_cosine']:.6f} | {neighbor_layers} |"
        )

    def add_pairs(title: str, rows: list[dict[str, Any]]) -> None:
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| rank | left | right | cosine |",
                "|---:|---|---|---:|",
            ]
        )
        for rank, row in enumerate(rows, 1):
            lines.append(
                f"| {rank} | {row['left']} | {row['right']} | {row['cosine']:.6f} |"
            )

    add_pairs("Strongest Positive Cross-Artifact Pairs", report["strongest_positive"])
    add_pairs("Strongest Negative Cross-Artifact Pairs", report["strongest_negative"])
    add_pairs("Strongest Absolute Cross-Artifact Pairs", report["strongest_absolute"])
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    diffmeans_dir = args.output_dir / "diffmeans"
    matrices_dir = args.output_dir / "layer_matrices"
    activations_dir = args.output_dir / "activations"
    diffmeans_dir.mkdir(parents=True, exist_ok=True)

    artifacts = load_diffmean_artifacts(args.source_dirs)
    artifact_summaries = write_per_artifact_outputs(artifacts, matrices_dir)
    layer_neighbors = []
    for artifact in artifacts:
        cosine = safe_cosine_matrix(artifact["vectors"])
        layer_neighbors.extend(layer_neighbor_rows(artifact, cosine, args.neighbor_k))

    bundle = global_bundle(artifacts)
    rows = bundle["rows"]
    pairs = pair_rows(rows, bundle["cosine"], include_same_artifact=False)
    report = top_pairs(pairs, args.top_k)

    torch.save(
        {
            "source_dirs": [str(path) for path in args.source_dirs],
            "artifact_summaries": artifact_summaries,
            "rows": rows,
            "labels": bundle["labels"],
            "diffmeans": bundle["vectors"],
            "cosine_matrix": bundle["cosine"],
        },
        diffmeans_dir / "qwen_layer_sweep_diffmeans_and_cosines.pt",
    )
    write_json(diffmeans_dir / "qwen_layer_sweep_diffmeans_metadata.json", artifact_summaries)
    write_json(args.output_dir / "layer_topk_neighbors.json", layer_neighbors)
    write_layer_neighbor_csv(args.output_dir / "layer_topk_neighbors.csv", layer_neighbors)
    write_json(args.output_dir / "top_cross_artifact_pairs.json", report)
    write_pair_csv(args.output_dir / "cross_artifact_pairs.csv", pairs)
    write_activation_metadata(artifacts, activations_dir)
    write_summary(args.output_dir / "analysis.md", artifact_summaries, rows, report, layer_neighbors)

    print(f"Loaded {len(artifacts)} Qwen diffmean artifacts")
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
