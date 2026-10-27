from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from layer_sweep import collect_activations, load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank layer-sweep examples by projection onto saved diffmean directions."
    )
    parser.add_argument(
        "--tensor-file",
        type=Path,
        required=True,
        help="Saved *_diffmeans.pt file from layer_sweep.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Markdown report path. Defaults next to the tensor file.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional JSON output path. Defaults next to the tensor file.",
    )
    parser.add_argument(
        "--layers",
        default="top10",
        help='Comma-separated layers, or "top10" to use the largest diffmean norms.',
    )
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def short_text(text: str, max_chars: int = 420) -> str:
    text = " ".join(str(text).split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def markdown_escape(text: object) -> str:
    return str(text).replace("|", "/").replace("\n", " ")


def selected_layers(args: argparse.Namespace, data: dict) -> list[int]:
    layers = [int(layer) for layer in data["layers"]]
    if args.layers != "top10":
        requested = [int(item.strip()) for item in args.layers.split(",") if item.strip()]
        missing = sorted(set(requested) - set(layers))
        if missing:
            raise ValueError(f"Requested layers not in tensor file: {missing}")
        return requested
    norms = sorted(
        ((layer, float(data["diffmeans"][layer].norm().item())) for layer in layers),
        key=lambda item: item[1],
        reverse=True,
    )
    return [layer for layer, _ in norms[:10]]


def main() -> None:
    args = parse_args()
    data = torch.load(args.tensor_file, map_location="cpu", weights_only=False)
    layers = selected_layers(args, data)

    output = args.output
    if output is None:
        output = args.tensor_file.with_name(args.tensor_file.stem + "_top_projection.md")
    json_output = args.json_output
    if json_output is None:
        json_output = args.tensor_file.with_name(args.tensor_file.stem + "_top_projection.json")

    model_key = data["model"]
    model, tokenizer, device = load_model(model_key)

    vectors = {}
    for layer in layers:
        vector = data["diffmeans"][layer].detach().cpu().float()
        norm = vector.norm()
        if norm.item() == 0:
            raise ValueError(f"Layer {layer} has zero diffmean norm")
        vectors[layer] = vector / norm

    rows = []
    for example in tqdm(data["examples"], desc="Projecting examples"):
        condition_a_acts, _, _ = collect_activations(
            example["condition_a_prompt"],
            model,
            tokenizer,
            device,
            layers,
            data["activation_kind"],
            data["position"],
        )
        condition_b_acts, _, _ = collect_activations(
            example["condition_b_prompt"],
            model,
            tokenizer,
            device,
            layers,
            data["activation_kind"],
            data["position"],
        )
        projections = {}
        cosines = {}
        for layer in layers:
            diff = (condition_a_acts[layer] - condition_b_acts[layer]).detach().cpu().float()
            projections[layer] = float(torch.dot(diff, vectors[layer]).item())
            diff_norm = diff.norm()
            cosines[layer] = (
                float(torch.dot(diff, vectors[layer]).item() / diff_norm.item())
                if diff_norm.item()
                else 0.0
            )

        rows.append(
            {
                "record_idx": example["record_idx"],
                "subject": example.get("subject"),
                "ground_truth": example.get("ground_truth"),
                "wrong_answer": example.get("wrong_answer"),
                "condition_a_answers": example.get("condition_a_answers"),
                "condition_b_answers": example.get("condition_b_answers"),
                "question": short_text(
                    example["condition_a_prompt"].split("Question:", 1)[-1]
                    if "Question:" in example["condition_a_prompt"]
                    else example["condition_a_prompt"]
                ),
                "projection_mean": sum(projections.values()) / len(projections),
                "cosine_mean": sum(cosines.values()) / len(cosines),
                "projections": projections,
                "cosines": cosines,
            }
        )

    rows_by_projection = sorted(rows, key=lambda row: row["projection_mean"], reverse=True)
    rows_by_cosine = sorted(rows, key=lambda row: row["cosine_mean"], reverse=True)
    top_projection = rows_by_projection[: args.top_k]
    bottom_projection = list(reversed(rows_by_projection[-args.top_k :]))
    top_cosine = rows_by_cosine[: args.top_k]

    report = {
        "tensor_file": str(args.tensor_file),
        "model": data["model"],
        "model_name": data["model_name"],
        "dataset": data["dataset"],
        "n": data["n"],
        "sample_count": data["sample_count"],
        "position": data["position"],
        "activation_kind": data["activation_kind"],
        "direction": data["direction"],
        "layers": layers,
        "top_projection": top_projection,
        "bottom_projection": bottom_projection,
        "top_cosine": top_cosine,
        "all_rows": rows,
    }
    json_output.write_text(json.dumps(report, indent=2))

    lines = [
        "# Top Projection Examples",
        "",
        f"- tensor_file: `{args.tensor_file}`",
        f"- model: {data['model_name']}",
        f"- dataset: {data['dataset']}",
        f"- n: {data['n']}",
        f"- direction: {data['direction']}",
        f"- layers: {', '.join(map(str, layers))}",
        f"- projection score: mean scalar projection onto unit diffmean vectors",
        "",
    ]

    def add_table(title: str, table_rows: list[dict]) -> None:
        lines.extend(
            [
                f"## {title}",
                "",
                "| rank | idx | subject | score | cos | target | wrong consensus | random baseline | question |",
                "|---:|---:|---|---:|---:|---|---|---|---|",
            ]
        )
        for rank, row in enumerate(table_rows, 1):
            lines.append(
                "| {rank} | {idx} | {subject} | {score:.4f} | {cos:.4f} | {target} | {wrong} | {baseline} | {question} |".format(
                    rank=rank,
                    idx=row["record_idx"],
                    subject=markdown_escape(row["subject"]),
                    score=row["projection_mean"],
                    cos=row["cosine_mean"],
                    target=markdown_escape(row["ground_truth"]),
                    wrong=markdown_escape(row["condition_a_answers"]),
                    baseline=markdown_escape(row["condition_b_answers"]),
                    question=markdown_escape(row["question"]),
                )
            )
        lines.append("")

    add_table(f"Top {args.top_k} By Projection", top_projection)
    add_table(f"Bottom {args.top_k} By Projection", bottom_projection)
    add_table(f"Top {args.top_k} By Cosine", top_cosine)

    output.write_text("\n".join(lines))
    print(f"Wrote {output}")
    print(f"Wrote {json_output}")


if __name__ == "__main__":
    main()
