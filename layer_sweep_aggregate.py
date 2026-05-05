from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

import oracle_utils
from layer_sweep import (
    CONSENSUS_PROMPT,
    MODEL_IDS,
    collect_activations,
    experiment_files,
    load_model,
    load_saved_experiment,
    maybe_sample_experiment_files,
    parse_experiment_file_name,
    query_oracle_vector,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate layer-sweep diffmeans over saved experiment prompts."
    )
    parser.add_argument("--models", default="qwen", help="Comma-separated model keys.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("experiment_layer_sweep_results_last_prompt"),
        help="Directory containing saved experiment result JSONs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("layer_sweep_aggregate_results"),
    )
    parser.add_argument(
        "--dataset",
        choices=["all", "mmlu", "bbh"],
        default="all",
        help="Optional dataset filter. Default aggregates each dataset separately.",
    )
    parser.add_argument("--layers", default="all", help='Comma-separated layer ids, or "all".')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional per-experiment record limit.",
    )
    parser.add_argument(
        "--sample-experiments",
        type=int,
        default=None,
        help="Optional number of experiment files to sample per model/dataset for smoke tests.",
    )
    parser.add_argument(
        "--activation-kind",
        choices=["residual", "delta"],
        default="residual",
        help="Use layer output residual stream, or per-layer residual update x_out - x_in.",
    )
    parser.add_argument(
        "--position",
        choices=["last_prompt_token", "assistant_start"],
        default="last_prompt_token",
        help="Activation position to compare when recomputing activations from saved prompts.",
    )
    parser.add_argument("--skip-oracle", action="store_true", default=True)
    parser.add_argument("--run-oracle", dest="skip_oracle", action="store_false")
    parser.add_argument(
        "--oracle-lora-path",
        default="adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()

def select_layers(model_name: str, layers_arg: str) -> list[int]:
    layer_count = oracle_utils.LAYER_COUNTS[model_name]
    if layers_arg == "all":
        return list(range(layer_count))
    layers = [int(layer.strip()) for layer in layers_arg.split(",") if layer.strip()]
    for layer in layers:
        if layer < 0 or layer >= layer_count:
            raise ValueError(f"Layer {layer} is out of range for {model_name} with {layer_count} layers")
    return layers


def aggregate_experiment_file(
    path: Path,
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
    layers: list[int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    info = parse_experiment_file_name(path)
    data = load_saved_experiment(path)
    examples = data["examples"]
    if args.limit is not None:
        examples = examples[: args.limit]

    if data.get("position") != args.position:
        raise ValueError(
            f"{path.name} was generated with position={data.get('position')}, "
            f"but the aggregate is running with --position={args.position}"
        )
    if data.get("activation_kind") and data.get("activation_kind") != args.activation_kind:
        raise ValueError(
            f"{path.name} was generated with activation_kind={data.get('activation_kind')}, "
            f"but the aggregate is running with --activation-kind={args.activation_kind}"
        )

    diff_sums = {layer: None for layer in layers}
    example_summaries = []

    for record_idx, example in enumerate(tqdm(examples, desc=path.name, leave=False)):
        condition_prompt = example["condition_prompt"]
        baseline_prompt = example["baseline_prompt"]
        condition_acts, condition_idx, condition_token = collect_activations(
            condition_prompt,
            model,
            tokenizer,
            device,
            layers,
            args.activation_kind,
            args.position,
        )
        baseline_acts, baseline_idx, baseline_token = collect_activations(
            baseline_prompt,
            model,
            tokenizer,
            device,
            layers,
            args.activation_kind,
            args.position,
        )

        for layer in layers:
            diff = condition_acts[layer] - baseline_acts[layer]
            diff_sums[layer] = diff if diff_sums[layer] is None else diff_sums[layer] + diff

        example_summaries.append(
            {
                "record_idx": record_idx,
                "subject": example.get("subject"),
                "question": example.get("question"),
                "ground_truth": example.get("ground_truth"),
                "main_wrong": example.get("main_wrong"),
                "da_position": example.get("da_position"),
                "da_answer": example.get("da_answer"),
                "model_answer": example.get("model_answer"),
                "is_correct": example.get("is_correct"),
                "conforms": example.get("conforms"),
                "random_baseline_answers": example.get("random_baseline_answers"),
                "condition_position_index": condition_idx,
                "baseline_position_index": baseline_idx,
                "condition_position_token": condition_token,
                "baseline_position_token": baseline_token,
                "condition_prompt": condition_prompt,
                "baseline_prompt": baseline_prompt,
            }
        )

    file_sample_count = len(example_summaries)
    file_diffmeans = {layer: diff_sums[layer] / file_sample_count for layer in layers}
    file_summary = {
        "experiment_file": str(path),
        "model": info["model"],
        "model_name": MODEL_IDS[info["model"]],
        "dataset": info["dataset"],
        "mode": info["mode"],
        "n": data.get("n"),
        "prompt_style": info["prompt_style"],
        "sample_count": file_sample_count,
        "position": args.position,
        "activation_kind": args.activation_kind,
        "baseline": data.get("baseline"),
        "diffmean_norms": {layer: float(file_diffmeans[layer].norm().item()) for layer in layers},
    }
    return {
        "summary": file_summary,
        "diffmeans": file_diffmeans,
        "examples": example_summaries,
    }


def write_analysis(results: dict[str, Any], output_file: Path) -> None:
    rows = results["layers"]
    top_norms = sorted(rows, key=lambda row: row["diffmean_norm"], reverse=True)[:10]
    true_layers = [
        row["layer"]
        for row in rows
        if row.get("oracle_responses", {}).get(CONSENSUS_PROMPT, "").lower().startswith("true")
    ]
    source_rows = results["source_files"]
    lines = [
        "# Aggregated Layer Sweep Analysis",
        "",
        f"- model: {results['model']}",
        f"- model_name: {results['model_name']}",
        f"- dataset: {results['dataset']}",
        f"- source_experiment_count: {results['source_experiment_count']}",
        f"- sample_count: {results['sample_count']}",
        f"- position: {results['position']}",
        f"- activation_kind: {results['activation_kind']}",
        "",
        "## Largest Diffmean Norms",
        "",
        "| layer | norm |",
        "|---:|---:|",
    ]
    lines.extend(f"| {row['layer']} | {row['diffmean_norm']:.6f} |" for row in top_norms)
    lines.extend(["", "## Oracle True Layers", ""])
    lines.append(", ".join(map(str, true_layers)) if true_layers else "None")
    lines.extend(["", "## Source Experiments", "", "| file | dataset | mode | prompt_style | n | samples |", "|---|---|---|---|---:|---:|"])
    for row in source_rows:
        lines.append(
            f"| {Path(row['experiment_file']).name} | {row['dataset']} | {row['mode']} | "
            f"{row['prompt_style']} | {row['n']} | {row['sample_count']} |"
        )
    output_file.write_text("\n".join(lines))


def aggregate_model(model_key: str, args: argparse.Namespace, dataset_filter: str) -> dict[str, Any]:
    model_name = MODEL_IDS[model_key]
    model_dir = args.output_dir / model_key
    model_dir.mkdir(parents=True, exist_ok=True)
    layers = select_layers(model_name, args.layers)

    search_dir = args.input_dir / model_key if (args.input_dir / model_key).exists() else args.input_dir
    files = experiment_files(search_dir, model_key)
    selected = []
    for path in files:
        info = parse_experiment_file_name(path)
        if dataset_filter != "all" and info["dataset"] != dataset_filter:
            continue
        selected.append(path)
    selected = maybe_sample_experiment_files(selected, model_key, args.sample_experiments, args.seed)

    if not selected:
        raise ValueError(f"No experiment files matched model={model_key} dataset={dataset_filter}")

    if args.dry_run:
        preview = []
        for path in selected[: min(3, len(selected))]:
            data = load_saved_experiment(path)
            preview.append(
                {
                    "file": path.name,
                    "dataset": data.get("dataset"),
                    "mode": data.get("mode"),
                    "prompt_style": data.get("prompt_style"),
                    "position": data.get("position"),
                    "activation_kind": data.get("activation_kind"),
                    "sample_count": len(data.get("examples", [])),
                }
            )
        return {
            "model": model_key,
            "model_name": model_name,
            "selected_experiments": len(selected),
            "preview": preview,
        }

    print(f"Loading {model_key}: {model_name}")
    model, tokenizer, device = load_model(model_key)
    if not args.skip_oracle:
        print(f"Loading oracle LoRA: {args.oracle_lora_path}")
        model.load_adapter(args.oracle_lora_path, adapter_name="oracle", is_trainable=False)

    aggregate_sums = {layer: None for layer in layers}
    source_files = []
    all_examples = []
    total_sources = 0

    for path in selected:
        per_file = aggregate_experiment_file(path, model, tokenizer, device, layers, args)
        total_sources += 1
        source_files.append(per_file["summary"])
        all_examples.extend(per_file["examples"])
        for layer in layers:
            diff = per_file["diffmeans"][layer]
            aggregate_sums[layer] = diff if aggregate_sums[layer] is None else aggregate_sums[layer] + diff

    diffmeans = {layer: aggregate_sums[layer] / total_sources for layer in layers}
    direction = "aggregate_over_saved_experiments"
    prompt_styles = {row["prompt_style"] for row in source_files}
    explain_slug = "mixed" if len(prompt_styles) > 1 else next(iter(prompt_styles))
    dataset_slug = dataset_filter
    run_slug = (
        f"layer_sweep_aggregate_{model_key}_{dataset_slug}_{args.position}_"
        f"{args.activation_kind}_{explain_slug}_n{len(source_files)}"
    )
    tensor_file = model_dir / f"{run_slug}_diffmeans.pt"
    json_file = model_dir / f"{run_slug}.json"
    analysis_file = model_dir / f"{run_slug}_analysis.md"

    torch.save(
        {
            "model": model_key,
            "model_name": model_name,
            "dataset": dataset_slug,
            "source_experiment_count": total_sources,
            "sample_count": len(all_examples),
            "layers": layers,
            "position": args.position,
            "activation_kind": args.activation_kind,
            "direction": direction,
            "diffmeans": diffmeans,
            "source_files": source_files,
        },
        tensor_file,
    )

    layer_results = []
    oracle_prompts = [
        "Describe this text.",
        "What belief or feature does this activation represent?",
        CONSENSUS_PROMPT,
    ]
    for layer in tqdm(layers, desc=f"Oracle sweep {model_key}", disable=args.skip_oracle):
        diffmean = diffmeans[layer]
        row = {
            "layer": layer,
            "diffmean_norm": float(diffmean.norm().item()),
            "oracle_responses": {},
        }
        if not args.skip_oracle:
            row["oracle_responses"] = query_oracle_vector(
                diffmean, layer, oracle_prompts, model, tokenizer, device
            )
        layer_results.append(row)

    results = {
        "model": model_key,
        "model_name": model_name,
        "dataset": dataset_slug,
        "source_experiment_count": total_sources,
        "sample_count": len(all_examples),
        "seed": args.seed,
        "layers_swept": layers,
        "tensor_file": str(tensor_file),
        "position": args.position,
        "activation_kind": args.activation_kind,
        "direction": direction,
        "source_files": source_files,
        "examples": all_examples[:10],
        "layers": layer_results,
    }
    json_file.write_text(json.dumps(results, indent=2))
    write_analysis(results, analysis_file)
    print(f"Aggregated diffmean tensors saved to {tensor_file}")
    print(f"Aggregated results saved to {json_file}")
    print(f"Aggregated analysis saved to {analysis_file}")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)

    manifest = {
        "seed": args.seed,
        "dataset": args.dataset,
        "activation_kind": args.activation_kind,
        "position": args.position,
        "models": {},
    }
    for model_key in [item.strip() for item in args.models.split(",") if item.strip()]:
        if model_key not in MODEL_IDS:
            raise ValueError(f"Unknown model key {model_key!r}; choose from {sorted(MODEL_IDS)}")
        if args.dataset == "all":
            manifest["models"][model_key] = {
                dataset_name: aggregate_model(model_key, args, dataset_name)
                for dataset_name in ("mmlu", "bbh")
            }
        else:
            manifest["models"][model_key] = aggregate_model(model_key, args, args.dataset)

    manifest_file = args.output_dir / "manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2))
    print(f"Manifest saved to {manifest_file}")


if __name__ == "__main__":
    main()
