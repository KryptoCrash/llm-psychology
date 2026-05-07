from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from tqdm import tqdm

from layer_sweep import collect_activations, load_model


EXPERIMENT_RE = re.compile(r"qwen_(mmlu|bbh)_(.+)_(base|explain)\.json$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Project mean pure assistant-start activations from saved Qwen experiments "
            "onto a layer_sweep diffmean vector."
        )
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path(
            "layer_sweep_results/qwen/"
            "layer_sweep_qwen_mmlu_mode_8_base_residual_assistant_start_"
            "all_wrong_minus_random_answers_n8_100_diffmeans.pt"
        ),
    )
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/qwen"))
    parser.add_argument("--output-dir", type=Path, default=Path("projection_results"))
    parser.add_argument("--layer", type=int, default=23)
    parser.add_argument("--position", default="assistant_start")
    parser.add_argument("--activation-kind", default="residual")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute experiment means even when a cache file exists.",
    )
    return parser.parse_args()


def discover_experiments(runs_dir: Path) -> list[Path]:
    paths = []
    for path in sorted(runs_dir.glob("*/*.json")):
        if EXPERIMENT_RE.fullmatch(path.name):
            paths.append(path)
    return paths


def load_reference(path: Path, layer: int) -> tuple[torch.Tensor, dict]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    ref = data["diffmeans"][layer].detach().cpu().float()
    norm = ref.norm()
    if norm.item() == 0:
        raise ValueError(f"Reference layer {layer} has zero norm")
    return ref, data


def experiment_info(path: Path) -> dict:
    match = EXPERIMENT_RE.fullmatch(path.name)
    if not match:
        raise ValueError(f"Unexpected experiment path: {path}")
    dataset, mode, prompt_style = match.groups()
    return {"dataset": dataset, "mode": mode, "prompt_style": prompt_style}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    reference, reference_data = load_reference(args.reference, args.layer)
    reference_unit = reference / reference.norm()
    experiments = discover_experiments(args.runs_dir)
    if not experiments:
        raise FileNotFoundError(f"No Qwen experiment JSON files found under {args.runs_dir}")

    print(f"Reference: {args.reference}")
    print(f"Reference layer: {args.layer}")
    print(f"Reference norm: {reference.norm().item():.6f}")
    print(f"Experiments: {len(experiments)}")

    missing_caches = [
        experiment_file
        for experiment_file in experiments
        if not (
            cache_dir
            / f"{experiment_file.stem}_layer{args.layer}_{args.activation_kind}_{args.position}.pt"
        ).exists()
    ]
    model = tokenizer = device = None
    if missing_caches or args.overwrite:
        model, tokenizer, device = load_model("qwen")
    else:
        print("Using cached experiment mean activations; model load skipped.")

    rows = []
    for experiment_file in tqdm(experiments, desc="Experiments"):
        info = experiment_info(experiment_file)
        cache_file = cache_dir / f"{experiment_file.stem}_layer{args.layer}_{args.activation_kind}_{args.position}.pt"
        if cache_file.exists() and not args.overwrite:
            cache = torch.load(cache_file, map_location="cpu", weights_only=False)
            mean_activation = cache["mean_activation"].float()
            position_counts = cache.get("position_counts", {})
            sample_count = int(cache["sample_count"])
        else:
            assert model is not None and tokenizer is not None and device is not None
            experiment = json.loads(experiment_file.read_text())
            records = experiment["results"]
            activation_sum = None
            position_counts: dict[str, int] = {}
            for record in tqdm(records, desc=experiment_file.stem, leave=False):
                acts, _, token = collect_activations(
                    record["prompt"],
                    model,
                    tokenizer,
                    device,
                    [args.layer],
                    args.activation_kind,
                    args.position,
                )
                activation = acts[args.layer].detach().cpu().float()
                activation_sum = activation if activation_sum is None else activation_sum + activation
                position_counts[token] = position_counts.get(token, 0) + 1
            sample_count = len(records)
            mean_activation = activation_sum / sample_count
            torch.save(
                {
                    "experiment_file": str(experiment_file),
                    "layer": args.layer,
                    "position": args.position,
                    "activation_kind": args.activation_kind,
                    "sample_count": sample_count,
                    "mean_activation": mean_activation,
                    "position_counts": position_counts,
                },
                cache_file,
            )

        dot = float(torch.dot(mean_activation, reference).item())
        scalar_projection = float(torch.dot(mean_activation, reference_unit).item())
        mean_norm = float(mean_activation.norm().item())
        cosine = scalar_projection / mean_norm if mean_norm else 0.0
        rows.append(
            {
                "experiment_file": str(experiment_file),
                "experiment": experiment_file.name,
                **info,
                "sample_count": sample_count,
                "layer": args.layer,
                "position": args.position,
                "activation_kind": args.activation_kind,
                "dot_with_reference": dot,
                "projection_onto_unit_reference": scalar_projection,
                "mean_activation_norm": mean_norm,
                "cosine_with_reference": cosine,
                "position_counts": position_counts,
                "cache_file": str(cache_file),
            }
        )

    rows.sort(key=lambda row: row["projection_onto_unit_reference"], reverse=True)

    out_json = args.output_dir / (
        f"qwen_experiment_mean_activation_projection_layer{args.layer}_"
        f"onto_{args.reference.stem}.json"
    )
    out_md = out_json.with_suffix(".md")
    payload = {
        "reference": str(args.reference),
        "reference_metadata": {
            "model": reference_data.get("model"),
            "model_name": reference_data.get("model_name"),
            "dataset": reference_data.get("dataset"),
            "mode": reference_data.get("mode"),
            "n": reference_data.get("n"),
            "position": reference_data.get("position"),
            "activation_kind": reference_data.get("activation_kind"),
            "direction": reference_data.get("direction"),
            "layer": args.layer,
            "reference_norm": float(reference.norm().item()),
        },
        "rows": rows,
    }
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Qwen Experiment Mean Activation Projection",
        "",
        f"- reference: `{args.reference}`",
        f"- reference layer: {args.layer}",
        f"- reference norm: {reference.norm().item():.6f}",
        f"- projected activation: pure mean residual activation at `{args.position}`",
        f"- experiment count: {len(rows)}",
        "",
        "| rank | experiment | dataset | mode | prompt | projection | cosine | mean_norm | n |",
        "|---:|---|---|---|---|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(rows, 1):
        lines.append(
            "| {rank} | {experiment} | {dataset} | {mode} | {prompt_style} | {proj:.6f} | {cos:.6f} | {norm:.6f} | {n} |".format(
                rank=rank,
                experiment=row["experiment"],
                dataset=row["dataset"],
                mode=row["mode"],
                prompt_style=row["prompt_style"],
                proj=row["projection_onto_unit_reference"],
                cos=row["cosine_with_reference"],
                norm=row["mean_activation_norm"],
                n=row["sample_count"],
            )
        )
    out_md.write_text("\n".join(lines))

    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
