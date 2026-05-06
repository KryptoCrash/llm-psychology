"""Project cached response activations onto layer-wise diffmean directions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project activation_cache shards onto a diffmeans .pt direction."
    )
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--diffmeans-path", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, default=None)
    return parser.parse_args()


def load_manifest(cache_dir: Path) -> dict[str, Any]:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    return json.loads(manifest_path.read_text())


def summarize_diffmeans(diffmeans: dict[int, torch.Tensor]) -> dict[int, dict[str, Any]]:
    out = {}
    for layer, vector in diffmeans.items():
        vec = vector.detach().float().cpu().squeeze()
        norm = float(vec.norm().item())
        out[int(layer)] = {
            "norm": norm,
            "unit": vec / norm if norm > 0 else None,
        }
    return out


def scalar_stats(values: torch.Tensor) -> dict[str, float]:
    values = values.float()
    return {
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "mean_abs": float(values.abs().mean().item()),
    }


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.cache_dir)
    diff_obj = torch.load(args.diffmeans_path, map_location="cpu", weights_only=False)
    diffmeans = {int(k): v for k, v in diff_obj["diffmeans"].items()}
    diff_summary = summarize_diffmeans(diffmeans)
    layers = [int(layer) for layer in manifest["layers"] if int(layer) in diffmeans]

    accum = {
        layer: {
            "count": 0,
            "sum": 0.0,
            "sum_sq": 0.0,
            "sum_abs": 0.0,
            "min": None,
            "max": None,
            "pos_sum": None,
            "pos_count": None,
            "cos_sum": 0.0,
            "cos_count": 0,
        }
        for layer in layers
    }
    record_rows = []

    records = manifest["records"]
    for record in tqdm(records, desc="Projecting shards"):
        shard_path = args.cache_dir / record["shard"]
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        record_summary = {
            "record_idx": record["record_idx"],
            "shard": record["shard"],
            "model_answer": record.get("model_answer"),
            "is_correct": record.get("is_correct"),
            "conforms": record.get("conforms"),
            "layers": {},
        }
        for layer in layers:
            unit = diff_summary[layer]["unit"]
            if unit is None:
                continue
            acts = shard["activations_by_layer"][layer].float()
            projections = acts @ unit
            act_norms = acts.norm(dim=-1)
            cosines = projections / act_norms.clamp_min(1e-12)

            state = accum[layer]
            count = int(projections.numel())
            state["count"] += count
            state["sum"] += float(projections.sum().item())
            state["sum_sq"] += float((projections * projections).sum().item())
            state["sum_abs"] += float(projections.abs().sum().item())
            pmin = float(projections.min().item())
            pmax = float(projections.max().item())
            state["min"] = pmin if state["min"] is None else min(state["min"], pmin)
            state["max"] = pmax if state["max"] is None else max(state["max"], pmax)
            pos_sum = projections.detach().cpu()
            if state["pos_sum"] is None:
                state["pos_sum"] = pos_sum.clone()
                state["pos_count"] = torch.ones_like(pos_sum, dtype=torch.long)
            else:
                if len(pos_sum) > len(state["pos_sum"]):
                    pad_len = len(pos_sum) - len(state["pos_sum"])
                    state["pos_sum"] = torch.cat(
                        [state["pos_sum"], torch.zeros(pad_len, dtype=state["pos_sum"].dtype)]
                    )
                    state["pos_count"] = torch.cat(
                        [state["pos_count"], torch.zeros(pad_len, dtype=state["pos_count"].dtype)]
                    )
                state["pos_sum"][: len(pos_sum)] += pos_sum
                state["pos_count"][: len(pos_sum)] += 1
            state["cos_sum"] += float(cosines.sum().item())
            state["cos_count"] += int(cosines.numel())

            stats = scalar_stats(projections)
            stats["mean_cosine"] = float(cosines.mean().item())
            record_summary["layers"][str(layer)] = stats
        record_rows.append(record_summary)

    layer_rows = []
    for layer in layers:
        state = accum[layer]
        if state["count"] == 0:
            continue
        mean = state["sum"] / state["count"]
        variance = max(0.0, state["sum_sq"] / state["count"] - mean * mean)
        row = {
            "layer": layer,
            "diffmean_norm": diff_summary[layer]["norm"],
            "token_count": state["count"],
            "projection_mean": mean,
            "projection_std": variance ** 0.5,
            "projection_min": state["min"],
            "projection_max": state["max"],
            "projection_mean_abs": state["sum_abs"] / state["count"],
            "mean_cosine": state["cos_sum"] / state["cos_count"],
        }
        pos_mean = state["pos_sum"] / state["pos_count"].clamp_min(1)
        row["position_mean_first_20"] = [float(x) for x in pos_mean[:20]]
        row["position_mean_last_20"] = [float(x) for x in pos_mean[-20:]]
        layer_rows.append(row)

    output = {
        "cache_dir": str(args.cache_dir),
        "diffmeans_path": str(args.diffmeans_path),
        "cache_manifest": {
            "source_file": manifest.get("source_file"),
            "model": manifest.get("model"),
            "dataset": manifest.get("dataset"),
            "mode": manifest.get("mode"),
            "prompt_style": manifest.get("prompt_style"),
            "alignment": manifest.get("alignment"),
            "activation_kind": manifest.get("activation_kind"),
            "record_count": manifest.get("record_count"),
            "layers": manifest.get("layers"),
        },
        "diffmeans_metadata": {
            key: diff_obj.get(key)
            for key in [
                "experiment_file",
                "dataset",
                "mode",
                "prompt_style",
                "position",
                "activation_kind",
                "baseline",
                "direction",
                "sample_count",
            ]
        },
        "layers": layer_rows,
        "records": record_rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2))

    if args.output_md is not None:
        top_abs = sorted(layer_rows, key=lambda r: abs(r["projection_mean"]), reverse=True)[:10]
        top_cos = sorted(layer_rows, key=lambda r: abs(r["mean_cosine"]), reverse=True)[:10]
        lines = [
            "# Response Activation Projection",
            "",
            f"- cache: `{args.cache_dir}`",
            f"- diffmeans: `{args.diffmeans_path}`",
            f"- source experiment: `{manifest.get('source_file')}`",
            f"- diffmean direction: `{diff_obj.get('direction')}`",
            f"- diffmean position: `{diff_obj.get('position')}`",
            "",
            "## Top Layers By Absolute Mean Projection",
            "",
            "| layer | diffmean norm | mean projection | std | mean abs | mean cosine |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
        for row in top_abs:
            lines.append(
                f"| {row['layer']} | {row['diffmean_norm']:.6f} | "
                f"{row['projection_mean']:.6f} | {row['projection_std']:.6f} | "
                f"{row['projection_mean_abs']:.6f} | {row['mean_cosine']:.6f} |"
            )
        lines.extend(
            [
                "",
                "## Top Layers By Absolute Mean Cosine",
                "",
                "| layer | diffmean norm | mean projection | mean cosine |",
                "|---:|---:|---:|---:|",
            ]
        )
        for row in top_cos:
            lines.append(
                f"| {row['layer']} | {row['diffmean_norm']:.6f} | "
                f"{row['projection_mean']:.6f} | {row['mean_cosine']:.6f} |"
            )
        args.output_md.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
