"""Cache per-layer activations for generated response tokens.

The raw experiment JSONs written by multi_actor.py already contain
prompt_token_ids and response_token_ids. This script reconstructs each
prompt+response sequence, runs the matching model under teacher forcing, and
saves one activation shard per result record.

Default input discovery targets the current full experiment set:
    runs/{llama,qwen}/{mmlu,bbh}/*.json

Output layout:
    activation_cache/{model}/{dataset}/{experiment_stem}/
      manifest.json
      record_00000.pt
      record_00001.pt
      ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is expected in this repo env
    tqdm = None


MODEL_IDS = {
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen": "Qwen/Qwen3-8B",
}

LAYER_COUNTS = {
    "meta-llama/Llama-3.1-8B-Instruct": 32,
    "Qwen/Qwen3-8B": 36,
}

HIDDEN_SIZES = {
    "llama": 4096,
    "qwen": 4096,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache response-token activations for saved multi_actor experiments."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("runs"),
        help="Directory containing raw experiment JSONs. Default: runs/",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("activation_cache"),
        help="Directory to write activation cache shards. Default: activation_cache/",
    )
    parser.add_argument(
        "--models",
        default="llama,qwen",
        help="Comma-separated model keys to process. Default: llama,qwen.",
    )
    parser.add_argument(
        "--datasets",
        default="mmlu,bbh",
        help="Comma-separated datasets to process. Default: mmlu,bbh.",
    )
    parser.add_argument(
        "--modes",
        default=None,
        help="Optional comma-separated modes, e.g. 1,3,10,qd,da.",
    )
    parser.add_argument(
        "--prompt-styles",
        default="base,explain",
        help="Comma-separated prompt styles to process: base,explain.",
    )
    parser.add_argument(
        "--layers",
        default="all",
        help='Comma-separated layer ids, or "all".',
    )
    parser.add_argument(
        "--alignment",
        choices=["response", "prediction"],
        default="response",
        help=(
            "response caches activations at generated-token positions. "
            "prediction caches positions that predicted each generated token "
            "(prompt last token for the first response token)."
        ),
    )
    parser.add_argument(
        "--save-dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
        help="Dtype used for saved activation tensors. Default: auto.",
    )
    parser.add_argument(
        "--limit-experiments",
        type=int,
        default=None,
        help="Optional number of experiment files per model to process.",
    )
    parser.add_argument(
        "--limit-records",
        type=int,
        default=None,
        help="Optional number of records per experiment to process.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected experiments and storage estimate without loading models.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing record shards.",
    )
    return parser.parse_args()


def split_csv(value: str | None) -> set[str] | None:
    if value is None:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def infer_prompt_style(path: Path, data: dict[str, Any]) -> str:
    if path.stem.endswith("_explain"):
        return "explain"
    if path.stem.endswith("_base"):
        return "base"
    results = data.get("results") or []
    if results and "Reasoning:" in str(results[0].get("prompt", "")):
        return "explain"
    return "base"


def raw_experiment_files(input_dir: Path) -> list[Path]:
    canonical_files = sorted(input_dir.glob("*/*/*.json"))
    candidates = canonical_files if canonical_files else sorted(input_dir.rglob("*.json"))
    files = []
    for path in candidates:
        if path.name == "manifest.json" or path.name.endswith("_results.json"):
            continue
        files.append(path)
    return files


def load_experiment(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    if not isinstance(data.get("results"), list):
        return None
    required = {"model", "dataset", "mode"}
    if not required.issubset(data):
        return None
    return data


def select_experiments(args: argparse.Namespace) -> list[tuple[Path, dict[str, Any], str]]:
    models = split_csv(args.models) or set(MODEL_IDS)
    datasets = split_csv(args.datasets) or {"mmlu", "bbh"}
    modes = split_csv(args.modes)
    prompt_styles = split_csv(args.prompt_styles) or {"base", "explain"}

    selected = []
    per_model_counts: dict[str, int] = {}
    for path in raw_experiment_files(args.input_dir):
        data = load_experiment(path)
        if data is None:
            continue
        model_key = data["model"]
        dataset = data["dataset"]
        mode = str(data["mode"])
        prompt_style = infer_prompt_style(path, data)
        if model_key not in models or dataset not in datasets:
            continue
        if modes is not None and mode not in modes:
            continue
        if prompt_style not in prompt_styles:
            continue
        if args.limit_experiments is not None:
            count = per_model_counts.get(model_key, 0)
            if count >= args.limit_experiments:
                continue
            per_model_counts[model_key] = count + 1
        selected.append((path, data, prompt_style))
    return selected


def layer_ids_for_model(model_key: str, layers_arg: str) -> list[int]:
    layer_count = LAYER_COUNTS[MODEL_IDS[model_key]]
    if layers_arg == "all":
        return list(range(layer_count))
    layers = [int(item.strip()) for item in layers_arg.split(",") if item.strip()]
    for layer in layers:
        if layer < 0 or layer >= layer_count:
            raise ValueError(
                f"Layer {layer} is out of range for {MODEL_IDS[model_key]} "
                f"with {layer_count} layers"
            )
    return layers


def response_positions(prompt_len: int, response_len: int, alignment: str) -> list[int]:
    if response_len == 0:
        return []
    if alignment == "response":
        return list(range(prompt_len, prompt_len + response_len))
    if prompt_len == 0:
        raise ValueError("prediction alignment requires a non-empty prompt")
    return list(range(prompt_len - 1, prompt_len + response_len - 1))


def save_dtype_from_arg(value: str, fallback: Any) -> Any:
    import torch

    if value == "auto":
        return fallback
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[value]


def model_input_device(model: Any) -> Any:
    return model.get_input_embeddings().weight.device


def load_model(model_key: str) -> tuple[Any, Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = MODEL_IDS[model_key]
    if torch.cuda.is_available():
        dtype = torch.bfloat16
    elif torch.backends.mps.is_available():
        dtype = torch.float16
    else:
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer, dtype


def collect_record_activations(
    model: Any,
    input_ids: list[int],
    positions: list[int],
    layers: list[int],
    save_dtype: Any,
) -> dict[int, Any]:
    import torch

    import oracle_utils

    if not positions:
        hidden_size = int(model.config.hidden_size)
        return {
            layer: torch.empty((0, hidden_size), dtype=save_dtype)
            for layer in layers
        }

    max_pos = max(positions)
    if max_pos >= len(input_ids):
        raise ValueError(f"Position {max_pos} is outside input length {len(input_ids)}")

    activations: dict[int, Any] = {}
    module_to_layer: dict[Any, int] = {}
    handles = []

    def hook_fn(module: Any, _inputs: Any, outputs: Any) -> None:
        layer = module_to_layer[module]
        resid = outputs[0] if isinstance(outputs, tuple) else outputs
        activations[layer] = resid[0, positions, :].detach().to("cpu", dtype=save_dtype)

    for layer in layers:
        submodule = oracle_utils.get_hf_submodule(model, layer)
        module_to_layer[submodule] = layer
        handles.append(submodule.register_forward_hook(hook_fn))

    device = model_input_device(model)
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(ids, dtype=torch.long, device=device)
    try:
        with torch.no_grad():
            model(input_ids=ids, attention_mask=attention_mask, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    missing = [layer for layer in layers if layer not in activations]
    if missing:
        raise RuntimeError(f"Did not collect activations for layers: {missing}")
    return activations


def record_metadata(record: dict[str, Any], record_idx: int, shard_name: str) -> dict[str, Any]:
    return {
        "record_idx": record_idx,
        "shard": shard_name,
        "subject": record.get("subject"),
        "question": record.get("question"),
        "model_answer": record.get("model_answer"),
        "is_correct": record.get("is_correct"),
        "conforms": record.get("conforms"),
        "generation_attempts": record.get("generation_attempts"),
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(manifest, indent=2))
    tmp_path.replace(path)


def cache_experiment(
    *,
    path: Path,
    data: dict[str, Any],
    prompt_style: str,
    model: Any,
    tokenizer: Any,
    model_dtype: Any,
    args: argparse.Namespace,
) -> None:
    import torch

    model_key = data["model"]
    dataset = data["dataset"]
    layers = layer_ids_for_model(model_key, args.layers)
    save_dtype = save_dtype_from_arg(args.save_dtype, model_dtype)
    records = data["results"]
    if args.limit_records is not None:
        records = records[: args.limit_records]

    out_dir = args.output_dir / model_key / dataset / path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest = {
        "source_file": str(path),
        "model": model_key,
        "model_name": MODEL_IDS[model_key],
        "dataset": dataset,
        "mode": data.get("mode"),
        "n": data.get("n"),
        "prompt_style": prompt_style,
        "alignment": args.alignment,
        "activation_kind": "residual",
        "layers": layers,
        "save_dtype": str(save_dtype).replace("torch.", ""),
        "record_count": len(records),
        "records": [],
    }

    iterator = enumerate(records)
    if tqdm is not None:
        iterator = tqdm(iterator, total=len(records), desc=path.name)

    for record_idx, record in iterator:
        shard_name = f"record_{record_idx:05d}.pt"
        shard_path = out_dir / shard_name
        manifest["records"].append(record_metadata(record, record_idx, shard_name))
        if shard_path.exists() and not args.force:
            continue

        prompt_ids = list(record["prompt_token_ids"])
        response_ids = list(record["response_token_ids"])
        full_ids = prompt_ids + response_ids
        positions = response_positions(len(prompt_ids), len(response_ids), args.alignment)
        source_position_token_ids = [full_ids[position] for position in positions]
        source_position_tokens = tokenizer.convert_ids_to_tokens(source_position_token_ids)
        activations = collect_record_activations(
            model=model,
            input_ids=full_ids,
            positions=positions,
            layers=layers,
            save_dtype=save_dtype,
        )

        torch.save(
            {
                "source_file": str(path),
                "record_idx": record_idx,
                "model": model_key,
                "model_name": MODEL_IDS[model_key],
                "dataset": dataset,
                "mode": data.get("mode"),
                "prompt_style": prompt_style,
                "alignment": args.alignment,
                "activation_kind": "residual",
                "layers": layers,
                "prompt_len": len(prompt_ids),
                "response_len": len(response_ids),
                "positions": positions,
                "response_token_ids": response_ids,
                "response_tokens": record.get("response_tokens", []),
                "source_position_token_ids": source_position_token_ids,
                "source_position_tokens": source_position_tokens,
                "activations_by_layer": activations,
            },
            shard_path,
        )
        del activations
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_manifest(manifest_path, manifest)


def dry_run(experiments: list[tuple[Path, dict[str, Any], str]], args: argparse.Namespace) -> None:
    total_tokens_by_model = {model: 0 for model in MODEL_IDS}
    total_records = 0
    print(f"Selected experiments: {len(experiments)}")
    for path, data, prompt_style in experiments:
        records = data["results"]
        if args.limit_records is not None:
            records = records[: args.limit_records]
        response_tokens = sum(len(record.get("response_token_ids", [])) for record in records)
        layers = layer_ids_for_model(data["model"], args.layers)
        total_tokens_by_model[data["model"]] += response_tokens * len(layers)
        total_records += len(records)
        print(
            f"{path} | model={data['model']} dataset={data['dataset']} "
            f"mode={data['mode']} style={prompt_style} records={len(records)} "
            f"response_tokens={response_tokens} layers={len(layers)}"
        )

    dtype_bytes = 2 if args.save_dtype in {"auto", "float16", "bfloat16"} else 4
    estimated = 0
    for model_key, token_layer_count in total_tokens_by_model.items():
        estimated += token_layer_count * HIDDEN_SIZES[model_key] * dtype_bytes
    print(f"\nTotal records: {total_records}")
    print(f"Estimated activation payload: {estimated / (1024 ** 3):.2f} GiB")
    print("Estimate excludes shard/manifest overhead.")


def main() -> None:
    args = parse_args()
    experiments = select_experiments(args)
    if args.dry_run:
        dry_run(experiments, args)
        return

    if not experiments:
        print("No matching experiment files found.")
        return

    experiments_by_model: dict[str, list[tuple[Path, dict[str, Any], str]]] = {}
    for item in experiments:
        experiments_by_model.setdefault(item[1]["model"], []).append(item)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    import torch

    torch.set_grad_enabled(False)
    for model_key, model_experiments in experiments_by_model.items():
        print(f"Loading {model_key}: {MODEL_IDS[model_key]}")
        model, tokenizer, model_dtype = load_model(model_key)
        try:
            for path, data, prompt_style in model_experiments:
                cache_experiment(
                    path=path,
                    data=data,
                    prompt_style=prompt_style,
                    model=model,
                    tokenizer=tokenizer,
                    model_dtype=model_dtype,
                    args=args,
                )
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
