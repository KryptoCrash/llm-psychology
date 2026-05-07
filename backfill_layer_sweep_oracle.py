from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

import oracle_utils
from experiment_layer_sweep import (
    CONSENSUS_PROMPT,
    MODEL_IDS,
    load_model,
    write_analysis,
)


ORACLE_PROMPTS = [
    "Describe this text.",
    "What belief or feature does this activation represent?",
    CONSENSUS_PROMPT,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill oracle responses for saved layer-sweep diffmean artifacts."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("experiment_layer_sweep_results_last_prompt"),
    )
    parser.add_argument("--models", default="llama,qwen", help="Comma-separated model keys.")
    parser.add_argument(
        "--oracle-lora-path",
        default="adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate oracle responses even when a layer already has them.",
    )
    parser.add_argument(
        "--limit-experiments",
        type=int,
        default=None,
        help="Optional limit per model for smoke testing.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Maximum new tokens for each oracle response.",
    )
    return parser.parse_args()


def load_manifest(results_dir: Path) -> dict[str, Any]:
    manifest_file = results_dir / "manifest.json"
    if not manifest_file.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_file}")
    return json.loads(manifest_file.read_text())


def needs_oracle(results: dict[str, Any], overwrite: bool) -> bool:
    if overwrite:
        return True
    return any(not row.get("oracle_responses") for row in results.get("layers", []))


def oracle_responses_for_layer(
    *,
    diffmean: torch.Tensor,
    layer: int,
    model,
    tokenizer,
    device: torch.device,
    max_new_tokens: int,
) -> dict[str, str]:
    oracle_inputs = [
        oracle_utils.create_oracle_input(
            prompt=prompt,
            layer=layer,
            num_positions=1,
            tokenizer=tokenizer,
            acts_BD=diffmean.unsqueeze(0),
        )
        for prompt in ORACLE_PROMPTS
    ]
    responses = oracle_utils._run_evaluation(
        eval_data=oracle_inputs,
        model=model,
        tokenizer=tokenizer,
        submodule=oracle_utils.get_hf_submodule(model, 1),
        device=device,
        dtype=torch.float32,
        lora_path="oracle",
        steering_coefficient=1.0,
        generation_kwargs={
            "do_sample": False,
            "temperature": 0.0,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
        },
    )
    return dict(zip(ORACLE_PROMPTS, [response.strip() for response in responses]))


def backfill_experiment(
    *,
    json_file: Path,
    model,
    tokenizer,
    device: torch.device,
    overwrite: bool,
    max_new_tokens: int,
) -> bool:
    results = json.loads(json_file.read_text())
    if not needs_oracle(results, overwrite):
        return False

    tensor_file = Path(results["tensor_file"])
    tensor_data = torch.load(tensor_file, map_location="cpu", weights_only=False)
    diffmeans = tensor_data["diffmeans"]

    changed = False
    for row in tqdm(results["layers"], desc=json_file.name, leave=False):
        if row.get("oracle_responses") and not overwrite:
            continue
        layer = int(row["layer"])
        row["oracle_responses"] = oracle_responses_for_layer(
            diffmean=diffmeans[layer],
            layer=layer,
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=max_new_tokens,
        )
        changed = True

    if changed:
        json_file.write_text(json.dumps(results, indent=2))
        analysis_file = json_file.with_name(json_file.stem + "_analysis.md")
        write_analysis(results, analysis_file)
    return changed


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.results_dir)
    model_keys = [item.strip() for item in args.models.split(",") if item.strip()]

    for model_key in model_keys:
        if model_key not in manifest["models"]:
            raise KeyError(f"{model_key!r} not found in {args.results_dir / 'manifest.json'}")

        experiments = manifest["models"][model_key]["experiments"]
        if args.limit_experiments is not None:
            experiments = experiments[: args.limit_experiments]

        pending = []
        for experiment in experiments:
            stem = Path(experiment["experiment_file"]).stem
            json_file = (
                args.results_dir
                / model_key
                / f"{stem}_{manifest['activation_kind']}.json"
            )
            results = json.loads(json_file.read_text())
            if needs_oracle(results, args.overwrite):
                pending.append(json_file)

        print(f"{model_key}: {len(pending)} experiments need oracle backfill")
        if not pending:
            continue

        print(f"Loading {model_key}: {MODEL_IDS[model_key]}")
        model, tokenizer, device = load_model(model_key)
        print(f"Loading oracle LoRA: {args.oracle_lora_path}")
        model.load_adapter(args.oracle_lora_path, adapter_name="oracle", is_trainable=False)
        model.set_adapter("oracle")

        for json_file in tqdm(pending, desc=f"Backfilling {model_key}"):
            backfill_experiment(
                json_file=json_file,
                model=model,
                tokenizer=tokenizer,
                device=device,
                overwrite=args.overwrite,
                max_new_tokens=args.max_new_tokens,
            )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
