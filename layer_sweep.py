import argparse
import json
import random
import re
from pathlib import Path

import torch
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import oracle_utils
import prompts


MODEL_IDS = {
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen": "Qwen/Qwen3-8B",
}
LETTER = prompts.LETTER
CONSENSUS_PROMPT = "Does this activation represent pressure toward an incorrect consensus? Answer only True or False."
EXPERIMENT_FILE_RE = re.compile(r"(llama|qwen)_(mmlu|bbh)_(.+)_(base|explain)(?:_(residual|delta))?$")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="qwen", help="Comma-separated model keys.")
    parser.add_argument("--input-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("layer_sweep_results"))
    parser.add_argument("--dataset", choices=["mmlu", "bbh"], default="mmlu")
    parser.add_argument(
        "--mode",
        default=None,
        help="Optional multi_actor mode: 1-10, qd, or da. Overrides --n/--qd/--contrast.",
    )
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument(
        "--limit",
        default="100",
        help='Number of questions to sample, or "all" for the full available pool.',
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sample-experiments",
        type=int,
        default=None,
        help="Accepted for config parity with experiment_layer_sweep.py; synthetic sweeps do not use experiment files.",
    )
    parser.add_argument("--layers", default="all", help='Comma-separated layers, or "all".')
    parser.add_argument(
        "--contrast",
        choices=["all_wrong", "random_deviant"],
        default="all_wrong",
        help=(
            "Condition A to compare against the baseline. random_deviant uses a wrong "
            "majority answer with one random participant choosing a different answer."
        ),
    )
    parser.add_argument(
        "--baseline",
        choices=["no_participants", "random_answers", "same_prompt"],
        default="random_answers",
        help=(
            "Condition B. no_participants uses a neutral prompt. random_answers shows "
            "all prior participants with independently random answers. same_prompt is a zero control."
        ),
    )
    parser.add_argument(
        "--position",
        choices=["last_prompt_token", "assistant_start"],
        default="assistant_start",
        help=(
            "Activation position to compare. last_prompt_token captures the final token of "
            "the experiment prompt; assistant_start captures the first assistant-template token."
        ),
    )
    parser.add_argument(
        "--activation-kind",
        choices=["residual", "delta"],
        default="residual",
        help="Use layer output residual stream, or per-layer residual update x_out - x_in.",
    )
    parser.add_argument("--explain", action="store_true", help="Use the reasoning prompt variant.")
    parser.add_argument("--skip-oracle", action="store_true", default=True)
    parser.add_argument("--run-oracle", dest="skip_oracle", action="store_false")
    parser.add_argument(
        "--oracle-lora-path",
        default="adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--qd", action="store_true")
    args = parser.parse_args()
    args.limit = None if str(args.limit).lower() == "all" else int(args.limit)
    return args


def parse_experiment_file_name(path: Path) -> dict[str, str]:
    match = EXPERIMENT_FILE_RE.fullmatch(path.stem)
    if not match:
        raise ValueError(f"Unexpected experiment filename: {path.name}")
    model, dataset, mode, prompt_style, activation_kind = match.groups()
    return {
        "model": model,
        "dataset": dataset,
        "mode": mode,
        "prompt_style": prompt_style,
        "explain": prompt_style == "explain",
        "activation_kind": activation_kind,
    }


def experiment_files(input_dir: Path, model: str) -> list[Path]:
    files = []
    for path in sorted(input_dir.glob(f"{model}_*_*.json")):
        if path.name.endswith("_correct.json") or path.name.endswith("_results.json"):
            continue
        files.append(path)
    return files


def maybe_sample_experiment_files(
    files: list[Path],
    model: str,
    sample_experiments: int | None,
    seed: int,
) -> list[Path]:
    if sample_experiments is None or sample_experiments >= len(files):
        return files
    rng = random.Random(f"{seed}:sample_experiments:{model}")
    return sorted(rng.sample(files, sample_experiments))


def load_saved_experiment(path: Path):
    return json.loads(path.read_text())


def flat_input_ids(tokenized) -> list[int]:
    if isinstance(tokenized, dict) or hasattr(tokenized, "keys"):
        tokenized = tokenized["input_ids"]
    if hasattr(tokenized, "ids"):
        tokenized = tokenized.ids
    if hasattr(tokenized, "tolist"):
        tokenized = tokenized.tolist()
    while isinstance(tokenized, (list, tuple)) and tokenized and not isinstance(tokenized[0], int):
        first = tokenized[0]
        if hasattr(first, "ids"):
            tokenized = first.ids
        elif hasattr(first, "tolist"):
            tokenized = first.tolist()
        else:
            tokenized = first
    return list(tokenized)


def find_subsequence(values: list[int], pattern: list[int]) -> int | None:
    if not pattern or len(pattern) > len(values):
        return None
    for start in range(len(values) - len(pattern), -1, -1):
        if values[start : start + len(pattern)] == pattern:
            return start
    return None


def chat_inputs(
    prompt: str,
    tokenizer: AutoTokenizer,
    device: torch.device,
    position_name: str,
) -> tuple[dict[str, torch.Tensor], int, str]:
    messages = [{"role": "user", "content": prompt}]
    kwargs = {}
    if "Qwen" in getattr(tokenizer, "name_or_path", ""):
        kwargs["enable_thinking"] = True
    prompt_inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        **kwargs,
    )
    without_generation = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors=None,
        **kwargs,
    )
    without_generation_ids = flat_input_ids(without_generation)
    input_ids = prompt_inputs["input_ids"][0].tolist()
    if position_name == "last_prompt_token":
        position = len(without_generation_ids) - 1
    else:
        assistant_start_patterns = [
            "<|im_start|>assistant\n",
            "<|start_header_id|>assistant<|end_header_id|>\n\n",
        ]
        position = None
        for pattern_text in assistant_start_patterns:
            pattern_ids = tokenizer.encode(pattern_text, add_special_tokens=False)
            position = find_subsequence(input_ids, pattern_ids)
            if position is not None:
                break
        if position is None:
            position = len(without_generation_ids)
    if position >= len(input_ids):
        position = len(input_ids) - 1
    inputs = {key: value.to(device) for key, value in prompt_inputs.items()}
    return inputs, position, tokenizer.decode([input_ids[position]])


def collect_activations(
    prompt: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    layers: list[int],
    activation_kind: str,
    position_name: str,
) -> tuple[dict[int, torch.Tensor], int, str]:
    if hasattr(model, "set_adapter"):
        model.set_adapter("default")

    inputs, position, token = chat_inputs(prompt, tokenizer, device, position_name)
    if activation_kind == "delta":
        return collect_layer_deltas(model, inputs, layers, position), position, token

    acts_by_layer = oracle_utils._collect_target_activations(
        model=model,
        inputs_BL=inputs,
        act_layers=layers,
        target_lora_path=None,
    )
    return (
        {layer: acts_by_layer[layer][0, position].detach().cpu() for layer in layers},
        position,
        token,
    )


def collect_layer_deltas(
    model: AutoModelForCausalLM,
    inputs: dict[str, torch.Tensor],
    layers: list[int],
    position: int,
) -> dict[int, torch.Tensor]:
    deltas = {}
    module_to_layer = {}
    handles = []

    def hook_fn(module, module_inputs, outputs):
        layer = module_to_layer[module]
        x_in = module_inputs[0]
        x_out = outputs[0] if isinstance(outputs, tuple) else outputs
        deltas[layer] = (x_out[:, position, :] - x_in[:, position, :]).detach().cpu()

    for layer in layers:
        submodule = oracle_utils.get_hf_submodule(model, layer)
        module_to_layer[submodule] = layer
        handles.append(submodule.register_forward_hook(hook_fn))

    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        for handle in handles:
            handle.remove()

    return {layer: deltas[layer][0] for layer in layers}


def query_oracle_vector(
    vector: torch.Tensor,
    layer: int,
    prompts: list[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
) -> dict[str, str]:
    responses = {}
    for prompt in prompts:
        oracle_input = oracle_utils.create_oracle_input(
            prompt=prompt,
            layer=layer,
            num_positions=1,
            tokenizer=tokenizer,
            acts_BD=vector.unsqueeze(0),
        )
        responses[prompt] = oracle_utils._run_evaluation(
            eval_data=[oracle_input],
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
                "max_new_tokens": 100,
                "pad_token_id": tokenizer.pad_token_id,
            },
        )[0].strip()
    return responses


def write_analysis(results: dict, output_file: Path) -> None:
    rows = results["layers"]
    top_norms = sorted(rows, key=lambda row: row["diffmean_norm"], reverse=True)[:10]
    true_layers = [
        row["layer"]
        for row in rows
        if row.get("oracle_responses", {}).get(CONSENSUS_PROMPT, "").lower().startswith("true")
    ]

    lines = [
        "# Layer Sweep Analysis",
        "",
        f"- dataset: {results['dataset']}",
        f"- explain: {results['explain']}",
        f"- contrast: {results['direction']}",
        f"- n: {results['n']}",
        f"- sample_count: {results['sample_count']}",
        f"- position: {results['position']}",
        "",
        "## Largest Diffmean Norms",
        "",
        "| layer | norm |",
        "|---:|---:|",
    ]
    lines.extend(f"| {row['layer']} | {row['diffmean_norm']:.6f} |" for row in top_norms)
    lines.extend(["", "## Oracle True Layers", ""])
    lines.append(", ".join(map(str, true_layers)) if true_layers else "None")
    lines.extend(["", "## Per-Layer Oracle Responses", ""])
    for row in rows:
        lines.append(f"### Layer {row['layer']}")
        lines.append(f"- norm: {row['diffmean_norm']:.6f}")
        for prompt, response in row.get("oracle_responses", {}).items():
            lines.append(f"- {prompt}: {response}")
        lines.append("")
    output_file.write_text("\n".join(lines))


def effective_mode(args: argparse.Namespace) -> tuple[int, bool, bool, str, str]:
    if args.mode is not None:
        n, use_qd, use_da = prompts.parse_mode(args.mode)
        contrast = "random_deviant" if use_da else "all_wrong"
        return n, use_qd, use_da, contrast, args.mode
    mode = "qd" if args.qd else str(args.n)
    return args.n, args.qd, args.contrast == "random_deviant", args.contrast, mode


def load_model(model_key: str) -> tuple[AutoModelForCausalLM, AutoTokenizer, torch.device]:
    model_name = MODEL_IDS[model_key]
    dtype = torch.bfloat16
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", dtype=dtype)
    model.eval()
    model.add_adapter(LoraConfig(), adapter_name="default")
    return model, tokenizer, device


def process_model(model_key: str, args: argparse.Namespace) -> dict:
    n, use_qd, use_da, contrast, mode = effective_mode(args)
    if n < 2:
        raise ValueError("--n/--mode must describe at least 2 participants for a social contrast")

    model_name = MODEL_IDS[model_key]
    model_dir = args.output_dir / model_key
    model_dir.mkdir(parents=True, exist_ok=True)
    layer_count = oracle_utils.LAYER_COUNTS[model_name]
    layers = (
        list(range(layer_count))
        if args.layers == "all"
        else [int(layer.strip()) for layer in args.layers.split(",") if layer.strip()]
    )

    print(f"Loading {args.limit} {args.dataset.upper()} questions")
    questions = prompts.load_questions(args.input_dir, args.dataset, args.limit, args.seed)
    limit_slug = args.limit if args.limit is not None else "all"
    questions_file = model_dir / f"{args.dataset}_{limit_slug}_seed_{args.seed}.json"
    questions_file.write_text(json.dumps(questions, indent=2))

    rng = random.Random(args.seed)
    direction = f"{contrast}_minus_{args.baseline}"
    explain_slug = "explain" if args.explain else "base"
    mode_slug = f"mode_{mode}"
    run_slug = (
        f"layer_sweep_{model_key}_{args.dataset}_{mode_slug}_{explain_slug}_"
        f"{args.activation_kind}_{args.position}_{direction}_n{n}_{limit_slug}"
    )
    checkpoint_file = model_dir / f"{run_slug}_checkpoint.pt"

    dry_examples = []
    if args.dry_run:
        rows_for_preview = questions[: min(3, len(questions))]
    else:
        print(f"Loading {model_key}: {model_name}")
        model, tokenizer, device = load_model(model_key)
        print("Model loaded successfully")
        rows_for_preview = questions

    diff_sums = {layer: None for layer in layers}
    examples = []

    for record_idx, row in enumerate(tqdm(rows_for_preview, desc=f"Collecting diffmeans {model_key}", disable=args.dry_run)):
        if args.dataset == "mmlu":
            ground_truth = LETTER[int(row["answer"])]
            wrong_answer = rng.choice([letter for letter in LETTER if letter != ground_truth])
            example_answer = rng.choice(LETTER)
        else:
            ground_truth = row["target"].strip()
            wrong_answer = prompts.wrong_bbh_answer(ground_truth, rng)
            _, example_answer = prompts.bbh_format(ground_truth, rng)

        all_wrong_answers = [wrong_answer] * (n - 1)
        deviant_participant = rng.randint(1, n - 1)
        deviant_answer = prompts.da_answer(args.dataset, ground_truth, wrong_answer, rng)
        random_deviant_answers = list(all_wrong_answers)
        random_deviant_answers[deviant_participant - 1] = deviant_answer
        random_baseline_answers = [
            rng.choice(prompts.answer_pool(args.dataset, ground_truth, wrong_answer)) for _ in range(n - 1)
        ]

        condition_a_answers = random_deviant_answers if use_da else all_wrong_answers
        condition_a_prompt = prompts.build_social_prompt(
            row,
            args.dataset,
            n,
            condition_a_answers,
            use_qd,
            args.explain,
            example_answer,
        )
        if args.baseline == "random_answers":
            condition_b_answers = random_baseline_answers
            condition_b_prompt = prompts.build_social_prompt(
                row,
                args.dataset,
                n,
                condition_b_answers,
                use_qd,
                args.explain,
                example_answer,
                qd_random=use_qd,
            )
        elif args.baseline == "same_prompt":
            condition_b_answers = condition_a_answers
            condition_b_prompt = condition_a_prompt
        else:
            condition_b_answers = None
            condition_b_prompt = prompts.build_no_participants_prompt(
                row, args.dataset, args.explain, example_answer
            )

        example_record = {
            "record_idx": record_idx,
            "subject": row.get("subject"),
            "ground_truth": ground_truth,
            "wrong_answer": wrong_answer,
            "example_answer": example_answer,
            "all_wrong_answers": all_wrong_answers,
            "random_deviant_answers": random_deviant_answers,
            "deviant_participant": deviant_participant,
            "deviant_answer": deviant_answer,
            "deviant_is_correct": str(deviant_answer).lower() == str(ground_truth).lower(),
            "condition_a_answers": condition_a_answers,
            "condition_b": args.baseline,
            "condition_b_answers": condition_b_answers,
            "condition_a_prompt": condition_a_prompt,
            "condition_b_prompt": condition_b_prompt,
        }
        if args.dry_run:
            dry_examples.append(example_record)
            continue

        condition_a_acts, condition_a_idx, condition_a_token = collect_activations(
            condition_a_prompt, model, tokenizer, device, layers, args.activation_kind, args.position
        )
        condition_b_acts, condition_b_idx, condition_b_token = collect_activations(
            condition_b_prompt, model, tokenizer, device, layers, args.activation_kind, args.position
        )

        for layer in layers:
            diff = condition_a_acts[layer] - condition_b_acts[layer]
            diff_sums[layer] = diff if diff_sums[layer] is None else diff_sums[layer] + diff

        example_record.update(
            {
                "condition_a_position_index": condition_a_idx,
                "condition_b_position_index": condition_b_idx,
                "condition_a_position_token": condition_a_token,
                "condition_b_position_token": condition_b_token,
            }
        )
        examples.append(example_record)

        if (record_idx + 1) % 10 == 0:
            torch.save(
                {
                    "processed": record_idx + 1,
                    "layers": layers,
                    "diff_sums": diff_sums,
                    "examples": examples,
                },
                checkpoint_file,
            )

    if args.dry_run:
        return {"model": model_key, "model_name": model_name, "preview_examples": dry_examples}

    diffmeans = {layer: diff_sums[layer] / len(questions) for layer in layers}
    tensor_file = model_dir / f"{run_slug}_diffmeans.pt"
    torch.save(
        {
            "model": model_key,
            "model_name": model_name,
            "dataset": args.dataset,
            "mode": mode,
            "explain": args.explain,
            "qd": use_qd,
            "n": n,
            "sample_count": len(questions),
            "layers": layers,
            "position": args.position,
            "activation_kind": args.activation_kind,
            "direction": direction,
            "baseline": args.baseline,
            "diffmeans": diffmeans,
            "examples": examples,
        },
        tensor_file,
    )
    print(f"Diffmean tensors saved to {tensor_file}")

    oracle_prompts = [
        "Describe this text.",
        "What belief or feature does this activation represent?",
        CONSENSUS_PROMPT,
    ]
    if not args.skip_oracle:
        print(f"Loading oracle LoRA: {args.oracle_lora_path}")
        model.load_adapter(args.oracle_lora_path, adapter_name="oracle", is_trainable=False)
        print("Oracle loaded successfully")

    layer_results = []
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
        "dataset": args.dataset,
        "mode": mode,
        "explain": args.explain,
        "qd": use_qd,
        "n": n,
        "sample_count": len(questions),
        "seed": args.seed,
        "layers_swept": layers,
        "questions_file": str(questions_file),
        "tensor_file": str(tensor_file),
        "position": args.position,
        "activation_kind": args.activation_kind,
        "direction": direction,
        "baseline": args.baseline,
        "condition_a": (
            "prior participants mostly give the same wrong answer; one deviant prior participant uses the multi_actor da answer"
            if use_da
            else "all prior participants give the same wrong answer"
        ),
        "condition_b": (
            "all prior participants are shown with independently random answers"
            if args.baseline == "random_answers"
            else (
                "baseline prompt is identical to the condition prompt"
                if args.baseline == "same_prompt"
                else "no other participants are shown; neutral prompt"
            )
        ),
        "examples": examples,
        "layers": layer_results,
    }

    json_file = model_dir / f"{run_slug}.json"
    json_file.write_text(json.dumps(results, indent=2))
    analysis_file = model_dir / f"{run_slug}_analysis.md"
    write_analysis(results, analysis_file)
    print(f"Layer sweep results saved to {json_file}")
    print(f"Layer sweep analysis saved to {analysis_file}")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)

    if args.sample_experiments is not None:
        print("--sample-experiments is accepted for config parity; synthetic layer_sweep runs one generated sweep.")

    manifest = {
        "seed": args.seed,
        "dataset": args.dataset,
        "limit": args.limit,
        "activation_kind": args.activation_kind,
        "baseline": args.baseline,
        "position": args.position,
        "models": {},
    }
    for model_key in [item.strip() for item in args.models.split(",") if item.strip()]:
        if model_key not in MODEL_IDS:
            raise ValueError(f"Unknown model key {model_key!r}; choose from {sorted(MODEL_IDS)}")
        result = process_model(model_key, args)
        manifest["models"][model_key] = result

    manifest_file = args.output_dir / "manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2))
    print(f"Manifest saved to {manifest_file}")


if __name__ == "__main__":
    main()
