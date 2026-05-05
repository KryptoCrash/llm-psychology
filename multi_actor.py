import argparse
import json
import random
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import prompts

MODEL_IDS = {
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen": "Qwen/Qwen3-8B",
}

LETTER = prompts.LETTER


class CausalLMRunner:
    def __init__(self, model_id, device=None):
        self.device = device or get_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.padding_side = "left"
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.generation_config = self.model.generation_config
        self.generation_config.pad_token_id = self.tokenizer.pad_token_id

    def generate(self, input_ids, attention_mask=None, max_new_tokens=50, do_sample=True, **kwargs):
        with torch.inference_mode():
            sequences = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
        return {"sequences": sequences}


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_name, device=None):
    torch.set_grad_enabled(False)
    device = device or get_device()
    model = CausalLMRunner(MODEL_IDS[model_name], device=device)
    return model, device


def batched(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def parse_answer(dataset, response):
    if dataset == "mmlu":
        m = re.search(r"<answer>\s*([A-Za-z])\s*</answer>", response)
        return m.group(1).upper() if m else ""
    m = re.search(r"<answer>(.*?)</answer>", response, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else ""


def run_experiment(model_name, dataset, mode, explain=False, model=None, device=None, output_file=None, batch_size=None):
    n, _, use_da = prompts.parse_mode(mode)
    if model is None:
        model, device = load_model(model_name, device=device)
    elif device is None:
        device = get_device()
    if batch_size is None:
        batch_size = 8 if explain else 16

    questions = prompts.load_combined_questions(dataset)

    rng = random.Random(42)
    tokenizer = model.tokenizer
    assert tokenizer is not None, "The loaded model must expose a tokenizer."

    def token_strings(token_ids):
        return tokenizer.convert_ids_to_tokens(token_ids)

    def decode_tokens(token_ids):
        return tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    correct = 0
    parseable = 0
    examples = []

    for index, row in enumerate(questions):
        prompt_info = prompts.build_experiment_prompt(row, dataset, mode, explain, rng)
        choices = prompt_info["choices"]
        ground_truth_idx = prompt_info["ground_truth_idx"]
        ground_truth = prompt_info["ground_truth"]
        prompt = prompt_info["prompt"]
        main_wrong = prompt_info["main_wrong"]
        da_pos = prompt_info["da_position"]
        da_ans = prompt_info["da_answer"]

        prompt_token_ids = tokenizer(prompt)["input_ids"]
        examples.append({
            "index": index,
            "row": row,
            "choices": choices,
            "ground_truth_idx": ground_truth_idx,
            "ground_truth": ground_truth,
            "prompt": prompt,
            "prompt_token_ids": prompt_token_ids,
            "prompt_tokens": token_strings(prompt_token_ids),
            "main_wrong": main_wrong,
            "da_position": da_pos,
            "da_answer": da_ans,
            "generation_attempts": 0,
            "model_answer": "",
            "response_token_ids": [],
            "response_tokens": [],
            "raw_response": "",
        })

    max_tokens = 300 if explain else 50
    pending = examples
    for _ in range(10):
        if not pending:
            break
        next_pending = []
        pending_by_length = sorted(pending, key=lambda example: len(example["prompt_token_ids"]))
        for batch in batched(pending_by_length, batch_size):
            prompts = [example["prompt"] for example in batch]
            model_inputs = tokenizer(prompts, return_tensors="pt", padding=True)
            model_inputs = {key: value.to(device) for key, value in model_inputs.items()}
            prompt_width = model_inputs["input_ids"].shape[1]
            generation_output = model.generate(
                model_inputs["input_ids"],
                attention_mask=model_inputs.get("attention_mask"),
                max_new_tokens=max_tokens,
                do_sample=True,
                output_logits=True,
            )
            sequences = (
                generation_output["sequences"]
                if isinstance(generation_output, dict)
                else generation_output.sequences
            )
            for batch_index, example in enumerate(batch):
                example["generation_attempts"] += 1
                response_token_ids = sequences[batch_index, prompt_width:].detach().cpu().tolist()
                response = decode_tokens(response_token_ids)
                answer = parse_answer(dataset, response)
                example["response_token_ids"] = response_token_ids
                example["response_tokens"] = token_strings(response_token_ids)
                example["raw_response"] = response.strip()
                example["model_answer"] = answer
                if not answer:
                    next_pending.append(example)
        pending = next_pending

    records = []
    for example in sorted(examples, key=lambda item: item["index"]):
        row = example["row"]
        answer = example["model_answer"]
        ground_truth = example["ground_truth"]
        if answer:
            parseable += 1

        if dataset == "mmlu":
            is_correct = answer == ground_truth
        else:
            is_correct = answer.lower() == ground_truth.lower()

        if is_correct:
            correct += 1

        record = {
            "subject": row["subject"],
            "question": row["question"],
            "prompt": example["prompt"],
            "prompt_token_ids": example["prompt_token_ids"],
            "prompt_tokens": example["prompt_tokens"],
            "generation_attempts": example["generation_attempts"],
            "model_answer": answer,
            "is_correct": is_correct,
            "response_token_ids": example["response_token_ids"],
            "response_tokens": example["response_tokens"],
            "raw_response": example["raw_response"],
        }
        if dataset == "mmlu":
            record["choices"] = example["choices"]
            record["answer"] = example["ground_truth_idx"]
        else:
            record["target"] = ground_truth
        if n >= 2:
            record["main_wrong"] = example["main_wrong"]
            if use_da:
                record["da_position"] = example["da_position"]
                record["da_answer"] = example["da_answer"]
            if dataset == "mmlu":
                record["conforms"] = (answer == example["main_wrong"]) if answer else None
            else:
                record["conforms"] = (answer.lower() == example["main_wrong"].lower()) if answer else None
        records.append(record)

    attempts = len(records)
    accuracy = correct / parseable if parseable else 0.0

    summary = {
        "model": model_name,
        "dataset": dataset,
        "mode": mode,
        "n": n,
        "accuracy": accuracy,
        "attempts": attempts,
        "parseable": parseable,
        "correct": correct,
    }
    if n >= 2:
        conformed = [r for r in records if r.get("conforms") is True]
        summary["conformity_rate"] = len(conformed) / parseable if parseable else 0.0
    summary["results"] = records

    if output_file is None:
        explain_str = "explain" if explain else "base"
        output_file = f"{model_name}_{dataset}_{mode}_{explain_str}.json"

    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"model={model_name} dataset={dataset} mode={mode} explain={explain} | {correct} correct / {parseable} parseable (accuracy={accuracy:.3f})")
    print(f"Full results saved to {output_file}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", required=True,
        help="1-10: n participants (no qd); qd: 10 participants with question distillation; da: 10 participants devil's advocate",
    )
    parser.add_argument("--model", choices=["llama", "qwen"], default="llama")
    parser.add_argument("--dataset", choices=["mmlu", "bbh"], default="mmlu")
    parser.add_argument("--explain", action="store_true", help="Chain-of-thought: model reasons step-by-step before answering")
    parser.add_argument("--batch-size", type=int, default=None, help="Number of prompts to generate at once")
    args = parser.parse_args()

    try:
        prompts.parse_mode(args.mode)
    except ValueError as exc:
        parser.error(str(exc))

    run_experiment(args.model, args.dataset, args.mode, args.explain, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
