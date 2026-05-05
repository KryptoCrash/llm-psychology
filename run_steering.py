"""Run activation steering on Qwen3-8B.

Prompts file format: plain text, one prompt per line. Blank lines are ignored.
Output format: JSONL, one object per prompt with keys {"prompt", "completion",
"layer", "alpha"}.
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from steering import steer

MODEL_NAME = "Qwen/Qwen3-8B"


def load_prompts(path: Path) -> list[str]:
    lines = path.read_text().splitlines()
    return [ln for ln in (l.strip() for l in lines) if ln]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vector-path", type=Path, required=True,
                    help="Path to a .pt file containing a single tensor of shape [4096].")
    ap.add_argument("--prompts-path", type=Path, required=True,
                    help="Plain text file, one prompt per line.")
    ap.add_argument("--layer", type=int, required=True, help="Layer index in 0..35.")
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--output-path", type=Path, required=True)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    if not (0 <= args.layer <= 35):
        raise ValueError(f"--layer must be in 0..35, got {args.layer}")

    vector = torch.load(args.vector_path, map_location="cpu")
    if not isinstance(vector, torch.Tensor):
        raise TypeError(f"Expected a torch.Tensor in {args.vector_path}, got {type(vector)}")
    vector = vector.squeeze()
    if vector.ndim != 1 or vector.shape[0] != 4096:
        raise ValueError(f"Expected vector of shape [4096], got {tuple(vector.shape)}")

    prompts = load_prompts(args.prompts_path)
    if not prompts:
        raise ValueError(f"No prompts found in {args.prompts_path}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    completions = steer(
        model=model,
        tokenizer=tokenizer,
        layer=args.layer,
        vector=vector,
        alpha=args.alpha,
        prompts=prompts,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w") as f:
        for prompt, completion in zip(prompts, completions):
            f.write(json.dumps({
                "prompt": prompt,
                "completion": completion,
                "layer": args.layer,
                "alpha": args.alpha,
            }) + "\n")

    print(f"Wrote {len(completions)} completions to {args.output_path}")


if __name__ == "__main__":
    main()
