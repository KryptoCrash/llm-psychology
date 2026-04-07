# pyright: reportMissingImports=false
from __future__ import annotations

import argparse
from typing import Any

try:
    from .utils import QuestionRecord, base_prompt, conformity_prompt, load_questions
except ImportError:
    from utils import QuestionRecord, base_prompt, conformity_prompt, load_questions


DEFAULT_MODEL_NAME = "Qwen/Qwen3-8B"


def default_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def default_dtype(device: str) -> Any:
    import torch

    if device == "cuda":
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    return torch.float32


def load_qwen3_model(
    model_name: str = DEFAULT_MODEL_NAME,
    device: str | None = None,
    dtype: Any = None,
    default_prepend_bos: bool | None = None,
    no_processing: bool = False,
) -> Any:
    from transformer_lens import HookedTransformer

    resolved_device = device or default_device()
    resolved_dtype = dtype or default_dtype(resolved_device)
    load_fn = (
        HookedTransformer.from_pretrained_no_processing
        if no_processing
        else HookedTransformer.from_pretrained
    )
    return load_fn(
        model_name,
        device=resolved_device,
        dtype=resolved_dtype,
        default_padding_side="left",
        default_prepend_bos=default_prepend_bos,
    )


def complete_prompt(
    model: Any,
    prompt: str,
    *,
    max_new_tokens: int = 8,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float = 1.0,
    stop_at_eos: bool = True,
    prepend_bos: bool | None = None,
) -> str:
    effective_temperature = temperature
    effective_top_k = top_k
    if temperature <= 0:
        effective_temperature = 1.0
        effective_top_k = 1 if top_k is None else top_k

    generated = model.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=effective_temperature,
        top_k=effective_top_k,
        top_p=top_p,
        stop_at_eos=stop_at_eos,
        prepend_bos=prepend_bos,
        return_type="str",
        verbose=False,
    )
    if not isinstance(generated, str):
        raise TypeError("Expected HookedTransformer.generate to return a string.")

    if generated.startswith(prompt):
        return generated[len(prompt) :].strip()
    return generated.strip()


def complete_base_prompt(
    model: Any, question: QuestionRecord, **generation_kwargs: Any
) -> str:
    return complete_prompt(model, base_prompt(question), **generation_kwargs)


def complete_conformity_prompt(
    model: Any,
    question: QuestionRecord,
    k: int,
    previous_answers: list[str] | None = None,
    **generation_kwargs: Any,
) -> str:
    prompt = conformity_prompt(question, k, previous_answers)
    return complete_prompt(model, prompt, **generation_kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Qwen3-8B completions on dataset-backed prompts."
    )
    parser.add_argument(
        "--mode",
        choices=["base", "conformity"],
        default="base",
        help="Which prompt style to use.",
    )
    parser.add_argument(
        "--question-index",
        type=int,
        default=0,
        help="Index into data.json.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Participant count/current participant for conformity prompts.",
    )
    parser.add_argument(
        "--previous-answer",
        action="append",
        default=[],
        help="Previous participant answer. Pass multiple times for multiple participants.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=8,
        help="Maximum tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Use 0.0 for greedy-ish decoding.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="TransformerLens/Hugging Face model name.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Explicit device override, e.g. cpu, cuda, or mps.",
    )
    parser.add_argument(
        "--no-processing",
        action="store_true",
        help="Disable TransformerLens preprocessing when loading the model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions = load_questions()
    question = questions[args.question_index]
    model = load_qwen3_model(
        model_name=args.model_name,
        device=args.device,
        no_processing=args.no_processing,
    )

    if args.mode == "base":
        prompt = base_prompt(question)
        completion = complete_base_prompt(
            model,
            question,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
    else:
        prompt = conformity_prompt(question, args.k, args.previous_answer)
        completion = complete_conformity_prompt(
            model,
            question,
            args.k,
            args.previous_answer,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

    print(prompt)
    print()
    print(completion)


if __name__ == "__main__":
    main()
