"""Create a compact summary of saved activation-oracle layer-sweep outputs."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_FILE = ROOT / "paper" / "generated" / "qwen_activation_oracle_summary.json"
CONSENSUS_PROMPT = (
    "Does this activation represent pressure toward an incorrect consensus? "
    "Answer only True or False."
)
BELIEF_PROMPT = "What belief or feature does this activation represent?"

SOURCE_FILES = [
    (
        "layer_sweep_results/layer_sweep_residual_all_wrong_minus_random_answers_n8_100.json",
        "layer_sweep_results/qwen/layer_sweep_qwen_mmlu_mode_8_base_residual_assistant_start_all_wrong_minus_random_answers_n8_100.json",
    ),
    (
        "layer_sweep_results/layer_sweep_residual_all_wrong_minus_no_participants_n8_100.json",
        "layer_sweep_results/qwen/layer_sweep_qwen_mmlu_mode_8_base_residual_assistant_start_all_wrong_minus_no_participants_n8_100.json",
    ),
    (
        "layer_sweep_results/layer_sweep_delta_all_wrong_minus_random_answers_n8_100.json",
        "layer_sweep_results/qwen/layer_sweep_qwen_mmlu_mode_8_base_delta_assistant_start_all_wrong_minus_random_answers_n8_100.json",
    ),
    (
        "layer_sweep_results/layer_sweep_delta_all_wrong_minus_no_participants_n8_100.json",
        "layer_sweep_results/qwen/layer_sweep_qwen_mmlu_mode_8_base_delta_assistant_start_all_wrong_minus_no_participants_n8_100.json",
    ),
]


def resolve_path(candidates: tuple[str, ...]) -> str:
    for path in candidates:
        if (ROOT / path).exists():
            return path
    raise FileNotFoundError(f"None of these files exist: {candidates}")


def load_local(candidates: tuple[str, ...]) -> tuple[str, dict]:
    path = resolve_path(candidates)
    return path, json.loads((ROOT / path).read_text())


def is_true_response(response: str) -> bool:
    return response.lower().startswith("true")


def summarize_run(candidates: tuple[str, ...]) -> dict:
    path, data = load_local(candidates)
    true_layers = [
        row["layer"]
        for row in data["layers"]
        if is_true_response(row.get("oracle_responses", {}).get(CONSENSUS_PROMPT, ""))
    ]
    top_norm_layers = sorted(
        data["layers"],
        key=lambda row: row["diffmean_norm"],
        reverse=True,
    )[:10]
    summary = {
        "source_file": path,
        "activation_kind": data["activation_kind"],
        "direction": data["direction"],
        "n": data["n"],
        "sample_count": data["sample_count"],
        "position": data["position"],
        "oracle_true_layers": true_layers,
        "top_norm_layers": [
            {
                "layer": row["layer"],
                "norm": row["diffmean_norm"],
                "oracle_consensus": is_true_response(
                    row.get("oracle_responses", {}).get(CONSENSUS_PROMPT, "")
                ),
            }
            for row in top_norm_layers
        ],
    }
    if (
        data["activation_kind"] == "residual"
        and data["direction"] == "all_wrong_minus_random_answers"
    ):
        by_layer = {row["layer"]: row for row in data["layers"]}
        summary["representative_belief_responses"] = [
            {
                "layer": layer,
                "response": by_layer[layer]["oracle_responses"][BELIEF_PROMPT],
            }
            for layer in [23, 24, 26, 31, 33]
        ]
    return summary


def main() -> None:
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    _, first = load_local(SOURCE_FILES[0])
    summary = {
        "source": "local layer_sweep_results activation-oracle outputs",
        "model_name": first["model_name"],
        "oracle_lora_path": "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B",
        "oracle_prompts": [
            "Describe this text.",
            BELIEF_PROMPT,
            CONSENSUS_PROMPT,
        ],
        "runs": [summarize_run(path) for path in SOURCE_FILES],
    }
    OUT_FILE.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
