"""Orchestrate a steered + baseline pair of multi_actor runs.

Loads Qwen3-8B once, picks the multi_actor mode from the sign of --alpha
(positive → '3', negative → '10'), and calls multi_actor.run_experiment
twice with the same seed: once with the steering hook attached, once without.
Two JSON files are written.

For a single steered run (no baseline) or any other mode/dataset, call
multi_actor.py directly with --vector-path/--layer/--alpha.
"""
import argparse
from pathlib import Path

from multi_actor import MODEL_IDS, CausalLMRunner, run_experiment
from steering import load_vector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vector-path", type=Path, required=True,
                    help="Path to a .pt with a [4096] tensor or a 'diffmeans' dict.")
    ap.add_argument("--layer", type=int, required=True, help="Layer index in 0..35.")
    ap.add_argument("--alpha", type=float, required=True,
                    help="Sign picks multi_actor mode (positive→'3', negative→'10'); "
                         "magnitude is the steering scale. Cannot be 0.")
    ap.add_argument("--output-path", type=Path, required=True,
                    help="Steered output JSON; baseline written to <stem>_baseline<suffix>.")
    ap.add_argument("--dataset", choices=["mmlu", "bbh"], default="mmlu")
    ap.add_argument("--explain", action="store_true")
    ap.add_argument("--normalize", action="store_true",
                    help="Divide the steering vector by its L2 norm before applying alpha.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Torch RNG seed (same value used for both runs for fair comparison).")
    ap.add_argument("--batch-size", type=int, default=None)
    args = ap.parse_args()

    if args.alpha == 0:
        raise ValueError("--alpha must be non-zero (positive→mode '3', negative→mode '10').")
    if not (0 <= args.layer <= 35):
        raise ValueError(f"--layer must be in 0..35, got {args.layer}")

    mode = "3" if args.alpha > 0 else "10"
    print(f"alpha={args.alpha} → multi_actor mode='{mode}'")

    vector = load_vector(args.vector_path, args.layer)
    if args.normalize:
        norm = vector.float().norm()
        if norm == 0:
            raise ValueError("Cannot normalize a zero vector.")
        vector = (vector.float() / norm).to(vector.dtype)

    runner = CausalLMRunner(MODEL_IDS["qwen"])

    steered_path = args.output_path
    baseline_path = steered_path.with_name(
        steered_path.stem + "_baseline" + steered_path.suffix
    )

    print(f"\n=== STEERED  layer={args.layer} alpha={args.alpha} normalize={args.normalize} ===")
    run_experiment(
        "qwen", args.dataset, mode, args.explain,
        model=runner, output_file=str(steered_path), batch_size=args.batch_size,
        steering_vector=vector, steering_layer=args.layer, steering_alpha=args.alpha,
        seed=args.seed,
    )

    print(f"\n=== BASELINE  (alpha=0, same prompts, same seed) ===")
    run_experiment(
        "qwen", args.dataset, mode, args.explain,
        model=runner, output_file=str(baseline_path), batch_size=args.batch_size,
        seed=args.seed,
    )

    print(f"\nSteered:  {steered_path}")
    print(f"Baseline: {baseline_path}")


if __name__ == "__main__":
    main()
