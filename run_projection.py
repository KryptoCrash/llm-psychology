"""Projection ablation: remove the conformity direction from every layer's
residual stream.

Given a .pt file (layer-sweep diffmeans format) and a layer L, load the
"conformity vector" v = diffmeans[L]. At every transformer block, replace
the residual stream h with its projection onto v's orthogonal complement:

    h ← h − (h · v̂) v̂,   v̂ = v / ‖v‖

This is the projection analogue of run_ablation.py's constant-vector mode:
the same vector (taken from layer L) is used to define the direction
removed at every block, but instead of adding α·v we subtract the
component of h along v.

Output filename written under --output-dir:
  qwen_{dataset}_{MODE}_projection_layer{L}.json

Existing files are skipped unless --force is given.
"""
import argparse
import json
from contextlib import contextmanager
from pathlib import Path

import torch

from multi_actor import MODEL_IDS, CausalLMRunner, run_experiment
from steering import make_resid_project_hook


# Hard-coded experiment parameters (mirroring run_ablation.py).
MODEL_NAME = "qwen"
MODE = "10"
MAX_LAYER = 35  # qwen3-8B has 36 transformer blocks (0..35)
VECTOR_PATH = Path(
    "layer_sweep_results/qwen/"
    "layer_sweep_qwen_mmlu_mode_8_base_residual_assistant_start_all_wrong_minus_random_answers_n8_100_diffmeans.pt"
)


def projection_path(out_dir: Path, dataset: str, source_layer: int) -> Path:
    return out_dir / f"{MODEL_NAME}_{dataset}_{MODE}_projection_layer{source_layer}.json"


def load_diffmeans_dict(path: Path) -> dict[int, torch.Tensor]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not (isinstance(obj, dict) and "diffmeans" in obj):
        raise ValueError(f"{path} must be a dict with a 'diffmeans' key (layer-sweep format).")
    out: dict[int, torch.Tensor] = {}
    for L, v in obj["diffmeans"].items():
        v = v.squeeze()
        if v.ndim != 1 or v.shape[0] != 4096:
            raise ValueError(f"layer {L}: expected shape [4096], got {tuple(v.shape)}")
        out[int(L)] = v
    return out


@contextmanager
def all_layer_projection_hook(runner: CausalLMRunner, vector: torch.Tensor):
    """Register a projection hook on every transformer block of
    runner.model.model.layers, using the same `vector` to define the
    direction removed. All hooks are removed on context exit."""
    layers = runner.model.model.layers
    handles = []
    try:
        for layer_module in layers:
            handles.append(layer_module.register_forward_hook(make_resid_project_hook(vector)))
        yield runner
    finally:
        for h in handles:
            h.remove()


def annotate_summary(path: Path, n_layers: int, source_layer: int, vector_path: Path) -> None:
    with path.open("r") as f:
        summary = json.load(f)
    new_summary = {k: v for k, v in summary.items() if k != "results"}
    new_summary["steering"] = {
        "layer": "all",
        "operation": "orthogonal_projection",
        "source_layer": source_layer,
        "vectors": f"projection: every layer's residual is projected orthogonal to diffmeans[{source_layer}]",
        "vector_path": str(vector_path),
        "n_layers_hooked": n_layers,
    }
    new_summary["results"] = summary.get("results", [])
    with path.open("w") as f:
        json.dump(new_summary, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=Path("runs"),
                    help="Directory for output JSONs (default: runs/).")
    ap.add_argument("--dataset", choices=["mmlu", "bbh"], default="mmlu")
    ap.add_argument("--explain", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing output file instead of skipping it.")
    ap.add_argument("--layer", type=int, required=True,
                    help=f"Layer index (0..{MAX_LAYER}) whose diffmean vector defines the "
                         f"conformity direction to project out at every transformer block.")
    ap.add_argument("--vector-path", type=Path, default=VECTOR_PATH,
                    help="Path to the .pt file containing the layer→vector dict.")
    args = ap.parse_args()

    if not (0 <= args.layer <= MAX_LAYER):
        raise ValueError(f"--layer must be in 0..{MAX_LAYER} for {MODEL_NAME}, got {args.layer}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    diffmeans = load_diffmeans_dict(args.vector_path)
    if args.layer not in diffmeans:
        raise KeyError(f"--layer {args.layer} not in diffmeans (available: {sorted(diffmeans)})")
    vector = diffmeans[args.layer]

    out_path = projection_path(args.output_dir, args.dataset, args.layer)
    if out_path.exists() and not args.force:
        print(f"[skip] projection already exists: {out_path.name}")
        return

    print(f"\nModel: {MODEL_NAME} ({MODEL_IDS[MODEL_NAME]})")
    print(f"Vector: diffmeans[{args.layer}] from {args.vector_path.name} "
          f"(‖v‖={vector.norm().item():.4f}, dim={vector.shape[0]})")
    print(f"Mode: {MODE}  ·  dataset: {args.dataset}  ·  seed: {args.seed}")
    print(f"Hook: orthogonal projection at every layer (removes the diffmeans[{args.layer}] direction)")
    print(f"Output: {out_path}")

    runner = CausalLMRunner(MODEL_IDS[MODEL_NAME])
    n_layers = len(runner.model.model.layers)
    print(f"\nModel has {n_layers} transformer layers (will hook all).")

    print(f"\nPROJECTION mode={MODE} layer{args.layer} (seed={args.seed})")
    with all_layer_projection_hook(runner, vector):
        run_experiment(
            MODEL_NAME, args.dataset, MODE, args.explain,
            model=runner, output_file=str(out_path), batch_size=args.batch_size,
            seed=args.seed,
        )
    annotate_summary(out_path, n_layers, args.layer, args.vector_path)
    print(f"  wrote {out_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
