"""Ablation: hook every transformer layer of qwen with a diffmean vector.

Two modes:
  * --layer L (0..35): use diffmeans[L] as a CONSTANT vector applied at every
    transformer block (layer L's vector subtracted from every layer's residual).
  * no --layer:        per-layer vectors — each block i is hooked with
    diffmeans[i] (default).

For each alpha, one ablation pass is run. The matching baseline (alpha=0) is
opt-in via --baseline.

Output filenames written under --output-dir:
  --layer L:   qwen_{dataset}_{mode}_ablation_layer{L}_alpha{a:g}.json
  per-layer:   qwen_{dataset}_{mode}_ablation_perlayer_alpha{a:g}.json
  baseline:    qwen_{dataset}_{mode}_baseline.json

Existing files are skipped. A baseline is also considered to exist if any file
matching qwen_{dataset}_10_*baseline*.json is already present.
"""
import argparse
import json
from contextlib import contextmanager
from pathlib import Path

import torch

from multi_actor import MODEL_IDS, CausalLMRunner, run_experiment
from steering import make_resid_add_hook


# Default experiment parameters from the original patching branch.
MODEL_NAME = "qwen"
DEFAULT_MODE = "10"
DEFAULT_ALPHAS = [-0.15, -0.2]
MAX_LAYER = 35  # qwen3-8B has 36 transformer blocks (0..35)
VECTOR_PATH = Path(
    "layer_sweep_results/qwen/"
    "layer_sweep_qwen_mmlu_mode_8_base_residual_assistant_start_all_wrong_minus_random_answers_n8_100_diffmeans.pt"
)


def fmt_alpha(a: float) -> str:
    """1.0→'1', 2.0→'2', 0.5→'0.5', -1.5→'-1.5'."""
    return f"{a:g}"


def ablation_path(out_dir: Path, dataset: str, mode: str, alpha: float, source_layer: int | None) -> Path:
    """`source_layer is None` → per-layer-vectors filename;
    integer → constant-vector-from-layer-{L} filename."""
    if source_layer is None:
        suffix = "perlayer"
    else:
        suffix = f"layer{source_layer}"
    return out_dir / f"{MODEL_NAME}_{dataset}_{mode}_ablation_{suffix}_alpha{fmt_alpha(alpha)}.json"


def baseline_path(out_dir: Path, dataset: str, mode: str) -> Path:
    return out_dir / f"{MODEL_NAME}_{dataset}_{mode}_baseline.json"


def find_existing_baseline(out_dir: Path, dataset: str, mode: str) -> Path | None:
    """Match canonical {mode}_baseline.json or any legacy {mode}_*baseline*.json."""
    canonical = baseline_path(out_dir, dataset, mode)
    if canonical.exists():
        return canonical
    matches = sorted(out_dir.glob(f"{MODEL_NAME}_{dataset}_{mode}_*baseline*.json"))
    return matches[0] if matches else None


def load_diffmeans_dict(path: Path) -> dict[int, torch.Tensor]:
    """Load the layer→tensor dict once. Validates every tensor is shape [4096]."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not (isinstance(obj, dict) and "diffmeans" in obj):
        raise ValueError(
            f"{path} must be a dict with a 'diffmeans' key (layer-sweep format)."
        )
    out: dict[int, torch.Tensor] = {}
    for L, v in obj["diffmeans"].items():
        v = v.squeeze()
        if v.ndim != 1 or v.shape[0] != 4096:
            raise ValueError(f"layer {L}: expected shape [4096], got {tuple(v.shape)}")
        out[int(L)] = v
    return out


@contextmanager
def per_layer_ablation_hook(runner: CausalLMRunner, diffmeans: dict[int, torch.Tensor], alpha: float):
    """Register a residual-add hook on every transformer block of
    runner.model.model.layers, where each block L adds `alpha * diffmeans[L]`
    to its residual stream. All hooks are removed on context exit.

    Raises KeyError if any layer index is missing from the diffmeans dict.
    """
    layers = runner.model.model.layers
    missing = [i for i in range(len(layers)) if i not in diffmeans]
    if missing:
        raise KeyError(f"diffmeans is missing vectors for layers {missing}; "
                       f"have {sorted(diffmeans)}")
    handles = []
    try:
        for i, layer_module in enumerate(layers):
            handles.append(layer_module.register_forward_hook(make_resid_add_hook(diffmeans[i], alpha)))
        yield runner
    finally:
        for h in handles:
            h.remove()


def annotate_summary(path: Path, alpha: float, n_layers: int, source_layer: int | None) -> None:
    """Re-open the JSON written by run_experiment and add ablation metadata
    (run_experiment doesn't know about the hook we wrap it in)."""
    with path.open("r") as f:
        summary = json.load(f)
    new_summary = {}
    for k, v in summary.items():
        if k == "results":
            continue
        new_summary[k] = v
    steering = {
        "layer": "all",
        "alpha": alpha,
        "n_layers_hooked": n_layers,
    }
    if source_layer is None:
        steering["vectors"] = "per-layer diffmean (layer L hooked with diffmeans[L])"
    else:
        steering["vectors"] = f"constant: diffmeans[{source_layer}] applied at every layer"
        steering["source_layer"] = source_layer
    new_summary["steering"] = steering
    new_summary["results"] = summary.get("results", [])
    with path.open("w") as f:
        json.dump(new_summary, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=Path("runs"),
                    help="Directory for output JSONs (default: runs/).")
    ap.add_argument("--dataset", choices=["mmlu", "bbh"], default="mmlu")
    ap.add_argument("--mode", choices=["3", "10"], default=DEFAULT_MODE)
    ap.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS)
    ap.add_argument("--explain", action="store_true")
    ap.add_argument("--seed", type=int, default=42,
                    help="Torch RNG seed (same value used for every run for fair comparison).")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing output files instead of skipping them.")
    ap.add_argument("--baseline", action="store_true",
                    help="Also run the matching unsteered baseline (alpha=0). Off by default.")
    ap.add_argument("--layer", type=int, default=None,
                    help=f"If set (0..{MAX_LAYER}), apply diffmeans[--layer] as a CONSTANT "
                         f"vector at every transformer block. If omitted, each block i is "
                         f"hooked with its own diffmeans[i] (per-layer ablation).")
    args = ap.parse_args()

    if args.layer is not None and not (0 <= args.layer <= MAX_LAYER):
        raise ValueError(f"--layer must be in 0..{MAX_LAYER} for {MODEL_NAME}, got {args.layer}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    diffmeans = load_diffmeans_dict(VECTOR_PATH)
    if args.layer is not None and args.layer not in diffmeans:
        raise KeyError(f"--layer {args.layer} not in diffmeans (available: {sorted(diffmeans)})")

    # Build the dict the hook will use: per-layer (default) or constant from --layer.
    if args.layer is None:
        effective_diffmeans = diffmeans
    else:
        effective_diffmeans = {i: diffmeans[args.layer] for i in diffmeans.keys()}

    # Plan: optional baseline (only if --baseline), then ablation runs.
    plan = []  # list of (kind, path, alpha-or-None)
    if args.baseline:
        existing = find_existing_baseline(args.output_dir, args.dataset, args.mode)
        if existing and not args.force:
            print(f"[skip] mode={args.mode} baseline already exists: {existing.name}")
        else:
            plan.append(("baseline", baseline_path(args.output_dir, args.dataset, args.mode), None))

    for a in sorted(args.alphas):
        sp = ablation_path(args.output_dir, args.dataset, args.mode, a, args.layer)
        if sp.exists() and not args.force:
            print(f"[skip] alpha={fmt_alpha(a)} mode={args.mode}: {sp.name}")
        else:
            plan.append(("ablation", sp, a))

    if not plan:
        print("\nNothing to run — all targets already exist. Use --force to overwrite.")
        return

    n_baselines = sum(1 for p in plan if p[0] == "baseline")
    n_ablations = sum(1 for p in plan if p[0] == "ablation")
    print(f"\nModel: {MODEL_NAME} ({MODEL_IDS[MODEL_NAME]})")
    if args.layer is None:
        print(f"Vectors: per-layer diffmeans from {VECTOR_PATH.name} "
              f"(layers {min(diffmeans)}..{max(diffmeans)})")
        hook_desc = "each layer L hooked with diffmeans[L] (per-layer)"
    else:
        print(f"Vectors: CONSTANT diffmeans[{args.layer}] from {VECTOR_PATH.name}")
        hook_desc = f"every layer hooked with the same diffmeans[{args.layer}]"
    print(f"Mode: {args.mode}  ·  alphas: {args.alphas}")
    print(f"Hook: {hook_desc}")
    print(f"Planned runs: {n_baselines} baseline(s) + {n_ablations} ablation = {len(plan)} total")
    label = "per-layer" if args.layer is None else f"layer{args.layer}"
    for kind, path, a in plan:
        if kind == "baseline":
            print(f"  - mode={args.mode} baseline                       → {path.name}")
        else:
            print(f"  - mode={args.mode} {label} alpha={fmt_alpha(a):<6}         → {path.name}")

    runner = CausalLMRunner(MODEL_IDS[MODEL_NAME])
    n_layers = len(runner.model.model.layers)
    print(f"\nModel has {n_layers} transformer layers (will hook all).")

    for i, (kind, path, a) in enumerate(plan, 1):
        print(f"\n[{i}/{len(plan)}]", end=" ")
        if kind == "baseline":
            print(f"BASELINE mode={args.mode} (alpha=0, seed={args.seed})")
            run_experiment(
                MODEL_NAME, args.dataset, args.mode, args.explain,
                model=runner, output_file=str(path), batch_size=args.batch_size,
                seed=args.seed,
            )
        else:
            tag = "per-layer" if args.layer is None else f"layer{args.layer}-constant"
            print(f"ABLATION mode={args.mode} {tag} alpha={a} (seed={args.seed})")
            with per_layer_ablation_hook(runner, effective_diffmeans, a):
                run_experiment(
                    MODEL_NAME, args.dataset, args.mode, args.explain,
                    model=runner, output_file=str(path), batch_size=args.batch_size,
                    seed=args.seed,
                )
            annotate_summary(path, a, n_layers, args.layer)
        print(f"  wrote {path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
