"""Sweep multi_actor runs across one or more layers and alphas.

Loads Qwen3-8B once and the diffmeans dict from the .pt once, then for each
(layer, alpha) pair:
  - positive alpha → multi_actor mode '3'
  - negative alpha → multi_actor mode '10'
runs one steered pass. Baselines (alpha=0) are layer-independent — exactly
one run per mode is performed for the whole sweep, regardless of how many
layers are swept. All runs share --seed for fair comparison.

Output filenames written under --output-dir:
  steered:  {model}_{dataset}_{mode}_layer{L}_alpha{a:g}.json
  baseline: {model}_{dataset}_{mode}_baseline.json

Existing files are skipped (so re-running with extra layers/alphas is cheap).
A baseline is also considered to exist if any file matching
  {model}_{dataset}_{mode}_*baseline*.json
is already present, which covers legacy *_layerN_*baseline.json names.
"""
import argparse
from pathlib import Path

import torch

from multi_actor import MODEL_IDS, CausalLMRunner, run_experiment


MODEL_NAME = "qwen"  # run_steering currently only loads Qwen3-8B


def fmt_alpha(a: float) -> str:
    """1.0→'1', 2.0→'2', 0.5→'0.5', -1.5→'-1.5'."""
    return f"{a:g}"


def steered_path(out_dir: Path, dataset: str, mode: str, layer: int, alpha: float) -> Path:
    return out_dir / f"{MODEL_NAME}_{dataset}_{mode}_layer{layer}_alpha{fmt_alpha(alpha)}.json"


def baseline_path(out_dir: Path, dataset: str, mode: str) -> Path:
    return out_dir / f"{MODEL_NAME}_{dataset}_{mode}_baseline.json"


def find_existing_baseline(out_dir: Path, dataset: str, mode: str) -> Path | None:
    """Match canonical {mode}_baseline.json or any legacy {mode}_layerN_*baseline*.json."""
    canonical = out_dir / f"{MODEL_NAME}_{dataset}_{mode}_baseline.json"
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vector-path", type=Path, required=True,
                    help="Path to a .pt with a 'diffmeans' dict (layer-sweep format).")
    ap.add_argument("--layers", type=int, nargs="+", required=True,
                    help="One or more layer indices in 0..35.")
    ap.add_argument("--alphas", type=float, nargs="+", required=True,
                    help="One or more nonzero alphas. Positive→mode '3', negative→mode '10'. "
                         "One baseline per mode present (layer-independent).")
    ap.add_argument("--output-dir", type=Path, default=Path("runs"),
                    help="Directory for output JSONs (default: runs/).")
    ap.add_argument("--dataset", choices=["mmlu", "bbh"], default="mmlu")
    ap.add_argument("--explain", action="store_true")
    ap.add_argument("--seed", type=int, default=42,
                    help="Torch RNG seed (same value used for every run for fair comparison).")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing output files instead of skipping them.")
    args = ap.parse_args()

    if any(a == 0 for a in args.alphas):
        raise ValueError("--alphas must all be nonzero (positive→mode '3', negative→mode '10').")
    bad_layers = [L for L in args.layers if not (0 <= L <= 35)]
    if bad_layers:
        raise ValueError(f"--layers must all be in 0..35, got {bad_layers}")
    if len(set(args.layers)) != len(args.layers):
        raise ValueError(f"--layers contains duplicates: {args.layers}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    diffmeans = load_diffmeans_dict(args.vector_path)
    missing = [L for L in args.layers if L not in diffmeans]
    if missing:
        raise KeyError(f"layers {missing} not in diffmeans (available: {sorted(diffmeans)})")

    # Determine modes that will appear (drives which baselines we need).
    modes_present: set[str] = set()
    for a in args.alphas:
        modes_present.add("3" if a > 0 else "10")

    # Plan: baselines first (layer-independent, one per mode), then steered.
    plan = []  # list of (kind, path, mode, layer-or-None, alpha-or-None)
    for mode in sorted(modes_present):
        existing = find_existing_baseline(args.output_dir, args.dataset, mode)
        if existing and not args.force:
            print(f"[skip] mode={mode} baseline already exists: {existing.name}")
        else:
            plan.append(("baseline", baseline_path(args.output_dir, args.dataset, mode), mode, None, None))

    for L in sorted(args.layers):
        for a in sorted(args.alphas):
            mode = "3" if a > 0 else "10"
            sp = steered_path(args.output_dir, args.dataset, mode, L, a)
            if sp.exists() and not args.force:
                print(f"[skip] layer={L} alpha={fmt_alpha(a)} mode={mode}: {sp.name}")
            else:
                plan.append(("steered", sp, mode, L, a))

    if not plan:
        print("\nNothing to run — all targets already exist. Use --force to overwrite.")
        return

    n_baselines = sum(1 for p in plan if p[0] == "baseline")
    n_steered = sum(1 for p in plan if p[0] == "steered")
    print(f"\nPlanned runs: {n_baselines} baseline(s) + {n_steered} steered = {len(plan)} total")
    for kind, path, mode, L, a in plan[:30]:
        if kind == "baseline":
            print(f"  - mode={mode:<3} baseline                  → {path.name}")
        else:
            print(f"  - mode={mode:<3} layer={L:<2} alpha={fmt_alpha(a):<6}  → {path.name}")
    if len(plan) > 30:
        print(f"  ... and {len(plan) - 30} more")

    runner = CausalLMRunner(MODEL_IDS[MODEL_NAME])

    for i, (kind, path, mode, L, a) in enumerate(plan, 1):
        print(f"\n[{i}/{len(plan)}]", end=" ")
        if kind == "baseline":
            print(f"BASELINE mode={mode} (alpha=0, seed={args.seed})")
            run_experiment(
                MODEL_NAME, args.dataset, mode, args.explain,
                model=runner, output_file=str(path), batch_size=args.batch_size,
                seed=args.seed,
            )
        else:
            print(f"STEERED mode={mode} layer={L} alpha={a} seed={args.seed}")
            run_experiment(
                MODEL_NAME, args.dataset, mode, args.explain,
                model=runner, output_file=str(path), batch_size=args.batch_size,
                steering_vector=diffmeans[L], steering_layer=L, steering_alpha=a,
                seed=args.seed,
            )
        print(f"  wrote {path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
