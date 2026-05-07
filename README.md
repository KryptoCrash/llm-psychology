# LLM Psychology Reproducibility

This branch contains the code needed to reproduce the NeurIPS paper results and
only the minimal checked-in data that the code reads directly:
`combined_mmlu_correct.json` and `combined_bbh_correct.json`. Experiment
outputs, caches, activation tensors, generated figures, and paper summary files
are intentionally not tracked; the commands below recreate them.

## Setup

Use Python 3.11 or newer. A GPU is expected for Qwen3-8B model runs.

```bash
uv sync
source .venv/bin/activate
```

If Hugging Face requires credentials for model or adapter downloads:

```bash
huggingface-cli login
```

## Compute and Runtime

The reported experiments were run on a node with 2x NVIDIA H100 GPUs.
Approximate wall-clock runtimes are:

- Behavioral inference: 1 hour.
- Activation patching and steering: 2.5 hours.
- Activation caching, DiffMean construction, and activation-oracle runs: 1 hour.
- All remaining analyses and figure generation: under 30 minutes each.

## Licenses and Upstream Assets

This repository's code and derived paper artifacts are released under the MIT
License; see `LICENSE`. Upstream assets retain their own licenses and terms:

- `Qwen/Qwen3-8B`: Apache-2.0.
- MMLU (`hendrycks/test`): MIT License.
- BIG-Bench Hard (`suzgunmirac/BIG-Bench-Hard`): MIT License.
- `adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B`: used as a
  Hugging Face adapter checkpoint under its upstream model-card terms; the
  release does not redistribute the adapter weights.

## Behavioral Experiments

The paper evaluates `Qwen/Qwen3-8B` on 100 MMLU and 100 BBH items that the model
answers correctly without social context. The fixed item subsets are
`combined_mmlu_correct.json` and `combined_bbh_correct.json`.

Regenerate all 48 behavioral runs:

```bash
python qwen_experiment.py
```

Regenerate one condition:

```bash
python multi_actor.py --model qwen --dataset mmlu --mode 10 --output-file qwen_mmlu_10_base.json
python multi_actor.py --model qwen --dataset bbh --mode da --explain --output-file qwen_bbh_da_explain.json
```

Regenerate behavioral tables, confidence intervals, and paper figures:

```bash
python data_analysis/bootstrap_qwen_results.py
python data_analysis/plot_qwen_participants.py
```

Outputs:

- `paper/generated/qwen_bootstrap_results.json`
- `paper/generated/qwen_condition_bootstrap.csv`
- `paper/generated/qwen_contrast_tests.csv`
- `paper/generated/qwen_trend_bootstrap.csv`
- `paper/figures/qwen_conformity_by_participants.{pdf,png}`
- `paper/figures/qwen_accuracy_by_participants.{pdf,png}`
- `paper/figures/qwen_mitigation_conformity.{pdf,png}`

## Diffmean Directions and Activation Oracles

The main activation direction is the MMLU `n=8`
`all_wrong_minus_random_answers` residual diffmean at assistant start. The paper
also uses the MMLU `n=10` residual diffmean as a projection check.

Recompute the diffmean tensors:

```bash
python layer_sweep.py --models qwen --dataset mmlu --mode 8 --limit 100 --activation-kind residual --position assistant_start --baseline random_answers --output-dir layer_sweep_results
python layer_sweep.py --models qwen --dataset mmlu --mode 10 --limit 100 --activation-kind residual --position assistant_start --baseline random_answers --output-dir layer_sweep_results
python layer_sweep.py --models qwen --dataset bbh --mode 4 --limit 100 --activation-kind residual --position assistant_start --baseline random_answers --output-dir layer_sweep_results
```

Recompute the activation-oracle runs used in the paper:

```bash
python layer_sweep.py --models qwen --dataset mmlu --mode 8 --limit 100 --activation-kind residual --position assistant_start --baseline random_answers --run-oracle --output-dir layer_sweep_results
python layer_sweep.py --models qwen --dataset mmlu --mode 8 --limit 100 --activation-kind residual --position assistant_start --baseline no_participants --run-oracle --output-dir layer_sweep_results
python layer_sweep.py --models qwen --dataset mmlu --mode 8 --limit 100 --activation-kind delta --position assistant_start --baseline random_answers --run-oracle --output-dir layer_sweep_results
python layer_sweep.py --models qwen --dataset mmlu --mode 8 --limit 100 --activation-kind delta --position assistant_start --baseline no_participants --run-oracle --output-dir layer_sweep_results
```

Summarize oracle outputs for the paper:

```bash
python data_analysis/summarize_activation_oracle_outputs.py
```

Output:

- `paper/generated/qwen_activation_oracle_summary.json`

## Projection Analysis

Projection analysis computes mean layer-23 assistant-start residual activations
for each saved Qwen condition and projects each condition mean onto the MMLU
diffmean direction.

```bash
python project_experiment_activations.py \
  --reference layer_sweep_results/qwen/layer_sweep_qwen_mmlu_mode_8_base_residual_assistant_start_all_wrong_minus_random_answers_n8_100_diffmeans.pt \
  --runs-dir . \
  --output-dir projection_results \
  --layer 23

python project_experiment_activations.py \
  --reference layer_sweep_results/qwen/layer_sweep_qwen_mmlu_mode_10_base_residual_assistant_start_all_wrong_minus_random_answers_n10_100_diffmeans.pt \
  --runs-dir . \
  --output-dir projection_results \
  --layer 23

python data_analysis/projection_correlation_analysis.py
```

Outputs:

- `projection_results/qwen_experiment_mean_activation_projection_layer23_onto_*.json`
- `paper/generated/qwen_projection_correlation_summary.json`
- `paper/generated/qwen_projection_correlations.csv`
- `paper/generated/qwen_projection_prompt_deltas.csv`

The projection script writes activation caches under `projection_results/cache/`.
Pass `--overwrite` to recompute activations from model forward passes.

## Layer-Wise Cosine Structure

The layer-wise cosine structure section uses cosine similarities between
diffmean vectors, primarily the MMLU `n=8` residual diffmeans.

```bash
python analyze_qwen_diffmean_cosines.py \
  --source-dirs layer_sweep_results/qwen \
  --output-dir diffmean_cosine_results/qwen_layer_sweep_recomputed \
  --neighbor-k 5 \
  --top-k 25
```

Outputs:

- `diffmean_cosine_results/qwen_layer_sweep_recomputed/analysis.md`
- `diffmean_cosine_results/qwen_layer_sweep_recomputed/layer_topk_neighbors.csv`
- `diffmean_cosine_results/qwen_layer_sweep_recomputed/layer_topk_neighbors.json`
- `diffmean_cosine_results/qwen_layer_sweep_recomputed/layer_matrices/`

The plotted values in the paper are copied from these layer-neighbor results
into `paper/sections/qwen_activation_projection_section.tex`.

## Activation Steering

The paper heatmap is generated from the single-layer steering sweep. The
multi-layer steering table is generated from the `qwen_*_ablation/` all-layer
runs. These commands write regenerated outputs under
`runs/activation_patching/`, which is intentionally ignored because it contains
large derived JSON files.

Single-layer steering sweep, using the MMLU `n=8` diffmean tensor:

```bash
VECTOR=layer_sweep_results/qwen/layer_sweep_qwen_mmlu_mode_8_base_residual_assistant_start_all_wrong_minus_random_answers_n8_100_diffmeans.pt

python run_steering.py --model qwen --dataset mmlu --vector-path "$VECTOR" --layers $(seq 0 35) --alphas -4 -2 -1 1 2 4 --output-dir runs/activation_patching/qwen_mmlu --seed 42
python run_steering.py --model qwen --dataset bbh --vector-path "$VECTOR" --layers $(seq 0 35) --alphas -4 -2 -1 1 2 4 --output-dir runs/activation_patching/qwen_bbh --seed 42
```

All-layer single-vector steering with the layer-23 MMLU diffmean:

```bash
python run_ablation.py --dataset mmlu --mode 3 --alphas 0.1 --layer 23 --baseline --output-dir runs/activation_patching/qwen_mmlu_ablation --seed 42
python run_ablation.py --dataset mmlu --mode 10 --alphas -0.1 --layer 23 --baseline --output-dir runs/activation_patching/qwen_mmlu_ablation --seed 42
python run_ablation.py --dataset bbh --mode 3 --alphas 0.1 --layer 23 --baseline --output-dir runs/activation_patching/qwen_bbh_ablation --seed 42
python run_ablation.py --dataset bbh --mode 10 --alphas -0.1 --layer 23 --baseline --output-dir runs/activation_patching/qwen_bbh_ablation --seed 42
```

Copy or regenerate the paper heatmap before building the paper:

- `paper/figures/fig2_heatmap_dconf.png`

## Build the Paper

```bash
cd paper
latexmk -pdf main.tex
```

The main paper source is `paper/main.tex`; section files are in
`paper/sections/`.
