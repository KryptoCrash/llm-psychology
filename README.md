# Mechanistic Interpretability of Conformity in LLMs

Code, data, and results for the paper *Mechanistic Interpretability of
Conformity in LLMs*. We study peer-majority conformity in `Qwen/Qwen3-8B` on
MMLU and BIG-Bench Hard, find a late-middle-layer residual-stream DiffMean
direction associated with incorrect consensus pressure, and show that steering
along it both increases and reduces conformity.

## Repository Layout

```
data/                        Fixed 100-item MMLU and BBH subsets used in every experiment
item_selection/              Scripts that built data/ (no-context answering + intersection)
multi_actor.py               Conformity prompts, generation, and scoring (core library)
run_behavioral.py            Runs all 48 behavioral conditions
layer_sweep.py               DiffMean directions and activation-oracle queries
oracle_utils.py              Activation-oracle helpers
project_experiment_activations.py   Layer-23 condition-mean projections
analyze_qwen_diffmean_cosines.py    Layer-wise DiffMean cosine structure
top_projection_examples.py   Items with the largest projection onto a DiffMean
steering.py                  Residual-stream steering hooks
run_steering.py              Per-layer steering sweep (Experiment 1)
run_ablation.py              All-layer single-vector steering (Experiment 2)
data_analysis/               Bootstrap statistics, paper plots, oracle/projection summaries
results/
  behavioral/{mmlu,bbh}/     48 behavioral runs (n=1..10, QD, DA; base and explain)
  layer_sweep/               DiffMean tensors and activation-oracle outputs
  projection/                Layer-23 projection results
  activation_patching/       Steering figures and all-layer steering runs
paper/                       LaTeX source, figures, and generated tables
```

## Paper-to-Code Map

| Paper result | Code | Checked-in output |
| --- | --- | --- |
| Figs. 1-3, conformity statistics (Sec. 3) | `run_behavioral.py`, `data_analysis/bootstrap_qwen_results.py`, `data_analysis/plot_qwen_participants.py` | `results/behavioral/`, `paper/generated/`, `paper/figures/` |
| DiffMean directions (Sec. 4) | `layer_sweep.py` | `results/layer_sweep/qwen/*_diffmeans.pt` |
| Activation oracles, Fig. 4 (Sec. 4) | `layer_sweep.py --run-oracle`, `data_analysis/summarize_activation_oracle_outputs.py` | `results/layer_sweep/layer_sweep_{residual,delta}_*.json`, `paper/generated/qwen_activation_oracle_summary.json` |
| Projection analysis, Table 1 (Sec. 4) | `project_experiment_activations.py`, `data_analysis/projection_correlation_analysis.py` | `results/projection/`, `paper/generated/qwen_projection_*` |
| Layer-wise cosine structure, Fig. 5 (Sec. 4) | `analyze_qwen_diffmean_cosines.py` | values in `paper/sections/qwen_activation_projection_section.tex` |
| Per-layer steering heatmap, Fig. 6 (Sec. 5) | `run_steering.py` | `paper/figures/fig2_heatmap_dconf.png`, `results/activation_patching/figures/` |
| All-layer steering, Table 2 (Sec. 5) | `run_ablation.py` | `results/activation_patching/qwen_{mmlu,bbh}_ablation/` |

The raw per-layer, per-alpha steering sweep JSONs (~480 MB) are not checked in;
regenerate them with `run_steering.py` as shown below.

## Setup

Use Python 3.11 or newer. A GPU is expected for Qwen3-8B model runs. Run every
command from the repository root.

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
- `meta-llama/Llama-3.1-8B-Instruct` (item selection only): Llama 3.1 Community License.
- MMLU (`hendrycks/test`): MIT License.
- BIG-Bench Hard (`suzgunmirac/BIG-Bench-Hard`): MIT License.
- `adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B`: used as a
  Hugging Face adapter checkpoint under its upstream model-card terms; the
  release does not redistribute the adapter weights.

## Item Selection

`data/combined_mmlu_correct.json` and `data/combined_bbh_correct.json` hold the
100 MMLU and 100 BBH items used in every experiment. Each item was answered
correctly, with no social context, by both `Qwen/Qwen3-8B` and
`meta-llama/Llama-3.1-8B-Instruct`. To rebuild them:

```bash
python item_selection/build_item_sets.py
```

This answers successive 200-question batches with
`item_selection/answer_without_context.py` and appends the intersection of the
two models' correct answers until each set has 100 items. Per-batch outputs go
to `results/item_selection/` (not tracked).

## Behavioral Experiments

Regenerate all 48 behavioral runs (written to `results/behavioral/{mmlu,bbh}/`):

```bash
python run_behavioral.py
```

Regenerate one condition:

```bash
python multi_actor.py --model qwen --dataset mmlu --mode 10
python multi_actor.py --model qwen --dataset bbh --mode da --explain
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

## DiffMean Directions and Activation Oracles

The main activation direction is the MMLU `n=8`
`all_wrong_minus_random_answers` residual DiffMean at the assistant-start token.
The paper also uses the MMLU `n=10` residual DiffMean as a projection check.

Recompute the DiffMean tensors (written to `results/layer_sweep/qwen/`):

```bash
python layer_sweep.py --models qwen --dataset mmlu --mode 8 --limit 100 --activation-kind residual --position assistant_start --baseline random_answers
python layer_sweep.py --models qwen --dataset mmlu --mode 10 --limit 100 --activation-kind residual --position assistant_start --baseline random_answers
python layer_sweep.py --models qwen --dataset bbh --mode 4 --limit 100 --activation-kind residual --position assistant_start --baseline random_answers
```

Recompute the activation-oracle runs used in the paper:

```bash
python layer_sweep.py --models qwen --dataset mmlu --mode 8 --limit 100 --activation-kind residual --position assistant_start --baseline random_answers --run-oracle
python layer_sweep.py --models qwen --dataset mmlu --mode 8 --limit 100 --activation-kind residual --position assistant_start --baseline no_participants --run-oracle
python layer_sweep.py --models qwen --dataset mmlu --mode 8 --limit 100 --activation-kind delta --position assistant_start --baseline random_answers --run-oracle
python layer_sweep.py --models qwen --dataset mmlu --mode 8 --limit 100 --activation-kind delta --position assistant_start --baseline no_participants --run-oracle
```

Summarize oracle outputs for the paper:

```bash
python data_analysis/summarize_activation_oracle_outputs.py
```

Output: `paper/generated/qwen_activation_oracle_summary.json`

## Projection Analysis

Projection analysis computes mean layer-23 assistant-start residual activations
for each behavioral condition in `results/behavioral/` and projects each
condition mean onto the MMLU DiffMean direction.

```bash
python project_experiment_activations.py \
  --reference results/layer_sweep/qwen/layer_sweep_qwen_mmlu_mode_8_base_residual_assistant_start_all_wrong_minus_random_answers_n8_100_diffmeans.pt \
  --layer 23

python project_experiment_activations.py \
  --reference results/layer_sweep/qwen/layer_sweep_qwen_mmlu_mode_10_base_residual_assistant_start_all_wrong_minus_random_answers_n10_100_diffmeans.pt \
  --layer 23

python data_analysis/projection_correlation_analysis.py
```

Outputs:

- `results/projection/qwen_experiment_mean_activation_projection_layer23_onto_*.{json,md}`
- `paper/generated/qwen_projection_correlation_summary.json`
- `paper/generated/qwen_projection_correlations.csv`
- `paper/generated/qwen_projection_prompt_deltas.csv`

The projection script caches activations under `results/projection/cache/`.
Pass `--overwrite` to recompute activations from model forward passes.

## Layer-Wise Cosine Structure

The layer-wise cosine structure section uses cosine similarities between
DiffMean vectors, primarily the MMLU `n=8` residual DiffMeans.

```bash
python analyze_qwen_diffmean_cosines.py --neighbor-k 5 --top-k 25
```

Outputs (in `results/diffmean_cosine/qwen_layer_sweep/`, not tracked):
`analysis.md`, `layer_topk_neighbors.{csv,json}`, and `layer_matrices/`. The
plotted values in Fig. 5 are copied from the layer-neighbor results into
`paper/sections/qwen_activation_projection_section.tex`.

## Activation Steering

Fig. 6 comes from the single-layer steering sweep, and Table 2 from the
all-layer runs in `results/activation_patching/qwen_*_ablation/`.

Single-layer steering sweep (Experiment 1), using the MMLU `n=8` DiffMean:

```bash
VECTOR=results/layer_sweep/qwen/layer_sweep_qwen_mmlu_mode_8_base_residual_assistant_start_all_wrong_minus_random_answers_n8_100_diffmeans.pt

python run_steering.py --model qwen --dataset mmlu --vector-path "$VECTOR" --layers $(seq 0 35) --alphas -4 -2 -1 1 2 4 --output-dir results/activation_patching/qwen_mmlu --seed 42
python run_steering.py --model qwen --dataset bbh --vector-path "$VECTOR" --layers $(seq 0 35) --alphas -4 -2 -1 1 2 4 --output-dir results/activation_patching/qwen_bbh --seed 42
```

All-layer single-vector steering (Experiment 2) with the layer-23 MMLU DiffMean:

```bash
python run_ablation.py --dataset mmlu --mode 3 --alphas 0.1 --layer 23 --baseline --output-dir results/activation_patching/qwen_mmlu_ablation --seed 42
python run_ablation.py --dataset mmlu --mode 10 --alphas -0.1 --layer 23 --baseline --output-dir results/activation_patching/qwen_mmlu_ablation --seed 42
python run_ablation.py --dataset bbh --mode 3 --alphas 0.1 --layer 23 --baseline --output-dir results/activation_patching/qwen_bbh_ablation --seed 42
python run_ablation.py --dataset bbh --mode 10 --alphas -0.1 --layer 23 --baseline --output-dir results/activation_patching/qwen_bbh_ablation --seed 42
```

## Build the Paper

The build needs the official NeurIPS 2026 style file `neurips_2026.sty`, which
is not redistributed here; download it from the NeurIPS website into `paper/`.

```bash
cd paper
latexmk -pdf main.tex
```

The main paper source is `paper/main.tex`; section files are in
`paper/sections/`.
