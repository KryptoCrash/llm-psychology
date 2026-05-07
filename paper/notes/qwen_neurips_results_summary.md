# Qwen inference results and NeurIPS presentation plan

Branch: `qwen_results`

Primary generator: `qwen_experiment.py`, which runs `multi_actor.run_experiment`
over 48 cells:

- Models: Qwen only (`Qwen/Qwen3-8B`)
- Datasets: `mmlu`, `bbh`
- Social conditions: participant count `n=1..10`, plus `qd` and `da`
- Prompting variants: `base`, `explain`

The branch contains all 48 expected Qwen result files:

- `qwen_mmlu_{1..10}_{base,explain}.json`
- `qwen_mmlu_qd_{base,explain}.json`
- `qwen_mmlu_da_{base,explain}.json`
- `qwen_bbh_{1..10}_{base,explain}.json`
- `qwen_bbh_qd_{base,explain}.json`
- `qwen_bbh_da_{base,explain}.json`

The branch also contains 48 analogous Llama result files, which are useful as a
secondary model comparison but were not produced by `qwen_experiment.py`.

## Important NeurIPS 2026 constraints from `neurips_2026.tex`

- Use `neurips_2026.sty`; do not modify the style file.
- Main submission is anonymous by default: use `\usepackage{neurips_2026}` with
  no `final` or `preprint` option.
- Main content is limited to 9 pages including figures. Acknowledgments,
  references, checklist, and optional appendices do not count.
- Critical evidence must be in the main paper, not only the appendix.
- Abstract is a single paragraph.
- Tables should use `booktabs`, no vertical rules, table caption above table.
- Figure captions go below figures and should state the key takeaway.
- `neurips_2026.tex` includes `\input{checklist.tex}`, but `checklist.tex` is
  currently missing from the repo, so the paper shell will not compile until the
  checklist file is added and filled.

## Qwen aggregate results

Accuracy below is `correct / parseable`, matching the JSON summaries. Strict
accuracy treats unparseable responses as wrong and is noted where it materially
changes the story.

| Dataset | Prompt | Baseline acc | MA avg acc | MA avg conf | MA n=10 acc | MA n=10 conf | QD acc | QD conf | DA acc | DA conf |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MMLU | base | 0.990 | 0.793 | 0.202 | 0.637 | 0.363 | 0.914 | 0.049 | 0.698 | 0.291 |
| MMLU | explain | 0.970 | 0.902 | 0.066 | 0.867 | 0.082 | 0.898 | 0.020 | 0.887 | 0.103 |
| BBH | base | 0.970 | 0.717 | 0.221 | 0.610 | 0.390 | 0.720 | 0.190 | 0.570 | 0.350 |
| BBH | explain | 0.950 | 0.861 | 0.099 | 0.808 | 0.152 | 0.840 | 0.130 | 0.778 | 0.141 |

Main pattern: Qwen starts near ceiling on the selected 100-item sets, but base
prompting shows large conformity under repeated incorrect peer answers. Asking
for explanations suppresses conformity and preserves accuracy, especially at
large group sizes.

## Scaling with group size

For the standard multi-actor condition (`n=2..10`):

| Dataset | Prompt | r(n, accuracy) | r(n, conformity) |
|---|---:|---:|---:|
| MMLU | base | -0.972 | 0.966 |
| MMLU | explain | -0.630 | 0.724 |
| BBH | base | -0.700 | 0.846 |
| BBH | explain | -0.519 | 0.710 |

This is the cleanest paper figure: group size strongly increases conformity and
decreases accuracy, with explanation prompting flattening both slopes.

## QD and DA against matched n=10 MA

| Dataset | Prompt | QD acc delta | QD conf delta | DA acc delta | DA conf delta |
|---|---:|---:|---:|---:|---:|
| MMLU | base | +0.276 | -0.313 | +0.060 | -0.072 |
| MMLU | explain | +0.031 | -0.061 | +0.019 | +0.021 |
| BBH | base | +0.110 | -0.200 | -0.040 | -0.040 |
| BBH | explain | +0.032 | -0.022 | -0.030 | -0.010 |

QD is the strongest mitigation for MMLU base and helps BBH base conformity, but
it is not uniformly accuracy-improving once explanation prompting is already
enabled. DA is weaker and sometimes harms accuracy, especially on BBH.

## Parseability caveat

Qwen MMLU base has nontrivial unparseable rates under social pressure. Average
MA parseable accuracy is 0.793, but average strict MA accuracy is 0.694. The
main paper should either:

- report strict accuracy as the primary metric and parseable-conditioned
  accuracy as secondary; or
- keep parseable-conditioned accuracy but include parse rate in the same table.

For BBH, parseability is nearly perfect in base prompting, so this caveat is
mostly a MMLU/Qwen-format issue.

## Supporting probes

The older standalone Qwen probes are not outputs of `qwen_experiment.py`, but
they document question-bank construction:

| File | Attempts | Parseable | Correct | Accuracy |
|---|---:|---:|---:|---:|
| `qwen_mmlu_results.json` | 200 | 196 | 138 | 0.704 |
| `qwen_bbh_results.json` | 200 | 187 | 89 | 0.476 |
| `combined_mmlu_correct.json` | 100 records | - | - | - |
| `combined_bbh_correct.json` | 100 records | - | - | - |

The experimental set is therefore a selected set of questions the combined
models can answer correctly, not a random benchmark evaluation. The paper should
frame the task as a controlled susceptibility test, not as benchmark accuracy.

## Suggested NeurIPS paper framing

Working title:
**Do language models conform? Social pressure, repetition, and deliberation in
multi-agent prompts**

Core claim:
LLMs exhibit an Asch-like conformity effect in controlled multiple-choice and
short-answer settings: repeated incorrect peer answers cause the model to abandon
answers it can otherwise produce correctly. The effect scales with group size,
is partly driven by textual repetition, and is attenuated by explanation-style
deliberation.

Recommended main-paper figures:

1. Line plot: conformity rate vs. number of prior participants, faceted by
   dataset and prompt type. This should be Figure 1.
2. Line plot: strict accuracy vs. number of prior participants, same facets.
3. Bar plot at n=10: MA vs QD vs DA conformity and accuracy.
4. Optional compact model-comparison panel: Qwen vs Llama average conformity.

Recommended main-paper tables:

1. Experimental design table: dataset, selected items, model, prompt variant,
   social condition, metric definitions.
2. Aggregate result table like the Qwen table above, with strict accuracy and
   parse rate added.

Recommended appendix material:

- Full per-condition tables for all 96 Qwen/Llama files.
- Subject-level conformity hotspots.
- Prompt templates.
- Examples of conforming and nonconforming generations.
- Parse-failure analysis.

Risks to address before submission:

- The current runs are single-seed stochastic generations (`do_sample=True` in
  `multi_actor.py`), so confidence intervals should be bootstrap intervals over
  items at minimum; ideally rerun several seeds.
- The selected 100-item sets create a controlled intervention benchmark, not an
  estimate of general MMLU/BBH performance.
- Qwen explanation outputs sometimes contain malformed or rambling text despite
  parseable final answers; qualitative examples should be chosen carefully and
  not overinterpreted as faithful reasoning.
- `checklist.tex` is missing.
