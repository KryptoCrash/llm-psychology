# llm-psychology

Minimal retrieval layer for reimplementing *Conformity in Large Language Models* ([arXiv:2410.12428](https://arxiv.org/abs/2410.12428)).

This repo intentionally only contains dataset acquisition/parsing code for the paper's main datasets:

- `MMLU` via `cais/mmlu`
- `CommonsenseQA` via `tau/commonsense_qa`
- `BBH object_counting` via `lukaemon/bbh`
- `PopQA` via `akariasai/PopQA`
- `Politiscale` via the public `questions.md` source
- `OpinionsQA` via a public Hugging Face mirror, with the original authors' repo/CodaLab worksheet referenced in the manifest

## Setup

```bash
uv sync
```

## Fetch datasets

Fetch everything used in the paper:

```bash
uv run fetch-paper-datasets --dataset all
```

Fetch a subset:

```bash
uv run fetch-paper-datasets --dataset mmlu --dataset popqa
```

Outputs are written under `data/raw/<dataset>/` as JSONL plus a `manifest.json` per dataset.

## Notes

- The script only retrieves/parses source data. It does not include prompting, inference, evaluation, or mitigation code.
- `OpinionsQA` is distributed by the original authors through CodaLab; the default retriever uses a public mirror that exposes the question set directly. The manifest records both the mirror and the original project reference.
- Check upstream dataset licenses before redistribution.
