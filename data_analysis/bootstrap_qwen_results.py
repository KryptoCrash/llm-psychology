#!/usr/bin/env python3
"""Bootstrap Qwen inference results from saved JSON files.

This script does not run model inference. It resamples the saved per-item
records to estimate item-level uncertainty for accuracy, conformity, parse rate,
and selected paired contrasts. Accuracy is correct / parseable.
"""

from __future__ import annotations

import csv
import json
import math
import random
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper" / "generated"
BOOTSTRAP_SAMPLES = 10_000
PERMUTATION_SAMPLES = 10_000
SEED = 20260507

RESULT_RE = re.compile(
    r"^qwen_(mmlu|bbh)_(1|2|3|4|5|6|7|8|9|10|qd|da)_(base|explain)\.json$"
)


def item_key(dataset: str, record: dict) -> tuple:
    if dataset == "mmlu":
        return (
            record["subject"],
            record["question"],
            tuple(record["choices"]),
            int(record["answer"]),
        )
    return (
        record["subject"],
        record["question"],
        record["target"].strip(),
    )


def is_parseable(record: dict) -> bool:
    return bool(record.get("model_answer"))


def is_correct(record: dict) -> bool:
    return bool(record.get("is_correct"))


def is_conforming(record: dict) -> bool:
    return record.get("conforms") is True


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def ci(values: list[float]) -> tuple[float, float]:
    cleaned = sorted(v for v in values if not math.isnan(v))
    return percentile(cleaned, 0.025), percentile(cleaned, 0.975)


def metric(records: list[dict], indices: list[int] | None = None) -> dict[str, float]:
    if indices is None:
        indices = list(range(len(records)))
    selected = [records[i] for i in indices]
    n = len(selected)
    parseable = sum(is_parseable(r) for r in selected)
    correct = sum(is_correct(r) for r in selected)
    conforms = sum(is_conforming(r) for r in selected)
    has_conformity = any("conforms" in r for r in selected)
    return {
        "items": n,
        "parseable": parseable,
        "correct": correct,
        "conforms": conforms,
        "accuracy": correct / parseable if parseable else float("nan"),
        "parse_rate": parseable / n if n else float("nan"),
        "conformity": conforms / parseable if has_conformity and parseable else float("nan"),
    }


def bootstrap_metric(records: list[dict], rng: random.Random) -> dict[str, tuple[float, float]]:
    n = len(records)
    samples = {
        "accuracy": [],
        "parse_rate": [],
        "conformity": [],
    }
    for _ in range(BOOTSTRAP_SAMPLES):
        indices = [rng.randrange(n) for _ in range(n)]
        m = metric(records, indices)
        for key in samples:
            samples[key].append(m[key])
    return {key: ci(values) for key, values in samples.items()}


def bootstrap_delta(
    before: list[dict],
    after: list[dict],
    metric_name: str,
    rng: random.Random,
) -> tuple[float, float, float]:
    assert len(before) == len(after)
    n = len(before)
    observed = metric(after)[metric_name] - metric(before)[metric_name]
    samples = []
    for _ in range(BOOTSTRAP_SAMPLES):
        indices = [rng.randrange(n) for _ in range(n)]
        samples.append(metric(after, indices)[metric_name] - metric(before, indices)[metric_name])
    low, high = ci(samples)
    return observed, low, high


def paired_permutation_p(
    before: list[dict],
    after: list[dict],
    outcome_name: str,
    rng: random.Random,
) -> float:
    """Two-sided paired randomization test for paired binary outcomes.

    Accuracy tests are restricted to items parseable in both conditions, matching
    the paper's accuracy denominator.
    """

    if outcome_name == "accuracy":
        pairs = [
            (1 if is_correct(b) else 0, 1 if is_correct(a) else 0)
            for b, a in zip(before, after)
            if is_parseable(b) and is_parseable(a)
        ]
        before_values = [b for b, _ in pairs]
        after_values = [a for _, a in pairs]
    elif outcome_name == "conformity":
        before_values = [1 if is_conforming(r) else 0 for r in before]
        after_values = [1 if is_conforming(r) else 0 for r in after]
    else:
        raise ValueError(outcome_name)

    diffs = [a - b for b, a in zip(before_values, after_values)]
    observed = mean(diffs)
    extreme = 0
    for _ in range(PERMUTATION_SAMPLES):
        permuted = [d if rng.random() < 0.5 else -d for d in diffs]
        if abs(mean(permuted)) >= abs(observed) - 1e-12:
            extreme += 1
    return (extreme + 1) / (PERMUTATION_SAMPLES + 1)


def pearson(xs: list[float], ys: list[float]) -> float:
    mx = mean(xs)
    my = mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = sum((x - mx) ** 2 for x in xs)
    den_y = sum((y - my) ** 2 for y in ys)
    return num / math.sqrt(den_x * den_y) if den_x and den_y else float("nan")


def bootstrap_trend(
    records_by_n: dict[int, list[dict]],
    metric_name: str,
    rng: random.Random,
) -> tuple[float, float, float]:
    ns = sorted(records_by_n)
    observed = pearson(ns, [metric(records_by_n[n])[metric_name] for n in ns])
    samples = []
    item_count = len(next(iter(records_by_n.values())))
    for _ in range(BOOTSTRAP_SAMPLES):
        indices = [rng.randrange(item_count) for _ in range(item_count)]
        samples.append(pearson(ns, [metric(records_by_n[n], indices)[metric_name] for n in ns]))
    low, high = ci(samples)
    return observed, low, high


def fmt_ci(value: float, low: float, high: float) -> str:
    if math.isnan(value):
        return "--"
    return f"{value:.3f} [{low:.3f}, {high:.3f}]"


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rng = random.Random(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[tuple[str, str, str], list[dict]] = {}
    keys_by_dataset: dict[str, list[tuple]] = {}

    for path in sorted(ROOT.glob("qwen_*.json")):
        match = RESULT_RE.match(path.name)
        if not match:
            continue
        dataset, mode, prompt = match.groups()
        with path.open() as f:
            data = json.load(f)
        records = data["results"]
        results[(dataset, mode, prompt)] = records
        keys = [item_key(dataset, r) for r in records]
        if dataset not in keys_by_dataset:
            keys_by_dataset[dataset] = keys
        elif keys != keys_by_dataset[dataset]:
            raise ValueError(f"Item order mismatch in {path.name}")

    expected = 2 * 12 * 2
    if len(results) != expected:
        raise ValueError(f"Expected {expected} Qwen files, found {len(results)}")

    condition_rows = []
    for (dataset, mode, prompt), records in sorted(results.items()):
        m = metric(records)
        intervals = bootstrap_metric(records, rng)
        condition_rows.append(
            {
                "dataset": dataset,
                "mode": mode,
                "prompt": prompt,
                "items": m["items"],
                "parseable": m["parseable"],
                "correct": m["correct"],
                "conforms": m["conforms"],
                "accuracy": f"{m['accuracy']:.6f}",
                "accuracy_ci_low": f"{intervals['accuracy'][0]:.6f}",
                "accuracy_ci_high": f"{intervals['accuracy'][1]:.6f}",
                "parse_rate": f"{m['parse_rate']:.6f}",
                "parse_rate_ci_low": f"{intervals['parse_rate'][0]:.6f}",
                "parse_rate_ci_high": f"{intervals['parse_rate'][1]:.6f}",
                "conformity": "" if math.isnan(m["conformity"]) else f"{m['conformity']:.6f}",
                "conformity_ci_low": "" if math.isnan(intervals["conformity"][0]) else f"{intervals['conformity'][0]:.6f}",
                "conformity_ci_high": "" if math.isnan(intervals["conformity"][1]) else f"{intervals['conformity'][1]:.6f}",
            }
        )

    contrast_specs = []
    for dataset in ("mmlu", "bbh"):
        for prompt in ("base", "explain"):
            contrast_specs.extend(
                [
                    (dataset, prompt, "baseline_to_ma10", "1", "10"),
                    (dataset, prompt, "ma10_to_qd", "10", "qd"),
                    (dataset, prompt, "ma10_to_da", "10", "da"),
                ]
            )
        contrast_specs.append((dataset, "base_to_explain", "ma10_explain_effect", "10_base", "10_explain"))

    contrast_rows = []
    for dataset, prompt, name, before_mode, after_mode in contrast_specs:
        if prompt == "base_to_explain":
            before = results[(dataset, "10", "base")]
            after = results[(dataset, "10", "explain")]
            prompt_label = "n10"
        else:
            before = results[(dataset, before_mode, prompt)]
            after = results[(dataset, after_mode, prompt)]
            prompt_label = prompt

        acc_delta = bootstrap_delta(before, after, "accuracy", rng)
        parse_delta = bootstrap_delta(before, after, "parse_rate", rng)
        row = {
            "dataset": dataset,
            "prompt": prompt_label,
            "contrast": name,
            "before": before_mode,
            "after": after_mode,
            "accuracy_delta": f"{acc_delta[0]:.6f}",
            "accuracy_delta_ci_low": f"{acc_delta[1]:.6f}",
            "accuracy_delta_ci_high": f"{acc_delta[2]:.6f}",
            "accuracy_permutation_p": f"{paired_permutation_p(before, after, 'accuracy', rng):.6f}",
            "parse_rate_delta": f"{parse_delta[0]:.6f}",
            "parse_rate_delta_ci_low": f"{parse_delta[1]:.6f}",
            "parse_rate_delta_ci_high": f"{parse_delta[2]:.6f}",
        }
        if before_mode != "1":
            conf_delta = bootstrap_delta(before, after, "conformity", rng)
            row.update(
                {
                    "conformity_delta": f"{conf_delta[0]:.6f}",
                    "conformity_delta_ci_low": f"{conf_delta[1]:.6f}",
                    "conformity_delta_ci_high": f"{conf_delta[2]:.6f}",
                    "conformity_permutation_p": f"{paired_permutation_p(before, after, 'conformity', rng):.6f}",
                }
            )
        else:
            row.update(
                {
                    "conformity_delta": "",
                    "conformity_delta_ci_low": "",
                    "conformity_delta_ci_high": "",
                    "conformity_permutation_p": "",
                }
            )
        contrast_rows.append(row)

    trend_rows = []
    for dataset in ("mmlu", "bbh"):
        for prompt in ("base", "explain"):
            records_by_n = {n: results[(dataset, str(n), prompt)] for n in range(2, 11)}
            for metric_name in ("accuracy", "conformity"):
                observed, low, high = bootstrap_trend(records_by_n, metric_name, rng)
                trend_rows.append(
                    {
                        "dataset": dataset,
                        "prompt": prompt,
                        "metric": metric_name,
                        "pearson_r": f"{observed:.6f}",
                        "pearson_r_ci_low": f"{low:.6f}",
                        "pearson_r_ci_high": f"{high:.6f}",
                    }
                )

    write_csv(
        OUT_DIR / "qwen_condition_bootstrap.csv",
        condition_rows,
        list(condition_rows[0].keys()),
    )
    write_csv(
        OUT_DIR / "qwen_contrast_tests.csv",
        contrast_rows,
        list(contrast_rows[0].keys()),
    )
    write_csv(
        OUT_DIR / "qwen_trend_bootstrap.csv",
        trend_rows,
        list(trend_rows[0].keys()),
    )

    with (OUT_DIR / "qwen_bootstrap_results.json").open("w") as f:
        json.dump(
            {
                "bootstrap_samples": BOOTSTRAP_SAMPLES,
                "permutation_samples": PERMUTATION_SAMPLES,
                "seed": SEED,
                "note": (
                    "Intervals and tests quantify item-level uncertainty "
                    "conditional on the saved generations and peer-answer assignments."
                ),
                "conditions": condition_rows,
                "contrasts": contrast_rows,
                "trends": trend_rows,
            },
            f,
            indent=2,
        )

    print(f"Loaded {len(results)} Qwen condition files.")
    print(f"Wrote {OUT_DIR / 'qwen_condition_bootstrap.csv'}")
    print(f"Wrote {OUT_DIR / 'qwen_contrast_tests.csv'}")
    print(f"Wrote {OUT_DIR / 'qwen_trend_bootstrap.csv'}")
    print(f"Wrote {OUT_DIR / 'qwen_bootstrap_results.json'}")


if __name__ == "__main__":
    main()
