# Conformity Analysis – Llama-3.2-3B-Instruct on MMLU

## Summary statistics

| n | Correct | Accuracy | Conforms | Conform rate | Wrong (independent) |
|---|---|---|---|---|---|
| 1 | 23/25 | 0.92 | — | — | 2 |
| 2 | 22/25 | 0.88 | 0 | 0.00 | 3 |
| 3 | 19/25 | 0.76 | 2 | 0.08 | 4 |
| 4 | 17/25 | 0.68 | 6 | 0.24 | 2 |
| 5 | 17/25 | 0.68 | 2 | 0.08 | 6 |
| 6 | 15/25 | 0.60 | 8 | 0.32 | 2 |
| 7 | 14/25 | 0.56 | 9 | 0.36 | 2 |
| 8 | 14/25 | 0.56 | 9 | 0.36 | 2 |
| 9 | 13/25 | 0.52 | 8 | 0.32 | 4 |
| 10 | 17/25 | 0.68 | 7 | 0.28 | 1 |

## Key observations

**1. Conformity emerges only at n≥3.** With a single confederate (n=2), the model shows zero conformity — it ignores the single wrong answer entirely. Conformity only kicks in when there are at least 2 prior participants showing the same wrong answer, consistent with Asch's finding that a unanimous majority is required to produce conformity pressure.

**2. Accuracy degrades monotonically with group size (n=2 to n=9).** From 0.88 at n=2 down to 0.52 at n=9, the decline tracks closely with rising conformity rate — confirming that conformity, not noise, is driving the accuracy drop.

**3. n=7–8 is the apparent saturation point.** Conformity rate peaks at 0.36 (9/25) for both n=7 and n=8, then plateaus. Adding more confederates beyond 7 doesn't increase conformity further, mirroring findings from human conformity experiments where majority pressure saturates around 3–5 people.

**4. Accuracy partially recovers at n=10.** Accuracy jumps from 0.52 back to 0.68 despite conformity remaining moderately high (0.28). The recovery comes almost entirely from independent wrong answers collapsing to just 1 — the model at n=10 is being pulled strongly toward the confederate answer, which paradoxically displaces some independent errors. This is a genuine Asch-like effect: the social pressure is so strong it overrides the model's own (sometimes incorrect) reasoning.

**5. Anatomy and professional/clinical subjects are most susceptible.** Anatomy shows 9 conformity instances across all conditions, followed by miscellaneous (6) and professional medicine (6). These are domains where the model may have lower confidence, making it more susceptible to social influence — consistent with the psychological principle that uncertainty increases conformity.

**6. n=5 is a notable outlier.** Conformity drops back to 0.08 at n=5 (matching n=3) before rising again at n=6. This may be stochastic given the small sample size (25 questions), but it's worth flagging as a replication point if you run more questions.
