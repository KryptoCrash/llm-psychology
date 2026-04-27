# Question Distillation vs Multi-Actor: Conformity Analysis

**Hypothesis (from Sharma et al. 2024, arXiv:2410.12428):** Presenting repeated identical responses as a single summary line (question distillation, QD) reduces LLM conformity to incorrect peer answers, because textual repetition is a primary driver of social conformity in LLMs.

Model: `meta-llama/Llama-3.2-3B-Instruct` | Dataset: `correct_questions.json` | N per condition: 25 questions

---

## 1. Conformity Rates by Group Size (n)

| n  | MA conformity | QD conformity | Delta (QD−MA) | Direction    |
|----|--------------|--------------|---------------|--------------|
| 2  | 0.00         | 0.16         | +0.16         | QD higher    |
| 3  | 0.08         | 0.24         | +0.16         | QD higher    |
| 4  | 0.24         | 0.04         | −0.20         | QD lower     |
| 5  | 0.08         | 0.20         | +0.12         | QD higher    |
| 6  | 0.32         | 0.20         | −0.12         | QD lower     |
| 7  | 0.36         | 0.12         | −0.24         | QD lower     |
| 8  | 0.36         | 0.16         | −0.20         | QD lower     |
| 9  | 0.32         | 0.04         | −0.28         | QD lower     |
| 10 | 0.28         | 0.16         | −0.12         | QD lower     |
| **AVG** | **0.227** | **0.147** | **−0.080** | **QD lower** |

---

## 2. Conformity Trend with n (Pearson r)

| Condition | r (conformity vs n) | Interpretation |
|-----------|---------------------|----------------|
| MA        | **+0.786**          | Strong positive — more repetitions = more conformity |
| QD        | **−0.290**          | Weak/flat — conformity decoupled from group size |

This is the most striking finding. In the standard multi-actor condition, conformity scales strongly with group size: each additional line repeating the wrong answer increases pressure. Under QD, conformity is essentially flat regardless of how many "participants" came before, because the model only ever sees one summary sentence regardless of n.

---

## 3. Accuracy by Group Size (n)

| n  | MA accuracy | QD accuracy | Delta (QD−MA) |
|----|-------------|-------------|---------------|
| 1  | 0.92        | 0.88        | −0.04         |
| 2  | 0.88        | 0.64        | −0.24         |
| 3  | 0.76        | 0.64        | −0.12         |
| 4  | 0.68        | 0.72        | +0.04         |
| 5  | 0.68        | 0.60        | −0.08         |
| 6  | 0.60        | 0.68        | +0.08         |
| 7  | 0.56        | 0.60        | +0.04         |
| 8  | 0.56        | 0.56        | 0.00          |
| 9  | 0.52        | 0.76        | +0.24         |
| 10 | 0.68        | 0.64        | −0.04         |
| **AVG** | **0.684** | **0.672** | **−0.012** |

| Condition | r (accuracy vs n) | Interpretation |
|-----------|-------------------|----------------|
| MA        | **−0.830**        | Strong negative — accuracy erodes as group pressure increases |
| QD        | **−0.391**        | Weaker decline — accuracy more stable across group sizes |

---

## 4. Was the Hypothesis Proven?

**Partially.** The hypothesis holds at n ≥ 6, where QD consistently produces lower conformity than MA (by 12–28 percentage points). At large group sizes, removing repetition clearly reduces the anchoring effect.

However, the hypothesis **fails at small n (2, 3, 5)**, where QD conformity is actually higher than MA. A possible explanation: at low n, the authoritative, declarative framing of the QD summary ("All 1 participant before you has chosen X") may be *more* persuasive than a single formatted response line. The summary implies consensus rather than presenting a data point.

**Overall average**: QD reduced conformity from 22.7% to 14.7% — a 35% relative reduction — supporting the hypothesis at the aggregate level.

---

## 5. Key Insights

**Repetition drives conformity in MA, but not linearly.** The strong r=0.786 correlation between n and conformity in MA means each additional copy of the wrong answer adds meaningful pressure. This directly implicates textual repetition as a mechanism, not just the presence of peer disagreement.

**QD breaks the conformity-scaling relationship.** By collapsing n−1 response lines into one sentence, QD removes the repetition signal. The model's conformity rate no longer tracks group size. This is consistent with the paper's core claim.

**QD does not reliably improve accuracy.** Despite lower conformity, QD accuracy is nearly identical to MA on average (67.2% vs 68.4%). This suggests that QD reduces capitulation to wrong answers but doesn't fully restore baseline performance — possibly because the distilled framing itself introduces some bias at low n.

**Small-n anomaly: the framing effect.** At n=2 and n=3, QD conformity (16%, 24%) exceeds MA (0%, 8%). This suggests the natural-language summary format ("All N participants chose X") carries an implicit authority or consensus signal that individual response lines do not, particularly when group size is small enough that repetition hasn't yet accumulated in MA.

**Baseline (n=1) accuracy difference (MA=0.92 vs QD=0.88)** is likely sampling noise from `do_sample=True` with 25 questions; both conditions use identical prompts at n=1.

---

## 6. Summary Verdict

| Claim | Result |
|-------|--------|
| QD reduces conformity overall | **Supported** (22.7% → 14.7%, −35%) |
| QD reduces conformity at all n | **Not supported** (fails at n=2,3,5) |
| Repetition drives MA conformity | **Strongly supported** (r=0.786) |
| QD preserves accuracy | **Roughly neutral** (avg Δ=−0.012) |
| QD conformity scales with n | **Not supported** (r=−0.290, essentially flat) |

The results broadly replicate the paper's finding that question distillation attenuates conformity, with the nuance that the effect is n-dependent and can reverse at small group sizes due to a framing/authority effect.
