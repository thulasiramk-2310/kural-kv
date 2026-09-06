# Results

Every number here was produced by a run in this repository, under the protocol in
[METHODS.md](METHODS.md). Findings are graded by what the evidence actually
supports, and the grades are part of the result:

- **Established** — replicates on samples it was not selected against, or the
  effect is large relative to the instrument.
- **Suggestive** — direction consistent, magnitude inside the resolution limit,
  or not yet replicated held-out.
- **Null** — measured and flat, reported as such.
- **Withdrawn** — produced by a run later found invalid, retained here so the
  record shows what was claimed and why it was dropped.

Primary arm: Qwen2.5-1.5B-Instruct, bf16, GQA with 12 query heads over 2 KV
heads. Benchmark: RULER `niah_single_1`. Findings 1 and 2 carry held-out
replications at n = 29-80; the remainder are n = 20, where one sample is 0.05.

---

## 1. Retaining a fact is not sufficient for using it

**Established.** The largest and most robust result in the study.

Selection recall is the fraction of (layer, KV head) units whose top-k retains
*every* token of the answer. It is measured by applying the identical selection
the policy applies, so it asks precisely what the policy kept.

Measured twice: once on the samples the study was built on, and once on samples
used for no selection of any kind.

**Held-out (seed 200):**

| budget | 2048 recall / accuracy | 16384 recall / accuracy | Δ recall | Δ accuracy |
|---|---|---|---|---|
| 181 | 0.822 / 0.867 | 0.795 / **0.433** | 0.027 | **0.434** |
| 362 | 0.898 / 0.900 | 0.861 / **0.350** | 0.037 | **0.550** |

**Original (seed 0):**

| budget | 2048 recall / accuracy | 16384 recall / accuracy |
|---|---|---|
| 91 | 0.163 / 0.000 | 0.114 / 0.000 |
| 181 | 0.830 / 0.800 | 0.793 / 0.250 |
| 362 | 0.893 / 0.900 | 0.856 / 0.300 |

Recall reproduces to within 0.02 on every cell across the two sample sets, which
is the strongest replication in the study.

**Selection recall is context-invariant; accuracy is not.** Held-out, the two
contexts differ by **0.027 in recall and 0.434 in accuracy** at 181 entries, and
by 0.037 and 0.550 at 362. SnapKV retains the answer at essentially the same rate
at both lengths, and the model can only use it at the shorter one.

(n: 2K recall 58, 16K recall 29, accuracy 60 each. Two samples were skipped where
the answer string could not be located in the prompt, rather than being scored as
failures.)

The same point from the other direction: at 16384, raising the budget from 181 to
2900 entries lifts accuracy 0.250 → 0.800 while lifting target recall only
0.793 → 0.914. Those 2,700 additional entries buy 0.55 of accuracy and 0.12 of
target retention. **Most of what a larger budget provides is not the answer's own
keys.**

Every scoring heuristic in this literature is built on identifying important
tokens and keeping them. This says that frame is incomplete, with a measurement
rather than an argument.

**Partial retention is worthless, and a metric must reflect it.** At 91 entries,
*some* answer token survives in 85–89% of units while *every* token survives in
11–16%, and accuracy is 0.000 at both contexts. A recall metric counting partial
hits would have reported ~87% success at a budget that retrieves nothing.

### Mechanism: unexplained, three hypotheses rejected

Stated because the absence matters more than a guess would.

1. **Query localisation** — that the diagnostic's unique key concentrates
   attention while RULER's spreads it. Rejected, and inverted: over an 8x context
   increase, entries covering 90% of the policy's scoring mass grow **3.39x** on
   the diagnostic (which follows the absolute axis) and **1.36x** on RULER (which
   follows the proportional one). The task whose mass spreads is the one behaving
   absolutely.
2. **Scoring-mass concentration** generally. Rejected with the same measurement,
   taken against the signal the policy actually ranks on.
3. **Key eviction** — that the needle's key is dropped while its value survives,
   leaving the answer unbindable. Rejected: key recall is 0.942 (2K) vs 0.921
   (16K) at 181 entries, where accuracy is 0.800 vs 0.250, and keys survive
   *better* than values at every budget.

No fourth hypothesis was pursued. The remaining candidate is how attention over a
compacted cache reallocates its mass, which is not probed here.

---

## 2. Head-level reallocation does nothing; depth-level reallocation gives a small real gain

All methods compared at **equal total retained entries** (see METHODS for why
allocated and stored differ for an unequal allocator). Unit of analysis is the
sample, with deltas averaged over budgets within each sample.

| set | comparison | n | mean | SE | t | 95% CI |
|---|---|---|---|---|---|---|
| seed 0 *(grid chosen here)* | Pyramid − Snap | 20 | +0.0750 | 0.0283 | +2.65 | [+0.020, +0.131] |
| seed 100 *(held out)* | Pyramid − Snap | 20 | +0.0300 | 0.0391 | +0.77 | [−0.047, +0.107] |
| seed 200 *(held out)* | Pyramid − Snap | 60 | +0.0433 | 0.0208 | +2.09 | [+0.003, +0.084] |
| **all held-out pooled** | **Pyramid − Snap** | **80** | **+0.0400** | 0.0183 | **+2.19** | **[+0.004, +0.076]** |
| **all held-out pooled** | **Ada-KV − Snap** | **80** | **−0.0050** | 0.0107 | **−0.47** | **[−0.026, +0.016]** |

**Ada-KV is flat, and now bounded. Established.** Across 80 held-out samples the
effect is −0.005 with a 95% interval of [−0.026, +0.016]. This is a stronger
statement than "not significant": any true benefit from reallocating budget
across KV groups is **smaller than 0.026** on this task and model. That is
consistent with the failure being downstream of selection, since Ada-KV's dials
only change *which* entries selection keeps.

**PyramidKV is established but small.** On the samples the grid was chosen from
it measured +0.075. On 80 samples used for no selection whatsoever it is
**+0.040, 95% CI [+0.004, +0.076]**. The interval excludes zero, so the effect is
real; it is also roughly half the figure the selection-contaminated set gave,
which is what a held-out test is for. Reported as established at that reduced
magnitude, with the lower bound noted as close to zero.

The two allocation axes therefore separate cleanly: **depth reallocation produces
a small real gain; head reallocation produces nothing measurable.**

One observation survives independent of magnitude: **PyramidKV's schedule
contains no context term**, yet it is level with SnapKV at 2K and positive at
16K. A context-dependent outcome from a context-independent rule is not explained
by better selection, since selection recall is context-invariant (finding 1).

### Ada-KV's storage overhead grows as budget loosens

| budget | 181 | 362 | 630 | 724 | 1451 | 2900 |
|---|---|---|---|---|---|---|
| stored / allocated | 1.15x | 1.27x | 1.35x | 1.36x | 1.40x | 1.40x |

Contrary to expectation, the penalty is *smallest* at the tightest budget, where
compression matters most: at tight budgets the pooled top-k has little room to
skew, heads receive near-equal counts, and the dense tensor wastes little. At
equal **stored** entries Ada-KV averages −0.043 against SnapKV, and costs about
3x its selection time (23.5 ms vs 8.0 ms p50 at 16K).

This overhead is a property of **dense storage**, not of the method. Paged
attention holds ragged per-head lengths natively and pays none of it.

---

## 3. The correct budget axis is a property of the task

**Established for these two tasks; no general claim.**

| matched on | 2048 | 16384 | gap |
|---|---|---|---|
| 181 entries | 0.80 | 0.28 | +0.52 |
| 362 entries | 0.90 | 0.33 | +0.57 |
| 8.86% of context | 0.80 | 0.72 | +0.08 |
| 17.70% of context | 0.90 | 0.89 | +0.01 |

On RULER the proportional axis is approximately right and the absolute axis badly
wrong. On the synthetic diagnostic the opposite held: both contexts transitioned
at the same absolute budgets, 0.00 at 45 entries and 1.00 at 181, across an
eightfold context change.

**A previous version of this document claimed the absolute axis was correct for
retrieval generally.** That was measured on the diagnostic and did not replicate
on the reported benchmark; it is retracted. Any quoted budget must state both its
axis and its context length.

---

## 4. Distractor-rich retrieval survives nothing below half the cache

**Established.** SnapKV on RULER `niah_multikey_2` at 16384, against a 0.900
full-cache baseline:

| retained | 33–724 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|
| % of context | 0.20–4.42% | 6.25% | 12.50% | 25.00% | 50.01% |
| score | 0.000 | 0.000 | 0.000 | 0.000 | 0.300 |

Where the haystack is built from other needles, so filler differs from the target
only in key and value, half the cache recovers a third of the baseline and
everything below it returns zero. Published budgets in this literature are
single-digit percentages, where this task scores nothing.

---

## 5. The head-budget lever shrinks 16x from MHA to GQA

**Established** (config arithmetic, reproduced by `scripts/budget_units.py`).

| model | attn | q heads | kv heads | group | budget units |
|---|---|---|---|---|---|
| Phi-3.5-mini | MHA | 32 | 32 | 1 | **32** |
| Mistral-7B-v0.3 | GQA | 32 | 8 | 4 | **8** |
| Qwen2.5-7B | GQA | 28 | 4 | 7 | **4** |
| Qwen2.5-3B | GQA | 16 | 2 | 8 | **2** |
| Qwen2.5-1.5B | GQA | 12 | 2 | 6 | **2** |
| Qwen2.5-0.5B | GQA | 14 | 2 | 7 | **2** |

Per-head budget allocation is only actionable per KV group, so an allocator has
`num_key_value_heads` dials. Published per-head methods were validated at the
wide end. Note the lever does not scale with model size within the Qwen family —
it is pinned at 2 from 0.5B to 3B while query heads move 14 → 16.

---

## 6. LongBench QA cannot measure eviction at this model scale

**Null, with a mechanism.** Reported as a result rather than a failed attempt.

Full cache versus 33 retained entries, Qwen2.5-1.5B, documents filtered to ≥16384
natural tokens:

| task | n | full | evicted | range | sd | answer unchanged |
|---|---|---|---|---|---|---|
| musique | 20 | 0.194 | 0.081 | +0.113 | 0.36 | 30% |
| hotpotqa | 66 | 0.321 | 0.237 | +0.085 | 0.21 | 38% |
| narrativeqa | 20 | 0.127 | 0.128 | −0.002 | −0.01 | 10% |
| 2wikimqa | 5 | 0.000 | 0.000 | 0.000 | 0.00 | 40% |
| multifieldqa_en | 1 | 0.000 | 0.000 | 0.000 | 0.00 | 0% |

Nothing clears one standard deviation, and narrativeqa scores *identically* with
33 retained entries as with a full 16K cache. The `answer unchanged` column is
the mechanism: with the context evicted, 30–40% of answers are byte-identical to
the full-cache answer, because the model is answering from parametric knowledge.
**Eviction cannot damage what the model was not using.**

No sample count fixes this. On hotpotqa the entire eviction effect is 0.19
standard deviations, so resolving a difference *within* that effect is
unreachable at any n available.

A second constraint on task choice: requiring documents long enough for a
controlled cross-context comparison leaves multifieldqa_en with **one** qualifying
document and 2wikimqa with five.

---

## Withdrawn

**The MHA contrast arm.** A Phi-3.5-mini run appeared to show Ada-KV flat with 32
budget units, which would have established that GQA narrowness was not what
flattened it. That run was invalidated by the attention-mask defect (METHODS),
and the conclusion is withdrawn. **Whether "two dials" flattened Ada-KV is
currently unknown**, and it is the most valuable open question left.

---

## Limitations

**No held-out set except for finding 2.** Every other result comes from
`seed=0`. Where a held-out set was constructed the effect halved, which is the
best available estimate of how much the others may be inflated.

**n = 20 to 60.** One sample is 0.05, so no individual cell is a measured
effect. Finding 2 rests on 80 held-out samples; the remaining findings are n=20
and rest on consistency of sign across budgets.

**One model carries the study.** Llama-3.2-1B is gated and unrun; Phi-3.5-mini is
capped at 2K by a LongRoPE defect and its arm is withdrawn. Findings are
Qwen2.5-1.5B unless stated.

**Generation quality is unmeasured.** Section 6 explains why: the available
generation benchmark has no dynamic range at this scale. Whether the section 1
result holds for generation as well as retrieval is open.

---

## Reproduction traps found

Recorded in METHODS with detail. Each produced plausible output rather than an
error, and each cost real time:

1. **Position on resume after eviction** — `position_ids` defaults to the
   *compacted* cache length, silently renumbering the continuation and making a
   reproduction report eviction as more damaging than it is.
2. **One causal mask for every layer** — a layer-varying budget cannot be run
   without bypassing mask construction.
3. **Registering an attention function is not enough** — mask construction
   dispatches separately on the same name, and an unregistered name silently
   skips the causal mask, making prefill bidirectional while decode stays correct.
4. **Qwen2.5 overflows fp16** — NaN logits on a plain forward pass, with every
   shape check passing.
5. **Phi-3.5-mini past position 4096** — LongRoPE degeneration with logits still
   finite, so a finiteness assert does not catch it.
