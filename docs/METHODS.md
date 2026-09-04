# Methods protocol

The rules every measurement in this repository is produced under. Recorded
before the results they govern, so that the conditions are checkable rather than
reconstructed after the fact.

## Numerical validity

**Precision is 2 bytes per element. The specific 2-byte dtype is a per-model
numerical choice, not a study parameter.**

Qwen2.5 has activation outliers that overflow fp16. The model emits NaN logits
on a plain 256-token forward pass — no cache, no chunking, no eviction — while
every structural property remains correct: shapes valid, nothing raised, RoPE
phases intact, memory accounting clean. `argmax` then returns token 0
indefinitely. The primary arm therefore runs bf16. Llama-3.2-1B and Phi-3.5-mini
are stable in fp16 and use it. Memory cost is identical either way, so no
measurement is affected by the choice.

**Finiteness is asserted before any number is recorded.** This applies to every
run, not only to gate checks: a run that produces a non-finite intermediate
fails loudly rather than writing a row. The failure above is invisible to shape
checks, crash detection, and memory accounting simultaneously, so it must be
tested for directly.

**Probe and benchmark inputs are real text.** Random token ids cannot
distinguish a working model from a broken one — degenerate output is expected
from gibberish input, which is exactly how the fp16 defect above stayed hidden
through a full passing gate run.

**Degenerate output is treated as a numerics bug until proven otherwise, never
as eviction damage.**

## Position handling under eviction

RoPE is applied to K before it enters the cache. Surviving entries therefore
carry the phase they were built with, and after an eviction their original
positions are non-contiguous. That is correct and is never undone: eviction
indexes the cache and never recomputes, re-rotates, or renumbers it.

The continuation is the part that silently breaks. `cache_position` and
`position_ids` are the same number in every ordinary forward pass, so
`transformers` collapses them and defaults `position_ids` to the cache length.
After eviction that length is the *compacted* one, so the next token is assigned
a RoPE phase from the middle of the surviving range and lands behind keys it
should follow. Nothing raises; quality simply degrades.

The two indices are therefore tracked separately throughout:

- `cache_position` — physical slot written to, tracks the compacted cache
- `position_ids` — logical position for RoPE, tracks the original sequence

`scripts/gate_check.py` asserts the invariant on every run by comparing
surviving keys bitwise against a pre-eviction reference, so a regression fails
rather than degrading quietly.

**Consequence for reproduction.** An implementation that gets this wrong reports
eviction as *more damaging than it is*. It is therefore a candidate explanation
for variance between published results, not only an implementation detail. The
naive and corrected implementations are to be measured against each other at
matched method and budget; the size of that gap is a result in its own right.

## Attention and scoring

Attention runs with `attn_implementation="eager"` throughout. Fused kernels do
not expose the attention scores that eviction policies score against.

**Scoring uses a 32–64 token observation window, never the full attention
matrix.** This constraint belongs to the scoring code itself and is not
satisfied by any probe length used elsewhere; the gate check's probe is sized to
stress the eviction path and is not a scoring window.

Eager attention materialises a `[1, q_heads, q_len, kv_len]` score matrix, so a
single-shot long prefill does not fit in 8GB. Long contexts are prefilled in
chunks. K and V are projections of the hidden states, so the resulting cache is
bit-identical to a single-shot prefill; only peak activation differs.

**`prefill_chunk` is 128 throughout.** At 16384 tokens, chunk sizes of 512 and
256 both fail on this GPU with allocation errors despite roughly 7.6 GiB
reported free — the score matrix is the largest single allocation and no
contiguous block of that size is obtainable. `expandable_segments:True` does not
help, placing the fragmentation below PyTorch's caching allocator. Chunk 128
clears it reliably (six of six across in-process repeats and cold processes) at
3.65 GiB peak. The chunk is held fixed across every context length so that
measured curves reflect context scaling rather than chunk-size effects.

## Reading a scaling exponent

Prefill cost is fitted as log(time) against log(context), and **the exponent
quoted is the local one at the top of the measured range, not the global fit.**

A global log-log fit over a range whose low end is dominated by fixed per-chunk
overhead is dragged toward linear regardless of how the attention term is
growing, and the smaller the chunk the stronger that pull. This is not a
hypothetical: the first version of `scripts/prefill_timing.py` keyed its verdict
off the global fit and reported "near-linear" for a curve whose top-of-range
exponent was 1.75 — a wrong methodological conclusion, arrived at from correct
measurements. Any scaling claim in this repository is read from the top of the
range, with the global fit reported for reference only.

The exponent at the top of the range is also not a ceiling. Attention is only
part of prefill: the MLP and the projections are linear in context, so the total
sits between the linear and quadratic terms and creeps toward 2 as the attention
term comes to dominate. A measured 1.75 across 8K→16K should be expected to rise
further above 16K, and must not be treated as a converged value.

## What distractor-rich retrieval costs

RULER's `niah_multikey_2` builds its haystack from other needles, so the filler
differs from the target only in key and value. A scoring policy cannot succeed by
noticing that needles look unlike filler, which is the cue the noise haystack and
the synthetic diagnostic both leave available.

SnapKV on Qwen2.5-1.5B at 16384 context, n=20, against a 0.900 full-cache
baseline:

| retained | % of context | score |
|---|---|---|
| 33 – 724 | 0.20 – 4.42% | 0.000 |
| 1024 | 6.25% | 0.000 |
| 2048 | 12.50% | 0.000 |
| 4096 | 25.00% | 0.000 |
| 8192 | 50.01% | 0.300 |

Nothing survives below half the cache, and half the cache recovers only a third
of the baseline. The same variant at 2048 context is already at 0.000 by 33
entries and reaches only 0.050 at 362.

This bounds what the method delivers on retrieval where distractors resemble the
target. Published budgets in this literature sit in the single-digit percentages;
at those budgets this task returns zero. The number is reported because the
distinction between "eviction preserves quality at 5% budget" and "eviction
preserves quality at 5% budget on tasks whose targets are lexically distinctive"
is the difference between a usable method and a benchmark artefact.

## Comparing methods: equal total retained entries

**Methods are compared at equal total retained KV entries, never at equal
per-unit budgets.** This rule is fixed before the methods are implemented,
because the units differ between them and choosing afterwards would be choosing
the answer.

SnapKV selects a single budget applied per KV head. PyramidKV allocates across
layers on a schedule, so its budget is per layer and its floor is a per-layer
floor whose total depends on that schedule. Ada-KV allocates across heads, which
on this model means across two KV groups rather than the twelve query heads its
paper assumes. Giving each method "budget B" means three different totals and
three different memory footprints, and the comparison would then be measuring
allocation arithmetic rather than policy quality.

Total retained entries is the axis because it is what the study's headline metric
is about: memory. Two methods retaining the same number of entries occupy the
same cache, whatever internal distribution produced it.

Both axes are reported for every configuration — absolute retained entries and
retained fraction of context — because neither can be assumed to be the
comparable one (see below). A quoted budget without its axis and its context
length is not interpretable.

## Budget floors from mandatory retained windows

SnapKV always retains its observation window, so with the protocol's 32-token
minimum the smallest coherent budget is 33 entries. Budgets below that are not
merely unusual, they are undefined for the method: the window alone exceeds them.

This is a property of any method with a mandatory retained region, and it is
recorded here because it constrains comparison. A method that must retain a
window cannot be compared against one that need not at budgets near the floor,
because at 33 entries the first is spending its entire budget on a fixed region
while the second is still selecting. Comparisons at very low budgets must either
be restricted to methods with the same floor, or report the floor alongside the
result. This will apply directly when PyramidKV and Ada-KV are added.

## Intra-group aggregation

On MHA there is one index per head and each query head owns its KV outright. On
GQA a KV entry is shared by every query head in its group, so an eviction score
must be formed from several query heads' preferences, and those heads may
disagree. The reference implementations never faced this because the group size
was 1.

Two consequences run through the code. Selection is per KV head, which requires a
`gather` rather than an `index_select`, because no single per-sequence index
serves all heads. And the per-query-head scores must be collapsed onto their
shared entry by an explicit rule:

- **`sum` (default).** Total attention mass the group directs at the entry. An
  entry that matters a little to all six heads outranks one that matters greatly
  to a single head, which matches the fact that evicting it harms all six.
- **`max`.** Retain the entry if any head in the group needs it. Protects
  minority heads, at the cost of spending budget on entries most of the group
  ignores.
- **`mean`.** For **uniform** group sizes this is `sum` divided by a constant, a
  monotone transform that leaves the top-k ranking unchanged. Verified
  empirically: `mean` and `sum` produce bit-identical generations. It is not an
  independent alternative on such a model, which halves the ablation to `sum`
  versus `max`. The equivalence depends on uniformity and would break on a model
  with ragged group sizes, where the divisor differs per group and the ranking
  can move; it is retained for that case.

This is a design decision, not a detail, and it has no correct answer inherited
from the literature. Observation-window scores are therefore cached **per query
head**, shape `[q_heads, L]`, and the reduction is applied at policy time. That
costs roughly 5% more sidecar at 16K and makes the aggregation an ablation that
re-runs against existing prefill caches instead of forcing a re-prefill.

The choice is measurable: at 2048 context and 3% budget, `sum` and `max` retain
different entries and produce different generations from the same cache. Whether
intra-group disagreement costs accuracy is an open measurement, and it is the
part of this study with no counterpart in the MHA results being reproduced.

## Prefill cache equivalence

The harness reads a prefill from disk when one exists for the exact identity
`(model, dtype, context, window, prefill_chunk, seed)`, and rejects a cache built
under any other configuration rather than silently reusing it.

Equivalence is verified through the harness itself, not only at the tensor level:
the same samples run live and then from cache must produce identical rows. The
check compares generated text as well as grades, because matching only on
correctness would pass while the underlying generations differed. At 2048 context
and 3% budget the two runs agree on every field, including the incorrect answers
— `60941`, `84608`, `1731`.

The incorrect answers are the load-bearing part of that comparison. A
grade-only check passes whenever both paths happen to fail, which at a harsh
budget is most of the time, so it would certify a broken cache path as sound. A
hallucinated code is a function of precisely which keys survived eviction, so
agreement on it constrains the cache contents rather than only the verdict.

## Context range

**The primary arm is capped at 16K. This is a methods decision, not a memory
limit.**

Qwen2.5-1.5B's `max_position_embeddings` is 32768. That is the top of its
trained range rather than a hard wall — a 32K run executes — but it sits where
RoPE extrapolation begins degrading quality, and that degradation is
indistinguishable from eviction damage in a quality-versus-budget plot. The
confound would land precisely on the most interesting budget point.

4K–16K is also the range most published eviction methods report in, so the
comparison stays fair. 32K measurements belong to the Llama arm, whose ceiling
is 131072 and which therefore has real margin.

## Prefill caching and which methods it may serve

Prefill is attention-bound at the top of the studied range, so the full
post-prefill state is cached once per (model, context, sample) and reused across
every method and budget rather than recomputed.

A cache entry is `(K, V, prompt_score_accumulator, Q_window)`. The two sidecar
tensors are what let a cached entry serve scoring methods: the accumulated
attention over the prompt is a pure function of prefill, as are the
observation-window queries. Together they add about 3.4% to the cache size at
16K, which is not a reason to omit them.

**The boundary that decides eligibility is not prompt-versus-decode scoring. It
is whether a method evicts during prefill.**

- A method that applies its budget *after* prefill can always be served from a
  cached full prefill. Its scoring inputs are prefill outputs, whether it reads
  the prompt tail (SnapKV) or an accumulator over the whole prompt (H2O's prompt
  phase). H2O's decode-phase accumulation stays live, but decoding is live in
  every design, so nothing is lost.
- A method that evicts *during* prefill — any streaming variant that bounds its
  own peak memory as it goes — never observes the full cache. Feeding it a cached
  full prefill silently benchmarks a different algorithm. Such a method must run
  live prefill, and no sidecar changes that.

Every method added to the study is classified against this boundary before it is
run from the cache. The distinction is easy to miss precisely because running the
wrong one from a cache produces plausible output rather than an error.

### Cache round trip is verified, not assumed

`scripts/cache_roundtrip_check.py` checks the round trip end to end across two
separate processes, because a same-process comparison can share state that hides
the failure. It asserts a bitwise field-by-field fingerprint of the reconstructed
cache and exact equality of decoded token ids against a live prefill.

The check is end-to-end rather than tensor-level because the three failure modes
that matter all survive a tensor comparison: non-contiguous tensors whose backing
storage serialises differently than intended, an incompletely reconstructed
`Cache` object that still runs while missing per-layer state, and — the one that
actually bites — the logical position on resume.

A loaded cache has no memory of what position the next token should claim. When
the entry was evicted before saving, cache length and logical position differ,
and defaulting to cache length reintroduces the RoPE desynchronisation from a
different direction. The logical position is therefore persisted in the sidecar.
Verified: at 2048 tokens with half the cache evicted, resuming at the correct
position reproduces the live continuation exactly, while resuming at the cache
length does not. Both continuations, verbatim, from the same cache and the same
first token:

    correct position (2048):  " Bletchley Park drove the construction of Colossus, the first programmable electronic computer."
    naive position   (1024):  " the first programmable electronic computer. Charles Babbage designed the the the the the the the the the"

No error is raised in either case. The second is fluent for a clause before
collapsing into repetition, which is what makes this class of bug expensive: it
looks like a working system for long enough to be mistaken for eviction damage.

Note also what the un-evicted case does. Without eviction, cache length and
logical position coincide, the naive resume matches exactly, and the test
certifies the persisted position as unnecessary — immediately before a harness
begins saving post-eviction states under a guarantee that was never exercised.
Both paths are therefore checked.

## Tasks: what is diagnostic and what is reported

**The methods lesson from this repository's own retraction.** The synthetic
needle produced a clean, reproducible result that held across an eightfold change
in context, and it was wrong — wrong in the sense that it was a property of the
task rather than of eviction, and it did not replicate on RULER. It was never
noisy or unstable. That is the more dangerous failure, because instability
announces itself and a stable artefact does not. A study reporting it as a
retrieval result would have shipped a confident false claim about the budget axis
that the entire field reports on. The separation of diagnostic from benchmark is
what caught it, and the cost of not separating them would have been the paper.

The synthetic needle task in `scripts/harness.py` builds its own prompt — filler
text, a five-digit code at a randomised depth, a question asking for it back —
and grades by exact substring match. It is a **diagnostic, not a reported
benchmark**. No number from it is comparable to a published result, because no
one else runs it.

It is retained because that exactness is what makes the pipeline debuggable. The
harness's first run scored 0.00 on the full-cache baseline, which is impossible
if the pipeline is intact, and that unambiguity located three silent bugs. When a
benchmark score later looks strange, the synthetic task is run first to establish
whether the pipeline is sound before eviction is blamed. A fuzzy grader cannot
serve that purpose.

Reported retrieval numbers come from RULER, which is generated from a config
rather than downloaded and defines standard variants; the synthetic task is
approximately RULER's single-needle case. Reported generation-quality numbers
come from LongBench, whose metrics are F1 and ROUGE rather than exact match.
Neither is wired up yet.

## The budget axis is task-dependent

Budgets are specified as an absolute count of retained KV entries, but **which
axis a result should be read on depends on the task, and this was established
only after an earlier conclusion here proved wrong.**

On the *synthetic needle diagnostic*, retrieval tracked the absolute retained
count and was essentially independent of context length: contexts of 2048 and
16384 transitioned at the same absolute budgets, 0.00 at 45 entries and 1.00 at
181, despite an eightfold difference in context. That was recorded here as a
general result about retrieval. It is not one.

On **RULER `niah_single_1`, the reported benchmark, it does not replicate.**
Same model, same policy, n=20, relative to each context's own full-cache
baseline:

| matched on | 2048 | 16384 | gap |
|---|---|---|---|
| 181 entries | 0.80 | 0.28 | +0.52 |
| 362 entries | 0.90 | 0.33 | +0.57 |
| 8.86% of context | 0.80 | 0.72 | +0.08 |
| 17.70% of context | 0.90 | 0.89 | +0.01 |

Matched on absolute entries the two contexts disagree by more than half the
score. Matched on percentage they agree to within 0.08, and at the larger budget
to within 0.01. On RULER the proportional axis is approximately correct and the
absolute axis is badly wrong — the opposite of the diagnostic task.

A query-localisation mechanism was proposed for this: that the synthetic needle
names a city occurring nowhere else, so the query concentrates on a fixed set of
entries at any context, whereas RULER's adjective-noun key against a homogeneous
haystack spreads the relevant mass proportionally. **It was tested and it is
wrong.**

The measurement is taken from the scores `prefill_with_scores` produces — the
summed 32-token observation window, reduced over the GQA group — because that is
the signal SnapKV ranks on. Entries needed to cover 50% and 90% of that mass,
median over layers, heads and 6 samples:

| task | context | k50 | k90 | k90 as % of context |
|---|---|---|---|---|
| synthetic needle | 2048 | 5 | 183 | 8.92% |
| synthetic needle | 16384 | 5 | 621 | 3.79% |
| ruler niah_single_1 | 2048 | 7 | 56 | 2.71% |
| ruler niah_single_1 | 16384 | 8 | 76 | 0.46% |

Across an eightfold context increase k90 grows 3.39x on the diagnostic and 1.36x
on RULER; proportional spreading would predict roughly 8x for the task following
the proportional axis. The result is not merely negative, it is inverted: the
diagnostic, which follows the **absolute** axis, is the task whose scoring mass
spreads with context, while RULER, which follows the **proportional** axis, stays
concentrated. Concentration of the policy's own scoring signal does not explain
the axis difference in either direction.

Nor does k90 predict where a task's budget transition sits. The diagnostic's k90
at 2048 is 183 against a transition at 181 entries, which looks like a match, but
at 16384 its k90 is 621 while the transition stays near 181. RULER's k90 at 16384
is 76 while it needs roughly 2900 entries to recover 0.80. The apparent agreement
at one point is coincidence.

**Superseded measurement.** An earlier version of this test used the attention of
the final prompt position rather than the observation-window scores, and reported
k90 growth of 0.99x and 1.34x. It rejected the hypothesis too, but against a
signal the policy does not use, so it was a rejection of a claim nobody was
making. It is recorded here because the correction matters: a finding resting on
the wrong measurement is the same class of error as the retracted absolute-budget
result, clean and reproducible and about something other than what it claims.

Two things follow. **No general claim is made about which axis is correct.** It
is a property of the task, and any result quoting a budget must say which axis it
was measured on and at what context length. And **the synthetic needle's
behaviour does not transfer to RULER**, which is the clearest possible argument
for why it is a diagnostic and not a benchmark: it was giving a clean, stable,
reproducible answer to a question, and the answer was specific to itself.

### The failure is downstream of selection

That measurement has now been made, and it depends on no hypothesis about
attention. `--measure-recall` applies exactly the selection `policy_snapkv`
applies -- same group reduction, same pooling, same forced window, same top-k --
and asks whether the answer's own token positions are among the kept indices. The
target is located by token-subsequence search against the ids that were actually
prefilled, so no decode/encode round trip can shift a boundary.

RULER `niah_single_1`, SnapKV, n=20. Selection recall is the fraction of
(layer, KV head) units retaining *every* token of the answer:

| budget | 2048 recall | 2048 accuracy | 16384 recall | 16384 accuracy |
|---|---|---|---|---|
| 91 | 0.163 | 0.000 | 0.114 | 0.000 |
| 181 | 0.830 | 0.800 | 0.793 | 0.250 |
| 362 | 0.893 | 0.900 | 0.856 | 0.300 |
| 1024 | — | — | 0.885 | 0.550 |
| 2900 | — | — | 0.914 | 0.800 |

**Selection recall is context-invariant; accuracy is not.** At 181 entries the two
contexts differ by 0.037 in recall and 0.550 in accuracy; at 362, by 0.037 and
0.600. SnapKV retains the target at essentially the same rate at both context
lengths, and the model can only use it at the shorter one.

So the 16K failure is not a ranking failure and not a failure to retain the
answer. It is downstream of selection. Note also that at 16384 the accuracy climb
from 0.250 to 0.800 between 181 and 2900 entries is accompanied by a recall climb
of only 0.793 to 0.914: the additional 2,700 entries buy 0.55 of accuracy while
buying 0.12 of target retention. Whatever those entries provide, it is mostly not
the answer's own keys.

Retaining a fact is therefore not sufficient for using it, and a budget's effect
at long context is not principally about whether the target survives. Any account
of eviction damage that reasons only about whether important tokens are kept is
incomplete.

**Partial retention is worthless.** At 91 entries, some token of the answer
survives in 85-89% of units while every token survives in 11-16%, and accuracy is
0.000 at both contexts. The distinction between "any" and "all" is the whole
signal; a recall metric that counted partial hits would have reported 87% success
at a budget that retrieves nothing.

### Keys are not preferentially evicted either

The obvious mechanism was that the needle's *key* -- what the query matches on --
might be evicted while its *value* survives, leaving the model holding the answer
with no way to bind it to the question. Traced with `--recall-target key`:

| budget | value recall 2K / 16K | key recall 2K / 16K | accuracy 2K / 16K |
|---|---|---|---|
| 91 | 0.163 / 0.114 | 0.746 / 0.755 | 0.000 / 0.000 |
| 181 | 0.830 / 0.793 | 0.942 / 0.921 | 0.800 / 0.250 |
| 362 | 0.893 / 0.856 | 0.963 / 0.933 | 0.900 / 0.300 |

Keys survive at essentially the same rate at both context lengths -- differing by
2.1 points at 181 entries where accuracy differs by 0.550 -- and they survive
*better* than values, not worse. The mechanism is not key eviction.

(The `any-token kept` figure is 100% for keys at every budget and should not be
read as a result: the key also appears in the question, which sits inside the
forced observation window. `all-tokens kept` is the meaningful column, since it
requires the needle's own occurrence to survive as well.)

**Three hypotheses have now been tested and rejected** -- query localisation,
scoring-mass concentration, and key eviction -- against a confirmed finding that
the failure is downstream of selection. No fourth is pursued. The remaining
candidate is how attention over a compacted cache reallocates its mass, which is
harder to probe and is not chased here. The finding stands without a mechanism:
retaining a fact is not sufficient for using it, and at long context the budget's
effect is mostly not about whether the target survives.

The grid remains absolute — 32, 45, 64, 91, 128, 181, 256, 362, 512, 1024 — with
percentage-matched points added when contexts are compared, since neither axis
can be assumed.

## Measurement

- Each model is compared only against its own full-cache baseline, never against
  another model. Cross-model numbers are not produced.
- Memory is the headline metric. Latency is secondary, because the hardware
  thermally throttles.
- Every latency number includes predictor and selection overhead.
- Fixed seeds. Every run configuration is logged to `results/` as JSON.
- No number appears in this repository that was not produced by a run in it.

## Hardware

Single RTX 4060 laptop, 8GB VRAM, 16GB system RAM. No cloud, no API models.

Models are pinned to the GPU with `device_map={"": 0}`, never `device_map="auto"`,
and `next(model.parameters()).device` is asserted to be `cuda:0` before any
benchmark run. Silent CPU offload would place part of the KV cache in system RAM
and invalidate every memory and latency figure, so the load is made to fail
loudly instead.
