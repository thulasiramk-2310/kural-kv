# kural-kv

Reproduction study of KV-cache eviction methods on small GQA models.

The protocol every measurement is produced under is recorded in
[docs/METHODS.md](docs/METHODS.md).

## What this study is asking

It began as a transfer question: published eviction findings are established
mostly on 7B MHA models, and this asks whether they hold on a 1B-class GQA model
that fits on one consumer GPU. That question is still here, and the GQA framing
below is still the reason it is worth asking.

The measurements have moved it somewhere more general. **Every scoring heuristic
in this literature is built on identifying important tokens and keeping them.**
This repository now has a direct measurement saying that frame is incomplete:
the answer's own entries survive selection at essentially the same rate at 2K and
16K context, and the model can only use them at 2K.

| budget | 2048 recall / accuracy | 16384 recall / accuracy |
|---|---|---|
| 181 entries | 0.830 / 0.800 | 0.793 / **0.250** |
| 362 entries | 0.893 / 0.900 | 0.856 / **0.300** |

Selection recall differs by 0.037 where accuracy differs by 0.550. And within
16K, raising the budget from 181 to 2900 entries buys 0.55 of accuracy while
buying only 0.12 of target retention — so most of what a larger budget provides
is not the answer's own keys.

So the question this repository is now answering is: **what actually governs
eviction damage?** Keeping the important tokens is measurably not the whole of
it. Three candidate mechanisms have been tested and rejected — query
localisation, scoring-mass concentration, and key eviction — and the finding is
reported without one, in
[docs/METHODS.md](docs/METHODS.md).

## Why GQA is the right setting for it

Published KV-cache eviction methods — Ada-KV, LKV, LU-KV — allocate cache budget
**per attention head**. On the MHA models they were developed against, mostly 7B,
that unit is well defined: every query head owns its own KV entries, so a per-head
budget is directly actionable.

The small models this study runs are GQA. On Qwen2.5-1.5B, two KV heads serve
twelve query heads, in groups of six. A KV entry is shared by every query head in
its group, so evicting it removes it for all six at once. The unit of eviction is
the group, not the head, and a query head cannot be given a budget different from
its groupmates.

Two consequences follow:

1. Head-level budget allocation — reported in those papers as the dominant
   factor — has **two dials on this model** rather than twelve. Whether that
   matters is a measurement, not a deduction.
2. Within a group, the query heads may disagree about which tokens matter. An
   eviction that is cheap for one can be expensive for another, and a single
   group-level decision cannot satisfy both.

On the primary arm, Qwen2.5-1.5B, that leaves **`budget_units: 2`** — twelve
query heads and two dials. Ada-KV's premise is that heads differ in attention
concentration, so budget moves from concentrated heads to dispersed ones. When
six query heads share one KV group, and two of the six are concentrated while
four are dispersed, no allocation serves both. The allocation is not per head.
It is per committee.

LKV's headline result — that the learned global budget contributes more than the
selection policy itself — meets the same narrowed lever.

**Stated before the run, so it is a prediction rather than a rationalisation.**
These two results compose rather than conflict: Ada-KV has two dials on this
model, and the recall finding says the thing those dials control — which entries
selection keeps — is not what binds at long context. So the prediction is *little
or no gain over SnapKV at matched total retained entries at 16K*.

All four outcomes are enumerated here **before the sweep finished**, so that
whichever lands is read against a written expectation rather than explained
afterwards:

| Ada-KV (head axis) | PyramidKV (layer axis) | reading |
|---|---|---|
| flat | gains | prediction holds; the two axes separate cleanly |
| gains | gains | prediction wrong; both axes do something despite two dials |
| gains | flat | prediction wrong in the *opposite* direction — head reallocation matters and depth does not, which contradicts the 2K/16K pattern already observed and would need explaining, not accommodating |
| flat | flat | neither axis matters at this scale; allocation is not the lever, consistent with the downstream-of-selection finding |

The third row is the one that would be easiest to rationalise after the fact and
hardest to justify, so it is written down with that noted.

### How wide is the lever, across models

Per-head budget allocation is only actionable per KV group, so the number of
dials an allocator has is `num_key_value_heads`. Reproduced by
`scripts/budget_units.py`, which reads `config.json` only — no weights, no GPU:

| model | attn | layers | q heads | kv heads | group | budget units |
|---|---|---|---|---|---|---|
| microsoft/Phi-3.5-mini-instruct | MHA | 32 | 32 | 32 | 1 | **32** |
| mistralai/Mistral-7B-Instruct-v0.3 | GQA | 32 | 32 | 8 | 4 | **8** |
| Qwen/Qwen2.5-7B-Instruct | GQA | 28 | 28 | 4 | 7 | **4** |
| Qwen/Qwen2.5-3B-Instruct | GQA | 36 | 16 | 2 | 8 | **2** |
| Qwen/Qwen2.5-1.5B-Instruct | GQA | 28 | 12 | 2 | 6 | **2** |
| Qwen/Qwen2.5-0.5B-Instruct | GQA | 24 | 14 | 2 | 7 | **2** |

The head-budget lever shrinks 16x from MHA to aggressive GQA. Published per-head
allocation methods were validated at the wide end of that range. Llama-3.2-1B
sits at 8 by specification; it is absent from the table because its repo is
gated and this project does not record numbers it has not read.

## Method

- **Qwen2.5-1.5B-Instruct** — bf16, GQA. Primary arm. 4K–16K context.
  12 query heads over 2 KV heads: group size 6, so 2 budget units.
- **Llama-3.2-1B-Instruct** — fp16, GQA. Second arm, 4K–32K context.
  32 query heads over 8 KV heads: group size 4, so 8 budget units.
- **Phi-3.5-mini** — 4-bit NF4, MHA. Contrast arm. 4K–12K context.

Each model is compared only against its own full-cache baseline, never against
another. The MHA arm exists to show what the same method does when per-head
budgeting is genuinely available, not to produce a cross-model comparison.

Two GQA arms at different group sizes, 6 and 4, are what let the study say
anything about group width rather than about one model's quirks.

### Why the primary arm is capped at 16K

Capped at 16K to stay clear of Qwen2.5-1.5B's 32768 position ceiling.

32768 is the top of the model's trained range, not a hard wall — a 32K run
executes. But it sits where RoPE extrapolation begins degrading quality, and
that degradation is indistinguishable from eviction damage in a quality-vs-budget
plot. The confound would land precisely on the most interesting budget point, so
the range stops short of it deliberately.

16K also leaves the comparison fair: 4K–16K is the range most published eviction
methods report in. The Llama arm, whose ceiling is 131072, carries the 32K
measurements where there is real margin.

This cap is a methods decision, not a memory limit.

Memory is the headline metric. Latency is reported but secondary, since the
hardware thermally throttles, and every latency figure includes predictor and
selection overhead.

Attention is run with `attn_implementation="eager"` throughout, because fused
kernels do not expose the attention scores the eviction policies score against.
Scoring uses a 32–64 token observation window rather than the full attention
matrix.

Precision is 2 bytes per element; which 2-byte dtype is a per-model numerical
choice, not a study parameter. See [docs/METHODS.md](docs/METHODS.md).

All runs are on a single RTX 4060 laptop, 8GB VRAM. No cloud, no API models.

## Methods note: RoPE phase under eviction

`cache_position` and `position_ids` are the same number in every ordinary forward
pass, so `transformers` collapses them and defaults `position_ids` to
`past_key_values.get_seq_length()`. Eviction is the one operation that breaks
that identity.

RoPE is applied to K before it enters the cache. After dropping half a
512-entry cache, physical slot 256 holds a key with phase 510, while the next
token arrives claiming phase 256. Every retained key past the midpoint then sits
at a wrong relative distance to the query. The model still produces fluent text.
It simply gets worse, and the damage is indistinguishable from "eviction hurt
quality."

The consequence for the literature is the part worth stating plainly: **a
reproduction that gets this wrong reports eviction as more damaging than it
actually is.** It is therefore a candidate explanation for variance between
published results, not merely an implementation detail.

This repository keeps the two indices separate — `cache_position` for the
physical slot, `position_ids` for the original sequence position — and gate 5 of
`scripts/gate_check.py` asserts the invariant on every run by comparing
surviving keys bitwise against a pre-eviction reference, so a regression fails
loudly instead of degrading quietly.

Planned: measure the same method at the same budget under both a naive and a
RoPE-correct implementation. If the curves separate measurably, the size of that
gap is a result in its own right.

## Methods note: numerical validity is checked, not assumed

Qwen2.5 has activation outliers that overflow fp16. The model then emits NaN
logits on a plain 256-token forward — no cache, no chunking, no eviction — while
every structural property stays correct: shapes valid, nothing crashes, RoPE
phases intact, memory sweep clean. `argmax` silently returns token 0 forever.

The only visible symptom is degenerate text, which in an eviction study reads as
"eviction hurt quality." So `scripts/gate_check.py` asserts finite logits and
attentions as its own gate, and this rule holds for every run in the repository,
not only gate checks: a run producing a non-finite intermediate fails loudly
rather than writing a row.

Related: probes use real text. Random token ids cannot distinguish a working
model from a broken one, which is precisely how this defect stayed hidden.

## Prefill cost

Measured on the primary arm, bf16, **`prefill_chunk=128`**, warm weights, one
discarded warmup iteration and three timed repeats per length
(`scripts/prefill_timing.py`, `results/prefill_timing.json`):

| context | median | ratio vs previous | local exponent | peak GiB |
|---|---|---|---|---|
| 1024 | 0.339s | — | — | 2.93 |
| 2048 | 0.568s | 1.68x | 0.75 | 2.98 |
| 4096 | 1.365s | 2.40x | 1.26 | 3.08 |
| 8192 | 3.969s | 2.91x | 1.54 | 3.27 |
| 16384 | 13.316s | 3.36x | **1.75** | 3.65 |

**Prefill is attention-bound at the top of the range.** The local exponent rises
monotonically to 1.75 across 8K→16K, which is the quadratic attention term
overtaking the linear per-token work. Prefill caching is therefore mandatory, not
an optimisation.

The global log-log fit is 1.34 and is *not* the number to quote. At small
contexts prefill is dominated by fixed per-chunk overhead, and the smaller the
chunk the more chunks there are, so the low end drags the global fit toward
linear regardless of how the attention term is growing. The exponent at the top
of the measured range is the signal.

Repeat spread is under 2% from 2K up, and resident allocation between repeats is
flat to 0.008 GiB, so these are prefill cost and not allocator behaviour.

**This supersedes the prefill timings in the gate-check log**, which were taken
while that run was also downloading weights and was thrashing the allocator. It
recorded 49.3s at 16K; the same context now measures 13.3s, 3.7x faster despite a
four-times-smaller chunk.

### 16K requires `--prefill-chunk 128`

At 16384 tokens, `prefill_chunk` 512 and 256 both fail on this GPU — 512 with a
driver-level `CUDA error: out of memory`, 256 with
`CUBLAS_STATUS_EXECUTION_FAILED` — while `nvidia-smi` reports ~7.6 GiB free. The
failure is not exhaustion. The eager score matrix is the largest single
allocation in the run, and the allocator cannot obtain a contiguous block of that
size even though the total is available.

`expandable_segments:True` does not fix it, which places the fragmentation below
PyTorch's caching allocator rather than inside it. Halving the chunk quarters the
score matrix, and 128 clears it: verified over three in-process repeats and three
independent cold processes, six of six, with peak allocation of 3.65 GiB — lower
than the 4.597 GiB the failing chunk-512 configuration reached.

So 16K is reliable, not "16K sometimes", but only at chunk 128. Benchmarks in
this repository fix the chunk at 128 for the whole sweep so the range is
uniformly reachable.

## Planned: prefill caching

One prefill per (model, context, sample) serves every method and every budget,
because prefill produces the full KV cache and each policy then reduces that same
cache. A 25-configuration sweep collapses to one prefill pass plus 25 cheap
decodes. At 13.3s per 16K prefill and an exponent of 1.75, prefill is the
dominant cost of any sweep that does not cache it. At 0.44 GiB per 16K sample, 100 samples is ~44 GB — affordable against
200 GB if deleted per task. Cache K and V as stored; never cache attentions,
which are recomputable and enormous.

This is only valid for methods that score from the prompt alone. SnapKV
qualifies, since its observation window is the prompt tail. H2O accumulates
attention across decoding and needs live scores, so a cached prefill would
silently benchmark something that is not H2O. Verify per method before reusing a
cache.

## Status

Nothing is measured yet.

`scripts/gate_check.py` verifies the environment assumptions the study rests on
before any eviction code is written: CUDA and VRAM, a 2-byte eager load pinned to
GPU, attention scores actually returned, the 8-vs-32 head counts that make this a
GQA study, eviction followed by continued decoding, and peak memory across
context lengths. It writes `results/gate_check.json`.

```bash
# Primary arm: Qwen2.5-1.5B-Instruct, 4K-16K
python scripts/gate_check.py

# Llama arm, once access is granted: the only arm that goes to 32K
python scripts/gate_check.py --model-id meta-llama/Llama-3.2-1B-Instruct \
    --context-lengths 4096 8192 16384 32768
```

If the sweep OOMs, it is the eager score matrix during prefill, not the KV cache.
Lower `--prefill-chunk` before changing anything else.

Two constraints that gate check established and that any later script inherits:

- Eager attention materialises a `[1, 32, q_len, kv_len]` score matrix, so a
  single-shot long prefill will not fit in 8GB. Long contexts are prefilled in
  chunks. The resulting cache is bit-identical; only peak activation differs.
- RoPE is applied to K before it enters the cache, so surviving entries keep
  non-contiguous original positions after an eviction. Continuation must be
  indexed by original position, not by compacted cache length, or the gaps are
  silently closed and quality degrades invisibly.

No number appears in this repository that was not produced by a run in it.
