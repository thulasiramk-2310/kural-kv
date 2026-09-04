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
