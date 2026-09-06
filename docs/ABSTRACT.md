# Abstract

KV-cache eviction methods are built on a common premise: identify the tokens that
matter and keep them. We test that premise directly on a 1B-class GQA model
running on a single 8GB consumer GPU, and find it incomplete.

Measuring whether an answer's own cache entries survive selection — by applying
the identical top-k the policy applies — we find that **selection recall is
invariant to context length while accuracy is not**. On RULER `niah_single_1`
with SnapKV at a budget of 181 entries, the answer's entries survive in 0.822 of
attention units at 2K context and 0.795 at 16K, a difference of 0.027, while
accuracy falls from 0.867 to 0.433. Both figures come from samples used for no
selection of any kind, and recall reproduces to within 0.02 of the original
measurement on every cell. Raising the 16K budget from 181 to 2,900
entries lifts accuracy by 0.55 and target retention by only 0.12. **Most of what
a larger budget buys is not the answer's own keys.** Retaining a fact is not
sufficient for using it, and an account of eviction damage that reasons only
about which tokens are kept cannot explain this.

We tested three mechanisms and rejected all three: query localisation,
concentration of the policy's own scoring mass, and eviction of the needle's key
as distinct from its value. One is rejected in the inverted direction — the task
that follows an absolute budget axis is the one whose scoring mass spreads with
context, not the one that stays concentrated. We report the effect without a
mechanism rather than supply one.

Against that result we evaluate three allocation strategies at equal total
retained entries. **Reallocating budget across KV heads (Ada-KV) does nothing**:
across 80 held-out samples the effect is −0.005 with a 95% interval of
[−0.026, +0.016], bounding any true benefit below 0.026. This is consistent with the failure being downstream of
selection, since head-level dials only change which entries selection keeps.
Reallocating across layers (PyramidKV) gives a **small real gain**: +0.075 on the
samples the budget grid was chosen from falls to **+0.040 (95% CI [+0.004,
+0.076])** across 80 held-out samples — real, but roughly half the
selection-contaminated estimate.

Three further results bear on how this literature is evaluated. The correct
budget axis — absolute retained entries versus fraction of context — is a
property of the task, not of eviction; a claim to the contrary derived from our
own synthetic diagnostic failed to replicate on RULER and is retracted here.
Retrieval against distractors that resemble the target survives nothing below
half the cache, where published budgets are single-digit percentages. And
LongBench QA cannot measure eviction at this model scale: with the context fully
evicted, 30–40% of answers are byte-identical to the full-cache answer, because
the model is answering from parametric knowledge. Eviction cannot damage what the
model was not using.

The setting is what makes the head-allocation result sharp. On grouped-query
attention a KV entry is shared by every query head in its group, so the unit of
allocation is the group: this model has 2 dials where the methods assume 32. That
narrowing is not incidental — across published models the head-budget lever spans
32 dials under MHA to 2 under aggressive GQA, and per-head allocation methods
were validated at the wide end.

All results are single-model. The two central findings carry held-out
replications at n = 29–80; the remainder are n = 20. We report
findings graded by what the evidence supports, including one retraction and one
withdrawal, and document five implementation traps — position handling on resume
after eviction, layer-varying budgets under a shared causal mask, attention-kernel
registration that silently skips mask construction, fp16 overflow, and LongRoPE
degeneration past its boundary — each of which produces plausible output rather
than an error, and any of which would silently corrupt a reproduction.

Code, protocol and per-run logs: https://github.com/thulasiramk-2310/kural-kv
