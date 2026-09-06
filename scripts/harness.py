#!/usr/bin/env python3
"""Eviction benchmark harness.

Narrow on purpose: one task, one baseline, one method, producing one row per
configuration. The harness has more moving parts than anything else in the study
-- sample construction, prefill, scoring, policy application, decode, grading,
logging -- so the path is made to work before methods are added. If five methods
were added at once and the numbers looked wrong, there would be no way to tell
which layer was lying.

Loop order is sample-outer, config-inner, and that is load-bearing rather than
stylistic. A cached 16K entry costs ~2.2s to read and rebuild on the GPU, which
is dominated by the host-to-device rebuild rather than the NVMe read. Config-outer
would reload the same sample once per configuration: at 100 samples x 25 configs
that is 2500 loads and about 1.5 hours of pure loading. Sample-outer loads each
entry once. Reversing this later means rewriting the loop, so it is fixed now.

Every policy runs against a clone, and the source cache is asserted unchanged
afterwards. Without that, configuration 2 silently runs on configuration 1's
leftovers and the second row of every sweep is wrong.
"""

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from gate_check import DTYPES, MODEL_ID, decode, load_model
from longbench import LongBenchTask
from ruler import RulerTask

REPO_ROOT = Path(__file__).resolve().parent.parent

FILLER = [
    "The archive room held decades of municipal records in numbered boxes.",
    "Rain moved across the valley in slow grey sheets that afternoon.",
    "The committee met on Tuesdays and rarely finished before dark.",
    "Copper pipes ran the length of the basement, warm to the touch.",
    "Someone had left a bicycle leaning against the loading dock again.",
    "The ledger was written in three different hands across four decades.",
    "Wind pushed the shutters open twice before anyone thought to latch them.",
    "A card index by the window listed every tenant since the war.",
]
NEEDLE = "The secret access code for {city} is {code}."
QUESTION = "What is the secret access code for {city}? Answer with the number only."
CITIES = ["Madurai", "Trichy", "Salem", "Erode", "Vellore", "Thanjavur", "Kanchipuram"]


# --------------------------------------------------------------------------
# Task
# --------------------------------------------------------------------------
def build_sample(tokenizer, context_len, seed, device):
    """A retrieval task whose grading is exact.

    Eviction damage is hard to read off a fluency metric, because a model that
    has lost the relevant keys still produces fluent text. Retrieval makes the
    failure legible: either the code survives in the cache and is recalled, or it
    does not. Depth is randomised per sample so a policy cannot score well by
    favouring one region of the context.
    """
    rng = random.Random(seed)
    city = rng.choice(CITIES)
    code = str(rng.randint(10000, 99999))
    body = []
    while True:
        body.append(rng.choice(FILLER))
        if len(tokenizer(" ".join(body)).input_ids) > context_len:
            break
    depth = rng.random()
    body.insert(max(1, int(len(body) * depth)), NEEDLE.format(city=city, code=code))
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": " ".join(body) + "\n\n"
          + QUESTION.format(city=city)}],
        tokenize=False, add_generation_prompt=True)
    ids = tokenizer(text, return_tensors="pt").input_ids[:, -context_len:].to(device)
    return {"input_ids": ids, "code": code, "city": city,
            "depth": round(depth, 3), "seed": seed}


class NeedleTask:
    """The synthetic needle, wrapped to match the LongBench task interface.

    A diagnostic, not a reported benchmark: the prompt is built here, so no
    number from it is comparable to a published one. It is kept because its
    grader is exact, which is what makes a broken pipeline unambiguous.
    """

    name = "needle"
    metric_name = "exact"
    max_gen = 16

    def __init__(self, tokenizer, context_len, seed):
        self.tok, self.context_len, self.seed = tokenizer, context_len, seed

    def __len__(self):
        return 10 ** 9      # generated on demand

    def sample(self, i, device):
        s = build_sample(self.tok, self.context_len, self.seed + i, device)
        return {"input_ids": s["input_ids"], "reference": s["code"],
                "max_gen": self.max_gen, "id": "needle-%d" % s["seed"],
                "meta": {"depth": s["depth"]}}

    def score(self, text, reference):
        return 1.0 if reference in text else 0.0


def truncate_at_stop(tokens, stop_ids):
    """Cut the generation at the first stop token.

    Without this the model answers, emits <|im_end|>, and then keeps going into
    template noise, which pollutes both the grading string and the eyeball check.
    """
    for i, t in enumerate(tokens):
        if t in stop_ids:
            return tokens[:i]
    return tokens


def sprint(line):
    """Print, surviving a console that cannot encode the model's output.

    A Windows cp1252 stdout raises UnicodeEncodeError on generated text
    containing non-Latin-1 characters. That killed a 16K sweep after three
    samples, which is an expensive way to lose a run to a print statement.
    """
    try:
        print(line)
    except UnicodeEncodeError:
        print(line.encode("ascii", "backslashreplace").decode("ascii"))


def pctiles(xs):
    """Median with p10/p90.

    Never a mean: selection timing carries a warmup outlier that moves a mean
    and does not move a median. Returns None when warmup has consumed every
    sample, rather than reporting a statistic from nothing.
    """
    if not xs:
        return {"sel_p50_ms": None, "sel_p10_ms": None, "sel_p90_ms": None, "sel_n": 0}
    xs = sorted(xs)
    def q(f):
        return xs[min(len(xs) - 1, max(0, int(round(f * (len(xs) - 1)))))] * 1000
    return {"sel_p50_ms": round(q(0.5), 2), "sel_p10_ms": round(q(0.1), 2),
            "sel_p90_ms": round(q(0.9), 2), "sel_n": len(xs)}


def grade(text, code):
    """Exact: the code is present in the generation or it is not."""
    return code in text


# --------------------------------------------------------------------------
# Prefill and observation-window scoring
# --------------------------------------------------------------------------
def prefill_with_scores(model, ids, chunk, window, device):
    """Prefill, and score every cached position against the observation window.

    The window is the last `window` tokens of the prompt, which is what SnapKV
    scores against. Its attention is captured once and reduced immediately to a
    per-position score; the full [q_heads, window, L] tensor is dropped, since
    keeping it would be O(L * window) for no purpose.

    Scores are kept PER QUERY HEAD, shape [q_heads, L], and the reduction over
    the GQA group happens at policy time. That costs about 5% more sidecar at
    16K and buys the intra-group aggregation ablation for free: the choice of
    sum, max or mean can be re-run against an existing cache instead of forcing
    a re-prefill.
    """
    cache = DynamicCache()
    total = ids.shape[1]
    head = ids[:, :total - window]
    out = None
    for start in range(0, head.shape[1], chunk):
        piece = head[:, start:start + chunk]
        past = cache.get_seq_length()
        out = model(input_ids=piece, past_key_values=cache,
                    attention_mask=torch.ones(1, past + piece.shape[1],
                                              dtype=torch.long, device=device),
                    cache_position=torch.arange(past, past + piece.shape[1], device=device),
                    use_cache=True, logits_to_keep=1)
        cache = out.past_key_values

    past = cache.get_seq_length()
    out = model(input_ids=ids[:, total - window:], past_key_values=cache,
                attention_mask=torch.ones(1, past + window, dtype=torch.long, device=device),
                cache_position=torch.arange(past, past + window, device=device),
                use_cache=True, output_attentions=True, logits_to_keep=1)
    cache = out.past_key_values
    scores = [a[0].float().sum(dim=1) for a in out.attentions]   # [q_heads, L]
    return cache, scores, out.logits, cache.get_seq_length()


def find_target_positions(tok, input_ids, value):
    """Token span(s) of the answer value inside the prompt.

    Searched as a token subsequence against the exact ids that were prefilled,
    rather than by re-tokenising decoded text, so there is no decode/encode
    round-trip that could shift a boundary. Both the bare and space-prefixed
    tokenisations are tried, since the value sits mid-sentence.
    """
    ids = input_ids[0].tolist()
    spans = []
    for variant in (value, " " + value):
        c = tok(variant, add_special_tokens=False).input_ids
        if not c:
            continue
        for i in range(len(ids) - len(c) + 1):
            if ids[i:i + len(c)] == c:
                spans.append((i, i + len(c)))
    # De-duplicate overlapping hits from the two tokenisations.
    spans.sort()
    merged = []
    for a, b in spans:
        if merged and a < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def target_survives(scores, kv_heads, mode, budget, window, pool_kernel, spans, L):
    """Does SnapKV's top-k retain the answer's own entries?

    Applies exactly the selection `policy_snapkv` applies -- same group
    reduction, same pooling, same forced observation window, same top-k -- and
    asks whether the target's token positions are among the kept indices.

    Separates a ranking failure from a budget failure without assuming anything
    about attention: if the target survives selection and the answer is still
    wrong, the failure is downstream of selection.
    """
    want = set()
    for a, b in spans:
        want.update(range(a, b))
    if not want:
        return None
    any_hits, all_hits, total = 0, 0, 0
    for layer_scores in scores:
        sc = reduce_group(layer_scores, kv_heads, mode).clone()
        if pool_kernel > 1:
            sc = torch.nn.functional.max_pool1d(
                sc.unsqueeze(0), kernel_size=pool_kernel, stride=1,
                padding=pool_kernel // 2).squeeze(0)[:, :L]
        sc[:, L - window:] = float("inf")
        idx = sc.topk(min(budget, L), dim=-1).indices
        for h in range(idx.shape[0]):
            kept = set(idx[h].tolist())
            hit = want & kept
            total += 1
            any_hits += bool(hit)
            all_hits += (len(hit) == len(want))
    return {"any": any_hits / total, "all": all_hits / total,
            "target_tokens": len(want), "units": total}


def score_concentration(scores, kv_heads, mode, fracs=(0.5, 0.9)):
    """Entries needed to cover each fraction of SnapKV's own scoring mass.

    Deliberately computed from the scores `prefill_with_scores` produces -- the
    summed observation-window attention, reduced over the GQA group -- because
    that is the signal the policy ranks on. An earlier version of this
    measurement used the attention of the final prompt position instead, which is
    a related but different quantity, and a hypothesis about why the policy
    behaves differently across tasks has to be tested against the signal the
    policy actually uses.

    Returns the median over layers and KV heads.
    """
    import statistics as _st
    out = {f: [] for f in fracs}
    for layer_scores in scores:
        a = reduce_group(layer_scores, kv_heads, mode).float()
        a = a / a.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        srt, _ = a.sort(dim=-1, descending=True)
        cum = srt.cumsum(dim=-1)
        for f in fracs:
            out[f] += ((cum < f).sum(dim=-1) + 1).tolist()
    return {f: _st.median(v) for f, v in out.items()}


def reduce_group(s, kv_heads, mode):
    """Collapse per-query-head scores onto their shared KV entry.

    This aggregation is a design decision with no counterpart in the MHA papers,
    where the group size is 1 and each query head owns its KV outright. Here six
    query heads share one entry and may disagree about its value, so a single
    vote must be formed from six preferences.

    sum  -- default. Total attention mass the group directs at the entry. An
            entry mattering a little to all six ranks above one mattering a lot
            to a single head, which matches the fact that eviction harms all six.
    max  -- keep the entry if ANY head in the group needs it. Protects minority
            heads at the cost of spending budget on entries most of the group
            ignores.
    mean -- sum normalised by group size; identical ranking to sum for uniform
            group sizes, kept for comparability across models whose groups differ.
    """
    q_heads, L = s.shape
    grouped = s.view(kv_heads, q_heads // kv_heads, L)
    if mode == "sum":
        return grouped.sum(dim=1)
    if mode == "max":
        return grouped.max(dim=1).values
    if mode == "mean":
        return grouped.mean(dim=1)
    raise ValueError(mode)


# --------------------------------------------------------------------------
# Cache handling
# --------------------------------------------------------------------------
def clone_cache(cache):
    new = DynamicCache()
    for i, layer in enumerate(cache.layers):
        new.update(layer.keys.clone(), layer.values.clone(), i)
    return new


def cache_signature(cache):
    """Structural + value signature, for asserting a policy did not mutate."""
    return [(tuple(l.keys.shape), float(l.keys.float().sum()),
             float(l.values.float().sum())) for l in cache.layers]


def prefill_cache_path(cache_dir, model_id, dtype, task, context, window, chunk, seed):
    tag = model_id.replace("/", "__")
    tsk = task.replace(":", "-")
    return Path(cache_dir) / f"{tag}_{dtype}_{tsk}_ctx{context}_w{window}_c{chunk}_s{seed}.pt"


def save_prefill(path, cache, scores, first, prompt_len, identity):
    """Persist a prefill. `identity` records the config that produced it, so a
    cache built under different settings is rejected rather than silently used."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "identity": identity,
        "prompt_len": prompt_len,
        # The logical position the continuation must claim. A loaded cache has no
        # memory of this, and defaulting to cache length is wrong after eviction.
        "next_position": prompt_len,
        "first_token": int(first),
        "layers": [{"keys": l.keys.contiguous().cpu(),
                    "values": l.values.contiguous().cpu()} for l in cache.layers],
        "scores": [s.contiguous().cpu() for s in scores],
    }, path)


def load_prefill(path, device, identity):
    d = torch.load(path, map_location="cpu", weights_only=True)
    # Compatibility, deliberately narrow: a key ABSENT from an older cache takes
    # the current default, because the property it names did not exist yet and
    # could only have had that value. A key that is present and differs still
    # rejects. Widening this to tolerate mismatches would silently reuse a cache
    # built under different settings, which is what the identity exists to stop.
    DEFAULTS = {"quant": "none", "min_natural_tokens": 0}
    cached = dict(d["identity"])
    for k, v in DEFAULTS.items():
        cached.setdefault(k, v)
    if cached != identity:
        raise SystemExit(
            f"cached prefill at {path} was built under a different config:\n"
            f"  cached:   {cached}\n  requested: {identity}")
    # Free each CPU tensor as it reaches the GPU. A 16K entry is ~470 MiB, and
    # holding the whole decoded payload while building the GPU copy costs that
    # twice; on a 16GB machine with the model resident that is enough for the OS
    # to kill the run mid-sweep, which it did. Entries are dropped as consumed.
    cache = DynamicCache()
    layers = d.pop("layers")
    for i in range(len(layers)):
        e = layers[i]
        cache.update(e["keys"].to(device).contiguous(),
                     e["values"].to(device).contiguous(), i)
        layers[i] = None
    sc = d.pop("scores")
    scores = []
    for i in range(len(sc)):
        scores.append(sc[i].to(device))
        sc[i] = None
    out = (cache, scores, d["first_token"], d["prompt_len"], d["next_position"])
    del d, layers, sc
    return out


# --------------------------------------------------------------------------
# Policies. Each returns a NEW cache; none mutates its input.
# --------------------------------------------------------------------------
def install_head_mask_attention(model):
    """Register an eager attention that honours a per-layer, per-head key mask.

    Ada-KV gives different KV heads different numbers of entries. A dense cache
    stores one sequence length for every head, so the shorter heads must be
    padded -- and the padding has to be excluded from attention or it is not
    Ada-KV, it is SnapKV at the larger budget.

    The model's own mask cannot express this. It builds one causal mask per
    attention *type*, shared by every layer, so it can carry neither a per-layer
    nor a per-head pattern. The mask therefore travels on the attention module
    itself, set by the policy and read here.

    Registered once, and inert unless a module carries `_head_mask`, so every
    other policy is unaffected.
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.qwen2.modeling_qwen2 import repeat_kv

    def eager_head_masked(module, query, key, value, attention_mask,
                          scaling=None, dropout=0.0, **kwargs):
        key_states = repeat_kv(key, module.num_key_value_groups)
        value_states = repeat_kv(value, module.num_key_value_groups)
        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        hm = getattr(module, "_head_mask", None)
        if hm is not None:
            # hm is [kv_heads, L] with True on padded slots; expand to the query
            # heads that share each KV head. Decoding appends a token to the
            # cache each step, and those are real entries, so the mask is
            # extended with False rather than resized.
            kvh, L = hm.shape
            grown = attn_weights.shape[-1] - L
            if grown > 0:
                hm = torch.cat([hm, torch.zeros(kvh, grown, dtype=torch.bool,
                                                device=hm.device)], dim=1)
                L = hm.shape[1]
            m = hm.repeat_interleave(module.num_key_value_groups, dim=0)
            attn_weights = attn_weights.masked_fill(
                m.view(1, kvh * module.num_key_value_groups, 1, L),
                torch.finfo(attn_weights.dtype).min)
        attn_weights = torch.nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        return attn_output.transpose(1, 2).contiguous(), attn_weights

    ALL_ATTENTION_FUNCTIONS.register("eager_head_masked", eager_head_masked)

    # Registering an attention function is NOT enough. Mask construction
    # dispatches separately, on the same name:
    #     mask_interface = ALL_MASK_ATTENTION_FUNCTIONS[config._attn_implementation]
    # An unregistered name does not raise -- it falls through to a default that
    # skips the causal mask, so prefill attends BIDIRECTIONALLY and the model
    # silently reads the whole prompt as unordered context. Decode is unaffected
    # because a single query token needs no causal mask, which is exactly what
    # makes this invisible: generation still runs, and only multi-token prefill
    # is wrong. The eager mask builder is therefore registered under the same
    # name as the attention function.
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    ALL_MASK_ATTENTION_FUNCTIONS.register(
        "eager_head_masked", ALL_MASK_ATTENTION_FUNCTIONS["eager"])

    model.config._attn_implementation = "eager_head_masked"
    return model


def clear_head_masks(model):
    for layer in model.model.layers:
        if hasattr(layer.self_attn, "_head_mask"):
            del layer.self_attn._head_mask


def adakv_allocation(scores_layer, kv_heads, mode, budget, window, pool_kernel, L):
    """Ada-KV: pool scores across heads and take one global top-k.

    SnapKV gives every head the same count. Ada-KV pools the heads' scores within
    a layer and selects globally, so a head holding more of the attention mass
    receives more entries and a flat head receives fewer. The total is the same
    `budget * kv_heads`, which is what makes it a reallocation rather than a
    budget increase.

    On this model that pooling is across two KV groups rather than the twelve
    query heads the paper assumes -- the mechanism has two dials here.

    Returns per-head sorted index tensors, which may differ in length.
    """
    sc = reduce_group(scores_layer, kv_heads, mode).clone()
    if pool_kernel > 1:
        sc = torch.nn.functional.max_pool1d(
            sc.unsqueeze(0), kernel_size=pool_kernel, stride=1,
            padding=pool_kernel // 2).squeeze(0)[:, :L]
    sc[:, L - window:] = float("inf")       # window is mandatory for every head
    total = min(budget * kv_heads, kv_heads * L)
    flat = sc.reshape(-1)
    chosen = flat.topk(total).indices
    heads = chosen // L
    pos = chosen % L
    out = []
    for h in range(kv_heads):
        idx = pos[heads == h]
        if idx.numel() == 0:                 # never leave a head empty
            idx = torch.arange(L - 1, L, device=sc.device)
        out.append(idx.sort().values)
    return out


def policy_adakv(cache, ctx):
    """Ada-KV with masked dense storage.

    Heads receive different counts, the tensor is sized to the largest, and the
    padded slots are masked out of attention so they contribute nothing. That
    keeps the semantics exact at the cost of storing more than is allocated --
    a dense-storage penalty, not a property of the method; paged attention holds
    ragged lengths natively and pays none of it.
    """
    new = DynamicCache()
    masks, allocated, stored = [], 0, 0
    for i, layer in enumerate(cache.layers):
        k, v = layer.keys, layer.values
        _, kv_heads, L, D = k.shape
        if ctx["budget"] >= L:
            new.update(k.clone(), v.clone(), i)
            masks.append(None)
            allocated += kv_heads * L
            stored += kv_heads * L
            continue
        idxs = adakv_allocation(ctx["scores"][i], kv_heads, ctx["group_reduce"],
                                ctx["budget"], ctx["window"], ctx["pool_kernel"], L)
        width = max(int(t.numel()) for t in idxs)
        gather = torch.zeros(kv_heads, width, dtype=torch.long, device=k.device)
        pad = torch.zeros(kv_heads, width, dtype=torch.bool, device=k.device)
        for h, t in enumerate(idxs):
            gather[h, :t.numel()] = t
            if t.numel() < width:
                gather[h, t.numel():] = t[-1]      # slot content is irrelevant,
                pad[h, t.numel():] = True          # it is masked out of attention
            allocated += int(t.numel())
        stored += kv_heads * width
        g = gather.unsqueeze(0).unsqueeze(-1).expand(1, kv_heads, width, D)
        new.update(k.gather(2, g).contiguous(), v.gather(2, g).contiguous(), i)
        masks.append(pad)
    ctx["_adakv_masks"] = masks
    ctx["_adakv_allocated"] = allocated
    ctx["_adakv_stored"] = stored
    return new


def pyramid_budgets(num_layers, budget, window, beta=20):
    """Per-layer budgets for PyramidKV, summing to `num_layers * budget`.

    PyramidKV's claim is that lower layers attend broadly and upper layers
    concentrate, so budget should decrease with depth rather than be uniform.
    The schedule is the paper's arithmetic sequence over the *selectable* portion
    of the budget -- the part above the mandatory observation window:

        min = selectable // beta,  max = selectable * 2 - min

    with layers interpolating linearly from max down to min.

    The total is then corrected to hit `num_layers * budget` exactly, because
    this study compares methods at equal TOTAL retained entries. Rounding a
    linear ramp otherwise leaves a few entries of drift, and a method that
    quietly retained 1% more would look better for the wrong reason.
    """
    sel = max(1, budget - window)
    if num_layers == 1:
        return [budget]
    lo = max(1, sel // beta)
    hi = max(lo, sel * 2 - lo)
    step = (hi - lo) / (num_layers - 1)
    raw = [max(1, int(round(hi - i * step))) for i in range(num_layers)]
    # Correct drift so the totals match exactly, spreading the adjustment over
    # the layers with the most room rather than dumping it on one.
    target = sel * num_layers
    i = 0
    while sum(raw) != target:
        d = 1 if sum(raw) < target else -1
        j = max(range(num_layers), key=lambda k: raw[k]) if d < 0 else i % num_layers
        if d < 0 and raw[j] <= 1:
            break
        raw[j] += d
        i += 1
    return [r + window for r in raw]


def _select(scores_layer, k, v, budget, window, ctx, kv_heads):
    """Shared top-k selection: group reduction, pooling, forced window, gather.

    Every policy in this study selects the same way and differs only in how much
    budget each layer or head is given. Keeping the selection identical is what
    makes the comparison about allocation rather than about scoring.
    """
    _, _, L, D = k.shape
    if budget >= L:
        return k.clone(), v.clone()
    sc = reduce_group(scores_layer, kv_heads, ctx["group_reduce"]).clone()
    if ctx["pool_kernel"] > 1:
        sc = torch.nn.functional.max_pool1d(
            sc.unsqueeze(0), kernel_size=ctx["pool_kernel"], stride=1,
            padding=ctx["pool_kernel"] // 2).squeeze(0)[:, :L]
    sc[:, L - window:] = float("inf")
    idx, _ = sc.topk(budget, dim=-1).indices.sort(dim=-1)
    g = idx.unsqueeze(0).unsqueeze(-1).expand(1, kv_heads, budget, D)
    return k.gather(2, g).contiguous(), v.gather(2, g).contiguous()


def policy_pyramid(cache, ctx):
    """PyramidKV: same selection as SnapKV, budget tapered across layers.

    Uniform-per-layer allocation is SnapKV; this differs only in the schedule,
    so any difference in the result is attributable to allocation across depth
    and to nothing else.
    """
    n_layers = len(cache.layers)
    budgets = pyramid_budgets(n_layers, ctx["budget"], ctx["window"])
    new = DynamicCache()
    for i, layer in enumerate(cache.layers):
        kv_heads = layer.keys.shape[1]
        kk, vv = _select(ctx["scores"][i], layer.keys, layer.values,
                         min(budgets[i], layer.keys.shape[2]), ctx["window"],
                         ctx, kv_heads)
        new.update(kk, vv, i)
    return new


def policy_full(cache, ctx):
    """Full-cache baseline. Every method is compared against this, per model."""
    return clone_cache(cache)


def policy_snapkv(cache, ctx):
    """SnapKV: keep the top-scoring positions per KV group, plus the window.

    Selection is per KV head, so different groups keep different positions. That
    requires a gather rather than an index_select, because no single per-sequence
    index serves all heads -- a structural difference from the MHA reference
    implementations, where one index per head suffices. Keys retain the RoPE
    phase they were written with; nothing is recomputed or renumbered.
    """
    budget, window = ctx["budget"], ctx["window"]
    new = DynamicCache()
    for i, layer in enumerate(cache.layers):
        k, v = layer.keys, layer.values                       # [1, kv_heads, L, D]
        _, kv_heads, L, D = k.shape
        if budget >= L:
            new.update(k.clone(), v.clone(), i)
            continue
        s = reduce_group(ctx["scores"][i], kv_heads, ctx["group_reduce"]).clone()
        if ctx["pool_kernel"] > 1:
            s = torch.nn.functional.max_pool1d(
                s.unsqueeze(0), kernel_size=ctx["pool_kernel"],
                stride=1, padding=ctx["pool_kernel"] // 2).squeeze(0)[:, :L]
        # The observation window is always retained: it is the most recent
        # context and it is what the scores were computed from.
        s[:, L - window:] = float("inf")
        idx, _ = s.topk(budget, dim=-1).indices.sort(dim=-1)
        gather = idx.unsqueeze(0).unsqueeze(-1).expand(1, kv_heads, budget, D)
        new.update(k.gather(2, gather).contiguous(),
                   v.gather(2, gather).contiguous(), i)
    return new


POLICIES = {"full": policy_full, "snapkv": policy_snapkv,
            "pyramidkv": policy_pyramid, "adakv": policy_adakv}


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--quant", choices=("none", "nf4"), default="none",
                    help="nf4 quantises weights only; the KV cache stays 2 bytes")
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    ap.add_argument("--task", default="needle",
                    help="'needle' (synthetic diagnostic), 'ruler:<variant>' "
                         "(reported retrieval benchmark), or 'longbench:<task>'")
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--min-natural-tokens", type=int, default=0,
                    help="LongBench only: keep documents whose untruncated prompt "
                         "is at least this long. Set to the largest context under "
                         "comparison so every sample saturates every context, "
                         "otherwise context length is confounded with document length")
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--policies", nargs="+", default=["full", "snapkv"])
    ap.add_argument("--budget-fractions", type=float, nargs="+", default=[0.5],
                    help="retained fraction(s) of the prompt's KV entries. Multiple "
                         "values sweep inside the config-inner loop, so each sample's "
                         "prefill is loaded once for the whole grid")
    ap.add_argument("--budget-tokens", type=int, nargs="*", default=None,
                    help="absolute retained entry count(s); overrides fractions. "
                         "Retrieval may depend on how many entries survive rather "
                         "than on what share of the prompt they are")
    ap.add_argument("--window", type=int, default=32,
                    help="observation window, 32-64 per protocol")
    ap.add_argument("--group-reduce", choices=("sum", "max", "mean"), default="sum",
                    help="how six query heads' preferences become one vote on "
                         "their shared KV entry. Re-runnable against an existing "
                         "prefill cache, since scores are stored per query head")
    ap.add_argument("--pool-kernel", type=int, default=7)
    ap.add_argument("--min-baseline", type=float, default=0.5,
                    help="refuse to write results if the full-cache baseline falls "
                         "below this. A weak baseline measures the pipeline, not "
                         "the method; the zero-baseline rule missed a 0.400 case")
    ap.add_argument("--allow-weak-baseline", action="store_true",
                    help="acknowledge a baseline below --min-baseline and write "
                         "results anyway, recording that it was overridden")
    ap.add_argument("--recall-target", choices=("value", "key"), default="value",
                    help="which part of the needle to trace through selection")
    ap.add_argument("--measure-recall", action="store_true",
                    help="report whether the answer's own entries survive top-k "
                         "at each budget, then skip decoding. Separates a ranking "
                         "failure from a budget failure")
    ap.add_argument("--measure-spread", action="store_true",
                    help="report how many entries hold 50%% and 90%% of SnapKV's "
                         "scoring mass, then skip the policies. Tests the budget-"
                         "axis hypothesis against the signal the policy ranks on")
    ap.add_argument("--timing-warmup", type=int, default=1,
                    help="leading samples discarded from timing stats PER "
                         "CONFIGURATION, since each config re-warms its own kernels")
    ap.add_argument("--prefill-chunk", type=int, default=128)
    ap.add_argument("--decode-tokens", type=int, default=0,
                    help="0 uses the task's own generation length "
                         "(LongBench specifies one per task)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-dir", type=Path, default=None,
                    help="reuse prefills across runs; built on first miss")
    ap.add_argument("--no-cache-write", action="store_true",
                    help="read the prefill cache but do not populate it")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "benchmark.json")
    args = ap.parse_args()

    assert 32 <= args.window <= 64, "protocol: observation window is 32-64 tokens"
    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model_id)
    model = load_model(args.model_id, args.dtype, args.quant)
    device = next(model.parameters()).device
    assert device.type == "cuda"
    if "adakv" in args.policies:
        install_head_mask_attention(model)
    stop_ids = {i for i in (tok.eos_token_id,
                            tok.convert_tokens_to_ids("<|im_end|>"),
                            tok.convert_tokens_to_ids("<|endoftext|>"))
                if isinstance(i, int) and i >= 0}
    # The task is part of the cache identity: a different task is a different
    # prompt, so reusing a cache across tasks would silently evaluate the wrong
    # text.
    if args.task.startswith("longbench:"):
        task = LongBenchTask(args.task.split(":", 1)[1], args.context, tok,
                             min_natural_tokens=args.min_natural_tokens)
    elif args.task.startswith("ruler:"):
        task = RulerTask(args.task.split(":", 1)[1], args.context, tok, seed=args.seed)
    elif args.task == "needle":
        task = NeedleTask(tok, args.context, args.seed)
    else:
        raise SystemExit("unknown task %r" % args.task)
    n_samples = min(args.samples, len(task))
    if n_samples < args.samples:
        print("note: task has only %d samples; running %d" % (len(task), n_samples))
    decode_tokens = args.decode_tokens or task.max_gen
    identity = {"model_id": args.model_id, "dtype": args.dtype,
                "quant": args.quant,
                "task": args.task, "context": args.context,
                "min_natural_tokens": args.min_natural_tokens,
                "window": args.window, "prefill_chunk": args.prefill_chunk}
    print(f"{args.model_id}  {args.dtype}  task={args.task} "
          f"({task.metric_name}, gen={decode_tokens})  ctx={args.context}  "
          f"budgets={args.budget_tokens or args.budget_fractions}  "
          f"window={args.window}  "
          f"group_reduce={args.group_reduce}  "
          f"cache={'on' if args.cache_dir else 'off'}\n")

    rows, spread, recall, hits, misses = [], [], [], 0, 0
    for n in range(n_samples):
        seed = args.seed + n
        sample = task.sample(n, device)
        path = (prefill_cache_path(args.cache_dir, args.model_id, args.dtype,
                                   args.task, args.context, args.window,
                                   args.prefill_chunk, seed)
                if args.cache_dir else None)

        t0 = time.perf_counter()
        if path is not None and path.exists():
            cache, scores, first_id, prompt_len, next_pos = load_prefill(
                path, device, identity)
            source, hits = "cache", hits + 1
        else:
            with torch.no_grad():
                cache, scores, logits, prompt_len = prefill_with_scores(
                    model, sample["input_ids"], args.prefill_chunk, args.window, device)
            assert torch.isfinite(logits).all(), "non-finite prefill logits"
            first_id, next_pos = int(logits[0, -1].argmax()), prompt_len
            source, misses = "live", misses + 1
            if path is not None and not args.no_cache_write:
                save_prefill(path, cache, scores, first_id, prompt_len, identity)
        prefill_s = time.perf_counter() - t0

        if args.measure_recall:
            kvh = model.config.num_key_value_heads
            # The needle carries a key and a value. The value is the answer;
            # the key is what the query matches on. Keys evicted while values
            # survive would leave the model holding the answer with no way to
            # bind it to the question.
            if args.recall_target == "key":
                value = (sample.get("meta") or {}).get("key")
                if value is None:
                    raise SystemExit(f"task {args.task!r} exposes no key to trace")
            else:
                ref = sample["reference"]
                value = ref[0] if isinstance(ref, (list, tuple)) else ref
            spans = find_target_positions(tok, sample["input_ids"], str(value))
            if not spans:
                sprint(f"  s{n} target {value!r} NOT LOCATED in prompt; skipped")
                del cache, scores
                torch.cuda.empty_cache()
                continue
            budgets = ([max(args.window + 1, b) for b in args.budget_tokens]
                       if args.budget_tokens
                       else [max(args.window + 1, int(prompt_len * f))
                             for f in args.budget_fractions])
            for b in budgets:
                r = target_survives(scores, kvh, args.group_reduce, b, args.window,
                                    args.pool_kernel, spans, prompt_len)
                recall.append({"sample": n, "budget": b, "prompt_len": prompt_len,
                               "target_tokens": r["target_tokens"],
                               "any_token_kept": round(r["any"], 4),
                               "all_tokens_kept": round(r["all"], 4)})
                sprint(f"  s{n} budget {b:>6}  target {r['target_tokens']} tok  "
                       f"any-kept {r['any']:>6.1%}  all-kept {r['all']:>6.1%}")
            del cache, scores
            torch.cuda.empty_cache()
            continue

        if args.measure_spread:
            kvh = model.config.num_key_value_heads
            c = score_concentration(scores, kvh, args.group_reduce)
            spread.append({"sample": n, "prompt_len": prompt_len,
                           "k50": c[0.5], "k90": c[0.9],
                           "k50_fraction": round(c[0.5] / prompt_len, 6),
                           "k90_fraction": round(c[0.9] / prompt_len, 6)})
            sprint(f"  s{n} prompt {prompt_len:>6}  k50 {c[0.5]:>6.0f} "
                   f"({c[0.5]/prompt_len:>7.4%})  k90 {c[0.9]:>6.0f} "
                   f"({c[0.9]/prompt_len:>7.4%})")
            del cache, scores
            torch.cuda.empty_cache()
            continue

        first = torch.tensor(first_id, device=device)
        before = cache_signature(cache)

        # Config-inner: policy x budget, all against this one loaded prefill.
        # `full` ignores the budget, so it is run once rather than per grid point.
        if args.budget_tokens:
            budgets = [(f"{b}tok", max(args.window + 1, b)) for b in args.budget_tokens]
        else:
            budgets = [(f"{f:g}", max(args.window + 1, int(prompt_len * f)))
                       for f in args.budget_fractions]
        configs = []
        for name in args.policies:
            configs += ([(name, "-", prompt_len)] if name == "full"
                        else [(name, label, b) for label, b in budgets])

        for name, blabel, budget in configs:
            ctx = {"scores": scores, "budget": budget, "window": args.window,
                   "group_reduce": args.group_reduce, "pool_kernel": args.pool_kernel}
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            with torch.no_grad():
                evicted = POLICIES[name](cache, ctx)
                torch.cuda.synchronize()
                select_s = time.perf_counter() - t1
                # Read before decode: decoding appends, so reading afterwards
                # reports budget + decode_tokens. Summed over layers, not read
                # from layer 0: PyramidKV tapers budget with depth, so layer 0 is
                # its largest layer and would overstate what it retained by ~80%.
                # Total entries is also the axis methods are compared on.
                kept_total = sum(l.keys.shape[2] for l in evicted.layers)
                kept = kept_total // len(evicted.layers)
                if name == "adakv":
                    for lyr, m in zip(model.model.layers, ctx["_adakv_masks"]):
                        if m is None:
                            if hasattr(lyr.self_attn, "_head_mask"):
                                del lyr.self_attn._head_mask
                        else:
                            lyr.self_attn._head_mask = m
                produced, _ = decode(model, evicted, first, decode_tokens,
                                     device, start_position=next_pos)
                if name == "adakv":
                    clear_head_masks(model)
            torch.cuda.synchronize()
            total_s = time.perf_counter() - t1

            assert cache_signature(cache) == before, (
                f"policy {name!r} mutated the source cache; every later "
                f"configuration for this sample would run on its leftovers")

            # `first` comes from the prefill logits and is part of the answer.
            produced = truncate_at_stop([first_id] + produced, stop_ids)
            text = tok.decode(produced, skip_special_tokens=True)
            score = task.score(text, sample["reference"])
            rows.append({
                "sample": n, "seed": seed, "id": sample["id"],
                "meta": sample.get("meta"),
                "policy": name, "budget_label": blabel, "budget": budget,
                "group_reduce": args.group_reduce,
                "prefill_source": source, "prefill_seconds": round(prefill_s, 4),
                "prompt_len": prompt_len, "kept": kept,
                "kept_total": kept_total, "n_layers": len(evicted.layers),
                # For an unequal allocator these differ: a dense cache stores
                # max(n_h) per head while the method allocates sum(n_h).
                "allocated_total": ctx.get("_adakv_allocated", kept_total),
                "stored_total": ctx.get("_adakv_stored", kept_total),
                "kept_fraction": round(kept / prompt_len, 4),
                "selection_seconds": round(select_s, 4),   # latency includes selection
                "total_seconds": round(total_s, 4),
                "score": round(score, 4), "correct": score >= 1.0,
                "expected": sample["reference"], "generated": text.strip()[:160],
            })
            sprint(f"  s{n} {name:<7}{blabel:>7} [{source:>5}] kept {kept:>6}"
                   f" ({kept/prompt_len:>6.2%})  "
                   f"{score:>5.2f}  {text.strip()[:34]!r}")
            del evicted
        del cache, scores
        torch.cuda.empty_cache()

    if args.measure_recall:
        import statistics as _st
        print()
        print("budget   n   any-token kept   all-tokens kept")
        agg = {}
        for r in recall:
            agg.setdefault(r["budget"], []).append(r)
        for b in sorted(agg):
            v = agg[b]
            print(f"{b:>6} {len(v):>3}   {_st.mean(x['any_token_kept'] for x in v):>12.1%}   "
                  f"{_st.mean(x['all_tokens_kept'] for x in v):>14.1%}")
        payload = {"run": {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
                           "task": args.task, "context": args.context,
                           "measures": "fraction of (layer, kv_head) units whose "
                                       "SnapKV top-k retains the answer's tokens"},
                   "config": {k: (str(v) if isinstance(v, Path) else v)
                              for k, v in vars(args).items()},
                   "by_budget": {str(b): {
                       "n": len(agg[b]),
                       "any_token_kept": round(_st.mean(x["any_token_kept"] for x in agg[b]), 4),
                       "all_tokens_kept": round(_st.mean(x["all_tokens_kept"] for x in agg[b]), 4)}
                       for b in sorted(agg)},
                   "rows": recall}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Logged to {args.out}")
        return

    if args.measure_spread:
        import statistics as _st
        med = {k: _st.median([r[k] for r in spread])
               for k in ("k50", "k90", "k50_fraction", "k90_fraction")}
        print(f"\nmedian over {len(spread)} samples: "
              f"k50 {med['k50']:.0f} ({med['k50_fraction']:.4%})  "
              f"k90 {med['k90']:.0f} ({med['k90_fraction']:.4%})")
        payload = {"run": {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
                           "task": args.task, "context": args.context,
                           "measures": "entries covering 50%/90% of SnapKV's "
                                       "group-reduced observation-window scores"},
                   "config": {k: (str(v) if isinstance(v, Path) else v)
                              for k, v in vars(args).items()},
                   "median": med, "rows": spread}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Logged to {args.out}")
        return

    summary = {}
    for key in dict.fromkeys((r["policy"], r["budget_label"]) for r in rows):
        sel = [r for r in rows if (r["policy"], r["budget_label"]) == key]
        summary[f"{key[0]}@{key[1]}"] = {
            "policy": key[0], "budget_label": key[1], "n": len(sel),
            "mean_kept": round(sum(r["kept"] for r in sel) / len(sel), 1),
            "score": round(sum(r["score"] for r in sel) / len(sel), 4),
            "mean_kept_fraction": round(sum(r["kept_fraction"] for r in sel) / len(sel), 4),
            # Warmup discarded per configuration, not per run: each config
            # re-warms its own kernels, so the first sample of every one is slow.
            **pctiles([r["selection_seconds"] for r in sel][args.timing_warmup:]),
        }
    print(f"\npolicy   budget      kept    kept%   n  {task.metric_name:>8}   sel p50 / p10-p90 ms")
    for v in summary.values():
        t = (f"{v['sel_p50_ms']:>6.1f} / {v['sel_p10_ms']:.1f}-{v['sel_p90_ms']:.1f}"
             if v.get("sel_p50_ms") is not None else "   n/a")
        print(f"{v['policy']:<8} {v['budget_label']:>7}  {v['mean_kept']:>8.0f} "
              f"{v['mean_kept_fraction']:>7.2%} {v['n']:>3}  {v['score']:>8.3f}   {t}")
    print(f"prefill cache: {hits} hit, {misses} miss")

    # A baseline materially below what the task should yield means the pipeline
    # is suspect, not that the task is hard. The earlier rule only fired at
    # exactly 0.000, and a Phi arm with a 0.400 baseline -- broken by a LongRoPE
    # defect and a truncated generation length -- read as "hard task" and was
    # reported before it was verified. The guard now fires on "materially low",
    # and requires an explicit acknowledgement rather than a judgement call.
    baseline = next((v["score"] for v in summary.values()
                     if v["policy"] == "full"), None)
    weak = baseline is not None and baseline < args.min_baseline
    if weak and not args.allow_weak_baseline:
        print(f"\nREFUSING TO WRITE: full-cache baseline {baseline:.3f} is below "
              f"--min-baseline {args.min_baseline}.")
        print("A baseline this low measures the pipeline, not the method. Verify "
              "the model, prompt and generation length independently first.")
        print("Re-run with --allow-weak-baseline to record it deliberately.")
        return 1

    payload = {
        "run": {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "task": args.task, "metric": task.metric_name,
                "full_cache_baseline": baseline,
                "weak_baseline_override": bool(weak and args.allow_weak_baseline),
                "decode_tokens": decode_tokens,
                "prefill_cache_hits": hits, "prefill_cache_misses": misses},
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "summary": summary, "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Logged to {args.out}")


if __name__ == "__main__":
    main()
