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

from gate_check import DTYPES, MODEL_ID, decode

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


def truncate_at_stop(tokens, stop_ids):
    """Cut the generation at the first stop token.

    Without this the model answers, emits <|im_end|>, and then keeps going into
    template noise, which pollutes both the grading string and the eyeball check.
    """
    for i, t in enumerate(tokens):
        if t in stop_ids:
            return tokens[:i]
    return tokens


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


def prefill_cache_path(cache_dir, model_id, dtype, context, window, chunk, seed):
    tag = model_id.replace("/", "__")
    return Path(cache_dir) / f"{tag}_{dtype}_ctx{context}_w{window}_c{chunk}_s{seed}.pt"


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
    if d["identity"] != identity:
        raise SystemExit(
            f"cached prefill at {path} was built under a different config:\n"
            f"  cached:   {d['identity']}\n  requested: {identity}")
    cache = DynamicCache()
    for i, e in enumerate(d["layers"]):
        cache.update(e["keys"].to(device).contiguous(),
                     e["values"].to(device).contiguous(), i)
    scores = [s.to(device) for s in d["scores"]]
    return cache, scores, d["first_token"], d["prompt_len"], d["next_position"]


# --------------------------------------------------------------------------
# Policies. Each returns a NEW cache; none mutates its input.
# --------------------------------------------------------------------------
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


POLICIES = {"full": policy_full, "snapkv": policy_snapkv}


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--policies", nargs="+", default=["full", "snapkv"])
    ap.add_argument("--budget-fraction", type=float, default=0.5)
    ap.add_argument("--window", type=int, default=32,
                    help="observation window, 32-64 per protocol")
    ap.add_argument("--group-reduce", choices=("sum", "max", "mean"), default="sum",
                    help="how six query heads' preferences become one vote on "
                         "their shared KV entry. Re-runnable against an existing "
                         "prefill cache, since scores are stored per query head")
    ap.add_argument("--pool-kernel", type=int, default=7)
    ap.add_argument("--prefill-chunk", type=int, default=128)
    ap.add_argument("--decode-tokens", type=int, default=24)
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
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, dtype=DTYPES[args.dtype],
        attn_implementation="eager", device_map={"": 0}).eval()
    device = next(model.parameters()).device
    assert device.type == "cuda"
    stop_ids = {i for i in (tok.eos_token_id,
                            tok.convert_tokens_to_ids("<|im_end|>"),
                            tok.convert_tokens_to_ids("<|endoftext|>"))
                if isinstance(i, int) and i >= 0}
    identity = {"model_id": args.model_id, "dtype": args.dtype,
                "context": args.context, "window": args.window,
                "prefill_chunk": args.prefill_chunk}
    print(f"{args.model_id}  {args.dtype}  ctx={args.context}  "
          f"budget={args.budget_fraction}  window={args.window}  "
          f"group_reduce={args.group_reduce}  "
          f"cache={'on' if args.cache_dir else 'off'}\n")

    rows, hits, misses = [], 0, 0
    for n in range(args.samples):
        seed = args.seed + n
        sample = build_sample(tok, args.context, seed, device)
        path = (prefill_cache_path(args.cache_dir, args.model_id, args.dtype,
                                   args.context, args.window, args.prefill_chunk, seed)
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

        first = torch.tensor(first_id, device=device)
        before = cache_signature(cache)
        budget = max(args.window + 1, int(prompt_len * args.budget_fraction))
        ctx = {"scores": scores, "budget": budget, "window": args.window,
               "group_reduce": args.group_reduce, "pool_kernel": args.pool_kernel}

        for name in args.policies:
            t1 = time.perf_counter()
            with torch.no_grad():
                evicted = POLICIES[name](cache, ctx)
                select_s = time.perf_counter() - t1
                # Read before decode: decoding appends, so reading afterwards
                # reports budget + decode_tokens.
                kept = evicted.layers[0].keys.shape[2]
                produced, _ = decode(model, evicted, first, args.decode_tokens,
                                     device, start_position=next_pos)
            total_s = time.perf_counter() - t1

            assert cache_signature(cache) == before, (
                f"policy {name!r} mutated the source cache; every later "
                f"configuration for this sample would run on its leftovers")

            # `first` comes from the prefill logits and is part of the answer.
            produced = truncate_at_stop([first_id] + produced, stop_ids)
            text = tok.decode(produced, skip_special_tokens=True)
            rows.append({
                "sample": n, "seed": seed, "depth": sample["depth"],
                "policy": name, "group_reduce": args.group_reduce,
                "prefill_source": source, "prefill_seconds": round(prefill_s, 4),
                "prompt_len": prompt_len, "kept": kept,
                "kept_fraction": round(kept / prompt_len, 4),
                "selection_seconds": round(select_s, 4),   # latency includes selection
                "total_seconds": round(total_s, 4),
                "correct": grade(text, sample["code"]),
                "expected": sample["code"], "generated": text.strip()[:120],
            })
            print(f"  s{n} {name:<8} [{source:>5}] kept {kept:>6}/{prompt_len} "
                  f"({kept/prompt_len:.0%})  {'HIT ' if rows[-1]['correct'] else 'miss'}  "
                  f"sel {select_s*1000:>6.1f}ms  {text.strip()[:40]!r}")
            del evicted
        del cache, scores
        torch.cuda.empty_cache()

    summary = {}
    for name in args.policies:
        sel = [r for r in rows if r["policy"] == name]
        summary[name] = {
            "n": len(sel),
            "accuracy": round(sum(r["correct"] for r in sel) / len(sel), 4),
            "mean_kept_fraction": round(sum(r["kept_fraction"] for r in sel) / len(sel), 4),
            "mean_selection_seconds": round(sum(r["selection_seconds"] for r in sel) / len(sel), 4),
        }
    print("\npolicy    n   accuracy  kept   sel ms")
    for name, v in summary.items():
        print(f"{name:<9} {v['n']:<3} {v['accuracy']:>7.2f}  "
              f"{v['mean_kept_fraction']:>5.0%}  {v['mean_selection_seconds']*1000:>6.1f}")
    print(f"prefill cache: {hits} hit, {misses} miss")

    payload = {
        "run": {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "task": "needle_in_haystack", "grading": "exact substring match",
                "prefill_cache_hits": hits, "prefill_cache_misses": misses},
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "summary": summary, "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Logged to {args.out}")


if __name__ == "__main__":
    main()
