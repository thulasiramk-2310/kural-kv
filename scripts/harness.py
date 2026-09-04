#!/usr/bin/env python3
"""Eviction benchmark harness, first end-to-end path.

Deliberately narrow: one task, one baseline, one method, one budget, producing
one row per configuration. The harness has more moving parts than anything else
in the study -- sample construction, prefill, scoring, policy application,
decode, grading, logging -- so the path is made to work before methods are added.
If five methods were added at once and the numbers looked wrong, there would be
no way to tell which layer was lying.

Loop order is sample-outer, config-inner, and that is load-bearing rather than
stylistic. A cached 16K entry costs ~2.2s to read and rebuild on the GPU, which
is dominated by the host-to-device rebuild rather than the NVMe read. Config-outer
would reload the same sample once per configuration: at 100 samples x 25 configs
that is 2500 loads and about 1.5 hours of pure loading. Sample-outer loads each
entry once, 100 loads, under four minutes. Reversing this later means rewriting
the loop, so it is fixed now.

Every policy therefore runs against a clone, and the source cache is asserted
unchanged afterwards. Without that, configuration 2 silently runs on
configuration 1's leftovers and the second row of every sweep is wrong.
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
# Task: needle in a haystack
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
    needle = NEEDLE.format(city=city, code=code)
    question = QUESTION.format(city=city)

    body = []
    while True:
        body.append(rng.choice(FILLER))
        if len(tokenizer(" ".join(body)).input_ids) > context_len:
            break
    depth = rng.random()
    at = max(1, int(len(body) * depth))
    body.insert(at, needle)

    messages = [{"role": "user",
                 "content": " ".join(body) + "\n\n" + question}]
    text = tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
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
# Prefill with SnapKV-style observation-window scores
# --------------------------------------------------------------------------
def prefill_with_scores(model, ids, chunk, window, device):
    """Prefill, and score every cached position against the observation window.

    The window is the last `window` tokens of the prompt, which is what SnapKV
    scores against. Its attention is captured once, reduced immediately to a
    per-position score, and the full [q_heads, window, L] tensor is dropped --
    keeping it would be O(L) memory for no purpose.

    The reduction is over the GQA group, not the query head, and that is the
    whole point of the study: a KV entry is shared by every query head in its
    group, so a per-query-head score is not actionable. Scores are summed within
    each group, which is the only defensible aggregation when the eviction
    decision must serve all members of the group at once.
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
    tail = ids[:, total - window:]
    out = model(input_ids=tail, past_key_values=cache,
                attention_mask=torch.ones(1, past + window, dtype=torch.long, device=device),
                cache_position=torch.arange(past, past + window, device=device),
                use_cache=True, output_attentions=True, logits_to_keep=1)
    cache = out.past_key_values

    cfg = model.config
    group = cfg.num_attention_heads // cfg.num_key_value_heads
    scores = []
    for attn in out.attentions:                       # [1, q_heads, window, L]
        s = attn[0].float().sum(dim=1)                # [q_heads, L]
        s = s.view(cfg.num_key_value_heads, group, -1).sum(dim=1)   # [kv_heads, L]
        scores.append(s)
    logits = out.logits
    return cache, scores, logits, cache.get_seq_length()


# --------------------------------------------------------------------------
# Cache handling
# --------------------------------------------------------------------------
def clone_cache(cache):
    new = DynamicCache()
    for i, layer in enumerate(cache.layers):
        new.update(layer.keys.clone(), layer.values.clone(), i)
    return new


def cache_signature(cache):
    """Cheap structural + value signature, for asserting a policy did not mutate."""
    return [(tuple(l.keys.shape), float(l.keys.float().sum()),
             float(l.values.float().sum())) for l in cache.layers]


# --------------------------------------------------------------------------
# Policies. Each returns a NEW cache; none mutates its input.
# --------------------------------------------------------------------------
def policy_full(cache, scores, budget, window):
    """Full-cache baseline. Every method is compared against this, per model."""
    return clone_cache(cache)


def policy_snapkv(cache, scores, budget, window, pool_kernel=7):
    """SnapKV: keep the top-scoring positions per KV group, plus the window.

    Selection is per KV head, so different groups keep different positions. That
    requires a gather rather than an index_select, because there is no single
    per-sequence index that serves all heads. Keys keep the RoPE phase they were
    written with; nothing is recomputed or renumbered.
    """
    new = DynamicCache()
    for i, layer in enumerate(cache.layers):
        k, v = layer.keys, layer.values                  # [1, kv_heads, L, D]
        _, kv_heads, L, D = k.shape
        if budget >= L:
            new.update(k.clone(), v.clone(), i)
            continue

        s = scores[i].clone()                            # [kv_heads, L]
        # SnapKV pools scores over neighbouring positions so a selected token
        # brings its local context rather than being kept in isolation.
        if pool_kernel > 1:
            s = torch.nn.functional.max_pool1d(
                s.unsqueeze(0), kernel_size=pool_kernel,
                stride=1, padding=pool_kernel // 2).squeeze(0)[:, :L]
        # The observation window is always retained: it is the most recent
        # context and it is what the scores were computed from.
        s[:, L - window:] = float("inf")

        idx = s.topk(budget, dim=-1).indices             # [kv_heads, budget]
        idx, _ = idx.sort(dim=-1)                        # keep positions ordered
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
    ap.add_argument("--budget-fraction", type=float, default=0.5,
                    help="fraction of the prompt's KV entries retained")
    ap.add_argument("--window", type=int, default=32,
                    help="observation window, 32-64 per protocol")
    ap.add_argument("--prefill-chunk", type=int, default=128)
    ap.add_argument("--decode-tokens", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
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
    print(f"{args.model_id}  {args.dtype}  ctx={args.context}  "
          f"budget={args.budget_fraction}  window={args.window}\n")

    rows = []
    for n in range(args.samples):
        sample = build_sample(tok, args.context, args.seed + n, device)

        # --- outer: prefill this sample ONCE, shared by every policy ---------
        with torch.no_grad():
            cache, scores, logits, prompt_len = prefill_with_scores(
                model, sample["input_ids"], args.prefill_chunk, args.window, device)
        assert torch.isfinite(logits).all(), "non-finite prefill logits"
        first = logits[0, -1].argmax()
        before = cache_signature(cache)
        budget = max(args.window + 1, int(prompt_len * args.budget_fraction))

        # --- inner: every policy against a clone of that one prefill ---------
        for name in args.policies:
            t0 = time.perf_counter()
            with torch.no_grad():
                evicted = POLICIES[name](cache, scores, budget, args.window)
                select_s = time.perf_counter() - t0
                # Measured before decode: decoding appends to the cache, so
                # reading it afterwards reports budget + decode_tokens.
                kept = evicted.layers[0].keys.shape[2]
                produced, _ = decode(model, evicted, first, args.decode_tokens,
                                     device, start_position=prompt_len)
            total_s = time.perf_counter() - t0
            # `first` comes from the prefill logits and is part of the answer.
            produced = [int(first)] + produced
            produced = truncate_at_stop(produced, stop_ids)

            assert cache_signature(cache) == before, (
                f"policy {name!r} mutated the source cache; every later "
                f"configuration for this sample would run on its leftovers")

            text = tok.decode(produced, skip_special_tokens=True)
            rows.append({
                "sample": n, "seed": sample["seed"], "depth": sample["depth"],
                "policy": name, "prompt_len": prompt_len,
                "kept": kept, "kept_fraction": round(kept / prompt_len, 4),
                # Latency includes selection, per protocol.
                "selection_seconds": round(select_s, 4),
                "total_seconds": round(total_s, 4),
                "correct": grade(text, sample["code"]),
                "expected": sample["code"], "generated": text.strip()[:120],
            })
            print(f"  s{n} {name:<8} kept {kept:>5}/{prompt_len} "
                  f"({kept/prompt_len:.0%})  {'HIT ' if rows[-1]['correct'] else 'miss'}  "
                  f"sel {select_s*1000:>6.1f}ms  tot {total_s:.2f}s  {text.strip()[:44]!r}")
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

    payload = {
        "run": {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "task": "needle_in_haystack", "grading": "exact substring match"},
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "summary": summary,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nLogged to {args.out}")


if __name__ == "__main__":
    main()
