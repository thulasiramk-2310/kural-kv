#!/usr/bin/env python3
"""Test why the budget axis differs between the diagnostic and RULER.

The synthetic needle task showed retrieval tracking the absolute retained entry
count, independent of context length. RULER `niah_single_1` shows the opposite:
the proportional axis is approximately right and the absolute one is badly wrong.
The proposed explanation is about where attention lands.

    The synthetic needle names a city that occurs nowhere else in the prompt, so
    the query localises onto a small fixed set of entries and a fixed count
    suffices however much filler surrounds it. RULER's key is an adjective-noun
    pair against a homogeneous haystack, so the relevant mass is spread across
    many partial matches whose number grows with context.

That is directly measurable. This script takes the attention of the final query
position over the whole prompt, reduces it over the GQA group (the unit eviction
actually acts on), and reports how many entries hold 50% and 90% of the mass.

    If the diagnostic's concentration is roughly CONSTANT in entries across 2K
    and 16K while RULER's grows roughly PROPORTIONALLY, the hypothesis holds.

The claim being tested is narrow: it concerns these two tasks on this model. It
is not a claim about attention in general, and nothing here should be extended
into one.
"""

import argparse
import json
import statistics as st
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from gate_check import DTYPES, MODEL_ID
from harness import NeedleTask
from ruler import RulerTask

REPO_ROOT = Path(__file__).resolve().parent.parent


def final_query_attention(model, ids, chunk, device):
    """Attention of the LAST prompt position over every key, per KV group.

    Returns a list, one entry per layer, of [kv_heads, L] probabilities summing
    to 1 along the last axis after the group reduction is renormalised.
    """
    cache = DynamicCache()
    total = ids.shape[1]
    head = ids[:, :total - 1]
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
    out = model(input_ids=ids[:, total - 1:], past_key_values=cache,
                attention_mask=torch.ones(1, past + 1, dtype=torch.long, device=device),
                cache_position=torch.arange(past, past + 1, device=device),
                use_cache=True, output_attentions=True, logits_to_keep=1)
    cfg = model.config
    group = cfg.num_attention_heads // cfg.num_key_value_heads
    per_layer = []
    for attn in out.attentions:                        # [1, q_heads, 1, L]
        a = attn[0, :, -1, :].float()                  # [q_heads, L]
        a = a.view(cfg.num_key_value_heads, group, -1).sum(dim=1)   # [kv_heads, L]
        a = a / a.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        per_layer.append(a.cpu())
    del cache, out
    torch.cuda.empty_cache()
    return per_layer


def concentration(per_layer, fracs=(0.5, 0.9)):
    """Entries needed to cover each mass fraction, averaged over layers/heads."""
    out = {f: [] for f in fracs}
    for a in per_layer:
        srt, _ = a.sort(dim=-1, descending=True)
        cum = srt.cumsum(dim=-1)
        for f in fracs:
            k = (cum < f).sum(dim=-1) + 1          # entries to reach fraction f
            out[f] += k.tolist()
    return {f: st.median(v) for f, v in out.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    ap.add_argument("--contexts", type=int, nargs="+", default=[2048, 16384])
    ap.add_argument("--tasks", nargs="+", default=["needle", "ruler:niah_single_1"])
    ap.add_argument("--samples", type=int, default=6)
    ap.add_argument("--prefill-chunk", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "attention_spread.json")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, dtype=DTYPES[args.dtype],
        attn_implementation="eager", device_map={"": 0}).eval()
    device = next(model.parameters()).device
    assert device.type == "cuda"

    rows = []
    print(f"{'task':<22} {'ctx':>6} {'k50':>7} {'k90':>8} {'k50/ctx':>9} {'k90/ctx':>9}")
    for tname in args.tasks:
        for ctx in args.contexts:
            task = (RulerTask(tname.split(":", 1)[1], ctx, tok, seed=args.seed)
                    if tname.startswith("ruler:") else NeedleTask(tok, ctx, args.seed))
            k50s, k90s, lens = [], [], []
            for i in range(args.samples):
                s = task.sample(i, device)
                with torch.no_grad():
                    per_layer = final_query_attention(
                        model, s["input_ids"], args.prefill_chunk, device)
                c = concentration(per_layer)
                k50s.append(c[0.5]); k90s.append(c[0.9])
                lens.append(s["input_ids"].shape[1])
            k50, k90, L = st.median(k50s), st.median(k90s), st.median(lens)
            rows.append({"task": tname, "context": ctx, "prompt_tokens": L,
                         "k50_entries": k50, "k90_entries": k90,
                         "k50_fraction": round(k50 / L, 6),
                         "k90_fraction": round(k90 / L, 6), "n": args.samples})
            print(f"{tname:<22} {ctx:>6} {k50:>7.0f} {k90:>8.0f} "
                  f"{k50/L:>9.4%} {k90/L:>9.4%}")

    print("\ngrowth from the smallest to the largest context "
          "(constant in entries => absolute axis; constant in fraction => proportional):")
    for tname in args.tasks:
        r = [x for x in rows if x["task"] == tname]
        if len(r) < 2:
            continue
        lo, hi = r[0], r[-1]
        ctx_ratio = hi["context"] / lo["context"]
        print(f"  {tname:<22} context x{ctx_ratio:.0f}  "
              f"k50 x{hi['k50_entries']/max(lo['k50_entries'],1e-9):.2f}  "
              f"k90 x{hi['k90_entries']/max(lo['k90_entries'],1e-9):.2f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "run": {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "measures": "attention of the final prompt position over all keys, "
                            "summed within each GQA group and renormalised; "
                            "k50/k90 are the entries needed to cover that mass"},
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "rows": rows}, indent=2), encoding="utf-8")
    print(f"\nLogged to {args.out}")


if __name__ == "__main__":
    main()
