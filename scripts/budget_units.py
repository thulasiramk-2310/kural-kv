#!/usr/bin/env python3
"""Log the head-budget lever width for a set of models.

Per-head budget allocation -- the mechanism Ada-KV, LKV and LU-KV attribute their
gains to -- is only actionable per KV group, because a KV entry is shared by every
query head in its group. So the real number of dials a budget allocator has is
num_key_value_heads, not num_attention_heads.

This reads config.json only. No weights, no GPU, no inference.
"""

import argparse
import json
from pathlib import Path

from transformers import AutoConfig

REPO_ROOT = Path(__file__).resolve().parent.parent

# Spans MHA through aggressive GQA. Gated repos are reported, never guessed.
DEFAULT_MODELS = [
    "microsoft/Phi-3.5-mini-instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "google/gemma-2-2b-it",
]


def probe(model_id):
    try:
        c = AutoConfig.from_pretrained(model_id)
    except Exception as exc:
        return {"model": model_id, "status": type(exc).__name__}
    q = getattr(c, "num_attention_heads", None)
    kv = getattr(c, "num_key_value_heads", None) or q
    if not q:
        return {"model": model_id, "status": "no head counts in config"}
    return {
        "model": model_id,
        "status": "ok",
        "layers": getattr(c, "num_hidden_layers", None),
        "query_heads": q,
        "kv_heads": kv,
        "group_size": q // kv,
        "budget_units": kv,
        "attention": "MHA" if kv == q else "GQA",
        "max_position_embeddings": getattr(c, "max_position_embeddings", None),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "budget_units.json")
    args = ap.parse_args()

    rows = [probe(m) for m in args.models]
    ok = [r for r in rows if r["status"] == "ok"]

    width = max((len(r["model"]) for r in rows), default=10)
    print(f"{'model':<{width}}  {'attn':<4} {'layers':>6} {'q':>4} {'kv':>4} "
          f"{'group':>6} {'budget_units':>13}")
    for r in rows:
        if r["status"] != "ok":
            print(f"{r['model']:<{width}}  -- unavailable: {r['status']}")
            continue
        print(f"{r['model']:<{width}}  {r['attention']:<4} {r['layers']:>6} "
              f"{r['query_heads']:>4} {r['kv_heads']:>4} {r['group_size']:>6} "
              f"{r['budget_units']:>13}")

    if ok:
        widest, narrowest = max(ok, key=lambda r: r["budget_units"]), min(ok, key=lambda r: r["budget_units"])
        print(f"\nThe head-budget lever spans {narrowest['budget_units']} to "
              f"{widest['budget_units']} dials, a "
              f"{widest['budget_units'] // narrowest['budget_units']}x range: "
              f"{widest['model']} down to {narrowest['model']}.")
        print("Published per-head allocation methods were validated at the wide end.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nLogged to {args.out}")


if __name__ == "__main__":
    main()
