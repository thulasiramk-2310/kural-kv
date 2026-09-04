#!/usr/bin/env python3
"""Isolated prefill cost curve.

The prefill timings recorded during gate check are contaminated: those runs were
also downloading weights, and an earlier fp16 run was thrashing the CUDA
allocator on OOM-retry. The superlinearity they suggest is real, but its
magnitude is not measurable from them.

This measures prefill alone, under conditions that let the number mean something:
warm weights, a discarded warmup iteration, repeats per context length, and
torch.cuda.synchronize() around the timed region so GPU time is not hidden behind
asynchronous dispatch.

It reports, per context length, each repeat's wall time and peak allocation, then
fits log(time) against log(context) to give the scaling exponent:

  exponent near 2  -- attention-bound, as expected for eager. Prefill dominates
                      the benchmark and prefill caching is mandatory.
  exponent near 1  -- the earlier 13x jump was mostly allocator thrash, and the
                      planned sweep is far cheaper than budgeted for.

Two diagnostics separate warmup from a leak:
  - resident allocation at the start of each repeat, after cleanup. Flat means
    memory is being released; climbing means something is retained and the
    timings are measuring that rather than prefill.
  - first-vs-rest timing. A slow first iteration with flat memory is warmup.
"""

import argparse
import json
import math
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from gate_check import DTYPES, MODEL_ID, build_probe_ids, chunked_prefill, kv_cache_bytes

REPO_ROOT = Path(__file__).resolve().parent.parent


def fit_exponent(lengths, times):
    """Least-squares slope of log(time) vs log(length)."""
    n = len(lengths)
    if n < 2:
        return None
    xs = [math.log(v) for v in lengths]
    ys = [math.log(v) for v in times]
    mx, my = sum(xs) / n, sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    ap.add_argument("--context-lengths", type=int, nargs="+",
                    default=[2048, 4096, 8192, 16384])
    ap.add_argument("--prefill-chunk", type=int, default=512,
                    help="held fixed across all context lengths, so the curve "
                         "measures context scaling and not chunk-size effects")
    ap.add_argument("--repeats", type=int, default=3, help="timed repeats per length")
    ap.add_argument("--warmup", type=int, default=1, help="discarded leading iterations")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "prefill_timing.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable; refusing to report timings.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, dtype=DTYPES[args.dtype],
        attn_implementation="eager", device_map={"": 0}).eval()
    device = next(model.parameters()).device
    assert device.type == "cuda", f"model is on {device}, not cuda"
    print(f"{args.model_id}  {args.dtype}  eager  on {torch.cuda.get_device_name(0)}")
    print(f"prefill_chunk={args.prefill_chunk}  warmup={args.warmup}  repeats={args.repeats}\n")

    rows = []
    for length in args.context_lengths:
        ids = build_probe_ids(tokenizer, length, device)
        reps = []
        for i in range(args.warmup + args.repeats):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            resident_before = torch.cuda.memory_allocated()

            t0 = time.perf_counter()
            with torch.no_grad():
                cache, out = chunked_prefill(model, ids, args.prefill_chunk, device)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            # Standing rule: never record a number from a non-finite forward pass.
            if not torch.isfinite(out.logits).all():
                raise SystemExit(
                    f"non-finite logits at context={length} in {args.dtype}; "
                    "refusing to write a timing row")

            peak = torch.cuda.max_memory_allocated()
            kv = kv_cache_bytes(cache)
            del cache, out
            reps.append({
                "iteration": i,
                "role": "warmup" if i < args.warmup else "timed",
                "seconds": round(elapsed, 4),
                "peak_gib": round(peak / 1024 ** 3, 3),
                "resident_before_gib": round(resident_before / 1024 ** 3, 3),
                "kv_gib": round(kv / 1024 ** 3, 3),
            })
        del ids
        torch.cuda.empty_cache()

        timed = [r["seconds"] for r in reps if r["role"] == "timed"]
        resident = [r["resident_before_gib"] for r in reps]
        row = {
            "context_length": length,
            "repeats": reps,
            "median_seconds": round(sorted(timed)[len(timed) // 2], 4),
            "min_seconds": round(min(timed), 4),
            "max_seconds": round(max(timed), 4),
            "spread_ratio": round(max(timed) / min(timed), 3),
            "peak_gib": max(r["peak_gib"] for r in reps),
            "kv_gib": reps[-1]["kv_gib"],
            "resident_drift_gib": round(max(resident) - min(resident), 3),
            "warmup_ratio": (round(reps[0]["seconds"] / min(timed), 2)
                             if args.warmup and min(timed) > 0 else None),
        }
        rows.append(row)
        print(f"ctx {length:>6}  median {row['median_seconds']:>8.3f}s  "
              f"min {row['min_seconds']:>8.3f}s  max {row['max_seconds']:>8.3f}s  "
              f"peak {row['peak_gib']:>5.2f} GiB  "
              f"warmup x{row['warmup_ratio']}  drift {row['resident_drift_gib']} GiB")

    exponent = fit_exponent([r["context_length"] for r in rows],
                            [r["median_seconds"] for r in rows])
    max_drift = max(r["resident_drift_gib"] for r in rows)

    # The global fit is not the signal. At small contexts prefill is dominated by
    # fixed per-chunk overhead, and the smaller the chunk the more chunks there
    # are, so the low end drags the global fit toward linear regardless of how
    # the attention term is actually growing. The exponent at the TOP of the
    # measured range is what says whether attention is taking over; the global
    # fit is reported for reference only.
    top_exponent = None
    if len(rows) >= 2:
        a, b = rows[-2], rows[-1]
        top_exponent = (math.log(b["median_seconds"] / a["median_seconds"])
                        / math.log(b["context_length"] / a["context_length"]))

    e = top_exponent if top_exponent is not None else exponent
    verdict = ("attention-bound (near-quadratic): prefill dominates, caching is mandatory"
               if e and e >= 1.6 else
               "near-linear: earlier superlinearity was largely allocator thrash"
               if e and e <= 1.3 else
               "between linear and quadratic, trending toward attention-bound")

    if exponent:
        print(f"\nglobal log-log fit: {exponent:.3f}  "
              f"(reference only: per-chunk overhead flattens the low end)")
    if top_exponent:
        print(f"local exponent at top of range "
              f"({rows[-2]['context_length']}->{rows[-1]['context_length']}): {top_exponent:.3f}")
    print(f"verdict: {verdict}")
    print(f"max resident drift across repeats: {max_drift} GiB "
          f"({'flat, no leak' if max_drift <= 0.05 else 'CLIMBING -- investigate before trusting timings'})")

    payload = {
        "run": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "scaling_exponent_global": round(exponent, 4) if exponent else None,
            "scaling_exponent_top_of_range": round(top_exponent, 4) if top_exponent else None,
            "verdict": verdict,
            "max_resident_drift_gib": max_drift,
        },
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "device": torch.cuda.get_device_name(0),
        },
        "measurements": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nLogged to {args.out}")


if __name__ == "__main__":
    main()
