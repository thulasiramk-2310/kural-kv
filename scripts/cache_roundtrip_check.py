#!/usr/bin/env python3
"""Verify that a prefilled KV cache survives a disk round trip.

The prefill-caching design writes one cache per (model, context, sample) and
reuses it across every eviction method and budget. Before tens of GiB are written
under that assumption, this checks that a cache saved and reloaded in a FRESH
PROCESS produces the same generation as a live one.

Deliberately end-to-end rather than tensor-level. A tensor comparison would pass
on all three of the failure modes that actually matter:

  Non-contiguous tensors. Cache tensors that have been sliced or permuted carry
  strides; torch.save preserves the underlying storage, which can persist far
  more than intended or restore with a different layout. Saved tensors are forced
  contiguous and asserted.

  Incomplete Cache reconstruction. Rebuilding a v5 Cache from raw tensors
  requires the per-layer state to line up -- layer count, ordering, dtype, device
  and initialisation flags. A cache missing a field frequently still runs and
  produces subtly wrong output, so the reconstruction is fingerprinted field by
  field and compared across processes.

  Position on resume. A loaded cache has no memory of what position the next
  token should claim. If the continuation defaults to the cache length, the RoPE
  desynchronisation that eviction causes reappears from a different direction.
  The logical continuation position is therefore persisted in the sidecar and
  asserted, and --demo-naive-position demonstrates what its absence costs.

Usage (the two phases MUST be separate processes; same-process comparison can
share state that hides exactly what is being tested):

    python cache_roundtrip_check.py --phase save --dir <d>
    python cache_roundtrip_check.py --phase load --dir <d>
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from gate_check import (DTYPES, MODEL_ID, build_probe_ids, chunked_prefill,
                        decode, evict_half)

REPO_ROOT = Path(__file__).resolve().parent.parent


def fingerprint(cache):
    """Field-by-field description of a cache, comparable across processes."""
    layers = []
    for i, layer in enumerate(cache.layers):
        entry = {"layer": i}
        for name in ("keys", "values"):
            t = getattr(layer, name)
            entry[name] = {
                "shape": list(t.shape),
                "dtype": str(t.dtype),
                "device": t.device.type,
                "contiguous": bool(t.is_contiguous()),
                # Hash the raw bytes: equality here is bitwise equality.
                "sha256": hashlib.sha256(
                    t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
                ).hexdigest(),
            }
        entry["dtype"] = str(getattr(layer, "dtype", None))
        entry["device"] = str(getattr(layer, "device", None))
        entry["is_initialized"] = bool(getattr(layer, "is_initialized", False))
        layers.append(entry)
    return {
        "num_layers": len(cache.layers),
        "seq_length": cache.get_seq_length(),
        "offloading": bool(getattr(cache, "offloading", False)),
        "layers": layers,
    }


def save_cache(cache, path):
    payload = []
    for layer in cache.layers:
        k, v = layer.keys, layer.values
        # Force contiguity before serialising, so a strided view cannot drag its
        # whole backing storage to disk or restore with a different layout.
        k, v = k.contiguous(), v.contiguous()
        assert k.is_contiguous() and v.is_contiguous()
        payload.append({"keys": k.cpu(), "values": v.cpu()})
    torch.save(payload, path)


def load_cache(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    cache = DynamicCache()
    for idx, entry in enumerate(payload):
        k = entry["keys"].to(device).contiguous()
        v = entry["values"].to(device).contiguous()
        cache.update(k, v, idx)
    return cache


def build_model(model_id, dtype):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=DTYPES[dtype], attn_implementation="eager",
        device_map={"": 0}).eval()
    assert next(model.parameters()).device.type == "cuda"
    return tok, model


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=("save", "load"), required=True)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    ap.add_argument("--context", type=int, default=2048)
    ap.add_argument("--prefill-chunk", type=int, default=128)
    ap.add_argument("--decode-tokens", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tolerance", type=float, default=0.0,
                    help="max allowed absolute logit difference (default: exact)")
    ap.add_argument("--evict-half-before-save", action="store_true",
                    help="evict 50%% of the cache before saving, so cache length "
                         "and logical position diverge. This is the case the "
                         "persisted next_position exists for")
    ap.add_argument("--demo-naive-position", action="store_true",
                    help="also decode using the cache length as the position, to "
                         "show what the persisted logical position is protecting")
    args = ap.parse_args()

    args.dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.dir / "cache.pt"
    side_path = args.dir / "sidecar.json"
    ref_path = args.dir / "reference.pt"

    torch.manual_seed(args.seed)
    tok, model = build_model(args.model_id, args.dtype)
    device = next(model.parameters()).device
    ids = build_probe_ids(tok, args.context, device)

    if args.phase == "save":
        with torch.no_grad():
            cache, out = chunked_prefill(model, ids, args.prefill_chunk, device)
        assert torch.isfinite(out.logits).all(), "non-finite prefill logits"
        prompt_len = cache.get_seq_length()
        if args.evict_half_before_save:
            evict_half(cache)
            print(f"evicted: cache length {prompt_len} -> {cache.get_seq_length()}, "
                  f"logical position stays {prompt_len}")

        fp = fingerprint(cache)
        save_cache(cache, cache_path)
        side_path.write_text(json.dumps({
            "model_id": args.model_id, "dtype": args.dtype,
            "context": args.context, "prefill_chunk": args.prefill_chunk,
            "prompt_len": prompt_len,
            # The logical position the continuation must claim. Without this the
            # loaded cache has no way to know, and the default is wrong.
            "next_position": prompt_len,
            "seed": args.seed,
            "fingerprint": fp,
        }, indent=2), encoding="utf-8")

        first = out.logits[0, -1].argmax()
        with torch.no_grad():
            produced, _ = decode(model, cache, first, args.decode_tokens,
                                 device, start_position=prompt_len)
        torch.save({"first_token": int(first), "tokens": produced}, ref_path)
        print(f"saved cache: {cache_path} ({cache_path.stat().st_size / 2**20:.1f} MiB)")
        print(f"prompt_len={prompt_len}  next_position={prompt_len}")
        print(f"reference tokens: {produced}")
        print(f"reference text  : {tok.decode(produced)!r}")
        return 0

    # ---- load phase, fresh process ----------------------------------------
    side = json.loads(side_path.read_text(encoding="utf-8"))
    ref = torch.load(ref_path, weights_only=True)
    cache = load_cache(cache_path, device)

    ok = True
    fp_live, fp_saved = fingerprint(cache), side["fingerprint"]
    if fp_live == fp_saved:
        print(f"[PASS] cache fingerprint identical across processes "
              f"({fp_live['num_layers']} layers, seq {fp_live['seq_length']}, bitwise)")
    else:
        ok = False
        print("[FAIL] cache fingerprint differs")
        for key in ("num_layers", "seq_length", "offloading"):
            if fp_live[key] != fp_saved[key]:
                print(f"         {key}: saved={fp_saved[key]} loaded={fp_live[key]}")
        for a, b in zip(fp_saved["layers"], fp_live["layers"]):
            diffs = [k for k in a if a[k] != b[k]]
            if diffs:
                print(f"         layer {a['layer']}: differs in {diffs}")

    first = torch.tensor(ref["first_token"], device=device)
    with torch.no_grad():
        produced, _ = decode(model, cache, first, args.decode_tokens,
                             device, start_position=side["next_position"])

    if produced == ref["tokens"]:
        print(f"[PASS] {len(produced)} decoded token ids identical to live prefill")
        print(f"         {tok.decode(produced)!r}")
    else:
        ok = False
        first_div = next(i for i, (a, b) in enumerate(zip(ref["tokens"], produced)) if a != b)
        print(f"[FAIL] token ids diverge at position {first_div}")
        print(f"         live  : {ref['tokens']}")
        print(f"         loaded: {produced}")
        print("         structural problem -- do not chase tolerances")

    if args.demo_naive_position:
        cache2 = load_cache(cache_path, device)
        with torch.no_grad():
            naive, _ = decode(model, cache2, first, args.decode_tokens, device,
                              start_position=cache2.get_seq_length())
        same = naive == ref["tokens"]
        print(f"[{'INFO' if not same else 'WARN'}] naive position (cache length) "
              f"{'also matches -- the sidecar is not load-bearing here' if same else 'diverges, as expected'}")
        if not same:
            print(f"         {tok.decode(naive)!r}")

    print("\n" + ("ROUND TRIP VERIFIED" if ok else "ROUND TRIP FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
