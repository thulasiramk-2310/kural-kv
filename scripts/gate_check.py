#!/usr/bin/env python3
"""Environment gate check for kural-kv.

Verifies the assumptions the whole study rests on, before any eviction code
is written. Every gate prints PASS or FAIL and is logged to results/gate_check.json.

Gates:
  1. CUDA available, report total VRAM.
  2. Load the target model in FP16 with eager attention, pinned to GPU 0.
  3. Forward pass with use_cache=True, output_attentions=True. Report the
     layer-0 K-cache shape and attentions[0] shape. Loud failure if attentions
     is None -- that would mean eager attention is not actually in effect.
  4. Confirm the K shape carries fewer KV heads than there are query heads in
     the attention scores. That proves GQA, and gives the true budget unit.
  5. Evict 50% of cache entries along the sequence dimension, decode 20 more
     tokens, confirm no crash.
  6. Report torch.cuda.max_memory_allocated() at each requested context length.
"""

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

# Primary arm. The Llama arm is run explicitly with --model-id and, being the
# only model with real margin above 16K, is the one that carries 32K.
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}

# Real text, not random token ids. A gate that decodes gibberish cannot tell a
# working model from a numerically broken one -- the whole point of reading the
# gate 5 output is that it should be judgeable by eye.
PROBE_TEXT = (
    "The history of computing begins with mechanical calculation. "
    "Charles Babbage designed the Analytical Engine in the 1830s, and "
    "Ada Lovelace wrote the first algorithm intended for a machine. "
    "A century later, wartime codebreaking at Bletchley Park drove the "
    "construction of Colossus, the first programmable electronic computer. "
)
REPO_ROOT = Path(__file__).resolve().parent.parent


class Gates:
    """Collects PASS/FAIL results and prints them as they are decided."""

    def __init__(self):
        self.results = []

    def record(self, name, ok, **details):
        status = "PASS" if ok else "FAIL"
        self.results.append({"gate": name, "status": status, **details})
        print(f"[{status}] {name}")
        for k, v in details.items():
            # Console encoding must never decide a gate. A Windows cp1252 stdout
            # raises UnicodeEncodeError on non-Latin-1 model output, which would
            # otherwise surface as a spurious gate failure.
            line = f"         {k}: {v}"
            try:
                print(line)
            except UnicodeEncodeError:
                print(line.encode("ascii", "backslashreplace").decode("ascii"))
        return ok

    @property
    def all_passed(self):
        return all(r["status"] == "PASS" for r in self.results)


def layer0_k(past_key_values):
    """Return (layer-0 key tensor, access path used).

    The spec for this gate is written as past_key_values[0][0]. That legacy
    tuple indexing was removed in transformers v5 -- Cache no longer defines
    __getitem__ -- so we try it first and fall back to the v5 attribute path,
    recording which one the installed version actually supports.
    """
    try:
        return past_key_values[0][0], "past_key_values[0][0]"
    except TypeError:
        return past_key_values.layers[0].keys, "past_key_values.layers[0].keys"


def evict_half(cache):
    """Drop every other cache entry along the sequence dim, in every layer.

    A structural smoke test, not a scoring policy: it keeps even-numbered
    positions so the retained set is spread across the sequence rather than
    truncated to a prefix. Retained entries keep the RoPE phase they were
    written with, so their original positions stay non-contiguous afterwards.
    Continuation is indexed by original position, not by compacted length -- see
    decode() for why that distinction is the whole ballgame.
    """
    idx = None
    for layer in cache.layers:
        seq_len = layer.keys.shape[-2]
        idx = torch.arange(0, seq_len, 2, device=layer.keys.device)
        # index_select only. Never recompute, re-rotate, or renumber: the keys
        # already carry their original RoPE phase and must keep it.
        layer.keys = layer.keys.index_select(-2, idx).contiguous()
        layer.values = layer.values.index_select(-2, idx).contiguous()
    return idx


def chunked_prefill(model, input_ids, chunk_size, device):
    """Prefill the KV cache in chunks.

    Eager attention materialises a [batch, heads, q_len, kv_len] score matrix.
    A single-shot 32K prefill would need ~69 GiB for that tensor alone, so long
    contexts are prefilled in chunks: the score matrix becomes
    [1, q_heads, chunk_size, kv_len], which fits. K and V are projections of the
    hidden states, so the resulting cache is bit-identical to a single-shot
    prefill; only the peak activation differs.
    """
    cache = DynamicCache()
    out = None
    total = input_ids.shape[1]
    for start in range(0, total, chunk_size):
        piece = input_ids[:, start:start + chunk_size]
        past = cache.get_seq_length()
        cache_position = torch.arange(past, past + piece.shape[1], device=device)
        attention_mask = torch.ones(1, past + piece.shape[1], dtype=torch.long, device=device)
        out = model(
            input_ids=piece,
            past_key_values=cache,
            attention_mask=attention_mask,
            cache_position=cache_position,
            use_cache=True,
            logits_to_keep=1,
        )
        cache = out.past_key_values
    return cache, out


def decode(model, cache, next_token, n_steps, device, start_position):
    """Greedy decode n_steps tokens against an existing cache.

    start_position is the ORIGINAL sequence position of the next token, which
    after eviction is not the same as the cache length. RoPE is applied to K
    before it is written to the cache, so surviving entries keep the phase they
    were built with and their original positions stay non-contiguous. That is
    correct and must not be undone.

    The model defaults position_ids to `arange(q_len) + past_key_values.
    get_seq_length()`, i.e. the COMPACTED length. Left alone, that silently
    renumbers the continuation to close the gaps the eviction opened: after
    dropping half of a 512-token cache the next token would be given RoPE phase
    256 while the surviving keys hold phases up to 510, so the query lands
    *behind* keys it should follow. The cache never looks corrupt; quality just
    degrades for reasons that are invisible at this layer.

    So the two indices are passed explicitly and kept apart:
      cache_position -- physical slot to write into, tracks the compacted cache
      position_ids   -- logical position for RoPE, tracks the original sequence
    """
    produced = []
    token = next_token
    position = start_position
    # A per-layer budget schedule (PyramidKV) leaves layers with different cache
    # lengths, which the model's mask construction cannot express: the causal
    # mask is built once and shared by every layer of a given attention type, so
    # a mask sized from layer 0 fails a shape check in every shorter layer.
    # Passing attention_mask=None does not help, because the model then builds
    # the mask itself from the same single length.
    #
    # The model does accept a pre-built mask mapping and skips construction
    # entirely when given one, and eager attention skips the mask add when the
    # mask is None. For single-token decoding nothing needs masking anyway --
    # the new token follows everything cached -- so a ragged cache is decoded
    # with an explicitly empty mapping and each layer uses its own length.
    ragged = len({l.keys.shape[2] for l in cache.layers}) > 1
    ragged_mask = {t: None for t in getattr(model.config, "layer_types", None)
                   or ["full_attention"]}
    for _ in range(n_steps):
        past = cache.get_seq_length()
        cache_position = torch.tensor([past], device=device)
        position_ids = torch.tensor([[position]], device=device)
        attention_mask = (ragged_mask if ragged else
                          torch.ones(1, past + 1, dtype=torch.long, device=device))
        out = model(
            input_ids=token.view(1, 1),
            past_key_values=cache,
            attention_mask=attention_mask,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=True,
            logits_to_keep=1,
        )
        cache = out.past_key_values
        token = out.logits[0, -1].argmax()
        produced.append(int(token))
        position += 1
    return produced, cache


def build_probe_ids(tokenizer, length, device):
    """Tokenise real text and tile it to exactly `length` tokens."""
    base = tokenizer(PROBE_TEXT, return_tensors="pt").input_ids
    reps = -(-length // base.shape[1])
    return base.repeat(1, reps)[:, :length].to(device)


def kv_cache_bytes(cache):
    return sum(l.keys.numel() * l.keys.element_size()
               + l.values.numel() * l.values.element_size() for l in cache.layers)


def write_log(args, gates, started, exit_reason):
    payload = {
        "run": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": round(time.time() - started, 2),
            "exit_reason": exit_reason,
            "all_passed": gates.all_passed,
        },
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": transformers.__version__,
        },
        "gates": gates.results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nLogged to {args.out}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--context-lengths", type=int, nargs="+",
                    default=[4096, 8192, 16384],
                    help="context lengths for the memory sweep (gate 6). Stops at "
                         "16K by default: that is the primary arm's cap, chosen to "
                         "stay clear of Qwen2.5-1.5B's 32768 position ceiling, not "
                         "a memory limit. Pass 32768 explicitly for the Llama arm")
    ap.add_argument("--attn-probe-len", type=int, default=512,
                    help="sequence length for the output_attentions probe (gates 3-5). "
                         "Kept short on purpose: the attention tensor is O(L^2) per "
                         "layer. NOT a scoring observation window -- it is sized to "
                         "stress the eviction path, not to score tokens. Scoring code "
                         "must enforce its own 32-64 token window and must not inherit "
                         "this default")
    ap.add_argument("--expect-kv-heads", type=int, default=None,
                    help="override the expected KV head count in gate 4 "
                         "(default: the model config's num_key_value_heads)")
    ap.add_argument("--expect-attn-heads", type=int, default=None,
                    help="override the expected query head count in gate 4 "
                         "(default: the model config's num_attention_heads)")
    ap.add_argument("--prefill-chunk", type=int, default=512,
                    help="chunk size for eager prefill (gates 5-6)")
    ap.add_argument("--decode-tokens", type=int, default=20,
                    help="tokens to decode after eviction (gate 5)")
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="fp16",
                    help="weight/activation dtype. fp16 is the project default. "
                         "Qwen2.5 overflows in fp16 and emits NaN -- it needs bf16, "
                         "which costs the same 2 bytes per element and so leaves "
                         "every memory number unchanged")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "gate_check.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    gates = Gates()
    started = time.time()

    # ---- Gate 1: CUDA + VRAM ------------------------------------------------
    cuda_ok = torch.cuda.is_available()
    vram_gib = None
    device_name = None
    if cuda_ok:
        props = torch.cuda.get_device_properties(0)
        device_name = props.name
        vram_gib = round(props.total_memory / 1024 ** 3, 2)
    gates.record("cuda_available", cuda_ok, device=device_name, total_vram_gib=vram_gib)
    if not cuda_ok:
        print("\nCUDA is unavailable; the remaining gates cannot run.")
        write_log(args, gates, started, exit_reason="no_cuda")
        return 1

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()

    # ---- Gate 2: load model in FP16 with eager attention --------------------
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            dtype=DTYPES[args.dtype],
            attn_implementation="eager",
            # Never device_map="auto". "auto" silently offloads whatever does not
            # fit to CPU, which puts part of the KV cache in system RAM and makes
            # every memory and latency number meaningless. {"": 0} pins the whole
            # model to GPU 0 and raises if it does not fit -- a crash tells the
            # truth, a silent offload does not. It also streams weights shard by
            # shard straight to VRAM instead of staging the full model in CPU RAM.
            device_map={"": 0},
        ).eval()
        cfg = model.config
        param_device = next(model.parameters()).device
        load_ok = (model.dtype == DTYPES[args.dtype]
                   and cfg._attn_implementation == "eager"
                   and param_device.type == "cuda")
        gates.record("model_loaded_fp16_eager", load_ok,
                     model_id=args.model_id,
                     dtype=str(model.dtype),
                     param_device=str(param_device),
                     attn_implementation=cfg._attn_implementation,
                     num_hidden_layers=cfg.num_hidden_layers,
                     num_attention_heads=cfg.num_attention_heads,
                     num_key_value_heads=cfg.num_key_value_heads,
                     weights_gib=round(torch.cuda.memory_allocated() / 1024 ** 3, 3))
    except Exception as exc:
        gates.record("model_loaded_fp16_eager", False,
                     model_id=args.model_id, error=f"{type(exc).__name__}: {exc}")
        write_log(args, gates, started, exit_reason="model_load_failed")
        return 1

    if not load_ok:
        print("\nModel is not the requested dtype / eager / on cuda:0. Refusing to continue: "
              "any memory or latency number measured from here would be invalid.")
        write_log(args, gates, started, exit_reason="bad_model_placement")
        return 1

    probe_ids = build_probe_ids(tokenizer, args.attn_probe_len, device)

    # ---- Gate 3: attentions are actually returned ---------------------------
    with torch.no_grad():
        out = model(probe_ids, use_cache=True, output_attentions=True, logits_to_keep=1)

    attentions = out.attentions
    if attentions is None:
        gates.record("attentions_returned", False,
                     detail="attentions is None -- eager attention is not in effect, "
                            "attention scores cannot be read, the study cannot proceed")
        write_log(args, gates, started, exit_reason="attentions_none")
        raise SystemExit("FATAL: output_attentions=True returned None.")

    k0, access_path = layer0_k(out.past_key_values)
    k_shape = tuple(k0.shape)
    attn_shape = tuple(attentions[0].shape)
    gates.record("attentions_returned", True,
                 cache_access_path=access_path,
                 k_cache_layer0_shape=list(k_shape),
                 k_cache_layout="[batch, kv_heads, seq, head_dim]",
                 attentions_layer0_shape=list(attn_shape),
                 attentions_layout="[batch, attn_heads, q_len, kv_len]",
                 num_layers_with_attentions=len(attentions))

    # ---- Gate 3b: the forward pass is numerically valid --------------------
    # A NaN here makes every downstream number meaningless while every other
    # gate still passes: shapes stay correct, nothing crashes, and argmax
    # silently returns token 0 forever. Qwen2.5 in fp16 fails exactly this way.
    logits = out.logits.float()
    finite_logits = bool(torch.isfinite(logits).all())
    finite_attn = all(bool(torch.isfinite(a).all()) for a in attentions)
    gates.record("forward_pass_finite", finite_logits and finite_attn,
                 logits_finite=finite_logits,
                 attentions_finite=finite_attn,
                 logits_absmax=(round(logits.abs().max().item(), 2)
                                if finite_logits else "nan/inf"),
                 dtype=args.dtype,
                 hint=("" if finite_logits else
                       "non-finite activations. Qwen2.5 overflows in fp16; "
                       "rerun with --dtype bf16, which uses the same 2 bytes "
                       "per element and changes no memory number"))

    # ---- Gate 4: GQA, not MHA -----------------------------------------------
    # Expected head counts come from the model's own config unless overridden,
    # so this gate checks that the runtime tensors agree with what the model
    # declares -- a stronger claim than matching a constant typed into a script,
    # and one that holds for any GQA model rather than only Llama-3.2-1B.
    kv_heads_in_cache = k_shape[1]
    attn_heads_in_scores = attn_shape[1]
    expect_kv = args.expect_kv_heads if args.expect_kv_heads else cfg.num_key_value_heads
    expect_attn = args.expect_attn_heads if args.expect_attn_heads else cfg.num_attention_heads
    is_gqa = kv_heads_in_cache < attn_heads_in_scores
    gqa_ok = (kv_heads_in_cache == expect_kv
              and attn_heads_in_scores == expect_attn
              and is_gqa
              and attn_heads_in_scores % kv_heads_in_cache == 0)
    group = attn_heads_in_scores // kv_heads_in_cache if kv_heads_in_cache else 0
    gates.record("gqa_confirmed", gqa_ok,
                 kv_heads_in_k_cache=kv_heads_in_cache,
                 attn_heads_in_scores=attn_heads_in_scores,
                 expected_kv_heads=expect_kv,
                 expected_attn_heads=expect_attn,
                 expectation_source="cli" if args.expect_kv_heads else "model config",
                 is_gqa=is_gqa,
                 gqa_group_size=group,
                 budget_units=kv_heads_in_cache,
                 note=f"{kv_heads_in_cache} KV heads behind {attn_heads_in_scores} "
                      f"query heads: evicting one KV entry removes it for all "
                      f"{group} query heads in its group, so the study has "
                      f"{kv_heads_in_cache} budget units, not {attn_heads_in_scores}")

    del out, attentions
    torch.cuda.empty_cache()

    # ---- Gate 5: evict 50%, keep decoding -----------------------------------
    try:
        with torch.no_grad():
            cache, out = chunked_prefill(model, probe_ids, args.prefill_chunk, device)
            before = cache.get_seq_length()
            # Reference copy of the pre-eviction keys, to prove afterwards that
            # surviving entries were carried over untouched rather than rebuilt.
            reference = cache.layers[0].keys.clone()
            idx = evict_half(cache)
            after = cache.get_seq_length()
            kept = idx.numel()
            rope_preserved = torch.equal(cache.layers[0].keys,
                                         reference.index_select(-2, idx))
            del reference
            first_token = out.logits[0, -1].argmax()
            # Continue from the ORIGINAL position, not the compacted length.
            produced, cache = decode(model, cache, first_token, args.decode_tokens,
                                     device, start_position=before)
        # All-identical output is the signature of NaN logits, not of eviction
        # damage. Coherence still needs a human read, but degeneracy is catchable.
        distinct = len(set(produced))
        evict_ok = (after == kept
                    and after <= (before + 1) // 2
                    and rope_preserved
                    and distinct > 1
                    and len(produced) == args.decode_tokens
                    and cache.get_seq_length() == after + args.decode_tokens)
        gates.record("evict_half_then_decode", evict_ok,
                     cache_len_before=before,
                     cache_len_after_eviction=after,
                     fraction_kept=round(after / before, 4),
                     rope_phase_preserved=bool(rope_preserved),
                     retained_original_positions=f"0,2,4,...,{before - 2}",
                     continuation_position_ids=f"{before}..{before + args.decode_tokens - 1}",
                     continuation_cache_slots=f"{after}..{after + args.decode_tokens - 1}",
                     tokens_decoded=len(produced),
                     distinct_tokens=distinct,
                     final_cache_len=cache.get_seq_length(),
                     decoded_preview=tokenizer.decode(produced))
        del cache, out
    except Exception as exc:
        gates.record("evict_half_then_decode", False, error=f"{type(exc).__name__}: {exc}")
    torch.cuda.empty_cache()

    # ---- Gate 6: peak memory sweep ------------------------------------------
    sweep = []
    max_pos = getattr(cfg, "max_position_embeddings", None)
    for length in args.context_lengths:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        entry = {"context_length": length}
        if max_pos is not None and length > max_pos:
            entry.update(status="SKIPPED",
                         reason=f"exceeds max_position_embeddings={max_pos}")
            sweep.append(entry)
            print(f"[SKIP] memory @ {length}: exceeds max_position_embeddings={max_pos}")
            continue
        try:
            ids = torch.randint(0, cfg.vocab_size, (1, length), device=device)
            t0 = time.perf_counter()
            with torch.no_grad():
                cache, _ = chunked_prefill(model, ids, args.prefill_chunk, device)
            torch.cuda.synchronize()
            entry.update(
                status="OK",
                peak_allocated_gib=round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3),
                kv_cache_gib=round(kv_cache_bytes(cache) / 1024 ** 3, 3),
                prefill_seconds=round(time.perf_counter() - t0, 3),
                cache_len=cache.get_seq_length(),
            )
            print(f"[ OK ] memory @ {length:>6}: peak {entry['peak_allocated_gib']} GiB "
                  f"(KV {entry['kv_cache_gib']} GiB, {entry['prefill_seconds']}s)")
            del cache, ids
        except torch.cuda.OutOfMemoryError as exc:
            entry.update(status="OOM", error=str(exc).split("\n")[0])
            print(f"[FAIL] memory @ {length}: OOM")
        except Exception as exc:
            entry.update(status="ERROR", error=f"{type(exc).__name__}: {exc}")
            print(f"[FAIL] memory @ {length}: {type(exc).__name__}: {exc}")
        sweep.append(entry)

    sweep_ok = all(e["status"] in ("OK", "SKIPPED") for e in sweep)
    gates.record("memory_sweep", sweep_ok,
                 prefill_chunk=args.prefill_chunk,
                 measurements=sweep)

    write_log(args, gates, started, exit_reason=None)
    print("\n" + ("ALL GATES PASSED" if gates.all_passed else "ONE OR MORE GATES FAILED"))
    return 0 if gates.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
