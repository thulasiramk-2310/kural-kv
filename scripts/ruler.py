#!/usr/bin/env python3
"""RULER needle-in-a-haystack tasks, generated per NVIDIA's specification.

RULER replaces the synthetic needle as the *reported* retrieval benchmark. It is
generated from a config rather than downloaded, so the document scarcity that
made LongBench's long-context comparison awkward does not arise: any number of
samples can be produced at any context length, and every sample saturates the
context exactly.

Faithfulness is the point, so the generator, the needle string, the prompt
template and the answer prefix are taken from the RULER sources, cached under
`configs/ruler/` alongside the results that used them:

    needle:   "One of the special magic {type_needle_v} for {key} is: {value}."
    template: "Some special magic {type_needle_v} are hidden within the following
               text. Make sure to memorize it. I will quiz you about the
               {type_needle_v} afterwards.\\n{context}\\nWhat are all the special
               magic {type_needle_v} for {query} mentioned in the provided text?"

Variants implemented here are those whose haystack needs no external corpus:

    niah_single_1    noise haystack, word key -> 7-digit number
    niah_multikey_2  haystack built from OTHER needles, word key -> number
    niah_multikey_3  haystack built from other needles, uuid key -> uuid

`multikey_2` and `multikey_3` are the demanding cases: the haystack is itself
composed of distractor needles, so a policy cannot succeed by recognising that
needles look different from filler. The `essay` variants additionally require the
Paul Graham corpus and an NLTK sentence tokeniser and are not implemented; asking
for one raises rather than silently substituting a different haystack.

Scoring is RULER's `string_match_all`: the fraction of expected values that
appear in the prediction. For a single-needle task that is exact retrieval.
"""

import json
import random
import re
import uuid as _uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs" / "ruler"

NEEDLE = "One of the special magic {type_needle_v} for {key} is: {value}."
TEMPLATE = ("Some special magic {type_needle_v} are hidden within the following "
            "text. Make sure to memorize it. I will quiz you about the "
            "{type_needle_v} afterwards.\n{context}\nWhat are all the special "
            "magic {type_needle_v} for {query} mentioned in the provided text?")
ANSWER_PREFIX = (" The special magic {type_needle_v} for {query} mentioned in the "
                 "provided text are")
NOISE = ("The grass is green. The sky is blue. The sun is yellow. Here we go. "
         "There and back again.")
# RULER places needles at one of 40 evenly spaced depths.
DEPTHS = [round(i * 100 / 39) for i in range(40)]

# Subset of RULER's synthetic.yaml. Only haystacks needing no external corpus.
TASKS = {
    "niah_single_1":   dict(haystack="noise",  k="words", v="numbers", nk=1, nv=1, nq=1),
    "niah_multikey_2": dict(haystack="needle", k="words", v="numbers", nk=1, nv=1, nq=1),
    "niah_multikey_3": dict(haystack="needle", k="uuids", v="uuids",   nk=1, nv=1, nq=1),
}
UNSUPPORTED = {"niah_single_2", "niah_single_3", "niah_multikey_1",
               "niah_multivalue", "niah_multiquery"}


def _words():
    from wonderwords.random_word import _get_words_from_text_file as g
    nouns = g("nounlist.txt")
    adjs = g("adjectivelist.txt")
    return sorted({f"{a}-{n}" for a in adjs for n in nouns})


class RulerTask:
    """One RULER NIAH variant at a fixed context length."""

    name = "ruler"
    metric_name = "match_all"
    max_gen = 128          # RULER's tokens_to_generate for niah

    def __init__(self, variant, context_len, tokenizer, seed=0):
        if variant in UNSUPPORTED:
            raise SystemExit(
                f"{variant!r} uses the essay haystack, which needs the Paul Graham "
                f"corpus and an NLTK tokeniser. Not implemented; substituting a "
                f"different haystack would not be RULER. Available: {sorted(TASKS)}")
        if variant not in TASKS:
            raise SystemExit(f"unknown RULER variant {variant!r}; have {sorted(TASKS)}")
        self.variant = variant
        self.cfg = TASKS[variant]
        self.context_len = context_len
        self.tok = tokenizer
        self.seed = seed
        self.word_list = _words() if "words" in (self.cfg["k"], self.cfg["v"]) else []
        self.type_v = self.cfg["v"]
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        (CONFIG_DIR / "implemented_variants.json").write_text(
            json.dumps({"source": "NVIDIA/RULER scripts/data/synthetic",
                        "variants": TASKS, "needle": NEEDLE,
                        "template": TEMPLATE, "answer_prefix": ANSWER_PREFIX,
                        "noise_haystack": NOISE, "depths": DEPTHS}, indent=2),
            encoding="utf-8")
        # Chat-template overhead, reserved so the wrapped prompt cannot overflow
        # the context and get trimmed at the front.
        wrapped = tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}], tokenize=False, add_generation_prompt=True)
        self.overhead = len(tokenizer(wrapped).input_ids)

    def __len__(self):
        return 10 ** 9      # generated on demand

    def _rand(self, rng, kind):
        if kind == "numbers":
            return str(rng.randint(10 ** 6, 10 ** 7 - 1))
        if kind == "words":
            return rng.choice(self.word_list)
        if kind == "uuids":
            return str(_uuid.UUID(int=rng.getrandbits(128)))
        raise ValueError(kind)

    def sample(self, i, device):
        rng = random.Random(self.seed * 100003 + i)
        cfg = self.cfg
        keys, values, needles = [], [], []
        for _ in range(cfg["nk"]):
            keys.append(self._rand(rng, cfg["k"]))
            vals = [self._rand(rng, cfg["v"]) for _ in range(cfg["nv"])]
            values.append(vals)
            needles += [NEEDLE.format(type_needle_v=self.type_v, key=keys[-1], value=v)
                        for v in vals]

        # Distractor needles. In the `needle` haystack the filler is itself made
        # of needles, so a policy cannot win by learning that needles look
        # unlike filler -- which is exactly what the synthetic diagnostic allows.
        def distractor():
            k = self._rand(rng, cfg["k"])
            v = self._rand(rng, cfg["v"])
            return NEEDLE.format(type_needle_v=self.type_v, key=k, value=v)

        query = keys[0]
        answers = values[0]
        room = self.context_len - self.overhead
        shell = len(self.tok(TEMPLATE.format(type_needle_v=self.type_v, context="",
                                             query=query) + ANSWER_PREFIX.format(
            type_needle_v=self.type_v, query=query)).input_ids)
        target = room - shell - sum(len(self.tok(n).input_ids) for n in needles) - 8

        filler = []
        used = 0
        unit = NOISE if cfg["haystack"] == "noise" else None
        while used < target:
            piece = unit if unit is not None else distractor()
            filler.append(piece)
            used += len(self.tok(piece).input_ids)

        # Insert the real needle(s) at RULER's evenly spaced depths.
        for n in needles:
            depth = rng.choice(DEPTHS)
            at = min(len(filler), max(0, int(len(filler) * depth / 100)))
            filler.insert(at, n)
        context = " ".join(filler)

        prompt = TEMPLATE.format(type_needle_v=self.type_v, context=context,
                                 query=query) + ANSWER_PREFIX.format(
            type_needle_v=self.type_v, query=query)
        prompt = self.tok.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True)
        ids = self.tok(prompt, return_tensors="pt").input_ids
        if ids.shape[1] > self.context_len:      # trim filler, never the query
            over = ids.shape[1] - self.context_len
            cut = re.sub(r"\s+", " ", context)
            ctx_ids = self.tok(cut, return_tensors="pt").input_ids[0]
            context = self.tok.decode(ctx_ids[: max(1, len(ctx_ids) - over - 4)],
                                      skip_special_tokens=True)
            prompt = TEMPLATE.format(type_needle_v=self.type_v, context=context,
                                     query=query) + ANSWER_PREFIX.format(
                type_needle_v=self.type_v, query=query)
            prompt = self.tok.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False,
                add_generation_prompt=True)
            ids = self.tok(prompt, return_tensors="pt").input_ids
        return {"input_ids": ids.to(device), "reference": answers,
                "max_gen": self.max_gen, "id": f"{self.variant}-{i}",
                "meta": {"key": query, "n_filler": len(filler)}}

    def score(self, text, reference):
        """RULER's string_match_all: fraction of expected values present."""
        t = text.lower()
        return sum(str(r).lower() in t for r in reference) / len(reference)


if __name__ == "__main__":
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    for v in TASKS:
        t = RulerTask(v, 2048, tok)
        s = t.sample(0, "cpu")
        txt = tok.decode(s["input_ids"][0])
        ok = all(str(r) in txt for r in s["reference"])
        print(f"{v:<17} tokens={s['input_ids'].shape[1]:>5} refs={s['reference']} "
              f"needle_present={ok}")
        print(f"    tail: {txt[-120:]!r}")
