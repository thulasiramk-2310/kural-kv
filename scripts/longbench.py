#!/usr/bin/env python3
"""LongBench task loading, prompting and scoring.

LongBench is the generation-quality half of the study. Nearly every method being
reproduced reports on it, which is the point: unlike the synthetic needle task,
a number produced here is comparable to a published one.

That comparability is fragile and rests on details that are easy to get wrong:

  Prompt templates. A home-made prompt produces a number that cannot be compared
  to anything. The official `dataset2prompt.json` and `dataset2maxlen.json` are
  fetched from the LongBench repository and cached under `configs/longbench/`, so
  the exact templates a run used are pinned alongside its results.

  Middle truncation. When a prompt exceeds the context length LongBench keeps the
  first and last halves and drops the middle, because the instruction sits at the
  end and the task description at the start. Truncating from the end instead
  would remove the question and score a different task.

  Chat template. Applied for the tasks that expect it and skipped for the
  completion-style ones (code, and the few classification tasks), following the
  reference implementation.

The metric is LongBench's own token-level F1 after SQuAD-style normalisation,
taken as the maximum over the reference answers.

`datasets` cannot load THUDM/LongBench: version 5 dropped script-based loaders
and the repository ships one. The per-task JSONL files are read directly from the
`data.zip` published in the same repository, which is the same content the script
would have produced.
"""

import json
import re
import string
import zipfile
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "longbench" / "data"
CONFIG_DIR = REPO_ROOT / "configs" / "longbench"
CONFIG_BASE = "https://raw.githubusercontent.com/THUDM/LongBench/main/LongBench/config/"

# Following the reference implementation, these tasks are completion-style and
# are NOT wrapped in a chat template.
NO_CHAT_TEMPLATE = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}

# Metric per task. Only the F1 family is implemented so far; summarization
# (ROUGE) and code (edit similarity) are not yet wired up, and asking for such a
# task raises rather than silently scoring it with the wrong metric.
TASK_METRIC = {
    "multifieldqa_en": "qa_f1", "qasper": "qa_f1", "narrativeqa": "qa_f1",
    "hotpotqa": "qa_f1", "2wikimqa": "qa_f1", "musique": "qa_f1",
}


def ensure_configs():
    """Fetch and pin the official prompt/length configs."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for name in ("dataset2prompt.json", "dataset2maxlen.json"):
        path = CONFIG_DIR / name
        if not path.exists():
            import urllib.request
            with urllib.request.urlopen(CONFIG_BASE + name, timeout=30) as r:
                path.write_bytes(r.read())
        out[name] = json.loads(path.read_text(encoding="utf-8"))
    return out["dataset2prompt.json"], out["dataset2maxlen.json"]


def ensure_data():
    """Extract the per-task JSONL files from the published data.zip if needed."""
    if DATA_DIR.exists() and any(DATA_DIR.glob("*.jsonl")):
        return DATA_DIR
    from huggingface_hub import hf_hub_download
    zp = hf_hub_download("THUDM/LongBench", "data.zip", repo_type="dataset")
    DATA_DIR.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zp) as z:
        z.extractall(DATA_DIR.parent)
    return DATA_DIR


# --------------------------------------------------------------------------
# Metric: LongBench's qa_f1_score
# --------------------------------------------------------------------------
def normalize_answer(s):
    def remove_articles(t):
        return re.sub(r"\b(a|an|the)\b", " ", t)
    def white_space_fix(t):
        return " ".join(t.split())
    def remove_punc(t):
        return "".join(ch for ch in t if ch not in set(string.punctuation))
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def f1(prediction, ground_truth):
    pred = normalize_answer(prediction).split()
    gold = normalize_answer(ground_truth).split()
    common = Counter(pred) & Counter(gold)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / len(pred)
    recall = same / len(gold)
    return 2 * precision * recall / (precision + recall)


def qa_f1(prediction, references):
    """Maximum F1 over the reference answers, as LongBench scores it."""
    return max((f1(prediction, r) for r in references), default=0.0)


METRICS = {"qa_f1": qa_f1}


# --------------------------------------------------------------------------
class LongBenchTask:
    """One LongBench task, prompted and scored per the reference implementation."""

    name = "longbench"

    def __init__(self, task, context_len, tokenizer, min_natural_tokens=0):
        if task not in TASK_METRIC:
            raise SystemExit(
                f"task {task!r} has no implemented metric. Available: "
                f"{sorted(TASK_METRIC)}. Summarization (ROUGE) and code (edit "
                f"similarity) are not wired up; scoring them with F1 would be wrong.")
        prompts, maxlens = ensure_configs()
        data_dir = ensure_data()
        path = data_dir / f"{task}.jsonl"
        if not path.exists():
            raise SystemExit(f"missing {path}")
        self.task = task
        self.rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        self.template = prompts[task]
        self.max_gen = maxlens[task]
        self.metric = METRICS[TASK_METRIC[task]]
        self.metric_name = TASK_METRIC[task]
        self.context_len = context_len
        self.tok = tokenizer
        # Reserve the chat template's own tokens. Truncating the content to the
        # full context and templating afterwards overflows, and trimming the
        # overflow from the front decapitates the system block -- the prompt
        # then differs structurally from what the reference implementation runs.
        self.overhead = 0
        if task not in NO_CHAT_TEMPLATE:
            wrapped = tokenizer.apply_chat_template(
                [{"role": "user", "content": ""}], tokenize=False,
                add_generation_prompt=True)
            self.overhead = len(tokenizer(wrapped).input_ids)

        # Comparing one context length against another is only controlled if the
        # SAME documents are truncated to both. LongBench prompts vary widely in
        # natural length -- multifieldqa_en has a median of 7.8k tokens, so a run
        # at context 16384 would mostly not be at 16384 at all, and the
        # comparison would confound context length with document length. Filtering
        # to documents at least as long as the largest context under test makes
        # every sample saturate every context length being compared.
        if min_natural_tokens:
            keep = []
            for row in self.rows:
                n = len(tokenizer(self.template.format(**row),
                                  truncation=False).input_ids)
                if n >= min_natural_tokens:
                    keep.append(row)
            self.rows = keep
        self.min_natural_tokens = min_natural_tokens

    def __len__(self):
        return len(self.rows)

    def sample(self, i, device):
        row = self.rows[i % len(self.rows)]
        prompt = self.template.format(**row)
        ids = self.tok(prompt, truncation=False, return_tensors="pt").input_ids[0]

        # Middle truncation: the task description is at the start and the
        # question at the end, so both must survive. Cutting from the end would
        # drop the question and silently score a different task.
        room = self.context_len - self.overhead
        if len(ids) > room:
            half = room // 2
            prompt = (self.tok.decode(ids[:half], skip_special_tokens=True)
                      + self.tok.decode(ids[-half:], skip_special_tokens=True))

        if self.task not in NO_CHAT_TEMPLATE:
            prompt = self.tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True)
        input_ids = self.tok(prompt, return_tensors="pt").input_ids
        assert input_ids.shape[1] <= self.context_len, (
            f"prompt {input_ids.shape[1]} exceeds context {self.context_len}; "
            "truncation reserve is wrong")
        return {"input_ids": input_ids.to(device),
                "reference": row["answers"],
                "max_gen": self.max_gen,
                "id": row.get("_id", str(i)),
                "meta": {"length": row.get("length"), "dataset": row.get("dataset")}}

    def score(self, text, reference):
        return self.metric(text, reference)


if __name__ == "__main__":
    # Self-check of the metric against hand-worked cases, so a regression in
    # normalisation shows up here rather than as a quietly depressed benchmark.
    cases = [
        ("the Eiffel Tower", ["Eiffel Tower"], 1.0),
        ("Paris", ["Paris, France"], 2 * 1.0 * 0.5 / 1.5),
        ("completely wrong", ["Eiffel Tower"], 0.0),
    ]
    for pred, refs, want in cases:
        got = qa_f1(pred, refs)
        status = "ok " if abs(got - want) < 1e-9 else "FAIL"
        print(f"[{status}] qa_f1({pred!r}, {refs}) = {got:.4f} (want {want:.4f})")
