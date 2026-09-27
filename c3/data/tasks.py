"""GSM8K and MBPP as teacher-forced infilling documents (plan v2 Sec. 3.2, T1 step 2).

The question is the prompt and the reference answer is the masked target:
``chat_template(question) + answer + end-of-turn``, with only the answer and its
end-of-turn token maskable.  Masking the question would measure how well the
model reconstructs the task, not how calibrated its answer is.

Why the span is built this way rather than as "full rendering minus prompt":
LLaDA-8B-Instruct's template re-appends an assistant header after the last
message, so rendering [user, assistant] adds tokens that are not part of the
answer.  Instead the end-of-turn token is read off one probe rendering per
tokenizer (``<|eot_id|>`` for LLaDA, ``<|im_end|>`` for Dream) and appended to
the answer, which is exactly what the template emits after an assistant turn.

Two choices, both recorded in every document's ``meta``:

* GSM8K's ``<<48/2=24>>`` calculator annotations are stripped by default.  They
  are an artefact of the dataset's annotation tool, not text any model is asked
  to write, and teacher-forced tokens inside them would dominate the error rate.
* MBPP uses the prompt of the MBPP paper (task text plus the three test
  asserts), zero-shot, which is what the Instruct models' reported numbers use.

Datasets are pinned to hub revisions so the 1,319 / 500 / 164 documents are
fixed; HumanEval is here only for T12's pass@1.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TaskSpec:
    path: str
    config: str | None
    split: str
    revision: str        # hub commit, checked 2026-09-28
    n: int


TASKS = {
    "gsm8k": TaskSpec("openai/gsm8k", "main", "test",
                      "740312add88f781978c0658806c59bc2815b9866", 1319),
    "mbpp": TaskSpec("google-research-datasets/mbpp", "full", "test",
                     "4bb6404fdc6cacfda99d4ac4205087b89d32030c", 500),
    "humaneval": TaskSpec("openai/openai_humaneval", None, "test",
                          "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544", 164),
}

MBPP_PROMPT = ("You are an expert Python programmer, and here is your task: {text} "
               "Your code should pass these tests:\n\n{tests}")

_CALC = re.compile(r"<<[^<>]*>>")


def strip_calculator(answer: str) -> str:
    """Remove GSM8K's ``<<a*b=c>>`` annotations, keeping the surrounding text."""
    return _CALC.sub("", answer)


def question_answer(task: str, rec: dict, strip_calc: bool = True) -> tuple[str, str]:
    """The (prompt text, reference answer) pair of one raw record."""
    if task == "gsm8k":
        ans = rec["answer"]
        return rec["question"], strip_calculator(ans) if strip_calc else ans
    if task == "mbpp":
        q = MBPP_PROMPT.format(text=rec["text"].strip(), tests="\n".join(rec["test_list"]))
        return q, rec["code"].replace("\r\n", "\n").strip()
    if task == "humaneval":
        return rec["prompt"], rec["canonical_solution"]
    raise KeyError(f"unknown task {task!r}; known: {sorted(TASKS)}")


def load_task(task: str, jsonl: str | Path | None = None) -> list[dict]:
    """Raw records, each with a ``doc_id`` (its index in the split).

    Reads ``jsonl`` (as written by :func:`save_task`) when given, else the
    pinned hub revision via ``datasets``.
    """
    spec = TASKS[task]
    if jsonl is not None:
        with open(jsonl, encoding="utf-8") as fh:
            recs = [json.loads(line) for line in fh if line.strip()]
    else:
        from datasets import load_dataset

        ds = load_dataset(spec.path, spec.config, split=spec.split, revision=spec.revision)
        recs = [{"doc_id": i, **dict(r)} for i, r in enumerate(ds)]
    if len(recs) != spec.n:
        raise ValueError(f"{task}: {len(recs)} records, expected {spec.n}")
    return recs


def save_task(task: str, out: str | Path) -> dict:
    """Download a task split to JSONL; returns a small manifest (revision, count, hash)."""
    from ..provenance import sha256_file

    recs = load_task(task)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    spec = TASKS[task]
    return {"task": task, "path": spec.path, "config": spec.config, "split": spec.split,
            "revision": spec.revision, "n": len(recs), "file": out.name,
            "sha256": sha256_file(out)}


def end_of_turn_id(tokenizer) -> int:
    """The token the chat template emits right after an assistant message."""
    probe = "x7q"
    user = [{"role": "user", "content": "hi"}]
    prompt = list(tokenizer.apply_chat_template(user, add_generation_prompt=True, tokenize=True))
    full = list(tokenizer.apply_chat_template(user + [{"role": "assistant", "content": probe}],
                                              tokenize=True))
    ans = list(tokenizer(probe, add_special_tokens=False)["input_ids"])
    j = len(prompt) + len(ans)
    if full[:len(prompt)] == prompt and full[len(prompt):j] == ans and len(full) > j:
        return int(full[j])
    if tokenizer.eos_token_id is None:
        raise ValueError("cannot find the end-of-turn token and the tokenizer has no EOS")
    return int(tokenizer.eos_token_id)


def format_documents(task: str, records: list[dict], tokenizer, max_answer_tokens: int = 256,
                     strip_calc: bool = True):
    """Records -> :class:`c3.traces.collect.Document` with the answer as the span.

    The answer is truncated to ``max_answer_tokens`` (the plan's generation
    length); a truncated answer loses its end-of-turn token.
    """
    from ..traces.collect import Document

    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(f"{task}: the tokenizer has no chat template; plan v2 runs tasks "
                         f"on the Instruct checkpoints")
    eot = end_of_turn_id(tokenizer)
    docs = []
    for rec in records:
        q, a = question_answer(task, rec, strip_calc)
        prompt = list(tokenizer.apply_chat_template([{"role": "user", "content": q}],
                                                    add_generation_prompt=True, tokenize=True))
        answer = list(tokenizer(a, add_special_tokens=False)["input_ids"]) + [eot]
        truncated = len(answer) > max_answer_tokens
        answer = answer[:max_answer_tokens]
        docs.append(Document(
            doc_id=int(rec["doc_id"]), ids=prompt + answer,
            span=(len(prompt), len(prompt) + len(answer)), dataset=task,
            meta={"task_id": rec.get("task_id", rec["doc_id"]), "truncated": truncated,
                  "strip_calculator": bool(strip_calc and task == "gsm8k")}))
    return docs
