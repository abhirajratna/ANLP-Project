"""T12: free-running generation on GSM8K and MBPP (plan v2 T12; plan to the final, Parth 3).

One problem is decoded as LLaDA's reference sampler does it: the prompt, then
``gen_length`` mask tokens (256), filled block by block (``block_length`` 32) by one
commit policy through ``c3.decode.loop.decode``, greedy.

Two prompt formats (``PROMPTS``):

* ``chat`` (the sweep): the task's question (``c3.data.tasks.question_answer``) as one
  chat turn, built exactly as the task traces build theirs, so a calibrator fitted on
  those traces sees the same prompts;
* ``lmeval`` (the sanity check against published numbers): lm-evaluation-harness's
  GSM8K few-shot context -- ``"Question: {q}\\nAnswer: {a}"`` blocks joined by blank
  lines, the target being the full train-set answer -- sent as one chat turn and
  cut at lm-eval's stop strings, as Fast-dLLM's ``eval_llada.py`` does for
  LLaDA-8B-Instruct (``num_fewshot=5``, gen length 256, block 32).  The examples are
  drawn as lm-eval's single-process sampler draws them (``random.Random(seed)``,
  seed 1234, one draw per question in document order); a multi-process lm-eval run
  draws differently, so the match is close, not exact.

Every result keeps all ``gen_length`` generated ids and the index of the first stop
token, so the cut can be redone after the fact.  The completion is the text before
the first end-of-turn / EOS token (and, for ``lmeval``, before the first stop string).
Speed is counted over the whole generated region, as LLaDA and Fast-dLLM report it:
tokens per step = gen_length / forward passes.  Scoring (GSM8K exact match, MBPP
pass@1) is a separate step on the saved completions.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np

PROMPTS = ("chat", "lmeval")
LMEVAL = {
    "gsm8k": {"split": "train", "until": ("Question:", "</s>", "<|im_end|>"), "seed": 1234,
              "text": "Question: {question}\nAnswer:", "target": " {answer}",
              "delimiter": "\n\n"},
}


def task_prompt(task: str, rec: dict, tokenizer) -> list[int]:
    """The chat-formatted question of one task record, as the traces format it."""
    from ..data.tasks import question_answer

    q, _ = question_answer(task, rec)
    return list(tokenizer.apply_chat_template([{"role": "user", "content": q}],
                                              add_generation_prompt=True, tokenize=True))


def load_fewshot_pool(task: str, jsonl: str | Path | None = None) -> list[dict]:
    """The records lm-eval draws few-shot examples from (GSM8K: the train split, at the
    revision ``c3.data.tasks.TASKS`` pins), from ``jsonl`` when given."""
    if task not in LMEVAL:
        raise KeyError(f"no lm-eval few-shot format for {task!r}; have {sorted(LMEVAL)}")
    if jsonl is not None:
        with open(jsonl, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]
    from datasets import load_dataset

    from ..data.tasks import TASKS

    spec = TASKS[task]
    ds = load_dataset(spec.path, spec.config, split=LMEVAL[task]["split"], revision=spec.revision)
    return [dict(r) for r in ds]


def lmeval_contexts(task: str, records: list[dict], pool: list[dict], k: int,
                    seed: int | None = None) -> dict[int, str]:
    """{doc_id: lm-eval few-shot context}, drawing k examples per question in doc_id order."""
    fmt = LMEVAL[task]
    rnd = random.Random(fmt["seed"] if seed is None else seed)
    out = {}
    for rec in sorted(records, key=lambda r: int(r["doc_id"])):
        shots = rnd.sample(pool, k) if k else []
        blocks = [fmt["text"].format(**s) + fmt["target"].format(**s) for s in shots]
        out[int(rec["doc_id"])] = fmt["delimiter"].join(blocks + [fmt["text"].format(**rec)])
    return out


def lmeval_prompt(context: str, tokenizer) -> list[int]:
    """Fast-dLLM's LLaDA-Instruct path: the context as one user turn, rendered to text and
    tokenized with the tokenizer's defaults."""
    text = tokenizer.apply_chat_template([{"role": "user", "content": context}],
                                         add_generation_prompt=True, tokenize=False)
    return list(tokenizer(text)["input_ids"])


def stop_ids(tokenizer) -> set[int]:
    """Tokens that end a completion: the chat template's end-of-turn token and EOS."""
    from ..data.tasks import end_of_turn_id

    out = {int(end_of_turn_id(tokenizer))}
    if tokenizer.eos_token_id is not None:
        out.add(int(tokenizer.eos_token_id))
    return out


def first_stop(ids, stops) -> int | None:
    """Index of the first stop token in ``ids``, or None."""
    for i, t in enumerate(ids):
        if int(t) in stops:
            return i
    return None


def cut_at_stop(ids, stops) -> tuple[list[int], bool]:
    """(the tokens before the first stop token, whether a stop token was generated)."""
    ids = [int(t) for t in ids]
    i = first_stop(ids, stops)
    return (ids, False) if i is None else (ids[:i], True)


def cut_at_strings(text: str, until) -> str:
    """``text`` before the first occurrence of any of ``until`` (lm-eval's stop strings)."""
    for s in until or ():
        text = text.split(s)[0]
    return text


def generate(model, prompt_ids, policy, stops, gen_length: int = 256,
             block_length: int | None = 32) -> dict:
    """Decode ``gen_length`` masked slots after ``prompt_ids`` with ``policy``.

    Returns every generated id, the index of the first stop token (None if none), the
    completion ids before it, and the speed: forward passes, tokens committed, tokens per
    step and wall-clock seconds (including the per-step statistics every policy pays for).
    """
    from .loop import decode

    prompt = np.asarray(prompt_ids, dtype=np.int64)
    x = np.concatenate([prompt, np.full(gen_length, int(model.mask_token_id), np.int64)])
    t0 = time.perf_counter()
    res = decode(model, x, policy, block_length=block_length, record=False)
    seconds = time.perf_counter() - t0
    generated = [int(t) for t in res.tokens[prompt.size:]]
    stop = first_stop(generated, stops)
    return {"generated_ids": generated, "stop_index": stop,
            "completion_ids": generated if stop is None else generated[:stop],
            "stopped": stop is not None,
            "n_forward": int(res.n_forward), "n_committed": int(res.n_committed),
            "tokens_per_step": float(res.tokens_per_step), "seconds": float(seconds)}
