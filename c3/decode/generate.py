"""T12: free-running generation on GSM8K and MBPP (plan v2 T12; plan to the final, Parth 3).

One problem is decoded as LLaDA's reference sampler does it: the chat-formatted
question, then ``gen_length`` mask tokens (256), filled block by block
(``block_length`` 32) by one commit policy through ``c3.decode.loop.decode``, greedy.
The prompt is built exactly as the task traces build theirs
(``c3.data.tasks.question_answer`` under the chat template), so a calibrator fitted
on those traces sees the same prompts.

The completion is the generated text up to the first end-of-turn or EOS token: the
model fills the rest of the 256 slots with those tokens, as LLaDA does.  Speed is
counted over the whole generated region, as LLaDA and Fast-dLLM report it: tokens per
step = gen_length / forward passes.  Scoring (GSM8K exact match, MBPP pass@1) is a
separate step on the saved completions.
"""

from __future__ import annotations

import time

import numpy as np


def task_prompt(task: str, rec: dict, tokenizer) -> list[int]:
    """The chat-formatted question of one task record, as the traces format it."""
    from ..data.tasks import question_answer

    q, _ = question_answer(task, rec)
    return list(tokenizer.apply_chat_template([{"role": "user", "content": q}],
                                              add_generation_prompt=True, tokenize=True))


def stop_ids(tokenizer) -> set[int]:
    """Tokens that end a completion: the chat template's end-of-turn token and EOS."""
    from ..data.tasks import end_of_turn_id

    out = {int(end_of_turn_id(tokenizer))}
    if tokenizer.eos_token_id is not None:
        out.add(int(tokenizer.eos_token_id))
    return out


def cut_at_stop(ids, stops) -> tuple[list[int], bool]:
    """(the tokens before the first stop token, whether a stop token was generated)."""
    ids = [int(t) for t in ids]
    for i, t in enumerate(ids):
        if t in stops:
            return ids[:i], True
    return ids, False


def generate(model, prompt_ids, policy, stops, gen_length: int = 256,
             block_length: int | None = 32) -> dict:
    """Decode ``gen_length`` masked slots after ``prompt_ids`` with ``policy``.

    Returns the completion token ids (cut at the first stop token), whether a stop
    token was generated, and the speed: forward passes, tokens committed, tokens per
    step and wall-clock seconds (including the per-step statistics, which every policy
    pays for).
    """
    from .loop import decode

    prompt = np.asarray(prompt_ids, dtype=np.int64)
    x = np.concatenate([prompt, np.full(gen_length, int(model.mask_token_id), np.int64)])
    t0 = time.perf_counter()
    res = decode(model, x, policy, block_length=block_length, record=False)
    seconds = time.perf_counter() - t0
    completion, stopped = cut_at_stop(res.tokens[prompt.size:], stops)
    return {"completion_ids": completion, "stopped": stopped,
            "n_forward": int(res.n_forward), "n_committed": int(res.n_committed),
            "tokens_per_step": float(res.tokens_per_step), "seconds": float(seconds)}
