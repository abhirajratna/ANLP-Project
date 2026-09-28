"""GSM8K/MBPP as teacher-forced documents: the question is never masked (T1 step 2)."""
from dataclasses import dataclass, field

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from c3.assemble import load_traces
from c3.data.tasks import (TASKS, end_of_turn_id, format_documents, question_answer,
                           strip_calculator)
from c3.traces.collect import CollectConfig, collect
from c3.traces.models import ToyDiffusionLM


@dataclass
class SpyToy(ToyDiffusionLM):
    seen: list = field(default_factory=list)

    def logits(self, input_ids):
        self.seen.extend(input_ids.detach().cpu().numpy().copy())
        return super().logits(input_ids)


def _small_cfg(**kw):
    return CollectConfig(**{"mask_rates": (0.3, 0.7), "masks_per_rate": 2,
                            "decoder_k": (4, 8), **kw})



class StubChatTokenizer:
    """Character-level ids with a two-role chat template; 58 ends a turn."""

    chat_template = "stub"
    eos_token_id = 1

    def _enc(self, text):
        return [2 + (ord(c) % 50) for c in text]

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": self._enc(text)}

    def apply_chat_template(self, msgs, add_generation_prompt=False, tokenize=True):
        ids = []
        for m in msgs:
            ids += [55, 56 if m["role"] == "user" else 57] + self._enc(m["content"]) + [58]
        return ids + ([55, 57] if add_generation_prompt else [])


def test_task_documents_mask_only_the_answer(tmp_path):
    tok = StubChatTokenizer()
    assert end_of_turn_id(tok) == 58
    assert strip_calculator("3*2=<<3*2=6>>6 so") == "3*2=6 so"
    recs = [{"doc_id": 0, "question": "What is 3*2?", "answer": "3*2=<<3*2=6>>6\n#### 6"},
            {"doc_id": 1, "question": "And 4+4?", "answer": "4+4=<<4+4=8>>8\n#### 8"}]
    docs = format_documents("gsm8k", recs, tok)
    d = docs[0]
    prompt = tok.apply_chat_template([{"role": "user", "content": "What is 3*2?"}],
                                     add_generation_prompt=True)
    assert d.span == (len(prompt), len(prompt) + len("3*2=6\n#### 6") + 1)
    assert list(d.ids[:d.span[0]]) == prompt and d.ids[-1] == 58

    spy = SpyToy()
    df = load_traces(collect(spy, docs, _small_cfg(), tmp_path / "g.parquet", progress_every=0))
    assert set(df.dataset) == {"gsm8k"}
    for doc in docs:
        rows = df[df.doc_id == doc.doc_id]
        assert rows.position.min() >= doc.span[0] and rows.position.max() < doc.span[1]
    # No question token is ever masked in any input the model saw.
    starts = {len(d.ids): d.span[0] for d in docs}
    for x in spy.seen:
        s = starts[len(x)]
        assert not np.any(x[:s] == spy.mask_token_id)
    # Rates are fractions of the answer span.
    uni = df[df.mask_kind == "uniform"].groupby("state_id").first()
    assert np.allclose(uni.mask_rate, uni.n_masked / uni.span_len)


def test_task_documents_need_a_chat_template():
    tok = StubChatTokenizer()
    tok.chat_template = None
    with pytest.raises(ValueError, match="chat template"):
        format_documents("gsm8k", [{"doc_id": 0, "question": "q", "answer": "a"}], tok)


def test_task_specs_are_pinned_and_prompts_are_built():
    assert TASKS["gsm8k"].n == 1319 and TASKS["mbpp"].n == 500
    assert all(len(t.revision) == 40 for t in TASKS.values())
    q, a = question_answer("mbpp", {"text": "Add two numbers. ", "code": "def f(a,b):\r\n  return a+b",
                                    "test_list": ["assert f(1,2)==3", "assert f(0,0)==0"]})
    assert q.endswith("assert f(1,2)==3\nassert f(0,0)==0") and "\r" not in a
