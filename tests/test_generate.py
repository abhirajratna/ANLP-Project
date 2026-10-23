"""T12 driver: free-running generation (c3.decode.generate, scripts/run_t12.py) on the CPU toy."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from c3.decode.generate import cut_at_stop, generate, stop_ids, task_prompt
from c3.decode.policies import Threshold, TopKConfidence
from c3.traces.models import ToyDiffusionLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_t12  # noqa: E402


class ChatTok:
    """A tiny chat tokenizer: one id per character, <eot> = 1, EOS = 2."""
    eos_token_id = 2
    chat_template = "fake"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [10 + (ord(c) % 40) for c in text]}

    def apply_chat_template(self, msgs, add_generation_prompt=False, tokenize=True):
        # user turn: 3 ... 4; assistant turn: 5 ... <eot>; the generation prompt opens one
        ids = []
        for m in msgs:
            body = self(m["content"])["input_ids"]
            ids += [5] + body + [1] if m["role"] == "assistant" else [3] + body + [4]
        return ids + ([5] if add_generation_prompt else [])


def test_cut_at_stop():
    assert cut_at_stop([7, 8, 1, 9, 2], {1, 2}) == ([7, 8], True)
    assert cut_at_stop([7, 8], {1}) == ([7, 8], False)


def test_prompt_and_stop_tokens_follow_the_chat_template():
    tok = ChatTok()
    p = task_prompt("gsm8k", {"question": "ab", "answer": "x #### 1", "doc_id": 0}, tok)
    assert p == [3, 10 + ord("a") % 40, 10 + ord("b") % 40, 4, 5]     # user turn + generation prompt
    assert stop_ids(tok) == {1, 2}


@pytest.mark.parametrize("k", [1, 2, 4])
def test_generation_fills_the_region_block_by_block(k):
    rng = np.random.default_rng(0)
    prompt = rng.integers(10, 60, 8)
    model = ToyDiffusionLM(noise_dispersion=0.3)
    model.set_reference(np.concatenate([prompt, rng.integers(10, 60, 32)]))
    g = generate(model, prompt, TopKConfidence(k), stops={999}, gen_length=32, block_length=8)
    assert g["n_committed"] == 32 and g["n_forward"] == 32 // k
    assert g["tokens_per_step"] == pytest.approx(k)
    assert g["seconds"] > 0 and not g["stopped"] and len(g["completion_ids"]) == 32


class Exact:
    """Predicts the reference token at every position, confidently."""
    mask_token_id, input_device, name = 63, None, "exact"

    def __init__(self, ref):
        self.ref = torch.as_tensor(ref)

    def logits(self, t):
        L = t.shape[1]
        out = torch.full((1, L, 64), -5.0)
        out[0, torch.arange(L), self.ref[:L]] = 5.0
        return out


def test_the_completion_stops_at_the_first_stop_token():
    rng = np.random.default_rng(1)
    prompt = rng.integers(10, 60, 8)
    ref = np.concatenate([prompt, rng.integers(10, 60, 12), [1], rng.integers(10, 60, 19)])
    g = generate(Exact(ref), prompt, Threshold(0.5), stops={1}, gen_length=32, block_length=8)
    assert g["stopped"] and g["completion_ids"] == ref[8:20].tolist()
    assert g["n_forward"] == 4                              # every slot of a block clears 0.5


def test_driver_selects_the_half_limit_and_shard(tmp_path):
    recs = [{"doc_id": i} for i in range(20)]
    man = tmp_path / "split.json"
    man.write_text(json.dumps({str(i): i % 2 for i in range(20)}))
    got = run_t12.select_records(recs, man, "report", 6, 0, 1)
    assert [r["doc_id"] for r in got] == [1, 3, 5, 7, 9, 11]
    got = run_t12.select_records(recs, man, "report", 6, 1, 2)
    assert [r["doc_id"] for r in got] == [1, 3, 5, 7, 9, 11][0:0] + [1, 3, 5, 7, 9, 11][::1] \
        if False else [r for r in [1, 3, 5, 7, 9, 11] if r % 2 == 1] and [
            r["doc_id"] for r in got] == [1, 3, 5, 7, 9, 11]
    assert len(run_t12.select_records(recs, None, "all", 0, 0, 1)) == 20


def test_driver_grid_and_policies(tmp_path):
    g = tmp_path / "grid.json"
    g.write_text(json.dumps({"configs": [{"policy": "topk_conf", "k": 2},
                                         {"policy": "c3a", "alpha": 0.1}]}))
    cfgs = run_t12.load_configs(g)
    assert run_t12.config_key(cfgs[1]) == "c3a[alpha=0.1]"
    assert isinstance(run_t12.build_policy(cfgs[0]), TopKConfidence)
    with pytest.raises(SystemExit):
        run_t12.build_policy(cfgs[1])                       # C3a without a calibrator
    g.write_text(json.dumps([{"policy": "topk_conf", "k": 2}, {"policy": "topk_conf", "k": 2}]))
    with pytest.raises(SystemExit):
        run_t12.load_configs(g)
    g.write_text(json.dumps([{"policy": "beam", "k": 2}]))
    with pytest.raises(SystemExit):
        run_t12.load_configs(g)


def test_resume_skips_recorded_pairs_and_torn_lines(tmp_path):
    p = tmp_path / "t12.shard0of1.jsonl"
    p.write_text(json.dumps({"config": "topk_conf[k=2]", "doc_id": 3}) + "\n{\"config\": \"top")
    assert run_t12.done_pairs(p) == {("topk_conf[k=2]", 3)}
