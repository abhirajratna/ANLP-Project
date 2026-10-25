"""T12 driver: free-running generation (c3.decode.generate, scripts/run_t12.py).

The prompt, grid and resume logic run without torch; generation runs on the CPU toy.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from c3.decode.generate import (LMEVAL, cut_at_stop, cut_at_strings, first_stop, lmeval_contexts,
                                lmeval_prompt, stop_ids, task_prompt)
from c3.decode.policies import TopKConfidence

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
        if not tokenize:
            return "".join(f"<{m['role']}>{m['content']}" for m in msgs) + \
                ("<assistant>" if add_generation_prompt else "")
        ids = []
        for m in msgs:
            body = self(m["content"])["input_ids"]
            ids += [5] + body + [1] if m["role"] == "assistant" else [3] + body + [4]
        return ids + ([5] if add_generation_prompt else [])

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(i)) for i in ids)


def _recs(n=3):
    return [{"doc_id": i, "question": "q" * (i + 1), "answer": f"a #### {i}"} for i in range(n)]


# --------------------------------------------------------------------------- #
# prompts and stops (no torch)
# --------------------------------------------------------------------------- #

def test_cut_at_stop_and_strings():
    assert cut_at_stop([7, 8, 1, 9, 2], {1, 2}) == ([7, 8], True)
    assert cut_at_stop([7, 8], {1}) == ([7, 8], False)
    assert first_stop([7, 2, 1], {1, 2}) == 1 and first_stop([7], {1}) is None
    assert cut_at_strings("x = 4.\n\nQuestion: next", LMEVAL["gsm8k"]["until"]) == "x = 4.\n\n"
    assert cut_at_strings("no stop", None) == "no stop"


def test_prompt_and_stop_tokens_follow_the_chat_template():
    tok = ChatTok()
    p = task_prompt("gsm8k", {"question": "ab", "answer": "x #### 1", "doc_id": 0}, tok)
    assert p == [3, 10 + ord("a") % 40, 10 + ord("b") % 40, 4, 5]     # user turn + generation prompt
    assert stop_ids(tok) == {1, 2}


def test_lmeval_contexts_follow_lm_eval_gsm8k():
    pool = [{"question": f"P{i}", "answer": f"step {i}\n#### {i}"} for i in range(20)]
    recs = _recs(3)
    ctx = lmeval_contexts("gsm8k", recs, pool, k=2)
    first = ctx[0].split("\n\n")
    assert len(first) == 3 and first[-1] == "Question: q\nAnswer:"
    assert all(b.startswith("Question: P") and "\nAnswer: step " in b for b in first[:2])
    # one draw per question, in document order, from lm-eval's seed: reproducible and varied
    assert lmeval_contexts("gsm8k", recs[::-1], pool, k=2) == ctx
    assert ctx[0].split("\n\n")[:2] != ctx[1].split("\n\n")[:2]
    assert lmeval_contexts("gsm8k", recs, pool, k=0)[2] == "Question: qqq\nAnswer:"
    tok = ChatTok()
    assert lmeval_prompt(ctx[0], tok) == tok("<user>" + ctx[0] + "<assistant>")["input_ids"]


def test_lmeval_contexts_do_not_depend_on_the_number_of_shards():
    pool = [{"question": f"P{i}", "answer": f"step {i}\n#### {i}"} for i in range(20)]
    recs = [{"doc_id": i, "question": "q" * (i + 1)} for i in range(12)]
    full = run_t12.select_records(recs, None, "all", 10, 0, 1)
    one = run_t12.shard_contexts("gsm8k", full, full, pool, 3)
    two = {}
    for shard in (0, 1):
        mine = run_t12.select_records(recs, None, "all", 10, shard, 2)
        two.update(run_t12.shard_contexts("gsm8k", full, mine, pool, 3))
    assert one == two and sorted(two) == list(range(10))
    # what the driver did before: draws over the shard's own records change the examples
    own = lmeval_contexts("gsm8k", run_t12.select_records(recs, None, "all", 10, 1, 2), pool, 3)
    assert own[1] != one[1]


def test_driver_selects_the_half_limit_and_shard(tmp_path):
    recs = [{"doc_id": i} for i in range(20)]
    man = tmp_path / "split.json"
    man.write_text(json.dumps({str(i): i % 2 for i in range(20)}))
    got = run_t12.select_records(recs, man, "report", 6, 0, 1)
    assert [r["doc_id"] for r in got] == [1, 3, 5, 7, 9, 11]
    got = run_t12.select_records(recs, man, "report", 6, 0, 3)      # shard 0 of 3
    assert [r["doc_id"] for r in got] == [3, 9]
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


def test_a_torn_last_line_is_cut_before_appending(tmp_path):
    p = tmp_path / "t12.shard0of1.jsonl"
    p.write_text(json.dumps({"config": "topk_conf[k=2]", "doc_id": 3}) + "\n{\"config\": \"top")
    run_t12.repair_tail(p)
    assert p.read_text().endswith("}\n") and run_t12.done_pairs(p) == {("topk_conf[k=2]", 3)}


# --------------------------------------------------------------------------- #
# generation and the sweep (CPU toy)
# --------------------------------------------------------------------------- #

def _meta(**kw):
    m = {"model": "toy", "model_revision": None, "precision": "fp32", "task": "gsm8k",
         "task_jsonl_sha256": None, "manifest_sha256": None, "half": "all", "gen_length": 16,
         "block_length": 8, "decoding": "greedy", "prompt": {"format": "chat", "fewshot": 0},
         "configs_sha256": "g", "calibrator_sha256": None, "c3a_clamp": False,
         "pc_sampler_variant": "position-only", "shard": 0, "num_shards": 1,
         "provenance": {"git_commit": "abc", "git_dirty": False}}
    m.update(kw)
    return m


def _read(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


@pytest.mark.parametrize("k", [1, 2, 4])
def test_generation_fills_the_region_block_by_block(k):
    pytest.importorskip("torch")
    from c3.decode.generate import generate
    from c3.traces.models import ToyDiffusionLM

    rng = np.random.default_rng(0)
    prompt = rng.integers(10, 60, 8)
    model = ToyDiffusionLM(noise_dispersion=0.3)
    model.set_reference(np.concatenate([prompt, rng.integers(10, 60, 32)]))
    g = generate(model, prompt, TopKConfidence(k), stops={999}, gen_length=32, block_length=8)
    assert g["n_committed"] == 32 and g["n_forward"] == 32 // k
    assert g["tokens_per_step"] == pytest.approx(k)
    assert len(g["generated_ids"]) == 32 and g["stop_index"] is None
    assert g["completion_ids"] == g["generated_ids"] and not g["stopped"]


def test_the_completion_stops_at_the_first_stop_token_and_keeps_every_id():
    torch = pytest.importorskip("torch")
    from c3.decode.generate import generate
    from c3.decode.policies import Threshold

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

    rng = np.random.default_rng(1)
    prompt = rng.integers(10, 60, 8)
    ref = np.concatenate([prompt, rng.integers(10, 60, 12), [1], rng.integers(10, 60, 19)])
    g = generate(Exact(ref), prompt, Threshold(0.5), stops={1}, gen_length=32, block_length=8)
    assert g["stopped"] and g["stop_index"] == 12
    assert g["completion_ids"] == ref[8:20].tolist() and g["generated_ids"] == ref[8:].tolist()
    assert g["n_forward"] == 4                              # every slot of a block clears 0.5


def _sweep(tmp_path, model, configs, meta=None, calibrator=None, **kw):
    tok = ChatTok()
    recs = _recs(3)
    prompts = {r["doc_id"]: task_prompt("gsm8k", r, tok) for r in recs}
    return run_t12.run_sweep(model, tok, recs, prompts, configs, tmp_path / "out", "t12.shard0of1",
                             meta or _meta(), calibrator=calibrator, **kw)


def test_sweep_is_problem_major_resumes_and_refuses_a_different_run(tmp_path):
    pytest.importorskip("torch")
    from c3.traces.models import ToyDiffusionLM

    model = ToyDiffusionLM(noise_dispersion=0.3)
    cfgs = [{"policy": "topk_conf", "k": 4}, {"policy": "threshold", "tau": 0.5}]
    path = _sweep(tmp_path, model, cfgs)
    recs = _read(path)
    assert [r["doc_id"] for r in recs] == [0, 0, 1, 1, 2, 2]      # problems outermost
    meta = json.loads((path.parent / "t12.shard0of1.meta.json").read_text())
    assert all(r["run_id"] == meta["run_id"] and r["precision"] == "fp32"
               and r["git_commit"] == "abc" and len(r["generated_ids"]) == 16 for r in recs)
    # a killed write, then a restart: nothing redone, nothing lost, the meta notes the start
    with path.open("a") as fh:
        fh.write('{"config": "topk_conf[k=4]", "doc_')
    _sweep(tmp_path, model, cfgs)
    assert len(_read(path)) == 6
    meta = json.loads((path.parent / "t12.shard0of1.meta.json").read_text())
    assert len(meta["starts"]) == 2
    # a different run into the same directory is refused
    with pytest.raises(SystemExit, match="precision"):
        _sweep(tmp_path, model, cfgs, _meta(precision="fp16"))
    other = _meta(provenance={"git_commit": "def", "git_dirty": False})
    with pytest.raises(SystemExit, match="git_commit"):
        _sweep(tmp_path, model, cfgs, other)
    _sweep(tmp_path, model, cfgs, other, allow_mixed_commits=True)   # allowed when asked


def test_a_failing_problem_is_recorded_and_the_sweep_goes_on(tmp_path):
    pytest.importorskip("torch")
    from c3.traces.models import ToyDiffusionLM

    class Overflows(ToyDiffusionLM):
        def logits(self, input_ids):
            if input_ids.shape[1] == len(task_prompt("gsm8k", _recs()[1], ChatTok())) + 16:
                raise FloatingPointError("toy (fp16): 3 non-finite logits")
            return super().logits(input_ids)

    path = _sweep(tmp_path, Overflows(), [{"policy": "topk_conf", "k": 4}])
    recs = _read(path)
    assert [r["error"] is None for r in recs] == [True, False, True]
    assert recs[1]["completion"] is None and "non-finite" in recs[1]["error"]


def test_c3a_runs_through_the_sweep_and_logs_every_step(tmp_path):
    pytest.importorskip("torch")
    from c3.traces.models import ToyDiffusionLM
    from test_c3a_policy import _calibrator

    path = _sweep(tmp_path, ToyDiffusionLM(noise_dispersion=0.3), [{"policy": "c3a", "alpha": 0.3}],
                  calibrator=_calibrator())
    for r in _read(path):
        steps = r["c3a_steps"]
        assert len(steps) == r["n_forward"] and sum(s["n_commit"] for s in steps) == 16
        assert steps[0]["mask_rate"] == 1.0 and steps[0]["log_n_masked"] == pytest.approx(np.log(16))
