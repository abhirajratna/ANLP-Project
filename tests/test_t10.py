"""T10: the parallel-sequential disagreement proxy, the (B) proxies and the probe calibration.

What has to hold for the proxy to mean anything: rank 1 always agrees (k = 1 is
0 by construction), a model whose positions ignore each other never disagrees, a
model with dependence does, the chains for k = 2, 4, 8 nest, and on the probe
the states are exactly the grid's.  All on CPU with toy models.
"""
import json

import numpy as np
import pytest

from c3.disagreement import (load_fit_documents, spearman_draws, summarise_probe,
                             summarise_text)


def _toy(seed=0):
    pytest.importorskip("torch")
    from c3.traces.models import ToyDiffusionLM
    return ToyDiffusionLM(seed=seed)


def _run_one(model, doc_id, ids, **kw):
    """run_text on one document (the toy model knows one reference at a time)."""
    from c3.disagreement import run_text
    if hasattr(model, "set_reference"):
        model.set_reference(ids)
    return run_text(model, [(doc_id, ids)], log=lambda s: None, **kw)


def _concat(parts):
    return {c: np.concatenate([p[c] for p in parts]) for c in parts[0]}


def _toy_cols(model, n_docs=4, length=48, **kw):
    rng = np.random.default_rng(1)
    return _concat([_run_one(model, d, rng.integers(0, 60, length), **kw) for d in range(n_docs)])


def _per_k(summary, metric):
    return {r["k"]: r[metric]["mean"] for r in summary["per_k"] if metric in r}


def test_k1_is_zero_and_rank1_always_agrees():
    cols = _toy_cols(_toy())
    rank1 = cols["rank"] == 1
    assert np.array_equal(cols["parallel"][rank1], cols["sequential"][rank1])
    s = summarise_text(cols, n_boot=200)
    assert s["rank1_always_agrees"]
    assert _per_k(s, "disagreement")[1] == 0.0
    assert s["n_states"] == 4 * 10 and s["n_docs"] == 4
    below = {r["k"]: r for r in s["per_k_rates_below_1"]}
    sub = (cols["target_rate"] < 1.0) & (cols["rank"] <= 8)
    assert below[8]["conf"]["mean"] == pytest.approx(np.mean(
        [cols["conf"][sub & (cols["doc_id"] == d)].astype(np.float64).mean() for d in range(4)]),
        abs=1e-12)


def _independent(seed=0):
    """Each position's logits depend on its index alone, never on another position."""
    pytest.importorskip("torch")
    from c3.traces.models import ToyDiffusionLM

    class Independent(ToyDiffusionLM):
        def logits(self, input_ids):
            import torch
            B, L = input_ids.shape
            table = np.random.default_rng(self.seed).normal(0, 2.0, (L, self.vocab_size))
            return torch.as_tensor(np.repeat(table[None], B, axis=0), dtype=torch.float32)

    return Independent(seed=seed)


def test_positions_that_ignore_each_other_never_disagree():
    cols = _toy_cols(_independent())
    assert cols["rank"].max() == 8
    assert np.array_equal(cols["parallel"], cols["sequential"])
    s = summarise_text(cols, n_boot=200)
    for k in (1, 2, 4, 8):
        assert _per_k(s, "disagreement")[k] == 0.0


def _copy_toy():
    """Pairs (2j, 2j+1): the leader prefers token 1, the follower alone prefers token 2 but
    copies its leader whenever the leader holds a token.  Confidence falls along the
    sequence, so the ranking is L0, F0, L1, F1, ...; at a fully masked state the parallel
    commit is (1, 2, 1, 2, ...) and the sequential one (1, 1, 1, 1, ...)."""
    pytest.importorskip("torch")
    from c3.traces.models import DiffusionLM

    class CopyToy(DiffusionLM):
        name, vocab_size, mask_token_id, precision = "copy-toy", 8, 7, "fp32"
        model = None

        def logits(self, input_ids):
            import torch
            ids = input_ids.detach().cpu().numpy()
            B, L = ids.shape
            out = np.zeros((B, L, self.vocab_size))
            for b in range(B):
                for p in range(L):
                    s = 4.0 - 0.3 * (p // 2)
                    if p % 2 == 0:
                        out[b, p, 1], out[b, p, 2] = s, s - 0.5
                    elif ids[b, p - 1] != self.mask_token_id:
                        out[b, p, ids[b, p - 1]] = s + 5.0
                    else:
                        out[b, p, 2], out[b, p, 1] = s - 0.05, s - 0.55
            return torch.as_tensor(out, dtype=torch.float32)

    return CopyToy()


def test_a_model_with_dependence_disagrees():
    # The CPU toy's noise is seeded by the whole row, so filling one position moves
    # every other position's logits: dependence of a crude kind, but dependence.
    s = summarise_text(_toy_cols(_toy()), n_boot=200)
    assert _per_k(s, "disagreement")[8] > 0
    # The copy toy: exactly half of every committed set disagrees at the fully masked state.
    # The true tokens (3) are neither choice, so a chain that filled in the truth instead of
    # its own tokens would copy 3 and fail here.
    cols = _run_one(_copy_toy(), 0, np.full(16, 3))
    full = cols["rate_idx"] == 9
    assert np.all(cols["n_masked"][full] == 16)
    assert np.array_equal(cols["position"][full], np.arange(8))
    assert np.array_equal(cols["parallel"][full], np.tile([1, 2], 4))
    assert np.array_equal(cols["sequential"][full], np.ones(8, dtype=int))
    rates = summarise_text(cols, n_boot=50)["per_k_rate"]
    at_full = {r["k"]: r["disagreement"]["mean"] for r in rates if r["rate"] == 1.0}
    assert at_full == {1: 0.0, 2: 0.5, 4: 0.5, 8: 0.5}


def test_the_chains_for_k_2_4_8_nest():
    model = _toy(seed=3)
    rng = np.random.default_rng(5)
    docs = [rng.integers(0, 60, 40) for _ in range(3)]
    by_depth = {d: _concat([_run_one(model, i, x, depth=d, batch_size=3)
                            for i, x in enumerate(docs)]) for d in (2, 4, 8)}
    deep = by_depth[8]
    assert int(deep["sequential"].size) == 3 * 10 * 8
    for d in (2, 4):
        shallow, keep = by_depth[d], deep["rank"] <= d
        for c in ("doc_id", "rate_idx", "rank", "position", "parallel", "sequential", "conf"):
            assert np.array_equal(shallow[c], deep[c][keep]), (d, c)
    # the k = 2 numbers from a chain of 2 are the k = 2 numbers from a chain of 8
    two = summarise_text(by_depth[2], ks=(2,), b_ks=(1, 2), n_boot=100)
    eight = summarise_text(deep, n_boot=100)
    assert _per_k(two, "disagreement")[2] == _per_k(eight, "disagreement")[2]
    assert _per_k(two, "delta")[2] == _per_k(eight, "delta")[2]
    # and the sets nest: a disagreement among the top 2 is one among the top 4 and 8
    flag = deep["parallel"] != deep["sequential"]
    state = deep["doc_id"] * 100 + deep["rate_idx"]
    for st in np.unique(state):
        f = flag[state == st]
        assert f[:2].sum() <= f[:4].sum() <= f[:8].sum()


def test_delta_and_ce_follow_their_definitions():
    cols = _toy_cols(_toy(seed=2), n_docs=2)
    s = summarise_text(cols, n_boot=50)
    k2 = cols["rank"] <= 2
    by_doc = []
    for d in np.unique(cols["doc_id"]):
        m = k2 & (cols["doc_id"] == d)
        by_doc.append((np.mean(cols["conf"][m] - (cols["parallel"][m] == cols["true"][m])),
                       np.mean(-cols["logp_true"][m])))
    r2 = next(r for r in s["per_k"] if r["k"] == 2)
    assert r2["delta"]["mean"] == pytest.approx(np.mean([x for x, _ in by_doc]), rel=1e-6)
    assert r2["ce"]["mean"] == pytest.approx(np.mean([y for _, y in by_doc]), rel=1e-6)
    assert np.all(np.isfinite(cols["logp_true"])) and np.all(cols["logp_true"] <= 0)


# ------------------------------------------------------------------ documents

class _WordTokenizer:
    def __call__(self, text, truncation=True, max_length=None, add_special_tokens=False):
        ids = [len(w) for w in text.split()]
        return {"input_ids": ids[:max_length] if truncation else ids}


def test_fit_documents_in_order_and_the_report_half_is_never_parsed(tmp_path):
    corpus, manifest = tmp_path / "docs.jsonl", tmp_path / "split.json"
    split = {i: (1 if i % 3 == 0 else 0) for i in range(12)}
    with corpus.open("w") as fh:
        for i in [5, 0, 7, 2, 9, 1, 11, 4, 3, 8, 10, 6]:          # not in doc_id order
            if split[i] == 1:     # invalid JSON: parsing a report-half line would raise
                fh.write('{"doc_id": %d, "text": "REPORT HALF, NEVER PARSED\n' % i)
            else:
                n = 3 if i == 4 else 20                            # doc 4 is short
                fh.write(json.dumps({"doc_id": i, "id": f"x{i}", "text": "w " * n}) + "\n")
    manifest.write_text(json.dumps({str(k): v for k, v in split.items()}))
    docs, info = load_fit_documents(corpus, manifest, _WordTokenizer(), n_docs=4, max_tokens=10)
    assert [d for d, _ in docs] == [1, 2, 5, 7]
    assert info["skipped_short"] == [4] and info["missing"] == []
    assert all(x.size == 10 for _, x in docs)
    with pytest.raises(ValueError, match="only 7"):
        load_fit_documents(corpus, manifest, _WordTokenizer(), n_docs=9, max_tokens=10)


# ------------------------------------------------------------------ probe

def test_spearman_draws_matches_scipy_and_handles_ties():
    from scipy.stats import spearmanr
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(50, 15)), rng.normal(size=(50, 15))
    y[:, :3] = 0.0                                  # ties, as the rho = 0 cells have
    got = spearman_draws(x, y)
    want = [spearmanr(a, b)[0] for a, b in zip(x, y)]
    assert np.allclose(got, want)
    assert spearman_draws(np.arange(5.0), np.arange(5.0) ** 3)[0] == pytest.approx(1.0)
    assert np.isnan(spearman_draws(np.arange(5.0), np.zeros(5))[0])


# disagreements per instance in each (rho, k) cell, out of 8 fields: rising with (A)
_TARGET = {(0.5, 2): 1, (1.0, 2): 2, (0.5, 4): 3, (1.0, 4): 4}


def _probe_rows(rhos=(0.0, 0.5, 1.0), ks=(1, 2, 4), n=6):
    """Rows as probe_disagreement writes them, 8 fields per instance; (A) per field is
    rho (k - 1) / 80 plus a little noise, and exactly 0 (up to float residue) at rho = 0."""
    rows, rng = [], np.random.default_rng(0)
    for r, rho in enumerate(rhos):
        for i in range(n):
            for k in ks:
                left, step, todo = 8, 0, _TARGET.get((rho, k), 0)
                while left:
                    c = min(k, left)
                    nd = min(c - 1, todo)
                    A = 1e-17 if rho == 0 else rho * (k - 1) / 80 * c + rng.uniform(0, 1e-4)
                    rows.append({"instance_id": 100 * r + i, "rho": rho, "k": k, "step": step,
                                 "n_fields": 8, "n_commit": c, "A": A, "n_disagree": nd,
                                 "n_agree_true_parallel": c, "n_agree_true_sequential": c - nd})
                    left, step, todo = left - c, step + 1, todo - nd
    return rows


def test_summarise_probe_pools_per_instance_and_ranks_cells():
    grid = {"cells": [{"rule": rule, "k": k, "rho": rho,
                       "A": {"mean": -1e-18 if rho == 0 else 8 * rho * (k - 1) / 80},
                       "A_per_field": {"mean": -1e-18 if rho == 0 else rho * (k - 1) / 80},
                       "B_per_field": {"mean": 0.1}}
                      for rule in ("random", "confidence") for rho in (0.0, 0.5, 1.0)
                      for k in (1, 2, 4)]}
    out = summarise_probe(_probe_rows(), grid, n_boot=300, seed=0)
    cells = {(c["rho"], c["k"]): c for c in out["cells"]}
    assert cells[(0.0, 2)]["A_per_field_grid"] == 0.0          # the float residue is zero
    assert cells[(0.0, 2)]["A_per_field_recomputed"] == 0.0
    assert cells[(1.0, 4)]["A_per_field_grid"] == pytest.approx(3 / 80)
    assert cells[(1.0, 4)]["disagreement"]["mean"] == pytest.approx(4 / 8)   # pooled over steps
    assert cells[(0.5, 2)]["disagreement"]["mean"] == pytest.approx(1 / 8)
    assert cells[(1.0, 1)]["disagreement"]["mean"] == 0.0
    assert cells[(1.0, 4)]["acc_sequential"] == pytest.approx(4 / 8)
    sp = out["spearman"]
    assert sp["n_cells"] == 6                                    # the k = 1 cells are left out
    assert sp["rho_s"] == pytest.approx(1.0)
    assert sp["ci95"] == pytest.approx([1.0, 1.0])
    assert out["max_abs_cell_A_recomputed_minus_grid"] < 1e-4


@pytest.fixture(scope="module")
def gpt2():
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained("gpt2")
    except Exception as exc:                       # offline without a cache
        pytest.skip(f"gpt2 tokenizer unavailable: {exc}")


def test_probe_states_are_the_grids_states(gpt2):
    """The proxy is computed on exactly the states run_probe.py decodes."""
    torch = pytest.importorskip("torch")
    import pandas as pd

    from c3.disagreement import align_with_grid_steps, probe_disagreement
    from c3.probe.grid import prepare, run_trajectories
    from c3.probe.instances import ProbeSpec, eval_instances
    from c3.traces.models import LocalMDLM
    from c3.training.model import MaskedDiffusionLM, ModelConfig
    from pathlib import Path

    torch.manual_seed(0)
    model = LocalMDLM(path="", device="cpu")
    model.model = MaskedDiffusionLM(ModelConfig(d_model=32, n_layers=2, n_heads=2, d_ff=64)).eval()
    spec = ProbeSpec.load(Path(__file__).resolve().parents[1] / "artifacts/probe/spec.json")
    prepared = prepare(spec, eval_instances(spec, per_rho=4), gpt2, "gpt2")
    for k in (2, 4):
        grid_rows = run_trajectories(model, spec, prepared, k, "confidence", seed=0, batch_size=8)
        mine = probe_disagreement(model, spec, prepared, k, batch_size=8, log=lambda s: None)
        al = align_with_grid_steps(mine, pd.DataFrame(grid_rows))
        assert al["steps_here"] == al["steps_grid"] == len(grid_rows)
        assert al["steps_same_state"] == al["steps_bitwise"] == len(grid_rows)
        assert all(0 <= r["n_disagree"] <= r["n_commit"] - 1 for r in mine)
