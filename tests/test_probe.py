"""T9 JSON probe: generator, single-token values, exact p*, and (A) + (B) = KL.

The plan's unit test (T9 "Done when"): the decomposition matches the directly
computed KL to 1e-6 nats, on toy distributions and on real 110M outputs.  The
real-model check needs a checkpoint: set C3_PROBE_CKPT to a final.pt.
"""
import hashlib
import itertools
import json
import os
from pathlib import Path

import numpy as np
import pytest

from c3.probe import MAX_TOKENS, N_TRAIN, RHOS
from c3.probe.instances import (Field, Group, ProbeSpec, Schema, eval_instances, sample_values,
                                serialize, train_instances)
from c3.probe.truth import (MASKED, decompose, direct_kl, entropy, group_joint, joint_over,
                            marginals)

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "probe"
TOL = 1e-6


def toy_schema(rng, n_fields=5, nested=True) -> Schema:
    fields = []
    for i in range(n_fields):
        kind = ["str", "num", "bool", "enum"][i % 4]
        n = 2 if kind == "bool" else int(rng.integers(2, 5))
        values = (["true", "false"] if kind == "bool" else
                  [str(j) for j in range(n)] if kind == "num" else [f"v{i}x{j}" for j in range(n)])
        path = ("obj", f"f{i}") if nested and i in (1, 2) else (f"f{i}",)
        fields.append(Field(path, kind, values, rng.dirichlet(np.ones(n))))
    order = rng.permutation(n_fields)
    groups = []
    for j in range(0, n_fields - 1, 2):
        a, b = int(order[j]), int(order[j + 1])
        perm = rng.permutation(fields[b].n)
        groups.append(Group((a, b), perm[np.arange(fields[a].n) % fields[b].n]))
    if n_fields % 2:
        groups.append(Group((int(order[-1]),)))
    return Schema("toy", fields, groups)


def toy_spec(seed=0, n_schemas=4) -> ProbeSpec:
    rng = np.random.default_rng(seed)
    return ProbeSpec([toy_schema(rng) for _ in range(n_schemas)], RHOS, seed)


def brute_joint(schema: Schema, rho: float) -> np.ndarray:
    """p*(all fields) straight from the generative process, by enumeration."""
    shape = [f.n for f in schema.fields]
    full = np.zeros(shape)
    for vals in itertools.product(*[range(n) for n in shape]):
        p = 1.0
        for g in schema.groups:
            if len(g.members) == 1:
                p *= schema.fields[g.members[0]].p[vals[g.members[0]]]
            else:
                a, b = g.members
                pa, pb = schema.fields[a].p, schema.fields[b].p
                copy = rho if g.fmap[vals[a]] == vals[b] else 0.0
                p *= pa[vals[a]] * (copy + (1 - rho) * pb[vals[b]])
        full[vals] = p
    return full


def brute_conditional(schema, rho, obs, S) -> np.ndarray:
    full = brute_joint(schema, rho)
    idx = tuple(slice(None) if v == MASKED else int(v) for v in obs)
    cond = full[idx]                                   # axes: masked fields in order
    masked = [i for i, v in enumerate(obs) if v == MASKED]
    drop = tuple(a for a, i in enumerate(masked) if i not in S)
    cond = cond.sum(axis=drop) if drop else cond
    kept = [i for i in masked if i in S]
    cond = np.transpose(cond, [kept.index(i) for i in S])
    return cond / cond.sum()


# ------------------------------------------------------------------ generator

def test_generator_is_deterministic():
    spec = toy_spec()
    a, b = eval_instances(spec, per_rho=8), eval_instances(spec, per_rho=8)
    assert [x.to_dict() for x in a] == [x.to_dict() for x in b]
    t1 = [x.to_dict() for x in train_instances(spec, 50)]
    t2 = [x.to_dict() for x in train_instances(spec, 50)]
    assert t1 == t2
    assert [x.to_dict() for x in train_instances(spec, 10, start=40)] == t1[40:]
    other = [x.to_dict() for x in train_instances(spec, 50, seed=spec.seed + 1)]
    assert other != t1
    assert [x.to_dict() for x in train_instances(spec, 50, label="probe-val")] != t1


def test_serialisation_is_json_and_spans_hold_the_values():
    spec = toy_spec()
    for inst in eval_instances(spec, per_rho=4):
        sch = spec.schemas[inst.schema]
        text, spans = serialize(sch, inst.values, inst.rho)
        doc = json.loads(text)
        assert doc["_rho"] == f"{inst.rho:.2f}"
        for f, v, (s, e) in zip(sch.fields, inst.values, spans):
            assert text[s:e] == f.values[v]
            node = doc
            for key in f.path:
                node = node[key]
            assert str(node).lower() == f.values[v].lower()


def test_committed_eval_and_training_sets_regenerate_exactly():
    spec = ProbeSpec.load(ART / "spec.json")
    manifest = json.loads((ART / "manifest.json").read_text())
    with open(ART / "eval_instances.jsonl") as fh:
        stored = [json.loads(line) for line in fh]
    ev = eval_instances(spec, per_rho=manifest["eval"]["per_rho"])
    assert len(ev) == len(stored) == 1000
    for inst, rec in zip(ev, stored):
        assert inst.to_dict() == {k: rec[k] for k in ("instance_id", "schema", "rho", "values")}
        assert serialize(spec.schemas[inst.schema], inst.values, inst.rho)[0] == rec["text"]
    assert sorted({r["rho"] for r in stored}) == list(RHOS)
    h = hashlib.sha256()
    for inst in train_instances(spec, N_TRAIN):
        h.update(serialize(spec.schemas[inst.schema], inst.values, inst.rho)[0].encode() + b"\n")
    assert h.hexdigest() == manifest["train"]["text_sha256"]


def test_generator_matches_the_exact_distribution():
    rng = np.random.default_rng(3)
    sch = toy_schema(rng, n_fields=4, nested=False)
    for rho in (0.0, 0.5, 1.0):
        draws = np.array([sample_values(sch, rho, np.random.default_rng([7, i]))
                          for i in range(40_000)])
        for g in sch.groups:
            exact = group_joint(sch, g, rho)
            if len(g.members) == 1:
                emp = np.bincount(draws[:, g.members[0]], minlength=exact.size) / len(draws)
            else:
                a, b = g.members
                emp = np.zeros_like(exact)
                np.add.at(emp, (draws[:, a], draws[:, b]), 1.0 / len(draws))
            assert np.abs(emp - exact).max() < 0.012


# ------------------------------------------------------------------ single-token values

TOKENIZER_NAMES = ("gpt2", "llada-8b-base", "dream-7b-base")


@pytest.fixture(scope="module")
def tokenizers():
    pytest.importorskip("transformers")
    from scripts.build_probe import load_tokenizers
    try:
        return load_tokenizers(TOKENIZER_NAMES)
    except Exception as exc:                       # offline without a cache
        pytest.skip(f"tokenizers unavailable: {exc}")


def test_every_value_is_one_token_in_context_under_all_three_tokenizers(tokenizers):
    from c3.probe.grid import prepare
    spec = ProbeSpec.load(ART / "spec.json")
    ev = eval_instances(spec, per_rho=200)
    for name, tok in tokenizers.items():
        prepared = prepare(spec, ev, tok, name)    # raises unless one token, the spec's id
        assert max(p.ids.size for p in prepared) <= MAX_TOKENS
        for p in prepared:
            assert p.pos.size == len(p.inst.values)
    for sch in spec.schemas:
        for f in sch.fields:
            for name in TOKENIZER_NAMES:
                assert len(set(f.tokens[name])) == f.n      # distinct values, distinct tokens


def test_offsets_from_decoding_agree_with_fast_offsets(tokenizers):
    from c3.probe.tokens import token_offsets
    text = '{"_rho":"0.25","a":"debug","b":{"c":true,"d":7},"e":"alpha"}'
    for name in ("gpt2", "llada-8b-base"):
        tok = tokenizers[name]
        fast = token_offsets(tok, text)

        class Slow:
            is_fast = False
            def __call__(self, t, add_special_tokens=False):
                return tok(t, add_special_tokens=add_special_tokens)
            def decode(self, ids, clean_up_tokenization_spaces=False):
                return tok.decode(ids, clean_up_tokenization_spaces=False)
        assert token_offsets(Slow(), text) == fast


# ------------------------------------------------------------------ enumeration

def test_group_tables_match_brute_force_conditionals():
    rng = np.random.default_rng(11)
    for trial in range(40):
        sch = toy_schema(rng, n_fields=int(rng.integers(3, 6)), nested=False)
        rho = float(rng.choice(RHOS))
        vals = sample_values(sch, rho, rng)
        obs = np.where(rng.random(len(vals)) < 0.4, vals, MASKED)
        masked = [i for i in range(len(vals)) if obs[i] == MASKED]
        if not masked:
            continue
        S = sorted(int(i) for i in rng.choice(masked, size=int(rng.integers(1, len(masked) + 1)),
                                              replace=False))
        assert np.allclose(joint_over(sch, rho, obs, S), brute_conditional(sch, rho, obs, S),
                           atol=1e-13)


def test_marginals_and_entropies_of_a_pair():
    rng = np.random.default_rng(5)
    sch = toy_schema(rng, n_fields=2, nested=False)
    (g,) = sch.groups
    a, b = g.members
    obs = np.full(2, MASKED)
    for rho in RHOS:
        m = marginals(sch, rho, obs, [a, b])
        assert np.allclose(m[a], sch.fields[a].p)
        push = np.bincount(g.fmap, weights=sch.fields[a].p, minlength=sch.fields[b].n)
        assert np.allclose(m[b], rho * push + (1 - rho) * sch.fields[b].p)
    # rho = 1 with a bijective map: the follower adds no entropy, so TC = H(leader).
    pa = rng.dirichlet(np.ones(4))
    bij = Schema("bij", [Field(("a",), "str", list("wxyz"), pa),
                         Field(("b",), "str", list("WXYZ"), np.full(4, 0.25))],
                 [Group((0, 1), np.array([3, 1, 0, 2]))])
    logq = {0: np.log(np.full(4, 0.25)), 1: np.log(np.full(4, 0.25))}
    d = decompose(bij, 1.0, np.full(2, MASKED), [0, 1], logq)
    assert abs(d["A"] - entropy(pa)) < 1e-12


def test_term_A_is_zero_at_rho_zero_and_for_single_commits():
    rng = np.random.default_rng(2)
    for _ in range(30):
        sch = toy_schema(rng)
        obs = np.full(len(sch.fields), MASKED)
        logq = {i: np.log(rng.dirichlet(np.ones(f.n))) for i, f in enumerate(sch.fields)}
        everything = list(range(len(sch.fields)))
        assert abs(decompose(sch, 0.0, obs, everything, logq)["A"]) < 1e-12
        for rho in RHOS:
            for i in everything:
                assert abs(decompose(sch, rho, obs, [i], logq)["A"]) < 1e-12
        assert decompose(sch, 1.0, obs, everything, logq)["A"] > 0


# ------------------------------------------------------------------ the identity

def test_decomposition_equals_direct_kl_on_toy_distributions():
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(300):
        sch = toy_schema(rng, n_fields=int(rng.integers(2, 7)), nested=False)
        rho = float(rng.uniform(0, 1))
        vals = sample_values(sch, rho, rng)
        obs = np.where(rng.random(len(vals)) < 0.3, vals, MASKED)
        masked = [i for i in range(len(vals)) if obs[i] == MASKED]
        if not masked:
            continue
        S = [int(i) for i in rng.choice(masked, size=int(rng.integers(1, len(masked) + 1)),
                                        replace=False)]
        # q leaves mass outside the field's values, as a model over a whole vocabulary does
        logq = {i: np.log(rng.dirichlet(np.ones(f.n)) * rng.uniform(0.05, 1.0))
                for i, f in enumerate(sch.fields)}
        d = decompose(sch, rho, obs, S, logq)
        worst = max(worst, abs(d["A"] + d["B"] - direct_kl(sch, rho, obs, S, logq)))
    assert worst < TOL


def _tiny_110m(seed=0):
    torch = pytest.importorskip("torch")
    from c3.traces.models import LocalMDLM
    from c3.training.model import MaskedDiffusionLM, ModelConfig
    torch.manual_seed(seed)
    m = LocalMDLM(path="", device="cpu")
    m.model = MaskedDiffusionLM(ModelConfig(d_model=32, n_layers=2, n_heads=2, d_ff=64)).eval()
    return m


def _identity_on(model, tokenizers, n=25):
    from c3.probe.grid import prepare, run_trajectories, summarise
    spec = ProbeSpec.load(ART / "spec.json")
    ev = eval_instances(spec, per_rho=200)
    pick = [ev[i] for i in np.random.default_rng(0).choice(len(ev), size=n, replace=False)]
    prepared = prepare(spec, pick, tokenizers["gpt2"], "gpt2")
    rows = []
    for rule in ("random", "confidence"):
        for k in (1, 2, 4):
            rows += run_trajectories(model, spec, prepared, k, rule, seed=0, batch_size=8)
    ident = summarise(rows, n_boot=200)["identity"]
    assert ident["steps_checked"] > 0.5 * ident["steps_total"]
    assert ident["max_abs_err_nats"] < TOL
    return rows


def test_decomposition_equals_direct_kl_on_a_tiny_mdlm(tokenizers):
    rows = _identity_on(_tiny_110m(), tokenizers)
    assert all(r["A"] >= -1e-12 and r["B"] >= -1e-12 for r in rows)


@pytest.mark.skipif(not os.environ.get("C3_PROBE_CKPT"), reason="set C3_PROBE_CKPT to a 110M final.pt")
def test_decomposition_equals_direct_kl_on_real_110m_outputs(tokenizers):
    pytest.importorskip("torch")
    from c3.traces.models import build_model
    import torch
    model = build_model("local-110m", path=os.environ["C3_PROBE_CKPT"],
                        device="cuda" if torch.cuda.is_available() else "cpu")
    _identity_on(model, tokenizers, n=40)
