"""Schema-v2 trace pipeline (plan v2 T1): masks, schema, the T1 step-5 unit tests,
and shards/resume/merge, on the CPU toy model."""
import json
from dataclasses import dataclass, field

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from c3.assemble import SCALAR_COLUMNS, arm_labels, load_traces
from c3.endpoints.e1 import native_gap
from c3.frame import PositionFrame
from c3.stats import doc_boot_counts
from c3.traces.collect import (PRESETS, TRACE_COLUMNS, CollectConfig, Document, collect,
                               collect_shard, shard_documents)
from c3.traces.masks import (UNIFORM_RATES, TeacherForcedTrajectory, matched_uniform_masks,
                             run_trajectory, uniform_masks)
from c3.traces.models import ToyDiffusionLM, build_model, check_finite_logits
from c3.traces.precision import compare_precisions
from c3.traces.shards import MergeError, merge

PILOT = "artifacts/pilot/llada-8b-4bit-n20.parquet"


@dataclass
class SpyToy(ToyDiffusionLM):
    """The toy, recording every input it is shown."""

    seen: list = field(default_factory=list)

    def logits(self, input_ids):
        self.seen.extend(input_ids.detach().cpu().numpy().copy())
        return super().logits(input_ids)


def _docs(n=3, length=40, seed=0):
    rng = np.random.default_rng(seed)
    return [(i, rng.integers(0, 60, length)) for i in range(n)]


def _small_cfg(**kw):
    base = dict(mask_rates=(0.3, 0.7), masks_per_rate=2, decoder_k=(4, 8))
    base.update(kw)
    return CollectConfig(**base)


@pytest.fixture(scope="module")
def spy_trace(tmp_path_factory):
    model = SpyToy(noise_dispersion=0.3)
    docs = _docs()
    out = collect(model, docs, _small_cfg(), tmp_path_factory.mktemp("t") / "t.parquet",
                  progress_every=0)
    return model, dict(docs), load_traces(out), out


# --------------------------------------------------------------------------- #
# The three T1 step-5 tests
# --------------------------------------------------------------------------- #

def test_masked_count_equals_the_mask(spy_trace):
    model, docs, df, _ = spy_trace
    per_state = df.groupby("state_id").agg(n=("position", "size"), n_masked=("n_masked", "first"))
    assert (per_state.n == per_state.n_masked).all()
    # ... and equals the number of mask tokens in the input the model was shown.
    seen = {frozenset(np.flatnonzero(x == model.mask_token_id).tolist()) for x in model.seen}
    for _, rows in df.groupby("state_id"):
        assert frozenset(rows.position.tolist()) in seen


def test_teacher_forcing_shows_no_true_token_before_its_commit(spy_trace):
    model, docs, df, _ = spy_trace
    inputs = {frozenset(np.flatnonzero(x == model.mask_token_id).tolist()): x for x in model.seen}
    dec = df[df.mask_kind == "decoder"]
    for (doc, k), traj in dec.groupby(["doc_id", "k"]):
        x0 = docs[doc]
        committed_before: set = set()
        for step, rows in traj.groupby("step"):
            masked = set(rows.position.tolist())
            # Visible = exactly what was committed at earlier steps.
            assert set(range(x0.size)) - masked == committed_before
            x = inputs[frozenset(masked)]
            vis = np.array(sorted(committed_before), dtype=int)
            assert np.array_equal(x[vis], x0[vis])          # visible slots hold the truth
            assert np.all(x[list(masked)] == model.mask_token_id)
            committed_before |= set(rows.position[rows.committed].tolist())
        assert committed_before == set(range(x0.size))


@pytest.mark.parametrize("precision,tol", [("fp16", 0.02), ("bf16", 0.15)])
def test_half_precision_spot_check_against_fp32(tmp_path, precision, tol):
    docs = _docs(n=2, length=48)
    cfg = _small_cfg(decoder_k=(4,))
    ref = load_traces(collect(ToyDiffusionLM(noise_dispersion=0.3), docs, cfg,
                              tmp_path / "fp32.parquet", progress_every=0))
    alt = load_traces(collect(ToyDiffusionLM(noise_dispersion=0.3, precision=precision), docs,
                              cfg, tmp_path / f"{precision}.parquet", progress_every=0))
    assert set(alt.precision) == {precision}
    cmp = compare_precisions(ref, alt)
    n_uniform = ref[ref.mask_kind == "uniform"].state_id.nunique()
    assert cmp["n_states_matched"] >= n_uniform          # uniform masks always coincide
    assert 0 < cmp["max_abs_delta_ell"] < tol
    assert cmp["top1_agreement"] > 0.98


def test_precision_comparison_finds_where_a_trajectory_diverges(spy_trace):
    _, _, df, _ = spy_trace
    alt = df.copy()
    # Pretend the other precision committed differently from step 3 on (doc 0, k=4).
    late = (alt.mask_kind == "decoder") & (alt.doc_id == 0) & (alt.k == 4) & (alt.step >= 3)
    alt.loc[late, "position"] = alt.loc[late, "position"] + 1000
    cmp = compare_precisions(df, alt)
    assert cmp["trajectories_diverged"] == 1 and cmp["first_divergent_step"] == 3
    assert cmp["max_abs_delta_ell"] == 0.0


# --------------------------------------------------------------------------- #
# Schema and masks
# --------------------------------------------------------------------------- #

def test_schema_v2_columns_and_meta(spy_trace):
    _, _, df, out = spy_trace
    assert list(df.columns) == list(TRACE_COLUMNS)
    assert all(len(v) == 20 for v in df.topk_ids.head(50))
    # p_true is the top-1 probability exactly when the top-1 is right.
    right = df.agree == 1
    assert np.allclose(df.p_true[right], df.conf[right], atol=1e-6)
    assert (df.p_true[~right] <= df.conf[~right] + 1e-6).all()
    assert (df.margin >= 0).all() and (df.entropy > 0).all()
    meta = json.loads(out.with_suffix(".meta.json").read_text())
    assert meta["schema_version"] == 2 and meta["git_commit"]
    assert df.git_commit.nunique() == 1 and df.git_commit.iloc[0] == meta["git_commit"]


def test_default_grid_is_plan_v2():
    cfg = CollectConfig()
    assert cfg.mask_rates == UNIFORM_RATES == (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    assert cfg.masks_per_rate == 2 and cfg.decoder_k == (2, 4, 8)
    assert cfg.decoder_states is None and cfg.include_uniform_matched
    states = uniform_masks(256, cfg.mask_rates, np.random.default_rng(0), 1, per_rate=2)
    assert len(states) == 18
    assert [s.target_rate for s in states[:4]] == [0.1, 0.1, 0.2, 0.2]


def test_decoder_records_every_state_and_commits_k(spy_trace):
    _, docs, df, _ = spy_trace
    L = next(iter(docs.values())).size
    dec = df[df.mask_kind == "decoder"]
    for (doc, k), traj in dec.groupby(["doc_id", "k"]):
        sizes = traj.groupby("step").n_masked.first().to_numpy()
        assert np.array_equal(sizes, np.arange(L, 0, -k))           # every state
        per = traj.groupby("step").agg(c=("committed", "sum"), n=("n_masked", "first"))
        assert np.array_equal(per.c.to_numpy(), np.minimum(k, per.n.to_numpy()))
        com = traj[traj.committed]
        assert com.groupby("step").commit_rank.apply(lambda r: sorted(r) == list(
            range(1, len(r) + 1))).all()
        # The committed positions are the most confident ones.
        for _, st in traj.groupby("step"):
            assert st.ell[st.committed].min() >= st.ell[~st.committed].max() - 1e-12 \
                if (~st.committed).any() else True
    assert (df.commit_rank[df.mask_kind != "decoder"] == -1).all()
    assert not df.committed[df.mask_kind != "decoder"].any()


def test_delta_ell_is_the_change_since_the_previous_step(spy_trace):
    _, _, df, _ = spy_trace
    dec = df[(df.mask_kind == "decoder") & (df.doc_id == 0) & (df.k == 4)]
    assert dec[dec.step == 0].delta_ell.isna().all()
    prev = dec[dec.step == 2].set_index("position").ell
    cur = dec[dec.step == 3].set_index("position")
    assert np.allclose(cur.delta_ell, cur.ell - prev.loc[cur.index], atol=1e-5)
    assert df[df.mask_kind != "decoder"].delta_ell.isna().all()


def test_matched_masks_have_the_decoder_count_and_ignore_its_ranking():
    L = 64
    a = run_trajectory(lambda m: {"ell": np.random.default_rng(1).normal(size=int(m.sum()))},
                       L, k=8)
    b = run_trajectory(lambda m: {"ell": np.random.default_rng(2).normal(size=int(m.sum()))},
                       L, k=8)
    sa, sb = [s.state for s in a], [s.state for s in b]
    assert any(not np.array_equal(x.mask, y.mask) for x, y in zip(sa, sb))
    ma = matched_uniform_masks(sa, np.random.default_rng(5))
    mb = matched_uniform_masks(sb, np.random.default_rng(5))
    assert [m.n_masked for m in ma] == [s.n_masked for s in sa]
    assert all(m.kind == "uniform_matched" and m.k == 8 for m in ma)
    # Different decoder rankings, same counts -> identical matched masks.
    assert all(np.array_equal(x.mask, y.mask) for x, y in zip(ma, mb))


def test_masks_respect_a_maskable_span():
    region = (10, 30)
    for st in uniform_masks(40, (0.5, 0.9), np.random.default_rng(0), 1, per_rate=2,
                            maskable=region):
        assert not st.mask[:10].any() and not st.mask[30:].any()
    traj = TeacherForcedTrajectory(40, 4, maskable=region)
    assert traj.mask.sum() == 20 and not traj.mask[:10].any()


def test_batching_does_not_change_the_rows(tmp_path):
    docs = _docs(n=2)
    a = load_traces(collect(ToyDiffusionLM(), docs, _small_cfg(batch_size=1),
                            tmp_path / "a.parquet", progress_every=0))
    b = load_traces(collect(ToyDiffusionLM(), docs, _small_cfg(batch_size=5),
                            tmp_path / "b.parquet", progress_every=0))
    for c in ("state_id", "position", "ell", "committed", "commit_rank"):
        assert np.array_equal(a[c].to_numpy(), b[c].to_numpy()), c


def test_nonfinite_logits_are_refused(tmp_path):
    with pytest.raises(FloatingPointError, match="fp16"):
        check_finite_logits(torch.tensor([1.0, float("inf")]), "m", "fp16")

    class NaNToy(ToyDiffusionLM):
        def logits(self, input_ids):
            lg = super().logits(input_ids)
            lg[0, 0, 0] = float("nan")
            return lg

    with pytest.raises(FloatingPointError):
        collect(NaNToy(), _docs(n=1), _small_cfg(), tmp_path / "x.parquet")
    assert list(tmp_path.iterdir()) == []            # no partial file left behind


def test_toy_honours_precision():
    assert build_model("toy", precision="fp16").precision == "fp16"
    assert build_model("toy").precision == "fp32"        # the 4-bit default means nothing here


# --------------------------------------------------------------------------- #
# Old schema, analysis arms, native gap
# --------------------------------------------------------------------------- #

def test_v1_pilot_file_still_loads():
    df = load_traces(PILOT, columns=SCALAR_COLUMNS)
    assert {"policy", "k"} <= set(df.columns) and "p_true" not in df.columns
    assert set(df.k[df.mask_kind == "decoder"]) == {8}
    assert set(arm_labels(df)) == {"uniform", "decoder"}


def test_arms_split_decoders_by_k(spy_trace):
    _, _, df, _ = spy_trace
    arms = set(arm_labels(df))
    assert {"uniform", "decoder:topk_conf:k4", "decoder:topk_conf:k8",
            "uniform_matched:topk_conf:k4"} <= arms


def test_native_gap_from_stored_commits_matches_the_rebuild(spy_trace):
    _, _, df, _ = spy_trace
    arm = df[(df.mask_kind == "decoder") & (df.k == 4)].reset_index(drop=True)
    pos = PositionFrame.from_frame(arm)
    codes = np.unique(pos.doc_id)
    counts = doc_boot_counts(codes.size, 50, seed=0, label="t")
    stored = native_gap(pos, codes, counts, k=4, committed=arm.committed.to_numpy())
    rebuilt = native_gap(pos, codes, counts, k=4, min_masked=1)
    assert np.allclose(stored.point, rebuilt.point)


# --------------------------------------------------------------------------- #
# Sharding, resume, merge
# --------------------------------------------------------------------------- #

def test_shards_partition_the_documents():
    docs = [Document(i, np.zeros(4, int)) for i in (0, 1, 2, 5, 7, 8, 13)]
    parts = [shard_documents(docs, i, 3) for i in range(3)]
    ids = [d.doc_id for p in parts for d in p]
    assert sorted(ids) == [0, 1, 2, 5, 7, 8, 13] and len(ids) == len(set(ids))
    with pytest.raises(ValueError):
        shard_documents(docs, 3, 3)


def _shard(tmp_path, docs, shard=0, n=1, model=None, resume=False, **kw):
    return collect_shard(model or ToyDiffusionLM(), docs, _small_cfg(**kw), tmp_path / "run",
                         shard, n, resume=resume, part_docs=2, progress_every=0)


def test_resume_skips_finished_documents_and_merges_cleanly(tmp_path):
    docs = _docs(n=5)
    _shard(tmp_path, docs[:3])                       # a run that stopped after 3 documents
    with pytest.raises(FileExistsError):
        _shard(tmp_path, docs)
    spy = SpyToy()
    meta = json.loads(_shard(tmp_path, docs, model=spy, resume=True).read_text())
    assert meta["complete"]
    # Only documents 3 and 4 were scored: identify each input by its visible tokens.
    scored = set()
    for x in spy.seen:
        vis = x != spy.mask_token_id
        if vis.any():
            scored |= {i for i, x0 in docs if np.array_equal(x[vis], x0[vis])}
    assert scored == {3, 4}
    out = merge(tmp_path / "run", tmp_path / "run.parquet")
    df = load_traces(out)
    assert sorted(df.doc_id.unique()) == [0, 1, 2, 3, 4]
    assert not df.duplicated(["state_id", "position"]).any()


def test_resume_refuses_a_different_configuration(tmp_path):
    _shard(tmp_path, _docs(n=2))
    with pytest.raises(ValueError, match="config"):
        _shard(tmp_path, _docs(n=2), resume=True, topk=10)


def test_merge_rejects_mismatched_or_incomplete_shards(tmp_path):
    docs = _docs(n=4)
    _shard(tmp_path, docs, 0, 2)
    _shard(tmp_path, docs, 1, 2, model=ToyDiffusionLM(precision="fp16"))
    with pytest.raises(MergeError, match="precision"):
        merge(tmp_path / "run", tmp_path / "m.parquet")

    other = tmp_path / "b"
    other.mkdir()
    _shard(other, docs, 0, 2)
    m1 = _shard(other, docs, 1, 2)
    meta = json.loads(m1.read_text())
    meta["complete"] = False
    m1.write_text(json.dumps(meta))
    with pytest.raises(MergeError, match="incomplete"):
        merge(other / "run", other / "m.parquet")

    dup = tmp_path / "c"
    dup.mkdir()
    m0 = _shard(dup, docs, 0, 2)
    m1 = _shard(dup, docs, 1, 2)
    meta1 = json.loads(m1.read_text())
    meta1["parts"].append(json.loads(m0.read_text())["parts"][0])   # a part written twice
    m1.write_text(json.dumps(meta1))
    with pytest.raises(MergeError, match="appears in"):
        merge(dup / "run", dup / "m.parquet")
    assert not (dup / "m.parquet").exists()


def test_t2_pilot_preset():
    p = PRESETS["t2-pilot"]
    cfg = CollectConfig(**p["config"])
    assert p["model"] == "llada-8b-base"
    assert len(cfg.mask_rates) == 5 and cfg.masks_per_rate == 2
    assert cfg.decoder_k == (4,) and cfg.decoder_states is None and cfg.topk == 20


# --------------------------------------------------------------------------- #
# Command line: collection, floor, single-shard merge
# --------------------------------------------------------------------------- #

def _run(*args):
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    return subprocess.run([sys.executable, *map(str, args)], check=True, capture_output=True,
                          text=True, cwd=root)


def test_cli_floor_run_and_single_shard_merge(tmp_path):
    common = ["--model", "toy", "--device", "cpu", "--limit", "3", "--max-tokens", "40",
              "--mask-rates", "0.3", "0.7", "--decoder-k", "4"]
    _run("scripts/collect_traces.py", *common, "--repeat", "2", "--out", tmp_path / "f.parquet")
    _run("scripts/measure_floor.py", "--traces", tmp_path / "f.rep0.parquet",
         tmp_path / "f.rep1.parquet", "--out", tmp_path / "floor.json")
    floor = json.loads((tmp_path / "floor.json").read_text())
    assert floor["plan_v2"]["floor_delta"] == 0.0 and floor["bitwise_identical"]
    assert {"uniform", "decoder", "uniform_matched"} <= set(floor["plan_v2"]["per_arm"])

    _run("scripts/collect_traces.py", *common, "--resume", "--out", tmp_path / "s")
    df = load_traces(tmp_path / "s.parquet")
    assert sorted(df.doc_id.unique()) == [0, 1, 2]
    meta = json.loads((tmp_path / "s.meta.json").read_text())
    assert meta["n_docs"] == 3 and meta["schema_version"] == 2
