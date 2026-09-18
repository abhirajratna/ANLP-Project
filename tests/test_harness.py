"""The trace harness end to end, on CPU, with no model download."""
import numpy as np

from c3.assemble import frame_from_traces, load_traces
from c3.traces.collect import CollectConfig, collect
from c3.traces.masks import decoder_trajectory_masks, uniform_masks
from c3.traces.models import ToyDiffusionLM, _to_logodds


def test_uniform_masks_respect_the_minimum():
    states = uniform_masks(128, rates=(0.01, 0.5), rng=np.random.default_rng(0), min_masked=32)
    assert all(s.n_masked >= 32 for s in states)


def test_decoder_trajectory_commits_progressively():
    rng = np.random.default_rng(0)
    states = decoder_trajectory_masks(lambda m: rng.random(int(m.sum())),
                                      length=128, tokens_per_step=8,
                                      max_states=5, min_masked=32)
    counts = [s.n_masked for s in states]
    assert counts == sorted(counts, reverse=True)
    assert all(s.kind == "decoder" for s in states)


def test_decoder_trajectory_spans_the_whole_decode():
    """Regression: taking the first N steps returns only near-fully-masked states.

    On LLaDA-8B that produced an agreement rate of 0.038 and no usable spread in
    confidence, because the model was predicting everything from nothing.
    """
    rng = np.random.default_rng(0)
    states = decoder_trajectory_masks(lambda m: rng.random(int(m.sum())),
                                      length=256, tokens_per_step=8,
                                      max_states=4, min_masked=32)
    rates = [s.rate for s in states]
    assert max(rates) > 0.95, "trajectory should start near fully masked"
    assert min(rates) < 0.30, f"trajectory should reach a sparsely masked state, got {rates}"
    assert len(states) <= 4


def test_saturated_confidence_is_clipped_and_counted():
    ell, n_sat = _to_logodds(np.array([1.0, 0.5, 0.0]), eps=1e-6)
    assert np.all(np.isfinite(ell)) and n_sat == 2


def test_collect_to_frame_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    docs = [(i, rng.integers(0, 60, 96)) for i in range(6)]
    out = collect(ToyDiffusionLM(noise_dispersion=0.5),
                  docs, CollectConfig(mask_rates=(0.5,), decoder_states=1), 
                  tmp_path / "t.parquet", progress_every=0)
    df = load_traces(out)
    assert {"uniform", "decoder"} <= set(df.mask_kind.unique())
    assert df.agree.isin([0, 1]).all() and np.isfinite(df.ell).all()
    frame = frame_from_traces(out, mask_kind="uniform", n_pools_per_size=8)
    assert len(frame) > 0 and set(frame.m.tolist()) <= {2, 4, 8, 16, 32}
    assert (tmp_path / "t.meta.json").exists()
