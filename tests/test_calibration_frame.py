"""The proposal-era commitment gap, and document-level splitting.

The v2 gap (sign conf - agree, document-first) is tested in test_endpoints.py.
"""
import numpy as np

from c3 import POOL_SIZES
from c3.calibration import commitment_gap, gap_slope
from c3.frame import AnalysisFrame
from c3.simulate import simulate

BASE = dict(n_docs=40, n_states_per_doc=3, n_positions=256, n_pools_per_size=8)


def test_gap_reported_for_every_pool_size():
    fr, _ = simulate(dgp="homoskedastic", seed=11, **BASE)
    gaps = commitment_gap(fr, n_boot=100)
    assert [g.m for g in gaps] == list(POOL_SIZES)
    assert all(g.ci_low <= g.gap <= g.ci_high for g in gaps)


def test_gap_widens_with_field_size_when_confidence_is_noisy():
    """The winner's curse as a decoder experiences it: bigger field, worse gap."""
    fr, _ = simulate(dgp="homoskedastic", sigma=1.5, seed=11, **BASE)
    gaps = commitment_gap(fr, n_boot=100)
    assert gaps[-1].gap < gaps[0].gap
    assert gap_slope(gaps) < 0


def test_split_is_by_document_and_disjoint():
    fr, _ = simulate(dgp="homoskedastic", seed=11, **BASE)
    a, b = fr.split_by_document(0.5, seed=0)
    assert set(a.doc_id) & set(b.doc_id) == set()
    assert len(a) + len(b) == len(fr)
    # States must not straddle the halves, or the fixed effects leak.
    assert set(a.state_id) & set(b.state_id) == set()


def test_frame_rejects_ragged_columns():
    import pytest
    n = 10
    ok = dict(doc_id=np.zeros(n), state_id=np.zeros(n), position=np.zeros(n),
              ell=np.zeros(n), agree=np.zeros(n), m=np.zeros(n), r=np.zeros(n))
    AnalysisFrame(**ok)
    with pytest.raises(ValueError):
        AnalysisFrame(**{**ok, "ell": np.zeros(n - 1)})


def test_winners_selects_rank_one_only():
    fr, _ = simulate(dgp="homoskedastic", seed=11, **BASE)
    assert np.all(fr.winners().r == 1)
