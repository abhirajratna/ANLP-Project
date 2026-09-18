"""Stage 0: dispersion estimation and the written gate."""
import numpy as np
import pytest

from c3.dispersion import estimate_dispersion
from c3.power import required_documents, stage0_gate
from c3.simulate import simulate

BASE = dict(n_docs=40, n_states_per_doc=3, n_positions=256, n_pools_per_size=4)


def test_dispersion_near_zero_when_noise_is_uniform():
    fr, _ = simulate(dgp="heteroskedastic", tau=0.0, positional_noise=True, seed=21, **BASE)
    assert estimate_dispersion(fr).dispersion < 0.15


def test_dispersion_detected_when_it_tracks_position():
    fr, _ = simulate(dgp="heteroskedastic", tau=1.2, positional_noise=True, seed=21, **BASE)
    assert estimate_dispersion(fr).dispersion > 0.3


def test_idiosyncratic_dispersion_is_not_identified():
    """Documented limitation: with no observable correlate the estimate collapses."""
    hi, _ = simulate(dgp="heteroskedastic", tau=1.2, positional_noise=False, seed=21, **BASE)
    assert estimate_dispersion(hi).dispersion < 0.2


def test_required_sample_grows_as_the_effect_shrinks():
    big = required_documents(-0.20, 0.05, 20).n_required
    small = required_documents(-0.05, 0.05, 20).n_required
    assert small > big


def test_noise_floor_can_bind_instead_of_the_estimate():
    r = required_documents(-0.01, 0.05, 20, floor_effect=0.05)
    assert r.binding_bar == "noise_floor"
    assert r.target_effect == pytest.approx(0.10)


def test_gate_requires_both_conditions():
    assert stage0_gate(500, 0.5).proceed_full
    assert not stage0_gate(5000, 0.5).proceed_full        # sample too large
    assert not stage0_gate(500, 0.05).proceed_full        # dispersion too small
    assert not stage0_gate(5000, 0.05).proceed_full


def test_gate_records_its_reasons():
    g = stage0_gate(5000, 0.05)
    assert len(g.reasons) == 2 and "EXCEEDS" in g.reasons[0]
    assert "dispersion" in g.to_dict()
