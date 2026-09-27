"""Document-level inference and calibration metrics (plan v2 Sec. 5)."""
import numpy as np
import pytest

from c3.metrics import (ece_equal_mass, equal_mass_bins, isotonic_fit, smooth_ece,
                        smooth_ece_at)
from c3.power import required_documents_sd
from c3.stats import (boot_pvalue, declare, doc_boot_counts, grouped_doc_mean, holm,
                      percentile_ci)


# ---------------------------------------------------------------- stats

def test_doc_first_mean_weights_documents_equally():
    # Doc 0 has 10x the rows of doc 1; a row-weighted mean would sit near 1.
    values = np.r_[np.ones(100), np.zeros(10)]
    doc = np.r_[np.zeros(100, int), np.ones(10, int)]
    counts = doc_boot_counts(2, 200, seed=0)
    g = grouped_doc_mean(values, doc, 2, np.zeros(110, int), counts)
    assert g.point[0] == pytest.approx(0.5)


def test_within_document_weights():
    values = np.array([1.0, 0.0, 0.0])
    doc = np.zeros(3, int)
    g = grouped_doc_mean(values, doc, 1, np.zeros(3, int), doc_boot_counts(1, 10),
                         weights=np.array([2.0, 1.0, 1.0]))
    assert g.point[0] == pytest.approx(0.5)


def test_shared_resamples_give_tighter_contrast_than_independent():
    rng = np.random.default_rng(0)
    n_docs = 60
    doc_effect = rng.normal(0, 1.0, n_docs)          # shared across levels
    rows_doc = np.repeat(np.arange(n_docs), 20)
    level = np.tile(np.r_[np.zeros(10, int), np.ones(10, int)], n_docs)
    values = doc_effect[rows_doc] + 0.05 * level + rng.normal(0, 0.1, rows_doc.size)
    counts = doc_boot_counts(n_docs, 2000, seed=1)
    g = grouped_doc_mean(values, rows_doc, n_docs, level, counts)
    _, joint = g.contrast(1, 0)
    indep = g.draws[:, 1] - np.random.default_rng(2).permutation(g.draws[:, 0])
    assert np.std(joint) < 0.5 * np.std(indep)
    lo, hi = percentile_ci(joint)
    assert lo > 0                                     # the planted +0.05 is detected


def test_boot_counts_are_multinomial_over_documents():
    c = doc_boot_counts(25, 300, seed=4)
    assert c.shape == (300, 25) and np.all(c.sum(axis=1) == 25)
    assert np.array_equal(c, doc_boot_counts(25, 300, seed=4))


def test_holm_matches_textbook():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    adj, rej = holm(p)
    # sorted: .005*4=.02, .01*3=.03, .03*2=.06, .04*1=.04 -> monotone .02,.03,.06,.06
    assert np.allclose(adj, [0.03, 0.06, 0.06, 0.02])
    assert rej.tolist() == [True, False, False, True]


def test_holm_matches_statsmodels():
    sm = pytest.importorskip("statsmodels.stats.multitest")
    p = np.random.default_rng(3).uniform(0, 0.2, 12)
    ours, _ = holm(p)
    theirs = sm.multipletests(p, method="holm")[1]
    assert np.allclose(ours, theirs)


def test_boot_pvalue_and_declare():
    assert boot_pvalue(np.full(999, 0.3)) == pytest.approx(2 / 1000)
    assert boot_pvalue(np.r_[-np.ones(500), np.ones(500)]) == pytest.approx(1.0)
    assert declare(0.05, (0.01, 0.09), floor=0.01)
    assert not declare(0.05, (0.01, 0.09), floor=0.03)     # below 2 x floor
    assert not declare(0.05, (-0.01, 0.09), floor=0.0)     # CI covers zero


def test_required_documents_matches_the_plan_formula():
    # N = 7.85 sigma_d^2 / delta_min^2
    assert required_documents_sd(0.02, 0.02) == 8
    assert required_documents_sd(0.10, 0.02) == int(np.ceil(7.849 * 25))
    with pytest.raises(ValueError):
        required_documents_sd(0.1, 0.0)


# ---------------------------------------------------------------- metrics

def test_isotonic_is_monotone_and_pools_violators():
    fit = isotonic_fit(np.array([1, 2, 3, 4]), np.array([1.0, 3.0, 2.0, 4.0]))
    assert fit.y.tolist() == [1.0, 2.5, 2.5, 4.0]
    x = np.random.default_rng(0).normal(size=500)
    y = (np.random.default_rng(1).random(500) < 1 / (1 + np.exp(-x))).astype(float)
    f = isotonic_fit(x, y)
    assert np.all(np.diff(f.y) >= -1e-12)
    # Weighted least squares preserves the total.
    assert f.predict(x).sum() == pytest.approx(y.sum())


def test_isotonic_merges_ties_and_clips_outside():
    f = isotonic_fit(np.array([0, 0, 1, 1]), np.array([0.0, 1.0, 1.0, 1.0]))
    assert f.predict(np.array([-5.0, 0.0, 5.0])).tolist() == [0.5, 0.5, 1.0]


def test_equal_mass_bins_are_balanced_even_with_ties():
    conf = np.r_[np.full(900, 0.999), np.linspace(0, 1, 600)]
    b = equal_mass_bins(conf, 15)
    counts = np.bincount(b, minlength=15)
    assert counts.max() - counts.min() <= 1


def test_ece_near_zero_when_calibrated_and_recovers_constant_offset():
    rng = np.random.default_rng(0)
    c = rng.uniform(0.05, 0.95, 100_000)
    a = (rng.random(c.size) < c).astype(float)
    assert ece_equal_mass(c, a).ece < 0.01
    c2 = np.full(100_000, 0.8)
    a2 = (rng.random(c2.size) < 0.7).astype(float)
    assert ece_equal_mass(c2, a2).ece == pytest.approx(0.1, abs=0.01)


def test_ece_bootstrap_interval_brackets_the_estimate():
    rng = np.random.default_rng(1)
    n_docs, per = 40, 250
    doc = np.repeat(np.arange(n_docs), per)
    c = rng.uniform(0.3, 0.99, doc.size)
    a = (rng.random(doc.size) < c - 0.05).astype(float)
    r = ece_equal_mass(c, a, 15, doc, doc_boot_counts(n_docs, 500, seed=2))
    assert r.ci[0] <= r.ece <= r.ci[1]


def test_smooth_ece_small_when_calibrated_large_when_not():
    rng = np.random.default_rng(0)
    c = rng.uniform(0.05, 0.95, 50_000)
    a = (rng.random(c.size) < c).astype(float)
    s, bw = smooth_ece(c, a)
    assert s < 0.02 and bw > 0
    c2 = np.clip(c + 0.1, 0, 1)
    assert smooth_ece(c2, a)[0] > 0.05
    assert smooth_ece_at(c2, a, 0.05) > smooth_ece_at(c, a, 0.05)
