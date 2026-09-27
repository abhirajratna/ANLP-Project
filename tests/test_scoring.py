"""Batched per-position statistics against a float64 numpy reference."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from c3.decode.scoring import position_stats


def _reference(z, mask_id, truth):
    z = z.astype(np.float64).copy()
    z[mask_id] = -np.inf
    p = np.exp(z - z.max())
    p /= p.sum()
    order = np.argsort(-p, kind="stable")
    t1 = order[0]
    others = np.delete(z, t1)
    lse = np.log(np.sum(np.exp(others[np.isfinite(others)] - z[t1]))) + z[t1]
    ent = -np.sum(p[p > 0] * np.log(p[p > 0]))
    return dict(top1=t1, conf=p[t1], ell=z[t1] - lse, p_true=p[truth], entropy=ent,
                margin=p[order[0]] - p[order[1]], topk_ids=order[:5], topk_p=p[order[:5]])


def test_matches_reference_for_every_masked_position():
    rng = np.random.default_rng(0)
    B, L, V, mask_id = 3, 7, 40, 39
    logits = rng.normal(0, 2, (B, L, V))
    masked = rng.random((B, L)) < 0.5
    truth = rng.integers(0, V - 1, (B, L))
    out = position_stats(torch.as_tensor(logits, dtype=torch.float32), masked, mask_id,
                         true_ids=truth, topk=5, chunk=4)
    assert out["rows"].size == masked.sum()
    for i, (b, c) in enumerate(zip(out["rows"], out["cols"])):
        ref = _reference(logits[b, c], mask_id, truth[b, c])
        assert out["top1"][i] == ref["top1"]
        for f in ("conf", "ell", "p_true", "entropy", "margin"):
            assert out[f][i] == pytest.approx(ref[f], abs=1e-5), f
        assert np.array_equal(out["topk_ids"][i], ref["topk_ids"])
        assert np.allclose(out["topk_p"][i], ref["topk_p"], atol=1e-6)


def test_mask_token_excluded_and_ell_exact_when_saturated():
    V, mask_id = 10, 9
    z = np.zeros((1, 2, V))
    z[0, 0, mask_id] = 50.0           # mask has the largest raw logit
    z[0, 1, 3] = 60.0                 # conf rounds to 1.0 in float32
    out = position_stats(torch.as_tensor(z, dtype=torch.float32), np.ones((1, 2), bool), mask_id)
    assert mask_id not in out["topk_ids"][0]
    assert out["top1"][1] == 3
    assert np.isfinite(out["ell"]).all()
    assert out["ell"][1] == pytest.approx(60.0 - np.log(8.0), abs=1e-4)
    assert np.isnan(out["p_true"]).all()


def test_no_masked_positions():
    out = position_stats(torch.zeros(2, 3, 5), np.zeros((2, 3), bool), 4, topk=3)
    assert out["rows"].size == 0 and out["topk_ids"].shape == (0, 3)
