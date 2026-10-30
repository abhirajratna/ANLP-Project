"""The decode loop and the T12 commit policies, on the CPU toy model."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from c3.decode.loop import decode
from c3.decode.policies import (KLASS, EntropyBounded, Margin, PCSampler, StepContext, Threshold,
                                TopKConfidence, unigram_log_freq)
from c3.traces.masks import run_trajectory
from c3.traces.models import ToyDiffusionLM


def _setup(L=40, prompt=8, seed=0):
    rng = np.random.default_rng(seed)
    x0 = rng.integers(0, 60, L)
    model = ToyDiffusionLM(noise_dispersion=0.3)
    model.set_reference(x0)
    x = x0.copy()
    x[prompt:] = model.mask_token_id
    return model, x0, x


def _check_run(res, x_init, mask_id):
    committed = np.concatenate([s.positions for s in res.steps])
    assert sorted(committed.tolist()) == np.flatnonzero(x_init == mask_id).tolist()
    assert not np.any(res.tokens == mask_id)                          # terminated
    assert res.n_forward == len(res.steps)
    assert res.tokens_per_step == pytest.approx(committed.size / res.n_forward)


@pytest.mark.parametrize("policy_cls", [TopKConfidence, Margin, PCSampler])
@pytest.mark.parametrize("k", [1, 3, 8])
def test_fixed_k_policies_commit_k_per_step(policy_cls, k):
    model, x0, x = _setup()
    res = decode(model, x, policy_cls(k))
    _check_run(res, x, model.mask_token_id)
    for s in res.steps:
        assert s.positions.size == min(k, s.n_eligible)
    assert np.array_equal(res.tokens[:8], x0[:8])                    # the prompt is untouched


def test_threshold_commits_exactly_the_confident_ones_or_one():
    model, _, x = _setup()
    res = decode(model, x, Threshold(tau=0.5))
    _check_run(res, x, model.mask_token_id)
    for s in res.steps:
        if s.positions.size > 1:
            assert np.all(s.conf >= 0.5)
    stats = {"ell": np.array([3.0, -1.0, 2.0]), "conf": np.array([0.95, 0.2, 0.9])}
    order, n = Threshold(0.9).select(stats, None)
    assert n == 2 and list(order[:2]) == [0, 2]
    assert Threshold(0.99).select(stats, None)[1] == 1                # fallback: the max


def test_entropy_bounded_prefix_rule():
    stats = {"entropy": np.array([0.5, 0.1, 0.3, 2.0])}
    order, n = EntropyBounded(gamma=0.45).select(stats, None)
    assert list(order) == [1, 2, 0, 3]
    assert n == 3            # prefixes: 0 <= g, 0.1 <= g, 0.4 <= g, 0.9 > g
    assert EntropyBounded(gamma=0.0).select(stats, None)[1] == 1
    model, _, x = _setup()
    _check_run(decode(model, x, EntropyBounded(0.5)), x, model.mask_token_id)


def test_klass_needs_a_stable_history_and_confidence():
    stats = {"ell": np.array([2.0, 1.0, 0.0]), "conf": np.array([0.95, 0.92, 0.5])}
    pos = np.array([10, 11, 12])
    ctx = StepContext(step=3, positions=pos, kl_history={10: [1e-4, 1e-4], 11: [0.5, 1e-4],
                                                         12: [1e-5, 1e-5]})
    order, n = KLASS(tau=0.9, eps_kl=1e-3, history=2).select(stats, ctx)
    assert n == 1 and order[0] == 0            # 11 moved too much, 12 is not confident
    order, n = KLASS(tau=0.99, eps_kl=1e-3).select(stats, ctx)
    assert n == 1 and order[0] == 0            # nothing stable: top-u by confidence
    model, _, x = _setup()
    res = decode(model, x, KLASS(tau=0.0, eps_kl=10.0, history=1))
    _check_run(res, x, model.mask_token_id)
    # Step 0 has no KL history (fallback: one token); at step 1 every position
    # has one small KL and passes, so the rest are committed in parallel.
    assert [s.positions.size for s in res.steps] == [1, 31]


def test_pc_sampler_positional_weight_decodes_left_to_right():
    model, _, x = _setup()
    res = decode(model, x, PCSampler(k=1, lam=50.0))
    firsts = [int(s.positions[0]) for s in res.steps]
    assert firsts == sorted(firsts)
    lf = unigram_log_freq([np.array([0, 0, 1])], vocab_size=4)
    assert lf.shape == (4,) and lf[0] > lf[1] > lf[2]
    assert PCSampler(1).variant == "position-only" and PCSampler(1, log_freq=lf).variant == "full"


def test_block_wise_decoding_stays_in_the_active_block():
    model, _, x = _setup(L=40, prompt=8)
    res = decode(model, x, TopKConfidence(3), block_length=8)
    _check_run(res, x, model.mask_token_id)
    for s in res.steps:
        lo = 8 + 8 * s.block
        assert np.all((s.positions >= lo) & (s.positions < lo + 8))
    assert [s.block for s in res.steps] == sorted(s.block for s in res.steps)


def test_teacher_forced_loop_reproduces_the_trace_trajectory():
    model, x0, _ = _setup(L=32, prompt=0)
    x = np.full_like(x0, model.mask_token_id)
    res = decode(model, x, TopKConfidence(4), teacher=x0)
    assert np.array_equal(res.tokens, x0)

    def score(mask):
        ids = x0.copy()
        ids[mask] = model.mask_token_id
        from c3.decode.scoring import position_stats
        lg = model.logits(torch.as_tensor(ids[None]))
        return position_stats(lg, mask[None], model.mask_token_id, true_ids=x0[None])

    traj = run_trajectory(score, 32, k=4)
    assert len(traj) == len(res.steps)
    for t, s in zip(traj, res.steps):
        pos = np.flatnonzero(t.state.mask)
        assert set(pos[t.committed].tolist()) == set(s.positions.tolist())
        assert np.all(np.isfinite(s.p_true))                           # a reference was given


def test_sampling_is_reproducible_by_seed():
    model, _, x = _setup()
    a = decode(model, x, TopKConfidence(2), sample=True, seed=3)
    b = decode(model, x, TopKConfidence(2), sample=True, seed=3)
    assert np.array_equal(a.tokens, b.tokens)


# --------------------------------------------------------------------------- #
# C3a in the decode loop (plan v2 T11); the policy alone is in test_c3a_policy.py
# --------------------------------------------------------------------------- #

class _StateSpy:
    """Top-1 per step, recording what a policy with ``needs_state`` is given."""

    needs_state = True

    def __init__(self):
        self.seen = []

    def select(self, stats, ctx):
        self.seen.append((ctx.positions.copy(), dict(ctx.state)))
        return np.argsort(-np.asarray(stats["ell"]), kind="stable"), 1


class _TopKWithState(TopKConfidence):
    """Top-k that asks for the state and ignores it."""

    needs_state = True

    def select(self, stats, ctx):
        assert ctx.state is not None
        return super().select(stats, ctx)


def test_state_features_cover_the_whole_masked_region_not_just_the_block():
    model, _, x = _setup(L=40, prompt=8)
    spy = _StateSpy()
    decode(model, x, spy, block_length=8)
    region = 32
    for step, (positions, st) in enumerate(spy.seen):
        n_masked = region - step                         # one commit per step
        assert st["mask_rate"] == pytest.approx(n_masked / region)
        assert st["log_n_masked"] == pytest.approx(np.log(n_masked))
        assert positions.size <= 8                       # the policy still sees only its block
    # the median is over every masked position: at step 0 that is all 32, and it equals
    # the median of a whole-span step-0 state
    whole = _StateSpy()
    decode(model, x, whole)
    assert spy.seen[0][1]["median_ell"] == pytest.approx(whole.seen[0][1]["median_ell"])


def test_c3a_decodes_from_one_per_step_to_a_block_per_step():
    from c3.decode.policies import C3a
    from test_c3a_policy import _calibrator

    cal = _calibrator()
    model, _, x = _setup(L=40, prompt=8)
    slow = decode(model, x, C3a(1e-12, cal), block_length=8)
    _check_run(slow, x, model.mask_token_id)
    assert slow.n_forward == 32                          # nothing reaches 1 - 1e-12: one per step
    fast = decode(model, x, C3a(1.0, cal), block_length=8)
    _check_run(fast, x, model.mask_token_id)
    assert fast.n_forward == 4                           # everything clears 0: a block per step


def test_computing_the_state_changes_no_decision():
    model, _, x = _setup()
    plain = decode(model, x, TopKConfidence(3), block_length=8)
    with_state = decode(model, x, _TopKWithState(3), block_length=8)
    assert np.array_equal(plain.tokens, with_state.tokens)
    assert plain.n_forward == with_state.n_forward
    assert [s.positions.tolist() for s in plain.steps] ==         [s.positions.tolist() for s in with_state.steps]
