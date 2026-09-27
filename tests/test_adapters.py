"""Model adapters and the trace harness: the correctness fixes plan v2 depends on."""
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from c3.assemble import load_traces
from c3.traces.collect import STATE_STRIDE, CollectConfig, collect
from c3.traces.models import (KNOWN_MODELS, MODEL_ALIASES, DiffusionLM, HFDiffusionLM,
                              ToyDiffusionLM, detect_variant)


class _FixedLogits(DiffusionLM):
    """Returns preset logits, so the scoring arithmetic can be checked exactly."""

    def __init__(self, logits, mask_token_id):
        self._lg = torch.as_tensor(logits, dtype=torch.float32)[None]
        self.mask_token_id = mask_token_id
        self.name = "fixed"

    def logits(self, input_ids):
        return self._lg


def test_log_odds_are_exact_and_finite_when_confidence_saturates():
    # top-1 logit 60 above the rest: conf rounds to 1.0 in float32, ell must not.
    lg = np.zeros((2, 5))
    lg[0, 1] = 60.0
    lg[1, 2] = 2.0
    out = _FixedLogits(lg, mask_token_id=4).score_masked(torch.zeros(1, 2), np.array([True, True]))
    ref0 = 60.0 - np.log(3.0)        # z1 - logsumexp(three zeros; mask token excluded)
    ref1 = 2.0 - np.log(3.0)
    assert np.all(np.isfinite(out.ell))
    assert out.ell[0] == pytest.approx(ref0, abs=1e-4)
    assert out.ell[1] == pytest.approx(ref1, abs=1e-5)
    assert out.n_saturated == 1


def test_mask_token_is_never_the_prediction():
    lg = np.zeros((1, 5))
    lg[0, 4] = 10.0                   # the mask token has the largest raw logit
    lg[0, 0] = 1.0
    out = _FixedLogits(lg, mask_token_id=4).score_masked(torch.zeros(1, 1), np.array([True]))
    assert out.top1[0] == 0


def test_dream_logits_are_shifted_right_by_one():
    class _Stub:
        def __call__(self, ids):
            L = ids.shape[1]
            lg = torch.zeros(1, L, 8)
            lg[0, torch.arange(L), torch.arange(L)] = 5.0   # position i predicts token i
            return type("O", (), {"logits": lg})()

    m = HFDiffusionLM(name="stub", path="none", mask_token_id=7, logit_shift=1)
    m.model = _Stub()
    lg = m.logits(torch.zeros(1, 4, dtype=torch.long))
    # After the shift, position i carries what the model emitted at i-1.
    assert lg[0, 2].argmax().item() == 1 and lg[0, 0].argmax().item() == 0
    assert KNOWN_MODELS["dream-7b-base"]["logit_shift"] == 1
    assert KNOWN_MODELS["llada-8b-base"]["logit_shift"] == 0


def test_registry_is_variant_explicit_with_pinned_revisions():
    for key, spec in KNOWN_MODELS.items():
        assert spec["variant"] in {"base", "instruct"}
        assert len(spec["revision"]) == 40
    assert set(MODEL_ALIASES.values()) <= set(KNOWN_MODELS)


def test_detect_variant_from_chat_template(tmp_path):
    inst, base = tmp_path / "inst", tmp_path / "base"
    inst.mkdir(); base.mkdir()
    (inst / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{x}}"}))
    (base / "tokenizer_config.json").write_text(json.dumps({"bos_token": "<s>"}))
    assert detect_variant(str(inst)) == "instruct"
    assert detect_variant(str(base)) == "base"
    assert detect_variant("GSAI-ML/LLaDA-8B-Base") is None


def test_input_device_for_a_plain_module():
    class _M(DiffusionLM):
        def __init__(self):
            self.model = torch.nn.Linear(2, 2)
    assert _M().input_device.type == "cpu"


def test_toy_model_is_informative_and_state_ids_unique(tmp_path):
    rng = np.random.default_rng(0)
    docs = [(i, rng.integers(0, 60, 96)) for i in range(8)]
    out = collect(ToyDiffusionLM(noise_dispersion=0.3), docs,
                  CollectConfig(mask_rates=(0.3, 0.7), decoder_states=3), tmp_path / "t.parquet",
                  progress_every=0)
    df = load_traces(out)
    # Agreement is neither ~0 (the old toy) nor ~1, and tracks confidence.
    assert 0.2 < df.agree.mean() < 0.95
    hi, lo = df.conf > df.conf.median(), df.conf <= df.conf.median()
    assert df.agree[hi].mean() > df.agree[lo].mean() + 0.1
    # State ids are unique per (doc, state) and decode back to the doc.
    per_state = df.groupby("state_id").doc_id.nunique()
    assert per_state.max() == 1
    assert np.all(df.state_id // STATE_STRIDE == df.doc_id)
    meta = json.loads((tmp_path / "t.meta.json").read_text())
    assert "provenance" in meta and meta["provenance"]["git_commit"]
    assert "n_saturated" in meta and meta["variant"] == "toy"
    assert {"variant", "state_idx", "seed"} <= set(df.columns)
