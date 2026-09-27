"""The 110M training stack on CPU with a tiny model: plan v2 T3 requirements."""
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from c3.training.budget import N_PARAMS_DEFAULT
from c3.training.data import DocWindowDataset, build_token_shards
from c3.training.exclusion import Exclusion, ExclusionEntry, token_hash
from c3.training.model import MaskedDiffusionLM, ModelConfig, mdlm_loss, nelbo_eval

ROOT = Path(__file__).resolve().parents[1]
TINY = dict(d_model=16, n_layers=2, n_heads=2, d_ff=32)


class FakeTokenizer:
    """Maps each character to a token id below GPT-2's EOS/pad (50256)."""

    def __call__(self, texts, add_special_tokens=False):
        if isinstance(texts, str):
            return {"input_ids": [ord(c) % 50000 for c in texts]}
        return {"input_ids": [[ord(c) % 50000 for c in t] for t in texts]}


def _hash(text):
    return "h" + str(abs(hash(text)) % 10 ** 12)


def _records(n, rng, min_len=20, max_len=700):
    out = []
    for i in range(n):
        L = int(rng.integers(min_len, max_len))
        text = "".join(chr(97 + int(c)) for c in rng.integers(0, 26, L))
        out.append({"id": f"doc{i}", "text": text})
    return out


def _manifest(tmp_path, entries):
    p = tmp_path / "excl.json"
    Exclusion(entries).save(p)
    return p


def _build(tmp_path, recs, entries, **kw):
    return build_token_shards(tmp_path / "shards", _manifest(tmp_path, entries), FakeTokenizer(),
                              target_tokens=10 ** 9, shard_tokens=20_000, records=recs,
                              content_hash_fn=_hash, **kw)


def test_param_constant_matches_model():
    assert ModelConfig().n_params == N_PARAMS_DEFAULT


def test_subs_mask_logit_is_minus_inf_and_padding_is_invisible():
    torch.manual_seed(0)
    cfg = ModelConfig(max_len=12, **TINY)
    m = MaskedDiffusionLM(cfg).eval()
    ids = torch.randint(0, 1000, (2, 12))
    valid = torch.ones(2, 12, dtype=torch.bool)
    valid[:, 8:] = False
    out = m(ids, valid)
    assert torch.isinf(out[..., cfg.mask_token_id]).all()
    ids2 = ids.clone()
    ids2[:, 8:] = 7                      # change only the padding
    out2 = m(ids2, valid)
    assert torch.allclose(out[:, :8], out2[:, :8], atol=1e-6)


def test_nelbo_of_a_uniform_model_is_log_vocab():
    """A model that predicts uniformly over the V-1 non-mask tokens has
    NELBO = log(V-1) per token exactly in expectation (E[#masked] = t n)."""
    V = 50258

    class Uniform(torch.nn.Module):
        def forward(self, ids, valid=None):
            lg = torch.zeros(*ids.shape, V)
            lg[..., V - 1] = float("-inf")
            return lg

    x = torch.randint(0, 50000, (8, 64))
    s, n = nelbo_eval(Uniform(), x, V - 1, n_t=64, seed=0)
    assert s / n == pytest.approx(math.log(V - 1), rel=0.03)


def test_loss_only_masks_real_tokens():
    torch.manual_seed(1)
    cfg = ModelConfig(max_len=16, **TINY)
    m = MaskedDiffusionLM(cfg)
    x = torch.randint(0, 1000, (32, 16))
    valid = torch.ones_like(x, dtype=torch.bool)
    valid[:, 10:] = False
    loss, stats = mdlm_loss(m, x, cfg.mask_token_id, valid)
    assert torch.isfinite(loss) and stats["real_tokens"] == 32 * 10
    loss.backward()


def test_exclusion_by_each_key_and_validation_split(tmp_path):
    rng = np.random.default_rng(0)
    recs = _records(300, rng)
    tok = FakeTokenizer()
    entries = [
        ExclusionEntry(role="main", id="doc3"),
        ExclusionEntry(role="main", content_hash=_hash(recs[10]["text"])),
        ExclusionEntry(role="pilot", token_hash=token_hash(tok(recs[20]["text"])["input_ids"])),
        ExclusionEntry(role="timesplit", id="not-in-stream", from_stream=False),
    ]
    idx = _build(tmp_path, recs, entries, val_fraction=0.05)
    ex = idx["exclusion"]
    assert idx["calibration_documents_skipped"] == 3
    assert ex["skipped_by_key"] == {"id": 1, "content": 1, "token": 1}
    assert ex["unmatched_from_stream"] == 0
    assert idx["val_documents"] > 0 and idx["documents"] + idx["val_documents"] == 297
    assert {"doc3", "doc10", "doc20"} <= set(ex["skipped_ids"])


def test_unmatched_stream_document_fails_the_build(tmp_path):
    rng = np.random.default_rng(0)
    recs = _records(50, rng)
    with pytest.raises(RuntimeError, match="never matched"):
        _build(tmp_path, recs, [ExclusionEntry(role="main", id="doc-that-was-never-streamed")])


def test_windows_never_cross_documents(tmp_path):
    rng = np.random.default_rng(1)
    # Each document uses its own letter, so a window's tokens identify its document.
    recs = [{"id": f"d{i}", "text": chr(97 + i % 26) * int(rng.integers(20, 900))} for i in range(60)]
    idx = _build(tmp_path, recs, [ExclusionEntry(role="main", id="d0")], val_fraction=0.0)
    path = tmp_path / "shards" / "index.json"
    ds = DocWindowDataset(path, seq_len=64, seed=0, mode="chunks", min_len=8)
    ids, valid = ds.batch(256)
    for row, v in zip(ids, valid):
        assert len(set(row[v].tolist())) == 1          # one document per window
        assert v[: v.sum()].all() and not v[v.sum():].any()
    pre = DocWindowDataset(path, seq_len=64, mode="prefix")
    assert pre.n_windows == idx["documents"]
    tok, n = pre.window(0)
    assert n == min(64, int(pre.doc_len[0]))
    a, _ = DocWindowDataset(path, seq_len=64).batch(8)
    ds.reseed(1, 0, 5)
    b, _ = ds.batch(8)
    ds.reseed(1, 1, 5)
    c, _ = ds.batch(8)
    assert not np.array_equal(b, c)                      # ranks draw different windows


def test_train_checkpoint_resume_guards_and_g2(tmp_path, monkeypatch):
    from c3.training import train as T
    from c3.training.model import ModelConfig as MC

    rng = np.random.default_rng(2)
    recs = _records(400, rng, 30, 400)
    _build(tmp_path, recs, [ExclusionEntry(role="main", id="doc1")], val_fraction=0.1)
    index = str(tmp_path / "shards" / "index.json")
    mcfg = MC(max_len=64, **TINY)
    cfg = T.TrainConfig(index_path=index, out_dir=str(tmp_path / "run"), seq_len=64,
                        batch_size=4, grad_accum=1, lr=1e-3, warmup_steps=2,
                        max_tokens=4 * 64 * 6, log_every=1, ckpt_every=2, eval_every=2,
                        eval_batches=2, ema_decay=0.9)
    final = T.train(cfg, mcfg)
    state = torch.load(final, map_location="cpu", weights_only=False)
    assert state["model_ema"] is not None and state["config_hash"]
    side = json.loads((tmp_path / "run" / "checkpoint.json").read_text())
    assert side["step"] == state["step"]
    hist = [json.loads(line) for line in (tmp_path / "run" / "history.jsonl").read_text().splitlines()]
    assert any("val_nelbo" in r for r in hist) and any("ema_val_nelbo" in r for r in hist)

    # A different configuration must not resume this checkpoint.
    bad = T.TrainConfig(**{**cfg.__dict__, "lr": 5e-4})
    with pytest.raises(RuntimeError, match="different run"):
        T.train(bad, mcfg)
    # Nor may a proposal-era checkpoint without a config hash.
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    ck = torch.load(tmp_path / "run" / "checkpoint.pt", map_location="cpu", weights_only=False)
    ck.pop("config_hash")
    torch.save(ck, legacy / "checkpoint.pt")
    with pytest.raises(RuntimeError, match="proposal-era"):
        T.train(T.TrainConfig(**{**cfg.__dict__, "out_dir": str(legacy)}), mcfg)

    # Fine-tuning starts from the weights only, in a new directory.
    ft = T.TrainConfig(**{**cfg.__dict__, "out_dir": str(tmp_path / "ft"), "init_from": str(final),
                          "max_tokens": 4 * 64 * 2})
    T.train(ft, mcfg)
    assert (tmp_path / "ft" / "final.pt").exists()

    # Gate G2 reads the history the trainer wrote.
    hist_path = tmp_path / "flat.jsonl"
    hist_path.write_text("\n".join(json.dumps({"step": s, "val_nelbo": 3.0 + 1e-6 * s})
                                   for s in range(0, 1000, 50)))
    out = tmp_path / "g2.json"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "check_g2.py"), "--history",
                    str(hist_path), "--out", str(out)], check=True, capture_output=True, cwd=ROOT)
    assert json.loads(out.read_text())["pass"] is True
    hist_path.write_text("\n".join(json.dumps({"step": s, "val_nelbo": 4.0 - 1e-3 * s})
                                   for s in range(0, 1000, 50)))
    subprocess.run([sys.executable, str(ROOT / "scripts" / "check_g2.py"), "--history",
                    str(hist_path), "--out", str(out)], check=True, capture_output=True, cwd=ROOT)
    assert json.loads(out.read_text())["pass"] is False


def test_budget_only_needs_no_torch_import():
    r = subprocess.run([sys.executable, "-c",
                        "import sys; sys.argv=['x','--budget-only'];"
                        "import runpy; runpy.run_path('scripts/train_110m.py', run_name='__main__')"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0 and "plan v2 default" in r.stdout
