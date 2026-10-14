"""The W&B results logger against a stub wandb module (no network)."""
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class _Run:
    def __init__(self, **kw):
        self.kw, self.summary, self.logged, self.artifacts = kw, {}, [], []
        self.url = "https://wandb.example/run"

    def log(self, d):
        self.logged.append(d)

    def log_artifact(self, a):
        self.artifacts.append(a)

    def finish(self):
        pass


class _Artifact:
    def __init__(self, name, type):
        self.name, self.type, self.files = name, type, []

    def add_file(self, p):
        self.files.append(p)


def _fake_wandb(runs):
    def init(**kw):
        runs.append(_Run(**kw))
        return runs[-1]
    return SimpleNamespace(init=init, Table=lambda columns, data: ("table", columns, data),
                           Artifact=_Artifact, teardown=lambda: None,
                           plot=SimpleNamespace(line_series=lambda **kw: ("line_series", kw)))


def _infill_results(tmp_path):
    rng = np.random.default_rng(0)
    d = tmp_path / "infill"
    d.mkdir()
    lines = []
    for pol, knobs in (("topk_conf", [1, 2, 4, 8]), ("c3set", [0.1, 0.5, 0.9])):
        for i, kv in enumerate(knobs):
            for doc in range(12):
                nf = 128 // kv if pol == "topk_conf" else [100, 40, 12][i]
                key = "k" if pol == "topk_conf" else "alpha"
                lines.append(json.dumps({"config": f"{pol}[{key}={kv}]", "policy": pol, key: kv,
                                         "doc_id": doc, "acc": float(rng.uniform(0.3, 0.5)),
                                         "n_forward": nf, "n_committed": 128,
                                         "tokens_per_step": 128 / nf, "step_all_right": 0.5,
                                         "seconds": 0.1, "tokens_sha1": "x"}))
    (d / "infill.shard0of1.jsonl").write_text("\n".join(lines) + "\n")
    out = tmp_path / "infill.json"
    subprocess.run([sys.executable, "scripts/analyze_infill.py", "--runs", str(d), "--n-boot", "100",
                    "--baselines", "topk_conf", "--out", str(out)], cwd=ROOT, check=True,
                   capture_output=True)
    return out


def test_logger_writes_summary_tables_and_artifact(tmp_path, monkeypatch, capsys):
    runs = []
    monkeypatch.setitem(sys.modules, "wandb", _fake_wandb(runs))
    sys.path.insert(0, str(ROOT / "scripts"))
    import wandb_log_results as w

    res = _infill_results(tmp_path)
    monkeypatch.setattr(sys, "argv", ["x", "--name", "t", "--group", "g", "--infill", str(res)])
    assert w.main() == 0
    out = capsys.readouterr().out
    assert "logging failed" not in out and "[wandb] logged" in out
    run = runs[0]
    assert run.kw["group"] == "g" and run.kw["name"] == "t"
    assert "infill/delta_best@tps2" in run.summary and "infill/delta_best@tps4_hi" in run.summary
    logged = {k for d in run.logged for k in d}
    assert {"infill/curves", "infill/matched", "infill/accuracy_vs_speed"} <= logged
    assert run.artifacts and run.artifacts[0].files == [str(res)]


def test_logger_never_fails_the_job(tmp_path, monkeypatch, capsys):
    def boom(**kw):
        raise RuntimeError("HTTP 403")
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=boom, teardown=lambda: None))
    sys.path.insert(0, str(ROOT / "scripts"))
    import wandb_log_results as w

    res = _infill_results(tmp_path)
    monkeypatch.setattr(sys, "argv", ["x", "--name", "t", "--infill", str(res)])
    assert w.main() == 0
    assert "disabled" in capsys.readouterr().out
