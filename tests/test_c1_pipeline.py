"""C1 sweep plumbing: zero-copy merge, run validation, closed-form design counts."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from c3.assemble import load_traces, trace_meta
from c3.traces.collect import CollectConfig, Document, collect_shard
from c3.traces.models import ToyDiffusionLM
from c3.traces.shards import MergeError, dataset_files, merge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import validate_run  # noqa: E402


def _run_two_shards(tmp_path, n_docs=4, length=24):
    rng = np.random.default_rng(0)
    docs = [Document(i, rng.integers(0, 60, length)) for i in range(n_docs)]
    cfg = CollectConfig(topk=5)
    for s in range(2):
        collect_shard(ToyDiffusionLM(), docs, cfg, tmp_path / "run", s, 2, part_docs=1,
                      progress_every=0)
    return tmp_path / "run"


def test_zero_copy_merge_reads_back_the_same_trace(tmp_path):
    stem = _run_two_shards(tmp_path)
    copied = load_traces(merge(stem, tmp_path / "copy.parquet"))
    man = merge(stem, tmp_path / "run.dataset.json")
    viewed = load_traces(man)
    key = ["state_id", "position"]
    a = copied.sort_values(key).reset_index(drop=True)
    b = viewed.sort_values(key).reset_index(drop=True)
    assert a.equals(b)
    meta = trace_meta(man)
    assert meta["format"] == "c3-trace-dataset-v1" and meta["n_docs"] == 4
    assert sum(f["n_rows"] for f in meta["files"]) == len(viewed) == meta["n_rows"]


def test_a_manifest_notices_a_changed_or_missing_part(tmp_path):
    stem = _run_two_shards(tmp_path)
    man = merge(stem, tmp_path / "run.dataset.json")
    files = dataset_files(man, verify=True)
    t = pq.read_table(files[0])
    pq.write_table(t.slice(0, t.num_rows - 1), files[0])          # a part rewritten later
    with pytest.raises(MergeError, match="changed"):
        dataset_files(man, verify=True)
    files[0].unlink()
    with pytest.raises(MergeError, match="missing"):
        dataset_files(man)


def test_closed_form_counts_for_the_plan_design():
    e = validate_run.expected_rows(256, CollectConfig().to_dict())
    assert e["decoder"] == e["uniform_matched"] == 16512 + 8320 + 4224
    assert e["passes"] == 466                       # slurm/CLUSTER.md
    assert e["uniform_mean"] == pytest.approx(2 * 256 * 4.5)
    no_match = CollectConfig(include_uniform_matched=False).to_dict()
    assert validate_run.expected_rows(256, no_match)["passes"] == 242


def test_validate_run_passes_a_clean_run_and_fails_a_corrupted_one(tmp_path):
    stem = _run_two_shards(tmp_path)
    man = merge(stem, tmp_path / "run.dataset.json")
    out = tmp_path / "c1" / "run.json"
    r = subprocess.run([sys.executable, "scripts/validate_run.py", "--dataset", man, "--run", "run",
                        "--out", out], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    res = json.loads(out.read_text())
    assert res["all_ok"] and res["summary"]["documents"] == 4
    assert res["checks"]["forward_passes"]["recorded"] == 4 * 60
    assert "not results" in res["note"]

    # Drop one decoder row from a part, and re-point the manifest at the new hash.
    f = dataset_files(man)[0]
    t = pq.read_table(f)
    kind = np.asarray(t.column("mask_kind").to_pylist())
    keep = np.ones(t.num_rows, bool)
    keep[np.flatnonzero(kind == "decoder")[0]] = False
    pq.write_table(t.filter(keep), f)
    m = json.loads(Path(man).read_text())
    from c3.provenance import sha256_file
    for e in m["files"]:
        if (Path(man).parent / e["file"]).resolve() == f.resolve():
            e["sha256"] = sha256_file(f)
    Path(man).write_text(json.dumps(m))
    r = subprocess.run([sys.executable, "scripts/validate_run.py", "--dataset", man, "--run", "run",
                        "--out", out], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 1
    bad = json.loads(out.read_text())["checks"]["rows_per_document"]
    assert not bad["ok"] and "decoder" in bad["first_bad"][0]["problems"][0]


class _OverflowToy(ToyDiffusionLM):
    """Overflows (inf logits) whenever document 2's first token is visible."""

    bad_first: int = -1

    def logits(self, input_ids):
        lg = super().logits(input_ids)
        ids = input_ids.detach().cpu().numpy()
        for b in range(ids.shape[0]):
            if ids[b, 0] == self.bad_first:
                lg[b, 0, 0] = float("inf")
        return lg


def test_an_overflowing_document_is_skipped_flagged_and_fails_validation(tmp_path):
    rng = np.random.default_rng(0)
    docs = [Document(i, rng.integers(0, 60, 24)) for i in range(4)]
    docs[2].ids[0] = 61                          # an id no other document starts with
    model = _OverflowToy()
    model.bad_first = 61
    cfg = CollectConfig(topk=5)
    with pytest.raises(FloatingPointError):
        collect_shard(model, docs, cfg, tmp_path / "a" / "run", 0, 1, part_docs=1, progress_every=0)
    meta = json.loads(collect_shard(model, docs, cfg, tmp_path / "b" / "run", 0, 1, part_docs=1,
                                    progress_every=0, on_nonfinite="skip").read_text())
    assert meta["complete"] and meta["doc_ids_failed"] == [2]
    assert sorted(meta["doc_ids_done"]) == [0, 1, 3]
    # A resume does not retry it.
    again = json.loads(collect_shard(model, docs, cfg, tmp_path / "b" / "run", 0, 1, resume=True,
                                     part_docs=1, progress_every=0,
                                     on_nonfinite="skip").read_text())
    assert again["doc_ids_failed"] == [2] and again["complete"]
    man = merge(tmp_path / "b" / "run", tmp_path / "b" / "run.dataset.json")
    m = trace_meta(man)
    assert m["doc_ids_failed"] == [2] and m["doc_ids"] == [0, 1, 3]
    out = tmp_path / "v.json"
    r = subprocess.run([sys.executable, "scripts/validate_run.py", "--dataset", man, "--run", "b",
                        "--out", out], cwd=ROOT, capture_output=True, text=True)
    res = json.loads(out.read_text())
    assert r.returncode == 1 and not res["checks"]["no_flagged_nonfinite_passes"]["ok"]
    assert res["checks"]["forward_passes"]["ok"]           # passes on doc 2 are not counted


def test_single_shard_runs_can_merge_zero_copy_or_not_at_all(tmp_path):
    common = [sys.executable, "scripts/collect_traces.py", "--model", "toy", "--device", "cpu",
              "--limit", "2", "--max-tokens", "16", "--mask-rates", "0.5", "--decoder-k", "8",
              "--resume"]
    subprocess.run([*common, "--merge-as", "dataset", "--out", tmp_path / "a"], cwd=ROOT, check=True,
                   capture_output=True)
    assert (tmp_path / "a.dataset.json").exists() and not (tmp_path / "a.parquet").exists()
    assert sorted(load_traces(tmp_path / "a.dataset.json").doc_id.unique()) == [0, 1]
    subprocess.run([*common, "--merge-as", "none", "--out", tmp_path / "b"], cwd=ROOT, check=True,
                   capture_output=True)
    assert not (tmp_path / "b.dataset.json").exists() and not (tmp_path / "b.parquet").exists()


def test_dev_subset_holds_only_fit_half_documents(tmp_path):
    stem = _run_two_shards(tmp_path, n_docs=6)
    man = merge(stem, tmp_path / "run.dataset.json")
    split = tmp_path / "split.json"
    split.write_text(json.dumps({str(i): i % 2 for i in range(6)}))     # 0, 2, 4 are fit
    out = tmp_path / "dev" / "trace.parquet"
    subprocess.run([sys.executable, "scripts/make_dev_subset.py", "--dataset", man,
                    "--manifest", split, "--out-trace", out,
                    "--out-manifest", tmp_path / "dev" / "m.json"], cwd=ROOT, check=True,
                   capture_output=True)
    docs = sorted(set(pq.read_table(out, columns=["doc_id"]).column("doc_id").to_pylist()))
    assert docs == [0, 2, 4]
    dev = json.loads((tmp_path / "dev" / "m.json").read_text())
    assert set(dev) == {"0", "2", "4"} and set(dev.values()) <= {0, 1}
    assert "DEVELOPMENT" in json.loads(out.with_suffix(".meta.json").read_text())["note"]


def test_max_doc_tokens_excludes_and_records_long_documents(tmp_path):
    common = [sys.executable, "scripts/collect_traces.py", "--model", "toy", "--device", "cpu",
              "--limit", "2", "--max-tokens", "16", "--mask-rates", "0.5", "--decoder-k", "8",
              "--resume", "--merge-as", "none"]
    subprocess.run([*common, "--max-doc-tokens", "15", "--out", tmp_path / "a"], cwd=ROOT,
                   check=True, capture_output=True)
    meta = json.loads((tmp_path / "a.shard0of1.meta.json").read_text())
    assert meta["excluded_too_long"] == [{"doc_id": 0, "tokens": 16}, {"doc_id": 1, "tokens": 16}]
    assert meta["doc_ids_expected"] == [] and meta["max_doc_tokens"] == 15
    subprocess.run([*common, "--max-doc-tokens", "16", "--out", tmp_path / "b"], cwd=ROOT,
                   check=True, capture_output=True)
    meta = json.loads((tmp_path / "b.shard0of1.meta.json").read_text())
    assert meta["excluded_too_long"] == [] and meta["doc_ids_expected"] == [0, 1]
