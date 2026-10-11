#!/usr/bin/env python3
"""Tokenise the probe's training and validation instances into 110M training shards.

    python scripts/build_probe_shards.py --spec artifacts/probe/spec.json --out-dir <scratch>/probe-shards

One document per instance (GPT-2 tokens, at most 256), in the trainer's
document-aligned format (``c3.training.data``), so fine-tuning reuses
scripts/train_110m.py with ``--init-from`` and ``--window-mode prefix``: every
training window is one whole instance.  The 200k training instances are
regenerated from the spec and checked against the text digest in
artifacts/probe/manifest.json before anything is written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from c3.probe import N_TRAIN, N_VAL
from c3.probe.instances import ProbeSpec, serialize, train_instances
from c3.provenance import sha256_file
from c3.training.data import _ShardWriter


def write(spec, label: str, n: int, tok, writer, batch: int = 4096) -> str:
    h = hashlib.sha256()
    texts = []

    def flush():
        for ids in tok(texts, add_special_tokens=False)["input_ids"]:
            if len(ids) > 256:
                raise ValueError(f"an instance has {len(ids)} > 256 tokens")
            writer.add(np.asarray(ids, dtype=np.uint16))
        texts.clear()

    for inst in train_instances(spec, n, label=label):
        text, _ = serialize(spec.schemas[inst.schema], inst.values, inst.rho)
        h.update(text.encode("utf-8") + b"\n")
        texts.append(text)
        if len(texts) == batch:
            flush()
    flush()
    writer.flush()
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", default="artifacts/probe/spec.json")
    ap.add_argument("--manifest", default="artifacts/probe/manifest.json")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    spec = ProbeSpec.load(a.spec)
    manifest = json.loads(Path(a.manifest).read_text())
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_w = _ShardWriter(out, "train", 50_000_000)
    val_w = _ShardWriter(out, "val", 5_000_000)
    train_sha = write(spec, "probe-train", N_TRAIN, tok, train_w)
    val_sha = write(spec, "probe-val", N_VAL, tok, val_w)
    for name, got in (("train", train_sha), ("val", val_sha)):
        if got != manifest[name]["text_sha256"]:
            raise SystemExit(f"{name} texts do not match the manifest ({got} vs "
                             f"{manifest[name]['text_sha256']}); the spec or generator changed")
    index = {"dataset": "c3-json-probe", "format": "doc-offsets-v2", "seq_len": 256,
             "total_tokens": train_w.total, "documents": train_w.n_docs,
             "val_tokens": val_w.total, "val_documents": val_w.n_docs, "val_fraction": None,
             "calibration_documents_skipped": 0, "exclusion": {},
             "spec_sha256": sha256_file(a.spec), "train_text_sha256": train_sha,
             "val_text_sha256": val_sha, "shards": train_w.shards, "val_shards": val_w.shards}
    (out / "index.json").write_text(json.dumps(index, indent=2))
    print(f"[probe-shards] {train_w.n_docs} train docs ({train_w.total/1e6:.1f}M tokens), "
          f"{val_w.n_docs} val docs -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
