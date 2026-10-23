#!/usr/bin/env python3
"""Build the T9 JSON probe: the spec, the 1,000 evaluation instances, the training-set record.

    git clone --filter=blob:none --no-checkout https://github.com/SchemaStore/schemastore.git runs/probe/schemastore
    git -C runs/probe/schemastore sparse-checkout set --no-cone 'src/schemas/json/*'
    git -C runs/probe/schemastore checkout 5e44c8eee8839250a8f6820aa516961b56ab0cff
    python scripts/build_probe.py --schemastore runs/probe/schemastore --out-dir artifacts/probe

Writes ``spec.json`` (schemas, fields, value lists with their token ids under
the three tokenizers, marginals, pairs), ``eval_instances.jsonl`` (200 per rho,
text included) and ``manifest.json`` (counts, the sha256 of every file and of
the 200k training and 2k validation texts, which are regenerated from the spec
on demand by ``c3.probe.instances.train_instances``).  Needs the GPT-2,
LLaDA-8B and Dream-7B tokenizers (downloaded from the Hub at pinned revisions).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from c3.probe import N_EVAL_PER_RHO, N_SCHEMAS, N_TRAIN, N_VAL, PROBE_SEED
from c3.probe.instances import eval_instances, serialize, train_instances
from c3.probe.spec import build_spec
from c3.provenance import collect as provenance, sha256_file

SCHEMASTORE_REPO = "https://github.com/SchemaStore/schemastore"
SCHEMASTORE_COMMIT = "5e44c8eee8839250a8f6820aa516961b56ab0cff"
TOKENIZERS = ("gpt2", "llada-8b-base", "dream-7b-base")


def load_tokenizers(names=TOKENIZERS) -> dict:
    from transformers import AutoTokenizer

    from c3.traces.models import KNOWN_MODELS
    out = {}
    for name in names:
        if name == "gpt2":
            out[name] = AutoTokenizer.from_pretrained("gpt2")
        else:
            spec = KNOWN_MODELS[name]
            out[name] = AutoTokenizer.from_pretrained(spec["path"], revision=spec["revision"],
                                                      trust_remote_code=True)
    return out


def text_digest(spec, instances) -> tuple[str, int]:
    h = hashlib.sha256()
    n = 0
    for inst in instances:
        text, _ = serialize(spec.schemas[inst.schema], inst.values, inst.rho)
        h.update(text.encode("utf-8") + b"\n")
        n += 1
    return h.hexdigest(), n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--schemastore", required=True, help="SchemaStore checkout at the pinned commit")
    ap.add_argument("--out-dir", default="artifacts/probe")
    ap.add_argument("--seed", type=int, default=PROBE_SEED)
    ap.add_argument("--n-schemas", type=int, default=N_SCHEMAS)
    ap.add_argument("--min-fields", type=int, default=6)
    a = ap.parse_args()

    store = Path(a.schemastore)
    commit = subprocess.run(["git", "-C", str(store), "rev-parse", "HEAD"], capture_output=True,
                            text=True, check=True).stdout.strip()
    if commit != SCHEMASTORE_COMMIT:
        raise SystemExit(f"SchemaStore checkout is at {commit}, not the pinned {SCHEMASTORE_COMMIT}")
    toks = load_tokenizers()
    spec = build_spec(store / "src" / "schemas" / "json", toks, n_schemas=a.n_schemas,
                      min_fields=a.min_fields, seed=a.seed)
    spec.meta["schemastore"] = {"repo": SCHEMASTORE_REPO, "commit": commit,
                                "dir": "src/schemas/json"}
    from c3.traces.models import KNOWN_MODELS
    spec.meta["tokenizers"] = {n: ("gpt2" if n == "gpt2" else
                                   {"path": KNOWN_MODELS[n]["path"],
                                    "revision": KNOWN_MODELS[n]["revision"]}) for n in toks}

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    spec.save(out / "spec.json")

    ev = eval_instances(spec, per_rho=N_EVAL_PER_RHO)
    with open(out / "eval_instances.jsonl", "w", encoding="utf-8") as fh:
        for inst in ev:
            text, spans = serialize(spec.schemas[inst.schema], inst.values, inst.rho)
            fh.write(json.dumps({**inst.to_dict(), "schema_name": spec.schemas[inst.schema].name,
                                 "text": text, "spans": spans}) + "\n")

    train_sha, n_train = text_digest(spec, train_instances(spec, N_TRAIN))
    val_sha, n_val = text_digest(spec, train_instances(spec, N_VAL, label="probe-val"))
    n_fields = [len(s.fields) for s in spec.schemas]
    kinds: dict[str, int] = {}
    for s in spec.schemas:
        for f in s.fields:
            kinds[f.kind] = kinds.get(f.kind, 0) + 1
    manifest = {
        "seed": a.seed, "n_schemas": len(spec.schemas), "rhos": list(spec.rhos),
        "fields_per_schema": {"min": min(n_fields), "max": max(n_fields),
                              "mean": sum(n_fields) / len(n_fields)},
        "field_kinds": kinds,
        "max_tokens": {n: max(s.source["n_tokens"][n] for s in spec.schemas) for n in toks},
        "files": {p: sha256_file(out / p) for p in ("spec.json", "eval_instances.jsonl")},
        "eval": {"n": len(ev), "per_rho": N_EVAL_PER_RHO},
        "train": {"n": n_train, "label": "probe-train", "text_sha256": train_sha},
        "val": {"n": n_val, "label": "probe-val", "text_sha256": val_sha},
        "schemastore": spec.meta["schemastore"], "provenance": provenance()}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"[probe] {len(spec.schemas)} schemas, {sum(n_fields)} fields {kinds}; "
          f"{len(ev)} eval instances, {n_train} train ({train_sha[:12]}), {n_val} val -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
