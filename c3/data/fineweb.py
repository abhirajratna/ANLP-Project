"""The held-out calibration corpus, and the exclusion manifest that keeps it held out.

Proposal Sec. 6: 2,000 documents sampled by fixed seed from FineWeb-Edu 10BT,
truncated to 256 tokens, and removed from the 110M model's training set.  The
removal is the part that is easy to get wrong and fatal if you do -- "held out"
is the only thing that makes the 110M arm's level claims mean anything -- so the
sample writes an explicit manifest of ids and content hashes, and the training
data loader refuses to run without consuming it.

Documents are truncated under each model's own tokeniser and never packed, so
every position lies inside a single document.  Token counts therefore differ
slightly across models, which the state fixed effects absorb.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

DATASET = "HuggingFaceFW/fineweb-edu"
SUBSET = "sample-10BT"


def fineweb_stream(streaming: bool = True):
    """FineWeb-Edu sample-10BT as an iterable of records, in the Hub's order.

    With ``C3_FINEWEB_DIR`` set, reads the parquet files that
    scripts/fetch_fineweb.py downloaded and verified; otherwise streams from the
    Hub.  The Hub config resolves ``sample/10BT/*`` to the same files in sorted
    order, so both yield the same records in the same order, and a corpus sampled
    from either is the committed one (scripts/check_corpus.py confirms it).
    """
    from datasets import load_dataset

    local = os.environ.get("C3_FINEWEB_DIR")
    if not local:
        return load_dataset(DATASET, name=SUBSET, split="train", streaming=streaming)
    files = sorted(str(p) for p in Path(local).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"C3_FINEWEB_DIR={local} holds no .parquet files")
    print(f"[data] reading {len(files)} local parquet files from {local}", flush=True)
    return load_dataset("parquet", data_files=files, split="train", streaming=streaming)


def content_hash(text: str) -> str:
    """Stable content fingerprint, used to exclude documents across tokenisers."""
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


@dataclass
class CorpusSpec:
    n_documents: int = 2000
    max_tokens: int = 256
    seed: int = 20260917
    reservoir_pool: int = 200_000     # documents streamed before sampling
    min_chars: int = 1200             # enough text to reach max_tokens

    def to_dict(self) -> dict:
        return asdict(self)


def sample_documents(spec: CorpusSpec, out_dir: str | Path, streaming: bool = True) -> Path:
    """Reservoir-sample the calibration corpus and write it with its manifest.

    Sampling streams the dataset -- from the verified local copy when
    C3_FINEWEB_DIR is set (see ``fineweb_stream``) -- and reads only the first
    ``reservoir_pool`` qualifying documents, never the whole subset.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(spec.seed)

    ds = fineweb_stream(streaming)
    reservoir: list[dict] = []
    seen = 0
    for rec in ds:
        text = rec.get("text", "")
        if len(text) < spec.min_chars:
            continue
        seen += 1
        item = {"id": rec.get("id", f"idx{seen}"), "text": text,
                "hash": content_hash(text)}
        if len(reservoir) < spec.n_documents:
            reservoir.append(item)
        else:
            j = int(rng.integers(0, seen))
            if j < spec.n_documents:
                reservoir[j] = item
        if seen >= spec.reservoir_pool:
            break

    if seen < spec.reservoir_pool:
        # A stream that ends early (e.g. a partial local copy) samples from a
        # smaller pool: a different corpus, not an error anything else would catch.
        raise RuntimeError(
            f"the stream ended after {seen} qualifying documents, before the "
            f"{spec.reservoir_pool}-document pool was filled")
    if len(reservoir) < spec.n_documents:
        raise RuntimeError(
            f"only {len(reservoir)} documents met the length filter, needed "
            f"{spec.n_documents}; raise reservoir_pool or lower min_chars")

    docs_path = out_dir / "calibration_docs.jsonl"
    with docs_path.open("w") as fh:
        for i, item in enumerate(reservoir):
            fh.write(json.dumps({"doc_id": i, **item}) + "\n")

    manifest = {
        "spec": spec.to_dict(),
        "dataset": DATASET, "subset": SUBSET,
        "n_documents": len(reservoir),
        "documents_streamed": seen,
        "excluded_hashes": sorted(item["hash"] for item in reservoir),
        "excluded_ids": sorted(str(item["id"]) for item in reservoir),
    }
    (out_dir / "exclusion_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[data] {len(reservoir)} docs -> {docs_path}")
    print(f"[data] exclusion manifest -> {out_dir / 'exclusion_manifest.json'}")
    return docs_path


def document_split(n_documents: int, seed: int = 20260917, frac: float = 0.5):
    """The 50/50 document-level split, fixed by seed (Sec. 6)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_documents)
    k = int(round(frac * n_documents))
    return np.sort(perm[:k]), np.sort(perm[k:])


def tokenize_documents(docs_path: str | Path, tokenizer, max_tokens: int = 256):
    """Truncate each document to its first ``max_tokens`` tokens, never packed."""
    out = []
    with Path(docs_path).open() as fh:
        for line in fh:
            rec = json.loads(line)
            ids = tokenizer(rec["text"], truncation=True, max_length=max_tokens,
                            add_special_tokens=False)["input_ids"]
            if len(ids) < max_tokens:
                continue          # short documents would change the field size
            out.append((rec["doc_id"], np.asarray(ids, dtype=np.int64)))
    return out


def load_exclusion(manifest_path: str | Path) -> set[str]:
    """Content hashes the 110M training stream must skip."""
    man = json.loads(Path(manifest_path).read_text())
    return set(man["excluded_hashes"])
