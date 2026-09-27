"""Training-stream preparation for the 110M model.

Three things matter here.

**Exclusion.**  The evaluation documents must never reach the model, because
the 110M arm's level claims rest on them being unseen.  ``c3.training.exclusion``
matches by FineWeb id, raw-text hash and first-256-token hash, and the build
fails if a document sampled from the stream was never matched.

**No packing** (plan v2 T3).  Documents are stored back to back, but every
shard carries the document boundaries (``offsets``), and training windows never
cross them.  A window is either a document's first ``seq_len`` tokens
(``prefix`` mode, exactly the evaluation setting) or any of its consecutive
``seq_len``-token chunks (``chunks`` mode, the default, which uses the whole
corpus); a short tail is padded and the padding is masked out of attention and
the loss.  The proposal-era loader, random crops across document boundaries,
remains as ``TokenShardDataset`` only so old runs can be inspected.

**A validation split.**  A small hash-selected fraction of documents is written
to separate ``val`` shards instead of the training shards -- excluded from
training by construction -- and is what the validation NELBO and gate G2 use.

The tokenised stream lives on node-local scratch and is staged to /share1,
which holds a few large files well (see slurm/CLUSTER.md).
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .exclusion import Exclusion, token_hash

# The corpus.  Kept here too so this module does not need c3.data at import time.
DATASET = "HuggingFaceFW/fineweb-edu"
SUBSET = "sample-10BT"


def _is_val(key: str, val_fraction: float) -> bool:
    if val_fraction <= 0:
        return False
    h = int.from_bytes(hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest(), "little")
    return (h % 1_000_000) < val_fraction * 1_000_000


class _ShardWriter:
    """Buffers documents into uint16 shards with a per-shard boundary array."""

    def __init__(self, out_dir: Path, prefix: str, shard_tokens: int):
        self.out_dir, self.prefix, self.shard_tokens = out_dir, prefix, shard_tokens
        self.buf = np.empty(shard_tokens, dtype=np.uint16)
        self.written = 0
        self.offsets: list[int] = [0]
        self.shards: list[dict] = []
        self.total = 0
        self.n_docs = 0

    def add(self, ids: np.ndarray) -> None:
        if ids.size == 0:
            return
        if ids.size > self.shard_tokens:
            ids = ids[: self.shard_tokens]
        if self.written + ids.size > self.shard_tokens:
            self.flush()
        self.buf[self.written:self.written + ids.size] = ids
        self.written += ids.size
        self.offsets.append(self.written)
        self.total += int(ids.size)
        self.n_docs += 1

    def flush(self) -> None:
        if self.written == 0:
            return
        k = len(self.shards)
        p = self.out_dir / f"{self.prefix}_{k:04d}.bin"
        o = self.out_dir / f"{self.prefix}_{k:04d}.offsets.npy"
        self.buf[: self.written].tofile(p)
        np.save(o, np.asarray(self.offsets, dtype=np.int64))
        self.shards.append({"path": str(p), "offsets": str(o), "tokens": int(self.written),
                            "documents": len(self.offsets) - 1})
        print(f"[data] {self.prefix} shard {k}: {self.written} tokens, "
              f"{len(self.offsets) - 1} docs -> {p}", flush=True)
        self.written = 0
        self.offsets = [0]


def build_token_shards(
    out_dir: str | Path,
    exclusion_manifests,
    tokenizer,
    target_tokens: int = 10_000_000_000,
    shard_tokens: int = 200_000_000,
    seq_len: int = 256,
    val_fraction: float = 0.0005,
    report_every: int = 20_000,
    records=None,
    content_hash_fn=None,
    fail_on_unmatched: bool = True,
) -> dict:
    """Stream, tokenise and shard the training corpus, minus the evaluation docs.

    ``records`` (an iterable of dicts with ``text`` and ``id``) replaces the
    HuggingFace stream in tests.  Writes ``train_*`` and ``val_*`` shards with
    boundary arrays and an ``index.json`` whose ``exclusion`` block is the audit
    trail that the held-out documents were held out.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifests = [exclusion_manifests] if isinstance(exclusion_manifests, (str, Path)) \
        else list(exclusion_manifests)
    excl = Exclusion.load(*manifests)
    if content_hash_fn is None:
        from ..data.fineweb import content_hash as content_hash_fn
    if records is None:
        from datasets import load_dataset
        records = load_dataset(DATASET, name=SUBSET, split="train", streaming=True)

    train_w = _ShardWriter(out_dir, "train", shard_tokens)
    val_w = _ShardWriter(out_dir, "val", min(shard_tokens, 50_000_000))
    t_start = time.time()
    n_seen = n_skipped = 0
    skipped_ids: list[str] = []

    def absorb(batch):
        nonlocal n_skipped
        texts = [r["text"] for r in batch]
        enc = tokenizer(texts, add_special_tokens=False)["input_ids"]
        for rec, ids in zip(batch, enc):
            rid = rec.get("id")
            why = excl.check(rec_id=rid, content=content_hash_fn(rec["text"]),
                             tokens=token_hash(ids))
            if why:
                n_skipped += 1
                if rid is not None and len(skipped_ids) < 100_000:
                    skipped_ids.append(str(rid))
                continue
            arr = np.asarray(ids, dtype=np.uint16)
            key = str(rid) if rid is not None else rec["text"][:512]
            (val_w if _is_val(key, val_fraction) else train_w).add(arr)
        return train_w.total >= target_tokens

    batch: list[dict] = []
    done = False
    for rec in records:
        if not rec.get("text"):
            continue
        batch.append(rec)
        n_seen += 1
        if len(batch) < 512:
            continue
        # Batch the tokeniser call: the Rust tokenisers release the GIL and
        # parallelise across a batch, which a per-document call cannot do.
        done = absorb(batch)
        batch = []
        if report_every and n_seen % report_every < 512:
            el = time.time() - t_start
            rate = train_w.total / max(1e-9, el)
            print(f"[data] {n_seen:,} docs | {train_w.total/1e9:.3f}B/{target_tokens/1e9:.1f}B tok "
                  f"| {rate/1e6:.2f}M tok/s | {n_skipped} excluded", flush=True)
        if done:
            break
    if not done and batch:
        absorb(batch)
    train_w.flush()
    val_w.flush()

    unmatched = excl.unmatched(from_stream_only=True)
    index = {"dataset": DATASET, "subset": SUBSET, "seq_len": seq_len, "format": "doc-offsets-v2",
             "total_tokens": int(train_w.total), "documents": int(train_w.n_docs),
             "val_tokens": int(val_w.total), "val_documents": int(val_w.n_docs),
             "val_fraction": val_fraction, "documents_streamed": n_seen,
             "calibration_documents_skipped": n_skipped,
             "exclusion": {**excl.summary(), "manifests": [str(m) for m in manifests],
                           "unmatched_from_stream": len(unmatched),
                           "skipped_ids": skipped_ids},
             "shards": train_w.shards, "val_shards": val_w.shards}
    (out_dir / "index.json").write_text(json.dumps(index, indent=2))
    print(f"[data] done: {train_w.total/1e9:.2f}B train tokens, {val_w.total/1e6:.1f}M val tokens, "
          f"{n_skipped} excluded; exclusion summary {excl.summary()}", flush=True)
    if unmatched and fail_on_unmatched:
        raise RuntimeError(
            f"{len(unmatched)} manifest documents sampled from this stream were never matched "
            f"(e.g. role={unmatched[0].role}, id={unmatched[0].id}).  Either the stream stopped "
            f"before reaching them or the keys do not line up; the shards may contain "
            f"evaluation documents.  index.json was written for inspection.")
    return index


class DocWindowDataset:
    """Windows of at most ``seq_len`` tokens that never cross a document boundary.

    ``mode="chunks"`` samples uniformly over every consecutive ``seq_len`` chunk
    of every document; ``mode="prefix"`` only over each document's first chunk
    (the evaluation setting).  Chunks shorter than ``min_len`` are skipped.
    ``batch`` returns ``(ids, valid)``: ``ids`` padded with ``pad_id`` and
    ``valid`` marking real tokens.  Sampling avoids materialising the window
    table: a cumulative window count per document is enough to draw a window
    index uniformly and locate it by binary search.
    """

    def __init__(self, index_path: str | Path, seq_len: int = 256, seed: int = 0,
                 mode: str = "chunks", split: str = "train", min_len: int = 16,
                 pad_id: int = 50256):
        if mode not in {"chunks", "prefix"}:
            raise ValueError(f"mode must be 'chunks' or 'prefix', got {mode!r}")
        index = json.loads(Path(index_path).read_text())
        if index.get("format") != "doc-offsets-v2":
            raise ValueError(f"{index_path} was built without document boundaries; rebuild the "
                             f"shards with this version (packed shards cannot avoid packing)")
        shards = index["shards" if split == "train" else "val_shards"]
        if not shards:
            raise ValueError(f"no {split} shards listed in {index_path}")
        self.seq_len, self.mode, self.min_len, self.pad_id = seq_len, mode, min_len, pad_id
        self.arrays, self.doc_start, self.doc_len, self.doc_shard = [], [], [], []
        for k, s in enumerate(shards):
            base = Path(index_path).parent
            path = Path(s["path"]); path = path if path.exists() else base / path.name
            opath = Path(s["offsets"]); opath = opath if opath.exists() else base / opath.name
            self.arrays.append(np.memmap(path, dtype=np.uint16, mode="r"))
            off = np.load(opath)
            self.doc_start.append(off[:-1])
            self.doc_len.append(np.diff(off))
            self.doc_shard.append(np.full(off.size - 1, k, dtype=np.int32))
        self.doc_start = np.concatenate(self.doc_start)
        self.doc_len = np.concatenate(self.doc_len)
        self.doc_shard = np.concatenate(self.doc_shard)
        if mode == "prefix":
            n_win = (self.doc_len >= min_len).astype(np.int64)
        else:
            full = self.doc_len // seq_len
            tail = self.doc_len - full * seq_len
            n_win = full + (tail >= min_len)
        self.cum = np.cumsum(n_win)
        self.n_windows = int(self.cum[-1]) if self.cum.size else 0
        if self.n_windows == 0:
            raise ValueError("no windows of at least min_len tokens")
        self.real_tokens = int(np.minimum(self.doc_len, seq_len).sum()) if mode == "prefix" \
            else int(self.doc_len.sum())
        self.rng = np.random.default_rng(seed)

    def reseed(self, *key) -> None:
        self.rng = np.random.default_rng([int(k) for k in key])

    def window(self, w: int) -> tuple[np.ndarray, int]:
        """Tokens of global window ``w`` and its length."""
        d = int(np.searchsorted(self.cum, w, side="right"))
        k = w - (int(self.cum[d - 1]) if d > 0 else 0)
        start = int(self.doc_start[d]) + k * self.seq_len
        n = int(min(self.seq_len, self.doc_start[d] + self.doc_len[d] - start))
        return self.arrays[self.doc_shard[d]][start:start + n], n

    def batch(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        ids = np.full((batch_size, self.seq_len), self.pad_id, dtype=np.int64)
        valid = np.zeros((batch_size, self.seq_len), dtype=bool)
        for b, w in enumerate(self.rng.integers(0, self.n_windows, size=batch_size)):
            toks, n = self.window(int(w))
            ids[b, :n] = toks
            valid[b, :n] = True
        return ids, valid

    def fixed_batches(self, n_batches: int, batch_size: int, seed: int = 0):
        """Deterministic batches for evaluation (the same every call)."""
        rng = np.random.default_rng(seed)
        picks = rng.choice(self.n_windows, size=min(self.n_windows, n_batches * batch_size),
                           replace=False)
        for i in range(0, picks.size, batch_size):
            chunk = picks[i:i + batch_size]
            ids = np.full((chunk.size, self.seq_len), self.pad_id, dtype=np.int64)
            valid = np.zeros((chunk.size, self.seq_len), dtype=bool)
            for b, w in enumerate(chunk):
                toks, n = self.window(int(w))
                ids[b, :n] = toks
                valid[b, :n] = True
            yield ids, valid


class TokenShardDataset:
    """PROPOSAL-ERA loader: fixed-length crops drawn across document boundaries.

    This is packing, which plan v2 forbids for the 110M model; it is kept only
    to read shards built before document boundaries were recorded.
    """

    def __init__(self, index_path: str | Path, seq_len: int = 256, seed: int = 0):
        index = json.loads(Path(index_path).read_text())
        if not index["shards"]:
            raise ValueError(f"no shards listed in {index_path}")
        self.seq_len = seq_len
        self.arrays = [np.memmap(s["path"], dtype=np.uint16, mode="r") for s in index["shards"]]
        self.sizes = np.array([a.size for a in self.arrays], dtype=np.int64)
        self.weights = self.sizes / self.sizes.sum()
        self.total_tokens = int(self.sizes.sum())
        self.rng = np.random.default_rng(seed)

    def batch(self, batch_size: int) -> np.ndarray:
        out = np.empty((batch_size, self.seq_len), dtype=np.int64)
        which = self.rng.choice(len(self.arrays), size=batch_size, p=self.weights)
        for b, w in enumerate(which):
            a = self.arrays[w]
            start = int(self.rng.integers(0, max(1, a.size - self.seq_len)))
            out[b] = a[start:start + self.seq_len].astype(np.int64)
        return out
