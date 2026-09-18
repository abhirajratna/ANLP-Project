"""Training-stream preparation for the 110M model.

Two things matter here.  The calibration documents must be removed, because the
110M arm's level claims rest entirely on those documents being genuinely unseen.
And the tokenised stream has to live on node-local ``/scratch`` (1.8 TB on Ada's
gnodes) rather than in ``/share1``, where this project has only tens of GB of
quota against a corpus that is 20 GB per 10B tokens at uint16.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..data.fineweb import DATASET, SUBSET, content_hash, load_exclusion


def build_token_shards(
    out_dir: str | Path,
    exclusion_manifest: str | Path,
    tokenizer,
    target_tokens: int = 6_000_000_000,
    shard_tokens: int = 200_000_000,
    seq_len: int = 256,
    eos_id: int | None = None,
    report_every: int = 20_000,
) -> dict:
    """Stream, tokenise and shard the training corpus, minus the calibration docs.

    Writes ``uint16`` shards (GPT-2's 50257-token vocabulary fits) plus an index
    recording how many documents were skipped, which is the audit trail that the
    held-out set was actually held out.
    """
    from datasets import load_dataset

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    excluded = load_exclusion(exclusion_manifest)
    if not excluded:
        raise ValueError("exclusion manifest is empty; refusing to build a training "
                         "stream that may contain the calibration corpus")

    shard_idx, written, total, n_docs, n_skipped = 0, 0, 0, 0, 0
    t_start = time.time()
    buf = np.empty(shard_tokens, dtype=np.uint16)
    shards: list[dict] = []

    def flush() -> None:
        nonlocal shard_idx, written
        if written == 0:
            return
        p = out_dir / f"train_{shard_idx:04d}.bin"
        buf[:written].tofile(p)
        shards.append({"path": str(p), "tokens": int(written)})
        print(f"[data] shard {shard_idx}: {written} tokens -> {p}", flush=True)
        shard_idx += 1
        written = 0

    ds = load_dataset(DATASET, name=SUBSET, split="train", streaming=True)

    def absorb(ids_list) -> bool:
        """Copy a batch of tokenised documents into the shard buffer.

        Filling the buffer with ``for tok in ids: buf[written] = tok`` means one
        Python iteration per token, which is six billion of them for a 6B-token
        corpus and takes longer than the training run it feeds. Slicing whole
        arrays in keeps it to a few thousand iterations.
        """
        nonlocal written, total
        for ids in ids_list:
            arr = np.asarray(ids, dtype=np.uint16)
            pos = 0
            while pos < arr.size:
                space = shard_tokens - written
                take = min(space, arr.size - pos)
                buf[written:written + take] = arr[pos:pos + take]
                written += take
                total += take
                pos += take
                if written == shard_tokens:
                    flush()
                if total >= target_tokens:
                    return True
        return False

    batch_texts: list[str] = []
    done = False
    for rec in ds:
        text = rec.get("text", "")
        if not text:
            continue
        if content_hash(text) in excluded:
            n_skipped += 1
            continue
        batch_texts.append(text)
        n_docs += 1
        if len(batch_texts) < 512:
            continue
        # Batch the tokeniser call: the Rust tokenisers release the GIL and
        # parallelise across a batch, which a per-document call cannot do.
        enc = tokenizer(batch_texts, add_special_tokens=False)["input_ids"]
        if eos_id is not None:
            enc = [e + [eos_id] for e in enc]
        batch_texts = []
        if absorb(enc):
            done = True
            break
        if report_every and n_docs % report_every < 512:
            el = time.time() - t_start
            rate = total / max(1e-9, el)
            eta = (target_tokens - total) / max(1.0, rate)
            print(f"[data] {n_docs:,} docs | {total/1e9:.3f}B/{target_tokens/1e9:.1f}B tokens "
                  f"| {rate/1e6:.2f}M tok/s | {n_skipped} calibration docs skipped "
                  f"| eta {eta/60:.0f} min", flush=True)

    if not done and batch_texts:
        enc = tokenizer(batch_texts, add_special_tokens=False)["input_ids"]
        if eos_id is not None:
            enc = [e + [eos_id] for e in enc]
        absorb(enc)

    flush()

    index = {"dataset": DATASET, "subset": SUBSET, "seq_len": seq_len,
             "total_tokens": int(total), "documents": n_docs,
             "calibration_documents_skipped": n_skipped,
             "exclusion_manifest": str(exclusion_manifest), "shards": shards}
    (out_dir / "index.json").write_text(json.dumps(index, indent=2))
    print(f"[data] done: {total/1e9:.2f}B tokens, {n_skipped} calibration docs skipped")
    return index


class TokenShardDataset:
    """Fixed-length crops drawn uniformly from memory-mapped uint16 shards."""

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
