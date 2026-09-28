"""The trace harness.

What is written to disk is the *position-level* table -- one row per (masked
state, masked position) carrying the plan v2 Sec. 3.4 fields.  Pools are drawn
later, at analysis time, from that table.

Keeping pools out of the stored artefact matters for two reasons.  It shrinks
the release by two orders of magnitude (every pool membership materialised would
be hundreds of millions of rows), and it lets the pool design be re-drawn --
different sizes, different counts, different seeds -- without re-running a
single forward pass.  That is what makes the proposal's promise to release the
traces, so the audit can be repeated without repeating the compute, actually
cheap to keep.

Schema v2 (Sec. 3.4) adds to v1 everything later tasks read that a forward pass
already computed: ``p_true`` (C2 needs CE), entropy, margin, the top-20 ids and
probabilities (temperature ablation, C3b), the change in ell since the previous
decoder step (KLASS/SWD's stability signal), and which positions the decoder
committed and at what rank (the native gap, T5 step 3).  Collecting them later
would mean running every forward pass twice.  v1 files still load
(``c3.assemble.load_traces``).

Every decoder state is scored by exactly one forward pass, and that pass both
fills the state's rows and chooses its commits.  States of one document are
scored in batches of ``batch_size`` (one document at a time, so there is no
padding).  Batch composition can change GPU reduction order, so a trace is only
bitwise reproducible at the batch size it was collected with; the floor must be
measured at that batch size.

Large runs are sharded by ``doc_id % num_shards`` and written as parts of
``part_docs`` documents.  A part is written to a temporary name and renamed when
complete, so a crash loses at most one part and ``resume`` skips every document
already in a complete part.  ``c3.traces.shards.merge`` joins the parts.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..rng import generator
from .masks import (DECODER_K, MASKS_PER_RATE, UNIFORM_RATES, MaskedState,
                    TeacherForcedTrajectory, matched_uniform_masks, maskable_mask,
                    subsample_evenly, uniform_masks)

SCHEMA_VERSION = 2

TRACE_COLUMNS = (
    "model", "variant", "revision", "precision", "dataset", "git_commit", "seed",
    "mask_kind", "policy", "k", "target_rate", "mask_rate", "step", "n_masked",
    "doc_id", "state_id", "state_idx", "doc_len", "span_start", "span_len",
    "position", "top1", "conf", "ell", "p_true", "entropy", "margin",
    "topk_ids", "topk_p", "delta_ell", "agree", "committed", "commit_rank",
)
"""Schema v2, one row per (masked state, masked position)."""

# A state id is doc_id * STATE_STRIDE + state index.  The proposal-era stride of
# 1000 would collide once a document has 1000 states, which recording every
# decoder state for several k and decoders reaches.
STATE_STRIDE = 100_000


@dataclass
class CollectConfig:
    mask_rates: tuple = UNIFORM_RATES
    masks_per_rate: int = MASKS_PER_RATE
    decoder_k: tuple = DECODER_K
    decoder_policies: tuple = ("topk_conf",)
    decoder_states: int | None = None   # None: every state (plan v2); n: n spread evenly (v1)
    min_masked: int = 1                 # decoder trajectory: smallest state recorded
    uniform_min_masked: int = 1         # uniform masks stay Bernoulli; pools skip m > |M|
    include_uniform: bool = True
    include_decoder: bool = True
    include_uniform_matched: bool = True
    topk: int = 20
    batch_size: int = 1
    eps: float = 1e-6
    seed: int = 20260917

    def to_dict(self) -> dict:
        """JSON-normalised (tuples become lists), so stored and live configs compare equal."""
        return json.loads(json.dumps(asdict(self)))


PRESETS = {
    # Plan v2 T2: 20 pilot documents on LLaDA-8B-Base, uniform masks at 5 rates
    # and one teacher-forced top-k trajectory with k = 4, full schema.  The five
    # rates are the suggested {0.1, 0.3, 0.5, 0.7, 0.9}; the plan does not list them.
    "t2-pilot": {
        "model": "llada-8b-base",
        "config": {"mask_rates": (0.1, 0.3, 0.5, 0.7, 0.9), "masks_per_rate": 2,
                   "decoder_k": (4,), "decoder_policies": ("topk_conf",),
                   "decoder_states": None, "min_masked": 1,
                   "include_uniform_matched": False, "topk": 20},
    },
}


@dataclass
class Document:
    """A token sequence and the region that may be masked.

    For raw text the region is the whole document; for a task it is the
    reference answer, so the question stays visible (T1 step 2).
    """

    doc_id: int
    ids: np.ndarray
    span: tuple | None = None      # [start, end) of the maskable region
    dataset: str = "fineweb"
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.ids = np.asarray(self.ids, dtype=np.int64)
        if self.span is None:
            self.span = (0, int(self.ids.size))
        s, e = int(self.span[0]), int(self.span[1])
        if not 0 <= s < e <= self.ids.size:
            raise ValueError(f"doc {self.doc_id}: span {self.span} outside [0, {self.ids.size})")
        self.span = (s, e)


def as_document(item, dataset: str = "fineweb") -> Document:
    """Accept a Document, (doc_id, ids) or (doc_id, ids, (start, end))."""
    if isinstance(item, Document):
        return item
    if len(item) == 2:
        return Document(int(item[0]), item[1], dataset=dataset)
    return Document(int(item[0]), item[1], span=tuple(item[2]), dataset=dataset)


def shard_documents(documents, shard: int, num_shards: int) -> list:
    """Documents with ``doc_id % num_shards == shard``.

    Keyed on the document id, not on list position, so a shard's membership does
    not change with ``--limit`` or with which documents a tokeniser dropped.
    """
    if not 0 <= shard < num_shards:
        raise ValueError(f"shard {shard} is not in [0, {num_shards})")
    return [d for d in documents if as_document(d).doc_id % num_shards == shard]


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def _score_masks(model, x0: np.ndarray, masks: list, cfg: CollectConfig,
                 counter: dict) -> list[dict]:
    """Per-position statistics for each mask, in position order, batched."""
    import torch

    from ..decode.scoring import position_stats
    from .models import check_finite_logits

    out: list[dict] = []
    dev = getattr(model, "input_device", None)
    for s in range(0, len(masks), max(1, cfg.batch_size)):
        batch = np.stack(masks[s:s + cfg.batch_size])
        ids = np.repeat(x0[None, :], batch.shape[0], axis=0)
        truth = ids.copy()
        ids[batch] = model.mask_token_id
        t = torch.as_tensor(ids, dtype=torch.long)
        if dev is not None:
            t = t.to(dev)
        with torch.no_grad():
            lg = model.logits(t)
            check_finite_logits(lg, model.name, getattr(model, "precision", "na"))
            # 256 rows at a time: at a 126k vocabulary each float32 temporary is
            # ~130 MB, which fits beside 9 GiB of weights on an 11 GB card.
            st = position_stats(lg, batch, model.mask_token_id, true_ids=truth, topk=cfg.topk,
                                chunk=256)
        counter["forward_passes"] += 1
        rows = st.pop("rows")
        st.pop("cols")
        bounds = np.flatnonzero(np.diff(rows)) + 1
        # rows are sorted (row-major nonzero), so the split is by batch element;
        # every mask has at least one masked position, so none is skipped.
        for piece in np.split(np.arange(rows.size), bounds):
            out.append({f: v[piece] for f, v in st.items()})
    if len(out) != len(masks):
        raise RuntimeError(f"scored {len(out)} states for {len(masks)} masks (an empty mask?)")
    return out


# --------------------------------------------------------------------------- #
# One document
# --------------------------------------------------------------------------- #

@dataclass
class _State:
    state: MaskedState
    stats: dict
    committed: np.ndarray | None = None
    commit_rank: np.ndarray | None = None
    delta_ell: np.ndarray | None = None


def document_states(model, doc: Document, cfg: CollectConfig,
                    counter: dict | None = None) -> list[_State]:
    """Every masked state of one document, scored, in state-index order.

    Order: uniform (rate by rate, then repeat), then decoder trajectories (policy,
    then k, then step), then the uniform-matched partner of each recorded decoder
    state.  Each family draws from its own random stream, so switching one off
    leaves the others' masks unchanged.
    """
    counter = counter if counter is not None else {"forward_passes": 0}
    x0 = doc.ids
    L = int(x0.size)
    region = maskable_mask(L, doc.span)
    if hasattr(model, "set_reference"):     # the CPU toy model only
        model.set_reference(x0)
    out: list[_State] = []

    if cfg.include_uniform:
        rng = generator(cfg.seed, "masks", str(doc.doc_id))
        uni = uniform_masks(L, cfg.mask_rates, rng, cfg.uniform_min_masked,
                            per_rate=cfg.masks_per_rate, maskable=doc.span)
        for st, stats in zip(uni, _score_masks(model, x0, [u.mask for u in uni], cfg, counter)):
            out.append(_State(st, stats))

    decoder_states: list[MaskedState] = []
    if cfg.include_decoder:
        trajs = [TeacherForcedTrajectory(L, int(k), policy, cfg.min_masked, region)
                 for policy in cfg.decoder_policies for k in cfg.decoder_k]
        # Advance all trajectories in lockstep so they can share batches.
        while True:
            active = [t for t in trajs if not t.done]
            if not active:
                break
            stats = _score_masks(model, x0, [t.mask.copy() for t in active], cfg, counter)
            for t, s in zip(active, stats):
                t.advance(s)
        for t in trajs:
            for step in subsample_evenly(t.steps, cfg.decoder_states):
                out.append(_State(step.state, step.stats, committed=step.committed,
                                  commit_rank=step.rank, delta_ell=step.delta_ell))
                decoder_states.append(step.state)

    if cfg.include_uniform_matched and decoder_states:
        rng = generator(cfg.seed, "matched", str(doc.doc_id))
        matched = matched_uniform_masks(decoder_states, rng, maskable=doc.span)
        scored = _score_masks(model, x0, [m.mask for m in matched], cfg, counter)
        out += [_State(st, s) for st, s in zip(matched, scored)]

    if len(out) >= STATE_STRIDE:
        raise ValueError(f"document {doc.doc_id} has {len(out)} states; state ids would collide")
    return out


def _identity(model, dataset: str) -> dict:
    from ..provenance import git_commit

    return {"model": model.name,
            "variant": getattr(model, "variant", None) or "na",
            "revision": getattr(model, "revision", None) or "na",
            "precision": getattr(model, "precision", "na"),
            "dataset": dataset,
            "git_commit": git_commit()[0]}


def arrow_schema(topk: int):
    import pyarrow as pa

    s, i64, f64, f32 = pa.string(), pa.int64(), pa.float64(), pa.float32()
    types = {
        "model": s, "variant": s, "revision": s, "precision": s, "dataset": s,
        "git_commit": s, "seed": i64, "mask_kind": s, "policy": s, "k": pa.int16(),
        "target_rate": f32, "mask_rate": f64, "step": i64, "n_masked": i64,
        "doc_id": i64, "state_id": i64, "state_idx": i64, "doc_len": i64,
        "span_start": i64, "span_len": i64, "position": i64, "top1": i64,
        "conf": f64, "ell": f64, "p_true": f32, "entropy": f32, "margin": f32,
        "topk_ids": pa.list_(pa.int32(), topk), "topk_p": pa.list_(f32, topk),
        "delta_ell": f32, "agree": i64, "committed": pa.bool_(), "commit_rank": pa.int32(),
    }
    return pa.schema([(c, types[c]) for c in TRACE_COLUMNS])


def document_table(model, doc: Document, cfg: CollectConfig, identity: dict,
                   counter: dict | None = None):
    """One document's rows as a pyarrow Table in the v2 schema.

    Built column-wise from numpy: at ~60k rows per document, a dict per row
    would cost more than the forward passes on a small model.
    """
    import pyarrow as pa

    counter = counter if counter is not None else {"forward_passes": 0}
    states = document_states(model, doc, cfg, counter)
    x0 = doc.ids
    cols: dict[str, list] = {c: [] for c in TRACE_COLUMNS if c not in identity}
    n_sat = 0
    for si, s in enumerate(states):
        st, stats = s.state, s.stats
        pos = np.flatnonzero(st.mask)
        n = pos.size
        conf = np.asarray(stats["conf"], dtype=np.float64)
        n_sat += int(np.sum(conf >= 1.0 - cfg.eps) + np.sum(conf <= cfg.eps))

        def const(v, dtype):
            return np.full(n, v, dtype=dtype)

        cols["seed"].append(const(cfg.seed, np.int64))
        cols["mask_kind"].append(const(st.kind, object))
        cols["policy"].append(const(st.policy, object))
        cols["k"].append(const(st.k, np.int16))
        cols["target_rate"].append(const(st.target_rate, np.float32))
        cols["mask_rate"].append(const(st.rate, np.float64))
        cols["step"].append(const(st.step, np.int64))
        cols["n_masked"].append(const(n, np.int64))
        cols["doc_id"].append(const(doc.doc_id, np.int64))
        cols["state_id"].append(const(doc.doc_id * STATE_STRIDE + si, np.int64))
        cols["state_idx"].append(const(si, np.int64))
        cols["doc_len"].append(const(x0.size, np.int64))
        cols["span_start"].append(const(doc.span[0], np.int64))
        cols["span_len"].append(const(doc.span[1] - doc.span[0], np.int64))
        cols["position"].append(pos.astype(np.int64))
        cols["top1"].append(np.asarray(stats["top1"], dtype=np.int64))
        cols["conf"].append(conf)
        cols["ell"].append(np.asarray(stats["ell"], dtype=np.float64))
        for f in ("p_true", "entropy", "margin"):
            cols[f].append(np.asarray(stats[f], dtype=np.float32))
        cols["topk_ids"].append(np.asarray(stats["topk_ids"], dtype=np.int32))
        cols["topk_p"].append(np.asarray(stats["topk_p"], dtype=np.float32))
        cols["delta_ell"].append(const(np.nan, np.float32) if s.delta_ell is None
                                 else s.delta_ell.astype(np.float32))
        cols["agree"].append((np.asarray(stats["top1"]) == x0[pos]).astype(np.int64))
        cols["committed"].append(const(False, bool) if s.committed is None else s.committed)
        cols["commit_rank"].append(const(-1, np.int32) if s.commit_rank is None
                                   else s.commit_rank.astype(np.int32))

    schema = arrow_schema(cfg.topk)
    arrays = {}
    n_rows = sum(c.size for c in cols["position"])
    for c in TRACE_COLUMNS:
        if c in identity:
            arrays[c] = pa.array([identity[c]] * n_rows, type=schema.field(c).type)
        elif c in ("topk_ids", "topk_p"):
            flat = np.concatenate(cols[c]).reshape(-1)
            arrays[c] = pa.FixedSizeListArray.from_arrays(
                pa.array(flat, type=schema.field(c).type.value_type), cfg.topk)
        else:
            arrays[c] = pa.array(np.concatenate(cols[c]), type=schema.field(c).type)
    table = pa.table([arrays[c] for c in TRACE_COLUMNS], schema=schema)
    return table, n_sat


def collect_document(model, doc_id: int, x0: np.ndarray, cfg: CollectConfig) -> list[dict]:
    """Every masked state for one document, as position-level rows (small inputs only)."""
    table, _ = document_table(model, Document(doc_id, x0), cfg, _identity(model, "na"))
    return table.to_pylist()


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #

class _PartWriter:
    """A parquet file written under a temporary name and renamed on close."""

    def __init__(self, path: Path, schema):
        import pyarrow.parquet as pq

        self.path = Path(path)
        self.tmp = self.path.with_name(self.path.name + ".tmp")
        self.writer = pq.ParquetWriter(self.tmp, schema, compression="zstd")
        self.n_rows = 0
        self.doc_ids: list[int] = []

    def write(self, table, doc_id: int) -> None:
        self.writer.write_table(table)
        self.n_rows += table.num_rows
        self.doc_ids.append(int(doc_id))

    def close(self) -> None:
        self.writer.close()
        self.tmp.replace(self.path)

    def abort(self) -> None:
        try:
            self.writer.close()
        finally:
            self.tmp.unlink(missing_ok=True)


def run_meta(model, cfg: CollectConfig, dataset: str, extra: dict | None = None) -> dict:
    """The identity every part of a run must share, plus provenance."""
    from ..provenance import collect as provenance

    ident = _identity(model, dataset)
    meta = {"schema_version": SCHEMA_VERSION, "config": cfg.to_dict(),
            "model": model.name, "variant": ident["variant"], "revision": ident["revision"],
            "path": getattr(model, "path", None), "precision": ident["precision"],
            "dataset": dataset, "git_commit": ident["git_commit"],
            "device_map": getattr(model, "device_map", None),
            "provenance": provenance(with_gpu=True)}
    if extra:
        meta.update(extra)
    return meta


IDENTITY_KEYS = ("schema_version", "config", "model", "variant", "revision", "precision",
                 "dataset")
"""Meta fields that must agree across the shards and parts of one run."""


def _collect_into(model, docs: list[Document], cfg: CollectConfig, next_path,
                  part_docs: int | None, progress_every: int, stats: dict) -> list[dict]:
    """Write ``docs`` into successive part files; returns a record per part."""
    ident = _identity(model, docs[0].dataset) if docs else None
    schema = arrow_schema(cfg.topk)
    parts: list[dict] = []
    writer = None
    t0 = time.time()
    counter = {"forward_passes": 0}
    try:
        for i, doc in enumerate(docs):
            if writer is None:
                writer = _PartWriter(next_path(), schema)
            table, sat = document_table(model, doc, cfg, ident, counter)
            writer.write(table, doc.doc_id)
            stats["n_saturated"] += sat
            if part_docs and len(writer.doc_ids) >= part_docs:
                writer.close()
                parts.append({"file": writer.path.name, "n_rows": writer.n_rows,
                              "doc_ids": writer.doc_ids})
                writer = None
            if progress_every and (i + 1) % progress_every == 0:
                hrs = (time.time() - t0) / 3600
                print(f"[collect] {i+1}/{len(docs)} docs, {counter['forward_passes']} forward "
                      f"passes, {(i+1)/max(hrs, 1e-9):.1f} docs/hour", flush=True)
        if writer is not None:
            writer.close()
            parts.append({"file": writer.path.name, "n_rows": writer.n_rows,
                          "doc_ids": writer.doc_ids})
            writer = None
    finally:
        if writer is not None:
            writer.abort()
        stats["forward_passes"] += counter["forward_passes"]
        stats["seconds"] += time.time() - t0
    secs = max(time.time() - t0, 1e-9)
    if docs:
        print(f"[collect] throughput: {len(docs)} docs, {counter['forward_passes']} forward "
              f"passes in {secs:.0f} s = {len(docs) / secs * 3600:.1f} docs/hour, "
              f"{counter['forward_passes'] / secs:.2f} passes/s", flush=True)
    return parts


def collect(model, documents, cfg: CollectConfig, out_path: str | Path,
            progress_every: int = 25, dataset: str = "fineweb") -> Path:
    """Run the harness over a corpus and write one parquet trace file (+ .meta.json)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    docs = [as_document(d, dataset) for d in documents]
    stats = {"n_saturated": 0, "forward_passes": 0, "seconds": 0.0}
    parts = _collect_into(model, docs, cfg, lambda: out_path, None, progress_every, stats)
    n_rows = sum(p["n_rows"] for p in parts)
    meta = run_meta(model, cfg, docs[0].dataset if docs else dataset,
                    {"n_rows": n_rows, "n_docs": len(docs),
                     "doc_ids": [d.doc_id for d in docs], **stats})
    out_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[collect] wrote {n_rows} rows -> {out_path}")
    return out_path


def shard_stem(stem: str | Path, shard: int, num_shards: int) -> Path:
    stem = Path(stem)
    return stem.with_name(f"{stem.name}.shard{shard}of{num_shards}")


def collect_shard(model, documents, cfg: CollectConfig, stem: str | Path, shard: int = 0,
                  num_shards: int = 1, resume: bool = False, part_docs: int = 50,
                  progress_every: int = 25, dataset: str = "fineweb",
                  extra_meta: dict | None = None) -> Path:
    """Collect shard ``shard`` of ``num_shards`` into part files; returns the shard meta path.

    Files: ``<stem>.shard{I}of{N}.p{J}.parquet`` and ``<stem>.shard{I}of{N}.meta.json``.
    With ``resume``, documents already in a complete part are skipped, and the
    stored identity (config, model, revision, precision, ...) must match this run.
    """
    base = shard_stem(stem, shard, num_shards)
    base.parent.mkdir(parents=True, exist_ok=True)
    meta_path = base.with_name(base.name + ".meta.json")
    docs = [as_document(d, dataset) for d in shard_documents(documents, shard, num_shards)]
    ds = docs[0].dataset if docs else dataset
    meta = run_meta(model, cfg, ds, extra_meta)
    meta.update({"shard": shard, "num_shards": num_shards,
                 "doc_ids_expected": sorted(d.doc_id for d in docs)})

    existing = sorted(base.parent.glob(base.name + ".p*.parquet"))
    old_parts: list[dict] = []
    stats = {"n_saturated": 0, "forward_passes": 0, "seconds": 0.0}
    if existing or meta_path.exists():
        if not resume:
            raise FileExistsError(f"{base}: parts already exist; pass resume=True (--resume) "
                                  f"to continue them, or choose a new --out")
        old = json.loads(meta_path.read_text()) if meta_path.exists() else None
        if old is not None:
            diff = [k for k in IDENTITY_KEYS if old.get(k) != meta.get(k)]
            if diff:
                raise ValueError(f"{meta_path}: cannot resume, the run differs in {diff}")
            for k in stats:
                stats[k] = old.get("stats", {}).get(k, stats[k])
            # Parts written before a requeue may come from another commit; keep them all.
            meta["part_commits"] = sorted(set(old.get("part_commits", [old.get("git_commit")]))
                                          | {meta["git_commit"]})
        old_parts = [part_record(p) for p in existing]
    done = {d for p in old_parts for d in p["doc_ids"]}
    todo = [d for d in docs if d.doc_id not in done]
    print(f"[collect] shard {shard}/{num_shards}: {len(docs)} docs, {len(done)} already done, "
          f"{len(todo)} to go", flush=True)

    # Number new parts after the highest existing index, not the count: with a
    # gap (a part lost in transfer) the count names an existing, complete part,
    # and the rename below would silently replace it.
    idx = [int(m.group(1)) for m in (re.search(r"\.p(\d+)\.parquet$", q.name) for q in existing) if m]
    counter = [max(idx) + 1 if idx else 0]

    def next_path():
        p = base.with_name(f"{base.name}.p{counter[0]:04d}.parquet")
        counter[0] += 1
        return p

    def write_meta(parts, complete):
        meta["parts"] = parts
        meta["n_rows"] = sum(p["n_rows"] for p in parts)
        meta["doc_ids_done"] = sorted(d for p in parts for d in p["doc_ids"])
        meta["complete"] = complete
        meta["stats"] = stats
        # Atomic: a job killed mid-write (or a push racing it) must not leave a
        # truncated meta that every later resume fails to parse.
        tmp = meta_path.with_name(meta_path.name + ".tmp")
        tmp.write_text(json.dumps(meta, indent=2))
        tmp.replace(meta_path)

    write_meta(old_parts, complete=not todo)
    new_parts = _collect_into(model, todo, cfg, next_path, part_docs, progress_every, stats)
    parts = old_parts + new_parts
    covered = {d for p in parts for d in p["doc_ids"]}
    write_meta(parts, complete=set(meta["doc_ids_expected"]) <= covered)
    print(f"[collect] shard {shard}: {meta['n_rows']} rows in {len(parts)} parts -> {base}*")
    return meta_path


def part_record(path: Path) -> dict:
    """File name, row count and document ids of a complete part file."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    ids = pf.read(columns=["doc_id"]).column("doc_id").to_numpy()
    return {"file": Path(path).name, "n_rows": pf.metadata.num_rows,
            "doc_ids": sorted(int(d) for d in np.unique(ids))}
